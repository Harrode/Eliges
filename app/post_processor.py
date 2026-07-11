"""
后处理代码生成模块
功能：代码模板库、LLM代码生成、安全代码执行
"""
import json
import re
import ast
import math
import inspect
import os
import multiprocessing as mp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta


def _postprocess_worker(code: str, data: List[Dict], kwargs: Dict, output_queue):
    """Run user post-processing code in a short-lived child process."""
    try:
        executor = SafeCodeExecutor(use_subprocess=False)
        result = executor.execute(code, data, **kwargs)
        output_queue.put({"ok": True, "result": result})
    except Exception as e:
        output_queue.put({"ok": False, "error": str(e)})


class PostProcessingTemplates:
    """后处理代码模板库"""

    TEMPLATES = {
        '统计分析': '''
def post_process(results):
    """统计分析"""
    import pandas as pd

    if not results:
        return {'error': '没有数据'}

    df = pd.DataFrame(results)

    stats = {}

    # 年龄分布
    if 'age' in df.columns:
        age_stats = df['age'].describe()
        stats['age_stats'] = {
            'mean': round(age_stats['mean'], 1),
            'min': int(age_stats['min']),
            'max': int(age_stats['max']),
            'std': round(age_stats['std'], 1)
        }

    # 性别分布
    if 'gender' in df.columns:
        gender_dist = df['gender'].value_counts().to_dict()
        stats['gender_distribution'] = gender_dist

    # 科室分布
    if 'department' in df.columns:
        dept_dist = df['department'].value_counts().to_dict()
        stats['department_distribution'] = dept_dist

    return stats
''',

        '数据筛选': '''
def post_process(results, conditions):
    """数据筛选"""
    filtered = []

    for patient in results:
        if meets_conditions(patient, conditions):
            filtered.append(patient)

    return filtered

def meets_conditions(patient, conditions):
    """检查患者是否满足条件"""
    for field, value in conditions.items():
        if field == 'age_min':
            if patient.get('age', 0) < value:
                return False
        elif field == 'age_max':
            if patient.get('age', 0) > value:
                return False
        elif field == 'gender':
            if patient.get('gender') != value:
                return False
        elif field == 'department':
            if patient.get('department') != value:
                return False

    return True
''',

        '时间分析': '''
def post_process(results):
    """时间分析"""
    import pandas as pd
    from datetime import datetime

    if not results:
        return {'error': '没有数据'}

    df = pd.DataFrame(results)

    time_stats = {}

    # 入院日期分析
    if 'admission_date' in df.columns:
        df['admission_date'] = pd.to_datetime(df['admission_date'], errors='coerce')
        valid_dates = df['admission_date'].dropna()

        if not valid_dates.empty:
            time_stats['admission'] = {
                'earliest': valid_dates.min().strftime('%Y-%m-%d'),
                'latest': valid_dates.max().strftime('%Y-%m-%d'),
                'count': len(valid_dates)
            }

    # 出院日期分析
    if 'discharge_date' in df.columns:
        df['discharge_date'] = pd.to_datetime(df['discharge_date'], errors='coerce')
        valid_dates = df['discharge_date'].dropna()

        if not valid_dates.empty:
            time_stats['discharge'] = {
                'earliest': valid_dates.min().strftime('%Y-%m-%d'),
                'latest': valid_dates.max().strftime('%Y-%m-%d'),
                'count': len(valid_dates)
            }

    # 住院时长分析
    if 'admission_date' in df.columns and 'discharge_date' in df.columns:
        df['stay_duration'] = (df['discharge_date'] - df['admission_date']).dt.days
        valid_stays = df['stay_duration'].dropna()

        if not valid_stays.empty:
            time_stats['stay_duration'] = {
                'mean': round(valid_stays.mean(), 1),
                'min': int(valid_stays.min()),
                'max': int(valid_stays.max())
            }

    return time_stats
''',

        '检验结果分析': '''
def post_process(results):
    """检验结果分析"""
    import pandas as pd

    if not results:
        return {'error': '没有数据'}

    lab_stats = {}

    for patient in results:
        lab_results = patient.get('lab_results', [])
        for lab in lab_results:
            name = lab.get('name')
            value = lab.get('value')

            if name and value is not None:
                if name not in lab_stats:
                    lab_stats[name] = []
                lab_stats[name].append(value)

    # 计算统计信息
    summary = {}
    for name, values in lab_stats.items():
        if values:
            import numpy as np
            summary[name] = {
                'count': len(values),
                'mean': round(np.mean(values), 2),
                'min': round(min(values), 2),
                'max': round(max(values), 2),
                'std': round(np.std(values), 2)
            }

    return summary
''',

        '生成报告': '''
def post_process(results, query):
    """生成报告"""
    report = []

    report.append(f"查询条件: {query}")
    report.append(f"命中患者数: {len(results)}")
    report.append("")

    # 患者列表
    report.append("患者列表:")
    for i, patient in enumerate(results[:10], 1):
        name = patient.get('name', '未知')
        age = patient.get('age', '未知')
        gender = patient.get('gender', '未知')
        dept = patient.get('department', '未知')
        diagnosis = patient.get('diagnosis', '未知')

        report.append(f"{i}. {name}, {age}岁, {gender}, {dept}, {diagnosis}")

    return "\\n".join(report)
''',

        '导出数据': '''
def post_process(results, format='json'):
    """导出数据"""
    import json

    if format == 'json':
        return json.dumps(results, ensure_ascii=False, indent=2)
    elif format == 'csv':
        import pandas as pd
        df = pd.DataFrame(results)
        return df.to_csv(index=False)
    else:
        return results
''',

        '字段提取': '''
def post_process(results, fields=None):
    """提取指定字段，默认返回患者核心信息"""
    selected_fields = fields or ['patient_id', 'name', 'age', 'gender', 'department', 'diagnosis']
    output = []

    for item in results:
        output.append({field: item.get(field) for field in selected_fields if field in item})

    return output
''',

        'TopN排序': '''
def post_process(results, sort_by='_score', limit=10, descending=True):
    """按指定字段排序并返回前N条"""
    def sort_value(item):
        value = item.get(sort_by)
        if value is None and sort_by == '_score':
            value = item.get('score', 0)
        return value if isinstance(value, (int, float)) else 0

    ranked = sorted(results, key=sort_value, reverse=descending)
    return ranked[:limit]
'''
    }

    def get_template(self, template_name: str) -> Optional[str]:
        """获取模板"""
        return self.TEMPLATES.get(template_name)

    def get_all_templates(self) -> Dict[str, str]:
        """获取所有模板"""
        return self.TEMPLATES.copy()

    def find_matching_template(self, user_request: str) -> Optional[str]:
        """查找匹配的模板"""
        user_request_lower = user_request.lower()

        # 关键词匹配
        keyword_mapping = {
            '统计': '统计分析',
            '分析': '统计分析',
            '分布': '统计分析',
            '筛选': '数据筛选',
            '过滤': '数据筛选',
            '时间': '时间分析',
            '日期': '时间分析',
            '住院': '时间分析',
            '检验': '检验结果分析',
            '检查': '检验结果分析',
            '化验': '检验结果分析',
            '报告': '生成报告',
            '总结': '生成报告',
            '导出': '导出数据',
            '保存': '导出数据',
            '字段': '字段提取',
            '提取': '字段提取',
            '只要': '字段提取',
            '返回': '字段提取',
            'top': 'TopN排序',
            '前': 'TopN排序',
            '最高': 'TopN排序',
            '排序': 'TopN排序',
        }

        for keyword, template_name in keyword_mapping.items():
            if keyword in user_request_lower:
                return self.TEMPLATES.get(template_name)

        return None


class CodeGenerator:
    """LLM 驱动的代码生成器"""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def generate_post_process_code(self, user_request: str, data_schema: Dict) -> str:
        """根据用户需求生成后处理代码"""
        prompt = f"""You are a Python data-analysis expert. Generate post-processing code based on the user request.

Data schema:
{json.dumps(data_schema, ensure_ascii=False, indent=2)}

User request:
{user_request}

Generate a function named post_process that accepts a results list and returns the processed data.

Requirements:
1. Code must be safe and must not perform dangerous operations
2. Use pandas for data analysis
3. Return JSON-serializable data
4. Function signature: def post_process(results):
5. Code must be complete and executable

Code:
```python
"""
        try:
            response = self.llm_client.chat.completions.create(
                model=os.getenv('LLM_MODEL', 'deepseek-v4-flash'),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000,
                extra_body={
                    "enable_thinking": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            code = response.choices[0].message.content
            return self._extract_code(code)
        except Exception as e:
            print(f'[CodeGenerator] Error generating code: {e}')
            return None

    def _extract_code(self, text: str) -> Optional[str]:
        """从 LLM 输出中提取代码"""
        # 尝试提取 ```python ... ``` 代码块
        code_match = re.search(r'```python\n(.*?)\n```', text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # 尝试提取 ``` ... ``` 代码块
        code_match = re.search(r'```\n(.*?)\n```', text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # 如果没有代码块，尝试提取 def post_process 开头的代码
        code_match = re.search(r'(def post_process.*?)(?=\n\n|\Z)', text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        return text.strip()


class SafeCodeExecutor:
    """安全代码执行器"""

    # 允许的内置函数
    ALLOWED_BUILTINS = {
        'len': len,
        'range': range,
        'enumerate': enumerate,
        'zip': zip,
        'sorted': sorted,
        'min': min,
        'max': max,
        'sum': sum,
        'abs': abs,
        'round': round,
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
        'list': list,
        'dict': dict,
        'tuple': tuple,
        'set': set,
        'isinstance': isinstance,
        'hasattr': hasattr,
        'getattr': getattr,
        'print': print,
    }

    # 禁止的函数
    BLOCKED_FUNCTIONS = [
        'exec', 'eval', 'compile',  # 代码执行
        'open', 'file',  # 文件操作
        'os.system', 'os.popen', 'subprocess',  # 系统命令
        '__import__', 'importlib',  # 模块导入
        'globals', 'locals',  # 命名空间
        'breakpoint', 'exit', 'quit',  # 调试/退出
    ]

    # 允许的模块
    ALLOWED_MODULES = ['pandas', 'numpy', 'json', 'datetime', 'math', 'statistics', 'collections']

    def __init__(self, use_subprocess: bool = True):
        self.execution_count = 0
        self.max_executions = 100  # 限制执行次数
        self.use_subprocess = use_subprocess
        self.timeout_seconds = int(os.getenv('POST_PROCESS_TIMEOUT_SECONDS', '3'))
        self.max_rows = int(os.getenv('POST_PROCESS_MAX_ROWS', '1000'))
        self.max_code_chars = int(os.getenv('POST_PROCESS_MAX_CODE_CHARS', '12000'))

    def _limited_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        """Only allow imports from the small analysis-oriented module allowlist."""
        root_name = name.split('.')[0]
        if root_name not in self.ALLOWED_MODULES:
            raise ImportError(f"不允许导入模块: {name}")
        return __import__(name, globals, locals, fromlist, level)

    def validate_code(self, code: str) -> Tuple[bool, str]:
        """验证代码安全性"""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"代码语法错误: {e}"

        # 检查是否包含危险函数
        for node in ast.walk(tree):
            # 检查函数调用
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.BLOCKED_FUNCTIONS:
                        return False, f"不允许调用函数: {node.func.id}"
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in self.BLOCKED_FUNCTIONS:
                        return False, f"不允许调用方法: {node.func.attr}"

            # 检查导入语句
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split('.')[0] not in self.ALLOWED_MODULES:
                            return False, f"不允许导入模块: {alias.name}"
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.split('.')[0] not in self.ALLOWED_MODULES:
                        return False, f"不允许导入模块: {node.module}"

            # 检查属性访问
            if isinstance(node, ast.Attribute):
                # 禁止访问 __ 属性
                if node.attr.startswith('__') and node.attr.endswith('__'):
                    return False, f"不允许访问特殊属性: {node.attr}"

        return True, "代码安全"

    def execute(self, code: str, data: List[Dict], **kwargs) -> Any:
        """安全执行代码"""
        # 检查执行次数限制
        if self.execution_count >= self.max_executions:
            raise RuntimeError("超过最大执行次数限制")
        if len(data or []) > self.max_rows:
            raise RuntimeError(f"后处理数据量超过限制: {len(data)} > {self.max_rows}")
        if len(code or "") > self.max_code_chars:
            raise RuntimeError(f"后处理代码长度超过限制: {len(code)} > {self.max_code_chars}")

        # 验证代码安全性
        is_safe, message = self.validate_code(code)
        if not is_safe:
            raise ValueError(f"代码不安全: {message}")

        if self.use_subprocess:
            return self._execute_with_timeout(code, data, **kwargs)

        return self._execute_in_current_process(code, data, **kwargs)

    def _execute_with_timeout(self, code: str, data: List[Dict], **kwargs) -> Any:
        output_queue = mp.Queue(maxsize=1)
        process = mp.Process(target=_postprocess_worker, args=(code, data, kwargs, output_queue))
        process.start()
        process.join(self.timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join(1)
            raise TimeoutError(f"后处理执行超时，限制为 {self.timeout_seconds} 秒")
        if output_queue.empty():
            raise RuntimeError("后处理执行失败：子进程未返回结果")
        payload = output_queue.get()
        if not payload.get("ok"):
            raise RuntimeError(payload.get("error", "后处理执行失败"))
        self.execution_count += 1
        return self._to_json_safe(payload.get("result"))

    def _execute_in_current_process(self, code: str, data: List[Dict], **kwargs) -> Any:
        """Execute code after validation. Used inside the isolated child process."""
        # 创建受限的执行环境
        safe_globals = {
            '__builtins__': self.ALLOWED_BUILTINS,
        }
        safe_globals['__builtins__']['__import__'] = self._limited_import

        # 导入允许的模块
        try:
            import pandas as pd
            import numpy as np
            import json
            import math
            safe_globals['pd'] = pd
            safe_globals['np'] = np
            safe_globals['json'] = json
            safe_globals['math'] = math
        except ImportError:
            pass

        # 添加其他参数
        safe_globals.update(kwargs)

        # 执行代码
        try:
            exec(code, safe_globals)
        except Exception as e:
            raise RuntimeError(f"代码执行失败: {e}")

        # 调用 post_process 函数
        post_process = safe_globals.get('post_process')
        if post_process is None:
            raise ValueError("代码中没有定义 post_process 函数")

        # 执行函数
        try:
            signature = inspect.signature(post_process)
            accepted_kwargs = {
                key: value for key, value in kwargs.items()
                if key in signature.parameters
            }
            result = post_process(data, **accepted_kwargs)
            self.execution_count += 1
            return self._to_json_safe(result)
        except Exception as e:
            raise RuntimeError(f"post_process 函数执行失败: {e}")

    def _to_json_safe(self, value: Any) -> Any:
        """Convert pandas/numpy scalars and NaN values to JSON-safe Python values."""
        if isinstance(value, dict):
            return {str(k): self._to_json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_json_safe(v) for v in value]
        if isinstance(value, tuple):
            return [self._to_json_safe(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            if np.isnan(value):
                return None
            return float(value)
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, (pd.Timestamp, datetime)):
            return value.isoformat()
        return value


class PostProcessingPipeline:
    """后处理流水线"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.templates = PostProcessingTemplates()
        self.code_generator = CodeGenerator(llm_client) if llm_client else None
        self.executor = SafeCodeExecutor()

    def process(self, user_request: str, results: List[Dict], **kwargs) -> Any:
        """执行后处理"""
        # 1. 检查是否有匹配的模板
        template = self.templates.find_matching_template(user_request)

        if template:
            code = template
        elif self.code_generator:
            # 2. 使用 LLM 生成代码
            data_schema = self._get_data_schema(results)
            code = self.code_generator.generate_post_process_code(
                user_request, data_schema
            )
            if not code:
                return {"error": "无法生成代码"}
        else:
            return {"error": "无法处理请求"}

        # 3. 安全执行代码
        try:
            processed_results = self.executor.execute(code, results, **kwargs)
            return processed_results
        except Exception as e:
            return {"error": f"处理失败: {str(e)}"}

    def _get_data_schema(self, results: List[Dict]) -> Dict:
        """获取数据结构"""
        if not results:
            return {}

        # 分析第一条数据的结构
        sample = results[0]
        schema = {}

        for key, value in sample.items():
            if isinstance(value, str):
                schema[key] = 'string'
            elif isinstance(value, (int, float)):
                schema[key] = 'number'
            elif isinstance(value, bool):
                schema[key] = 'boolean'
            elif isinstance(value, list):
                schema[key] = 'array'
            elif isinstance(value, dict):
                schema[key] = 'object'
            else:
                schema[key] = 'unknown'

        return schema

    def get_available_templates(self) -> List[str]:
        """获取可用模板列表"""
        return list(self.templates.get_all_templates().keys())


# 全局实例
_post_processing_pipeline = None


def get_post_processing_pipeline(llm_client=None) -> PostProcessingPipeline:
    """获取后处理流水线实例"""
    global _post_processing_pipeline
    if _post_processing_pipeline is None:
        _post_processing_pipeline = PostProcessingPipeline(llm_client)
    return _post_processing_pipeline
