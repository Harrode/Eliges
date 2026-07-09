"""
字段映射增强模块
功能：动态字段发现、模糊匹配、字段验证
"""
import json
import re
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher


class FieldMapper:
    """字段映射器"""

    def __init__(self, es_client=None, index_name: str = None):
        self.es_client = es_client
        self.index_name = index_name
        self.field_cache = {}
        self.field_aliases = {}
        self.field_types = {}

        # 加载字段别名配置
        self._load_field_aliases()

    def _load_field_aliases(self):
        """加载字段别名配置"""
        self.field_aliases = {
            # 诊断相关
            '诊断': 'diagnosis',
            '疾病': 'diagnosis',
            '病症': 'diagnosis',
            '病名': 'diagnosis',

            # 科室相关
            '科室': 'department',
            '科别': 'department',
            '就诊科室': 'department',

            # 年龄相关
            '年龄': 'age',
            '岁数': 'age',

            # 性别相关
            '性别': 'gender',
            '男女': 'gender',

            # 入院相关
            '入院': 'admission_date',
            '入院时间': 'admission_date',
            '入院日期': 'admission_date',
            '住院时间': 'admission_date',

            # 出院相关
            '出院': 'discharge_date',
            '出院时间': 'discharge_date',
            '出院日期': 'discharge_date',

            # 主诉相关
            '主诉': 'chief_complaint',
            '症状': 'chief_complaint',
            '不适': 'chief_complaint',

            # 现病史相关
            '现病史': 'history_present',
            '病史': 'history_present',
            '发病': 'history_present',

            # 既往史相关
            '既往史': 'past_history',
            '过去病史': 'past_history',

            # 药物相关
            '药物': 'medications',
            '用药': 'medications',
            '药品': 'medications',
            '处方': 'medications',

            # 检验相关
            '检验': 'lab_results',
            '检查': 'lab_results',
            '化验': 'lab_results',
            '实验室检查': 'lab_results',

            # 手术相关
            '手术': 'procedure_notes',
            '操作': 'procedure_notes',
            '治疗': 'procedure_notes',

            # 生命体征相关
            '生命体征': 'vital_signs',
            '体征': 'vital_signs',
            '血压': 'vital_signs.bp_systolic',
            '心率': 'vital_signs.heart_rate',
            '体温': 'vital_signs.temperature',
        }

    def _flatten_properties(self, properties: Dict, prefix: str = "") -> Dict[str, str]:
        """Flatten ES mapping properties into dot-notated field/type pairs."""
        fields = {}
        for field, config in properties.items():
            full_name = f"{prefix}.{field}" if prefix else field
            field_type = config.get('type', 'object')
            fields[full_name] = field_type

            nested_props = config.get('properties')
            if nested_props:
                fields.update(self._flatten_properties(nested_props, full_name))

        return fields

    def get_index_fields(self, index_name: str = None) -> List[str]:
        """从 ES 获取索引的所有字段"""
        target_index = index_name or self.index_name

        if target_index in self.field_cache:
            return self.field_cache[target_index]

        if not self.es_client:
            return []

        try:
            mapping = self.es_client.indices.get_mapping(index=target_index)
            properties = mapping[target_index]['mappings']['properties']
            fields = list(self._flatten_properties(properties).keys())
            self.field_cache[target_index] = fields
            return fields
        except Exception as e:
            print(f'[FieldMapper] Error getting fields for {target_index}: {e}')
            return []

    def get_field_types(self, index_name: str = None) -> Dict[str, str]:
        """获取字段类型"""
        target_index = index_name or self.index_name

        if target_index in self.field_types:
            return self.field_types[target_index]

        if not self.es_client:
            return {}

        try:
            mapping = self.es_client.indices.get_mapping(index=target_index)
            properties = mapping[target_index]['mappings']['properties']

            field_types = self._flatten_properties(properties)

            self.field_types[target_index] = field_types
            return field_types
        except Exception as e:
            print(f'[FieldMapper] Error getting field types for {target_index}: {e}')
            return {}

    def find_matching_field(self, user_input: str, index_name: str = None) -> Optional[str]:
        """根据用户输入查找匹配的字段"""
        # 1. 精确匹配
        if user_input in self.field_aliases:
            return self.field_aliases[user_input]

        # 2. 获取索引字段
        fields = self.get_index_fields(index_name)
        if not fields:
            return None

        # 3. 模糊匹配
        best_match = None
        best_score = 0.0

        for field in fields:
            # 计算相似度
            score = self._calculate_similarity(user_input, field)

            # 检查别名
            for alias, target_field in self.field_aliases.items():
                if target_field == field:
                    alias_score = self._calculate_similarity(user_input, alias)
                    score = max(score, alias_score)

            if score > best_score and score > 0.6:  # 阈值 0.6
                best_score = score
                best_match = field

        return best_match

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """计算字符串相似度"""
        # 使用 SequenceMatcher 计算相似度
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

    def validate_and_map_fields(self, conditions: Dict, index_name: str = None) -> Tuple[Dict, List[str]]:
        """验证并映射字段"""
        validated = {}
        warnings = []

        for field, value in conditions.items():
            # 检查是否是已知字段
            if field in ['diagnoses', 'age_min', 'age_max', 'gender', 'lab_tests',
                        'procedures', 'medications', 'symptoms', 'time_conditions',
                        'department', 'diagnoses_excluded', 'conditions_excluded',
                        'medications_excluded', 'lab_tests_excluded']:
                validated[field] = value
                continue

            # 尝试映射字段
            mapped_field = self.find_matching_field(field, index_name)

            if mapped_field:
                validated[mapped_field] = value
                warnings.append(f"字段 '{field}' 已映射到 '{mapped_field}'")
            else:
                # 保留原字段，但添加警告
                validated[field] = value
                warnings.append(f"字段 '{field}' 未找到匹配，保留原值")

        return validated, warnings

    def get_field_description(self, field: str) -> str:
        """获取字段描述"""
        descriptions = {
            'diagnosis': '诊断名称',
            'department': '科室',
            'age': '年龄',
            'gender': '性别',
            'admission_date': '入院日期',
            'discharge_date': '出院日期',
            'chief_complaint': '主诉',
            'history_present': '现病史',
            'past_history': '既往史',
            'medications': '药物',
            'lab_results': '检验结果',
            'procedure_notes': '手术/操作',
            'vital_signs': '生命体征',
        }
        return descriptions.get(field, field)

    def suggest_fields(self, partial_input: str, index_name: str = None, top_n: int = 5) -> List[Tuple[str, str, float]]:
        """建议可能的字段"""
        fields = self.get_index_fields(index_name)
        if not fields:
            return []

        suggestions = []

        for field in fields:
            # 计算相似度
            score = self._calculate_similarity(partial_input, field)

            # 检查别名
            for alias, target_field in self.field_aliases.items():
                if target_field == field:
                    alias_score = self._calculate_similarity(partial_input, alias)
                    score = max(score, alias_score)

            if score > 0.3:  # 较低的阈值用于建议
                description = self.get_field_description(field)
                suggestions.append((field, description, score))

        # 按分数排序
        suggestions.sort(key=lambda x: x[2], reverse=True)

        return suggestions[:top_n]


class FieldMappingConfig:
    """字段映射配置管理"""

    def __init__(self, config_path: str = 'config/field_mapping.json'):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f'[FieldMappingConfig] Error loading config: {e}')
            return {}

    def get_field_mapping(self, field_name: str) -> Optional[Dict]:
        """获取字段映射配置"""
        return self.config.get('field_mapping', {}).get(field_name)

    def get_all_mappings(self) -> Dict:
        """获取所有字段映射"""
        return self.config.get('field_mapping', {})

    def add_field_mapping(self, field_name: str, mapping: Dict):
        """添加字段映射"""
        if 'field_mapping' not in self.config:
            self.config['field_mapping'] = {}

        self.config['field_mapping'][field_name] = mapping
        self._save_config()

    def _save_config(self):
        """保存配置文件"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f'[FieldMappingConfig] Error saving config: {e}')


# 全局实例
_field_mapper = None
_field_mapping_config = None


def get_field_mapper(es_client=None, index_name: str = None) -> FieldMapper:
    """获取字段映射器实例"""
    global _field_mapper
    if _field_mapper is None:
        _field_mapper = FieldMapper(es_client, index_name)
    return _field_mapper


def get_field_mapping_config() -> FieldMappingConfig:
    """获取字段映射配置实例"""
    global _field_mapping_config
    if _field_mapping_config is None:
        _field_mapping_config = FieldMappingConfig()
    return _field_mapping_config
