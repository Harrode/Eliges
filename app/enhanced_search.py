"""
增强搜索模块
整合字段映射、跨索引查询、后处理代码生成
"""
import json
from typing import Dict, List, Optional, Any
from elasticsearch import Elasticsearch

from .field_mapper import FieldMapper, get_field_mapper, get_field_mapping_config
from .cross_index_search import IndexDiscovery, CrossIndexSearch, IndexRouter, get_cross_index_search, get_index_router
from .post_processor import PostProcessingPipeline, get_post_processing_pipeline


class EnhancedSearchEngine:
    """增强搜索引擎"""

    def __init__(self, es_client: Elasticsearch, llm_client=None):
        self.es_client = es_client
        self.llm_client = llm_client

        # 初始化组件
        self.field_mapper = get_field_mapper(es_client)
        self.index_router = get_index_router(es_client)
        self.cross_index_search = get_cross_index_search(es_client)
        self.post_processor = get_post_processing_pipeline(llm_client)

        # 当前索引
        self.current_index = 'n2c2_patients'

    def search(self, query: str, conditions: Dict, options: Dict = None) -> Dict:
        """增强搜索"""
        options = options or {}
        enable_cross_index = options.get('enable_cross_index', True)
        enable_field_mapping = options.get('enable_field_mapping', True)
        max_results = options.get('max_results', 100)

        result = {
            'query': query,
            'original_conditions': conditions.copy(),
            'processed_conditions': {},
            'field_mapping_warnings': [],
            'searched_indices': [],
            'results': [],
            'total': 0,
            'dsl': None
        }

        # 1. 字段映射增强
        if enable_field_mapping:
            processed_conditions, warnings = self.field_mapper.validate_and_map_fields(
                conditions, self.current_index
            )
            result['processed_conditions'] = processed_conditions
            result['field_mapping_warnings'] = warnings
        else:
            result['processed_conditions'] = conditions

        # 2. 跨索引查询
        if enable_cross_index:
            # 路由到相关索引
            target_indices = self.index_router.route(query, conditions)

            if target_indices:
                # 跨索引查询
                search_result = self.cross_index_search.search(
                    result['processed_conditions'],
                    query,
                    max_indices=3,
                    target_indices=target_indices
                )
                result['results'] = search_result.get('hits', [])[:max_results]
                result['total'] = search_result.get('total', 0)
                result['searched_indices'] = search_result.get('indices', [])
            else:
                # 降级到单索引查询
                result = self._fallback_single_index_search(query, result['processed_conditions'], result)
        else:
            # 单索引查询
            result = self._fallback_single_index_search(query, result['processed_conditions'], result)

        return result

    def _fallback_single_index_search(self, query: str, conditions: Dict, result: Dict) -> Dict:
        """降级到单索引查询"""
        try:
            # 构建查询 DSL
            dsl = self._build_dsl_for_index(conditions, self.current_index)

            # 执行查询
            es_result = self.es_client.search(
                index=self.current_index,
                body=dsl,
                size=100
            )

            # 处理结果
            hits = []
            for hit in es_result['hits']['hits']:
                hit['_source']['_score'] = hit.get('_score', 0)
                hits.append(hit['_source'])

            result['results'] = hits
            result['total'] = es_result['hits']['total']['value']
            result['searched_indices'] = [self.current_index]
            result['dsl'] = dsl

        except Exception as e:
            print(f'[EnhancedSearch] Error in fallback search: {e}')
            result['error'] = str(e)

        return result

    def _build_dsl_for_index(self, conditions: Dict, index: str) -> Dict:
        """为指定索引构建 DSL"""
        must_clauses = []
        filter_clauses = []

        # 获取索引字段
        metadata = self.field_mapper.get_index_fields(index)
        field_names = metadata if metadata else []

        for field, value in conditions.items():
            # 处理特殊字段
            if field == 'diagnoses':
                # 诊断搜索
                for diag in value:
                    must_clauses.append({
                        "multi_match": {
                            "query": diag,
                            "fields": ["diagnosis^3", "chief_complaint^2", "history_present^2"],
                            "type": "best_fields",
                            "fuzziness": "AUTO"
                        }
                    })

            elif field == 'age_min':
                filter_clauses.append({"range": {"age": {"gte": value}}})

            elif field == 'age_max':
                filter_clauses.append({"range": {"age": {"lte": value}}})

            elif field == 'gender':
                gender_variants = {
                    'male': ['male', 'Male', 'M', '男', '男性'],
                    'female': ['female', 'Female', 'F', '女', '女性'],
                }.get(value, [value])
                filter_clauses.append({"terms": {"gender": gender_variants}})

            elif field == 'department':
                filter_clauses.append({"term": {"department": value}})

            elif field == 'diagnoses_excluded':
                # 排除诊断
                for diag in value:
                    must_clauses.append({
                        "bool": {
                            "must_not": {
                                "multi_match": {
                                    "query": diag,
                                    "fields": ["diagnosis", "chief_complaint", "history_present"]
                                }
                            }
                        }
                    })

            elif field == 'time_conditions':
                # 时间条件
                for tc in value:
                    if isinstance(tc, dict):
                        time_field = tc.get('field', 'admission_date')
                        operator = tc.get('operator', '')
                        time_value = tc.get('value', '')

                        if operator == 'since':
                            # 解析日期
                            import re
                            year_match = re.search(r'(\d{4})', str(time_value))
                            if year_match:
                                year = int(year_match.group(1))
                                month = 1

                                month_match = re.search(r'(\d{4})年(\d{1,2})月?', str(time_value))
                                if month_match:
                                    month = int(month_match.group(2))
                                else:
                                    month_match = re.search(r'(\d{4})-(\d{1,2})', str(time_value))
                                    if month_match:
                                        month = int(month_match.group(2))

                                start_date = f"{year}-{month:02d}-01"
                                filter_clauses.append({"range": {time_field: {"gte": start_date}}})

                        elif operator == 'between':
                            value_start = tc.get('value_start', '')
                            value_end = tc.get('value_end', '')

                            if value_start and value_end:
                                # 解析起始日期
                                import re
                                start_match = re.search(r'(\d{4})-(\d{1,2})', str(value_start))
                                end_match = re.search(r'(\d{4})-(\d{1,2})', str(value_end))

                                if start_match and end_match:
                                    start_year = int(start_match.group(1))
                                    start_month = int(start_match.group(2))
                                    end_year = int(end_match.group(1))
                                    end_month = int(end_match.group(2))

                                    start_date = f"{start_year}-{start_month:02d}-01"

                                    # 计算结束日期（月末）
                                    if end_month == 12:
                                        end_date = f"{end_year + 1}-01-01"
                                    else:
                                        end_date = f"{end_year}-{end_month + 1:02d}-01"

                                    filter_clauses.append({
                                        "range": {
                                            time_field: {
                                                "gte": start_date,
                                                "lt": end_date
                                            }
                                        }
                                    })

                        elif operator == 'year':
                            # 年份匹配
                            import re
                            year_match = re.search(r'(\d{4})', str(time_value))
                            if year_match:
                                year = int(year_match.group(1))
                                start_date = f"{year}-01-01"
                                end_date = f"{year + 1}-01-01"
                                filter_clauses.append({
                                    "range": {
                                        time_field: {
                                            "gte": start_date,
                                            "lt": end_date
                                        }
                                    }
                                })

                        elif operator == 'recent':
                            # 最近N个月
                            from datetime import datetime, timedelta
                            months = int(time_value)
                            cutoff_date = (datetime.now() - timedelta(days=months * 30)).strftime('%Y-%m-%d')
                            filter_clauses.append({"range": {time_field: {"gte": cutoff_date}}})

        # 构建完整查询
        if must_clauses or filter_clauses:
            return {
                "query": {
                    "bool": {
                        "must": must_clauses if must_clauses else [{"match_all": {}}],
                        "filter": filter_clauses
                    }
                }
            }
        else:
            return {"query": {"match_all": {}}}

    def post_process(self, user_request: str, results: List[Dict], **kwargs) -> Any:
        """后处理"""
        return self.post_processor.process(user_request, results, **kwargs)

    def get_field_suggestions(self, partial_input: str, top_n: int = 5) -> List[Dict]:
        """获取字段建议"""
        suggestions = self.field_mapper.suggest_fields(partial_input, self.current_index, top_n)
        return [
            {'field': s[0], 'description': s[1], 'score': s[2]}
            for s in suggestions
        ]

    def get_available_indices(self) -> List[str]:
        """获取可用索引"""
        return self.index_router.index_discovery.get_all_indices()

    def get_index_fields(self, index_name: str = None) -> List[str]:
        """获取索引字段"""
        return self.field_mapper.get_index_fields(index_name or self.current_index)

    def get_available_templates(self) -> List[str]:
        """获取可用模板"""
        return self.post_processor.get_available_templates()

    def set_current_index(self, index_name: str):
        """设置当前索引"""
        self.current_index = index_name
        # 清除缓存
        self.field_mapper.field_cache.clear()


# 全局实例
_enhanced_search_engine = None


def get_enhanced_search_engine(es_client: Elasticsearch, llm_client=None) -> EnhancedSearchEngine:
    """获取增强搜索引擎实例"""
    global _enhanced_search_engine
    if _enhanced_search_engine is None:
        _enhanced_search_engine = EnhancedSearchEngine(es_client, llm_client)
    return _enhanced_search_engine
