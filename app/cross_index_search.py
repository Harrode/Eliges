"""
跨索引查询模块
功能：索引发现、跨索引查询、结果合并
"""
import json
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


class IndexDiscovery:
    """索引发现器"""

    def __init__(self, es_client):
        self.es_client = es_client
        self.index_cache = {}
        self.index_metadata = {}

    def get_all_indices(self) -> List[str]:
        """获取所有可用索引"""
        try:
            indices = self.es_client.cat.indices(format='json')
            allowed = {
                item.strip()
                for item in os.getenv('ALLOWED_ES_INDICES', 'n2c2_patients,mimic_patients,emr_records,n2c2_large,n2c2_50').split(',')
                if item.strip()
            }
            # 过滤掉系统索引（以.开头）
            return [idx['index'] for idx in indices if not idx['index'].startswith('.') and idx['index'] in allowed]
        except Exception as e:
            print(f'[IndexDiscovery] Error getting indices: {e}')
            return []

    def get_index_metadata(self, index_name: str) -> Dict:
        """获取索引元数据"""
        if index_name in self.index_metadata:
            return self.index_metadata[index_name]

        try:
            # 获取索引映射
            mapping = self.es_client.indices.get_mapping(index=index_name)
            properties = mapping[index_name]['mappings']['properties']

            # 获取字段信息
            fields = self._flatten_fields(properties)

            # 获取文档数量
            count = self.es_client.count(index=index_name)['count']

            metadata = {
                'fields': fields,
                'field_names': list(fields.keys()),
                'doc_count': count
            }

            self.index_metadata[index_name] = metadata
            return metadata

        except Exception as e:
            print(f'[IndexDiscovery] Error getting metadata for {index_name}: {e}')
            return {}

    def _get_field_description(self, field: str) -> str:
        """获取字段描述"""
        descriptions = {
            'patient_id': '患者ID',
            'name': '姓名',
            'age': '年龄',
            'gender': '性别',
            'diagnosis': '诊断',
            'department': '科室',
            'admission_date': '入院日期',
            'discharge_date': '出院日期',
            'chief_complaint': '主诉',
            'history_present': '现病史',
            'past_history': '既往史',
            'medications': '药物',
            'lab_results': '检验结果',
            'procedure_notes': '手术/操作',
            'vital_signs': '生命体征',
            'discharge_summary': '出院小结',
        }
        return descriptions.get(field, field)

    def _flatten_fields(self, properties: Dict, prefix: str = "") -> Dict:
        """Flatten mapping fields, keeping nested/object children discoverable."""
        fields = {}
        for field, config in properties.items():
            full_name = f"{prefix}.{field}" if prefix else field
            field_type = config.get('type', 'object')
            fields[full_name] = {
                'type': field_type,
                'description': self._get_field_description(full_name),
                'path': prefix or None
            }

            nested_props = config.get('properties')
            if nested_props:
                fields.update(self._flatten_fields(nested_props, full_name))

        return fields

    def find_relevant_indices(self, query: str, conditions: Dict, top_n: int = 3) -> List[Tuple[str, float]]:
        """根据查询内容查找相关索引"""
        all_indices = self.get_all_indices()
        relevant_indices = []

        for index in all_indices:
            metadata = self.get_index_metadata(index)
            if not metadata:
                continue

            # 计算相关性分数
            score = self._calculate_relevance(query, conditions, metadata)

            if score > 0.3:  # 阈值
                relevant_indices.append((index, score, metadata['doc_count']))

        # 按分数排序
        relevant_indices.sort(key=lambda x: x[1], reverse=True)

        # 返回前N个索引
        return [(idx[0], idx[1]) for idx in relevant_indices[:top_n]]

    def _calculate_relevance(self, query: str, conditions: Dict, metadata: Dict) -> float:
        """计算查询与索引的相关性"""
        score = 0.0
        field_names = metadata.get('field_names', [])

        semantic_requirements = {
            'diagnoses': ['diagnosis', 'chief_complaint', 'history_present', 'discharge_summary'],
            'diagnoses_excluded': ['diagnosis', 'chief_complaint', 'history_present', 'discharge_summary'],
            'conditions_excluded': ['diagnosis', 'chief_complaint', 'history_present', 'discharge_summary'],
            'symptoms': ['chief_complaint', 'history_present'],
            'procedures': ['procedure_notes'],
            'medications': ['medications'],
            'medications_excluded': ['medications'],
            'lab_tests': ['lab_results.name', 'lab_results.value'],
            'lab_tests_excluded': ['lab_results.name', 'lab_results.value'],
            'age_min': ['age'],
            'age_max': ['age'],
            'gender': ['gender'],
            'department': ['department'],
            'time_conditions': ['admission_date', 'discharge_date'],
        }

        # 1. 检查条件中的字段或语义字段是否在索引中存在
        for field in conditions.keys():
            if field in field_names:
                score += 0.3
                continue
            required_fields = semantic_requirements.get(field, [])
            matched_required = sum(1 for required in required_fields if required in field_names)
            if required_fields:
                score += 0.25 * (matched_required / len(required_fields))

        # 2. 检查查询内容是否与索引字段相关
        query_lower = query.lower()
        for field in field_names:
            if field.lower() in query_lower:
                score += 0.1

        # 3. 根据文档数量调整分数（优先选择数据量大的索引）
        doc_count = metadata.get('doc_count', 0)
        if doc_count > 100:
            score += 0.1
        elif doc_count > 10:
            score += 0.05

        return min(score, 1.0)


class CrossIndexSearch:
    """跨索引查询器"""

    def __init__(self, es_client, index_discovery: IndexDiscovery):
        self.es_client = es_client
        self.index_discovery = index_discovery
        self.max_result_size = min(int(os.getenv('ES_MAX_RESULT_SIZE', '50')), 100)
        self.query_timeout = os.getenv('ES_QUERY_TIMEOUT', '3s')

    def search(self, conditions: Dict, query: str, max_indices: int = 3, target_indices: Optional[List[str]] = None) -> Dict:
        """跨索引查询"""
        # 1. 发现相关索引
        if target_indices:
            available = set(self.index_discovery.get_all_indices())
            relevant_indices = [(idx, 1.0) for idx in target_indices if idx in available][:max_indices]
        else:
            relevant_indices = self.index_discovery.find_relevant_indices(
                query, conditions, top_n=max_indices
            )

        if not relevant_indices:
            return {'total': 0, 'hits': [], 'indices': []}

        # 2. 并行查询多个索引
        all_results = []
        searched_indices = []

        with ThreadPoolExecutor(max_workers=min(len(relevant_indices), 3)) as executor:
            future_to_index = {
                executor.submit(self._search_single_index, index, conditions, query): index
                for index, _ in relevant_indices
            }

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results = future.result()
                    if results:
                        all_results.extend(results)
                        searched_indices.append(index)
                except Exception as e:
                    print(f'[CrossIndexSearch] Error searching {index}: {e}')

        # 3. 合并并排序结果
        merged_results = self._merge_and_rank_results(all_results)

        return {
            'total': len(merged_results),
            'hits': merged_results,
            'indices': searched_indices
        }

    def _search_single_index(self, index: str, conditions: Dict, query: str = "") -> List[Dict]:
        """在单个索引上查询"""
        try:
            # 构建查询 DSL
            dsl = self._build_dsl(conditions, index, query)

            # 执行查询
            result = self.es_client.search(
                index=index,
                body=dsl,
                size=self.max_result_size,
                request_timeout=5
            )

            # 添加索引信息到结果
            hits = []
            for hit in result['hits']['hits']:
                hit['_source']['_index'] = index
                hit['_source']['_score'] = hit.get('_score', 0)
                hits.append(hit['_source'])

            return hits

        except Exception as e:
            print(f'[CrossIndexSearch] Error in _search_single_index for {index}: {e}')
            return []

    def _has_fields(self, field_names: List[str], *fields: str) -> bool:
        return all(field in field_names for field in fields)

    def _first_existing(self, field_names: List[str], candidates: List[str]) -> Optional[str]:
        for candidate in candidates:
            if candidate in field_names:
                return candidate
        return None

    def _text_fields(self, metadata: Dict, candidates: List[str]) -> List[str]:
        fields = metadata.get('fields', {})
        return [
            field for field in candidates
            if field in fields and fields[field].get('type') in ['text', 'keyword']
        ]

    def _add_multi_match(self, must_clauses: List[Dict], value: str, fields: List[str], boost: bool = False):
        if not value or not fields:
            return
        query_fields = fields
        if boost:
            boosts = {
                'diagnosis': '^3',
                'chief_complaint': '^2',
                'history_present': '^2',
            }
            query_fields = [f"{field}{boosts.get(field, '')}" for field in fields]
        must_clauses.append({
            "multi_match": {
                "query": value,
                "fields": query_fields,
                "type": "best_fields",
                "fuzziness": "AUTO"
            }
        })

    def _add_nested_lab_clause(self, clauses: List[Dict], field_names: List[str], lab_condition: Dict, negate: bool = False):
        if not self._has_fields(field_names, 'lab_results.name', 'lab_results.value'):
            return
        op_map = {'<': 'lt', '<=': 'lte', '>': 'gt', '>=': 'gte'}
        test_name = lab_condition.get('test')
        value = lab_condition.get('value')
        es_op = op_map.get(lab_condition.get('op', ''), 'gte')
        if not test_name or value is None:
            return
        nested = {
            "nested": {
                "path": "lab_results",
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"lab_results.name": test_name}},
                            {"range": {"lab_results.value": {es_op: value}}}
                        ]
                    }
                }
            }
        }
        if negate:
            clauses.append({"bool": {"must_not": [nested]}})
        else:
            clauses.append(nested)

    def _date_range_from_condition(self, tc: Dict) -> Optional[Dict]:
        operator = tc.get('operator', '')
        value = tc.get('value', '')

        if operator == 'recent':
            cutoff = (datetime.now() - timedelta(days=int(value) * 30)).strftime('%Y-%m-%d')
            return {'gte': cutoff}

        if operator in ['since', 'after']:
            match = re.search(r'(\d{4})(?:[-年](\d{1,2}))?', str(value))
            if match:
                year = int(match.group(1))
                month = int(match.group(2) or 1)
                return {'gte': f"{year}-{month:02d}-01"}

        if operator == 'before':
            match = re.search(r'(\d{4})(?:[-年](\d{1,2}))?', str(value))
            if match:
                year = int(match.group(1))
                month = int(match.group(2) or 1)
                return {'lt': f"{year}-{month:02d}-01"}

        if operator == 'year':
            match = re.search(r'(\d{4})', str(value))
            if match:
                year = int(match.group(1))
                return {'gte': f"{year}-01-01", 'lt': f"{year + 1}-01-01"}

        if operator == 'between':
            start_match = re.search(r'(\d{4})-(\d{1,2})', str(tc.get('value_start', '')))
            end_match = re.search(r'(\d{4})-(\d{1,2})', str(tc.get('value_end', '')))
            if start_match and end_match:
                start_year, start_month = int(start_match.group(1)), int(start_match.group(2))
                end_year, end_month = int(end_match.group(1)), int(end_match.group(2))
                if end_month == 12:
                    end_year, end_month = end_year + 1, 1
                else:
                    end_month += 1
                return {
                    'gte': f"{start_year}-{start_month:02d}-01",
                    'lt': f"{end_year}-{end_month:02d}-01",
                }

        return None

    def _build_dsl(self, conditions: Dict, index: str, query: str = "") -> Dict:
        """构建查询 DSL"""
        must_clauses = []
        filter_clauses = []

        # 获取索引字段
        metadata = self.index_discovery.get_index_metadata(index)
        field_names = metadata.get('field_names', [])

        text_candidates = ['diagnosis', 'chief_complaint', 'history_present', 'discharge_summary', 'past_history']
        searchable_text_fields = self._text_fields(metadata, text_candidates)

        # 根据语义条件构建查询
        for field, value in conditions.items():
            if field == 'diagnoses' and isinstance(value, list):
                for item in value:
                    self._add_multi_match(must_clauses, item, searchable_text_fields, boost=True)
                continue

            if field == 'diagnoses_excluded' and isinstance(value, list):
                for item in value:
                    if searchable_text_fields:
                        must_clauses.append({
                            "bool": {
                                "must_not": [{
                                    "multi_match": {
                                        "query": item,
                                        "fields": searchable_text_fields
                                    }
                                }]
                            }
                        })
                continue

            if field == 'conditions_excluded' and isinstance(value, list):
                for item in value:
                    if searchable_text_fields:
                        must_clauses.append({
                            "bool": {
                                "must_not": [{
                                    "multi_match": {
                                        "query": item,
                                        "fields": searchable_text_fields
                                    }
                                }]
                            }
                        })
                continue

            if field == 'symptoms' and isinstance(value, list):
                symptom_fields = self._text_fields(metadata, ['chief_complaint', 'history_present'])
                for item in value:
                    self._add_multi_match(must_clauses, item, symptom_fields)
                continue

            if field == 'procedures' and isinstance(value, list) and 'procedure_notes' in field_names:
                for item in value:
                    must_clauses.append({"match": {"procedure_notes": item}})
                continue

            if field == 'medications' and isinstance(value, list) and 'medications' in field_names:
                for item in value:
                    must_clauses.append({"match": {"medications": item}})
                continue

            if field == 'medications_excluded' and isinstance(value, list) and 'medications' in field_names:
                for item in value:
                    must_clauses.append({"bool": {"must_not": [{"match": {"medications": item}}]}})
                continue

            if field == 'age_min' and 'age' in field_names:
                filter_clauses.append({"range": {"age": {"gte": value}}})
                continue

            if field == 'age_max' and 'age' in field_names:
                filter_clauses.append({"range": {"age": {"lte": value}}})
                continue

            if field == 'gender' and 'gender' in field_names:
                gender_value = '男' if value == 'male' else '女'
                filter_clauses.append({"term": {"gender": gender_value}})
                continue

            if field == 'department' and 'department' in field_names:
                filter_clauses.append({"term": {"department": value}})
                continue

            if field == 'lab_tests' and isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        self._add_nested_lab_clause(filter_clauses, field_names, item)
                continue

            if field == 'lab_tests_excluded' and isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        self._add_nested_lab_clause(must_clauses, field_names, item, negate=True)
                continue

            if field == 'time_conditions' and isinstance(value, list):
                for tc in value:
                    if not isinstance(tc, dict):
                        continue
                    date_field = self._first_existing(field_names, [tc.get('field', ''), 'admission_date', 'discharge_date'])
                    date_range = self._date_range_from_condition(tc)
                    if date_field and date_range:
                        filter_clauses.append({"range": {date_field: date_range}})
                continue

            if field not in field_names:
                continue

            # 根据原始 ES 字段类型构建查询
            field_type = metadata.get('fields', {}).get(field, {}).get('type', 'text')

            if field_type == 'text':
                # 文本字段使用 match
                must_clauses.append({"match": {field: value}})
            elif field_type == 'keyword':
                # 关键字字段使用 term
                filter_clauses.append({"term": {field: value}})
            elif field_type in ['integer', 'long', 'float', 'double']:
                # 数值字段使用 range
                if isinstance(value, dict):
                    range_query = {}
                    if 'gte' in value:
                        range_query['gte'] = value['gte']
                    if 'lte' in value:
                        range_query['lte'] = value['lte']
                    if 'gt' in value:
                        range_query['gt'] = value['gt']
                    if 'lt' in value:
                        range_query['lt'] = value['lt']
                    filter_clauses.append({"range": {field: range_query}})
            elif field_type == 'date':
                # 日期字段使用 range
                if isinstance(value, dict):
                    range_query = {}
                    if 'gte' in value:
                        range_query['gte'] = value['gte']
                    if 'lte' in value:
                        range_query['lte'] = value['lte']
                    filter_clauses.append({"range": {field: range_query}})

        if not must_clauses and query and searchable_text_fields:
            self._add_multi_match(must_clauses, query, searchable_text_fields, boost=True)

        # 构建完整查询
        if must_clauses or filter_clauses:
            return {
                "query": {
                    "bool": {
                        "must": must_clauses if must_clauses else [{"match_all": {}}],
                        "filter": filter_clauses
                    }
                },
                "timeout": self.query_timeout
            }
        else:
            return {"query": {"match_all": {}}, "timeout": self.query_timeout}

    def _merge_and_rank_results(self, results: List[Dict]) -> List[Dict]:
        """合并并排序结果"""
        # 去重（基于患者ID）
        seen_ids = set()
        unique_results = []

        for result in results:
            patient_id = result.get('patient_id') or result.get('id')
            if patient_id and patient_id not in seen_ids:
                seen_ids.add(patient_id)
                unique_results.append(result)
            elif not patient_id:
                unique_results.append(result)

        # 按分数排序
        unique_results.sort(key=lambda x: x.get('_score', 0), reverse=True)

        return unique_results


class IndexRouter:
    """索引路由器"""

    # 路由规则
    ROUTING_RULES = {
        'patient': {
            'keywords': ['患者', '病人', '诊断', '疾病', '科室'],
            'indices': ['n2c2_patients', 'mimic_patients']
        },
        'lab': {
            'keywords': ['检验', '检查', '化验', 'eGFR', '肌酐', '血糖'],
            'indices': ['lab_results']
        },
        'medication': {
            'keywords': ['药物', '用药', '药品', '处方'],
            'indices': ['medications']
        },
        'procedure': {
            'keywords': ['手术', '操作', '治疗'],
            'indices': ['procedures']
        }
    }

    def __init__(self, es_client):
        self.es_client = es_client
        self.index_discovery = IndexDiscovery(es_client)

    def route(self, query: str, conditions: Dict) -> List[str]:
        """根据查询路由到相应索引"""
        target_indices = set()

        # 1. 根据关键词路由
        query_lower = query.lower()
        for category, rule in self.ROUTING_RULES.items():
            for keyword in rule['keywords']:
                if keyword in query_lower:
                    target_indices.update(rule['indices'])
                    break

        # 2. 根据条件字段路由
        for field in conditions.keys():
            if field in ['diagnoses', 'age_min', 'age_max', 'gender', 'department']:
                target_indices.update(self.ROUTING_RULES['patient']['indices'])
            elif field in ['lab_tests', 'lab_tests_excluded']:
                target_indices.update(self.ROUTING_RULES['lab']['indices'])
            elif field in ['medications', 'medications_excluded']:
                target_indices.update(self.ROUTING_RULES['medication']['indices'])
            elif field in ['procedures']:
                target_indices.update(self.ROUTING_RULES['procedure']['indices'])

        # 3. 如果没有匹配的路由，使用索引发现
        if not target_indices:
            relevant_indices = self.index_discovery.find_relevant_indices(
                query, conditions, top_n=3
            )
            target_indices = [idx[0] for idx in relevant_indices]

        return list(target_indices)


# 全局实例
_cross_index_search = None
_index_router = None


def get_cross_index_search(es_client) -> CrossIndexSearch:
    """获取跨索引查询器实例"""
    global _cross_index_search
    if _cross_index_search is None:
        index_discovery = IndexDiscovery(es_client)
        _cross_index_search = CrossIndexSearch(es_client, index_discovery)
    return _cross_index_search


def get_index_router(es_client) -> IndexRouter:
    """获取索引路由器实例"""
    global _index_router
    if _index_router is None:
        _index_router = IndexRouter(es_client)
    return _index_router
