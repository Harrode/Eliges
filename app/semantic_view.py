"""
Semantic view layer for safe ES query construction.

The LLM emits logical conditions; this module maps only approved logical fields
to approved ES query fragments. It does not inspect ES mappings during the
normal path, so the ES schema stays hidden behind a controlled view.
"""
import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


class SemanticViewBuilder:
    """Build ES DSL from a controlled semantic view configuration."""

    ALLOWED_DSL_KEYS = {
        "query", "bool", "must", "filter", "must_not", "should",
        "multi_match", "match", "match_all", "term", "terms", "ids", "values", "range", "nested", "path",
        "fields", "type", "fuzziness", "minimum_should_match",
        "highlight", "number_of_fragments", "fragment_size", "pre_tags",
        "post_tags", "require_field_match", "size", "timeout",
        "gte", "lte", "gt", "lt"
    }

    REQUIRED_FIELD_KEYS = {
        "multi_match": ["es_fields"],
        "must_not_multi_match": ["es_fields"],
        "range_gte": ["es_field"],
        "range_lte": ["es_field"],
        "term": ["es_field"],
        "match": ["es_field"],
        "must_not_match": ["es_field"],
        "nested_range": ["nested_path", "name_field", "value_field"],
        "must_not_nested_range": ["nested_path", "name_field", "value_field"],
        "time_filter": ["allowed_date_fields", "default_date_field"],
    }

    def __init__(self, config_path: str = "config/semantic_views.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.validation_errors = self.validate_config()

    def _load_config(self) -> Dict:
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[SemanticView] Failed to load {self.config_path}: {e}")
            return {}

    def get_view(self, view_name: Optional[str] = None) -> Dict:
        views = self.config.get("views", {})
        selected = view_name or self.config.get("default_view")
        return views.get(selected, {})

    def validate_config(self) -> List[str]:
        """Validate semantic view configuration before it is used to build DSL."""
        errors: List[str] = []
        version = str(self.config.get("version", ""))
        if not re.match(r"^\d+\.\d+$", version):
            errors.append("semantic view config version must use '<major>.<minor>' format")

        default_view = self.config.get("default_view")
        views = self.config.get("views")
        if not isinstance(views, dict) or not views:
            errors.append("semantic view config must define non-empty views")
            return errors
        if default_view not in views:
            errors.append(f"default_view '{default_view}' is not present in views")

        for view_name, view in views.items():
            allowed_indices = view.get("allowed_indices", [])
            if not isinstance(allowed_indices, list) or not allowed_indices:
                errors.append(f"view '{view_name}' must define allowed_indices")

            max_size = view.get("max_size", 50)
            if not isinstance(max_size, int) or max_size < 1 or max_size > 100:
                errors.append(f"view '{view_name}' max_size must be an integer between 1 and 100")

            allowed_query_types = set(view.get("allowed_query_types", []))
            exposed_fields = view.get("exposed_fields", {})
            if not isinstance(exposed_fields, dict) or not exposed_fields:
                errors.append(f"view '{view_name}' must expose at least one field")

            for field_name, field_config in exposed_fields.items():
                query_type = field_config.get("query_type")
                if query_type not in allowed_query_types:
                    errors.append(f"field '{field_name}' query_type '{query_type}' is not allowed by view '{view_name}'")
                    continue
                for required_key in self.REQUIRED_FIELD_KEYS.get(query_type, []):
                    if required_key not in field_config:
                        errors.append(f"field '{field_name}' missing required key '{required_key}'")

        return errors

    def build(self, conditions: Dict, query: str, index_name: Optional[str] = None, view_name: Optional[str] = None) -> Dict:
        if self.validation_errors:
            return {
                "ok": False,
                "error": "; ".join(self.validation_errors),
                "body": None,
                "meta": {"source": "semantic_view", "validation_errors": self.validation_errors}
            }

        view = self.get_view(view_name)
        if not view:
            return {
                "ok": False,
                "error": "semantic view config not found",
                "body": None,
                "meta": {}
            }

        target_index = self._resolve_index(view, index_name)
        exposed_fields = view.get("exposed_fields", {})
        must_clauses: List[Dict] = []
        filter_clauses: List[Dict] = []
        warnings: List[str] = []
        used_fields: List[str] = []
        unsupported_fields: List[str] = []

        for field_name, value in conditions.items():
            field_config = exposed_fields.get(field_name)
            if not field_config:
                unsupported_fields.append(field_name)
                warnings.append(f"语义视图未暴露字段: {field_name}")
                continue

            added = self._append_clause(field_name, value, field_config, must_clauses, filter_clauses, warnings)
            if added:
                used_fields.append(field_name)

        has_exclusion = any(k in conditions for k in ["diagnoses_excluded", "conditions_excluded", "medications_excluded", "lab_tests_excluded"])
        if not must_clauses and not filter_clauses:
            fallback = view.get("fallback_full_text", {})
            if query and not has_exclusion:
                self._append_multi_match(query, fallback, must_clauses)
                used_fields.append("__full_text__")
            else:
                must_clauses.append({"match_all": {}})

        body = {
            "query": {
                "bool": {
                    "must": must_clauses if must_clauses else [{"match_all": {}}],
                    "filter": filter_clauses
                }
            },
            "highlight": self._build_highlight(view),
            "size": min(int(view.get("max_size", 50)), 100),
            "timeout": view.get("timeout", "3s")
        }

        dsl_errors = self.validate_dsl(body)
        if dsl_errors:
            return {
                "ok": False,
                "error": "; ".join(dsl_errors),
                "body": None,
                "meta": {"source": "semantic_view", "dsl_errors": dsl_errors}
            }

        return {
            "ok": True,
            "index": target_index,
            "body": body,
            "meta": {
                "view": view_name or self.config.get("default_view"),
                "index": target_index,
                "used_fields": used_fields,
                "unsupported_fields": unsupported_fields,
                "warnings": warnings,
                "covered_all_conditions": len(unsupported_fields) == 0,
                "source": "semantic_view"
            }
        }

    def validate_dsl(self, body: Dict) -> List[str]:
        """Reject DSL structures outside the semantic view allowlist."""
        errors: List[str] = []

        def walk(value: Any, path: str = "$"):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key not in self.ALLOWED_DSL_KEYS and not self._is_field_name_key(path):
                        errors.append(f"DSL key '{key}' is not allowed at {path}")
                    walk(child, f"{path}.{key}")
            elif isinstance(value, list):
                for idx, item in enumerate(value):
                    walk(item, f"{path}[{idx}]")

        walk(body)
        return errors

    def _is_field_name_key(self, path: str) -> bool:
        return (
            path.endswith(".match")
            or path.endswith(".term")
            or path.endswith(".terms")
            or path.endswith(".range")
            or path.endswith(".highlight.fields")
        )

    def _resolve_index(self, view: Dict, index_name: Optional[str]) -> str:
        allowed = set(view.get("allowed_indices", []))
        if index_name and (not allowed or index_name in allowed):
            return index_name
        return view.get("default_index", index_name or "")

    def _append_clause(self, field_name: str, value: Any, field_config: Dict, must_clauses: List[Dict], filter_clauses: List[Dict], warnings: List[str]) -> bool:
        query_type = field_config.get("query_type")

        if query_type == "multi_match" and isinstance(value, list):
            for item in value:
                self._append_multi_match(item, field_config, must_clauses)
            return bool(value)

        if query_type == "must_not_multi_match" and isinstance(value, list):
            for item in value:
                # Use match_phrase for each field to avoid single-character token matching
                should_clauses = []
                for field in field_config.get("es_fields", []):
                    should_clauses.append({"match_phrase": {field: item}})
                must_clauses.append({
                    "bool": {
                        "must_not": [{
                            "bool": {
                                "should": should_clauses,
                                "minimum_should_match": 1
                            }
                        }]
                    }
                })
            return bool(value)

        if query_type == "range_gte" and value is not None:
            filter_clauses.append({"range": {field_config["es_field"]: {"gte": value}}})
            return True

        if query_type == "range_lte" and value is not None:
            filter_clauses.append({"range": {field_config["es_field"]: {"lte": value}}})
            return True

        if query_type == "term" and value is not None:
            es_field = field_config["es_field"]
            if es_field == "gender":
                # Gender may be stored in English or Chinese across indices.
                variants = {
                    "male": ["male", "Male", "M", "男", "男性"],
                    "female": ["female", "Female", "F", "女", "女性"],
                }.get(value, [value])
                filter_clauses.append({"terms": {es_field: variants}})
            else:
                mapped_value = field_config.get("value_mapping", {}).get(value, value)
                filter_clauses.append({"term": {es_field: mapped_value}})
            return True

        if query_type == "match" and isinstance(value, list):
            for item in value:
                must_clauses.append({"match": {field_config["es_field"]: item}})
            return bool(value)

        if query_type == "must_not_match" and isinstance(value, list):
            for item in value:
                must_clauses.append({"bool": {"must_not": [{"match": {field_config["es_field"]: item}}]}})
            return bool(value)

        if query_type == "nested_range" and isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    filter_clauses.append(self._nested_range_clause(item, field_config))
            return bool(value)

        if query_type == "must_not_nested_range" and isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    must_clauses.append({"bool": {"must_not": [self._nested_range_clause(item, field_config)]}})
            return bool(value)

        if query_type == "time_filter" and isinstance(value, list):
            added = False
            for item in value:
                if not isinstance(item, dict):
                    continue
                date_field = self._safe_date_field(item.get("field"), field_config)
                date_range = self._date_range(item)
                if date_field and date_range:
                    filter_clauses.append({"range": {date_field: date_range}})
                    added = True
            return added

        warnings.append(f"语义字段 {field_name} 的查询类型暂不支持: {query_type}")
        return False

    def _append_multi_match(self, value: str, field_config: Dict, must_clauses: List[Dict]):
        if not value:
            return
        clause = {
            "multi_match": {
                "query": value,
                "fields": field_config.get("es_fields", []),
                "type": "best_fields"
            }
        }
        if field_config.get("fuzziness"):
            clause["multi_match"]["fuzziness"] = field_config["fuzziness"]
        must_clauses.append(clause)

    def _nested_range_clause(self, item: Dict, field_config: Dict) -> Dict:
        aliases = field_config.get("aliases", {})
        op_mapping = field_config.get("op_mapping", {})
        test_name = item.get("test", "")
        possible_names = aliases.get(test_name, [test_name])
        name_queries = [{"match": {field_config["name_field"]: name}} for name in possible_names if name]
        es_op = op_mapping.get(item.get("op", ""), "gte")

        return {
            "nested": {
                "path": field_config["nested_path"],
                "query": {
                    "bool": {
                        "must": [
                            {"bool": {"should": name_queries, "minimum_should_match": 1}},
                            {"range": {field_config["value_field"]: {es_op: item.get("value", 0)}}}
                        ]
                    }
                }
            }
        }

    def _safe_date_field(self, requested: Optional[str], field_config: Dict) -> Optional[str]:
        allowed = field_config.get("allowed_date_fields", [])
        if requested in allowed:
            return requested
        return field_config.get("default_date_field")

    def _date_range(self, condition: Dict) -> Optional[Dict]:
        operator = condition.get("operator", "")
        value = condition.get("value", "")

        if operator == "recent":
            try:
                cutoff = datetime.now() - timedelta(days=int(value) * 30)
                return {"gte": cutoff.strftime("%Y-%m-%d")}
            except Exception:
                return None

        if operator in ["since", "after"]:
            date_value = self._parse_year_month(value)
            return {"gte": date_value} if date_value else None

        if operator == "before":
            date_value = self._parse_year_month(value)
            return {"lt": date_value} if date_value else None

        if operator == "year":
            match = re.search(r"(\d{4})", str(value))
            if match:
                year = int(match.group(1))
                return {"gte": f"{year}-01-01", "lt": f"{year + 1}-01-01"}

        if operator == "between":
            start = self._parse_year_month(condition.get("value_start", ""))
            end = self._parse_year_month(condition.get("value_end", ""), next_month=True)
            if start and end:
                return {"gte": start, "lt": end}

        return None

    def _parse_year_month(self, value: Any, next_month: bool = False) -> Optional[str]:
        match = re.search(r"(\d{4})(?:[-年](\d{1,2}))?", str(value))
        if not match:
            return None
        year = int(match.group(1))
        month = int(match.group(2) or 1)
        if next_month:
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1
        return f"{year}-{month:02d}-01"

    def _build_highlight(self, view: Dict) -> Dict:
        fields = {}
        for field in view.get("highlight_fields", []):
            fields[field] = {
                "number_of_fragments": 2,
                "fragment_size": 150,
                "pre_tags": ["<mark>"],
                "post_tags": ["</mark>"]
            }
        if "diagnosis" in fields:
            fields["diagnosis"]["number_of_fragments"] = 0
        return {"fields": fields, "require_field_match": False}


_semantic_view_builder = None


def get_semantic_view_builder() -> SemanticViewBuilder:
    global _semantic_view_builder
    if _semantic_view_builder is None:
        _semantic_view_builder = SemanticViewBuilder()
    return _semantic_view_builder
