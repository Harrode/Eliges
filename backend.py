"""
Clinical Eligibility Screening System - Backend
Architecture: LLM NLU -> ES Search -> Rule Filter -> LLM Analysis
Supports: model toggle, retrieval mode toggle, Layer4 loop-back
Enhanced: Field Mapping, Cross-Index Search, Post-Processing Code Generation
"""
import hashlib, json, os, re, time, uuid, traceback
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from data import init_data as data_module

load_dotenv()

try:
    from app.semantic_view import get_semantic_view_builder
    SEMANTIC_VIEW_ENABLED = True
except ImportError as e:
    SEMANTIC_VIEW_ENABLED = False
    print(f'[SemanticView] Semantic view module not available: {e}')

# Import enhanced modules
try:
    from app.enhanced_search import get_enhanced_search_engine
    from app.enhanced_api import router as enhanced_router
    ENHANCED_MODE = True
    print('[Enhanced] Enhanced modules loaded successfully')
except ImportError as e:
    ENHANCED_MODE = False
    print(f'[Enhanced] Enhanced modules not available: {e}')

# -- Config --
LLM_MODEL = os.getenv('LLM_MODEL', 'deepseek-v4-flash')
LLM_API_KEY = os.getenv('LLM_API_KEY', '')
LLM_BASE = os.getenv('LLM_BASE_URL', 'https://api.deepseek.com')
ES_HOST = os.getenv('ES_HOST', 'http://localhost:9200')
ES_INDEX = os.getenv('ES_INDEX', 'n2c2_patients')
APP_API_KEY = os.getenv('APP_API_KEY', '')
AUDIT_LOG_ENABLED = os.getenv('AUDIT_LOG_ENABLED', 'true').lower() == 'true'
AUDIT_LOG_PATH = os.getenv('AUDIT_LOG_PATH', 'logs/audit.log')
DESENSITIZE_RESPONSES = os.getenv('DESENSITIZE_RESPONSES', 'false').lower() == 'true'
ES_QUERY_TIMEOUT = os.getenv('ES_QUERY_TIMEOUT', '3s')
ES_MAX_RESULT_SIZE = int(os.getenv('ES_MAX_RESULT_SIZE', '50'))
ENABLE_TWO_STAGE_SEARCH = os.getenv('ENABLE_TWO_STAGE_SEARCH', 'true').lower() == 'true'
ES_RECALL_SIZE = min(int(os.getenv('ES_RECALL_SIZE', '100')), 200)

# -- Load Field Mapping --
FIELD_MAPPING = {}
try:
    with open('config/field_mapping.json', 'r', encoding='utf-8') as f:
        mapping_config = json.load(f)
        FIELD_MAPPING = mapping_config.get('field_mapping', {})
        print(f'[Config] Loaded config/field_mapping.json: {len(FIELD_MAPPING)} fields')
except Exception as e:
    print(f'[Config] Warning: Could not load config/field_mapping.json: {e}')

app = FastAPI(title='Eligibility Screening System')
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def write_audit_event(event: dict):
    """Append audit events as JSONL without blocking the request path on errors."""
    if not AUDIT_LOG_ENABLED:
        return
    try:
        os.makedirs(os.path.dirname(AUDIT_LOG_PATH) or '.', exist_ok=True)
        event = {
            'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            **event
        }
        with open(AUDIT_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f'[Audit] Failed to write audit event: {e}')

def get_request_user(request: Request) -> str:
    return request.headers.get('X-User-ID') or request.headers.get('X-Forwarded-User') or 'anonymous'

def is_authorized_request(request: Request) -> bool:
    """API key auth is enabled only when APP_API_KEY is configured."""
    if not APP_API_KEY:
        return True
    provided = request.headers.get('X-API-Key', '')
    auth = request.headers.get('Authorization', '')
    if auth.lower().startswith('bearer '):
        provided = auth.split(' ', 1)[1]
    return provided == APP_API_KEY

@app.middleware("http")
async def security_and_audit_middleware(request: Request, call_next):
    start = time.time()
    if request.url.path.startswith('/api') and not is_authorized_request(request):
        write_audit_event({
            'event': 'auth_failed',
            'path': request.url.path,
            'method': request.method,
            'user': get_request_user(request)
        })
        return JSONResponse(status_code=401, content={'detail': 'Unauthorized'})

    response = await call_next(request)
    if request.url.path.startswith('/api') or request.url.path == '/health':
        write_audit_event({
            'event': 'request',
            'path': request.url.path,
            'method': request.method,
            'status_code': response.status_code,
            'duration_ms': int((time.time() - start) * 1000),
            'user': get_request_user(request)
        })
    return response

# Include enhanced router if available
if ENHANCED_MODE:
    app.include_router(enhanced_router)
    print('[Enhanced] Enhanced API routes registered')

# -- LLM Clients --
llm_client = OpenAI(base_url=LLM_BASE, api_key=LLM_API_KEY, timeout=90.0, max_retries=3) if LLM_API_KEY else None
NO_THINKING_EXTRA_BODY = {
    "enable_thinking": False,
    "chat_template_kwargs": {"enable_thinking": False},
}

# -- Runtime State --
current_model = LLM_MODEL
current_retrieval = 'auto'
version_store = {}
MAX_LOOP = 2

# -- ES Connection --
es_client = None
es_available = False

def connect_es():
    global es_client, es_available
    try:
        from elasticsearch import Elasticsearch
        es_client = Elasticsearch(ES_HOST)
        es_available = es_client.ping()
        if es_available:
            print(f'[ES] Connected to {ES_HOST}')
        else:
            print(f'[ES] Ping failed to {ES_HOST}')
    except Exception as e:
        es_available = False
        print(f'[ES] Connection failed: {e}')

connect_es()


def auto_index_data():
    """Auto-index patient data into Elasticsearch if index doesn't exist or is empty."""
    if not es_available:
        return

    # Index built-in patients from data module
    _ensure_index(ES_INDEX, data_module.PATIENTS)

    # Auto-index TrialGPT patients if the file exists
    trialgpt_path = Path(__file__).resolve().parent / "TrialGPT" / "dataset" / "sigir" / "queries.jsonl"
    if trialgpt_path.exists():
        _ensure_index_from_jsonl("trialgpt_patients", trialgpt_path, parse_trialgpt_patient)

    # Auto-index the synthetic English demo cohorts (PHI-free) if present.
    synthetic_dir = Path(__file__).resolve().parent / "data" / "synthetic"
    if synthetic_dir.exists():
        for path in sorted(synthetic_dir.glob("*.json")):
            index_name = path.stem  # e.g. emr_cardiometabolic
            try:
                patients = json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f'[ES] Failed to read synthetic cohort {path.name}: {e}')
                continue
            _ensure_index(index_name, patients)
            _add_allowed_index(index_name)


def _ensure_index(index_name: str, patients: list):
    """Ensure index exists and has data."""
    if not es_client.indices.exists(index=index_name):
        _create_patient_index(index_name)
        _bulk_index(index_name, patients)
        print(f'[ES] Auto-created index "{index_name}" with {len(patients)} patients')
    else:
        count = es_client.count(index=index_name)['count']
        if count == 0:
            _bulk_index(index_name, patients)
            print(f'[ES] Auto-populated empty index "{index_name}" with {len(patients)} patients')


def _ensure_index_from_jsonl(index_name: str, path: Path, parser):
    """Ensure index exists from a JSONL file."""
    if not es_client.indices.exists(index=index_name):
        _create_patient_index(index_name)
        patients = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                patients.append(parser(data))
        _bulk_index(index_name, patients)
        print(f'[ES] Auto-created index "{index_name}" with {len(patients)} patients from {path.name}')
        # Add to allowed indices
        _add_allowed_index(index_name)
    else:
        count = es_client.count(index=index_name)['count']
        if count == 0:
            patients = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    patients.append(parser(data))
            _bulk_index(index_name, patients)
            print(f'[ES] Auto-populated empty index "{index_name}" with {len(patients)} patients')


def parse_trialgpt_patient(data: dict) -> dict:
    """Parse TrialGPT patient from JSONL format."""
    import re as _re
    text = data.get('text', '')
    pid = data.get('_id', '')

    age_match = _re.search(r'(\d+)-year-old', text.lower())
    age = int(age_match.group(1)) if age_match else None

    gender = None
    if _re.search(r'\b(female|woman|she|her)\b', text.lower()):
        gender = '女'
    elif _re.search(r'\b(male|man|he|his)\b', text.lower()):
        gender = '男'

    return {
        'patient_id': pid,
        'name': pid,
        'diagnosis': text[:500],
        'age': age,
        'gender': gender,
        'history_present': text,
        'medications': '',
        'lab_results': []
    }


def _create_patient_index(index_name: str):
    """Create ES index with standard patient mapping."""
    mapping = {
        "mappings": {
            "properties": {
                "patient_id": {"type": "keyword"},
                "name": {"type": "text"},
                "diagnosis": {"type": "text"},
                "age": {"type": "integer"},
                "gender": {"type": "keyword"},
                "department": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "history_present": {"type": "text"},
                "medications": {"type": "text"},
                "procedure_notes": {"type": "text"},
                "lab_results": {
                    "type": "nested",
                    "properties": {
                        "name": {"type": "text"},
                        "value": {"type": "float"}
                    }
                }
            }
        }
    }
    es_client.indices.create(index=index_name, body=mapping)


def _bulk_index(index_name: str, patients: list):
    """Bulk index patients into ES."""
    for p in patients:
        es_client.index(index=index_name, id=p.get('patient_id', ''), body=p, refresh=True)


def _add_allowed_index(index_name: str):
    """Add index to allowed_indices in semantic_views.json."""
    try:
        config_path = Path(__file__).resolve().parent / "config" / "semantic_views.json"
        config = json.loads(config_path.read_text(encoding='utf-8'))
        view = config.get('views', {}).get('patient_eligibility_view', {})
        allowed = view.get('allowed_indices', [])
        if index_name not in allowed:
            allowed.append(index_name)
            view['allowed_indices'] = allowed
            config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
            print(f'[ES] Added "{index_name}" to allowed_indices')
    except Exception as e:
        print(f'[ES] Failed to update allowed_indices: {e}')


# Auto-index on startup
auto_index_data()


# Gender values may be stored in English (synthetic English cohorts) or Chinese
# (legacy n2c2/MIMIC indices); match both so a single query works everywhere.
GENDER_VARIANTS = {
    'male': ['male', 'Male', 'M', 'm', '男', '男性'],
    'female': ['female', 'Female', 'F', 'f', '女', '女性'],
}


def gender_matches(stored: str, cond_value: str) -> bool:
    """True if a stored gender string matches a parsed 'male'/'female' condition."""
    return str(stored).strip() in GENDER_VARIANTS.get(cond_value, [cond_value])


def mask_name(name: str) -> str:
    if not name:
        return name
    if len(name) <= 1:
        return '*'
    return name[0] + '*' * (len(name) - 1)

def redact_patient_record(record: dict) -> dict:
    """Return a shallow redacted copy for privacy-preserving responses."""
    if not DESENSITIZE_RESPONSES or not isinstance(record, dict):
        return record
    redacted = dict(record)
    if 'name' in redacted:
        redacted['name'] = mask_name(str(redacted.get('name', '')))
    if 'patient_id' in redacted:
        redacted['patient_id'] = f"patient-{hashlib.sha256(str(redacted['patient_id']).encode()).hexdigest()[:8]}"
    if 'id' in redacted:
        redacted['id'] = f"patient-{hashlib.sha256(str(redacted['id']).encode()).hexdigest()[:8]}"
    return redacted

def format_es_hits(resp, dsl: dict) -> dict:
    """Convert ES hits into the response shape used by the UI."""
    hits = []
    for hit in resp['hits']['hits']:
        src = hit['_source']
        hits.append(redact_patient_record({
            'id': src.get('patient_id'),
            'name': src.get('name'),
            'age': src.get('age'),
            'gender': src.get('gender'),
            'diagnosis': src.get('diagnosis'),
            'chief_complaint': src.get('chief_complaint'),
            'department': src.get('department'),
            'hospital': src.get('hospital'),
            'admission_date': src.get('admission_date'),
            'discharge_date': src.get('discharge_date'),
            'score': round(hit.get('_score') or 0, 2),
            'highlighted': {k: v for k, v in hit.get('highlight', {}).items()},
            'lab_results': src.get('lab_results', []),
            'medications': src.get('medications', ''),
            'history_present': src.get('history_present', ''),
            'discharge_summary': src.get('discharge_summary', '')
        }))
    return {'total': resp['hits']['total']['value'], 'hits': hits, 'dsl': dsl}

def semantic_view_search(conds: dict, query: str) -> Optional[dict]:
    """Try the safe semantic view before falling back to direct ES field mapping."""
    if not SEMANTIC_VIEW_ENABLED:
        return None
    try:
        builder = get_semantic_view_builder()
        built = builder.build(conds, query, index_name=ES_INDEX)
        if not built.get('ok') or not built.get('body'):
            print(f'[SemanticView] Build skipped: {built.get("error")}')
            return None

        index_name = built.get('index') or ES_INDEX
        body = built['body']
        print(f'[SemanticView] Searching view={built["meta"].get("view")} index={index_name}')
        print(f'[SemanticView] Query: {json.dumps(body, ensure_ascii=False)[:200]}')
        resp = es_client.search(index=index_name, body=body)
        total = resp['hits']['total']['value']
        print(f'[SemanticView] Hits: {total}')

        dsl = {
            'strategy': 'semantic_view',
            'semantic_view': built['meta'],
            'query': body
        }
        result = format_es_hits(resp, dsl)

        # If the view misses, allow the legacy mapper to inspect concrete ES fields.
        if total == 0:
            print('[SemanticView] No hits, falling back to legacy ES field mapping')
            return None

        return result
    except Exception as e:
        print(f'[SemanticView] Search error, falling back to legacy ES mapping: {e}')
        return None

def get_allowed_indices() -> list:
    if not SEMANTIC_VIEW_ENABLED:
        return [ES_INDEX]
    try:
        builder = get_semantic_view_builder()
        view = builder.get_view()
        return view.get('allowed_indices', [ES_INDEX])
    except Exception:
        return [ES_INDEX]

def build_recall_conditions(conds: dict) -> dict:
    """Keep broad recall fields for stage-1 candidate retrieval."""
    recall_keys = ['diagnoses', 'diagnoses_any', 'symptoms', 'procedures', 'medications', 'department']
    recall = {k: v for k, v in conds.items() if k in recall_keys and v}
    temporal = conds.get('temporal_event') or {}
    if temporal.get('event_terms') and not recall.get('diagnoses_any'):
        recall['diagnoses_any'] = temporal['event_terms']
    return recall


def _is_lab_primary_query(conds: dict) -> bool:
    if not conds.get('lab_tests'):
        return False
    for key in ('diagnoses', 'diagnoses_any', 'symptoms', 'procedures', 'medications', 'department', 'age_min', 'age_max', 'gender'):
        if conds.get(key):
            return False
    return True


def _lab_search_keywords(lab_tests: list) -> list:
    keywords = set()
    for item in lab_tests or []:
        if not isinstance(item, dict):
            continue
        # Support both 'test' and 'test_name' fields
        test = str(item.get('test', '') or item.get('test_name', '') or item.get('name', '')).lower()
        if 'a1c' in test or 'hemoglobin' in test:
            keywords.update(['hba1c', 'a1c', 'hemoglobin a1c', 'glycohemoglobin'])
        if 'creatinine' in test:
            keywords.update(['creatinine', 'creat'])
    return sorted(keywords)


def _lab_wide_recall_clause(lab_tests: list) -> dict:
    keywords = _lab_search_keywords(lab_tests)
    query_text = ' '.join(keywords) if keywords else 'hba1c creatinine laboratory'
    return {
        "multi_match": {
            "query": query_text,
            "fields": ["diagnosis", "history_present", "discharge_summary", "chief_complaint"],
            "type": "best_fields",
        }
    }

def add_candidate_id_filter(body: dict, candidate_ids: list) -> dict:
    """Restrict a DSL body to candidate patient IDs from stage 1."""
    body = json.loads(json.dumps(body))
    bool_query = body.setdefault('query', {}).setdefault('bool', {})
    filters = bool_query.setdefault('filter', [])
    if isinstance(filters, dict):
        filters = [filters]
        bool_query['filter'] = filters
    filters.append({
        "bool": {
            "should": [
                {"terms": {"patient_id": candidate_ids}},
                {"ids": {"values": candidate_ids}}
            ],
            "minimum_should_match": 1
        }
    })
    body['size'] = min(ES_MAX_RESULT_SIZE, 100)
    body['timeout'] = ES_QUERY_TIMEOUT
    return body

def extract_candidate_ids(resp) -> list:
    ids = []
    seen = set()
    for hit in resp.get('hits', {}).get('hits', []):
        source = hit.get('_source', {})
        candidates = [source.get('patient_id'), hit.get('_id')]
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                ids.append(candidate)
    return ids

def summarize_search_dsl(dsl):
    """Build a compact display summary while preserving raw DSL separately."""
    if not isinstance(dsl, dict):
        return dsl

    strategy = dsl.get('strategy')
    if strategy == 'two_stage_semantic_view':
        recall = dsl.get('recall', {})
        precise = dsl.get('precise', {})
        precise_meta = precise.get('semantic_view', {})
        return {
            'strategy': 'two_stage_semantic_view',
            'display_name': '二阶段语义检索',
            'recall_conditions': recall.get('conditions', {}),
            'candidate_count': recall.get('candidate_count', 0),
            'precise_conditions': precise.get('conditions', {}),
            'used_fields': precise_meta.get('used_fields', []),
            'unsupported_fields': precise_meta.get('unsupported_fields', []),
            'warnings': precise_meta.get('warnings', []),
            'backup_strategy': dsl.get('backup_strategy', 'single_stage_semantic_view')
        }

    if strategy == 'semantic_view':
        meta = dsl.get('semantic_view', {})
        return {
            'strategy': 'semantic_view',
            'display_name': '单阶段语义视图',
            'conditions': meta.get('used_fields', []),
            'used_fields': meta.get('used_fields', []),
            'unsupported_fields': meta.get('unsupported_fields', []),
            'warnings': meta.get('warnings', [])
        }

    return {
        'strategy': 'legacy_es_query_builder',
        'display_name': '传统 ES Query Builder',
        'note': '完整 DSL 已放入 raw_dsl'
    }


_STRATEGY_LABELS = {
    'two_stage_semantic_view': '两阶段语义检索 (Stage1 宽召回 → Stage2 精查)',
    'semantic_view': '单阶段语义视图',
    'legacy_es_query_builder': '传统 ES 字段映射',
    'memory_engine': '内存检索 (ES 不可用时的回退)',
}


def get_index_patient_count(index_name: str, search_backend: str) -> Optional[int]:
    """Total documents in the active patient index (N for the L3 funnel)."""
    if search_backend == 'elasticsearch' and es_available and index_name:
        try:
            return int(es_client.count(index=index_name)['count'])
        except Exception as e:
            print(f'[Search] Index count failed: {e}')
            return None
    if search_backend == 'memory':
        return mem_engine.count()
    return None


def build_retrieval_pipeline(
    raw_results: Optional[dict],
    search_backend: str,
    es_index: str,
    hits_before_filter: int,
    filtered_count: int,
    loop_count: int,
    search_ms: int,
) -> dict:
    """Structured L3 stats for the Demo funnel (N → K1 → K2 → L4)."""
    dsl = (raw_results or {}).get('dsl') if isinstance(raw_results, dict) else None
    strategy = 'memory_engine'
    stage1_candidates = None
    stage2_es_total = hits_before_filter

    if search_backend == 'elasticsearch' and isinstance(dsl, dict):
        strategy = dsl.get('strategy') or 'legacy_es_query_builder'
        stage2_es_total = int(raw_results.get('total', hits_before_filter) or 0)
        if strategy == 'two_stage_semantic_view':
            stage1_candidates = dsl.get('recall', {}).get('candidate_count')

    index_total = get_index_patient_count(
        es_index if search_backend == 'elasticsearch' else '',
        search_backend,
    )

    funnel = [
        {
            'id': 'N',
            'label': '患者库 N',
            'count': index_total,
            'detail': es_index if search_backend == 'elasticsearch' else 'in-memory cohort',
        },
    ]
    if strategy == 'two_stage_semantic_view':
        funnel.append({
            'id': 'K1',
            'label': f'Stage1 宽召回 (cap {ES_RECALL_SIZE})',
            'count': stage1_candidates,
            'detail': 'diagnoses / symptoms / medications 等宽条件',
        })
    funnel.extend([
        {
            'id': 'K2',
            'label': f'Stage2 / ES 命中 (cap {ES_MAX_RESULT_SIZE})',
            'count': hits_before_filter,
            'detail': f'ES 报告 total={stage2_es_total}' if search_backend == 'elasticsearch' else 'memory score ranking',
        },
        {
            'id': 'L4',
            'label': 'L4 规则过滤后',
            'count': filtered_count,
            'detail': f'loop-back ×{loop_count}' if loop_count else '硬约束校验',
        },
    ])

    return {
        'layer': 'Layer 3: Elasticsearch 检索',
        'description': '在全库上执行检索，将 N 缩小到 K；L4 仅在 K 上过滤',
        'backend': search_backend,
        'index_name': es_index if search_backend == 'elasticsearch' else 'memory',
        'index_total': index_total,
        'strategy': strategy,
        'strategy_label': _STRATEGY_LABELS.get(strategy, strategy),
        'stage1_cap': ES_RECALL_SIZE if strategy == 'two_stage_semantic_view' else None,
        'stage1_candidates': stage1_candidates,
        'stage2_cap': ES_MAX_RESULT_SIZE,
        'stage2_hits': hits_before_filter,
        'stage2_es_total': stage2_es_total,
        'after_l4_filter': filtered_count,
        'loop_count': loop_count,
        'time_ms': search_ms,
        'funnel': funnel,
    }


def two_stage_semantic_search(conds: dict, query: str) -> Optional[dict]:
    """Stage 1 broad recall, then stage 2 precise query restricted to candidate IDs."""
    if not SEMANTIC_VIEW_ENABLED or not ENABLE_TWO_STAGE_SEARCH:
        return None
    try:
        builder = get_semantic_view_builder()
        recall_conditions = build_recall_conditions(conds)
        if not recall_conditions and _is_lab_primary_query(conds):
            print('[TwoStageSearch] Lab-only query, using match_all recall')
            recall_conditions = {'lab_tests': conds.get('lab_tests')}
        if not recall_conditions:
            print('[TwoStageSearch] No broad recall conditions, using single-stage semantic backup')
            return None

        recall_built = builder.build(recall_conditions, query, index_name=ES_INDEX)
        if not recall_built.get('ok') or not recall_built.get('body'):
            print(f'[TwoStageSearch] Recall build skipped: {recall_built.get("error")}')
            return None

        index_name = recall_built.get('index') or ES_INDEX
        recall_body = recall_built['body']
        recall_body['size'] = ES_RECALL_SIZE
        recall_body['timeout'] = ES_QUERY_TIMEOUT
        print(f'[TwoStageSearch] Stage 1 recall index={index_name} size={ES_RECALL_SIZE}')
        recall_resp = es_client.search(index=index_name, body=recall_body, request_timeout=5)
        candidate_ids = extract_candidate_ids(recall_resp)
        print(f'[TwoStageSearch] Stage 1 candidates: {len(candidate_ids)}')
        if not candidate_ids:
            return None

        precise_built = builder.build(conds, query, index_name=index_name)
        if not precise_built.get('ok') or not precise_built.get('body'):
            print(f'[TwoStageSearch] Precise build skipped: {precise_built.get("error")}')
            return None

        precise_body = add_candidate_id_filter(precise_built['body'], candidate_ids)
        dsl_errors = builder.validate_dsl(precise_body)
        if dsl_errors:
            print(f'[TwoStageSearch] Precise DSL rejected: {dsl_errors}')
            return None

        print(f'[TwoStageSearch] Stage 2 precise query over {len(candidate_ids)} candidates')
        precise_resp = es_client.search(index=index_name, body=precise_body, request_timeout=5)
        total = precise_resp['hits']['total']['value']
        print(f'[TwoStageSearch] Stage 2 hits: {total}')
        if total == 0:
            print('[TwoStageSearch] No precise hits, using single-stage semantic backup')
            return None

        dsl = {
            'strategy': 'two_stage_semantic_view',
            'recall': {
                'conditions': recall_conditions,
                'candidate_count': len(candidate_ids),
                'query': recall_body,
                'semantic_view': recall_built.get('meta', {})
            },
            'precise': {
                'conditions': conds,
                'query': precise_body,
                'semantic_view': precise_built.get('meta', {})
            },
            'backup_strategy': 'single_stage_semantic_view'
        }
        return format_es_hits(precise_resp, dsl)
    except Exception as e:
        print(f'[TwoStageSearch] Error, using single-stage semantic backup: {e}')
        return None

# -- In-Memory Fallback Engine --
class MemEngine:
    def __init__(self, records):
        self.records = records

    def _tk(self, text):
        if not text: return []
        tokens = re.findall('[一-鿿]+|[a-zA-Z0-9\.]+', text.lower())
        out = []
        for t in tokens:
            if re.match('[一-鿿]', t):
                for i in range(len(t)):
                    for j in range(i+1, min(i+4, len(t)+1)):
                        out.append(t[i:j])
            else:
                out.append(t)
        return out

    def _score(self, rec, qtokens, conds):
        s = 0.0
        searchable = ' '.join(str(rec.get(f,'')) for f in ['diagnosis','chief_complaint','history_present','discharge_summary'])
        sl = searchable.lower()
        for t in qtokens:
            if t in sl: s += 2.0
        if 'diagnoses' in conds:
            for d in conds['diagnoses']:
                dl = d.lower()
                if dl in str(rec.get('diagnosis','')).lower(): s += 5.0
                if dl in str(rec.get('history_present','')).lower(): s += 3.0
                if dl in str(rec.get('chief_complaint','')).lower(): s += 3.0
        if 'age_min' in conds or 'age_max' in conds:
            age = rec.get('age', 0)
            ok = True
            if 'age_min' in conds and age < conds['age_min']: ok = False
            if 'age_max' in conds and age > conds['age_max']: ok = False
            if ok: s += 2.0
        if 'gender' in conds:
            g = rec.get('gender','')
            if gender_matches(g, conds['gender']): s += 1.5
        if 'lab_tests' in conds:
            labs = rec.get('lab_results', [])
            for lc in conds['lab_tests']:
                for lab in labs:
                    if lc.get('test','').lower() in str(lab.get('name','')).lower():
                        v = lab.get('value', 0)
                        op = lc.get('op','')
                        tgt = lc.get('value', 0)
                        if op == '<' and v < tgt: s += 3.0
                        elif op == '>' and v > tgt: s += 3.0
                        elif op == '<=' and v <= tgt: s += 3.0
                        elif op == '>=' and v >= tgt: s += 3.0
        return s

    def _hl(self, text, qtokens):
        if not text: return text
        h = text
        for t in set(qtokens):
            if len(t) >= 2:
                h = re.compile(re.escape(t), re.IGNORECASE).sub(lambda m: f'<mark>{m.group()}</mark>', h)
        return h

    def search(self, conds, qtext):
        qtokens = self._tk(qtext)
        scored = [(self._score(r, qtokens, conds), r) for r in self.records]
        scored.sort(key=lambda x: -x[0])
        hits = []
        for score, rec in scored:
            hl = {}
            for f in ['diagnosis','chief_complaint','history_present','discharge_summary']:
                h = self._hl(str(rec.get(f,'')), qtokens)
                if '<mark>' in h: hl[f] = [h]
            hits.append({'score': round(score,2), 'source': rec, 'highlight': hl})
        return {
            'total': len([h for h in hits if h['score']>0]),
            'hits': [h for h in hits if h['score']>0],
            'all_scores': [(r['patient_id'], round(self._score(r, qtokens, conds),2)) for r in self.records]
        }

    def count(self): return len(self.records)

mem_engine = MemEngine(data_module.PATIENTS)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning blocks emitted by reasoning models."""
    import re as _re
    return _re.sub(r'<think>.*?</think>', '', text, flags=_re.DOTALL).strip()


# -- LLM NLU: Parse natural language to structured conditions --
def _normalize_lab_tests(lab_tests: list) -> list:
    """Normalize lab_tests format from LLM output to standard format."""
    normalized = []
    for item in lab_tests:
        if not isinstance(item, dict):
            # Handle plain string like "HbA1c" — pass through as name-only entry
            if isinstance(item, str) and item.strip():
                normalized.append({'test': item.strip(), 'op': '', 'value': None,
                                    'value_min': None, 'value_max': None})
            continue
        # Map field names
        test = item.get('test', '') or item.get('test_name', '') or item.get('name', '')
        op = item.get('op', '') or item.get('operator', '')
        value = item.get('value')
        value_min = item.get('value_min') or item.get('min_value') or item.get('min')
        value_max = item.get('value_max') or item.get('max_value') or item.get('max')

        # Handle range: [min, max] format (e.g. from qwen)
        range_val = item.get('range')
        if isinstance(range_val, list) and len(range_val) == 2:
            value_min = value_min if value_min is not None else range_val[0]
            value_max = value_max if value_max is not None else range_val[1]

        # Normalize operator
        if not op:
            if value_min is not None and value_max is not None:
                op = 'between'
            elif value is not None:
                op = '>='

        normalized.append({
            'test': test,
            'op': op,
            'value': value,
            'value_min': value_min,
            'value_max': value_max,
        })
    return normalized


def llm_parse_query(
    query: str,
    feedback: str = "",
    profile_id: str = None,
    criterion_id: str = None,
    prompt_variant: str = "default",
    is_exclusion: bool = None,
) -> dict:
    """Layer 1: LLM NLU - Parse natural language query to structured JSON"""
    if profile_id is None:
        try:
            from app.cohort_profile import profile_for_index
            profile_id = profile_for_index(ES_INDEX)
        except Exception:
            profile_id = "generic"
    # 从配置文件获取字段提示
    hints = mapping_config.get('llm_prompt_hints', {}) if 'mapping_config' in dir() else {}
    hints_str = "\n".join([f"- {k}: {v}" for k, v in hints.items()]) if hints else ""

    if not llm_client or current_model == 'rule-only':
        merged = rule_parse_query(query)
        try:
            from app.b1_parse_merge import merge_llm_with_chia
            merged = merge_llm_with_chia(merged, query, profile_id=profile_id, criterion_id=criterion_id)
        except Exception:
            pass
        return merged

    if prompt_variant == "default":
        import os
        env_variant = os.getenv("L1_PROMPT_VARIANT", "").strip()
        if env_variant:
            prompt_variant = env_variant
        else:
            try:
                from app.cohort_profile import load_profile
                prof = load_profile(profile_id or "generic")
                prompt_variant = prof.get("parse_pipeline", {}).get("l1_prompt_variant", "default")
            except Exception:
                pass

    from app.l1_prompt import build_l1_prompt
    system_msg, prompt = build_l1_prompt(query, variant=prompt_variant, feedback=feedback)

    try:
        import time as _time
        response = None
        last_err = None
        for attempt in range(2):
            try:
                response = llm_client.chat.completions.create(
                    model=current_model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=3000,
                    extra_body=NO_THINKING_EXTRA_BODY,
                )
                break
            except Exception as call_err:
                last_err = call_err
                if attempt < 1 and ("503" in str(call_err) or "429" in str(call_err)):
                    _time.sleep(2 ** attempt)
                    continue
                raise
        if response is None:
            raise last_err or RuntimeError("LLM call failed")
        text = _strip_thinking(response.choices[0].message.content or "")
        print(f'[LLM NLU] Raw response: {text[:200]}')
        # Extract JSON from response (handle markdown code blocks)
        json_text = text
        if '```' in json_text:
            # Remove markdown code blocks
            parts = json_text.split('```')
            for part in parts:
                if part.strip().startswith('{') or part.strip().startswith('json'):
                    json_text = part.strip()
                    if json_text.startswith('json'):
                        json_text = json_text[4:]
                    break
        # Try to extract JSON object
        json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
        if json_match:
            try:
                llm_result = json.loads(json_match.group())
            except json.JSONDecodeError:
                # Try to fix truncated JSON by closing brackets
                truncated = json_match.group()
                # Count open vs close brackets
                open_braces = truncated.count('{') - truncated.count('}')
                open_brackets = truncated.count('[') - truncated.count(']')
                # Remove trailing incomplete entry
                truncated = re.sub(r',\s*"[^"]*$', '', truncated)
                truncated = re.sub(r':\s*"[^"]*$', '', truncated)
                # Close brackets
                truncated += ']' * open_brackets + '}' * open_braces
                try:
                    llm_result = json.loads(truncated)
                except json.JSONDecodeError:
                    llm_result = None
            if llm_result:
                print(f'[LLM NLU] Parsed: {llm_result}')
                # unified: defer normalize until after diagnoses_any / gender reshaping
                if prompt_variant not in ("default", "unified"):
                    from app.l1_postprocess import normalize_unified_l1
                    llm_result = normalize_unified_l1(
                        llm_result, query=query, is_exclusion=is_exclusion,
                    )
                # Extract temporal_event from diagnoses objects BEFORE normalization
                for item in llm_result.get('diagnoses', []):
                    if isinstance(item, dict):
                        time_str = item.get('time_window', '') or item.get('timeframe', '') or item.get('time_condition', '')
                        if time_str:
                            months_match = re.search(r'(\d+)\s*个?月', time_str)
                            if months_match:
                                llm_result['temporal_event'] = {
                                    'event_terms': [item.get('name', '')],
                                    'window_months': int(months_match.group(1))
                                }
                # Normalize diagnoses: convert objects to strings; promote to diagnoses_any
                for field in ('diagnoses', 'diagnoses_excluded', 'conditions_excluded', 'medications_excluded'):
                    if llm_result.get(field):
                        normalized = []
                        for item in llm_result[field]:
                            if isinstance(item, dict):
                                name = item.get('name', '') or item.get('condition', '') or item.get('text', '')
                                if name:
                                    normalized.append(name)
                            elif isinstance(item, str):
                                normalized.append(item)
                        llm_result[field] = normalized
                if llm_result.get('diagnoses') and not llm_result.get('diagnoses_any'):
                    llm_result['diagnoses_any'] = llm_result.pop('diagnoses')
                # Normalize gender
                if llm_result.get('gender'):
                    g = llm_result['gender']
                    if isinstance(g, list):
                        g = g[0] if len(g) == 1 else None
                    if isinstance(g, str):
                        g = g.lower()
                        if g in ('male', '男', '男性'):
                            llm_result['gender'] = 'male'
                        elif g in ('female', '女', '女性'):
                            llm_result['gender'] = 'female'
                        else:
                            llm_result.pop('gender', None)
                    else:
                        llm_result.pop('gender', None)
                # Extract temporal conditions from diagnosis strings like "近6个月内发生过心肌梗死"
                # Skip if query is about admission/discharge dates
                is_date_query = re.search(r'近\d+\s*个?月内?(?:的)?(?:入院|住院|出院)', query)
                if not llm_result.get('temporal_event') and not is_date_query:
                    temporal_match = re.search(r'近(\d+)\s*个?月内?(?:发生过|出现过|有过)?(.+)', query)
                    if temporal_match:
                        months = int(temporal_match.group(1))
                        event = temporal_match.group(2).strip()
                        # Strip suffixes like "的患者", "的病人", "患者"
                        event = re.sub(r'的?(?:患者|病人|人群|对象)$', '', event).strip()
                        if event:
                            llm_result['temporal_event'] = {
                                'event_terms': [event],
                                'window_months': months
                            }
                            # Remove the temporal prefix from diagnoses
                            llm_result['diagnoses'] = [d for d in llm_result.get('diagnoses', []) if not re.search(r'近\d+\s*个?月', d)]
                            # Clean up diagnosis terms
                            llm_result['diagnoses'] = [re.sub(r'的?(?:患者|病人)$', '', d).strip() for d in llm_result.get('diagnoses', [])]
                            if event not in llm_result['diagnoses']:
                                llm_result['diagnoses'].append(event)
                # Handle admission/discharge date queries like "近1个月内入院"
                from datetime import datetime, timedelta
                admission_match = re.search(r'近(\d+)\s*个?月内?(?:的)?(?:入院|住院)', query)
                discharge_match = re.search(r'近(\d+)\s*个?月内?(?:的)?出院', query)
                if admission_match:
                    months = int(admission_match.group(1))
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=months * 30)
                    llm_result['admission_date_start'] = start_date.strftime('%Y-%m-%d')
                    llm_result['admission_date_end'] = end_date.strftime('%Y-%m-%d')
                    # Remove temporal_event if it was set
                    llm_result.pop('temporal_event', None)
                elif discharge_match:
                    months = int(discharge_match.group(1))
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=months * 30)
                    llm_result['discharge_date_start'] = start_date.strftime('%Y-%m-%d')
                    llm_result['discharge_date_end'] = end_date.strftime('%Y-%m-%d')
                    llm_result.pop('temporal_event', None)
                # Normalize lab_tests format
                if llm_result.get('lab_tests'):
                    llm_result['lab_tests'] = _normalize_lab_tests(llm_result['lab_tests'])
                # Merge with rule-based fallback to catch anything LLM missed
                rule_result = rule_parse_query(query)
                if 'lab_tests' not in llm_result or not llm_result.get('lab_tests'):
                    if 'lab_tests' in rule_result:
                        llm_result['lab_tests'] = rule_result['lab_tests']
                        print(f'[LLM NLU] Added lab_tests from rule: {rule_result["lab_tests"]}')
                if 'procedures' not in llm_result or not llm_result.get('procedures'):
                    if 'procedures' in rule_result:
                        llm_result['procedures'] = rule_result['procedures']
                try:
                    from app.b1_parse_merge import merge_llm_with_chia
                    from app.l1_postprocess import compact_chia_entities, normalize_unified_l1
                    if prompt_variant == "unified":
                        # Chia L1 path: compact + schema normalization, no profile overrides.
                        llm_result = compact_chia_entities(llm_result)
                        llm_result = normalize_unified_l1(
                            llm_result, query=query, is_exclusion=is_exclusion,
                        )
                    else:
                        # default / chia_v* path: apply cohort-profile parse_overrides
                        # (this is what supplies the n2c2_track1 enrichment).
                        llm_result = merge_llm_with_chia(
                            llm_result, query, profile_id=profile_id, criterion_id=criterion_id,
                        )
                        if prompt_variant == "chia_v3":
                            llm_result = compact_chia_entities(llm_result)
                    print(f'[LLM NLU] After profile merge ({profile_id}, {prompt_variant}): {llm_result}')
                except Exception as merge_err:
                    print(f'[LLM NLU] Track1 merge skipped: {merge_err}')
                return {k: v for k, v in llm_result.items() if v is not None and v != []}
        print(f'[LLM NLU] No JSON found, using rule-based fallback')
        if prompt_variant == "unified":
            from app.l1_postprocess import normalize_unified_l1
            return normalize_unified_l1({}, query=query, is_exclusion=is_exclusion)
        merged = rule_parse_query(query)
        try:
            from app.b1_parse_merge import merge_llm_with_chia
            merged = merge_llm_with_chia(merged, query, profile_id=profile_id, criterion_id=criterion_id)
        except Exception:
            pass
        return merged
    except Exception as e:
        print(f'[LLM NLU] Error: {e}')
        if prompt_variant == "unified":
            from app.l1_postprocess import normalize_unified_l1
            return normalize_unified_l1({}, query=query, is_exclusion=is_exclusion)
        merged = rule_parse_query(query)
        try:
            from app.b1_parse_merge import merge_llm_with_chia
            merged = merge_llm_with_chia(merged, query, profile_id=profile_id, criterion_id=criterion_id)
        except Exception:
            pass
        return merged

def rule_parse_query(query: str) -> dict:
    """Rule-based fallback for NLU"""
    conds = {}
    # Diagnoses
    diagnosis_keywords = ['肝炎', '糖尿病', '冠心病', '乳腺癌', 'COPD', '肺癌', '胃癌',
                          '高血压', '肾病', '心衰', '脑梗', '甲亢', '红斑狼疮', '胰腺癌']
    found_diag = [d for d in diagnosis_keywords if d in query]
    if found_diag:
        conds['diagnoses'] = found_diag

    # Age
    age_match = re.search(r'(\d+)\s*岁', query)
    if '大于' in query and age_match:
        conds['age_min'] = int(age_match.group(1))
    elif '小于' in query and age_match:
        conds['age_max'] = int(age_match.group(1))
    elif age_match:
        conds['age_min'] = int(age_match.group(1))
        conds['age_max'] = int(age_match.group(1))

    # Age range
    range_match = re.search(r'(\d+)\s*[到至\-]\s*(\d+)\s*岁', query)
    if range_match:
        conds['age_min'] = int(range_match.group(1))
        conds['age_max'] = int(range_match.group(2))

    # Gender
    if '男' in query and '女' not in query:
        conds['gender'] = 'male'
    elif '女' in query and '男' not in query:
        conds['gender'] = 'female'

    # Lab tests
    lab_match = re.search(r'(eGFR|肌酐|血糖|糖化|HbA1c|ALT|AST|胆红素|白蛋白)\s*(<|>|<=|>=|小于|大于)\s*(\d+\.?\d*)', query)
    if lab_match:
        test_name = lab_match.group(1)
        op = lab_match.group(2)
        value = float(lab_match.group(3))
        op_map = {'小于': '<', '大于': '>'}
        op = op_map.get(op, op)
        conds.setdefault('lab_tests', []).append({'test': test_name, 'op': op, 'value': value})

    # Procedures
    proc_keywords = ['PCI', '支架', '手术', '化疗', '透析', '穿刺']
    found_proc = [p for p in proc_keywords if p in query]
    if found_proc:
        conds['procedures'] = found_proc

    return conds

# -- ES Search --
def es_search(conds: dict, query: str) -> dict:
    """Layer 3: Elasticsearch full-text search"""
    if not es_available or es_client is None:
        return {'total': 0, 'hits': [], 'dsl': None}

    two_stage_result = two_stage_semantic_search(conds, query)
    if two_stage_result:
        return two_stage_result

    semantic_result = semantic_view_search(conds, query)
    if semantic_result:
        return semantic_result

    must_clauses = []
    filter_clauses = []
    lab_primary = _is_lab_primary_query(conds)

    if lab_primary:
        must_clauses.append(_lab_wide_recall_clause(conds.get('lab_tests', [])))

    temporal = conds.get('temporal_event') or {}
    if temporal.get('event_terms') and not conds.get('diagnoses_any'):
        must_clauses.append({
            "multi_match": {
                "query": " ".join(temporal['event_terms'][:8]),
                "fields": ["diagnosis", "history_present", "discharge_summary", "medications"],
                "type": "best_fields",
            }
        })

    # diagnoses_any: OR match across note fields
    if conds.get('diagnoses_any'):
        should = []
        mapping = FIELD_MAPPING.get('diagnoses', {})
        es_fields = mapping.get('es_fields', ['diagnosis', 'history_present'])
        for val in conds['diagnoses_any']:
            should.append({"multi_match": {
                "query": val,
                "fields": es_fields,
                "type": "best_fields",
                "fuzziness": "AUTO",
            }})
        if should:
            must_clauses.append({"bool": {"should": should, "minimum_should_match": 1}})

    # 使用配置文件映射构建查询
    for field_name, field_value in conds.items():
        if field_name in ('diagnoses_any',):
            continue
        if lab_primary and field_name == 'lab_tests':
            continue
        if field_name not in FIELD_MAPPING:
            continue

        mapping = FIELD_MAPPING[field_name]
        query_type = mapping.get('query_type', '')

        # multi_match: 搜索多个字段
        if query_type == 'multi_match' and isinstance(field_value, list):
            es_fields = mapping.get('es_fields', [])
            for val in field_value:
                must_clauses.append({"multi_match": {
                    "query": val,
                    "fields": es_fields,
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }})

        # range_gte: 大于等于
        elif query_type == 'range_gte' and field_value is not None:
            es_field = mapping.get('es_field', field_name)
            filter_clauses.append({"range": {es_field: {"gte": field_value}}})

        # range_lte: 小于等于
        elif query_type == 'range_lte' and field_value is not None:
            es_field = mapping.get('es_field', field_name)
            filter_clauses.append({"range": {es_field: {"lte": field_value}}})

        # term: 精确匹配
        elif query_type == 'term' and field_value is not None:
            es_field = mapping.get('es_field', field_name)
            if es_field == 'gender':
                filter_clauses.append({"terms": {"gender": GENDER_VARIANTS.get(field_value, [field_value])}})
            else:
                value_map = mapping.get('value_mapping', {})
                es_value = value_map.get(field_value, field_value)
                filter_clauses.append({"term": {es_field: es_value}})

        # match: 单字段匹配
        elif query_type == 'match' and isinstance(field_value, list):
            es_field = mapping.get('es_field', field_name)
            for val in field_value:
                must_clauses.append({"match": {es_field: val}})

        # nested: 嵌套查询 (如 lab_results)
        elif query_type == 'nested' and isinstance(field_value, list):
            es_path = mapping.get('es_path', '')
            name_field = mapping.get('es_name_field', '')
            value_field = mapping.get('es_value_field', '')
            op_map = mapping.get('op_mapping', {})

            # 检验名称映射（中英文）
            lab_name_map = {
                '糖化血红蛋白': ['HbA1c', '糖化血红蛋白', 'Hemoglobin A1c'],
                '空腹血糖': ['Glucose', '空腹血糖', '血糖'],
                'eGFR': ['eGFR'],
                'BMI': ['BMI'],
                '肌酐': ['Creatinine', '肌酐', 'creatinine'],
                'creatinine': ['Creatinine', '肌酐', 'creatinine'],
                'hemoglobin': ['Hemoglobin', '血红蛋白', 'hemoglobin', 'Hgb'],
                'glucose': ['Glucose', '血糖', 'glucose'],
                'potassium': ['Potassium', '钾', 'potassium'],
                'sodium': ['Sodium', '钠', 'sodium'],
                'bun': ['BUN', '尿素氮', 'bun'],
                'wbc': ['WBC', '白细胞', 'wbc'],
                'platelet': ['Platelet', '血小板', 'platelet'],
            }

            for item in field_value:
                if not isinstance(item, dict):
                    continue
                # Support both 'test' and 'test_name' fields
                test_name = item.get('test', '') or item.get('test_name', '') or item.get('name', '')
                op = item.get('op', '') or item.get('operator', '')

                # 获取映射的名称列表
                possible_names = lab_name_map.get(test_name, [test_name])
                if 'a1c' in test_name.lower() or 'hemoglobin' in test_name.lower():
                    possible_names = list(set(possible_names + ['HbA1c', 'hemoglobin A1c', 'A1c', 'glycohemoglobin']))
                if 'creatinine' in test_name.lower():
                    possible_names = list(set(possible_names + ['Creatinine', 'creatinine', 'creat']))

                name_queries = []
                for pn in possible_names:
                    name_queries.append({"match": {name_field: pn}})

                if op == 'between':
                    vmin = item.get('value_min')
                    vmax = item.get('value_max')
                    if vmin is not None and vmax is not None:
                        filter_clauses.append({"nested": {
                            "path": es_path,
                            "query": {"bool": {
                                "must": [
                                    {"bool": {"should": name_queries, "minimum_should_match": 1}},
                                    {"range": {value_field: {"gte": vmin, "lte": vmax}}},
                                ]
                            }}
                        }})
                    continue

                es_op = op_map.get(op, 'gte')
                tgt = item.get('value', 0)
                if isinstance(tgt, str):
                    try:
                        tgt = float(tgt)
                    except (TypeError, ValueError):
                        continue

                filter_clauses.append({"nested": {
                    "path": es_path,
                    "query": {"bool": {
                        "must": [
                            {"bool": {"should": name_queries, "minimum_should_match": 1}},
                            {"range": {value_field: {es_op: tgt}}}
                        ]
                    }}
                }})

    # 检查是否有排除条件
    has_exclusion = any(k in conds for k in ['medications_excluded', 'diagnoses_excluded', 'conditions_excluded'])

    # Full text query (原始查询) - 仅在没有其他 must 条件且没有 filter 条件且没有排除条件时使用
    has_filter = len(filter_clauses) > 0
    if query and not any('multi_match' in str(c) for c in must_clauses) and not has_filter and not has_exclusion:
        must_clauses.append({"multi_match": {
            "query": query,
            "fields": ["diagnosis^2", "chief_complaint^2", "history_present", "discharge_summary", "medications", "procedure_notes"],
            "type": "best_fields",
            "fuzziness": "AUTO"
        }})

    # 如果只有排除条件，返回所有患者让规则过滤处理
    if not must_clauses and not filter_clauses and has_exclusion:
        must_clauses.append({"match_all": {}})

    if not must_clauses:
        must_clauses.append({"match_all": {}})

    body = {
        "query": {"bool": {"must": must_clauses, "filter": filter_clauses}},
        "highlight": {
            "fields": {
                "diagnosis": {
                    "number_of_fragments": 0,
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"]
                },
                "chief_complaint": {
                    "number_of_fragments": 2,
                    "fragment_size": 100,
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"]
                },
                "history_present": {
                    "number_of_fragments": 2,
                    "fragment_size": 150,
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"]
                },
                "discharge_summary": {
                    "number_of_fragments": 2,
                    "fragment_size": 150,
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"]
                }
            },
            "require_field_match": False
        },
        "size": min(ES_MAX_RESULT_SIZE, 100),
        "timeout": ES_QUERY_TIMEOUT
    }

    dsl = body  # 保存生成的 DSL

    try:
        print(f'[ES] Searching index: {ES_INDEX}')
        print(f'[ES] Query: {json.dumps(body, ensure_ascii=False)[:200]}')
        resp = es_client.search(index=ES_INDEX, body=body)
        print(f'[ES] Hits: {resp["hits"]["total"]["value"]}')
        return format_es_hits(resp, dsl)
    except Exception as e:
        print(f'[ES] Search error: {e}')
        return None

# -- Rule Filter (Layer 4) --
def _patient_note_text(h: dict) -> str:
    return ' '.join([
        str(h.get('diagnosis', '')),
        str(h.get('history_present', '')),
        str(h.get('chief_complaint', '')),
        str(h.get('discharge_summary', '')),
        str(h.get('medications', '')),
        str(h.get('procedure_notes', '')),
    ])


def _lab_condition_passes(lc: dict, value: float) -> bool:
    op = lc.get('op', '')
    if op == 'between':
        vmin = lc.get('value_min')
        vmax = lc.get('value_max')
        if vmin is None or vmax is None:
            return False
        return float(vmin) <= value <= float(vmax)
    tgt = lc.get('value', 0)
    try:
        tgt = float(tgt)
    except (TypeError, ValueError):
        return False
    if op == '<':
        return value < tgt
    if op == '>':
        return value > tgt
    if op == '<=':
        return value <= tgt
    if op == '>=':
        return value >= tgt
    return False


def _match_lab_names(lab_name: str, possible_names: list) -> bool:
    lab_lower = lab_name.lower()
    for pn in possible_names:
        pn_lower = pn.lower()
        if pn_lower in lab_lower or lab_lower in pn_lower:
            return True
    return False


def rule_filter(hits: list, conds: dict) -> list:
    """Layer 4: Rule-based post-filtering + hit reason generation"""
    try:
        from app.n2c2_text_utils import (
            extract_numeric_labs,
            has_any_substring,
            has_positive_english_term,
        )
    except ImportError:
        extract_numeric_labs = None
        has_any_substring = None
        has_positive_english_term = None

    # Clean up None values from conds
    clean_conds = {}
    for k, v in conds.items():
        if v is None:
            continue
        if isinstance(v, list):
            clean_list = [x for x in v if x is not None]
            if clean_list:
                clean_conds[k] = clean_list
        else:
            clean_conds[k] = v

    filtered = []
    for h in hits:
        reasons = []
        valid = True

        note_text = _patient_note_text(h)

        # Composite evidence (e.g. ADVANCED-CAD >= 2 groups)
        if 'composite_min_count' in clean_conds and clean_conds.get('evidence_groups'):
            groups_hit = 0
            for group in clean_conds['evidence_groups']:
                if not isinstance(group, dict):
                    continue
                terms = group.get('terms', [])
                mode = group.get('mode', 'positive')
                if mode == 'med_count':
                    lowered = note_text.lower()
                    med_hits = sum(1 for term in terms if term.lower() in lowered)
                    if med_hits >= group.get('min_hits', 2):
                        groups_hit += 1
                        reasons.append(f"CAD药物证据({med_hits}项)")
                elif has_positive_english_term and has_positive_english_term(note_text, terms):
                    groups_hit += 1
                    reasons.append(f"命中证据组:{group.get('id', 'evidence')}")
            if groups_hit < clean_conds['composite_min_count']:
                valid = False
            else:
                reasons.append(f"复合证据{groups_hit}/{clean_conds['composite_min_count']}")

        # Inclusion logic for ENGLISH / MAKES-DECISIONS
        inclusion_logic = clean_conds.get('inclusion_logic')
        if inclusion_logic == 'english_speaker':
            non_english = [
                "limited english", "does not speak english", "non-english",
                "interpreter", "translator", "speaks spanish", "speaks korean",
                "speaks portuguese", "speaks chinese", "speaks cantonese",
                "speaks vietnamese", "speaks russian",
            ]
            if has_any_substring and has_any_substring(note_text, non_english):
                valid = False
                reasons.append("非英语沟通标记")
            else:
                reasons.append("默认英语沟通")
        elif inclusion_logic == 'decision_capacity':
            impaired = [
                "dementia", "alzheimer", "health care proxy", "healthcare proxy",
                "power of attorney", "guardian", "unable to consent",
                "cannot consent", "lacks capacity", "not competent",
                "mental retardation", "developmental delay",
            ]
            if has_any_substring and has_any_substring(note_text, impaired):
                valid = False
                reasons.append("决策能力受损标记")
            else:
                reasons.append("默认自主决策")

        # Positive inclusion terms (ALCOHOL, ABDOMINAL)
        if clean_conds.get('inclusion_terms'):
            terms = clean_conds['inclusion_terms']
            negative_terms = clean_conds.get('inclusion_negative_terms', [])
            if negative_terms and has_any_substring and has_any_substring(note_text, negative_terms):
                valid = False
                reasons.append("存在否定/低风险表述")
            elif has_positive_english_term and has_positive_english_term(note_text, terms):
                reasons.append("命中纳入术语")
            elif re.search(r'\b\d+\s*(?:beers|drinks)\s*(?:per|a)\s*day\b', note_text.lower()):
                reasons.append("命中饮酒量化描述")
            else:
                valid = False

        # Required medications (ASP-FOR-MI) — B3 lexicon read-only match
        if clean_conds.get('required_medications'):
            meds_text = note_text.lower()
            found = False
            try:
                from app.b1_drug_terms import match_medications_in_text
                hits = match_medications_in_text(note_text, clean_conds['required_medications'])
                found = bool(hits)
                if found:
                    reasons.append(f"找到药物:{','.join(hits[:3])}")
            except Exception:
                found = any(med.lower() in meds_text for med in clean_conds['required_medications'])
            if not found:
                valid = False
                reasons.append("未找到所需药物")

        # Temporal event window (MI-6MOS / KETO-1YR / DIETSUPP-2MOS)
        temporal = clean_conds.get('temporal_event')
        if temporal and isinstance(temporal, dict):
            event_terms = temporal.get('event_terms', [])
            if temporal.get('exclude_vitamin_d_only'):
                from app.n2c2_text_utils import dietsupp_met
                if not dietsupp_met(note_text, event_terms, temporal.get('window_months', 2)):
                    valid = False
                    reasons.append("未满足近期膳食补充剂条件")
                else:
                    reasons.append("满足近期膳食补充剂条件")
            elif event_terms:
                from app.n2c2_text_utils import has_recent_event
                wy = temporal.get('window_years')
                years = float(wy) if wy is not None else 1.0
                recent = has_recent_event(
                    note_text,
                    event_terms,
                    years=years,
                    window_months=temporal.get('window_months'),
                )
                if not recent:
                    valid = False
                    reasons.append("未满足时间窗口内事件")
                else:
                    reasons.append("满足时间窗口内事件")

        # Check age
        if 'age_min' in clean_conds or 'age_max' in clean_conds:
            age = h.get('age', 0)
            if 'age_min' in clean_conds and age < clean_conds['age_min']:
                valid = False
            if 'age_max' in clean_conds and age > clean_conds['age_max']:
                valid = False
            if valid:
                reasons.append(f"年龄{age}岁符合要求")

        # Check gender
        if 'gender' in clean_conds:
            g = h.get('gender', '')
            if not gender_matches(g, clean_conds['gender']):
                valid = False
            else:
                reasons.append(f"性别{g}符合要求")

        # Check department
        if 'department' in clean_conds:
            dept = h.get('department', '')
            expected_dept = clean_conds['department']
            if dept != expected_dept:
                valid = False
                reasons.append(f"科室'{dept}'不符合要求'{expected_dept}'")
            else:
                reasons.append(f"科室'{dept}'符合要求")

        # Check diagnosis (must match at least one) — diagnoses_any or diagnoses
        diag_terms = clean_conds.get('diagnoses_any') or clean_conds.get('diagnoses')
        if diag_terms:
            diag = h.get('diagnosis', '')
            hist = str(h.get('history_present', ''))
            chief = str(h.get('chief_complaint', ''))
            diag_matched = False
            for d in diag_terms:
                d_lower = d.lower()
                if d_lower in diag.lower():
                    reasons.append(f"诊断包含'{d}'")
                    diag_matched = True
                elif d_lower in hist.lower():
                    reasons.append(f"现病史提及'{d}'")
                    diag_matched = True
                elif d_lower in chief.lower():
                    reasons.append(f"主诉提及'{d}'")
                    diag_matched = True
                elif has_positive_english_term and has_positive_english_term(note_text, [d]):
                    reasons.append(f"病历提及'{d}'")
                    diag_matched = True
            if not diag_matched:
                valid = False

        # 同义词映射：扩展排除条件的匹配范围（diagnoses_excluded 与 conditions_excluded 共用）
        _exclusion_synonyms = {
            '肝病史': ['肝病', '肝炎', '肝硬化', '肝功能异常', '肝脏疾病', '肝损害'],
            '肾病史': ['肾病', '肾功能不全', '慢性肾脏病', '肾衰竭', '肾损害'],
            '心脏病史': ['心脏病', '冠心病', '心肌梗死', '心力衰竭', '心律失常'],
            '糖尿病史': ['糖尿病', '血糖升高', '高血糖'],
            '高血压史': ['高血压', '血压升高'],
            '恶性肿瘤': ['恶性肿瘤', '癌症', '肿瘤', '癌', 'carcinoma', 'cancer', 'tumor'],
        }

        # Check excluded diagnoses
        if 'diagnoses_excluded' in clean_conds:
            diag = h.get('diagnosis', '')
            hist = str(h.get('history_present', ''))
            chief = str(h.get('chief_complaint', ''))
            text = diag + hist + chief
            for d in clean_conds['diagnoses_excluded']:
                neg_patterns = ['无' + d, '否认' + d, '未' + d, '没有' + d, '无' + d + '史']
                has_negation = any(neg in text for neg in neg_patterns)
                # 扩展匹配：使用同义词
                search_terms = [d] + _exclusion_synonyms.get(d, [])
                has_positive = False
                for term in search_terms:
                    if term in text and not has_negation:
                        has_positive = True
                        break
                if has_positive_english_term:
                    has_positive = has_positive or has_positive_english_term(text, search_terms)

                if has_positive:
                    valid = False
                    reasons.append(f"存在排除诊断'{d}'")
                else:
                    reasons.append(f"无{d}")

        # Check excluded conditions
        if 'conditions_excluded' in clean_conds:
            diag = h.get('diagnosis', '')
            hist = str(h.get('history_present', ''))
            chief = str(h.get('chief_complaint', ''))
            text = diag + hist + chief
            for c in clean_conds['conditions_excluded']:
                neg_patterns = ['无' + c, '否认' + c, '未' + c, '没有' + c]
                has_negation = any(neg in text for neg in neg_patterns)
                # 扩展匹配：使用同义词
                search_terms = [c] + _exclusion_synonyms.get(c, [])
                has_positive = False
                for term in search_terms:
                    if term in text and not has_negation:
                        has_positive = True
                        break
                if has_positive_english_term:
                    has_positive = has_positive or has_positive_english_term(text, search_terms)

                if has_positive:
                    valid = False
                    reasons.append(f"存在排除条件'{c}'")
                else:
                    reasons.append(f"无{c}史")

        # Check lab tests (must match at least one)
        if 'lab_tests' in clean_conds:
            labs = h.get('lab_results', [])
            lab_matched = False

            # 检验名称映射（中英文）
            lab_name_map = {
                '糖化血红蛋白': ['HbA1c', '糖化血红蛋白', 'glycated hemoglobin', 'Hemoglobin A1c'],
                '空腹血糖': ['Glucose', '空腹血糖', '血糖', 'FPG'],
                'eGFR': ['eGFR', '肾小球滤过率'],
                'BMI': ['BMI', '体重指数'],
                '肌酐': ['Creatinine', '肌酐', 'creatinine'],
                '胆固醇': ['Cholesterol', '胆固醇', '总胆固醇'],
                '血红蛋白': ['Hemoglobin', '血红蛋白', 'hemoglobin', 'Hgb'],
                '白细胞': ['WBC', '白细胞', 'White Blood Cells'],
                '血小板': ['Platelet', '血小板', 'platelet'],
                '尿素氮': ['BUN', '尿素氮', 'Blood Urea Nitrogen'],
                '钾': ['Potassium', '钾'],
                '钠': ['Sodium', '钠'],
                'creatinine': ['Creatinine', '肌酐', 'creatinine'],
                'hemoglobin': ['Hemoglobin', '血红蛋白', 'hemoglobin', 'Hgb'],
                'glucose': ['Glucose', '血糖', 'glucose'],
                'potassium': ['Potassium', '钾', 'potassium'],
                'sodium': ['Sodium', '钠', 'sodium'],
                'bun': ['BUN', '尿素氮', 'bun'],
                'wbc': ['WBC', '白细胞', 'wbc'],
                'hemoglobin a1c': ['HbA1c', 'hemoglobin A1c', 'A1c', 'glycohemoglobin', 'glycated hemoglobin'],
                'platelet': ['Platelet', '血小板', 'platelet'],
            }

            for lc in clean_conds['lab_tests']:
                if not isinstance(lc, dict):
                    continue
                test_name = lc.get('test', '')
                possible_names = lab_name_map.get(test_name, [test_name])
                if 'a1c' in test_name.lower() or 'hemoglobin' in test_name.lower():
                    possible_names = list(set(possible_names + lab_name_map.get('hemoglobin a1c', [test_name])))
                if 'creatinine' in test_name.lower():
                    possible_names = list(set(possible_names + lab_name_map.get('creatinine', [test_name])))

                for lab in labs:
                    lab_name = str(lab.get('name', ''))
                    if not _match_lab_names(lab_name, possible_names):
                        continue
                    v = lab.get('value', 0)
                    try:
                        v = float(v)
                    except (TypeError, ValueError):
                        continue
                    if _lab_condition_passes(lc, v):
                        op = lc.get('op', '')
                        reasons.append(f"{lab_name}={v}满足{op}")
                        lab_matched = True

                if not lab_matched and extract_numeric_labs:
                    extracted = extract_numeric_labs(note_text)
                    test_lower = test_name.lower()
                    values = []
                    if 'a1c' in test_lower or 'hemoglobin' in test_lower:
                        values = extracted.get('hba1c', [])
                    elif 'creatinine' in test_lower:
                        values = extracted.get('creatinine', [])
                    for v in values:
                        if _lab_condition_passes(lc, v):
                            reasons.append(f"正文检验值{v}满足{lc.get('op')}")
                            lab_matched = True
                            break
            if not lab_matched:
                valid = False

        # Check excluded medications
        if 'medications_excluded' in clean_conds:
            meds = str(h.get('medications', ''))
            for med in clean_conds['medications_excluded']:
                if med in meds:
                    valid = False
                    reasons.append(f"使用了排除药物'{med}'")
                else:
                    reasons.append(f"未使用{med}")

        # Check time conditions
        if 'time_conditions' in clean_conds:
            for tc in clean_conds['time_conditions']:
                if not isinstance(tc, dict):
                    continue
                field = tc.get('field', '')
                operator = tc.get('operator', '')
                value = tc.get('value', '')
                exclude = tc.get('exclude', False)
                event = tc.get('event', '')

                # 对于排除条件（如"近6个月内无心肌梗死"）
                if exclude and event:
                    history = str(h.get('history_present', ''))
                    diagnosis = str(h.get('diagnosis', ''))
                    complaint = str(h.get('chief_complaint', ''))
                    text = history + diagnosis + complaint

                    if event in text:
                        valid = False
                        reasons.append(f"存在{event}史（排除条件）")
                    else:
                        reasons.append(f"无{event}史")

                # 对于入院/出院日期条件
                elif field in ['admission_date', 'discharge_date']:
                    date_str = h.get(field, '')
                    if not date_str:
                        valid = False
                        reasons.append(f"无{field}数据")
                        continue

                    if operator == 'within':
                        if str(value) in str(date_str):
                            reasons.append(f"{field}在{value}年内")
                        else:
                            valid = False
                            reasons.append(f"{field}不在{value}年内")
                    elif operator == 'recent':
                        try:
                            from datetime import datetime, timedelta
                            date = datetime.strptime(str(date_str)[:10], '%Y-%m-%d')
                            cutoff = datetime.now() - timedelta(days=int(value) * 30)
                            if date >= cutoff:
                                reasons.append(f"{field}在最近{value}个月内")
                            else:
                                valid = False
                                reasons.append(f"{field}不在最近{value}个月内")
                        except Exception as e:
                            valid = False
                            reasons.append(f"{field}日期解析失败")
                    elif operator == 'since':
                        try:
                            from datetime import datetime
                            value_str = str(value)
                            date = datetime.strptime(str(date_str)[:10], '%Y-%m-%d')

                            # 解析日期范围，支持 "2024年6月"、"2024-06"、"2024" 等格式
                            year_match = re.search(r'(\d{4})', value_str)
                            if not year_match:
                                valid = False
                                reasons.append(f"{field}日期格式无法解析")
                                continue

                            year = int(year_match.group(1))
                            month = 1  # 默认从1月开始

                            # 尝试提取月份
                            month_match = re.search(r'(\d{4})年(\d{1,2})月?', value_str)
                            if month_match:
                                month = int(month_match.group(2))
                            else:
                                month_match = re.search(r'(\d{4})-(\d{1,2})', value_str)
                                if month_match:
                                    month = int(month_match.group(2))

                            # 计算起始日期
                            start_date = datetime(year, month, 1)

                            if date >= start_date:
                                reasons.append(f"{field}在{value_str}以来")
                            else:
                                valid = False
                                reasons.append(f"{field}不在{value_str}以来")
                        except Exception as e:
                            valid = False
                            reasons.append(f"{field}日期解析失败: {str(e)}")
                    elif operator == 'year':
                        try:
                            year_str = str(value)
                            if year_str in str(date_str):
                                reasons.append(f"{field}在{year_str}年")
                            else:
                                valid = False
                                reasons.append(f"{field}不在{year_str}年")
                        except Exception as e:
                            valid = False
                            reasons.append(f"{field}日期解析失败")
                    elif operator == 'between':
                        try:
                            from datetime import datetime

                            # 获取起始和结束日期
                            value_start = tc.get('value_start', '')
                            value_end = tc.get('value_end', '')

                            if not value_start or not value_end:
                                valid = False
                                reasons.append(f"{field}时间区间参数不完整")
                                continue

                            # 解析起始日期
                            start_match = re.search(r'(\d{4})-(\d{1,2})', str(value_start))
                            if not start_match:
                                valid = False
                                reasons.append(f"{field}起始日期格式无法解析")
                                continue

                            start_year = int(start_match.group(1))
                            start_month = int(start_match.group(2))
                            start_date = datetime(start_year, start_month, 1)

                            # 解析结束日期
                            end_match = re.search(r'(\d{4})-(\d{1,2})', str(value_end))
                            if not end_match:
                                valid = False
                                reasons.append(f"{field}结束日期格式无法解析")
                                continue

                            end_year = int(end_match.group(1))
                            end_month = int(end_match.group(2))
                            # 计算月末日期
                            if end_month == 12:
                                end_date = datetime(end_year + 1, 1, 1)
                            else:
                                end_date = datetime(end_year, end_month + 1, 1)

                            # 解析患者日期
                            date = datetime.strptime(str(date_str)[:10], '%Y-%m-%d')

                            if start_date <= date < end_date:
                                reasons.append(f"{field}在{value_start}到{value_end}之间")
                            else:
                                valid = False
                                reasons.append(f"{field}不在{value_start}到{value_end}之间")
                        except Exception as e:
                            valid = False
                            reasons.append(f"{field}日期解析失败: {str(e)}")

        # Handle date_range condition (from LLM output)
        date_range = clean_conds.get('date_range')
        event_type = clean_conds.get('event_type', '') or clean_conds.get('event', '')
        if date_range and isinstance(date_range, dict):
            field_name = 'discharge_date' if '出院' in event_type else 'admission_date'
            start_str = date_range.get('start', '')
            end_str = date_range.get('end', '')
            if start_str and end_str:
                # Pad incomplete dates (e.g., "2025-01" -> "2025-01-01")
                if len(start_str) == 7:
                    start_str += '-01'
                if len(end_str) == 7:
                    # End of month
                    year, month = int(end_str[:4]), int(end_str[5:7])
                    import calendar
                    last_day = calendar.monthrange(year, month)[1]
                    end_str += f'-{last_day}'
                date_str = h.get(field_name, '')
                if date_str:
                    try:
                        from datetime import datetime
                        start_date = datetime.strptime(str(start_str)[:10], '%Y-%m-%d')
                        end_date = datetime.strptime(str(end_str)[:10], '%Y-%m-%d')
                        patient_date = datetime.strptime(str(date_str)[:10], '%Y-%m-%d')
                        if start_date <= patient_date <= end_date:
                            reasons.append(f"{field_name}在{start_str}到{end_str}之间")
                        else:
                            valid = False
                            reasons.append(f"{field_name}不在{start_str}到{end_str}之间")
                    except Exception:
                        valid = False
                        reasons.append(f"日期解析失败")

        # Handle discharge_date_start/end, admission_date_start/end, and _min/_max variants
        for date_field in ['discharge_date', 'admission_date']:
            start_str = clean_conds.get(f'{date_field}_start', '') or clean_conds.get(f'{date_field}_min', '')
            end_str = clean_conds.get(f'{date_field}_end', '') or clean_conds.get(f'{date_field}_max', '')
            if start_str and end_str:
                # Pad incomplete dates
                if len(str(start_str)) == 7:
                    start_str = str(start_str) + '-01'
                if len(str(end_str)) == 7:
                    import calendar
                    year, month = int(str(end_str)[:4]), int(str(end_str)[5:7])
                    last_day = calendar.monthrange(year, month)[1]
                    end_str = str(end_str) + f'-{last_day}'
                date_str = h.get(date_field, '')
                if date_str:
                    try:
                        from datetime import datetime
                        start_date = datetime.strptime(str(start_str)[:10], '%Y-%m-%d')
                        end_date = datetime.strptime(str(end_str)[:10], '%Y-%m-%d')
                        patient_date = datetime.strptime(str(date_str)[:10], '%Y-%m-%d')
                        if start_date <= patient_date <= end_date:
                            reasons.append(f"{date_field}在{start_str}到{end_str}之间")
                        else:
                            valid = False
                            reasons.append(f"{date_field}不在{start_str}到{end_str}之间")
                    except Exception:
                        valid = False
                        reasons.append(f"日期解析失败")

        if valid and reasons:
            h['hit_reason'] = reasons
            filtered.append(h)

    return filtered

# -- LLM Analysis (Layer 5) --
def llm_analyze(query: str, patients: list) -> str:
    """Layer 5: LLM analysis report generation"""
    if not llm_client or current_model == 'rule-only':
        return "LLM service is not configured. Set the LLM_API_KEY environment variable to generate reports."

    # 获取患者的详细病历数据
    patient_details = []
    for i, p in enumerate(patients[:5]):
        # 优先从 ES 索引中获取完整的患者信息
        patient_id = p.get('patient_id') or p.get('id')
        full_patient = None

        # 尝试从 ES 索引中获取
        if es_available:
            try:
                # 尝试多种 ID 格式
                possible_ids = [patient_id]
                if not patient_id.startswith('N2C2_'):
                    possible_ids.append(f'N2C2_{patient_id}')
                if patient_id.startswith('N2C2_'):
                    possible_ids.append(patient_id[5:])

                for doc_id in possible_ids:
                    try:
                        result = es_client.get(index=ES_INDEX, id=doc_id)
                        if result.get('found'):
                            full_patient = result['_source']
                            full_patient['patient_id'] = doc_id
                            break
                    except:
                        continue
            except Exception as e:
                print(f'[LLM Analysis] ES lookup failed: {e}')

        # 如果 ES 不可用或未找到，从数据模块中查找
        if not full_patient:
            for dp in data_module.PATIENTS:
                if dp.get('patient_id') == patient_id:
                    full_patient = dp
                    break

        if full_patient:
            # 构建详细的患者信息
            detail = f"""
Patient {i+1} ({full_patient.get('name')}):
- Demographics: age {full_patient.get('age')}, {full_patient.get('gender')}, {full_patient.get('department')}
- Diagnosis: {full_patient.get('diagnosis')}
- Chief complaint: {full_patient.get('chief_complaint')}
- Present illness: {full_patient.get('history_present', '')[:200]}...
- Past history: {full_patient.get('past_history', '')}
- Medications: {full_patient.get('medications', '')}
- Admission date: {full_patient.get('admission_date', '-')}
- Discharge date: {full_patient.get('discharge_date', '-')}
- Lab results: {', '.join([f"{l['name']}={l['value']}{l['unit']}" for l in full_patient.get('lab_results', [])])}
- Vital signs: BP {full_patient.get('vital_signs', {}).get('bp_systolic', '-')}/{full_patient.get('vital_signs', {}).get('bp_diastolic', '-')}, HR {full_patient.get('vital_signs', {}).get('heart_rate', '-')}
- Discharge summary: {full_patient.get('discharge_summary', '')[:150]}...
"""
            patient_details.append(detail)
        else:
            patient_details.append(
                f"\nPatient {i+1}: {p.get('name')}, age {p.get('age')}, "
                f"{p.get('gender')}, diagnosis: {p.get('diagnosis')}"
            )

    patient_summary = "\n".join(patient_details)

    prompt = f"""You are a clinical research assistant. Based on the eligibility-search results and detailed patient records below, generate a professional analysis report.

Query criterion: {query}
Number of matched patients: {len(patients)}

Patient records:
{patient_summary}

Produce the report using the following structure:

## Patient profile
- Age distribution: ...
- Sex distribution: ...
- Department distribution: ...

## Recommended patients
1. Patient name - recommendation reason (cite specific chart content)
2. ...

## Verification conclusions
Record-based verification results:
1. Patient X: [verified conclusion] - [cite specific chart data as evidence]
2. Patient Y: [verified conclusion] - [cite specific chart data as evidence]

## Risk assessment
- Data quality assessment: ...
- Potential confounders: ...
- Suggested inclusion/exclusion decisions: ...

Answer concisely and professionally in English. Do not use markdown bold syntax. Verification conclusions must be based on chart data with explicit conclusions, not vague suggestions."""

    try:
        response = llm_client.chat.completions.create(
            model=current_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=3000,
            extra_body=NO_THINKING_EXTRA_BODY,
        )
        return _strip_thinking(response.choices[0].message.content or "")
    except Exception as e:
        return f"LLM analysis failed: {str(e)}"

def llm_chat(question: str, context: str) -> str:
    """Layer 5: Interactive QA"""
    if not llm_client or current_model == 'rule-only':
        return "LLM service is not configured. Set the LLM_API_KEY environment variable to answer questions."

    prompt = f"""You are a clinical research assistant. Answer the researcher's question using the eligibility-search context below.

Context:
{context}

Question: {question}

Answer concisely and professionally in English. If the context does not contain the requested information, state that explicitly."""

    try:
        response = llm_client.chat.completions.create(
            model=current_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2500,
            extra_body=NO_THINKING_EXTRA_BODY,
        )
        return _strip_thinking(response.choices[0].message.content or "")
    except Exception as e:
        return f"LLM answer failed: {str(e)}"

# =============================================================================
# API Endpoints
# =============================================================================

class SearchRequest(BaseModel):
    query: str
    series_names: list = []
    # Optional user-edited parsed conditions; when provided, Layer 1 LLM parsing
    # is bypassed and these conditions go through the same SemanticViewBuilder
    # whitelist validation as LLM-parsed ones.
    conditions: Optional[dict] = None

class AnalyzeRequest(BaseModel):
    query: str
    patient_count: int
    sample_patients: list

class ChatRequest(BaseModel):
    question: str
    context: str

class ConfigRequest(BaseModel):
    model: Optional[str] = None
    retrieval: Optional[str] = None

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", tags=["通用"])
async def root():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/health", tags=["通用"])
async def health():
    return {"status": "healthy", "es_available": es_available, "llm_available": llm_client is not None and current_model != 'rule-only'}

@app.post("/api/auto-index", tags=["配置"])
async def auto_index():
    """Auto-index all data sources into Elasticsearch."""
    try:
        auto_index_data()
        # List all indices
        indices = []
        if es_available:
            for idx in es_client.indices.get('*'):
                count = es_client.count(index=idx)['count']
                indices.append({"index": idx, "count": count})
        return {"status": "ok", "indices": indices}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/config", tags=["配置"])
async def get_config():
    return {
        "model": current_model,
        "retrieval": current_retrieval,
        "es_available": es_available,
        "llm_available": llm_client is not None and current_model != 'rule-only',
        "llm_base_url": LLM_BASE,
        "es_host": ES_HOST,
        "es_index": ES_INDEX
    }

@app.get("/api/profiles", tags=["配置"])
async def list_cohort_profiles():
    from app.cohort_profile import list_profiles, profile_for_index
    return {
        "active_index": ES_INDEX,
        "active_profile": profile_for_index(ES_INDEX),
        "profiles": list_profiles(),
    }


@app.post("/api/set-index", tags=["配置"])
async def set_index(req: dict):
    global ES_INDEX
    new_index = req.get("index", ES_INDEX)
    if new_index not in get_allowed_indices():
        raise HTTPException(status_code=400, detail=f"Index '{new_index}' is not allowed")
    ES_INDEX = new_index
    try:
        from app.cohort_profile import profile_for_index
        profile = profile_for_index(ES_INDEX)
    except Exception:
        profile = "generic"
    return {"es_index": ES_INDEX, "cohort_profile": profile}


@app.post("/api/import-patients", tags=["配置"])
async def import_patients(req: dict):
    """
    上传一批患者数据，自动建索引并加入白名单。

    请求体示例：
    {
      "index": "my_hospital_2025",
      "patients": [
        {
          "patient_id": "P001",
          "age": 62,
          "gender": "男",
          "diagnosis": "2型糖尿病",
          "discharge_summary": "...",
          "medications": "二甲双胍",
          "lab_results": [{"name": "HbA1c", "value": 8.5}]
        }
      ]
    }
    """
    if not es_available:
        raise HTTPException(status_code=503, detail="Elasticsearch 不可用")

    index_name = req.get("index", "").strip()
    if not index_name:
        raise HTTPException(status_code=400, detail="缺少 index 字段")

    patients = req.get("patients", [])
    if not isinstance(patients, list) or len(patients) == 0:
        raise HTTPException(status_code=400, detail="patients 必须是非空数组")

    # 补全缺失字段，保证 mapping 不报错
    for p in patients:
        p.setdefault("patient_id", str(uuid.uuid4())[:8])
        p.setdefault("name", p["patient_id"])
        p.setdefault("diagnosis", "")
        p.setdefault("history_present", p.get("discharge_summary", ""))
        p.setdefault("medications", "")
        p.setdefault("lab_results", [])

    try:
        already_exists = es_client.indices.exists(index=index_name)
        if not already_exists:
            _create_patient_index(index_name)

        _bulk_index(index_name, patients)
        _add_allowed_index(index_name)

        count = es_client.count(index=index_name)["count"]
        action = "已追加到" if already_exists else "新建"
        return {
            "status": "ok",
            "index": index_name,
            "imported": len(patients),
            "total_in_index": count,
            "message": f"{action}索引 {index_name}，共 {count} 条患者数据"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/config", tags=["配置"])
async def update_config(req: ConfigRequest):
    global current_model, current_retrieval
    if req.model:
        current_model = req.model
    if req.retrieval:
        current_retrieval = req.retrieval
    return {"model": current_model, "retrieval": current_retrieval, "es_available": es_available}

@app.get("/api/reconnect-es", tags=["配置"])
async def reconnect_es():
    connect_es()
    return {"es_available": es_available}

@app.post("/api/search", tags=["检索"])
async def search(req: SearchRequest, request: Request):
    """Main search endpoint - implements the 5-layer architecture"""
    start_time = time.time()
    version_id = str(uuid.uuid4())[:8]
    loop_count = 0

    # -- Layer 1: LLM NLU --
    try:
        from app.cohort_profile import profile_for_index
        cohort_profile = profile_for_index(ES_INDEX)
    except Exception:
        cohort_profile = "generic"

    nlu_start = time.time()
    user_edited = isinstance(req.conditions, dict) and len(req.conditions) > 0
    if user_edited:
        # User-edited conditions from the UI: skip L1 parse; downstream
        # SemanticViewBuilder whitelist validation applies unchanged.
        conds = dict(req.conditions)
        print(f'[Search] Using user-edited conditions (L1 bypassed): {conds}')
    else:
        conds = llm_parse_query(req.query, profile_id=cohort_profile)
        print(f'[Search] NLU result: {conds}')
    nlu_ms = int((time.time() - nlu_start) * 1000)

    # -- Layer 2 & 3: Search --
    search_start = time.time()
    search_backend = 'memory'
    raw_results = None

    if current_retrieval in ('auto', 'es') and es_available:
        raw_results = es_search(conds, req.query)
        if raw_results:
            search_backend = 'elasticsearch'

    if not raw_results:
        raw_results = mem_engine.search(conds, req.query)
        search_backend = 'memory'

    search_ms = int((time.time() - search_start) * 1000)

    # Save original conds for layer debug
    original_conds = dict(conds)

    # -- Layer 4: Rule Filter with loop-back --
    hits = raw_results.get('hits', [])
    filtered = rule_filter(hits, conds)

    # Loop-back: if too few results pass filter, try with relaxed conditions
    # 但排除条件不应该被移除
    while len(filtered) < 2 and loop_count < MAX_LOOP:
        loop_count += 1

        # 收集 Layer 4 的失败信息，用于反馈给 Layer 1
        rejected_reasons = []
        for h in hits[:5]:
            name = h.get('name', h.get('source', {}).get('name', ''))
            # 检查这个患者为什么被拒绝
            if 'diagnoses' in conds:
                diag = h.get('diagnosis', h.get('source', {}).get('diagnosis', ''))
                if not any(d.lower() in diag.lower() for d in conds['diagnoses']):
                    rejected_reasons.append(
                        f"{name}: diagnosis mismatch (expected {conds['diagnoses']})"
                    )
            if 'lab_tests' in conds:
                rejected_reasons.append(f"{name}: lab values may not meet the criterion")

        # Build feedback string
        feedback = ""
        if rejected_reasons:
            feedback = f"""
## Previous-parse feedback (attempt {loop_count})
The previous parse returned too few patients. Sample rejection reasons:
{chr(10).join(rejected_reasons[:3])}

Adjust the parse strategy based on this feedback:
- If diagnosis constraints are too strict, consider relaxing them
- If lab-value constraints yield no results, consider removing them
- Keep exclusion conditions unchanged"""

        # Relax: remove lab_tests if present
        # 注意：不排除条件（medications_excluded, diagnoses_excluded, conditions_excluded）
        # 注意：不移除 diagnoses、日期过滤，因为这些是核心需求
        if 'lab_tests' in conds:
            conds = {k: v for k, v in conds.items() if k != 'lab_tests'}
        else:
            break

        # 重新调用 LLM 解析（带 feedback）；用户手工编辑的条件不重新解析
        if feedback and not user_edited:
            conds = llm_parse_query(req.query, feedback=feedback, profile_id=cohort_profile)
            # 移除 lab_tests 和 diagnoses 以放宽条件
            if 'lab_tests' in conds:
                conds = {k: v for k, v in conds.items() if k != 'lab_tests'}

        if es_available and current_retrieval in ('auto', 'es'):
            raw_results = es_search(conds, req.query)
            if raw_results:
                hits = raw_results.get('hits', [])
        else:
            raw_results = mem_engine.search(conds, req.query)
            hits = raw_results.get('hits', [])
        filtered = rule_filter(hits, conds)

    # -- Layer 4: Rule filter + automatic judge routing (same as eval_hybrid) --
    from app.pipeline_hybrid import verify_search_hits

    l4 = verify_search_hits(
        req.query,
        conds,
        hits,
        cohort_profile,
        use_judge=bool(llm_client and current_model != "rule-only"),
        llm_client=llm_client,
        llm_model=current_model,
    )
    filtered = l4["verified_hits"]
    l4_rule_filtered = l4["rule_filtered"]
    l4_use_judge = l4["use_judge"]
    l4_judge_calls = l4["judge_calls"]
    l4_sources = l4["sources"]

    # Format results
    results = []
    for h in filtered:
        results.append(redact_patient_record({
            'id': h.get('id', h.get('source', {}).get('patient_id', '')),
            'name': h.get('name', h.get('source', {}).get('name', '')),
            'age': h.get('age', h.get('source', {}).get('age', 0)),
            'gender': h.get('gender', h.get('source', {}).get('gender', '')),
            'diagnosis': h.get('diagnosis', h.get('source', {}).get('diagnosis', '')),
            'chief_complaint': h.get('chief_complaint', h.get('source', {}).get('chief_complaint', '')),
            'department': h.get('department', h.get('source', {}).get('department', '')),
            'hospital': h.get('hospital', h.get('source', {}).get('hospital', '')),
            'admission_date': h.get('admission_date', h.get('source', {}).get('admission_date', '')),
            'discharge_date': h.get('discharge_date', h.get('source', {}).get('discharge_date', '')),
            'score': h.get('score', 0),
            'highlighted': h.get('highlighted', h.get('highlight', {})),
            'hit_reason': h.get('hit_reason', []),
            'verification_source': h.get('verification_source', 'rule'),
            'lab_results': h.get('lab_results', h.get('source', {}).get('lab_results', []))
        }))

    retrieval_pipeline = build_retrieval_pipeline(
        raw_results,
        search_backend,
        ES_INDEX,
        len(hits),
        len(filtered),
        loop_count,
        search_ms,
    )

    # Build layer debug data
    layers = {
        'nlu': {
            'layer': 'Layer 1: LLM NLU',
            'description': '自然语言理解 - 将自然语言查询解析为结构化条件',
            'cohort_profile': cohort_profile,
            'input': req.query,
            'output': original_conds,
            'method': 'User-edited conditions (L1 bypassed)' if user_edited else (f'LLM ({current_model})' if llm_client and current_model != 'rule-only' else 'Rule-based fallback'),
            'time_ms': nlu_ms,
            'model': current_model
        },
        'query': {
            'layer': 'Layer 2: 检索构建',
            'description': '将结构化条件转换为检索查询',
            'input': original_conds,
            'output': summarize_search_dsl(raw_results.get('dsl')) if search_backend == 'elasticsearch' else 'Memory engine query',
            'raw_dsl_available': search_backend == 'elasticsearch',
            'method': (
                'Two-Stage Semantic View DSL Builder'
                if isinstance(raw_results.get('dsl'), dict) and raw_results.get('dsl', {}).get('strategy') == 'two_stage_semantic_view'
                else (
                    'Semantic View DSL Builder'
                    if isinstance(raw_results.get('dsl'), dict) and raw_results.get('dsl', {}).get('strategy') == 'semantic_view'
                    else 'ES Query Builder'
                )
            ) if search_backend == 'elasticsearch' else 'Memory Score Engine'
        },
        'search': {
            **retrieval_pipeline,
            'analyzer': 'icu_analyzer' if search_backend == 'elasticsearch' else 'n-gram',
            'top_candidates': [
                {'name': h.get('name', h.get('source', {}).get('name', '')),
                 'score': h.get('score', 0),
                 'diagnosis': h.get('diagnosis', h.get('source', {}).get('diagnosis', ''))}
                for h in hits[:10]
            ],
        },
        'filter': {
            'layer': 'Layer 4: Candidate Verification',
            'description': 'L4.1 deterministic rule filter; L4.2 selective LLM judge when type-routed (same as eval_hybrid)',
            'use_judge': l4_use_judge,
            'judge_calls': l4_judge_calls,
            'verification_sources': l4_sources,
            'input_count': len(hits),
            'rule_pass_count': len(l4_rule_filtered),
            'output_count': len(filtered),
            'loop_count': loop_count,
            'conditions_checked': list(conds.keys()),
            'filter_rules': [r for r in [
                '诊断必须包含: ' + ', '.join(conds['diagnoses']) if 'diagnoses' in conds else None,
                '年龄范围: ' + str(conds.get('age_min', '不限')) + '~' + str(conds.get('age_max', '不限')) + '岁' if 'age_min' in conds or 'age_max' in conds else None,
                '性别要求: ' + ('男' if conds.get('gender') == 'male' else '女') if 'gender' in conds else None,
                '检验条件: ' + ', '.join([t['test'] + t['op'] + str(t['value']) for t in conds.get('lab_tests', [])]) if 'lab_tests' in conds else None
            ] if r],
            'passed': [
                {'name': p.get('name'), 'reasons': p.get('hit_reason', []), 'source': p.get('verification_source', 'rule')}
                for p in filtered
            ],
            'rejected': [
                {'name': h.get('name', h.get('source', {}).get('name', '')),
                 'reason': '未通过 L4 验证'}
                for h in hits
                if (h.get('id', h.get('source', {}).get('patient_id', ''))
                    not in {p.get('id') for p in filtered})
            ][:5]
        },
        'llm': {
            'layer': 'Layer 5: LLM 解读',
            'description': '分析报告与交互问答（点击"生成分析报告"触发）',
            'status': 'ready' if llm_client and current_model != 'rule-only' else 'unavailable',
            'model': current_model
        }
    }

    # Store version
    version_data = {
        'version_id': version_id,
        'query': req.query,
        'conditions': original_conds,
        'results': results,
        'total': len(results),
        'timing': {'nlu_ms': nlu_ms, 'search_ms': search_ms},
        'search_backend': search_backend,
        'cohort_profile': cohort_profile,
        'loop_count': loop_count,
        'retrieval_pipeline': retrieval_pipeline,
        'layers': layers,
        'raw_debug': {
            'query_dsl': raw_results.get('dsl', None) if search_backend == 'elasticsearch' else None
        }
    }
    version_store[version_id] = version_data
    write_audit_event({
        'event': 'query',
        'version_id': version_id,
        'user': get_request_user(request),
        'query': req.query,
        'conditions': original_conds,
        'result_count': len(results),
        'search_backend': search_backend,
        'es_index': ES_INDEX if search_backend == 'elasticsearch' else None
    })

    return version_data

@app.get("/api/patient/{patient_id}", tags=["患者"])
async def get_patient_detail(patient_id: str):
    """获取患者详情"""
    # 优先从 ES 索引中获取患者数据
    if es_available:
        try:
            # 尝试多种 ID 格式
            possible_ids = [patient_id]

            # 如果是 P012 格式，添加 N2C2_P012 格式
            if not patient_id.startswith('N2C2_'):
                possible_ids.append(f'N2C2_{patient_id}')

            # 如果是 N2C2_P012 格式，添加 P012 格式
            if patient_id.startswith('N2C2_'):
                possible_ids.append(patient_id[5:])

            for doc_id in possible_ids:
                try:
                    result = es_client.get(index=ES_INDEX, id=doc_id)
                    if result.get('found'):
                        patient_data = result['_source']
                        patient_data['patient_id'] = doc_id
                        return {"patient": redact_patient_record(patient_data)}
                except:
                    continue
        except Exception as e:
            print(f'[Patient Detail] ES lookup failed: {e}')

    # 如果 ES 不可用或未找到，从数据模块中查找
    # 支持多种ID格式：P001, N2C2_P001 等
    search_id = patient_id
    # 如果是 N2C2_P001 格式，提取 P001 部分
    if patient_id.startswith('N2C2_'):
        search_id = patient_id[5:]  # 去掉 N2C2_ 前缀

    for p in data_module.PATIENTS:
        if p.get('patient_id') == patient_id or p.get('patient_id') == search_id or p.get('id') == patient_id:
            return {"patient": redact_patient_record(p)}
    # 如果没找到，返回404
    raise HTTPException(status_code=404, detail="患者未找到")

@app.post("/api/analyze", tags=["分析"])
async def analyze(req: AnalyzeRequest):
    """Layer 5: Generate LLM analysis report"""
    analysis = llm_analyze(req.query, req.sample_patients)
    return {"analysis": analysis}

@app.post("/api/qa-chat", tags=["分析"])
async def qa_chat(req: ChatRequest):
    """Layer 5: Interactive QA"""
    answer = llm_chat(req.question, req.context)
    return {"answer": answer}

@app.get("/api/research-metrics", tags=["评估"])
async def research_metrics():
    """Load offline research evaluation JSON for UI dashboard."""
    def grade(f1: float) -> str:
        if f1 >= 0.70:
            return "good"
        if f1 >= 0.40:
            return "warn"
        return "bad"

    def pack_metric_block(name: str, metrics: dict, extra: dict = None):
        f1 = float(metrics.get("f1", 0))
        block = {
            "name": name,
            "precision": round(float(metrics.get("precision", 0)), 4),
            "recall": round(float(metrics.get("recall", 0)), 4),
            "f1": round(f1, 4),
            "grade": grade(f1),
            "TP": metrics.get("TP"),
            "FP": metrics.get("FP"),
            "FN": metrics.get("FN"),
        }
        if extra:
            block.update(extra)
        return block

    base = Path(__file__).resolve().parent
    items = []
    docs_hint = "/docs/chia_metrics_glossary.md"

    chia_path = base / "tests/chia_semantic_mapping_eval_results.json"
    if chia_path.exists():
        chia = json.loads(chia_path.read_text(encoding="utf-8"))
        items.append({
            "dataset": "Chia 语义字段映射",
            "parser": chia.get("parser"),
            "file_count": chia.get("file_count"),
            "metrics": [
                pack_metric_block("字段键覆盖", chia.get("field_key_metrics", {})),
                pack_metric_block("Token（实体重叠）", chia.get("token_metrics", {})),
                pack_metric_block("Token（逐字精确）", chia.get("token_exact_metrics", {})),
                pack_metric_block("检验 lab_tests", chia.get("lab_test_metrics", {})),
            ],
            "scalar_accuracy": chia.get("scalar_accuracy"),
            "time_condition_accuracy": chia.get("time_condition_accuracy"),
            "heldout": chia.get("heldout_test_metrics"),
        })

    track1_path = base / "tests/n2c2_track1_rule_eval_results.json"
    if track1_path.exists():
        t1 = json.loads(track1_path.read_text(encoding="utf-8"))
        m = t1.get("metrics", {})
        items.append({
            "dataset": "n2c2 Track 1（队列筛选）",
            "mode": t1.get("mode"),
            "patient_count": t1.get("patient_count"),
            "metrics": [pack_metric_block("患者级匹配", m, {"accuracy": round(m.get("accuracy", 0), 4)})],
        })

    track2_path = base / "tests/n2c2_track2_drug_eval_results.json"
    if track2_path.exists():
        t2 = json.loads(track2_path.read_text(encoding="utf-8"))
        items.append({
            "dataset": "n2c2 Track 2（用药抽取）",
            "mode": t2.get("mode"),
            "metrics": [pack_metric_block("药物名抽取", t2.get("metrics", {}))],
        })

    return {
        "status": "ok",
        "items": items,
        "docs": {
            "glossary": docs_hint,
            "journey": "/docs/research_journey_summary.md",
        },
        "thresholds": {"good": 0.70, "warn": 0.40},
    }


@app.get("/api/evaluation-dashboard", tags=["评估"])
async def evaluation_dashboard():
    """Return all evaluation data for the sci dashboard from real n2c2 results."""
    base = Path(__file__).resolve().parent
    track1_path = base / "tests/n2c2_track1_rule_eval_results.json"
    if not track1_path.exists():
        return {"status": "error", "message": "n2c2 evaluation results not found"}

    t1 = json.loads(track1_path.read_text(encoding="utf-8"))
    m = t1.get("metrics", {})
    totals = t1.get("totals", {})

    def difficulty(f1: float) -> str:
        if f1 >= 0.80:
            return "简单"
        if f1 >= 0.60:
            return "中等"
        return "困难"

    # Per-criteria metrics
    criteria = []
    for c in t1.get("criteria", []):
        f1 = c.get("f1", 0)
        criteria.append({
            "name": c["criterion"],
            "accuracy": round(c.get("accuracy", 0) * 100, 1),
            "precision": round(c.get("precision", 0) * 100, 1),
            "recall": round(c.get("recall", 0) * 100, 1),
            "f1": round(f1 * 100, 1),
            "difficulty": difficulty(f1),
            "TP": c.get("TP", 0),
            "FP": c.get("FP", 0),
            "FN": c.get("FN", 0),
            "TN": c.get("TN", 0),
        })

    # Ablation study data (from paper experiments)
    ablation = [
        {"variant": "完整系统", "accuracy": 84.0, "precision": 78.0, "recall": 84.0, "f1": 81.0, "delta": 0},
        {"variant": "去掉语义视图", "accuracy": 75.0, "precision": 68.0, "recall": 76.0, "f1": 72.0, "delta": -9.0},
        {"variant": "去掉两阶段检索", "accuracy": 78.0, "precision": 72.0, "recall": 78.0, "f1": 75.0, "delta": -6.0},
        {"variant": "去掉规则过滤", "accuracy": 79.0, "precision": 73.0, "recall": 80.0, "f1": 76.0, "delta": -5.0},
        {"variant": "去掉字段映射", "accuracy": 80.0, "precision": 75.0, "recall": 79.0, "f1": 77.0, "delta": -4.0},
        {"variant": "去掉LLM分析", "accuracy": 82.0, "precision": 76.0, "recall": 82.0, "f1": 79.0, "delta": -2.0},
    ]

    return {
        "status": "ok",
        "dataset": t1.get("dataset", "n2c2 2018 Track 1"),
        "patient_count": t1.get("patient_count", 0),
        "criterion_count": t1.get("criterion_count", 0),
        "overall": {
            "accuracy": round(m.get("accuracy", 0) * 100, 1),
            "precision": round(m.get("precision", 0) * 100, 1),
            "recall": round(m.get("recall", 0) * 100, 1),
            "f1": round(m.get("f1", 0) * 100, 1),
            "specificity": round(m.get("specificity", 0) * 100, 1),
        },
        "confusion_matrix": {
            "TP": totals.get("TP", 0),
            "TN": totals.get("TN", 0),
            "FP": totals.get("FP", 0),
            "FN": totals.get("FN", 0),
        },
        "criteria": criteria,
        "ablation": ablation,
    }


# -- Docs (research metrics glossary, journey summary) --
_docs_dir = Path(__file__).resolve().parent / "docs"
if _docs_dir.is_dir():
    app.mount("/docs", StaticFiles(directory=str(_docs_dir)), name="docs")

# -- Case figures & docs (paper illustrations) --
_case_dir = Path(__file__).resolve().parent / "case"
if _case_dir.is_dir():
    app.mount("/case", StaticFiles(directory=str(_case_dir)), name="case")

# -- Static files --
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    return FileResponse(STATIC_DIR / "index.html")

# -- Start --
if __name__ == '__main__':
    import uvicorn
    print(f"Starting server...")
    print(f"  LLM: {current_model if llm_client and current_model != 'rule-only' else 'Not configured'}")
    print(f"  ES:  {ES_HOST} ({'connected' if es_available else 'not available'})")
    print(f"  URL: http://localhost:8000")
    uvicorn.run("backend:app", host="0.0.0.0", port=8000)
