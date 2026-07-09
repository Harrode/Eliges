"""
增强 API 模块
提供字段映射、跨索引查询、后处理代码生成的 API 接口
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any

from .enhanced_search import get_enhanced_search_engine
from .field_mapper import get_field_mapper
from .cross_index_search import get_index_router, get_cross_index_search
from .post_processor import get_post_processing_pipeline


router = APIRouter(prefix="/api/enhanced", tags=["增强功能"])


class EnhancedSearchRequest(BaseModel):
    query: str
    conditions: Dict
    options: Optional[Dict] = None


class PostProcessRequest(BaseModel):
    request: str
    results: List[Dict]
    options: Optional[Dict] = None


class FieldSuggestRequest(BaseModel):
    partial_input: str
    top_n: int = 5


@router.post("/search")
async def enhanced_search(req: EnhancedSearchRequest):
    """增强搜索"""
    try:
        from elasticsearch import Elasticsearch
        import os

        es_host = os.getenv('ES_HOST', 'http://localhost:9200')
        es_client = Elasticsearch(es_host)

        engine = get_enhanced_search_engine(es_client)
        result = engine.search(req.query, req.conditions, req.options)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/post-process")
async def post_process(req: PostProcessRequest):
    """后处理"""
    try:
        from openai import OpenAI
        import os

        api_key = os.getenv('LLM_API_KEY', '')
        base_url = os.getenv('LLM_BASE_URL', 'https://api.deepseek.com')
        llm_client = OpenAI(base_url=base_url, api_key=api_key) if api_key else None

        pipeline = get_post_processing_pipeline(llm_client)
        result = pipeline.process(req.request, req.results, **(req.options or {}))

        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/field-suggest")
async def field_suggest(req: FieldSuggestRequest):
    """字段建议"""
    try:
        from elasticsearch import Elasticsearch
        import os

        es_host = os.getenv('ES_HOST', 'http://localhost:9200')
        es_client = Elasticsearch(es_host)

        engine = get_enhanced_search_engine(es_client)
        suggestions = engine.get_field_suggestions(req.partial_input, req.top_n)

        return {"suggestions": suggestions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/indices")
async def get_indices():
    """获取可用索引"""
    try:
        from elasticsearch import Elasticsearch
        import os

        es_host = os.getenv('ES_HOST', 'http://localhost:9200')
        es_client = Elasticsearch(es_host)

        engine = get_enhanced_search_engine(es_client)
        indices = engine.get_available_indices()

        return {"indices": indices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/indices/{index_name}/fields")
async def get_index_fields(index_name: str):
    """获取索引字段"""
    try:
        from elasticsearch import Elasticsearch
        import os

        es_host = os.getenv('ES_HOST', 'http://localhost:9200')
        es_client = Elasticsearch(es_host)

        engine = get_enhanced_search_engine(es_client)
        fields = engine.get_index_fields(index_name)

        return {"fields": fields}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates")
async def get_templates():
    """获取可用模板"""
    try:
        pipeline = get_post_processing_pipeline()
        templates = pipeline.get_available_templates()

        return {"templates": templates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "modules": {
            "field_mapper": "available",
            "cross_index_search": "available",
            "post_processor": "available"
        }
    }
