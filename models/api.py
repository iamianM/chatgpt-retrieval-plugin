from models.models import (
    Document,
    DocumentMetadataFilter,
    Query,
    QueryResult,
)
from pydantic import BaseModel
from typing import List, Optional


class UpsertRequest(BaseModel):
    documents: List[Document]


class UpsertResponse(BaseModel):
    ids: List[str]


class QueryRequest(BaseModel):
    queries: List[Query]
    search_topics: Optional[bool] = False
    
class ChatQueryRequest(BaseModel):
    queries: List[Query]
    messages: List[str]
    search_topics: Optional[bool] = False


class QueryResponse(BaseModel):
    results: List[QueryResult]
    
    
class QueryChartDataRequest(BaseModel):
    queries: List[str]
    time_intervals: List[str]
    top_k: int


class QueryChartDataResponse(BaseModel):
    results: List[List[float]]
    
class ChatQueryResult(BaseModel):
    response: List[str]


class DeleteRequest(BaseModel):
    ids: Optional[List[str]] = None
    filter: Optional[DocumentMetadataFilter] = None
    delete_all: Optional[bool] = False


class DeleteResponse(BaseModel):
    success: bool
    
class AskQuestionInput(BaseModel):
    question: str
    use_gpt4: bool = False
    temperature: float = 0.1
    max_response_tokens: int = 500
