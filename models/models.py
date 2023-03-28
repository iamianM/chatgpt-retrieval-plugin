from pydantic import BaseModel
from typing import List, Optional, Dict
from enum import Enum


class Source(str, Enum):
    email = "email"
    file = "file"
    chat = "chat"

class DocumentMetadata(BaseModel):
    source: Optional[Source] = None
    episode_id: Optional[int] = None
    podcast_id: Optional[int] = None
    mp3_url: Optional[str] = None
    created_at: Optional[str] = None
    duration: Optional[str] = None
    author: Optional[str] = None
    name: Optional[str] = None


class DocumentChunkMetadata(DocumentMetadata):
    document_id: Optional[str] = None


class DocumentChunk(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: DocumentChunkMetadata
    embedding: Optional[List[float]] = None
    sparse_values: Optional[Dict[str, List[float]]] = None


class DocumentChunkWithScore(DocumentChunk):
    score: float


class Document(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: Optional[DocumentMetadata] = None


class DocumentWithChunks(Document):
    chunks: List[DocumentChunk]


class DocumentMetadataFilter(BaseModel):
    document_id: Optional[str] = None
    start_date: Optional[str] = None  # any date string format
    end_date: Optional[str] = None  # any date string format
    episode_id: Optional[int] = None
    podcast_id: Optional[int] = None
    mp3_url: Optional[str] = None
    duration: Optional[str] = None
    author: Optional[str] = None
    name: Optional[str] = None


class Query(BaseModel):
    query: str
    filter: Optional[DocumentMetadataFilter] = None
    top_k: Optional[int] = 3


class QueryWithEmbedding(Query):
    embedding: List[float]
    sparse_indices: List[float]
    sparse_embedding: List[float]


class QueryResult(BaseModel):
    query: str
    results: List[DocumentChunkWithScore]
