from pydantic import BaseModel
from typing import List, Optional, Dict
from enum import Enum


class Source(str, Enum):
    email = "email"
    file = "file"
    chat = "chat"

class DocumentMetadata(BaseModel):
    source: Optional[Source] = None
    episode_id: Optional[str] = None
    podcast_id: Optional[str] = None
    mp3_url: Optional[str] = None
    created_at: Optional[int] = None
    episode_duration: Optional[str] = None
    author: Optional[str] = None
    name: Optional[str] = None
    slug: Optional[str] = None
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None
    text_metadata: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    keyphrases: Optional[List[str]] = None
    entities: Optional[List[str]] = None
    topic_label: Optional[str] = None
    namespace: Optional[str] = None


class DocumentChunkMetadata(DocumentMetadata):
    document_id: Optional[str] = None


class DocumentChunk(BaseModel):
    id: Optional[str] = None
    text: Optional[str] = None
    metadata: DocumentChunkMetadata = None
    embedding: Optional[List[float]] = None
    sparse_values: Optional[Dict[str, List]] = None
    created_at: Optional[int] = None


class DocumentChunkWithScore(DocumentChunk):
    score: float


class Document(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: Optional[DocumentMetadata] = None
    created_at: Optional[int] = None


class DocumentWithChunks(Document):
    chunks: List[DocumentChunk]


class DocumentMetadataFilter(BaseModel):
    document_id: Optional[str] = None
    start_date: Optional[str] = None  # any date string format
    end_date: Optional[str] = None  # any date string format
    episode_id: Optional[str] = None
    podcast_id: Optional[str] = None
    mp3_url: Optional[str] = None
    episode_duration: Optional[str] = None
    author: Optional[str] = None
    name: Optional[str] = None
    slug: Optional[str] = None
    chunk_id_start: Optional[int] = None
    chunk_id_end: Optional[int] = None
    namespace: Optional[str] = None


class Query(BaseModel):
    query: str
    filter: Optional[DocumentMetadataFilter] = None
    top_k: Optional[int] = 5
    include_embeddings: Optional[bool] = False


class QueryWithEmbedding(Query):
    embedding: List[float]
    sparse_values: Optional[Dict[str, List]] = None


class QueryResult(BaseModel):
    query: str
    results: List[DocumentChunkWithScore]
