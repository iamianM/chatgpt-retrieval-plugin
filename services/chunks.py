from typing import Dict, List, Optional, Tuple
import uuid
from models.models import Document, DocumentChunk, DocumentChunkMetadata

import tiktoken
import numpy as np
import re
import json
import os
from services.date import to_unix_timestamp

datastore_service = os.getenv('DATASTORE')

from services.openai import get_embeddings, get_sparse_embeddings

# Global variables
tokenizer = tiktoken.get_encoding(
    "cl100k_base"
)  # The encoding scheme to use for tokenization

# Constants
CHUNK_SIZE = 200  # The target size of each text chunk in tokens
MIN_CHUNK_SIZE_CHARS = 350  # The minimum size of each text chunk in characters
MIN_CHUNK_LENGTH_TO_EMBED = 5  # Discard chunks shorter than this
EMBEDDINGS_BATCH_SIZE = 128  # The number of embeddings to request at a time
MAX_NUM_CHUNKS = 10000  # The maximum number of chunks to generate from a text


def get_text_chunks(text: str, chunk_token_size: Optional[int]):
    """
    Split a text into chunks of ~CHUNK_SIZE tokens, based on punctuation and newline boundaries.

    Args:
        text: The text to split into chunks.
        chunk_token_size: The target size of each chunk in tokens, or None to use the default CHUNK_SIZE.

    Returns:
        A list of text chunks, each of which is a string of ~CHUNK_SIZE tokens.
    """
    # Return an empty list if the text is empty or whitespace
    if not text or text.isspace():
        return []

    # Tokenize the text
    chunks = json.loads(text)

    # Use the provided chunk token size or the default one
    chunk_size_goal = chunk_token_size or CHUNK_SIZE

    # Initialize a counter for the number of chunks
    num_chunks = 0

    text_metadatas = []
    start_timestamps = []
    end_timestamps = []
    chunk_text = ""
    result = []
    text_idx = 0
    
    # Loop until all tokens are consumed
    while chunks and num_chunks < MAX_NUM_CHUNKS:
        start_timestamps.append(f"[{chunks[0]['time']}]")
        text_metadata = []
        chunk_size = 0
        while chunk_size < chunk_size_goal and chunks:
            data = {'text_index': text_idx}
            chunk_text += " " + chunks[0]['content']
            chunk_size += len(tokenizer.encode(chunks[0]['content'], disallowed_special=()))
            text_idx += len(chunks[0]['content'])
            end_timestamp = f"[{chunks[0]['time']}]"
            data.update({k: v for k, v in chunks[0].items() if k != 'content'})
            # data.update({k: v for k, v in chunks[0].items()})
            # text_metadata.append("json.dumps(data)")
            chunks = chunks[1:]
            
        end_timestamps.append(end_timestamp)
        text_metadatas.append(text_metadata)

        # Find the last period or punctuation mark in the chunk
        last_punctuation = max(
            chunk_text.rfind("."),
            chunk_text.rfind("?"),
            chunk_text.rfind("!"),
            chunk_text.rfind("\n"),
        )

        # If there is a punctuation mark, and the last punctuation index is before MIN_CHUNK_SIZE_CHARS
        if last_punctuation != -1 and last_punctuation > MIN_CHUNK_SIZE_CHARS:
            chunk_text_to_append = chunk_text[: last_punctuation + 1]
            chunk_text = chunk_text[last_punctuation + 1:]
        else:
            chunk_text_to_append = chunk_text
            chunk_text = ""
            
        # Remove any newline characters and strip any leading or trailing whitespace
        chunk_text_to_append = chunk_text_to_append.replace("\n", " ").strip()

        if len(chunk_text_to_append) > MIN_CHUNK_LENGTH_TO_EMBED:
            # Append the chunk text to the list of chunks
            result.append(chunk_text_to_append)
        else:
            text_metadatas.pop()
            start_timestamps.pop()
            end_timestamps.pop()

        # Increment the number of chunks
        num_chunks += 1

    # Handle the remaining tokens
    if chunk_text:
        chunk_text = chunk_text.replace("\n", " ").strip()
        if len(chunk_text) > MIN_CHUNK_LENGTH_TO_EMBED:
            result.append(chunk_text)
            text_metadatas.append([])
            start_timestamps.append(end_timestamps[-1])
            end_timestamps.append(end_timestamps[-1])

    return result, text_metadatas, start_timestamps, end_timestamps


def create_document_chunks(
    doc: Document, chunk_token_size: Optional[int]
) -> Tuple[List[DocumentChunk], str]:
    """
    Create a list of document chunks from a document object and return the document id.

    Args:
        doc: The document object to create chunks from. It should have a text attribute and optionally an id and a metadata attribute.
        chunk_token_size: The target size of each chunk in tokens, or None to use the default CHUNK_SIZE.

    Returns:
        A tuple of (doc_chunks, doc_id), where doc_chunks is a list of document chunks, each of which is a DocumentChunk object with an id, a document_id, a text, and a metadata attribute,
        and doc_id is the id of the document object, generated if not provided. The id of each chunk is generated from the document id and a sequential number, and the metadata is copied from the document object.
    """
    # Check if the document text is empty or whitespace
    if not doc.text or doc.text.isspace():
        return [], doc.id or str(uuid.uuid4())

    # Generate a document id if not provided
    doc_id = doc.id or str(uuid.uuid4())

    # Split the document text into chunks
    text_chunks, text_metadatas, start_timestamps, end_timestamps = get_text_chunks(doc.text, chunk_token_size)

    metadata = (
        DocumentChunkMetadata(**doc.metadata.__dict__)
        if doc.metadata is not None
        else DocumentChunkMetadata()
    )
    metadata.document_id = doc_id

    # Initialize an empty list of chunks for this document
    doc_chunks = []

    # Assign each chunk a sequential number and create a DocumentChunk object
    for i, text_chunk in enumerate(text_chunks):
        chunk_id = f"{doc_id}_{i}"
        metadata_copy = metadata.copy()
        metadata_copy.start_timestamp = start_timestamps[i]
        metadata_copy.end_timestamp = end_timestamps[i]
        metadata_copy.text_metadata = text_metadatas[i]
        metadata_copy.created_at = to_unix_timestamp(metadata_copy.created_at) if type(metadata_copy.created_at) == str else metadata_copy.created_at
        doc_chunk = DocumentChunk(
            id=chunk_id,
            text=text_chunk,
            metadata=metadata_copy,
        )
        # Append the chunk object to the list of chunks for this document
        doc_chunks.append(doc_chunk)

    # Return the list of chunks and the document id
    return doc_chunks, doc_id


def get_document_chunks(
    documents: List[Document], chunk_token_size: Optional[int]
) -> Dict[str, List[DocumentChunk]]:
    """
    Convert a list of documents into a dictionary from document id to list of document chunks.

    Args:
        documents: The list of documents to convert.
        chunk_token_size: The target size of each chunk in tokens, or None to use the default CHUNK_SIZE.

    Returns:
        A dictionary mapping each document id to a list of document chunks, each of which is a DocumentChunk object
        with text, metadata, and embedding attributes.
    """
    # Initialize an empty dictionary of lists of chunks
    chunks: Dict[str, List[DocumentChunk]] = {}

    # Initialize an empty list of all chunks
    all_chunks: List[DocumentChunk] = []

    # Loop over each document and create chunks
    for doc in documents:
        doc_chunks, doc_id = create_document_chunks(doc, chunk_token_size)

        # Append the chunks for this document to the list of all chunks
        all_chunks.extend(doc_chunks)

        # Add the list of chunks for this document to the dictionary with the document id as the key
        chunks[doc_id] = doc_chunks

    # Check if there are no chunks
    if not all_chunks:
        return {}

    # Get all the embeddings for the document chunks in batches, using get_embeddings
    embeddings = []
    sparse_values = []
    for i in range(0, len(all_chunks), EMBEDDINGS_BATCH_SIZE):
        # Get the text of the chunks in the current batch
        batch_texts = [
            chunk.text for chunk in all_chunks[i : i + EMBEDDINGS_BATCH_SIZE]
        ]

        # Get the embeddings for the batch texts
        batch_embeddings = get_embeddings(batch_texts)
        embeddings.extend(batch_embeddings)
        
        if datastore_service == 'pinecone':
            batch_sparse_values = get_sparse_embeddings(batch_texts)
            sparse_values.extend(batch_sparse_values)

    # Update the document chunk objects with the embeddings
    for i, chunk in enumerate(all_chunks):
        # Assign the embedding from the embeddings list to the chunk object
        chunk.embedding = embeddings[i]
        if datastore_service == 'pinecone':
            chunk.sparse_values = sparse_values[i]

    return chunks
