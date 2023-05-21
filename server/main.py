import os
from typing import Optional
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Depends, Body, UploadFile, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from typing import List, Dict
from services.openai import get_chat_completion, openai_response, get_messages2

import logging
import logging.config

logging.config.fileConfig('logging_config.ini')
logger = logging.getLogger('sampleLogger')

import openai, json

SECRETS = json.load(open("./OPENAI_KEY.json"))
openai.api_key = SECRETS["OPENAI_KEY"]

from models.api import (
    DeleteRequest,
    DeleteResponse,
    QueryRequest,
    QueryResponse,
    QueryChartDataRequest,
    QueryChartDataResponse,
    UpsertRequest,
    UpsertResponse,
    ChatQueryRequest,
    ChatQueryResult,
    Query,
    AskQuestionInput,
)
from models.models import DocumentChunk, DocumentMetadataFilter
from datastore.factory import get_datastore
from services.file import get_document_from_file

from models.models import DocumentMetadata, Source

bearer_scheme = HTTPBearer()
BEARER_TOKEN = os.environ.get("BEARER_TOKEN")
assert BEARER_TOKEN is not None


def validate_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return credentials


app = FastAPI(dependencies=[Depends(validate_token)])
app.mount("/.well-known", StaticFiles(directory=".well-known"), name="static")

from starlette.middleware.base import BaseHTTPMiddleware
import traceback
class CatchAllMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.debug(f"Error: {e}\n{tb_str}")
            raise e from None
        return response
    
app.add_middleware(CatchAllMiddleware)
# Create a sub-application, in order to access just the query endpoint in an OpenAPI schema, found at http://0.0.0.0:8000/sub/openapi.json when the app is running locally
sub_app = FastAPI(
    title="Retrieval Plugin API",
    description="A retrieval API for querying and filtering documents based on natural language queries and metadata",
    version="1.0.0",
    servers=[{"url": "https://your-app-url.com"}],
    dependencies=[Depends(validate_token)],
)
sub_app.add_middleware(CatchAllMiddleware)
app.mount("/sub", sub_app)


@app.post(
    "/upsert-file",
    response_model=UpsertResponse,
)
async def upsert_file(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
):
    try:
        metadata_obj = (
            DocumentMetadata.parse_raw(metadata)
            if metadata
            else DocumentMetadata(source=Source.file)
        )
    except:
        metadata_obj = DocumentMetadata(source=Source.file)

    document = await get_document_from_file(file, metadata_obj)

    try:
        ids = await datastore.upsert([document])
        return UpsertResponse(ids=ids)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=f"str({e})")


@app.post(
    "/upsert",
    response_model=UpsertResponse,
)
async def upsert(
    request: UpsertRequest = Body(...),
):
    # try:
    ids = await datastore.upsert(request.documents)
    return UpsertResponse(ids=ids)
    # except Exception as e:
    #     print("Error:", e)
    #     raise HTTPException(status_code=500, detail="Internal Service Error")

@app.post(
    "/query",
    response_model=QueryResponse,
)
async def query(
    request: QueryRequest = Body(...),
):
    print(request)
    results = await datastore.query(
        request.queries, request.search_topics
    )
    return QueryResponse(results=results)


@app.delete(
    "/delete",
    response_model=DeleteResponse,
)
async def delete(
    request: DeleteRequest = Body(...),
):
    if not (request.ids or request.filter or request.delete_all):
        raise HTTPException(
            status_code=400,
            detail="One of ids, filter, or delete_all is required",
        )
    try:
        success = await datastore.delete(
            ids=request.ids,
            filter=request.filter,
            delete_all=request.delete_all,
        )
        return DeleteResponse(success=success)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.post(
    "/query_chart_data",
    response_model=QueryChartDataResponse,
)
async def query_chart_data(
    request: QueryChartDataRequest = Body(...),
):
    query_response = query(request)
    r = {}
    for k,v in query_response.items():
        rr = {}
        for t in v['results']:
            scores = [tt['score'] for tt in t['results']]
            rr[t['query']] = sum(scores)/len(scores)
            
        r[k] = rr
        
    return r

def call_chat(segments: List[dict], messages: List[str], q: str) -> str:
    messages_openai = [{
        "role": "system",
        "content": f"Given a request, try to answer it using the content of the file extracts above. Do your best to respond to the request. " \
        f"Give the answer in markdown format. "
        f"For your response use Answer: <answer or \"I couldn't find the answer to that question in your files\" or \"That's not a valid question.\">\n\n" \
        f"Files:\n{segments}\n" \
        f"Question: {q}\n\n" \
        f"Answer:"
    }]
    # for i in range(len(messages) - 1):
    #     messages_openai.append({"role": "user", "content": messages[i]})
    # messages_openai.append({"role": "user", "content": str(segments) + '\n\n' + q})

    print(messages)
    # print(messages_openai)
    # print(segments)
    completion = get_chat_completion(
        messages_openai,
    )
    
    return completion

def get_questions(segments: List[dict], messages: List[str]) -> str:
    messages_openai = [{
        "role": "system",
        "content": f"Given file segments, come up with 3 questions that a user might have where those files would be relevant in answering the questions. " \
        f"Files:\n{segments}\n" \
        f"Recommended Questions:"
    }]
    
    print(messages)
    completion = get_chat_completion(
        messages_openai,
    )
    
    return completion

@app.post(
    "/chat_query",
    response_model=ChatQueryResult,
)
async def chat_query(
    request: ChatQueryRequest = Body(...),
):
    token_len_max = 3900
    token_len_max -= 500
    top_k_max = 30
    model = "gpt-3.5-turbo"
    
    new_queries = []
    for q in request.queries:
        f = {}
        if q.filter:
            f = {k: v for k,v in q.filter.dict().items() if k in ['episode_id', 'podcast_id']}
        new_queries.append(Query.parse_obj({
            "query": q.query, 
            "filter": f, 
            "top_k": top_k_max
        }))
    request.queries = new_queries
    print(request.queries)
    
    query_response = await query(request)
    query_results = []
    for t in query_response.results:
        keys_to_extract = ['text', 'score', 'name', 'slug']
        query_results.append(
            [{key: tt.dict()['metadata'][key] if key in ['name', 'author', 'slug'] else tt.dict()[key] for key in keys_to_extract} for tt in t.results]
        )

    results = []
    print(query_results)
    for i in range(len(query_results)):
        _, messages = get_messages2(request.queries[i].query, query_results[i], token_len_max)
        results.append(await openai_response(model, messages, 500, 0.5))
        
    print(results)
    return {'response': results}

@app.post(
    "/recommend_questions",
    response_model=ChatQueryResult,
)
async def recommend_questions(
    request: ChatQueryRequest = Body(...),
):
    new_queries = []
    for q in request.queries:
        f = {}
        if q.filter:
            f = {k: v for k,v in q.filter.dict().items() if k in ['episode_id', 'podcast_id']}
        new_queries.append(Query.parse_obj({
            "query": "What is this about?", 
            "filter": f, 
            "top_k": q.top_k
        }))
    request.queries = new_queries
    print(request.queries)
    
    query_response = await query(request)
    query_results = []
    for t in query_response.results:
        keys_to_extract = ['text', 'score', 'name', 'author']
        query_results.append(
            [{key: tt.dict()['metadata'][key] if key in ['name', 'author'] else tt.dict()[key] for key in keys_to_extract} for tt in t.results]
        )

    results = []
    for i in range(len(query_results)):
        results.append(get_questions(query_results[i], request.messages))
        
    return {'response': results}

import numpy as np
import tiktoken
from fastapi.responses import StreamingResponse
import aiohttp

tokenizer = tiktoken.get_encoding(
    "cl100k_base"
)

def build_files_string(response_data: Dict) -> str:
    files_string = f"Podcast Author: {response_data['results'][0]['metadata']['author']}\n\nEpisode Slug: {response_data['results'][0]['metadata']['slug']}"

    for result in response_data['results']:
        file_text = result["text"]
        file_string = f"###\n{file_text}\n"
        files_string += file_string

    return files_string

def get_messages(question: str, segments: str) -> str:
    messages = [{
        "role": "system",
        "content": f"Given a request, try to answer it using the content of the podcast transcript segments below. Do your best to respond to the request. " \
        f"Give the answer in markdown format. "
        f"Use the following format:\n\nQuestion: <question>\n\nSegments:\n<###\n\"segment 1\">\n<###\n\"segment 2\">...\n\n"\
        f"Answer: <answer or \"I couldn't find the answer to that question in your files\" or \"That's not a valid question.\">\n\n" \
        f"Segments:\n{segments}\n" \
        f"Question: {question}\n\n" \
        f"Answer:"
    }]

    return messages

def shorten_responses(response_data, question):
    response_data['results'] = response_data['results'][:-1]
    files_string = build_files_string(response_data)
    messages = get_messages(question, files_string)
    token_len = len(tokenizer.encode(messages[0]['content'], disallowed_special=()))
    return messages, token_len

@app.post("/chat")
async def chat(input_data: AskQuestionInput):
    token_len_max = 8000 if input_data.use_gpt4 else 3900
    token_len_max -= input_data.max_response_tokens
    top_k_max = 60 if input_data.use_gpt4 else 30
    model = "gpt-4" if input_data.use_gpt4 else "gpt-3.5-turbo"
    
    top_k = top_k_max
    _filter = DocumentMetadataFilter()

    # print(input_data.question)
    queries = Query(
        query=input_data.question,
        filter=_filter,
        top_k=top_k,
    )
    
    response_data = await datastore.query([queries], False)
    response_data = response_data[0].dict()
        
    
    token_len = 100000
    while token_len > token_len_max:
        messages, token_len = shorten_responses(response_data, input_data.question)
        print('prompt_length', token_len)
            
    return ''.join([t.choices[0]['delta']['content'] for t in openai.ChatCompletion.create(
            messages=messages,
            model=model,
            max_tokens=input_data.max_response_tokens,
            temperature=input_data.temperature,
            stream=True) if 'content' in t.choices[0]['delta']])
            # stream=input_data.stream) if 'content' in t.choices[0]['delta']])
    
async def stream_openai_response(model, messages, max_tokens, temperature, stream):
    url = f"https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {openai.api_key}"}
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
        "model": model
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            async for line in resp.content:
                yield line
                
from tenacity import retry, wait_random_exponential, stop_after_attempt
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
@app.post("/chat_stream")
async def chat_stream(input_data: AskQuestionInput):    
    token_len_max = 8000 if input_data.use_gpt4 else 3900
    token_len_max -= input_data.max_response_tokens
    top_k_max = 60 if input_data.use_gpt4 else 30
    model = "gpt-4" if input_data.use_gpt4 else "gpt-3.5-turbo"
    
    top_k = top_k_max
    _filter = DocumentMetadataFilter()

    queries = Query(
        query=input_data.question,
        filter=_filter,
        top_k=top_k,
    )
    
    response_data = await datastore.query([queries], False)
    response_data = response_data[0].dict()

    token_len = 100000
    while token_len > token_len_max:
        messages, token_len = shorten_responses(response_data, input_data.question)

    return StreamingResponse(
        stream_openai_response(model, messages, input_data.max_response_tokens, input_data.temperature, True),
        media_type="application/json",
    )

@app.on_event("startup")
async def startup():
    global datastore
    datastore = await get_datastore()

def start():
    port = 8000
    try:
        uvicorn.run("server.main:app", host="0.0.0.0", port=port, reload=True, workers=8)
    except:
        uvicorn.run("server.main:app", host="0.0.0.0", port=port + 1, reload=True, workers=8)
