import os
import uvicorn
from fastapi import FastAPI, File, HTTPException, Depends, Body, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from typing import List
from services.openai import get_chat_completion

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
)
from datastore.factory import get_datastore
from services.file import get_document_from_file


app = FastAPI()
app.mount("/.well-known", StaticFiles(directory=".well-known"), name="static")

# Create a sub-application, in order to access just the query endpoint in an OpenAPI schema, found at http://0.0.0.0:8000/sub/openapi.json when the app is running locally
sub_app = FastAPI(
    title="Retrieval Plugin API",
    description="A retrieval API for querying and filtering documents based on natural language queries and metadata",
    version="1.0.0",
    servers=[{"url": "https://your-app-url.com"}],
)
app.mount("/sub", sub_app)

bearer_scheme = HTTPBearer()
BEARER_TOKEN = os.environ.get("BEARER_TOKEN")
assert BEARER_TOKEN is not None


def validate_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return credentials


@app.post(
    "/upsert-file",
    response_model=UpsertResponse,
)
async def upsert_file(
    file: UploadFile = File(...),
    token: HTTPAuthorizationCredentials = Depends(validate_token),
):
    document = await get_document_from_file(file)

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
    token: HTTPAuthorizationCredentials = Depends(validate_token),
):
    try:
        ids = await datastore.upsert(request.documents)
        return UpsertResponse(ids=ids)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.post(
    "/query",
    response_model=QueryResponse,
)
async def query_main(
    request: QueryRequest = Body(...),
    token: HTTPAuthorizationCredentials = Depends(validate_token),
):
    try:
        results = await datastore.query(
            request.queries,
        )
        return QueryResponse(results=results)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@sub_app.post(
    "/query",
    response_model=QueryResponse,
    # NOTE: We are describing the shape of the API endpoint input due to a current limitation in parsing arrays of objects from OpenAPI schemas. This will not be necessary in the future.
    description="Accepts search query objects array each with query and optional filter. Break down complex questions into sub-questions. Refine results by criteria, e.g. time / source, don't do this often. Split queries if ResponseTooLargeError occurs.",
)
async def query(
    request: QueryRequest = Body(...),
    token: HTTPAuthorizationCredentials = Depends(validate_token),
):
    try:
        results = await datastore.query(
            request.queries,
        )
        return QueryResponse(results=results)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.delete(
    "/delete",
    response_model=DeleteResponse,
)
async def delete(
    request: DeleteRequest = Body(...),
    token: HTTPAuthorizationCredentials = Depends(validate_token),
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
    messages_openai = [
        {
            "role": "system",
            "content": f"""
            You take in segments of podcast transcripts that have been found using embeddings.
            Using these segments to answer users questions. Answer no matter what.
            """,
        }
    ]
    for i in range(len(messages) - 1):
        messages_openai.append({"role": "user", "content": messages[i]})
    messages_openai.append({"role": "user", "content": str(segments) + '\n\n' + q})

    print(messages_openai)
    print(segments)
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
    new_queries = []
    for q in request.queries:
        f = {}
        if q.filter:
            f = {k: v for k,v in q.filter.dict() if k in ['episode_id', 'podcast_id']}
        new_queries.append(Query.parse_obj({
            "query": q.query, 
            "filter": f, 
            "top_k": q.top_k
        }))
    request.queries = new_queries
    
    query_response = await query(request)
    query_results = []
    for t in query_response.results:
        keys_to_extract = ['text', 'score', 'name', 'author']
        query_results.append(
            [{key: tt.dict()['metadata'][key] if key in ['name', 'author'] else tt.dict()[key] for key in keys_to_extract} for tt in t.results]
        )

    results = []
    for i in range(len(query_results)):
        results.append(call_chat(query_results[i], request.messages, request.queries[i].query))
        
    return {'response': results}

@app.on_event("startup")
async def startup():
    global datastore
    datastore = await get_datastore()

def start():
    attempts = 0
    worked = False
    port = 8000
    while not worked and attempts < 5:
        try:
            uvicorn.run("server.main:app", host="0.0.0.0", port=port, reload=True, workers=8)
        except KeyboardInterrupt:
            cvwdv
        except:
            attempts += 1
            port += 1
