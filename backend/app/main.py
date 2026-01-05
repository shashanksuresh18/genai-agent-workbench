import os
import sys
import json
import time
from uuid import uuid4
from typing import Any, Optional, TypedDict, List, Dict

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.rag.store import ensure_collection
from app.rag.ingest import ingest_text, search as rag_search

# Azure OpenAI (OpenAI python SDK v1.x)
from openai import AzureOpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError, APIStatusError

# LangGraph
from langgraph.graph import StateGraph, END


# ----------------------------
# Models (API contract)
# ----------------------------
class RunRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    mode: str = Field(default="rag")  # rag | chat (same for now)
    top_k: int = Field(default=5, ge=1, le=20)
    doc_id: Optional[str] = None


class StepTrace(BaseModel):
    name: str
    ms: int
    meta: Optional[dict[str, Any]] = None


class Citation(BaseModel):
    doc_id: str
    chunk_id: int
    source: Optional[str] = None


class RunResponse(BaseModel):
    request_id: str
    mode: str
    answer: str
    citations: list[Citation]
    steps: list[StepTrace]
    mock_llm: bool


class UploadRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=200000)
    source: str = Field(default="manual")
    doc_id: Optional[str] = None
    chunk_size: int = 800
    overlap: int = 120


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)
    doc_id: Optional[str] = None


class ErrorInfo(BaseModel):
    code: str
    message: str
    details: Optional[Any] = None


class ErrorResponse(BaseModel):
    request_id: str
    error: ErrorInfo


# ----------------------------
# App + config
# ----------------------------
app = FastAPI(title="genai-agent-workbench", version="0.2.0")

GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY", "").strip()
MOCK_LLM = os.getenv("MOCK_LLM", "true").lower() in {"1", "true", "yes", "y"}

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview").strip()
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()


def _get_aoai_client() -> AzureOpenAI:
    missing = []
    if not AZURE_OPENAI_ENDPOINT:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not AZURE_OPENAI_API_KEY:
        missing.append("AZURE_OPENAI_API_KEY")
    if not AZURE_OPENAI_DEPLOYMENT:
        missing.append("AZURE_OPENAI_DEPLOYMENT")
    if missing:
        raise HTTPException(status_code=500, detail=f"Server misconfigured: missing {', '.join(missing)}.")
    return AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )


# ----------------------------
# Request tracing
# ----------------------------
def _get_request_id(request: Request) -> str:
    rid = getattr(request.state, "request_id", None)
    if rid:
        return rid
    rid = request.headers.get("x-request-id") or str(uuid4())
    request.state.request_id = rid
    return rid


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid4())
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["x-request-id"] = rid
    return response


# ----------------------------
# Error envelope
# ----------------------------
def _error_json(request: Request, status: int, code: str, message: str, details: Any = None):
    rid = _get_request_id(request)
    payload = ErrorResponse(
        request_id=rid,
        error=ErrorInfo(code=code, message=message, details=details),
    ).model_dump()
    resp = JSONResponse(status_code=status, content=payload)
    resp.headers["x-request-id"] = rid
    return resp


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return _error_json(request, 400, "INVALID_REQUEST", "Request validation failed.", exc.errors())


@app.exception_handler(HTTPException)
async def http_handler(request: Request, exc: HTTPException):
    code = "HTTP_ERROR"
    if exc.status_code == 401:
        code = "UNAUTHORIZED"
    elif 400 <= exc.status_code < 500:
        code = "BAD_REQUEST"
    elif exc.status_code >= 500:
        code = "SERVER_ERROR"
    msg = exc.detail if isinstance(exc.detail, str) else "Request failed."
    return _error_json(request, exc.status_code, code, msg)


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    return _error_json(request, 500, "INTERNAL_ERROR", "Internal server error.")


# ----------------------------
# Auth
# ----------------------------
def _require_key(x_api_key: Optional[str]):
    if not GATEWAY_API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfigured: missing GATEWAY_API_KEY.")
    if not x_api_key or x_api_key != GATEWAY_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: missing or invalid x-api-key.")


# ----------------------------
# Startup
# ----------------------------
@app.on_event("startup")
def startup():
    try:
        ensure_collection()
        print("[startup] Qdrant collection ensured.", file=sys.stderr)
    except Exception as e:
        print(f"[startup] Qdrant init FAILED: {type(e).__name__}: {e}", file=sys.stderr)


# ----------------------------
# LangGraph RAG agent
# ----------------------------
class AgentState(TypedDict, total=False):
    question: str
    top_k: int
    doc_id: Optional[str]
    hits: List[Dict[str, Any]]
    answer: str
    citations: List[Dict[str, Any]]


def _node_retrieve(state: AgentState) -> AgentState:
    hits = rag_search(
        query=state["question"],
        top_k=int(state.get("top_k", 5)),
        doc_id=state.get("doc_id"),
    )
    return {"hits": hits}


def _node_generate(state: AgentState) -> AgentState:
    hits = state.get("hits", [])
    # Build a compact context block (don’t flood tokens)
    ctx_lines = []
    citations = []
    for h in hits[:8]:
        did = h.get("doc_id")
        cid = h.get("chunk_id")
        src = h.get("source")
        txt = (h.get("text") or "").strip()
        ctx_lines.append(f"[doc_id={did} chunk_id={cid} source={src}] {txt}")
        if did is not None and cid is not None:
            citations.append({"doc_id": str(did), "chunk_id": int(cid), "source": src})

    if MOCK_LLM:
        ans = f"(mock) Based on retrieved context: {ctx_lines[0] if ctx_lines else 'no context'}"
        return {"answer": ans, "citations": citations}

    client = _get_aoai_client()

    system = (
        "You are a banking-grade assistant. Use ONLY the provided context to answer.\n"
        "Return STRICT JSON with keys: answer (string), citations (array of objects {doc_id, chunk_id}).\n"
        "If context is insufficient, say so in the answer and return an empty citations list.\n"
        "No markdown. JSON only."
    )

    user = (
        f"QUESTION: {state['question']}\n\n"
        f"CONTEXT:\n" + "\n".join(ctx_lines[:8])
    )

    try:
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        raw = (resp.choices[0].message.content or "").strip()

        # Parse JSON response; if model misbehaves, fallback safely.
        try:
            obj = json.loads(raw)
            ans = str(obj.get("answer", "")).strip()
            cits = obj.get("citations", [])
            if not isinstance(cits, list):
                cits = []
            # sanitize citations shape
            cleaned = []
            for c in cits:
                if not isinstance(c, dict):
                    continue
                did = c.get("doc_id")
                cid = c.get("chunk_id")
                if did is None or cid is None:
                    continue
                cleaned.append({"doc_id": str(did), "chunk_id": int(cid), "source": None})
            return {"answer": ans, "citations": cleaned}
        except Exception:
            # fallback: answer raw, citations from retrieved hits
            return {"answer": raw, "citations": citations}

    except RateLimitError:
        raise HTTPException(status_code=503, detail="Upstream LLM is rate limiting. Try again.")
    except APITimeoutError:
        raise HTTPException(status_code=504, detail="Upstream LLM timed out.")
    except APIConnectionError:
        raise HTTPException(status_code=503, detail="Upstream LLM connection failed.")
    except APIStatusError:
        raise HTTPException(status_code=502, detail="Upstream LLM returned an error.")
    except Exception:
        raise HTTPException(status_code=502, detail="Upstream LLM request failed.")


_graph = StateGraph(AgentState)
_graph.add_node("retrieve", _node_retrieve)
_graph.add_node("generate", _node_generate)
_graph.set_entry_point("retrieve")
_graph.add_edge("retrieve", "generate")
_graph.add_edge("generate", END)
AGENT = _graph.compile()


# ----------------------------
# Routes
# ----------------------------
@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/api/upload")
def upload(req: UploadRequest, request: Request, x_api_key: Optional[str] = Header(default=None, alias="x-api-key")):
    _require_key(x_api_key)
    result = ingest_text(
        text=req.text,
        source=req.source,
        doc_id=req.doc_id,
        chunk_size=req.chunk_size,
        overlap=req.overlap,
    )
    return {"request_id": _get_request_id(request), **result}


@app.post("/api/search")
def search(req: SearchRequest, request: Request, x_api_key: Optional[str] = Header(default=None, alias="x-api-key")):
    _require_key(x_api_key)
    hits = rag_search(query=req.query, top_k=req.top_k, doc_id=req.doc_id)
    return {"request_id": _get_request_id(request), "hits": hits}


@app.post("/api/run", response_model=RunResponse)
def run(req: RunRequest, request: Request, x_api_key: Optional[str] = Header(default=None, alias="x-api-key")):
    _require_key(x_api_key)
    rid = _get_request_id(request)

    steps: list[StepTrace] = []

    t0 = time.perf_counter()
    state: AgentState = {"question": req.message, "top_k": req.top_k, "doc_id": req.doc_id}
    # retrieve
    out1 = _node_retrieve(state)
    state.update(out1)
    steps.append(StepTrace(name="retrieve", ms=int((time.perf_counter() - t0) * 1000), meta={"hits": len(state.get("hits", []))}))

    # generate (LLM)
    t1 = time.perf_counter()
    out2 = _node_generate(state)
    state.update(out2)
    steps.append(StepTrace(name="generate", ms=int((time.perf_counter() - t1) * 1000), meta={"mock": MOCK_LLM}))

    cits = state.get("citations", []) or []
    citations = [Citation(**c) for c in cits if isinstance(c, dict)]

    return RunResponse(
        request_id=rid,
        mode=req.mode,
        answer=state.get("answer", ""),
        citations=citations,
        steps=steps,
        mock_llm=MOCK_LLM,
    )
