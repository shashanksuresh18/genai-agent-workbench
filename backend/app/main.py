import os
import sys
import json
import time
import logging
from uuid import uuid4
from pathlib import Path
from typing import Any, Optional, TypedDict, List, Dict

# Load backend/.env for local dev (safe: doesn't override real env vars)
from dotenv import load_dotenv
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"  # backend/.env
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=False)

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from app.rag.store import ensure_collection
from app.rag.ingest import ingest_text, search as rag_search
from app.auth import auth_dep, auth_dep_metrics, AuthContext, AUTH_MODE

from openai import AzureOpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError, APIStatusError

from langgraph.graph import StateGraph, END


# ----------------------------
# Metrics
# ----------------------------
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["auth_type", "method", "path", "status"],
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)

RAG_SEARCH_DURATION = Histogram(
    "rag_search_duration_seconds",
    "RAG search duration in seconds",
    ["op"],
)

AOAI_CALLS_TOTAL = Counter("aoai_calls_total", "Azure OpenAI calls total")
AOAI_RETRIES_TOTAL = Counter("aoai_retries_total", "Azure OpenAI retries total")


# ----------------------------
# Models (API contract)
# ----------------------------
class RunRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    mode: str = Field(default="rag")
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
app = FastAPI(title="genai-agent-workbench", version="0.3.0")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, stream=sys.stdout, format="%(message)s")
logger = logging.getLogger("workbench")

MOCK_LLM = os.getenv("MOCK_LLM", "true").strip().lower() in {"1", "true", "yes", "y"}

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview").strip()
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()

AOAI_TIMEOUT_SECONDS = float(os.getenv("AOAI_TIMEOUT_SECONDS", "20"))
AOAI_MAX_RETRIES = int(os.getenv("AOAI_MAX_RETRIES", "2"))
AOAI_RETRY_BASE_MS = int(os.getenv("AOAI_RETRY_BASE_MS", "250"))


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
# Request + logs + metrics middleware
# ----------------------------
def _get_request_id(request: Request) -> str:
    rid = getattr(request.state, "request_id", None)
    if rid:
        return rid
    rid = request.headers.get("x-request-id") or str(uuid4())
    request.state.request_id = rid
    return rid


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid4())
    request.state.request_id = rid

    start = time.perf_counter()
    status_code = 500
    response = None

    try:
        response = await call_next(request)
        status_code = response.status_code

        # attach headers on success responses
        response.headers["x-request-id"] = rid
        response.headers["x-auth-mode"] = AUTH_MODE
        return response

    except Exception:
        status_code = 500
        raise

    finally:
        dur_s = max(0.0, time.perf_counter() - start)
        auth_type = getattr(request.state, "auth_type", None) or "none"

        HTTP_REQUESTS_TOTAL.labels(
            auth_type=auth_type,
            method=request.method,
            path=request.url.path,
            status=str(status_code),
        ).inc()

        HTTP_REQUEST_DURATION.labels(
            method=request.method,
            path=request.url.path,
        ).observe(dur_s)

        event = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "request_id": rid,
            "method": request.method,
            "path": request.url.path,
            "status": status_code,
            "duration_ms": int(dur_s * 1000),
            "auth_type": auth_type,
        }
        logger.info(json.dumps(event, separators=(",", ":")))


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
    resp.headers["x-auth-mode"] = AUTH_MODE
    return resp


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return _error_json(request, 400, "INVALID_REQUEST", "Request validation failed.", exc.errors())


@app.exception_handler(HTTPException)
async def http_handler(request: Request, exc: HTTPException):
    code = "HTTP_ERROR"
    if exc.status_code == 401:
        code = "UNAUTHORIZED"
    elif exc.status_code == 403:
        code = "FORBIDDEN"
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
# Startup
# ----------------------------
@app.on_event("startup")
def startup():
    try:
        ensure_collection()
        print(f"[startup] Qdrant collection ensured. ENV={ENV_PATH}", file=sys.stderr)
        print(f"[startup] AUTH_MODE={AUTH_MODE} MOCK_LLM={MOCK_LLM}", file=sys.stderr)
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
    with RAG_SEARCH_DURATION.labels(op="search").time():
        hits = rag_search(
            query=state["question"],
            top_k=int(state.get("top_k", 5)),
            doc_id=state.get("doc_id"),
        )
    return {"hits": hits}


def _aoai_chat_json(system: str, user: str) -> str:
    client = _get_aoai_client()

    for attempt in range(AOAI_MAX_RETRIES + 1):
        if attempt > 0:
            AOAI_RETRIES_TOTAL.inc()

        try:
            AOAI_CALLS_TOTAL.inc()
            resp = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                timeout=AOAI_TIMEOUT_SECONDS,
            )
            return (resp.choices[0].message.content or "").strip()

        except RateLimitError:
            if attempt >= AOAI_MAX_RETRIES:
                raise HTTPException(status_code=503, detail="Upstream LLM is rate limiting. Try again.")
        except APITimeoutError:
            if attempt >= AOAI_MAX_RETRIES:
                raise HTTPException(status_code=504, detail="Upstream LLM timed out.")
        except APIConnectionError:
            if attempt >= AOAI_MAX_RETRIES:
                raise HTTPException(status_code=503, detail="Upstream LLM connection failed.")
        except APIStatusError:
            if attempt >= AOAI_MAX_RETRIES:
                raise HTTPException(status_code=502, detail="Upstream LLM returned an error.")
        except Exception:
            if attempt >= AOAI_MAX_RETRIES:
                raise HTTPException(status_code=502, detail="Upstream LLM request failed.")

        sleep_s = (AOAI_RETRY_BASE_MS * (2 ** attempt)) / 1000.0
        time.sleep(sleep_s)

    raise HTTPException(status_code=502, detail="Upstream LLM request failed.")


def _node_generate(state: AgentState) -> AgentState:
    hits = state.get("hits", [])
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

    ctx_text = "\n".join(ctx_lines[:8])

    if MOCK_LLM:
        ans = f"(mock) Based on retrieved context: {ctx_lines[0] if ctx_lines else 'no context'}"
        return {"answer": ans, "citations": citations}

    system = (
        "You are a banking-grade assistant. Use ONLY the provided context to answer.\n"
        "Return STRICT JSON with keys: answer (string), citations (array of objects {doc_id, chunk_id}).\n"
        "If context is insufficient, say so in the answer and return an empty citations list.\n"
        "No markdown. JSON only."
    )
    user = f"QUESTION: {state['question']}\n\nCONTEXT:\n{ctx_text}"

    raw = _aoai_chat_json(system=system, user=user)

    try:
        obj = json.loads(raw)
        ans = str(obj.get("answer", "")).strip()
        cits = obj.get("citations", [])
        if not isinstance(cits, list):
            cits = []

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
        # fallback: raw model text + retrieved citations
        return {"answer": raw, "citations": citations}


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


@app.get("/metrics")
def metrics(_: AuthContext = Depends(auth_dep_metrics)):
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/whoami")
def whoami(request: Request, auth: AuthContext = Depends(auth_dep)):
    return {
        "request_id": _get_request_id(request),
        "auth_type": auth.auth_type,
        "oid": auth.oid,
        "tid": auth.tid,
        "upn": auth.upn,
    }


@app.post("/api/upload")
def upload(req: UploadRequest, request: Request, _: AuthContext = Depends(auth_dep)):
    result = ingest_text(
        text=req.text,
        source=req.source,
        doc_id=req.doc_id,
        chunk_size=req.chunk_size,
        overlap=req.overlap,
    )
    return {"request_id": _get_request_id(request), **result}


@app.post("/api/search")
def search(req: SearchRequest, request: Request, _: AuthContext = Depends(auth_dep)):
    hits = rag_search(query=req.query, top_k=req.top_k, doc_id=req.doc_id)
    return {"request_id": _get_request_id(request), "hits": hits}


@app.post("/api/run", response_model=RunResponse)
def run(req: RunRequest, request: Request, auth: AuthContext = Depends(auth_dep)):
    rid = _get_request_id(request)
    steps: list[StepTrace] = []

    t0 = time.perf_counter()
    state: AgentState = {"question": req.message, "top_k": req.top_k, "doc_id": req.doc_id}

    out1 = _node_retrieve(state)
    state.update(out1)
    steps.append(
        StepTrace(
            name="retrieve",
            ms=int((time.perf_counter() - t0) * 1000),
            meta={"hits": len(state.get("hits", []))},
        )
    )

    t1 = time.perf_counter()
    out2 = _node_generate(state)
    state.update(out2)
    steps.append(
        StepTrace(
            name="generate",
            ms=int((time.perf_counter() - t1) * 1000),
            meta={
                "mock": MOCK_LLM,
                "auth_type": auth.auth_type,
                "timeout_s": AOAI_TIMEOUT_SECONDS,
                "max_retries": AOAI_MAX_RETRIES,
                "ctx_chars": sum(len((h.get("text") or "")) for h in (state.get("hits") or [])[:8]),
                "ctx_chunks": min(len(state.get("hits") or []), 8),
            },
        )
    )

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
