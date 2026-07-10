"""FastAPI application for the KGraph chatbot.

Replaces the Flask app with async endpoints and native SSE streaming.
All routes are thin wrappers around the ChatPipeline orchestrator.
"""

import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from pipeline.orchestrator import ChatPipeline


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


# ---------------------------------------------------------------------------
# Application lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

pipeline: ChatPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage pipeline lifecycle: connect on startup, close on shutdown."""
    global pipeline

    pipeline = ChatPipeline(
        openai_api_key=_require_env("OPENAI_API_KEY"),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=_require_env("NEO4J_PASSWORD"),
        model=os.getenv("LLM_MODEL", "gpt-4o"),
    )
    await pipeline.start()
    print("Pipeline ready.")

    yield

    await pipeline.shutdown()
    print("Pipeline shut down.")


app = FastAPI(
    title="KGraph Chatbot",
    description="Temporal Knowledge Graph chatbot with narrative memory",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str


class ClearResponse(BaseModel):
    message: str


class GraphDataResponse(BaseModel):
    nodes: list[dict]
    edges: list[dict]


# ---------------------------------------------------------------------------
# SSE stream helper
# ---------------------------------------------------------------------------

async def event_stream(message: str):
    """Convert pipeline events to SSE format.

    Each event is sent as:
        data: {"type": "status", "content": "Checking timeline..."}\n\n
    """
    async for event in pipeline.handle_message(message):
        data = json.dumps({"type": event.type.value, "content": event.content})
        yield f"data: {data}\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the chatbot frontend."""
    return templates.TemplateResponse(request, "index.html")


@app.post("/chat-stream")
async def chat_stream(body: ChatRequest):
    """Stream chat responses via Server-Sent Events.

    The frontend receives events with these types:
    - status: Natural language progress message (e.g. "Checking timeline...")
    - token: A chunk of the response text
    - done: Response complete
    - rejected: Message blocked (moderation or conflict)
    - error: Something went wrong
    """
    return StreamingResponse(
        event_stream(body.message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/clear", response_model=ClearResponse)
async def clear_memory():
    """Clear all conversation memory and reset the knowledge graph."""
    result = await pipeline.clear_memory()
    return ClearResponse(message=result)


@app.get("/graph-data", response_model=GraphDataResponse)
async def graph_data():
    """Get the knowledge graph data for frontend visualization."""
    data = await pipeline.get_graph_data()
    return GraphDataResponse(**data)


# ---------------------------------------------------------------------------
# Dev server entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    print(f"Starting server at http://127.0.0.1:{port}")
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=True)
