"""
AutoPM — AI Product Manager powered by HuggingFace + Notion MCP
Run: uvicorn main:app --reload

Architecture:
  HuggingFace Inference API  →  structured content generation
  Notion MCP (stdio server)  →  ALL Notion reads & writes via MCP protocol
  FastAPI backend             →  orchestrates both
"""

import os, json, logging
from datetime import date
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional
from huggingface_hub import InferenceClient
import httpx

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ModuleNotFoundError:
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None

log = logging.getLogger("autopm")

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


def get_env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def hf_api_key() -> str:
    return get_env("HF_API_KEY") or get_env("HF_TOKEN")


def notion_token() -> str:
    return get_env("NOTION_TOKEN") or get_env("NOTION_API_KEY")


def notion_parent_page_id() -> str:
    return get_env("NOTION_PARENT_PAGE_ID")


def hf_model() -> str:
    return get_env("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")

app = FastAPI(title="AutoPM")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


# ─── Notion transport layer (MCP primary, httpx fallback) ────────────────────

NOTION_API = "https://api.notion.com/v1"
NOTION_VER = "2022-06-28"


class NotionHTTPFallback:
    """Direct Notion REST client — used when MCP stdio is unavailable (e.g. Vercel)."""

    def _h(self):
        return {"Authorization": f"Bearer {notion_token()}",
                "Notion-Version": NOTION_VER, "Content-Type": "application/json"}

    @staticmethod
    def _required(payload: dict, key: str, tool: str):
        try:
            return payload.pop(key)
        except KeyError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Notion REST fallback missing required argument '{key}' for {tool}.",
            ) from exc

    @staticmethod
    def _json_response(response: httpx.Response) -> dict:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            raise HTTPException(
                status_code=502,
                detail=f"Notion REST request failed with HTTP {status}.",
            ) from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise HTTPException(
                status_code=502,
                detail="Notion REST returned invalid JSON.",
            ) from exc
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=502,
                detail="Notion REST returned an unexpected payload shape.",
            )
        return payload

    async def call_tool(self, tool: str, args: dict) -> dict:
        payload = dict(args)
        async with httpx.AsyncClient(timeout=30) as c:
            if tool == "API-post-page":
                r = await c.post(f"{NOTION_API}/pages", headers=self._h(), json=payload)
            elif tool == "API-post-search":
                r = await c.post(f"{NOTION_API}/search", headers=self._h(), json=payload)
            elif tool == "API-get-block-children":
                bid = self._required(payload, "block_id", tool)
                r = await c.get(f"{NOTION_API}/blocks/{bid}/children", headers=self._h(), params=payload)
            elif tool == "API-get-self":
                r = await c.get(f"{NOTION_API}/users/me", headers=self._h())
            elif tool == "API-patch-page":
                pid = self._required(payload, "page_id", tool)
                r = await c.patch(f"{NOTION_API}/pages/{pid}", headers=self._h(), json=payload)
            elif tool == "API-retrieve-a-page":
                pid = self._required(payload, "page_id", tool)
                r = await c.get(f"{NOTION_API}/pages/{pid}", headers=self._h())
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Unknown Notion tool: {tool}.",
                )
            return self._json_response(r)


def mcp_package_available() -> bool:
    return ClientSession is not None and StdioServerParameters is not None and stdio_client is not None


def notion_transport_name() -> str:
    return "mcp-stdio" if mcp_package_available() else "rest-fallback"


@asynccontextmanager
async def notion_mcp():
    """Spin up Notion MCP stdio server and yield a ClientSession."""
    token = notion_token()
    if not token:
        raise HTTPException(status_code=500, detail="NOTION_TOKEN not set")
    if not mcp_package_available():
        log.warning("mcp package is not installed; using Notion REST fallback.")
        yield NotionHTTPFallback()
        return
    params = StdioServerParameters(
        command="npx",
        args=["-y", "@notionhq/notion-mcp-server"],
        env={**os.environ, "NOTION_TOKEN": token},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


notion_session = notion_mcp


async def mcp_call(session, tool: str, args: dict) -> dict:
    if isinstance(session, NotionHTTPFallback):
        return await session.call_tool(tool, args)
    result = await session.call_tool(tool, args)
    return parse_mcp_tool_result(result, tool)


def parse_mcp_tool_result(result, tool: str) -> dict:
    content = getattr(result, "content", None) or []
    if not content:
        return {}
    text = getattr(content[0], "text", None)
    if text is None:
        raise HTTPException(
            status_code=502,
            detail=f"Notion MCP returned non-text content for {tool}.",
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Notion MCP returned invalid JSON for {tool}.",
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=502,
            detail=f"Notion MCP returned an unexpected payload shape for {tool}.",
        )
    return payload


def _rt(content: str) -> list:
    return [{"text": {"content": content}}]


def _heading(text: str, level: int = 2) -> dict:
    k = f"heading_{level}"
    return {"object": "block", "type": k, k: {"rich_text": _rt(text)}}


def _para(text: str) -> dict:
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": _rt(text)}}


def _bullet(text: str) -> dict:
    return {"object": "block", "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": _rt(text)}}


async def mcp_create_page(session, parent_id: str, title: str, children: list) -> dict:
    """Create a Notion page via MCP API-post-page tool."""
    result = await mcp_call(session, "API-post-page", {
        "parent": {"page_id": parent_id},
        "properties": {"title": {"title": _rt(title)}},
        "children": children[:100],
    })
    if result.get("status") and result["status"] >= 400:
        raise HTTPException(status_code=502,
            detail=f"Notion MCP error: {result.get('message', str(result)[:200])}")
    return result


async def mcp_search(session, query: str = "") -> list:
    result = await mcp_call(session, "API-post-search", {"query": query, "page_size": 50})
    return result.get("results", [])


async def mcp_get_children(session, block_id: str) -> list:
    result = await mcp_call(session, "API-get-block-children",
                            {"block_id": block_id, "page_size": 100})
    return result.get("results", [])


# ─── HuggingFace ─────────────────────────────────────────────────────────────

async def generate_text(system: str, user_msg: str) -> str:
    hf = InferenceClient(model=hf_model(), token=hf_api_key())
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
    out = ""
    for chunk in hf.chat_completion(messages=messages, max_tokens=4096, stream=True):
        if chunk.choices:
            d = chunk.choices[0].delta
            if d.content:
                out += d.content
    return out


def _parse_json(raw: str) -> dict:
    s, e = raw.find("{"), raw.rfind("}") + 1
    if s == -1 or e <= s:
        raise HTTPException(status_code=502, detail="Model did not return valid JSON")
    try:
        payload = json.loads(raw[s:e])
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="Model did not return valid JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=502, detail="Model did not return a JSON object")
    return payload


# ─── Block builders ──────────────────────────────────────────────────────────

PRD_SYSTEM = """You are AutoPM, an expert AI Product Manager.
Given a product idea, generate a detailed PRD in this JSON format (no markdown fences):
{
  "title": "PRD: <Product Name>",
  "problem": "...",
  "goals": ["goal1", "goal2", ...],
  "personas": [{"name": "...", "description": "..."}],
  "user_stories": ["As a ..., I want ..., so that ..."],
  "out_of_scope": ["..."],
  "epics": [
    {"name": "...", "stories": [
      {"name": "...", "priority": "High|Medium|Low", "points": 1|2|3|5|8, "tasks": ["...", "..."]}
    ]}
  ],
  "sprint1_goal": "...",
  "sprint1_stories": ["story names totaling ~20 points"]
}
Be thorough, specific, and actionable. At least 3 epics, 10 stories, 15 tasks."""


def build_prd_blocks(prd: dict) -> list:
    b = []
    b.append(_heading("Problem Statement"))
    b.append(_para(prd.get("problem", "")))
    b.append(_heading("Goals & Success Metrics"))
    for g in prd.get("goals", []):
        b.append(_bullet(g))
    b.append(_heading("User Personas"))
    for p in prd.get("personas", []):
        b.append(_bullet(f'{p.get("name","")}: {p.get("description","")}'))
    b.append(_heading("User Stories"))
    for s in prd.get("user_stories", []):
        b.append(_bullet(s))
    b.append(_heading("Out of Scope"))
    for o in prd.get("out_of_scope", []):
        b.append(_bullet(o))
    b.append(_heading("Sprint 1 Plan"))
    b.append(_para(prd.get("sprint1_goal", "")))
    for s in prd.get("sprint1_stories", []):
        b.append(_bullet(s))
    return b


def build_task_blocks(prd: dict) -> list:
    b = []
    for epic in prd.get("epics", []):
        b.append(_heading(f'Epic: {epic.get("name","")}'))
        for story in epic.get("stories", []):
            lbl = f'{story.get("name","")} [{story.get("priority","Med")}] ({story.get("points",0)}pts)'
            b.append(_heading(lbl, level=3))
            for task in story.get("tasks", []):
                b.append(_bullet(task))
    return b


# ─── Models ───────────────────────────────────────────────────────────────────

class PRDRequest(BaseModel):
    idea: str = Field(..., min_length=10, max_length=2000)
    parent_page_id: Optional[str] = None

class StandupRequest(BaseModel):
    parent_page_id: Optional[str] = None
    date: Optional[str] = Field(default=None, max_length=40)

class SprintRequest(BaseModel):
    sprint_num: int = Field(default=1, ge=1, le=99)
    parent_page_id: Optional[str] = None


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    with open(BASE_DIR / "static" / "index.html") as f:
        return f.read()


@app.post("/api/generate-prd")
async def generate_prd_route(req: PRDRequest):
    if not hf_api_key():
        raise HTTPException(status_code=500, detail="HF_API_KEY not set")
    if not notion_token():
        raise HTTPException(status_code=500, detail="NOTION_TOKEN not set")
    parent_id = req.parent_page_id or notion_parent_page_id()
    if not parent_id:
        raise HTTPException(status_code=400, detail="parent_page_id required")

    raw = await generate_text(PRD_SYSTEM, f"Product idea: {req.idea}")
    prd = _parse_json(raw)

    async with notion_session() as mcp:
        prd_page = await mcp_create_page(mcp, parent_id, prd.get("title", "PRD"),
                                         build_prd_blocks(prd))
        task_page = await mcp_create_page(mcp, parent_id,
                                          f'{prd.get("title","PRD")} — Tasks',
                                          build_task_blocks(prd))

    return {
        "status": "success",
        "prd_url": prd_page.get("url", ""),
        "tasks_url": task_page.get("url", ""),
        "epics": len(prd.get("epics", [])),
        "stories": sum(len(e.get("stories", [])) for e in prd.get("epics", [])),
        "tasks": sum(len(s.get("tasks", [])) for e in prd.get("epics", [])
                     for s in e.get("stories", [])),
    }


@app.post("/api/standup")
async def generate_standup(req: StandupRequest):
    if not hf_api_key():
        raise HTTPException(status_code=500, detail="HF_API_KEY not set")
    if not notion_token():
        raise HTTPException(status_code=500, detail="NOTION_TOKEN not set")
    parent_id = req.parent_page_id or notion_parent_page_id()
    if not parent_id:
        raise HTTPException(status_code=400, detail="parent_page_id required")
    today = req.date or str(date.today())

    async with notion_session() as mcp:
        # READ from Notion via MCP — get workspace context
        pages = await mcp_search(mcp, "")
        titles = []
        for p in pages[:20]:
            for prop in p.get("properties", {}).values():
                if prop.get("type") == "title":
                    t = "".join(x.get("plain_text", "") for x in prop.get("title", []))
                    if t:
                        titles.append(t)
        context = ", ".join(titles) if titles else "No existing pages"

        system = f"""You are AutoPM's standup agent. Based on the workspace context, generate a standup in JSON:
{{
  "completed": ["task1", "task2"],
  "in_progress": ["task3"],
  "blockers": ["blocker1"],
  "health": "On track — 60% done, velocity 18pts/sprint"
}}
Workspace pages: {context}
Today: {today}"""

        raw = await generate_text(system, f"Generate standup for {today}")
        data = _parse_json(raw)

        blocks = [_heading(f"Daily Standup — {today}")]
        blocks.append(_heading("Completed Yesterday", 3))
        for item in data.get("completed", []):
            blocks.append(_bullet(item))
        blocks.append(_heading("In Progress Today", 3))
        for item in data.get("in_progress", []):
            blocks.append(_bullet(item))
        blocks.append(_heading("Blockers", 3))
        for item in data.get("blockers", []):
            blocks.append(_bullet(item))
        blocks.append(_heading("Sprint Health", 3))
        blocks.append(_para(data.get("health", "No data")))

        # WRITE to Notion via MCP
        page = await mcp_create_page(mcp, parent_id, f"Standup — {today}", blocks)

    return {"status": "success", "standup_url": page.get("url", ""), **data}


@app.post("/api/plan-sprint")
async def plan_sprint(req: SprintRequest):
    if not hf_api_key():
        raise HTTPException(status_code=500, detail="HF_API_KEY not set")
    if not notion_token():
        raise HTTPException(status_code=500, detail="NOTION_TOKEN not set")
    parent_id = req.parent_page_id or notion_parent_page_id()
    if not parent_id:
        raise HTTPException(status_code=400, detail="parent_page_id required")

    async with notion_session() as mcp:
        # READ backlog from Notion via MCP
        pages = await mcp_search(mcp, "Tasks")
        task_ctx = []
        for p in pages[:5]:
            for prop in p.get("properties", {}).values():
                if prop.get("type") == "title":
                    t = "".join(x.get("plain_text", "") for x in prop.get("title", []))
                    if t:
                        task_ctx.append(t)
        backlog = ", ".join(task_ctx) if task_ctx else "No existing task pages"

        system = f"""You are AutoPM's sprint planner. Based on backlog context, generate a sprint plan in JSON:
{{
  "goal": "Sprint goal statement",
  "stories": [{{"name": "...", "points": 3, "priority": "High"}}],
  "total_points": 20,
  "capacity": "Assuming 2 developers, 10 working days",
  "definition_of_done": "All stories reviewed, tested, deployed"
}}
Existing task pages: {backlog}
Sprint number: {req.sprint_num}"""

        raw = await generate_text(system, f"Plan sprint {req.sprint_num}")
        data = _parse_json(raw)

        blocks = [_heading(f"Sprint {req.sprint_num} Plan")]
        blocks.append(_heading("Sprint Goal", 3))
        blocks.append(_para(data.get("goal", "")))
        blocks.append(_heading("Selected Stories", 3))
        for s in data.get("stories", []):
            blocks.append(_bullet(
                f'{s.get("name","")} — {s.get("points",0)}pts [{s.get("priority","")}]'))
        blocks.append(_heading("Capacity", 3))
        blocks.append(_para(data.get("capacity", "")))
        blocks.append(_heading("Definition of Done", 3))
        blocks.append(_para(data.get("definition_of_done", "")))

        # WRITE to Notion via MCP
        page = await mcp_create_page(mcp, parent_id, f"Sprint {req.sprint_num} Plan", blocks)

    return {
        "status": "success",
        "sprint_url": page.get("url", ""),
        "total_points": data.get("total_points", 0),
        "task_count": len(data.get("stories", [])),
    }


@app.get("/api/health")
async def health():
    mcp_ok = False
    if notion_token():
        try:
            async with notion_session() as mcp:
                me = await mcp_call(mcp, "API-get-self", {})
                mcp_ok = notion_transport_name() == "mcp-stdio" and bool(me.get("id"))
        except Exception:
            pass
    return {
        "status": "ok",
        "hf_key": bool(hf_api_key()),
        "notion_token": bool(notion_token()),
        "parent_page_id": bool(notion_parent_page_id()),
        "mcp_connected": mcp_ok,
        "notion_transport": notion_transport_name(),
    }
