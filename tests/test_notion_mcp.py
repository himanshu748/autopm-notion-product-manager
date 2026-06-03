from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main


class FakeServerParameters:
    def __init__(self, *, command: str, args: list[str], env: dict[str, str]) -> None:
        self.command = command
        self.args = args
        self.env = env


class FakeClientSession:
    def __init__(self, read: object, write: object) -> None:
        self.initialized = False

    async def __aenter__(self) -> "FakeClientSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def initialize(self) -> None:
        self.initialized = True

    async def call_tool(self, tool: str, args: dict) -> SimpleNamespace:
        assert self.initialized is True
        assert tool == "API-get-self"
        assert args == {}
        return SimpleNamespace(
            content=[SimpleNamespace(text='{"id":"notion-user","name":"AutoPM"}')]
        )


class FakeStdioClient:
    def __init__(self, params: FakeServerParameters) -> None:
        self.params = params

    async def __aenter__(self) -> tuple[object, object]:
        assert self.params.command == "npx"
        assert self.params.args == ["-y", "@notionhq/notion-mcp-server"]
        assert self.params.env["NOTION_TOKEN"] == "ntn_test"
        return object(), object()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


@pytest.mark.asyncio
async def test_notion_mcp_uses_official_stdio_server(monkeypatch):
    monkeypatch.setenv("NOTION_TOKEN", "ntn_test")
    monkeypatch.setattr(main, "StdioServerParameters", FakeServerParameters)
    monkeypatch.setattr(main, "ClientSession", FakeClientSession)
    monkeypatch.setattr(main, "stdio_client", lambda params: FakeStdioClient(params))

    async with main.notion_session() as session:
        result = await main.mcp_call(session, "API-get-self", {})

    assert result == {"id": "notion-user", "name": "AutoPM"}
    assert main.notion_transport_name() == "mcp-stdio"


@pytest.mark.asyncio
async def test_notion_mcp_requires_token(monkeypatch):
    monkeypatch.delenv("NOTION_TOKEN", raising=False)

    with pytest.raises(main.HTTPException, match="NOTION_TOKEN"):
        async with main.notion_session():
            pass


def test_health_does_not_require_credentials(monkeypatch):
    monkeypatch.delenv("HF_API_KEY", raising=False)
    monkeypatch.delenv("NOTION_TOKEN", raising=False)
    monkeypatch.delenv("NOTION_PARENT_PAGE_ID", raising=False)

    client = TestClient(main.app)
    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["hf_key"] is False
    assert payload["notion_token"] is False
    assert payload["parent_page_id"] is False


def test_prd_request_rejects_short_idea():
    client = TestClient(main.app)

    response = client.post("/api/generate-prd", json={"idea": "tiny"})

    assert response.status_code == 422


def test_standup_requires_parent_page_when_env_missing(monkeypatch):
    monkeypatch.setenv("HF_API_KEY", "hf_test")
    monkeypatch.setenv("NOTION_TOKEN", "ntn_test")
    monkeypatch.delenv("NOTION_PARENT_PAGE_ID", raising=False)

    client = TestClient(main.app)
    response = client.post("/api/standup", json={})

    assert response.status_code == 400
    assert response.json()["detail"] == "parent_page_id required"


@pytest.mark.asyncio
async def test_rest_fallback_does_not_mutate_args(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200
        text = "{}"

        def json(self):
            return {"object": "list", "results": []}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, headers=None, params=None):
            captured["params"] = dict(params)
            return FakeResponse()

    monkeypatch.setenv("NOTION_TOKEN", "ntn_test")
    monkeypatch.setattr(main.httpx, "AsyncClient", FakeClient)

    args = {"block_id": "block-1", "page_size": 100}
    result = await main.NotionHTTPFallback().call_tool("API-get-block-children", args)

    assert result == {"object": "list", "results": []}
    assert args == {"block_id": "block-1", "page_size": 100}
    assert captured["params"] == {"page_size": 100}
