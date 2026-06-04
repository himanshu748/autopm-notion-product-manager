from __future__ import annotations

from types import SimpleNamespace

import httpx
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
    monkeypatch.delenv("HF_TOKEN", raising=False)
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


def test_hf_token_alias_configures_health(monkeypatch):
    monkeypatch.delenv("HF_API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.delenv("NOTION_TOKEN", raising=False)
    monkeypatch.delenv("NOTION_PARENT_PAGE_ID", raising=False)

    client = TestClient(main.app)
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["hf_key"] is True


def test_notion_api_key_alias_configures_health(monkeypatch):
    monkeypatch.delenv("NOTION_TOKEN", raising=False)
    monkeypatch.setenv("NOTION_API_KEY", "ntn_test")
    monkeypatch.delenv("NOTION_PARENT_PAGE_ID", raising=False)

    client = TestClient(main.app)
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["notion_token"] is True


def test_prd_request_rejects_short_idea():
    client = TestClient(main.app)

    response = client.post("/api/generate-prd", json={"idea": "tiny"})

    assert response.status_code == 422


def test_parse_json_rejects_malformed_model_payload():
    with pytest.raises(main.HTTPException) as exc:
        main._parse_json('{"title": "Broken",')

    assert exc.value.status_code == 502
    assert exc.value.detail == "Model did not return valid JSON"


def test_parse_json_rejects_payload_without_object():
    with pytest.raises(main.HTTPException) as exc:
        main._parse_json('["not", "an", "object"]')

    assert exc.value.status_code == 502
    assert exc.value.detail == "Model did not return valid JSON"


def test_parse_mcp_tool_result_rejects_invalid_json():
    result = SimpleNamespace(content=[SimpleNamespace(text="not json")])

    with pytest.raises(main.HTTPException) as exc:
        main.parse_mcp_tool_result(result, "API-get-self")

    assert exc.value.status_code == 502
    assert exc.value.detail == "Notion MCP returned invalid JSON for API-get-self."


def test_parse_mcp_tool_result_rejects_non_object_payload():
    result = SimpleNamespace(content=[SimpleNamespace(text="[]")])

    with pytest.raises(main.HTTPException) as exc:
        main.parse_mcp_tool_result(result, "API-get-self")

    assert exc.value.status_code == 502
    assert exc.value.detail == (
        "Notion MCP returned an unexpected payload shape for API-get-self."
    )


def test_parse_mcp_tool_result_rejects_non_text_content():
    result = SimpleNamespace(content=[SimpleNamespace(data={"id": "notion-user"})])

    with pytest.raises(main.HTTPException) as exc:
        main.parse_mcp_tool_result(result, "API-get-self")

    assert exc.value.status_code == 502
    assert exc.value.detail == "Notion MCP returned non-text content for API-get-self."


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
    class FakeResponse:
        def __init__(self, payload=None, *, status_code=200, json_error=False):
            self.payload = payload or {"object": "list", "results": []}
            self.status_code = status_code
            self.json_error = json_error

        def raise_for_status(self):
            if self.status_code >= 400:
                request = httpx.Request("GET", "https://api.notion.com/v1/test")
                response = httpx.Response(self.status_code, request=request)
                raise httpx.HTTPStatusError(
                    "Notion error body with private details",
                    request=request,
                    response=response,
                )

        def json(self):
            if self.json_error:
                raise ValueError("not json")
            return self.payload

    class FakeClient:
        instances = []
        next_response = None

        def __init__(self, *args, **kwargs):
            self.calls = []
            self.instances.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, headers=None, params=None):
            self.calls.append(("get", url, dict(params or {}), None))
            return self.next_response or FakeResponse()

    monkeypatch.setenv("NOTION_TOKEN", "ntn_test")
    monkeypatch.setattr(main.httpx, "AsyncClient", FakeClient)

    args = {"block_id": "block-1", "page_size": 100}
    result = await main.NotionHTTPFallback().call_tool("API-get-block-children", args)

    assert result == {"object": "list", "results": []}
    assert args == {"block_id": "block-1", "page_size": 100}
    assert FakeClient.instances[0].calls == [
        (
            "get",
            f"{main.NOTION_API}/blocks/block-1/children",
            {"page_size": 100},
            None,
        )
    ]


@pytest.mark.asyncio
async def test_rest_fallback_rejects_unknown_tools():
    with pytest.raises(main.HTTPException) as exc_info:
        await main.NotionHTTPFallback().call_tool("API-delete-everything", {})

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Unknown Notion tool: API-delete-everything."


@pytest.mark.asyncio
async def test_rest_fallback_requires_tool_arguments():
    with pytest.raises(main.HTTPException) as exc_info:
        await main.NotionHTTPFallback().call_tool("API-get-block-children", {})

    assert exc_info.value.status_code == 500
    assert "block_id" in exc_info.value.detail


@pytest.mark.asyncio
async def test_rest_fallback_raises_sanitized_http_errors(monkeypatch):
    class FakeResponse:
        status_code = 401

        def raise_for_status(self):
            request = httpx.Request("GET", "https://api.notion.com/v1/test")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError(
                "Notion error body with private details",
                request=request,
                response=response,
            )

        def json(self):
            return {"error": "private"}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, headers=None):
            return FakeResponse()

    monkeypatch.setattr(main.httpx, "AsyncClient", FakeClient)

    with pytest.raises(main.HTTPException) as exc_info:
        await main.NotionHTTPFallback().call_tool("API-get-self", {})

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "Notion REST request failed with HTTP 401."
    assert "private" not in exc_info.value.detail
    assert "ntn_test" not in exc_info.value.detail


@pytest.mark.asyncio
async def test_rest_fallback_rejects_invalid_json(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError("not json")

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, headers=None):
            return FakeResponse()

    monkeypatch.setattr(main.httpx, "AsyncClient", FakeClient)

    with pytest.raises(main.HTTPException) as exc_info:
        await main.NotionHTTPFallback().call_tool("API-get-self", {})

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "Notion REST returned invalid JSON."


@pytest.mark.asyncio
async def test_rest_fallback_rejects_unexpected_payload_shape(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, headers=None):
            return FakeResponse()

    monkeypatch.setattr(main.httpx, "AsyncClient", FakeClient)

    with pytest.raises(main.HTTPException) as exc_info:
        await main.NotionHTTPFallback().call_tool("API-get-self", {})

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "Notion REST returned an unexpected payload shape."
