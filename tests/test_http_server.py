"""Tests for the remote HTTP MCP entrypoint."""

import json

import mcp.types as types
import pytest
from starlette.testclient import TestClient

from arxiv_mcp_server.http_server import create_app, create_mcp, get_abstract, search_papers


def _text_result(payload: str) -> list[types.TextContent]:
    return [types.TextContent(type="text", text=payload)]


def test_service_info_endpoint():
    with TestClient(create_app()) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "name": "arxiv-mcp-server",
        "transport": "streamable-http",
        "mcp_endpoint": "/mcp",
        "tools": ["search_papers", "get_abstract"],
    }


def test_healthz_endpoint():
    with TestClient(create_app()) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_create_mcp_disables_localhost_host_validation():
    server = create_mcp()

    assert server.settings.host == "0.0.0.0"
    assert server.settings.transport_security is not None
    assert server.settings.transport_security.enable_dns_rebinding_protection is False


@pytest.mark.asyncio
async def test_search_papers_delegates_to_existing_handler(mocker):
    mocked = mocker.patch(
        "arxiv_mcp_server.http_server.handle_search",
        side_effect=[
            _text_result(json.dumps({"total_results": 1, "papers": [{"id": "1"}]}))
        ],
    )

    result = await search_papers(
        query="agents",
        max_results=5,
        date_from="2024-01-01",
        date_to="2024-12-31",
        categories=["cs.AI"],
        sort_by="date",
    )

    mocked.assert_awaited_once_with(
        {
            "query": "agents",
            "max_results": 5,
            "date_from": "2024-01-01",
            "date_to": "2024-12-31",
            "categories": ["cs.AI"],
            "sort_by": "date",
        }
    )
    assert result == {"total_results": 1, "papers": [{"id": "1"}]}


@pytest.mark.asyncio
async def test_get_abstract_normalizes_plain_text_errors(mocker):
    mocker.patch(
        "arxiv_mcp_server.http_server.handle_get_abstract",
        side_effect=[_text_result("Error: upstream unavailable")],
    )

    result = await get_abstract("2401.12345")

    assert result == {"status": "error", "message": "upstream unavailable"}
