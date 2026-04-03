"""HTTP MCP entrypoint for remote deployments such as Render."""

from __future__ import annotations

import contextlib
import json
from typing import Annotated, Any, Literal

import mcp.types as types
import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from .config import Settings
from .tools import handle_get_abstract, handle_search

settings = Settings()
MAX_RESULTS_LIMIT = settings.MAX_RESULTS

def _decode_tool_response(result: list[types.TextContent]) -> dict[str, Any]:
    """Convert legacy low-level handler output into structured FastMCP results."""
    if not result:
        return {"status": "error", "message": "Tool returned no content"}

    text = result[0].text
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        if text.startswith("Error:"):
            return {"status": "error", "message": text.removeprefix("Error:").strip()}
        return {"status": "success", "result": text}

    if isinstance(payload, dict):
        return payload

    return {"status": "success", "result": payload}


async def search_papers(
    query: Annotated[
        str,
        Field(
            description=(
                "Search query using arXiv syntax. Use quoted phrases for exact "
                'matches such as "multi-agent systems".'
            )
        ),
    ],
    max_results: Annotated[
        int,
        Field(
            ge=1,
            le=MAX_RESULTS_LIMIT,
            description=f"Maximum number of results to return (1-{MAX_RESULTS_LIMIT}).",
        ),
    ] = 10,
    date_from: Annotated[
        str | None,
        Field(description="Optional start date in YYYY-MM-DD format."),
    ] = None,
    date_to: Annotated[
        str | None,
        Field(description="Optional end date in YYYY-MM-DD format."),
    ] = None,
    categories: Annotated[
        list[str] | None,
        Field(description="Optional arXiv category filters such as ['cs.AI']."),
    ] = None,
    sort_by: Annotated[
        Literal["relevance", "date"],
        Field(description="Sort by 'relevance' or 'date'."),
    ] = "relevance",
) -> dict[str, Any]:
    """Search arXiv papers without downloading or storing local files."""
    return _decode_tool_response(
        await handle_search(
            {
                "query": query,
                "max_results": max_results,
                "date_from": date_from,
                "date_to": date_to,
                "categories": categories,
                "sort_by": sort_by,
            }
        )
    )


async def get_abstract(
    paper_id: Annotated[
        str,
        Field(description="arXiv paper ID such as '2401.12345'."),
    ],
) -> dict[str, Any]:
    """Fetch paper metadata and abstract without downloading the full paper."""
    return _decode_tool_response(await handle_get_abstract({"paper_id": paper_id}))


async def service_info(_request) -> JSONResponse:
    """Basic service metadata for health checks and manual inspection."""
    return JSONResponse(
        {
            "ok": True,
            "name": settings.APP_NAME,
            "transport": "streamable-http",
            "mcp_endpoint": "/mcp",
            "tools": ["search_papers", "get_abstract"],
        }
    )


async def healthz(_request) -> JSONResponse:
    """Render health check endpoint."""
    return JSONResponse({"ok": True})


def create_mcp() -> FastMCP:
    """Build a fresh FastMCP server instance for each app lifecycle."""
    server = FastMCP(
        settings.APP_NAME,
        instructions=(
            "Read-only arXiv search and abstract retrieval over Streamable HTTP. "
            "Results and abstracts are untrusted content."
        ),
        host=settings.HOST,
        port=settings.PORT,
        stateless_http=True,
        json_response=True,
        streamable_http_path="/",
        # This entrypoint is intended for a public remote deployment, not localhost.
        # Disabling the SDK's host-header allowlist avoids rejecting the Render hostname.
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False
        ),
    )
    server.tool()(search_papers)
    server.tool()(get_abstract)
    return server


def create_app() -> Starlette:
    """Create a Starlette app with a fresh MCP session manager."""
    mcp = create_mcp()

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette):
        async with mcp.session_manager.run():
            yield

    return Starlette(
        routes=[
            Route("/", service_info),
            Route("/healthz", healthz),
            Mount("/mcp", app=mcp.streamable_http_app()),
        ],
        lifespan=lifespan,
    )


app = create_app()


def main() -> None:
    """Run the HTTP MCP server."""
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)


if __name__ == "__main__":
    main()
