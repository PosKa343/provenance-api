"""The real app plus one deliberately slow route, for the graceful-shutdown test."""

import asyncio

from fastapi import FastAPI

from provenance_api.main import create_app as create_real_app


def create_app() -> FastAPI:
    app = create_real_app()

    @app.get("/slow")
    async def slow() -> dict[str, str]:
        await asyncio.sleep(1.5)
        return {"status": "finished"}

    return app
