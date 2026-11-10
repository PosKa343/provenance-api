from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from provenance_api import __version__
from provenance_api.config import Settings
from provenance_api.db import create_engine, create_session_factory
from provenance_api.errors import install_error_handling
from provenance_api.ml.model import Classifier
from provenance_api.routes import admin, api_keys, auth, classify, health


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Loaded once per process: reading the model and tagger takes about half
        # a second, far too slow to repeat on every request.
        app.state.classifier = Classifier.load()
        engine = create_engine(settings.database_url)
        app.state.session_factory = create_session_factory(engine)
        yield
        await engine.dispose()

    app = FastAPI(
        title="Provenance API",
        version=__version__,
        summary="Classifies text as human-written or AI-generated.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    install_error_handling(app)
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(api_keys.router)
    app.include_router(classify.router)
    app.include_router(admin.router)
    return app
