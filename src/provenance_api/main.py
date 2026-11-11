from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from arq.connections import RedisSettings, create_pool
from fastapi import FastAPI

from provenance_api import __version__
from provenance_api.billing import gateway_from_settings
from provenance_api.config import Settings
from provenance_api.db import create_engine, create_session_factory
from provenance_api.errors import install_error_handling
from provenance_api.inference import InferencePool
from provenance_api.limits import Limiter
from provenance_api.logs import configure_logging
from provenance_api.observability import install_observability
from provenance_api.routes import (
    admin,
    api_keys,
    auth,
    billing,
    classify,
    health,
    jobs,
    usage,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_format, settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Built once per process: loading the model and starting the worker
        # processes takes seconds, far too slow to repeat on every request.
        app.state.inference = InferencePool(settings.inference_workers)
        await app.state.inference.warm_up()
        engine = create_engine(settings.database_url)
        app.state.session_factory = create_session_factory(engine)
        app.state.queue = await create_pool(RedisSettings.from_dsn(settings.redis_url))
        app.state.limiter = Limiter(app.state.queue)
        yield
        # Reached after the server has stopped taking requests and the ones in
        # flight have finished, so connections are closed cleanly, not cut.
        await app.state.queue.aclose()
        await engine.dispose()
        app.state.inference.close()

    app = FastAPI(
        title="Provenance API",
        version=__version__,
        summary="Classifies text as human-written or AI-generated.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.billing = gateway_from_settings(settings)
    install_error_handling(app)
    install_observability(app)
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(api_keys.router)
    app.include_router(classify.router)
    app.include_router(jobs.router)
    app.include_router(usage.router)
    app.include_router(billing.router)
    app.include_router(admin.router)
    return app
