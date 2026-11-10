from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from provenance_api.config import Settings
from provenance_api.ml.model import Classifier


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_classifier(request: Request) -> Classifier:
    classifier: Classifier = request.app.state.classifier
    return classifier


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    async with request.app.state.session_factory() as session:
        yield session


SettingsDep = Annotated[Settings, Depends(get_settings)]
ClassifierDep = Annotated[Classifier, Depends(get_classifier)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]
