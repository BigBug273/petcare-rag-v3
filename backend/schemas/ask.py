"""Request and response models for the ask API."""

from typing import Any

from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]] = []
