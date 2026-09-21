from typing import Literal

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=50, ge=1, le=500)

    strategy: Literal["greedy", "temperature", "top-k"] = "greedy"

    temperature: float = Field(default=1.0, gt=0)
    top_k: int = Field(default=50, ge=1)


class GenerateResponse(BaseModel):
    generated_text: str


class HealthResponse(BaseModel):
    status: str
    device: str
