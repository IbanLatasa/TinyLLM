from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import torch
from fastapi import FastAPI, Request

from tinyllm.inference.generate import (
    generate_greedy,
    generate_with_temperature,
    generate_with_top_k,
)
from tinyllm.tokenizer.bpe import TinyLLMTokenizer
from tinyllm.training.checkpoint import load_model_checkpoint

from .schemas import GenerateRequest, GenerateResponse, HealthResponse

CHECKPOINT_PATH = Path(
    os.getenv("TINYLLM_CHECKPOINT_PATH", "artifacts/checkpoints/step_500000.pt")
)

TOKENIZER_PATH = Path(
    os.getenv("TINYLLM_TOKENIZER_PATH", "artifacts/tokenizer/tokenizer.json")
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = TinyLLMTokenizer.from_file(TOKENIZER_PATH)

    model, checkpoint = load_model_checkpoint(CHECKPOINT_PATH, device)

    model.eval()

    app.state.device = device
    app.state.tokenizer = tokenizer
    app.state.model = model
    app.state.model_config = checkpoint["model_config"]

    yield


app = FastAPI(
    title="TinyLLM API",
    description="HTTP API for TinyLLM text generation",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return HealthResponse(status="ok", device=str(request.app.state.device))


@app.post("/generate", response_model=GenerateResponse)
def generate(request: Request, generation_request: GenerateRequest) -> GenerateResponse:
    model = request.app.state.model
    tokenizer = request.app.state.tokenizer
    device = request.app.state.device
    model_config = request.app.state.model_config

    token_ids = tokenizer.encode(generation_request.prompt)

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    eos_token_id = tokenizer.token_to_id("<eos>")

    if generation_request.strategy == "greedy":
        output_ids = generate_greedy(
            model=model,
            input_ids=input_ids,
            max_new_tokens=generation_request.max_new_tokens,
            context_length=model_config["context_length"],
            eos_token_id=eos_token_id,
        )

    elif generation_request.strategy == "temperature":
        output_ids = generate_with_temperature(
            model=model,
            input_ids=input_ids,
            max_new_tokens=generation_request.max_new_tokens,
            context_length=model_config["context_length"],
            temperature=generation_request.temperature,
            eos_token_id=eos_token_id,
        )

    else:
        output_ids = generate_with_top_k(
            model=model,
            input_ids=input_ids,
            max_new_tokens=generation_request.max_new_tokens,
            context_length=model_config["context_length"],
            top_k=generation_request.top_k,
            temperature=generation_request.temperature,
            eos_token_id=eos_token_id,
        )

    generated_text = tokenizer.decode(output_ids[0].tolist())

    return GenerateResponse(generated_text=generated_text)
