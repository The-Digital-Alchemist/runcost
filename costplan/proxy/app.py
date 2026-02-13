"""FastAPI proxy application with budget enforcement.

Endpoints:
    POST /v1/chat/completions  -- OpenAI-compatible proxy
    POST /v1/messages          -- Anthropic-compatible proxy (Claude Code)
    GET  /v1/budget            -- Query remaining budget
    POST /v1/budget/reset      -- Reset session budget
    GET  /health               -- Health check
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from costplan.core.pricing import PricingRegistry, PricingNotFoundError
from costplan.proxy.budget_state import ProxyBudgetState, ProxyBudgetExceeded
from costplan.proxy.forwarder import Forwarder
from costplan.proxy.stream import SSEParser, parse_openai_sse_usage
from costplan.proxy.cost import (
    compute_anthropic_cost,
    compute_openai_cost,
    heuristic_input_cost_estimate,
)

logger = logging.getLogger(__name__)


def _extract_text_length(messages: list) -> int:
    """Extract total text length from messages for heuristic estimation."""
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total += len(block.get("text", ""))
    return total


def create_app(
    budget: ProxyBudgetState,
    forwarder: Forwarder,
    openai_pricing: Optional[PricingRegistry] = None,
    anthropic_pricing: Optional[PricingRegistry] = None,
) -> FastAPI:
    """Create the FastAPI proxy application.

    Args:
        budget: Budget state for enforcement.
        forwarder: HTTP forwarder for upstream APIs.
        openai_pricing: Pricing registry for OpenAI models.
        anthropic_pricing: Pricing registry for Anthropic models.

    Returns:
        Configured FastAPI application.
    """
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await forwarder.close()

    app = FastAPI(
        title="CostPlan Proxy",
        description="LLM Economic Circuit Breaker — transparent budget enforcement proxy",
        version="0.1.0",
        lifespan=lifespan,
    )

    _openai_pricing = openai_pricing or PricingRegistry(provider_name="openai")
    _anthropic_pricing = anthropic_pricing or PricingRegistry(provider_name="anthropic")

    # --- Health ---

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "costplan-proxy"}

    # --- Budget ---

    @app.get("/v1/budget")
    async def get_budget():
        return await budget.stats()

    @app.post("/v1/budget/reset")
    async def reset_budget():
        await budget.reset()
        return {"status": "reset", "remaining": await budget.remaining()}

    # --- OpenAI-compatible proxy ---

    @app.post("/v1/chat/completions")
    async def proxy_openai_chat(request: Request):
        raw_body = await request.body()
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}},
            )

        model = body.get("model", "")
        messages = body.get("messages", [])
        is_stream = body.get("stream", False)

        # Pre-check: estimate cost
        try:
            text_len = _extract_text_length(messages)
            estimated_cost = heuristic_input_cost_estimate(text_len, model, _openai_pricing)
        except PricingNotFoundError:
            estimated_cost = 0.0  # Unknown model: skip pre-check, track actual

        try:
            await budget.pre_check(estimated_cost)
        except ProxyBudgetExceeded as e:
            remaining = e.remaining
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": str(e),
                        "type": "budget_exceeded",
                        "remaining_budget": remaining,
                    }
                },
                headers={
                    "X-CostPlan-Budget-Remaining": str(remaining),
                },
            )

        # Forward
        headers = dict(request.headers)

        if is_stream:
            # For streaming, inject stream_options to get usage in final chunk
            body.setdefault("stream_options", {})["include_usage"] = True
            raw_body = json.dumps(body).encode("utf-8")

            upstream = await forwarder.forward_openai(
                "/v1/chat/completions", raw_body, headers, stream=True,
            )

            usage_acc: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

            async def stream_with_tracking():
                try:
                    async for chunk in upstream.aiter_bytes():
                        parse_openai_sse_usage(chunk, usage_acc)
                        yield chunk
                finally:
                    await upstream.aclose()
                    # Record cost after stream
                    try:
                        cost = compute_openai_cost(
                            model,
                            usage_acc["prompt_tokens"],
                            usage_acc["completion_tokens"],
                            pricing=_openai_pricing,
                        )
                        await budget.record_actual(
                            cost,
                            model=model,
                            input_tokens=usage_acc["prompt_tokens"],
                            output_tokens=usage_acc["completion_tokens"],
                        )
                    except Exception:
                        logger.exception("Failed to record OpenAI streaming cost")

            resp_headers = {
                "content-type": "text/event-stream",
                "cache-control": "no-cache",
                "x-costplan-budget-remaining": str(await budget.remaining()),
            }
            return StreamingResponse(
                stream_with_tracking(),
                media_type="text/event-stream",
                headers=resp_headers,
            )

        else:
            upstream = await forwarder.forward_openai(
                "/v1/chat/completions", raw_body, headers, stream=False,
            )

            resp_body = upstream.content
            try:
                resp_json = json.loads(resp_body)
                usage = resp_json.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                cost = compute_openai_cost(model, prompt_tokens, completion_tokens, pricing=_openai_pricing)
                await budget.record_actual(
                    cost,
                    model=model,
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                )
            except Exception:
                logger.exception("Failed to record OpenAI cost")

            remaining = await budget.remaining()
            resp_headers = {
                k: v for k, v in upstream.headers.items()
                if k.lower() not in {"transfer-encoding", "content-encoding", "content-length"}
            }
            resp_headers["X-CostPlan-Budget-Remaining"] = str(remaining)

            return Response(
                content=resp_body,
                status_code=upstream.status_code,
                headers=resp_headers,
            )

    # --- Anthropic-compatible proxy (Claude Code) ---

    @app.post("/v1/messages")
    async def proxy_anthropic_messages(request: Request):
        raw_body = await request.body()
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}},
            )

        model = body.get("model", "")
        messages = body.get("messages", [])
        is_stream = body.get("stream", False)

        # Pre-check: estimate cost
        try:
            text_len = _extract_text_length(messages)
            estimated_cost = heuristic_input_cost_estimate(text_len, model, _anthropic_pricing)
        except PricingNotFoundError:
            estimated_cost = 0.0

        try:
            await budget.pre_check(estimated_cost)
        except ProxyBudgetExceeded as e:
            remaining = e.remaining
            # Return Anthropic-style error format
            return JSONResponse(
                status_code=429,
                content={
                    "type": "error",
                    "error": {
                        "type": "budget_exceeded",
                        "message": str(e),
                    },
                    "remaining_budget": remaining,
                },
                headers={
                    "X-CostPlan-Budget-Remaining": str(remaining),
                },
            )

        headers = dict(request.headers)

        if is_stream:
            upstream = await forwarder.forward_anthropic(
                "/v1/messages", raw_body, headers, stream=True,
            )

            sse_parser = SSEParser()

            async def stream_with_tracking():
                try:
                    async for chunk in upstream.aiter_bytes():
                        sse_parser.feed(chunk)
                        yield chunk
                finally:
                    await upstream.aclose()
                    # Finalize and record cost
                    usage = sse_parser.finalize()
                    try:
                        cost = compute_anthropic_cost(
                            model=model,
                            input_tokens=usage.input_tokens,
                            output_tokens=usage.output_tokens,
                            cache_read_tokens=usage.cache_read_input_tokens,
                            cache_creation_tokens=usage.cache_creation_input_tokens,
                            pricing=_anthropic_pricing,
                        )
                        await budget.record_actual(
                            cost,
                            model=model,
                            input_tokens=usage.input_tokens,
                            output_tokens=usage.output_tokens,
                            cache_read_tokens=usage.cache_read_input_tokens,
                            cache_creation_tokens=usage.cache_creation_input_tokens,
                        )
                        logger.info(
                            "Anthropic stream: model=%s input=%d output=%d "
                            "cache_read=%d cache_create=%d cost=$%.6f",
                            model, usage.input_tokens, usage.output_tokens,
                            usage.cache_read_input_tokens, usage.cache_creation_input_tokens,
                            cost,
                        )
                    except Exception:
                        logger.exception("Failed to record Anthropic streaming cost")

            resp_headers = {
                "content-type": "text/event-stream",
                "cache-control": "no-cache",
                "x-costplan-budget-remaining": str(await budget.remaining()),
            }
            return StreamingResponse(
                stream_with_tracking(),
                media_type="text/event-stream",
                headers=resp_headers,
            )

        else:
            upstream = await forwarder.forward_anthropic(
                "/v1/messages", raw_body, headers, stream=False,
            )

            resp_body = upstream.content
            try:
                resp_json = json.loads(resp_body)
                usage = resp_json.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                cache_read = usage.get("cache_read_input_tokens", 0)
                cache_create = usage.get("cache_creation_input_tokens", 0)
                cost = compute_anthropic_cost(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read,
                    cache_creation_tokens=cache_create,
                    pricing=_anthropic_pricing,
                )
                await budget.record_actual(
                    cost,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read,
                    cache_creation_tokens=cache_create,
                )
            except Exception:
                logger.exception("Failed to record Anthropic cost")

            remaining = await budget.remaining()
            resp_headers = {
                k: v for k, v in upstream.headers.items()
                if k.lower() not in {"transfer-encoding", "content-encoding", "content-length"}
            }
            resp_headers["X-CostPlan-Budget-Remaining"] = str(remaining)

            return Response(
                content=resp_body,
                status_code=upstream.status_code,
                headers=resp_headers,
            )

    return app
