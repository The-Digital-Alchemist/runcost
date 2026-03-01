"""FastAPI proxy application with budget enforcement.

Endpoints:
    GET  /                     -- Dashboard (budget status page)
    GET  /health               -- Health check
    GET  /v1/budget            -- Query remaining budget
    POST /v1/budget/reset       -- Reset session budget
    POST /v1/chat/completions  -- OpenAI-compatible proxy
    POST /v1/messages          -- Anthropic-compatible proxy (Claude Code)
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

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

    # --- Dashboard (minimal status page) ---

    DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CostPlan — Budget status</title>
  <style>
    :root {
      font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
      line-height: 1.6;
      --bg: #f1f5f9;
      --card: #fff;
      --text: #0f172a;
      --text-muted: #64748b;
      --border: #e2e8f0;
      --remaining: #0ea5e9;
      --ok: #22c55e;
      --locked: #ef4444;
      --btn-bg: #0f172a;
      --btn-hover: #1e293b;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #0f172a;
        --card: #1e293b;
        --text: #f1f5f9;
        --text-muted: #94a3b8;
        --border: #334155;
        --btn-bg: #38bdf8;
        --btn-hover: #7dd3fc;
      }
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; background: var(--bg); color: var(--text); padding: 2rem 1rem; }
    .wrap { max-width: 22rem; margin: 0 auto; }
    .header { margin-bottom: 1.5rem; }
    .header h1 { font-size: 1.125rem; font-weight: 600; letter-spacing: -0.02em; margin: 0; color: var(--text); }
    .header p { font-size: 0.8125rem; color: var(--text-muted); margin: 0.25rem 0 0; }
    .reset-info { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.25rem; }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.5rem;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    @media (prefers-color-scheme: dark) {
      .card { box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
    }
    .progress-wrap { margin-bottom: 1.25rem; }
    .progress-bar {
      height: 8px;
      background: var(--border);
      border-radius: 999px;
      overflow: hidden;
    }
    .progress-fill {
      height: 100%;
      border-radius: 999px;
      background: var(--remaining);
      transition: width 0.3s ease;
    }
    .progress-fill.low { background: var(--locked); }
    .progress-labels { display: flex; justify-content: space-between; font-size: 0.8125rem; margin-top: 0.5rem; color: var(--text-muted); }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem 1.5rem; }
    .stat { min-width: 0; }
    .stat .label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--text-muted); display: block; margin-bottom: 0.125rem; }
    .stat .value { font-size: 1rem; font-weight: 600; }
    .badge {
      display: inline-block;
      font-size: 0.75rem;
      font-weight: 600;
      padding: 0.2rem 0.5rem;
      border-radius: 6px;
    }
    .badge.ok { background: rgba(34, 197, 94, 0.15); color: var(--ok); }
    .badge.locked { background: rgba(239, 68, 68, 0.15); color: var(--locked); }
    @media (prefers-color-scheme: dark) {
      .badge.ok { background: rgba(34, 197, 94, 0.2); }
      .badge.locked { background: rgba(239, 68, 68, 0.2); }
    }
    .actions { margin-top: 1.25rem; padding-top: 1rem; border-top: 1px solid var(--border); }
    button {
      padding: 0.5rem 1rem;
      font-size: 0.875rem;
      font-weight: 500;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      background: var(--btn-bg);
      color: #fff;
      transition: background 0.15s ease;
    }
    @media (prefers-color-scheme: dark) { button { color: #0f172a; } }
    button:hover { background: var(--btn-hover); }
    button:active { transform: scale(0.98); }
    #toast { font-size: 0.8125rem; color: var(--ok); margin-top: 0.5rem; }
  </style>
</head>
<body>
  <div class="wrap">
    <header class="header">
      <h1>CostPlan</h1>
      <p>Budget status — refreshes every 5s</p>
      <p id="reset_info" class="reset-info"></p>
    </header>
    <div class="card">
      <div class="progress-wrap">
        <div class="progress-bar" role="progressbar" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">
          <div id="progress_fill" class="progress-fill" style="width: 100%"></div>
        </div>
        <div class="progress-labels">
          <span id="remaining_label">$0.00 remaining</span>
          <span id="session_budget">—</span>
        </div>
      </div>
      <div class="grid">
        <div class="stat"><span class="label">Spent</span><span id="spent" class="value">—</span></div>
        <div class="stat"><span class="label">Calls</span><span id="call_count" class="value">—</span></div>
        <div class="stat"><span class="label">Circuit</span><span id="locked" class="value">—</span></div>
      </div>
      <div class="actions">
        <button id="reset_btn">Reset session</button>
        <div id="toast"></div>
      </div>
    </div>
  </div>
  <script>
    function fmt(n) { return typeof n === "number" ? "$" + n.toFixed(2) : n; }
    function fmtReset(sec) {
      if (sec >= 86400) return (sec / 86400).toFixed(1) + "d";
      if (sec >= 3600) return (sec / 3600).toFixed(1) + "h";
      if (sec >= 60) return (sec / 60).toFixed(1) + "m";
      return sec + "s";
    }
    function render(data) {
      var pct = data.session_budget > 0 ? (data.remaining / data.session_budget) * 100 : 100;
      var fill = document.getElementById("progress_fill");
      fill.style.width = Math.max(0, Math.min(100, pct)) + "%";
      fill.classList.toggle("low", pct < 15);
      document.getElementById("remaining_label").textContent = fmt(data.remaining) + " remaining";
      document.getElementById("session_budget").textContent = "of " + fmt(data.session_budget);
      document.getElementById("spent").textContent = fmt(data.total_spent);
      document.getElementById("call_count").textContent = data.call_count;
      var el = document.getElementById("locked");
      el.textContent = data.locked ? "Locked" : "OK";
      el.className = "badge " + (data.locked ? "locked" : "ok");
      var ri = document.getElementById("reset_info");
      if (data.budget_window_seconds) {
        ri.textContent = "Rolling window: " + fmtReset(data.budget_window_seconds);
      } else if (data.reset_every_seconds) {
        var next = data.next_reset_at ? new Date(data.next_reset_at * 1000).toLocaleTimeString() : "";
        ri.textContent = "Resets every " + fmtReset(data.reset_every_seconds) + (next ? " • Next: " + next : "");
      } else {
        ri.textContent = "";
      }
    }
    function fetchBudget() {
      fetch("/v1/budget").then(function(r) { return r.json(); }).then(render).catch(function() {
        document.getElementById("remaining_label").textContent = "Error loading";
      });
    }
    document.addEventListener("DOMContentLoaded", function() {
      fetchBudget();
      setInterval(fetchBudget, 5000);
      document.getElementById("reset_btn").addEventListener("click", function() {
        fetch("/v1/budget/reset", { method: "POST" }).then(function(r) { return r.json(); }).then(function() {
          fetchBudget();
          var t = document.getElementById("toast");
          t.textContent = "Session reset.";
          setTimeout(function() { t.textContent = ""; }, 2000);
        });
      });
    });
  </script>
</body>
</html>
"""

    @app.get("/")
    async def dashboard():
        return HTMLResponse(DASHBOARD_HTML)

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

        logger.info("Anthropic request: model=%s stream=%s", model, is_stream)

        # Pre-check: estimate cost
        text_len = _extract_text_length(messages)
        estimated_cost = heuristic_input_cost_estimate(text_len, model, _anthropic_pricing)

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
