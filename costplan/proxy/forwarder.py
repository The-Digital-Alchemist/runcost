"""Async HTTP forwarder for proxying LLM API requests.

Uses httpx.AsyncClient for non-blocking request forwarding with streaming support.
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Default upstream API URLs
DEFAULT_OPENAI_URL = "https://api.openai.com"
DEFAULT_ANTHROPIC_URL = "https://api.anthropic.com"

# Headers to strip when forwarding (set by proxy itself)
STRIP_HEADERS = {"host", "content-length", "transfer-encoding"}


class Forwarder:
    """Async HTTP forwarder that proxies requests to upstream LLM APIs."""

    def __init__(
        self,
        openai_target: str = DEFAULT_OPENAI_URL,
        anthropic_target: str = DEFAULT_ANTHROPIC_URL,
        timeout: float = 120.0,
    ):
        """
        Args:
            openai_target: Base URL for OpenAI API.
            anthropic_target: Base URL for Anthropic API.
            timeout: Request timeout in seconds.
        """
        self._openai_target = openai_target.rstrip("/")
        self._anthropic_target = anthropic_target.rstrip("/")
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-init the httpx async client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout, connect=10.0),
                follow_redirects=True,
            )
        return self._client

    def _build_headers(self, original_headers: dict, extra_headers: Optional[dict] = None) -> dict:
        """Build forwarded headers, stripping proxy-specific ones."""
        headers = {
            k: v for k, v in original_headers.items()
            if k.lower() not in STRIP_HEADERS
        }
        if extra_headers:
            headers.update(extra_headers)
        return headers

    async def forward_openai(
        self,
        path: str,
        body: bytes,
        headers: dict,
        stream: bool = False,
    ) -> httpx.Response:
        """Forward a request to the OpenAI API.

        Args:
            path: API path (e.g. "/v1/chat/completions").
            body: Raw request body bytes.
            headers: Original request headers.
            stream: If True, return a streaming response.

        Returns:
            httpx.Response (streaming if stream=True).
        """
        client = await self._get_client()
        url = f"{self._openai_target}{path}"
        fwd_headers = self._build_headers(headers)

        if stream:
            req = client.build_request(
                "POST", url, content=body, headers=fwd_headers,
            )
            return await client.send(req, stream=True)
        else:
            return await client.post(url, content=body, headers=fwd_headers)

    async def forward_anthropic(
        self,
        path: str,
        body: bytes,
        headers: dict,
        stream: bool = False,
    ) -> httpx.Response:
        """Forward a request to the Anthropic API.

        Args:
            path: API path (e.g. "/v1/messages").
            body: Raw request body bytes.
            headers: Original request headers.
            stream: If True, return a streaming response.

        Returns:
            httpx.Response (streaming if stream=True).
        """
        client = await self._get_client()
        url = f"{self._anthropic_target}{path}"
        fwd_headers = self._build_headers(headers)

        if stream:
            req = client.build_request(
                "POST", url, content=body, headers=fwd_headers,
            )
            return await client.send(req, stream=True)
        else:
            return await client.post(url, content=body, headers=fwd_headers)

    async def close(self) -> None:
        """Close the underlying httpx client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
