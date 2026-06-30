"""HTTP client for the OKX exchange API (retry + rate-limit handling).

Migrated from old src/market_http.py (http_get_json) and the page-fetch part
of src/fetch_candles.py.
"""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

USER_AGENT = "ml-predictor-v2/0.1"
_429_RETRY_DELAYS = (2, 4, 8)   # seconds between retries on HTTP 429


class OKXClient:
    def __init__(self, base_url: str = "https://www.okx.com", timeout: float = 10.0):
        self.base_url = base_url
        self.timeout = timeout

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        query = urlencode(
            {k: v for k, v in (params or {}).items() if v is not None}
        )
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{query}"

        request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})

        last_exc: Exception | None = None
        for attempt, delay in enumerate((0, *_429_RETRY_DELAYS)):
            if delay:
                time.sleep(delay)
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                if exc.code == 429:
                    last_exc = exc
                    continue
                detail = exc.read().decode("utf-8", errors="replace")[:400]
                raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
            except URLError as exc:
                raise RuntimeError(f"Network error while requesting {url}: {exc.reason}") from exc

        raise RuntimeError(
            f"HTTP 429 after {len(_429_RETRY_DELAYS) + 1} attempts: {url}"
        ) from last_exc

    def history_candles(self, symbol: str, bar: str, after: str | None = None,
                        limit: int = 300) -> list[list[Any]]:
        """One page of history candles (paginates backwards via `after`)."""
        payload = self.get_json(
            "/api/v5/market/history-candles",
            {"instId": symbol, "bar": bar, "limit": limit, "after": after},
        )
        return payload.get("data", [])
