from __future__ import annotations

import json
import time
from typing import Any

import httpx
from pydantic import BaseModel, Field


class RequestRecord(BaseModel):
    method: str
    url: str
    request_headers: dict[str, str]
    request_body: str | None
    status_code: int
    response_headers: dict[str, str]
    response_body: str
    elapsed_ms: float

    def format_log(self) -> str:
        lines = [
            f">>> {self.method} {self.url}",
            "",
            *[f">>> {k}: {v}" for k, v in self.request_headers.items()],
            "",
        ]

        if self.request_body:
            try:
                parsed: object = json.loads(self.request_body)
                lines.append(json.dumps(parsed, indent=2, ensure_ascii=False))
            except (json.JSONDecodeError, TypeError):
                lines.append(self.request_body)

        lines.extend([
            "",
            f"<<< {self.status_code}",
            "",
            *[f"<<< {k}: {v}" for k, v in self.response_headers.items()],
            "",
        ])

        try:
            parsed_resp: object = json.loads(self.response_body)
            lines.append(json.dumps(parsed_resp, indent=2, ensure_ascii=False))
        except (json.JSONDecodeError, TypeError):
            lines.append(self.response_body)

        lines.extend([
            "",
            f"--- Elapsed: {self.elapsed_ms:.1f}ms",
        ])
        return "\n".join(lines)


JsonDict = dict[str, Any]


class LoggingHttpClient(BaseModel):
    base_url: str
    default_headers: dict[str, str] = Field(default_factory=dict)
    records: list[RequestRecord] = Field(default_factory=lambda: list[RequestRecord]())
    timeout: float = 60.0
    verify_ssl: bool = True

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: JsonDict | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, JsonDict | str]:
        """Send a request and return (status_code, response_body).

        Response body is parsed as JSON if possible, otherwise returned as string.
        """
        url = f"{self.base_url}{path}"
        merged_headers = {**self.default_headers, **(headers or {})}

        request_body_str: str | None = None
        if json_body is not None:
            request_body_str = json.dumps(json_body, ensure_ascii=False)

        start = time.monotonic()
        with httpx.Client(timeout=self.timeout, verify=self.verify_ssl) as client:
            resp = client.request(
                method,
                url,
                headers=merged_headers,
                content=request_body_str,
            )
        elapsed = (time.monotonic() - start) * 1000

        response_body = resp.text
        record = RequestRecord(
            method=method,
            url=url,
            request_headers=merged_headers,
            request_body=request_body_str,
            status_code=resp.status_code,
            response_headers=dict(resp.headers),
            response_body=response_body,
            elapsed_ms=elapsed,
        )
        self.records.append(record)

        try:
            parsed: JsonDict = resp.json()
            return resp.status_code, parsed
        except (json.JSONDecodeError, ValueError):
            return resp.status_code, response_body

    def request_stream(
        self,
        method: str,
        path: str,
        *,
        json_body: JsonDict | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, list[str]]:
        """Send a streaming request and return (status_code, list_of_sse_lines).

        Collects all SSE lines for both validation and logging.
        """
        url = f"{self.base_url}{path}"
        merged_headers = {**self.default_headers, **(headers or {})}

        request_body_str: str | None = None
        if json_body is not None:
            request_body_str = json.dumps(json_body, ensure_ascii=False)

        sse_lines: list[str] = []

        start = time.monotonic()
        with (
            httpx.Client(timeout=self.timeout, verify=self.verify_ssl) as client,
            client.stream(
                method,
                url,
                headers=merged_headers,
                content=request_body_str,
            ) as resp,
        ):
            status_code = resp.status_code
            resp_headers = dict(resp.headers)
            for line in resp.iter_lines():
                sse_lines.append(line)
        elapsed = (time.monotonic() - start) * 1000

        record = RequestRecord(
            method=method,
            url=url,
            request_headers=merged_headers,
            request_body=request_body_str,
            status_code=status_code,
            response_headers=resp_headers,
            response_body="\n".join(sse_lines),
            elapsed_ms=elapsed,
        )
        self.records.append(record)

        return status_code, sse_lines
