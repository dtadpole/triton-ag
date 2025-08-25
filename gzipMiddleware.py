# gunzip_middleware.py
import gzip
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Scope, Receive, Send

class GunzipRequestMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int = 100 * 1024 * 1024):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        enc = None
        for k, v in scope["headers"]:
            if k.lower() == b"content-encoding":
                enc = v.decode().lower()
                break
        if enc != "gzip":
            return await self.app(scope, receive, send)

        # Read full compressed body (simple & reliable)
        body = b""
        more = True
        while more:
            event = await receive()
            if event["type"] != "http.request":
                return await self.app(scope, receive, send)
            body += event.get("body", b"")
            if len(body) > self.max_bytes:
                return await PlainTextResponse("Payload too large", status_code=413)(scope, receive, send)
            more = event.get("more_body", False)

        try:
            body = gzip.decompress(body)
        except Exception:
            return await PlainTextResponse("Bad compressed payload", status_code=400)(scope, receive, send)

        # Strip encoding/length and set new Content-Length
        new_headers = [(k, v) for (k, v) in scope["headers"]
                       if k.lower() not in (b"content-encoding", b"content-length")]
        new_headers.append((b"content-length", str(len(body)).encode()))
        scope = {**scope, "headers": new_headers}

        sent = False
        async def new_receive():
            nonlocal sent, body
            if sent:
                return {"type": "http.request", "body": b"", "more_body": False}
            sent = True
            b = body; body = b""
            return {"type": "http.request", "body": b, "more_body": False}

        return await self.app(scope, new_receive, send)
