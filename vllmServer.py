import argparse
import asyncio
from types import SimpleNamespace
import uvicorn
from fastapi import FastAPI, Body
import httpx
import yaml

# vLLM's OpenAI server internals
from vllm.entrypoints.openai.api_server import (
    build_app,
    build_async_engine_client,
    init_app_state,
)

# ---- load configuration from vllm.yaml ----
def load_config():
    with open("vllm.yaml", "r") as f:
        config = yaml.safe_load(f)
    return SimpleNamespace(**config)

args = load_config()

# ---- assemble the FastAPI app that serves /v1/* ----
async def make_app() -> FastAPI:
    # create vLLM engine client and app, wire state (what `run_server` does)
    async with build_async_engine_client(args) as engine_client:
        app = build_app(args)
        vllm_config = await engine_client.get_vllm_config()
        await init_app_state(engine_client, vllm_config, app.state, args)

        # ---- ADD YOUR OWN ENDPOINTS HERE ----

        @app.get("/healthz")
        async def healthz():
            return {
                "ok": True,
                "model": vllm_config.model,
                "vllm_version": vllm_config.vllm_version,
            }

        @app.post("/my/summarize")
        async def summarize(payload: dict = Body(...)):
            """
            A demo endpoint that *internally* calls vLLM's OpenAI /v1/chat/completions
            using an ASGI loopback (no outbound HTTP).
            """
            user_text = payload.get("text", "")
            prompt = f"Summarize in 1 sentence:\n\n{user_text}"

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://local-asgi") as client:
                r = await client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {args.api_key}"},
                    json={
                        "model": args.served_model_name or args.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,
                    },
                )
            r.raise_for_status()
            data = r.json()
            return {"summary": data["choices"][0]["message"]["content"]}

        return app

async def main():
    app = await make_app()
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
