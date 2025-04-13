"""
server.py

Serve inference from LLM, with SSE, Websockets, streaming, and sessions.
"""

import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import socketio

from fabricai_inference_server.engine import load_default_engine
from fabricai_inference_server.redis_session_manager import RedisSessionManager
from fabricai_inference_server.exceptions import (
    QueueFullException,
    ModelNotFoundException,
)
from fabricai_inference_server.models import InferenceRequest
from fabricai_inference_server.settings import settings

fastapi_app = FastAPI()
sio = socketio.AsyncServer(cors_allowed_origins="*")
socketio_app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)

session_manager = RedisSessionManager()
_engine = None

# Concurrency
MAX_CONCURRENT_REQUESTS = settings.max_concurrent_requests
QUEUE_MAX_SIZE = settings.queue_max_size
sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
request_queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)


def get_engine():
    global _engine
    if _engine is None:
        _engine = load_default_engine()
    return _engine


@fastapi_app.get("/v1/status")
async def status():
    current_queue_size = request_queue.qsize()
    active_slots = MAX_CONCURRENT_REQUESTS - sem._value
    return {
        "queue_size": current_queue_size,
        "queue_max_size": QUEUE_MAX_SIZE,
        "active_requests": active_slots,
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
    }


async def join_queue_or_fail():
    if request_queue.full():
        raise HTTPException(429, detail="Request queue is full.")
    await request_queue.put(None)


async def leave_queue():
    request_queue.get_nowait()
    request_queue.task_done()


@fastapi_app.post("/v1/inference_sse")
async def inference_sse(inference_req: InferenceRequest):
    engine = get_engine()
    session_id = await session_manager.get_or_create_session(inference_req.session_id)
    await session_manager.add_message(session_id, "user", inference_req.prompt)
    full_prompt = await session_manager.build_prompt(session_id)

    if request_queue.full():
        raise QueueFullException()
    await request_queue.put(None)

    async def sse_generator():
        try:
            async with sem:
                await leave_queue()
                assistant_reply = []

                for token in engine.generate_stream(
                    prompt=full_prompt,
                    max_tokens=inference_req.max_tokens,
                    temperature=inference_req.temperature,
                    top_p=inference_req.top_p,
                    repeat_penalty=inference_req.repeat_penalty,
                    stop=inference_req.stop,
                ):
                    assistant_reply.append(token)
                    yield f"data: {token}\n\n"

                final_reply = "".join(assistant_reply).strip()
                await session_manager.add_message(session_id, "assistant", final_reply)

        except FileNotFoundError as e:
            raise ModelNotFoundException(str(e))
        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


@sio.event
async def connect(sid, environ, auth):
    # TODO: auth JWT
    print(f"Socket.IO: Client connected with sid={sid}")


@sio.event
async def disconnect(sid):
    print(f"Socket.IO: Client disconnected with sid={sid}")


@sio.on("inference_prompt")
async def handle_inference_prompt(sid, data):
    """
    data is a dict:
    {
      "session_id": optional,
      "prompt": "User message",
      "max_tokens": int,
      "temperature": float,
      ...
    }
    """
    try:
        inference_req = InferenceRequest(**data)
    except Exception as e:
        await sio.emit("inference_error", {"error": str(e)}, to=sid)
        return

    if request_queue.full():
        await sio.emit("inference_error", {"error": "Queue is full (429)."}, to=sid)
        return
    else:
        await request_queue.put(None)

    async with sem:
        await leave_queue()
        session_id = await session_manager.get_or_create_session(inference_req.session_id)
        await session_manager.add_message(session_id, "user", inference_req.prompt)
        full_prompt = await session_manager.build_prompt(session_id)
        engine = get_engine()
        assistant_reply = []

        try:
            for token in engine.generate_stream(
                prompt=full_prompt,
                max_tokens=inference_req.max_tokens,
                temperature=inference_req.temperature,
                top_p=inference_req.top_p,
                repeat_penalty=inference_req.repeat_penalty,
                stop=inference_req.stop,
            ):
                assistant_reply.append(token)
                await sio.emit("inference_token", {"token": token}, to=sid)
            final_reply = "".join(assistant_reply).strip()
            await session_manager.add_message(session_id, "assistant", final_reply)
            await sio.emit(
                "inference_complete",
                {"message": final_reply, "session_id": session_id},
                to=sid,
            )
        except FileNotFoundError as fnf_err:
            await sio.emit("inference_error", {"error": str(fnf_err)}, to=sid)
        except Exception as e:
            await sio.emit("inference_error", {"error": str(e)}, to=sid)


@fastapi_app.exception_handler(QueueFullException)
async def queue_full_exception_handler(request, exc: QueueFullException):
    return JSONResponse(status_code=429, content={"error": exc.message})


@fastapi_app.exception_handler(ModelNotFoundException)
async def model_not_found_exception_handler(request, exc: ModelNotFoundException):
    return JSONResponse(status_code=404, content={"error": exc.message})
