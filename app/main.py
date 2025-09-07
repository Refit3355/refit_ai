from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging

from app.routers import recommend
from app.core.embedding import init_model
from app.core.scheduler import (
    start_scheduler,
    refresh_embeddings,
    refresh_embeddings_beauty,
    refresh_embeddings_hair,
    refresh_embeddings_health,
    refresh_weather,
)
from app.core import state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup : 모델 미리 로드
    try:
        init_model()
        logger.info("[startup] model init success")
        start_scheduler()
        logger.info("[startup] scheduler started")
    except Exception as e:
        logger.exception("[startup] model init failed")
    yield

app = FastAPI(
    title="Refit AI Recommendation APIs",
    lifespan=lifespan,
)

# 라우터
app.include_router(recommend.router)

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/debug/cache-status")
def cache_status():
    return {
        "global_index": state.global_index is not None,
        "global_index_beauty": state.global_index_beauty is not None,
        "global_index_hair": state.global_index_hair is not None,
        "global_index_health": state.global_index_health is not None,
        "global_weather_ctx": state.global_weather_ctx is not None,
        "global_frames_all": state.global_frames_all is not None,
        "global_frames_beauty": state.global_frames_beauty is not None,
        "global_frames_hair": state.global_frames_hair is not None,
        "global_frames_health": state.global_frames_health is not None,
    }

@app.post("/debug/force-refresh")
def force_refresh():
    try:
        refresh_embeddings()
        refresh_embeddings_beauty()
        refresh_embeddings_hair()
        refresh_embeddings_health()
        refresh_weather()
        return {"ok": True, "msg": "모든 캐시 강제 갱신 완료"}
    except Exception as e:
        return {"ok": False, "error": str(e)}