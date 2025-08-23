from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging

from app.routers import recommend
from app.core.embedding import init_model
from app.core.scheduler import start_scheduler  

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
