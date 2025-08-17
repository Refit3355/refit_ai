from fastapi import APIRouter, HTTPException
from app.models.schemas import RecommendAiRequest, RecommendAiResponse

from app.core.reco_common import load_frames, simple_recommend
from app.core.reco_beauty import load_frames_beauty, recommend_beauty
from app.core.reco_all import load_frames_all, recommend_all
from app.core.reco_hair import load_frames_hair, recommend_hair
from app.core.reco_health import load_frames_health, recommend_health
from app.core.weather import fetch_weather_ctx

router = APIRouter(prefix="/products/recommendation", tags=["recommend"])

def _fallback_basic(req: RecommendAiRequest, weather, label: str):
    """기본 가벼운 추천으로 재시도"""
    frames = load_frames()
    q = f"member:{req.memberId} category:{req.preferCategoryId if req.preferCategoryId is not None else 'all'} {label}"
    df = simple_recommend(
        q, frames,
        topk=req.topk or 200,
        final=req.final or 10,
        product_type=req.productType,
        prefer_category_id=req.preferCategoryId
    )
    results = df.to_dict(orient="records") if not df.empty else []
    return RecommendAiResponse(weather=weather, query=q, targetEffects=[], results=results)

@router.post("/ai", response_model=RecommendAiResponse)
def recommend_ai(req: RecommendAiRequest):
    try:
        weather = fetch_weather_ctx(req.location or "서울")

        # 0) 전체
        if req.productType == 0:
            try:
                frames = load_frames_all()
                df = recommend_all(
                    member_id=req.memberId,
                    frames=frames,
                    prefer_category_id=req.preferCategoryId,
                    topk=req.topk or 200,
                    final=req.final or 10,
                    weather_ctx=weather
                )
                if df.empty:
                    return _fallback_basic(req, weather, "type:0(empty->basic)")
                results = df.to_dict(orient="records")
                q = f"member:{req.memberId} category:{req.preferCategoryId if req.preferCategoryId is not None else 'all'} type:0"
                return RecommendAiResponse(weather=weather, query=q, targetEffects=[], results=results)
            except Exception:
                return _fallback_basic(req, weather, "type:0(error->basic)")

        # 1) 뷰티
        if req.productType == 1:
            try:
                frames = load_frames_beauty()
                df = recommend_beauty(
                    member_id=req.memberId,
                    frames=frames,
                    prefer_category_id=req.preferCategoryId,
                    topk=req.topk or 200,
                    final=req.final or 10,
                    weather_ctx=weather
                )
                if df.empty:
                    return _fallback_basic(req, weather, "type:1(empty->basic)")
                results = df.to_dict(orient="records")
                q = f"member:{req.memberId} category:{req.preferCategoryId if req.preferCategoryId is not None else 'beauty(0-5)'} type:1"
                return RecommendAiResponse(weather=weather, query=q, targetEffects=[], results=results)
            except Exception:
                return _fallback_basic(req, weather, "type:1(error->basic)")

        # 2) 헤어
        if req.productType == 2:
            try:
                frames = load_frames_hair()
                df = recommend_hair(
                    member_id=req.memberId,
                    frames=frames,
                    prefer_category_id=req.preferCategoryId,
                    topk=req.topk or 200,
                    final=req.final or 10,
                    weather_ctx=weather
                )
                if df.empty:
                    return _fallback_basic(req, weather, "type:2(empty->basic)")
                results = df.to_dict(orient="records")
                q = f"member:{req.memberId} category:{req.preferCategoryId if req.preferCategoryId is not None else 'hair(6-7)'} type:2"
                return RecommendAiResponse(weather=weather, query=q, targetEffects=[], results=results)
            except Exception:
                return _fallback_basic(req, weather, "type:2(error->basic)")

        # 3) 건강기능식품
        if req.productType == 3:
            try:
                frames = load_frames_health()
                df = recommend_health(
                    member_id=req.memberId,
                    frames=frames,
                    prefer_category_id=req.preferCategoryId,
                    topk=req.topk or 200,
                    final=req.final or 10,
                    weather_ctx=weather
                )
                if df.empty:
                    return _fallback_basic(req, weather, "type:3(empty->basic)")
                results = df.to_dict(orient="records")
                q = f"member:{req.memberId} category:{req.preferCategoryId if req.preferCategoryId is not None else 'health(8-11)'} type:3"
                return RecommendAiResponse(weather=weather, query=q, targetEffects=[], results=results)
            except Exception:
                return _fallback_basic(req, weather, "type:3(error->basic)")

        # 그 외 : 기본 추천
        return _fallback_basic(req, weather, f"type:{req.productType}(default->basic)")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
