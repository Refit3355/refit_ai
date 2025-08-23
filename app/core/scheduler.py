from apscheduler.schedulers.background import BackgroundScheduler
import pytz
from datetime import datetime
import faiss

from app.core.reco_all import load_frames_all, attach_product_text, build_prod_effects
from app.core.embedding import encode_passages
from app.core.weather import fetch_weather_ctx 
from app.core import state


def refresh_weather():
    print("날씨 캐싱 시작:", datetime.now())
    try:
        state.global_weather_ctx = fetch_weather_ctx("서울")
        print("날씨 캐싱 완료:", state.global_weather_ctx)
    except Exception as e:
        print("날씨 캐싱 실패:", e)
        
def refresh_embeddings():
    print("임베딩 갱신 시작:", datetime.now())
    frames = load_frames_all()
    dp = attach_product_text(frames["df_product"], build_prod_effects(frames))
    vecs = encode_passages(dp["PRODUCT_TEXT"].tolist())
    ids = dp["PRODUCT_ID"].to_numpy(dtype=np.int64)

    state.global_index = faiss.IndexIDMap(faiss.IndexFlatIP(vecs.shape[1]))
    state.global_index.add_with_ids(vecs, ids)
    print("임베딩 갱신 완료:", datetime.now())
    
def refresh_embeddings_beauty():
    print("뷰티 임베딩 갱신 시작:", datetime.now())
    from app.core.reco_beauty import load_frames_beauty, attach_product_text, build_prod_effects
    frames = load_frames_beauty()
    dp = attach_product_text(frames["df_product"], build_prod_effects(frames))
    vecs = encode_passages(dp["PRODUCT_TEXT"].tolist())
    ids = dp["PRODUCT_ID"].to_numpy(dtype=np.int64)

    state.global_index_beauty = faiss.IndexIDMap(faiss.IndexFlatIP(vecs.shape[1]))
    state.global_index_beauty.add_with_ids(vecs, ids)
    print("뷰티 임베딩 갱신 완료:", datetime.now())

def refresh_embeddings_hair():
    print("헤어 임베딩 갱신 시작:", datetime.now())
    from app.core.reco_hair import load_frames_hair, attach_product_text, build_prod_effects
    frames = load_frames_hair()
    dp = attach_product_text(frames["df_product"], build_prod_effects(frames))
    vecs = encode_passages(dp["PRODUCT_TEXT"].tolist())
    ids = dp["PRODUCT_ID"].to_numpy(dtype=np.int64)

    state.global_index_hair = faiss.IndexIDMap(faiss.IndexFlatIP(vecs.shape[1]))
    state.global_index_hair.add_with_ids(vecs, ids)
    print("헤어 임베딩 갱신 완료:", datetime.now())
    
def refresh_embeddings_health():
    print("헬스 임베딩 갱신 시작:", datetime.now())
    from app.core.reco_health import load_frames_health, attach_product_text, build_prod_effects
    frames = load_frames_health()
    dp = attach_product_text(frames["df_product"], build_prod_effects(frames))
    vecs = encode_passages(dp["PRODUCT_TEXT"].tolist())
    ids = dp["PRODUCT_ID"].to_numpy(dtype=np.int64)

    state.global_index_health = faiss.IndexIDMap(faiss.IndexFlatIP(vecs.shape[1]))
    state.global_index_health.add_with_ids(vecs, ids)
    print("헬스 임베딩 갱신 완료:", datetime.now())

def start_scheduler():
    scheduler = BackgroundScheduler(timezone=pytz.timezone("Asia/Seoul"))
    scheduler.add_job(refresh_embeddings, "cron", hour=0, minute=0)       # 매일 자정 (전체)
    scheduler.add_job(refresh_embeddings_beauty, "cron", hour=0, minute=5)  # 매일 자정+5분 (뷰티)
    scheduler.add_job(refresh_embeddings_hair, "cron", hour=0, minute=10)  # 매일 자정+10분 (헤어)
    scheduler.add_job(refresh_embeddings_health, "cron", hour=0, minute=15)  # 매일 자정+15분 (헬스)
    
    scheduler.add_job(refresh_weather, "interval", minutes=30) # 30분마다 한 번씩 날씨 캐싱

    scheduler.start()
