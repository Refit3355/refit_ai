from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"

# 앱 시작 시 1회 로드하도록 외부에서 인스턴스 생성
_model = None

def init_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def encode_queries(texts: List[str]) -> np.ndarray:
    m = init_model()
    texts = [f"query: {t}" for t in texts]
    v = m.encode(texts, normalize_embeddings=True, batch_size=64)
    return v.astype("float32")

def encode_passages(texts: List[str]) -> np.ndarray:
    m = init_model()
    texts = [f"passage: {t}" for t in texts]
    v = m.encode(texts, normalize_embeddings=True, batch_size=64)
    return v.astype("float32")
