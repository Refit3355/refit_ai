from typing import Optional, List, Dict, Literal, Any
from pydantic import BaseModel, Field

# 요청
class RecommendAiRequest(BaseModel):
    memberId: int = Field(..., ge=1)
    productType: Literal[0, 1, 2, 3] = 0   # 0:전체, 1:뷰티, 2:헤어, 3:건강기능식품
    preferCategoryId: Optional[int] = None
    location: Optional[str] = "서울"
    topk: Optional[int] = Field(200, ge=1, le=10000)
    final: Optional[int] = Field(10, ge=1, le=200)

# 응답 아이템(엄격 스키마)
class RecommendItem(BaseModel):
    # 기본 식별/이름
    productId: int
    name: Optional[str] = None

    # 카테고리
    category: Optional[str] = None
    categoryId: Optional[int] = None

    # 스코어
    sim: Optional[float] = None
    effMatch: Optional[float] = None
    finalScore: Optional[float] = None

    # 상품 정보
    price: Optional[int] = None
    brand: Optional[str] = None
    stock: Optional[int] = None
    discountRate: Optional[int] = Field(default=None, ge=0, le=100)  # 예: 10 == 10%
    thumbnailUrl: Optional[str] = None

    # 인기도
    unitsSold: Optional[float] = 0
    ordersSold: Optional[float] = 0

# 응답
class RecommendAiResponse(BaseModel):
    weather: Dict[str, Any]
    query: str
    targetEffects: List[str]
    results: List[RecommendItem]
