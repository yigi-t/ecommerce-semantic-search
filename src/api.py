# -*- coding: utf-8 -*-
"""
FastAPI arama servisi.

    uvicorn src.api:app --reload --port 8000

GET /search?q=puantiyeli+kırmızı+elbise&limit=24
GET /products/{id}/outfit-recommendations?limit=4
GET /products/{id}/outfit-recommendations?limit=1&preference=cheaper&current_product_id={id}
POST /personalized-recommendations
GET /        → mini demo arayüzü
"""

import re
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.responses import FileResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from .outfit import (
    OUTFIT_EDIT_EXCLUDE_LIMIT,
    OUTFIT_PREFERRED_SIZE_LIMIT,
    OutfitEditValidationError,
    OutfitRecommender,
)
from .personalized import (
    PERSONALIZED_EXCLUDE_LIMIT,
    PERSONALIZED_ID_LENGTH_LIMIT,
    PERSONALIZED_RECENT_LIMIT,
    PERSONALIZED_SAVED_OUTFIT_LIMIT,
    PersonalizedRecommender,
)
from .search import SearchEngine


ProductId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=PERSONALIZED_ID_LENGTH_LIMIT,
    ),
]


class SavedOutfitSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_product_id: ProductId
    recommended_product_id: ProductId


class PersonalizedSizePreferences(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alpha: str | None = Field(default=None, max_length=40)
    numeric: str | None = Field(default=None, max_length=40)
    shoe: str | None = Field(default=None, max_length=40)
    jean_waist: str | None = Field(default=None, max_length=40)
    jean_length: str | None = Field(default=None, max_length=40)
    child: str | None = Field(default=None, max_length=40)

    @field_validator("*", mode="before")
    @classmethod
    def _strip_empty_values(cls, value):
        if value is None:
            return None
        clean = str(value).strip()
        return clean or None

    @model_validator(mode="after")
    def _validate_size_formats(self):
        if self.alpha and not re.fullmatch(
            r"(?:XXS|XS|S|M|L|XL|XXL|[3-6]XL)", self.alpha.upper()
        ):
            raise ValueError("Geçersiz harfli giyim bedeni")
        for field_name in ("numeric", "shoe", "jean_waist", "jean_length"):
            value = getattr(self, field_name)
            if value and not re.fullmatch(r"\d{2}", value):
                raise ValueError("Sayısal bedenler iki basamaklı olmalıdır")
        if bool(self.jean_waist) != bool(self.jean_length):
            raise ValueError("Jean bel ve boy bedeni birlikte gönderilmelidir")
        if self.child:
            child_match = re.fullmatch(
                r"(?:age|month):(\d{1,2})(?:-(\d{1,2}))?",
                self.child.lower(),
            )
            if not child_match:
                raise ValueError("Geçersiz çocuk veya bebek bedeni")
            child_start = int(child_match.group(1))
            child_end = int(child_match.group(2) or child_start)
            if child_start > child_end:
                raise ValueError("Çocuk veya bebek beden aralığı geçersiz")
        return self


class PersonalizedRecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recent_product_ids: list[ProductId] = Field(
        default_factory=list, max_length=PERSONALIZED_RECENT_LIMIT,
    )
    saved_outfits: list[SavedOutfitSignal] = Field(
        default_factory=list, max_length=PERSONALIZED_SAVED_OUTFIT_LIMIT,
    )
    size_preferences: PersonalizedSizePreferences = Field(
        default_factory=PersonalizedSizePreferences,
    )
    exclude_product_ids: list[ProductId] = Field(
        default_factory=list, max_length=PERSONALIZED_EXCLUDE_LIMIT,
    )
    limit: int = Field(default=12, ge=1, le=24)


app = FastAPI(title="Defacto Semantic Search", version="1.0")
engine: SearchEngine | None = None
outfit_recommender: OutfitRecommender | None = None
personalized_recommender: PersonalizedRecommender | None = None


@app.on_event("startup")
def _load():
    global engine, outfit_recommender, personalized_recommender
    engine = SearchEngine()   # modeller bir kez yüklenir
    outfit_recommender = OutfitRecommender(engine)
    personalized_recommender = PersonalizedRecommender(engine)


# Filtre panelindeki "Cinsiyet" seçenekleri: etiket -> (gender, age_group) kısıtı.
# None = o alan serbest. Birden çok seçilirse aralarında OR uygulanır.
_AUDIENCE_FILTERS = {
    "Kadın": ("Kadın", "Yetişkin"),
    "Erkek": ("Erkek", "Yetişkin"),
    "Unisex": ("Unisex", None),
    "Kız çocuk": ("Kadın", "Çocuk"),
    "Erkek çocuk": ("Erkek", "Çocuk"),
    "Bebek": (None, "Bebek"),
}


@app.get("/search")
def search(q: str = Query("", max_length=200),
           limit: int = Query(24, ge=1, le=100),
           color: Annotated[list[str] | None, Query()] = None,
           category: Annotated[list[str] | None, Query()] = None,
           audience: Annotated[list[str] | None, Query()] = None,
           pattern: Annotated[list[str] | None, Query()] = None,
           price_min: Annotated[float | None, Query(ge=0)] = None,
           price_max: Annotated[float | None, Query(ge=0)] = None,
           discount: Annotated[bool, Query()] = False):
    # Manuel filtre paneli seçimleri. Hepsi boşsa engine.search'e hiç
    # gönderilmez; mevcut /search davranışı ve yanıt sözleşmesi birebir korunur.
    def _clean(values):
        return {v.strip() for v in (values or []) if v and v.strip() and len(v) <= 40}
    filters: dict = {}
    if (colors := _clean(color)):
        filters["color"] = colors
    if (categories := _clean(category)):
        filters["category"] = categories
    audiences = [_AUDIENCE_FILTERS[a] for a in _clean(audience) if a in _AUDIENCE_FILTERS]
    if audiences:
        filters["audience"] = audiences
    if (patterns := _clean(pattern)):
        filters["patterns"] = patterns
    if price_min is not None:
        filters["price_min"] = price_min
    if price_max is not None:
        filters["price_max"] = price_max
    if discount:
        filters["discount"] = True
    # Boş sorgu + hiç filtre yoksa eskisi gibi 422; ama en az bir filtre varsa
    # metin yazmadan da "filtreyle gözat" araması yapılabilir.
    if not q.strip() and not filters:
        raise HTTPException(
            status_code=422, detail="Arama metni veya en az bir filtre gerekli")
    extra = {"filters": filters} if filters else {}
    result = engine.search(q, limit=limit, **extra)
    return {
        "query": q,
        "understood": result.applied_filters,   # sorgudan çıkarılan yapı
        "relaxed": result.relaxed,              # gevşetilen kısıtlar
        "count": len(result.products),
        "products": result.products,
        "facets": getattr(result, "facets", {}),  # dinamik filtre verisi
    }


@app.get("/products/{product_id}/outfit-recommendations")
def outfit_recommendations(
    product_id: str = Path(..., min_length=1, max_length=100),
    limit: int = Query(4, ge=1, le=8),
    preference: Annotated[
        Literal["default", "alternative", "cheaper", "different_color"],
        Query(),
    ] = "default",
    current_product_id: Annotated[
        str | None, Query(min_length=1, max_length=100),
    ] = None,
    exclude_product_id: Annotated[list[str] | None, Query()] = None,
    preferred_size: Annotated[list[str] | None, Query()] = None,
):
    """Seçilen ürünü tamamlayan, stoktaki mantıklı parçaları döndürür."""

    if outfit_recommender is None:
        raise HTTPException(status_code=503, detail="Servis henüz hazır değil")

    excluded_ids = {
        str(value).strip()
        for value in (exclude_product_id or [])
        if str(value).strip()
    }
    if (len(excluded_ids) > OUTFIT_EDIT_EXCLUDE_LIMIT
            or any(len(value) > 100 for value in excluded_ids)):
        raise HTTPException(
            status_code=422,
            detail=(
                f"En fazla {OUTFIT_EDIT_EXCLUDE_LIMIT} geçerli ürün dışlanabilir"
            ),
        )

    preferred_sizes = {
        str(value).strip()
        for value in (preferred_size or [])
        if str(value).strip()
    }
    if (len(preferred_sizes) > OUTFIT_PREFERRED_SIZE_LIMIT
            or any(len(value) > 40 for value in preferred_sizes)):
        raise HTTPException(
            status_code=422,
            detail=(
                f"En fazla {OUTFIT_PREFERRED_SIZE_LIMIT} geçerli beden tercihi "
                "gönderilebilir"
            ),
        )

    if preference == "default":
        result = outfit_recommender.recommend(
            product_id,
            limit=limit,
            preferred_sizes=preferred_sizes,
        )
    else:
        if current_product_id is None:
            raise HTTPException(
                status_code=422,
                detail="Kombini düzenlemek için mevcut tamamlayıcı ürün gereklidir",
            )
        try:
            result = outfit_recommender.recommend_replacement(
                product_id,
                current_product_id,
                preference=preference,
                excluded_product_ids=excluded_ids,
                limit=limit,
                preferred_sizes=preferred_sizes,
            )
        except OutfitEditValidationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
    if result is None:
        raise HTTPException(status_code=404, detail="Ürün bulunamadı")
    return result


@app.post("/personalized-recommendations")
def personalized_recommendations(request: PersonalizedRecommendationRequest):
    """Tarayıcıdaki ürün ID'lerinden stok güvenli kişisel öneriler üretir."""

    if personalized_recommender is None:
        raise HTTPException(status_code=503, detail="Servis henüz hazır değil")
    return personalized_recommender.recommend(
        recent_product_ids=request.recent_product_ids,
        saved_outfits=[outfit.model_dump() for outfit in request.saved_outfits],
        size_preferences=request.size_preferences.model_dump(exclude_none=True),
        exclude_product_ids=request.exclude_product_ids,
        limit=request.limit,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def home():
    return FileResponse("static/index.html")
