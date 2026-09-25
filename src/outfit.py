# -*- coding: utf-8 -*-
"""Detaylı aramadan bağımsız, kural tabanlı kombin önerileri.

Kaynak ürün yalnızca ``product_id`` ile alınır. Adaylar önce stok, kategori,
cinsiyet ve yaş grubu ile kesin olarak filtrelenir; ardından renk, başlıktaki
desen/kalıp sinyalleri, fiyat yakınlığı ve popülerlik ile yeniden sıralanır.

Bu modül ``SearchEngine.search`` akışına dokunmaz. Mevcut hybrid sorgu altyapısı
sadece tamamlayıcı ürün aday havuzunu üretmek için yeniden kullanılır.
"""

from __future__ import annotations

import math
import re
import uuid
from collections import Counter
from dataclasses import dataclass

from qdrant_client import models

import config
from .enrich import FITS, PATTERNS, tr_fold


@dataclass(frozen=True)
class OutfitPlan:
    """Bir kaynak kategori için gösterilecek tamamlayıcı ürün ailesi."""

    title: str
    label: str
    target_categories: tuple[str, ...]


TOP_CATEGORIES = frozenset({
    "Tişört", "Polo Tişört", "Tesettür Tişört", "Gömlek", "Bluz",
    "Atlet", "Tunik", "Kazak", "Sweatshirt", "Tesettür Sweatshirt",
    "Süveter",
})

BOTTOM_CATEGORIES = frozenset({
    "Pantolon", "Jean Pantolon", "Chino Pantolon",
    "Tesettür Pantolon", "Eşofman Altı", "Jogger",
    "Şort", "Jean Şort", "Bermuda", "Etek", "Tesettür Etek",
    "Jean Etek", "Tayt", "Kapri",
})

ONE_PIECE_CATEGORIES = frozenset({
    "Elbise", "Tesettür Elbise", "Tulum",
})

LAYER_CATEGORIES = frozenset({
    "Ceket", "Blazer Ceket", "Jean Ceket", "Deri Ceket", "Hırka",
    "Hırka / Bolero", "Tesettur Hırka", "Yelek", "Mont", "Kaban / Parka",
    "Parka", "Trençkot", "Yağmurluk",
})

_BOTTOM_TARGETS = (
    "Pantolon", "Jean Pantolon", "Etek", "Tesettür Pantolon",
    "Şort", "Jean Şort", "Bermuda", "Eşofman Altı", "Jogger", "Tayt",
    "Chino Pantolon", "Tesettür Etek", "Jean Etek",
)

_TOP_TARGETS = (
    "Tişört", "Polo Tişört", "Gömlek", "Bluz", "Kazak", "Sweatshirt",
    "Tunik", "Tesettür Tişört", "Tesettür Sweatshirt", "Atlet", "Süveter",
)

_LAYER_TARGETS = (
    "Blazer Ceket", "Ceket", "Hırka", "Hırka / Bolero", "Yelek",
    "Jean Ceket", "Deri Ceket", "Kemer", "Çanta",
)

# Beden tercihi giyim parçalarına uygulanır. Kemer, takı ve yüzük kendi ölçü
# sistemlerini kullanır; çanta ise standart/bedensizdir. Bunlar kullanıcının
# giyim bedeniyle karşılaştırılmamalıdır.
SIZE_PREFERENCE_EXEMPT_CATEGORIES = frozenset({
    "Kemer", "Çanta", "Takı", "Yüzük",
})


def _is_unambiguous_jean_bottom(category: str | None, title: str | None) -> bool:
    if category != "Jean":
        return False
    folded_title = tr_fold(title or "")
    excluded = ("etek", "sort", "ceket", "elbise", "tulum", "gomlek")
    return "pantolon" in folded_title and not any(word in folded_title for word in excluded)


def get_outfit_plan(
    category: str | None, title: str | None = None,
) -> OutfitPlan | None:
    """Feed'deki gerçek kategori adına göre kombin planını döndürür."""

    if category in TOP_CATEGORIES:
        return OutfitPlan(
            title="Bu üstü tamamlayan alt giyim önerileri",
            label="Alt giyim",
            target_categories=_BOTTOM_TARGETS,
        )
    if category in BOTTOM_CATEGORIES or _is_unambiguous_jean_bottom(category, title):
        return OutfitPlan(
            title="Bu alt parçayı tamamlayan üstler",
            label="Üst giyim",
            target_categories=_TOP_TARGETS,
        )
    if category in ONE_PIECE_CATEGORIES:
        return OutfitPlan(
            title="Bu parçayı tamamlayan katmanlar ve aksesuarlar",
            label="Tamamlayıcı parça",
            target_categories=_LAYER_TARGETS,
        )
    if category in LAYER_CATEGORIES:
        return OutfitPlan(
            title="Bu dış giyimi tamamlayan üstler",
            label="İç katman",
            target_categories=_TOP_TARGETS,
        )
    return None


# Güvenli, gündelik kombinler için renk öncelikleri. Bunlar kesin filtre değil;
# katalogdaki çeşitliliği korumak için sadece yeniden sıralama sinyalidir.
COLOR_PAIRS: dict[str, tuple[str, ...]] = {
    "Siyah": ("Ekru", "Beyaz", "Bej", "Gri", "İndigo", "Siyah", "Haki"),
    "Beyaz": ("İndigo", "Lacivert", "Siyah", "Bej", "Haki", "Kahve", "Gri"),
    "Ekru": ("Kahve", "Bej", "İndigo", "Haki", "Siyah", "Lacivert"),
    "Bej": ("Ekru", "Beyaz", "Kahve", "Haki", "Lacivert", "Siyah"),
    "Gri": ("Siyah", "Beyaz", "Ekru", "Lacivert", "Bordo", "Pembe"),
    "Antrasit": ("Ekru", "Beyaz", "Gri", "Siyah", "Bordo", "Mavi"),
    "Lacivert": ("Beyaz", "Ekru", "Bej", "Gri", "Kırmızı", "İndigo"),
    "İndigo": ("Beyaz", "Ekru", "Bej", "Gri", "Siyah", "Kahve"),
    "Kahve": ("Ekru", "Bej", "Haki", "Beyaz", "İndigo", "Siyah"),
    "Kırmızı": ("Siyah", "Ekru", "Beyaz", "Bej", "Lacivert", "Gri"),
    "Bordo": ("Ekru", "Bej", "Gri", "Siyah", "Lacivert"),
    "Mavi": ("Beyaz", "Ekru", "Bej", "Lacivert", "Gri", "Kahve"),
    "Yeşil": ("Ekru", "Bej", "Kahve", "Siyah", "Lacivert", "Beyaz"),
    "Haki": ("Ekru", "Bej", "Kahve", "Siyah", "İndigo", "Beyaz"),
    "Pembe": ("Beyaz", "Ekru", "Bej", "Gri", "Lacivert", "Kahve"),
    "Sarı": ("Lacivert", "Ekru", "Beyaz", "Bej", "Kahve", "Siyah"),
    "Turuncu": ("Ekru", "Bej", "Lacivert", "Kahve", "Siyah", "Beyaz"),
    "Mor": ("Ekru", "Gri", "Siyah", "Beyaz", "Bej", "Lacivert"),
    "Turkuaz": ("Ekru", "Beyaz", "Bej", "Lacivert", "Kahve"),
    "Çok Renkli": ("Siyah", "Ekru", "Beyaz", "Bej", "Gri", "Lacivert"),
}

NEUTRAL_COLORS = frozenset({
    "Siyah", "Beyaz", "Ekru", "Bej", "Gri", "Antrasit", "Lacivert",
    "İndigo", "Kahve", "Haki",
})

_NON_VISIBLE_PATTERNS = {"düz"}
_CLEAN_FITS = {"slim fit", "regular fit", "straight", "skinny"}
_LOOSE_FITS = {"oversize", "relax fit", "wide leg", "boyfriend"}
_SEASON_RE = re.compile(r"(?:\d{2})?(SP|SM|AU|WN|CW|NS)")
_WARM_SEASONS = {"SP", "SM"}
_COLD_SEASONS = {"AU", "WN", "CW"}
_AGE_SIZE_RE = re.compile(r"(\d{1,2})\s*[/\-]\s*(\d{1,2})\s*yas")
_MONTH_SIZE_RE = re.compile(r"(\d{1,2})\s*[/\-]\s*(\d{1,2})\s*ay")
_SINGLE_MONTH_SIZE_RE = re.compile(r"^(\d{1,2})\s*ay")
_NORMALIZED_SIZE_RANGE_RE = re.compile(
    r"^(age|month):(\d{1,2})(?:-(\d{1,2}))?$"
)

OUTFIT_EDIT_PREFERENCES = frozenset({
    "alternative", "cheaper", "different_color",
})
OUTFIT_EDIT_EXCLUDE_LIMIT = 100
OUTFIT_PREFERRED_SIZE_LIMIT = 16
_OUTFIT_EDIT_SCROLL_BATCH = 128
_OUTFIT_EDIT_SCROLL_TARGET = 32
_OUTFIT_EDIT_SCROLL_MAX = 4096
_OUTFIT_EDIT_PAYLOAD_FIELDS = [
    "id", "title", "category", "category_path", "color", "patterns",
    "price", "sale_price", "gender", "age_group", "availability",
    "total_stock", "in_stock_sizes", "weekly_sales", "image", "link",
]


class OutfitEditValidationError(ValueError):
    """Kombin düzenleme isteği mevcut ürün verisiyle uygulanamıyor."""


def _title_signals(product: dict, lexicon: dict) -> set[str]:
    """Açıklamadaki gürültüyü almadan yalnızca ürün başlığını yorumlar."""

    folded = tr_fold(product.get("title") or "")
    return {
        canon
        for canon, triggers in lexicon.items()
        if any(trigger in folded for trigger in triggers)
    }


def _visible_patterns(product: dict) -> set[str]:
    """Spesifik desenlerin yanında genel baskılı/desenli sinyalini de korur."""

    return _title_signals(product, PATTERNS) - _NON_VISIBLE_PATTERNS


def _color_score(source_color: str | None, candidate_color: str | None) -> float:
    if not source_color or not candidate_color:
        return 0.0
    preferred = COLOR_PAIRS.get(source_color, ())
    if candidate_color in preferred:
        return 2.4 - 0.18 * preferred.index(candidate_color)
    if source_color in COLOR_PAIRS.get(candidate_color, ()):
        return 1.2
    if candidate_color in NEUTRAL_COLORS:
        return 0.8
    if candidate_color == source_color:
        return 0.65
    return 0.15


def _pattern_score(source: dict, candidate: dict) -> float:
    source_patterns = _visible_patterns(source)
    candidate_patterns = _visible_patterns(candidate)
    if source_patterns and not candidate_patterns:
        return 1.4
    if source_patterns and candidate_patterns:
        return -0.8
    if not candidate_patterns:
        return 0.45
    return 0.1


def _fit_score(source: dict, candidate: dict) -> float:
    source_fits = _title_signals(source, FITS)
    candidate_fits = _title_signals(candidate, FITS)
    if (source_fits & _LOOSE_FITS
            and candidate_fits & _CLEAN_FITS
            and not candidate_fits & _LOOSE_FITS):
        return 0.9
    if (source_fits & _CLEAN_FITS
            and candidate_fits & _LOOSE_FITS
            and not source_fits & _LOOSE_FITS):
        return 0.65
    if source_fits and candidate_fits and source_fits & candidate_fits:
        return 0.35
    return 0.0


def _season_code(product: dict) -> str | None:
    match = _SEASON_RE.search(str(product.get("id") or "").upper())
    return match.group(1) if match else None


def _season_score(source: dict, candidate: dict) -> float:
    source_season = _season_code(source)
    candidate_season = _season_code(candidate)
    if not source_season or not candidate_season:
        return 0.0
    if source_season == candidate_season:
        return 0.55
    if "NS" in (source_season, candidate_season):
        return 0.2
    if ({source_season, candidate_season} <= _WARM_SEASONS
            or {source_season, candidate_season} <= _COLD_SEASONS):
        return 0.3
    return 0.0


def _price_score(source: dict, candidate: dict) -> float:
    source_price = source.get("sale_price") or source.get("price")
    candidate_price = candidate.get("sale_price") or candidate.get("price")
    if not source_price or not candidate_price:
        return 0.0
    actual_ratio = candidate_price / source_price
    if source.get("category") in TOP_CATEGORIES:
        expected_ratio = 1.6
    elif (source.get("category") in BOTTOM_CATEGORIES
          or _is_unambiguous_jean_bottom(
              source.get("category"), source.get("title"))):
        expected_ratio = 0.65
    elif source.get("category") in LAYER_CATEGORIES:
        expected_ratio = 0.65
    else:
        expected_ratio = 1.0
    ratio = max(actual_ratio / expected_ratio, expected_ratio / actual_ratio)
    if ratio <= 1.35:
        return 0.9
    if ratio <= 1.8:
        return 0.55
    if ratio <= 2.5:
        return 0.2
    return 0.0


def _effective_price(product: dict) -> float | None:
    """İndirimli fiyatı, yoksa normal fiyatı güvenli bir sayıya çevirir."""

    value = product.get("sale_price")
    if value is None:
        value = product.get("price")
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if price > 0 else None


def compatibility_score(
    source: dict,
    candidate: dict,
    plan: OutfitPlan,
    search_rank: int,
) -> float:
    """Adayı deterministik ve açıklanabilir sinyallerle puanlar."""

    try:
        category_rank = plan.target_categories.index(candidate.get("category"))
    except ValueError:
        category_rank = len(plan.target_categories)

    category_score = max(0.0, 2.0 - 0.14 * category_rank)
    semantic_rank_score = 1.2 / (1.0 + search_rank / 8.0)
    popularity = min(math.log1p(candidate.get("weekly_sales") or 0) / 4.0, 1.0)

    return (
        category_score
        + semantic_rank_score
        + _color_score(source.get("color"), candidate.get("color"))
        + _pattern_score(source, candidate)
        + _fit_score(source, candidate)
        + _season_score(source, candidate)
        + _price_score(source, candidate)
        + popularity
    )


def recommendation_reason(source: dict, candidate: dict) -> str:
    """Arayüzde gösterilecek kısa ve kural tabanlı gerekçeyi üretir."""

    reasons: list[str] = []
    source_patterns = _visible_patterns(source)
    candidate_patterns = _visible_patterns(candidate)

    if source_patterns and not candidate_patterns:
        reasons.append("Sade görünümü, kaynak parçanın desenini dengeler.")

    source_color = source.get("color")
    candidate_color = candidate.get("color")
    if candidate_color in COLOR_PAIRS.get(source_color, ()):
        reasons.append(
            f"{candidate_color} tonu, {source_color.lower()} parçayla dengeli bir palet kurar."
        )
    elif candidate_color in NEUTRAL_COLORS:
        reasons.append(f"{candidate_color} tonu kolay eşleşen, güvenli bir tamamlayıcıdır.")

    if _fit_score(source, candidate) >= 0.65:
        reasons.append("Kalıp dengesi silueti daha düzenli gösterir.")

    if not reasons:
        reasons.append("Tamamlayıcı ürün kategorisi bu parçayla doğal bir eşleşme sunar.")

    return " ".join(reasons[:2])


def _serialize_product(payload: dict) -> dict:
    """Kombin endpoint'i için arama kartlarıyla uyumlu temel ürün sözlüğü."""

    return {
        "id": payload.get("id"),
        "title": payload.get("title"),
        "category": payload.get("category"),
        "color": payload.get("color"),
        "patterns": payload.get("patterns", []),
        "price": payload.get("price"),
        "sale_price": payload.get("sale_price"),
        "gender": payload.get("gender"),
        "age_group": payload.get("age_group"),
        "sizes": payload.get("in_stock_sizes", []),
        "image": payload.get("image"),
        "link": payload.get("link"),
    }


class OutfitRecommender:
    """SearchEngine'in modellerini paylaşan bağımsız kombin servisi."""

    def __init__(self, search_engine):
        self.search_engine = search_engine

    def _get_source(self, product_id: str) -> dict | None:
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, product_id))
        records = self.search_engine.client.retrieve(
            collection_name=config.COLLECTION,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return None
        return records[0].payload or None

    def _scroll_replacement_candidates(
        self,
        qfilter: models.Filter,
        *,
        accept_payload,
        existing_product_ids: set[str],
        target_count: int,
    ) -> list:
        """Hybrid havuz post-filter'da tükenirse katalogda sayfalı ara.

        Bu yol yalnız kombin önerilerinde ve hybrid sonuçlar yetersiz kaldığında
        çalışır. Özellikle çocuk beden uyumu Qdrant filtresinden sonra
        doğrulandığı için ilk semantik havuzun dışında kalan güvenli adayları
        bulur; ana ``/search`` akışına dokunmaz.
        """

        scroll = getattr(self.search_engine.client, "scroll", None)
        if not callable(scroll):
            return []

        candidates = []
        seen_ids = set(existing_product_ids)
        offset = None
        scanned = 0
        while scanned < _OUTFIT_EDIT_SCROLL_MAX and len(candidates) < target_count:
            page_limit = min(
                _OUTFIT_EDIT_SCROLL_BATCH,
                _OUTFIT_EDIT_SCROLL_MAX - scanned,
            )
            records, next_offset = scroll(
                collection_name=config.COLLECTION,
                scroll_filter=qfilter,
                limit=page_limit,
                offset=offset,
                with_payload=_OUTFIT_EDIT_PAYLOAD_FIELDS,
                with_vectors=False,
            )
            if not records:
                break
            scanned += len(records)
            for record in records:
                payload = record.payload or {}
                product_id = str(payload.get("id") or "")
                if not product_id or product_id in seen_ids:
                    continue
                if not accept_payload(payload):
                    continue
                candidates.append(record)
                seen_ids.add(product_id)
                if len(candidates) >= target_count:
                    break
            if next_offset is None:
                break
            offset = next_offset
        return candidates

    @staticmethod
    def _candidate_filter(
        source: dict,
        plan: OutfitPlan,
        *,
        target_categories: tuple[str, ...] | None = None,
        excluded_product_ids: set[str] | None = None,
        effective_price_lt: float | None = None,
        excluded_color: str | None = None,
        preferred_sizes: set[str] | None = None,
    ) -> models.Filter:
        categories = target_categories or plan.target_categories
        must: list = [
            models.FieldCondition(
                key="availability",
                match=models.MatchValue(value="in stock"),
            ),
            models.FieldCondition(
                key="category",
                match=models.MatchAny(any=list(categories)),
            ),
            models.FieldCondition(
                key="total_stock",
                range=models.Range(gt=0),
            ),
        ]

        if source.get("gender"):
            genders = [source["gender"]]
            if source["gender"] != "Unisex":
                genders.append("Unisex")
            must.append(models.FieldCondition(
                key="gender", match=models.MatchAny(any=genders)))

        if source.get("age_group"):
            must.append(models.FieldCondition(
                key="age_group",
                match=models.MatchValue(value=source["age_group"]),
            ))

        if (preferred_sizes
                and OutfitRecommender._audience_cohort(source) == "adult"):
            # ``in_stock_sizes`` yalnız stok adedi pozitif varyantları içerir.
            # Standart beden ürünleri kişisel beden tercihinden bağımsızdır.
            allowed_sizes = sorted(preferred_sizes | {"STD", "One Size"})
            must.append(models.Filter(should=[
                models.FieldCondition(
                    key="in_stock_sizes",
                    match=models.MatchAny(any=allowed_sizes),
                ),
                models.FieldCondition(
                    key="category",
                    match=models.MatchAny(
                        any=sorted(SIZE_PREFERENCE_EXEMPT_CATEGORIES)
                    ),
                ),
            ]))

        if effective_price_lt is not None:
            # ``_effective_price`` indirimli fiyat yoksa normal fiyatı kullanır.
            # Qdrant aday havuzu da aynı kurala uymalı; aksi halde yalnızca
            # ``price`` bilgisi olan gerçekten ucuz ürünler sorguda elenir.
            must.append(models.Filter(should=[
                models.FieldCondition(
                    key="sale_price",
                    range=models.Range(gt=0, lt=effective_price_lt),
                ),
                models.Filter(must=[
                    models.IsEmptyCondition(
                        is_empty=models.PayloadField(key="sale_price"),
                    ),
                    models.FieldCondition(
                        key="price",
                        range=models.Range(gt=0, lt=effective_price_lt),
                    ),
                ]),
            ]))

        must_not: list = []
        if excluded_product_ids:
            point_ids = [
                str(uuid.uuid5(uuid.NAMESPACE_URL, product_id))
                for product_id in sorted(excluded_product_ids)
            ]
            must_not.append(models.HasIdCondition(has_id=point_ids))
        if excluded_color:
            must_not.append(models.FieldCondition(
                key="color",
                match=models.MatchValue(value=excluded_color),
            ))

        return models.Filter(must=must, must_not=must_not or None)

    @staticmethod
    def _query_text(source: dict, plan: OutfitPlan) -> str:
        parts = [
            source.get("title"),
            source.get("color"),
            source.get("material"),
            *plan.target_categories[:4],
        ]
        return " ".join(str(part) for part in parts if part)

    @staticmethod
    def _replacement_query_text(
        source: dict, current: dict, plan: OutfitPlan,
    ) -> str:
        """Mevcut tamamlayıcıya yakın, ama düzenlenebilir alternatif sorgusu."""

        parts = [
            current.get("title"),
            current.get("category"),
            current.get("material"),
            source.get("title"),
            source.get("color"),
            *plan.target_categories[:3],
        ]
        return " ".join(str(part) for part in parts if part)

    @staticmethod
    def _audience_cohort(product: dict) -> str:
        first_path_part = tr_fold(
            str(product.get("category_path") or "").split(" > ", 1)[0]
        )
        if "bebek" in first_path_part or "yenidogan" in first_path_part:
            return "baby"
        if "cocuk" in first_path_part or "genc" in first_path_part:
            return "child"
        return "adult"

    @staticmethod
    def _normalize_child_size(size) -> str:
        folded = tr_fold(str(size))
        if match := _AGE_SIZE_RE.search(folded):
            return f"age:{match.group(1)}-{match.group(2)}"
        if match := _MONTH_SIZE_RE.search(folded):
            return f"month:{match.group(1)}-{match.group(2)}"
        if match := _SINGLE_MONTH_SIZE_RE.search(folded):
            return f"month:{match.group(1)}"
        return re.sub(r"\s+", "", folded)

    @staticmethod
    def _normalized_size_matches(available: str, preferred: str) -> bool:
        """Çocuk/ay aralıklarında kapsayan stok bedenini de eşleşme sayar."""

        if available == preferred:
            return True
        available_range = _NORMALIZED_SIZE_RANGE_RE.fullmatch(available)
        preferred_range = _NORMALIZED_SIZE_RANGE_RE.fullmatch(preferred)
        if not available_range or not preferred_range:
            return False
        if available_range.group(1) != preferred_range.group(1):
            return False
        available_start = int(available_range.group(2))
        available_end = int(available_range.group(3) or available_start)
        preferred_start = int(preferred_range.group(2))
        preferred_end = int(preferred_range.group(3) or preferred_start)
        return (
            available_start <= preferred_start
            and available_end >= preferred_end
        )

    @staticmethod
    def _normalized_size_overlaps(first: str, second: str) -> bool:
        if first == second:
            return True
        first_range = _NORMALIZED_SIZE_RANGE_RE.fullmatch(first)
        second_range = _NORMALIZED_SIZE_RANGE_RE.fullmatch(second)
        if not first_range or not second_range:
            return False
        if first_range.group(1) != second_range.group(1):
            return False
        first_start = int(first_range.group(2))
        first_end = int(first_range.group(3) or first_start)
        second_start = int(second_range.group(2))
        second_end = int(second_range.group(3) or second_start)
        return max(first_start, second_start) <= min(first_end, second_end)

    @classmethod
    def _matches_preferred_size(
        cls, candidate: dict, preferred_sizes: set[str] | None,
    ) -> bool:
        """Adayın hatırlanan bedenlerden en az birinde stokta olduğunu doğrular."""

        if not preferred_sizes:
            return True
        if candidate.get("category") in SIZE_PREFERENCE_EXEMPT_CATEGORIES:
            return True
        candidate_sizes = {
            cls._normalize_child_size(size)
            for size in candidate.get("in_stock_sizes") or []
        }
        if candidate_sizes & {"std", "onesize"}:
            return True
        preferred = {
            cls._normalize_child_size(size)
            for size in preferred_sizes
        }
        return any(
            cls._normalized_size_matches(available, wanted)
            for available in candidate_sizes
            for wanted in preferred
        )

    @classmethod
    def _is_valid_candidate(
        cls, source: dict, candidate: dict, plan: OutfitPlan,
        preferred_sizes: set[str] | None = None,
    ) -> bool:
        if candidate.get("id") == source.get("id"):
            return False
        if not source.get("gender") or not source.get("age_group"):
            return False
        if not candidate.get("gender") or not candidate.get("age_group"):
            return False
        if candidate.get("availability") != "in stock":
            return False
        if (candidate.get("total_stock") or 0) <= 0:
            return False
        if candidate.get("category") not in plan.target_categories:
            return False
        if source.get("gender") and candidate.get("gender") not in {
            source["gender"], "Unisex",
        }:
            return False
        if (source.get("age_group")
                and candidate.get("age_group") != source.get("age_group")):
            return False
        if cls._audience_cohort(source) != cls._audience_cohort(candidate):
            return False
        if cls._audience_cohort(source) in {"baby", "child"}:
            source_sizes = {
                cls._normalize_child_size(size)
                for size in source.get("in_stock_sizes") or []
            }
            candidate_sizes = {
                cls._normalize_child_size(size)
                for size in candidate.get("in_stock_sizes") or []
            }
            if not source_sizes or not candidate_sizes:
                return False
            candidate_is_universal = bool(
                candidate_sizes & {"std", "onesize"}
            )
            has_shared_size = any(
                cls._normalized_size_overlaps(source_size, candidate_size)
                for source_size in source_sizes
                for candidate_size in candidate_sizes
            )
            if not candidate_is_universal and not has_shared_size:
                return False
        if not cls._matches_preferred_size(candidate, preferred_sizes):
            return False
        return True

    @staticmethod
    def _diversify(ranked_points: list, limit: int) -> list:
        """Aynı kategori/renkten tekdüze bir liste oluşmasını önler."""

        chosen, seen_ids, seen_titles = [], set(), set()
        category_counts: Counter = Counter()
        color_counts: Counter = Counter()

        for point in ranked_points:
            payload = point.payload or {}
            product_id = payload.get("id")
            normalized_title = tr_fold(payload.get("title") or "")
            category = payload.get("category")
            color = payload.get("color")
            if product_id in seen_ids:
                continue
            if normalized_title and normalized_title in seen_titles:
                continue
            if category_counts[category] >= 2 or color_counts[color] >= 2:
                continue
            chosen.append(point)
            seen_ids.add(product_id)
            seen_titles.add(normalized_title)
            category_counts[category] += 1
            color_counts[color] += 1
            if len(chosen) >= limit:
                return chosen

        # Küçük katalog dilimlerinde çeşitlilik kuralı listeyi eksik bırakmasın.
        for point in ranked_points:
            product_id = (point.payload or {}).get("id")
            normalized_title = tr_fold((point.payload or {}).get("title") or "")
            if product_id in seen_ids:
                continue
            if normalized_title and normalized_title in seen_titles:
                continue
            chosen.append(point)
            seen_ids.add(product_id)
            seen_titles.add(normalized_title)
            if len(chosen) >= limit:
                break
        return chosen

    def recommend(
        self,
        product_id: str,
        limit: int = 4,
        preferred_sizes: set[str] | None = None,
    ) -> dict | None:
        source = self._get_source(product_id)
        if source is None:
            return None

        plan = get_outfit_plan(source.get("category"), source.get("title"))
        source_product = _serialize_product(source)
        if plan is None:
            return {
                "source": source_product,
                "eligible": False,
                "title": "Bu ürün için kombin önerisi bulunmuyor",
                "summary": "Kombin önerileri şu anda giyim parçaları için hazırlanıyor.",
                "target_slot": None,
                "target_categories": [],
                "count": 0,
                "products": [],
            }
        if not source.get("gender") or not source.get("age_group"):
            return {
                "source": source_product,
                "eligible": False,
                "title": "Bu ürün için kombin önerisi hazırlanamadı",
                "summary": "Ürünün hedef kitle bilgisi eksik olduğu için güvenli bir eşleşme yapılamadı.",
                "target_slot": None,
                "target_categories": [],
                "count": 0,
                "products": [],
            }
        source_cohort = self._audience_cohort(source)
        if (source_cohort in {"baby", "child"}
                and not source.get("in_stock_sizes")):
            return {
                "source": source_product,
                "eligible": False,
                "title": "Bu ürün için kombin önerisi hazırlanamadı",
                "summary": "Kaynak ürünün stokta beden bilgisi olmadığı için güvenli bir eşleşme yapılamadı.",
                "target_slot": None,
                "target_categories": [],
                "count": 0,
                "products": [],
            }

        candidate_limit = (
            config.PREFETCH_LIMIT
            if source_cohort in {"baby", "child"}
            else min(
                config.PREFETCH_LIMIT,
                max(config.OUTFIT_CANDIDATE_POOL, limit * 12),
            )
        )
        candidate_filter = self._candidate_filter(
            source,
            plan,
            preferred_sizes=preferred_sizes,
        )
        points = self.search_engine._hybrid_query(
            self._query_text(source, plan),
            candidate_filter,
            candidate_limit,
        )
        points = [
            point for point in points
            if self._is_valid_candidate(
                source,
                point.payload or {},
                plan,
                preferred_sizes,
            )
        ]
        ranked = [
            point
            for _, point in sorted(
                enumerate(points),
                key=lambda item: compatibility_score(
                    source, item[1].payload or {}, plan, item[0]),
                reverse=True,
            )
        ]
        if (source_cohort in {"baby", "child"}
                and len(self._diversify(ranked, limit)) < limit):
            existing_ids = {
                str((point.payload or {}).get("id") or "")
                for point in points
            }
            fallback = self._scroll_replacement_candidates(
                candidate_filter,
                accept_payload=lambda candidate: self._is_valid_candidate(
                    source,
                    candidate,
                    plan,
                    preferred_sizes,
                ),
                existing_product_ids=existing_ids,
                target_count=max(_OUTFIT_EDIT_SCROLL_TARGET, limit * 16),
            )
            ranked.extend(
                point
                for _, point in sorted(
                    enumerate(fallback),
                    key=lambda item: compatibility_score(
                        source,
                        item[1].payload or {},
                        plan,
                        candidate_limit + item[0],
                    ),
                    reverse=True,
                )
            )
        selected = self._diversify(ranked, limit)

        products = []
        for point in selected:
            product = _serialize_product(point.payload or {})
            product["reason"] = recommendation_reason(source, point.payload or {})
            products.append(product)

        if products:
            summary = (
                "Renk, ürün türü, yaş grubu, fiyat dengesi ve hatırlanan bedenlerin "
                "stok durumu birlikte değerlendirildi."
                if preferred_sizes
                else "Renk, ürün türü, yaş grubu ve fiyat dengesi birlikte değerlendirildi."
            )
        else:
            summary = (
                "Hatırladığın bedenlerde stokta uygun bir tamamlayıcı bulunamadı."
                if preferred_sizes
                else "Bu parça için stokta uygun bir tamamlayıcı bulunamadı."
            )
        return {
            "source": source_product,
            "eligible": True,
            "title": plan.title,
            "summary": summary,
            "target_slot": plan.label,
            "target_categories": list(plan.target_categories),
            "count": len(products),
            "products": products,
        }

    def recommend_replacement(
        self,
        product_id: str,
        current_product_id: str,
        *,
        preference: str,
        excluded_product_ids: set[str] | None = None,
        limit: int = 1,
        preferred_sizes: set[str] | None = None,
    ) -> dict | None:
        """Bir kombin kartı için stateless ve güvenli bir alternatif döndürür.

        ``alternative`` normal uyum sırasındaki bir sonraki parçayı;
        ``cheaper`` aynı kategoride kesin olarak daha ucuz bir parçayı;
        ``different_color`` ise aynı kategoride farklı renkli en yakın parçayı
        arar. Varsayılan ``recommend`` akışı bu metottan tamamen bağımsızdır.
        """

        if preference not in OUTFIT_EDIT_PREFERENCES:
            raise OutfitEditValidationError("Geçersiz kombin düzenleme tercihi")

        source = self._get_source(product_id)
        current = self._get_source(current_product_id)
        if source is None or current is None:
            return None

        plan = get_outfit_plan(source.get("category"), source.get("title"))
        if plan is None:
            raise OutfitEditValidationError("Bu ürünün kombini düzenlenemiyor")
        if not self._is_valid_candidate(source, current, plan):
            raise OutfitEditValidationError(
                "Değiştirilecek ürün bu kombin için geçerli bir tamamlayıcı değil"
            )

        excluded_ids = {
            str(value).strip()
            for value in (excluded_product_ids or set())
            if str(value).strip()
        }
        excluded_ids.add(str(current_product_id))

        target_categories: tuple[str, ...] | None = None
        effective_price_lt: float | None = None
        excluded_color: str | None = None
        if preference == "cheaper":
            current_price = _effective_price(current)
            if current_price is None:
                raise OutfitEditValidationError(
                    "Mevcut tamamlayıcının fiyat bilgisi bulunmuyor"
                )
            target_categories = (current["category"],)
            effective_price_lt = current_price
        elif preference == "different_color":
            excluded_color = current.get("color")
            if not excluded_color:
                raise OutfitEditValidationError(
                    "Mevcut tamamlayıcının renk bilgisi bulunmuyor"
                )
            target_categories = (current["category"],)

        candidate_filter = self._candidate_filter(
            source,
            plan,
            target_categories=target_categories,
            excluded_product_ids=excluded_ids,
            effective_price_lt=effective_price_lt,
            excluded_color=excluded_color,
            preferred_sizes=preferred_sizes,
        )
        points = self.search_engine._hybrid_query(
            (
                self._query_text(source, plan)
                if preference == "alternative"
                else self._replacement_query_text(source, current, plan)
            ),
            candidate_filter,
            config.PREFETCH_LIMIT,
        )

        current_price = _effective_price(current)
        def is_replacement_candidate(candidate: dict) -> bool:
            candidate_id = str(candidate.get("id") or "")
            if not candidate_id or candidate_id in excluded_ids:
                return False
            if not self._is_valid_candidate(
                source, candidate, plan, preferred_sizes,
            ):
                return False
            if target_categories and candidate.get("category") not in target_categories:
                return False
            if preference == "cheaper":
                candidate_price = _effective_price(candidate)
                if (candidate_price is None or current_price is None
                        or candidate_price >= current_price):
                    return False
            if preference == "different_color":
                candidate_color = candidate.get("color")
                if not candidate_color or candidate_color == excluded_color:
                    return False
            return True

        candidates = [
            point for point in points
            if is_replacement_candidate(point.payload or {})
        ]
        ranked = [
            point
            for _, point in sorted(
                enumerate(candidates),
                key=lambda item: compatibility_score(
                    source, item[1].payload or {}, plan, item[0]),
                reverse=True,
            )
        ]
        if len(self._diversify(ranked, limit)) < limit:
            existing_ids = {
                str((point.payload or {}).get("id") or "")
                for point in candidates
            }
            fallback = self._scroll_replacement_candidates(
                candidate_filter,
                accept_payload=is_replacement_candidate,
                existing_product_ids=existing_ids,
                target_count=max(_OUTFIT_EDIT_SCROLL_TARGET, limit * 16),
            )
            ranked.extend(
                point
                for _, point in sorted(
                    enumerate(fallback),
                    key=lambda item: compatibility_score(
                        source,
                        item[1].payload or {},
                        plan,
                        config.PREFETCH_LIMIT + item[0],
                    ),
                    reverse=True,
                )
            )
        selected = self._diversify(ranked, limit)

        products = []
        for point in selected:
            payload = point.payload or {}
            product = _serialize_product(payload)
            reason = recommendation_reason(source, payload)
            if preference == "cheaper":
                reason = "Seçtiğin tamamlayıcıdan daha uygun fiyatlıdır. " + reason
            elif preference == "different_color":
                reason = "Aynı kategoride, farklı renkte benzer bir seçenektir. " + reason
            product["reason"] = reason
            products.append(product)

        summaries = {
            "alternative": "Aynı uyum kurallarıyla yeni tamamlayıcı parçalar değerlendirildi.",
            "cheaper": "Uyumu koruyan daha uygun fiyatlı parçalar değerlendirildi.",
            "different_color": (
                "Aynı kategorideki farklı renkli benzer ürünler değerlendirildi."
            ),
        }
        return {
            "source": _serialize_product(source),
            "eligible": True,
            "title": plan.title,
            "summary": summaries[preference],
            "target_slot": plan.label,
            "target_categories": list(plan.target_categories),
            "count": len(products),
            "products": products,
        }
