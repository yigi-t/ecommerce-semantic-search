# -*- coding: utf-8 -*-
"""Son gezilen ve kaydedilen ürünlerden stateless kişisel öneriler üretir.

Bu servis ana ``SearchEngine.search`` akışından bağımsızdır. Arama motorunun
yalnızca zaten yüklenmiş modellerini, Qdrant istemcisini ve düşük seviyeli
hybrid sorgusunu paylaşır; detaylı aramanın parser, filtre gevşetme ve sıralama
kurallarını değiştirmez.
"""

from __future__ import annotations

import math
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from qdrant_client import models

import config
from .enrich import tr_fold
from .outfit import OutfitRecommender, SIZE_PREFERENCE_EXEMPT_CATEGORIES


PERSONALIZED_RECENT_LIMIT = 10
PERSONALIZED_SAVED_OUTFIT_LIMIT = 20
PERSONALIZED_EXCLUDE_LIMIT = 100
PERSONALIZED_ID_LENGTH_LIMIT = 100

_RECENT_DECAY = 0.82
_SAVED_WEIGHT = 2.5
_SAVED_DECAY = 0.90
_CANDIDATE_POOL = max(config.PREFETCH_LIMIT, 160)
_SCROLL_BATCH = 128
_SCROLL_MAX = 2048
_SCROLL_TARGET = 48

_PROFILE_FIELDS = ("category", "color", "patterns", "fabrics", "fits")
_PROFILE_FIELD_LABELS = {
    "category": "kategori",
    "color": "renk",
    "patterns": "desen",
    "fabrics": "kumaş",
    "fits": "kalıp",
}
_ATTRIBUTE_WEIGHTS = {
    "category": 0.35,
    "color": 0.20,
    "patterns": 0.20,
    "fabrics": 0.15,
    "fits": 0.10,
}
_PAYLOAD_FIELDS = [
    "id", "title", "category", "category_path", "color", "patterns",
    "fabrics", "fits", "price", "sale_price", "gender", "age_group",
    "availability", "total_stock", "in_stock_sizes", "weekly_sales",
    "image", "link",
]

FOOTWEAR_CATEGORIES = frozenset({
    "Ayakkabı", "Günlük Ayakkabı", "Topuklu Ayakkabı", "Sandalet",
    "Terlik", "Ev Terliği", "Sneaker", "Bot", "Çizme", "Babet",
    "Loafer", "Çorap",
})

_ALPHA_TOKEN_RE = re.compile(r"^(?:XXS|XS|S|M|L|XL|XXL|[3-6]XL)$")
_NUMERIC_SIZE_RE = re.compile(r"^(\d{2})(?:([/\-])(\d{2}))?$")
_JEAN_SIZE_RE = re.compile(r"beden\s*(\d{2})\s*-\s*boy\s*(\d{2})")
_CHILD_SIZE_RE = re.compile(
    r"^(age|month):(\d{1,2})(?:-(\d{1,2}))?$"
)


@dataclass(frozen=True)
class AudienceLane:
    gender: str
    age_group: str
    cohort: str


@dataclass
class PreferenceProfile:
    features: dict[str, Counter]
    reference_price: float | None
    query_text: str


def _effective_price(product: dict) -> float | None:
    value = product.get("sale_price")
    if value is None or value <= 0:
        value = product.get("price")
    if value is None or value <= 0:
        return None
    return float(value)


def _clean_id(value) -> str:
    return str(value or "").strip()


def _unique(values: Iterable[str]) -> list[str]:
    result, seen = [], set()
    for value in values:
        clean = _clean_id(value)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        result.append(clean)
    return result


def _saved_ids(saved_outfit) -> tuple[str, str]:
    if isinstance(saved_outfit, dict):
        return (
            _clean_id(saved_outfit.get("source_product_id")),
            _clean_id(saved_outfit.get("recommended_product_id")),
        )
    return (
        _clean_id(getattr(saved_outfit, "source_product_id", "")),
        _clean_id(getattr(saved_outfit, "recommended_product_id", "")),
    )


def _valid_normalized_child_size(value: str) -> bool:
    match = _CHILD_SIZE_RE.fullmatch(value)
    if not match:
        return False
    start = int(match.group(2))
    end = int(match.group(3) or start)
    return start <= end


class PersonalizedRecommender:
    """Yerel kullanıcı sinyallerini Qdrant'taki güncel ürünlerle doğrular."""

    def __init__(self, search_engine):
        self.search_engine = search_engine

    @staticmethod
    def _empty_response() -> dict:
        return {
            "personalized": False,
            "strategy": "empty",
            "signal_count": 0,
            "summary": (
                "Sana özel öneri hazırlamak için henüz yeterli ürün etkileşimi yok."
            ),
            "count": 0,
            "products": [],
        }

    def _retrieve_signals(self, product_ids: list[str]) -> dict[str, dict]:
        if not product_ids:
            return {}
        point_ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, product_id))
            for product_id in product_ids
        ]
        records = self.search_engine.client.retrieve(
            collection_name=config.COLLECTION,
            ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )
        products = {}
        for record in records:
            payload = record.payload or {}
            product_id = _clean_id(payload.get("id"))
            if product_id and product_id in product_ids:
                products[product_id] = payload
        return products

    @staticmethod
    def _audience_lane(product: dict | None) -> AudienceLane | None:
        if not product:
            return None
        gender = _clean_id(product.get("gender"))
        age_group = _clean_id(product.get("age_group"))
        category_path = _clean_id(product.get("category_path"))
        if not gender or not age_group or not category_path:
            return None
        return AudienceLane(
            gender=gender,
            age_group=age_group,
            cohort=OutfitRecommender._audience_cohort(product),
        )

    @classmethod
    def _active_lane(
        cls,
        recent_ids: list[str],
        saved_pairs: list[tuple[str, str]],
        products: dict[str, dict],
    ) -> AudienceLane | None:
        ordered_ids = [
            *recent_ids,
            *(product_id for pair in saved_pairs for product_id in pair),
        ]
        for product_id in ordered_ids:
            lane = cls._audience_lane(products.get(product_id))
            if lane is not None:
                return lane
        return None

    @classmethod
    def _signal_weights(
        cls,
        recent_ids: list[str],
        saved_pairs: list[tuple[str, str]],
        products: dict[str, dict],
        lane: AudienceLane,
    ) -> tuple[dict[str, float], set[str]]:
        weights: dict[str, float] = {}
        saved_signal_ids: set[str] = set()

        for index, product_id in enumerate(recent_ids):
            product = products.get(product_id)
            if cls._audience_lane(product) != lane:
                continue
            weights[product_id] = (
                weights.get(product_id, 0.0) + _RECENT_DECAY ** index
            )

        for index, pair in enumerate(saved_pairs):
            pair_weight = _SAVED_WEIGHT * (_SAVED_DECAY ** index)
            for product_id in pair:
                product = products.get(product_id)
                if cls._audience_lane(product) != lane:
                    continue
                weights[product_id] = weights.get(product_id, 0.0) + pair_weight
                saved_signal_ids.add(product_id)

        return weights, saved_signal_ids

    @staticmethod
    def _build_profile(
        products: dict[str, dict], weights: dict[str, float],
    ) -> PreferenceProfile | None:
        features = {field: Counter() for field in _PROFILE_FIELDS}
        weighted_log_price = 0.0
        price_weight = 0.0

        for product_id, weight in weights.items():
            product = products[product_id]
            for field in _PROFILE_FIELDS:
                raw_value = product.get(field)
                values = raw_value if isinstance(raw_value, list) else [raw_value]
                for value in _unique(values):
                    if value:
                        features[field][value] += weight

            price = _effective_price(product)
            if price is not None:
                weighted_log_price += math.log(price) * weight
                price_weight += weight

        query_parts = []
        for field in _PROFILE_FIELDS:
            for value, weight in features[field].most_common(6):
                repetitions = max(1, min(4, round(weight)))
                query_parts.extend(
                    [f"{_PROFILE_FIELD_LABELS[field]} {value}"] * repetitions
                )
        if not query_parts:
            return None

        reference_price = (
            math.exp(weighted_log_price / price_weight)
            if price_weight > 0
            else None
        )
        return PreferenceProfile(
            features=features,
            reference_price=reference_price,
            query_text=" ".join(query_parts)[:1200],
        )

    @staticmethod
    def _candidate_filter(
        lane: AudienceLane, excluded_product_ids: set[str],
    ) -> models.Filter:
        genders = [lane.gender]
        if lane.gender != "Unisex":
            genders.append("Unisex")
        must = [
            models.FieldCondition(
                key="availability",
                match=models.MatchValue(value="in stock"),
            ),
            models.FieldCondition(
                key="total_stock",
                range=models.Range(gt=0),
            ),
            models.FieldCondition(
                key="gender",
                match=models.MatchAny(any=genders),
            ),
            models.FieldCondition(
                key="age_group",
                match=models.MatchValue(value=lane.age_group),
            ),
        ]
        must_not = None
        if excluded_product_ids:
            must_not = [models.HasIdCondition(has_id=[
                str(uuid.uuid5(uuid.NAMESPACE_URL, product_id))
                for product_id in sorted(excluded_product_ids)
            ])]
        return models.Filter(must=must, must_not=must_not)

    @staticmethod
    def _is_universal_size(value) -> bool:
        normalized = re.sub(r"\s+", "", tr_fold(str(value)))
        return normalized in {"std", "onesize"}

    @staticmethod
    def _alpha_tokens(value) -> list[str]:
        normalized = re.sub(r"\s+", "", tr_fold(str(value))).upper()
        parts = normalized.split("/")
        if len(parts) <= 2 and all(_ALPHA_TOKEN_RE.fullmatch(part) for part in parts):
            return parts
        return []

    @staticmethod
    def _numeric_size_matches(
        value, preferred: str, *, slash_is_range: bool = False,
    ) -> bool:
        normalized = re.sub(r"\s+", "", tr_fold(str(value)))
        match = _NUMERIC_SIZE_RE.fullmatch(normalized)
        if not match:
            return False
        try:
            wanted = int(preferred)
        except (TypeError, ValueError):
            return False
        start = int(match.group(1))
        if not match.group(2):
            return start == wanted
        end = int(match.group(3))
        if ((match.group(2) == "-" or slash_is_range)
                and start < end and end - start <= 10):
            return start <= wanted <= end
        return wanted in {start, end}

    @staticmethod
    def _jean_size(value) -> tuple[str, str] | None:
        match = _JEAN_SIZE_RE.search(tr_fold(str(value)))
        return (match.group(1), match.group(2)) if match else None

    @classmethod
    def _size_is_safe(
        cls,
        candidate: dict,
        lane: AudienceLane,
        size_preferences: dict[str, str | None],
    ) -> tuple[bool, bool]:
        """Return ``(allowed, remembered_size_matched)``.

        A preference becomes a hard constraint only when the candidate's size
        system is recognizable. Unknown systems stay eligible instead of being
        incorrectly discarded.
        """

        if candidate.get("category") in SIZE_PREFERENCE_EXEMPT_CATEGORIES:
            return True, False
        sizes = [
            str(value).strip()
            for value in candidate.get("in_stock_sizes") or []
            if str(value).strip()
        ]
        if any(cls._is_universal_size(value) for value in sizes):
            return True, False
        if not sizes:
            return True, False

        if candidate.get("category") in FOOTWEAR_CATEGORIES:
            preference = _clean_id(size_preferences.get("shoe"))
            numeric_sizes = [
                value for value in sizes
                if _NUMERIC_SIZE_RE.fullmatch(
                    re.sub(r"\s+", "", tr_fold(value))
                )
            ]
            if preference and numeric_sizes:
                matches = any(
                    cls._numeric_size_matches(
                        value, preference, slash_is_range=True,
                    )
                    for value in numeric_sizes
                )
                return matches, matches
            if lane.cohort not in {"baby", "child"}:
                return True, False

        if lane.cohort in {"baby", "child"}:
            preference = _clean_id(size_preferences.get("child"))
            normalized_sizes = [
                OutfitRecommender._normalize_child_size(value)
                for value in sizes
            ]
            child_sizes = [
                value for value in normalized_sizes
                if _valid_normalized_child_size(value)
            ]
            if not preference or not child_sizes:
                return True, False
            wanted = OutfitRecommender._normalize_child_size(preference)
            if not _valid_normalized_child_size(wanted):
                return True, False
            matches = any(
                OutfitRecommender._normalized_size_matches(available, wanted)
                for available in child_sizes
            )
            return matches, matches

        applicable_results = []
        jean_sizes = [size for value in sizes if (size := cls._jean_size(value))]
        waist = _clean_id(size_preferences.get("jean_waist"))
        length = _clean_id(size_preferences.get("jean_length"))
        if jean_sizes and waist and length:
            applicable_results.append((waist, length) in jean_sizes)

        alpha_sizes = [token for value in sizes for token in cls._alpha_tokens(value)]
        alpha = _clean_id(size_preferences.get("alpha")).upper()
        if alpha_sizes and alpha:
            applicable_results.append(alpha in alpha_sizes)

        numeric_sizes = [
            value for value in sizes
            if _NUMERIC_SIZE_RE.fullmatch(
                re.sub(r"\s+", "", tr_fold(value))
            )
        ]
        numeric = _clean_id(size_preferences.get("numeric"))
        if numeric_sizes and numeric:
            applicable_results.append(any(
                cls._numeric_size_matches(value, numeric)
                for value in numeric_sizes
            ))

        if not applicable_results:
            return True, False
        matches = any(applicable_results)
        return matches, matches

    @classmethod
    def _is_valid_candidate(
        cls,
        candidate: dict,
        lane: AudienceLane,
        excluded_product_ids: set[str],
        size_preferences: dict[str, str | None],
    ) -> bool:
        product_id = _clean_id(candidate.get("id"))
        if not product_id or product_id in excluded_product_ids:
            return False
        if candidate.get("availability") != "in stock":
            return False
        if (candidate.get("total_stock") or 0) <= 0:
            return False
        if not candidate.get("gender") or not candidate.get("age_group"):
            return False
        if not candidate.get("category_path"):
            return False
        if candidate.get("gender") not in {lane.gender, "Unisex"}:
            return False
        if candidate.get("age_group") != lane.age_group:
            return False
        if OutfitRecommender._audience_cohort(candidate) != lane.cohort:
            return False
        allowed, _ = cls._size_is_safe(candidate, lane, size_preferences)
        return allowed

    @staticmethod
    def _attribute_affinity(candidate: dict, profile: PreferenceProfile) -> float:
        weighted_score = 0.0
        available_weight = 0.0
        for field, field_weight in _ATTRIBUTE_WEIGHTS.items():
            preferences = profile.features[field]
            if not preferences:
                continue
            raw_value = candidate.get(field)
            candidate_values = raw_value if isinstance(raw_value, list) else [raw_value]
            candidate_values = {_clean_id(value) for value in candidate_values}
            candidate_values.discard("")
            strongest = max(preferences.values())
            matched = max(
                (preferences[value] for value in candidate_values if value in preferences),
                default=0.0,
            )
            weighted_score += field_weight * (matched / strongest)
            available_weight += field_weight
        return weighted_score / available_weight if available_weight else 0.0

    @staticmethod
    def _price_affinity(candidate: dict, reference_price: float | None) -> float:
        price = _effective_price(candidate)
        if price is None or reference_price is None:
            return 0.5
        return math.exp(-abs(math.log(price / reference_price)))

    @classmethod
    def _rank(
        cls, points: list, profile: PreferenceProfile,
    ) -> list:
        total = max(len(points), 1)
        scored = []
        for index, point in enumerate(points):
            payload = point.payload or {}
            semantic = 1.0 - (index / total)
            attributes = cls._attribute_affinity(payload, profile)
            price = cls._price_affinity(payload, profile.reference_price)
            popularity = min(
                math.log1p(payload.get("weekly_sales") or 0) / 5.0,
                1.0,
            )
            final_score = (
                semantic * 0.55
                + attributes * 0.28
                + price * 0.10
                + popularity * 0.07
            )
            scored.append((final_score, -index, point))
        scored.sort(key=lambda value: (value[0], value[1]), reverse=True)
        return [point for _, _, point in scored]

    @staticmethod
    def _diversify(ranked_points: list, limit: int) -> list:
        chosen, seen_ids, seen_titles = [], set(), set()
        category_counts: Counter = Counter()
        color_counts: Counter = Counter()

        for point in ranked_points:
            payload = point.payload or {}
            product_id = _clean_id(payload.get("id"))
            title = tr_fold(_clean_id(payload.get("title")))
            category = payload.get("category")
            color = payload.get("color")
            if not product_id or product_id in seen_ids:
                continue
            if title and title in seen_titles:
                continue
            if category_counts[category] >= 2 or color_counts[color] >= 2:
                continue
            chosen.append(point)
            seen_ids.add(product_id)
            if title:
                seen_titles.add(title)
            category_counts[category] += 1
            color_counts[color] += 1
            if len(chosen) >= limit:
                return chosen

        for point in ranked_points:
            payload = point.payload or {}
            product_id = _clean_id(payload.get("id"))
            title = tr_fold(_clean_id(payload.get("title")))
            if not product_id or product_id in seen_ids:
                continue
            if title and title in seen_titles:
                continue
            chosen.append(point)
            seen_ids.add(product_id)
            if title:
                seen_titles.add(title)
            if len(chosen) >= limit:
                break
        return chosen

    def _scroll_candidates(
        self,
        qfilter: models.Filter,
        *,
        lane: AudienceLane,
        excluded_product_ids: set[str],
        size_preferences: dict[str, str | None],
        existing_product_ids: set[str],
        target_count: int,
    ) -> list:
        scroll = getattr(self.search_engine.client, "scroll", None)
        if not callable(scroll):
            return []

        candidates, seen_ids = [], set(existing_product_ids)
        offset = None
        scanned = 0
        while scanned < _SCROLL_MAX and len(candidates) < target_count:
            records, next_offset = scroll(
                collection_name=config.COLLECTION,
                scroll_filter=qfilter,
                limit=min(_SCROLL_BATCH, _SCROLL_MAX - scanned),
                offset=offset,
                with_payload=_PAYLOAD_FIELDS,
                with_vectors=False,
            )
            if not records:
                break
            scanned += len(records)
            for record in records:
                payload = record.payload or {}
                product_id = _clean_id(payload.get("id"))
                if not product_id or product_id in seen_ids:
                    continue
                if not self._is_valid_candidate(
                    payload, lane, excluded_product_ids, size_preferences,
                ):
                    continue
                candidates.append(record)
                seen_ids.add(product_id)
                if len(candidates) >= target_count:
                    break
            if next_offset is None:
                break
            offset = next_offset
        return candidates

    @classmethod
    def _reason(
        cls,
        candidate: dict,
        profile: PreferenceProfile,
        lane: AudienceLane,
        size_preferences: dict[str, str | None],
    ) -> str:
        category = _clean_id(candidate.get("category"))
        color = _clean_id(candidate.get("color"))
        reasons = []
        if category and category in profile.features["category"]:
            reasons.append(f"İlgilendiğin {category.lower()} ürünleriyle aynı kategoride")
        if color and color in profile.features["color"]:
            reasons.append(f"beğendiğin {color.lower()} renk tercihine uyuyor")
        if not reasons:
            for field, label in (("patterns", "desen"), ("fabrics", "kumaş"),
                                 ("fits", "kalıp")):
                values = candidate.get(field) or []
                match = next(
                    (value for value in values if value in profile.features[field]),
                    None,
                )
                if match:
                    reasons.append(f"ilgilendiğin {match} {label} özelliğini taşıyor")
                    break
        if reasons:
            reason = "; ".join(reasons) + "."
        else:
            reason = "Son gezdiğin ürünlere anlamsal olarak benzeyen stoklu bir seçenektir."
        _, size_matched = cls._size_is_safe(candidate, lane, size_preferences)
        if size_matched:
            reason += " Hatırladığın beden stokta."
        return reason[0].upper() + reason[1:]

    @classmethod
    def _serialize_product(
        cls,
        point,
        profile: PreferenceProfile,
        lane: AudienceLane,
        size_preferences: dict[str, str | None],
    ) -> dict:
        payload = point.payload or {}
        score = getattr(point, "score", None)
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
            "score": round(score, 4) if score is not None else None,
            "reason": cls._reason(payload, profile, lane, size_preferences),
        }

    def recommend(
        self,
        *,
        recent_product_ids: list[str] | None = None,
        saved_outfits: list | None = None,
        size_preferences: dict[str, str | None] | None = None,
        exclude_product_ids: list[str] | set[str] | None = None,
        limit: int = 12,
    ) -> dict:
        recent_ids = _unique(recent_product_ids or [])[:PERSONALIZED_RECENT_LIMIT]
        saved_pairs = [
            pair for item in (saved_outfits or [])
            if any(pair := _saved_ids(item))
        ][:PERSONALIZED_SAVED_OUTFIT_LIMIT]
        flattened_saved_ids = [
            product_id
            for pair in saved_pairs
            for product_id in pair
            if product_id
        ]
        signal_ids = _unique([*recent_ids, *flattened_saved_ids])
        if not signal_ids:
            return self._empty_response()

        products = self._retrieve_signals(signal_ids)
        lane = self._active_lane(recent_ids, saved_pairs, products)
        if lane is None:
            return self._empty_response()

        weights, saved_signal_ids = self._signal_weights(
            recent_ids, saved_pairs, products, lane,
        )
        profile = self._build_profile(products, weights) if weights else None
        if profile is None:
            return self._empty_response()

        size_preferences = dict(size_preferences or {})
        explicit_excludes = set(_unique(exclude_product_ids or []))
        excluded_ids = set(signal_ids) | explicit_excludes
        qfilter = self._candidate_filter(lane, excluded_ids)
        raw_points = self.search_engine._hybrid_query(
            profile.query_text,
            qfilter,
            _CANDIDATE_POOL,
        )

        points, seen_candidate_ids = [], set()
        for point in raw_points:
            payload = point.payload or {}
            product_id = _clean_id(payload.get("id"))
            if product_id in seen_candidate_ids:
                continue
            if not self._is_valid_candidate(
                payload, lane, excluded_ids, size_preferences,
            ):
                continue
            points.append(point)
            seen_candidate_ids.add(product_id)

        ranked = self._rank(points, profile)
        if len(self._diversify(ranked, limit)) < limit:
            fallback = self._scroll_candidates(
                qfilter,
                lane=lane,
                excluded_product_ids=excluded_ids,
                size_preferences=size_preferences,
                existing_product_ids=seen_candidate_ids,
                target_count=max(_SCROLL_TARGET, limit * 8),
            )
            ranked = self._rank([*points, *fallback], profile)

        selected = self._diversify(ranked, limit)
        serialized = [
            self._serialize_product(
                point, profile, lane, size_preferences,
            )
            for point in selected
        ]
        signal_count = len(weights)
        strategy = (
            "personalized"
            if signal_count >= 2 or bool(saved_signal_ids)
            else "similar_to_recent"
        )
        has_recent_signals = any(product_id in weights for product_id in recent_ids)
        if strategy == "similar_to_recent":
            summary = (
                "Son gezdiğin ürüne benzeyen, hedef kitlesi ve stoku uygun "
                "seçenekler değerlendirildi."
            )
        elif saved_signal_ids and has_recent_signals:
            summary = (
                "Kaydettiğin kombinler ve son gezdiğin ürünlerdeki ortak "
                "tercihler güncel stokla birlikte değerlendirildi."
            )
        elif saved_signal_ids:
            summary = (
                "Kaydettiğin kombinlerdeki ortak tercihler güncel stokla "
                "birlikte değerlendirildi."
            )
        else:
            summary = (
                "Son gezdiğin ürünlerdeki ortak tercihler güncel stokla "
                "birlikte değerlendirildi."
            )
        if not serialized:
            has_size_preference = any(
                _clean_id(value) for value in size_preferences.values()
            )
            summary = (
                "Tercihlerine, hedef kitlene ve hatırlanan bedenine uyan "
                "stoklu yeni bir ürün bulunamadı."
                if has_size_preference
                else "Tercihlerine ve hedef kitlene uyan stoklu yeni bir ürün "
                     "bulunamadı."
            )
        return {
            "personalized": True,
            "strategy": strategy,
            "signal_count": signal_count,
            "summary": summary,
            "count": len(serialized),
            "products": serialized,
        }
