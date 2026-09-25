# -*- coding: utf-8 -*-
"""
Arama çekirdeği: sorgu anlama → hybrid (dense+BM25, RRF) → kademeli gevşetme.

Akış:
  1. parse_query: "puantiyeli kırmızı elbise" → yapılandırılmış niyet
  2. Niyetten Qdrant filtresi kurulur (must koşulları)
  3. Hybrid arama: dense + sparse prefetch, RRF füzyonu
  4. Sonuç MIN_RESULTS_BEFORE_RELAX altındaysa filtreler kademeli
     gevşetilir (önce desen, sonra renk, sonra diğerleri) — gevşetilen
     kısıt anlamsal skora "soft boost" olarak geri verilir. Böylece
     "kırmızı puantiyeli elbise" katalogda hiç yoksa kullanıcı boş sayfa
     yerine en yakın alternatifleri (siyah puantiyeli, düz kırmızı) görür
     ve yanıt hangi kısıtın gevşetildiğini raporlar.
"""

from dataclasses import dataclass, field

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

import config
from .query_parser import QueryIntent, parse_query

# Gevşetme sırası: en "opsiyonel" kısıttan en temel olana.
# Kategori ve cinsiyet en son gevşer — "elbise" arayan birine
# asla pantolon göstermeyiz; kategori pratikte hiç gevşetilmez.
RELAX_ORDER = ["lengths", "sleeves", "necklines", "fits",
               "fabrics", "patterns", "color", "age_group"]


@dataclass
class SearchResult:
    products: list
    applied_filters: dict
    relaxed: list = field(default_factory=list)
    intent: QueryIntent | None = None
    facets: dict = field(default_factory=dict)


class SearchEngine:
    def __init__(self):
        self.client = QdrantClient(url=config.QDRANT_URL,
                                   api_key=config.QDRANT_API_KEY)
        self.dense_model = TextEmbedding(
            config.DENSE_MODEL, cache_dir=config.FASTEMBED_CACHE_DIR
        )
        self.sparse_model = SparseTextEmbedding(
            config.SPARSE_MODEL, cache_dir=config.FASTEMBED_CACHE_DIR
        )

    # ------------------------------------------------------------------
    def _build_filter(self, intent: QueryIntent, exclude: set,
                      f: dict | None = None) -> models.Filter | None:
        must = [models.FieldCondition(key="availability",
                                      match=models.MatchValue(value="in stock"))]

        def kw(key, value):
            must.append(models.FieldCondition(
                key=key, match=models.MatchValue(value=value)))

        if intent.category and "category" not in exclude:
            kw("category", intent.category)
        if intent.color and "color" not in exclude:
            kw("color", intent.color)
        if intent.gender and "gender" not in exclude:
            kw("gender", intent.gender)
        if intent.age_group and "age_group" not in exclude:
            kw("age_group", intent.age_group)

        for key in ("patterns", "fabrics", "fits",
                    "necklines", "sleeves", "lengths"):
            vals = getattr(intent, key)
            if vals and key not in exclude:
                # Liste alanı: herhangi biri eşleşsin
                must.append(models.FieldCondition(
                    key=key, match=models.MatchAny(any=vals)))

        if intent.price_min is not None or intent.price_max is not None:
            must.append(models.FieldCondition(
                key="sale_price",
                range=models.Range(gte=intent.price_min, lte=intent.price_max)))

        # Manuel filtreler (renk/kategori/kitle/desen/fiyat) da aramaya girer;
        # böylece "erkek" arayıp "Tişört" seçince sonuçlar "erkek tişört" gibi gelir.
        if f:
            must.extend(self._manual_must(f))
        return models.Filter(must=must) if must else None

    # ------------------------------------------------------------------
    def _hybrid_query(self, text: str, qfilter, limit: int):
        dense_vec = next(self.dense_model.embed(["query: " + text])).tolist()
        sparse_raw = next(self.sparse_model.embed([text]))
        sparse_vec = models.SparseVector(indices=sparse_raw.indices.tolist(),
                                         values=sparse_raw.values.tolist())

        return self.client.query_points(
            collection_name=config.COLLECTION,
            prefetch=[
                models.Prefetch(query=dense_vec, using="dense",
                                filter=qfilter, limit=config.PREFETCH_LIMIT),
                models.Prefetch(query=sparse_vec, using="bm25",
                                filter=qfilter, limit=config.PREFETCH_LIMIT),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        ).points

    # ------------------------------------------------------------------
    def search(self, q: str, limit: int = 24, filters: dict | None = None) -> SearchResult:
        intent = parse_query(q)
        exclude: set = set()
        relaxed: list = []

        # Manuel filtreler (Filtre paneli) ve/veya "indirimli" istendiğinde daha
        # geniş aday havuzu çek; bu filtreler Qdrant tarafında değil, sonuçlar
        # limite kırpılmadan hemen önce uygulama tarafında uygulanır. Hiçbiri yoksa
        # mevcut davranış birebir korunur (havuz da, _build_filter de değişmez).
        f = dict(filters or {})
        # "indirimli" sorgusu da manuel "Sadece indirimli" gibi davranır.
        f["discount"] = intent.discounted or bool(f.get("discount"))
        narrowed = self._has_manual_filters(f)
        facets = self._compute_facets(intent, f)

        # Filtre-only "gözat": sorgu metni yoksa ama filtre varsa, semantik
        # sıralama yerine doğrudan filtreye uyan ürünleri Qdrant'tan tara.
        # Metin varsa mevcut hybrid + gevşetme + post-filter yolu birebir korunur.
        if narrowed and not intent.semantic_text.strip():
            records = self._scroll_by_filters(f, config.PREFETCH_LIMIT)
            records = [r for r in records if self._passes_manual_filters(r.payload, f)]
            products = [self._to_product(r) for r in records[:limit]]
            return SearchResult(products=products,
                                applied_filters=intent.active_filters(),
                                relaxed=[], intent=intent, facets=facets)

        fetch_limit = config.PREFETCH_LIMIT if narrowed else limit

        points = self._hybrid_query(intent.semantic_text,
                                    self._build_filter(intent, exclude, f), fetch_limit)

        # Kademeli gevşetme
        for key in RELAX_ORDER:
            if len(points) >= config.MIN_RESULTS_BEFORE_RELAX:
                break
            if not getattr(intent, key):
                continue
            exclude.add(key)
            relaxed.append(key)
            points = self._hybrid_query(intent.semantic_text,
                                        self._build_filter(intent, exclude, f),
                                        fetch_limit)

        # Gevşetilmiş kısıtları soft-boost olarak geri ver: payload'ında
        # gevşetilen niteliği hâlâ taşıyan ürünler öne gelsin.
        if relaxed and points:
            def boost(pt):
                score = 0
                for key in relaxed:
                    want = getattr(intent, key)
                    have = pt.payload.get(key)
                    if not want:
                        continue
                    if isinstance(want, list):
                        if set(want) & set(have or []):
                            score += 1
                    elif have == want:
                        score += 1
                return score
            points.sort(key=boost, reverse=True)

        # Manuel filtreler + indirim: sonuçları daralt, sonra istenen limite kırp.
        # Hiçbir filtre yoksa (narrowed False) bu blok atlanır; davranış aynı.
        if narrowed:
            points = [pt for pt in points
                      if self._passes_manual_filters(pt.payload, f)][:limit]

        products = [self._to_product(pt) for pt in points]
        return SearchResult(products=products,
                            applied_filters=intent.active_filters(),
                            relaxed=relaxed, intent=intent, facets=facets)

    @staticmethod
    def _is_discounted(payload) -> bool:
        """sale_price geçerli ve normal fiyatın altında mı?"""
        try:
            price = float(payload.get("price"))
            sale = float(payload.get("sale_price"))
        except (TypeError, ValueError):
            return False
        return 0 < sale < price

    @staticmethod
    def _has_manual_filters(f) -> bool:
        if any(f.get(k) for k in
               ("color", "category", "gender", "age_group", "audience", "patterns")):
            return True
        if f.get("price_min") is not None or f.get("price_max") is not None:
            return True
        return bool(f.get("discount"))

    @staticmethod
    def _passes_manual_filters(payload, f) -> bool:
        """Manuel filtre paneli seçimlerini uygulama tarafında uygular."""
        if f.get("color") and payload.get("color") not in f["color"]:
            return False
        if f.get("category") and payload.get("category") not in f["category"]:
            return False
        if f.get("gender") and payload.get("gender") not in f["gender"]:
            return False
        if f.get("age_group") and payload.get("age_group") not in f["age_group"]:
            return False
        audience = f.get("audience")
        if audience and not any(
                (g is None or payload.get("gender") == g)
                and (a is None or payload.get("age_group") == a)
                for g, a in audience):
            return False
        if f.get("patterns") and not (set(payload.get("patterns") or []) & f["patterns"]):
            return False
        price_min = f.get("price_min")
        price_max = f.get("price_max")
        if price_min is not None or price_max is not None:
            value = payload.get("sale_price") or payload.get("price")
            try:
                value = float(value)
            except (TypeError, ValueError):
                return False
            if price_min is not None and value < price_min:
                return False
            if price_max is not None and value > price_max:
                return False
        if f.get("discount") and not SearchEngine._is_discounted(payload):
            return False
        return True

    @staticmethod
    def _manual_must(f) -> list:
        """Manuel filtrelerin Qdrant koşulları (indirim hariç; o post-filtredir)."""
        conditions = []
        if f.get("color"):
            conditions.append(models.FieldCondition(
                key="color", match=models.MatchAny(any=sorted(f["color"]))))
        if f.get("category"):
            conditions.append(models.FieldCondition(
                key="category", match=models.MatchAny(any=sorted(f["category"]))))
        if f.get("gender"):
            conditions.append(models.FieldCondition(
                key="gender", match=models.MatchAny(any=sorted(f["gender"]))))
        if f.get("age_group"):
            conditions.append(models.FieldCondition(
                key="age_group", match=models.MatchAny(any=sorted(f["age_group"]))))
        if f.get("audience"):
            should = []
            for g, a in f["audience"]:
                sub = []
                if g is not None:
                    sub.append(models.FieldCondition(
                        key="gender", match=models.MatchValue(value=g)))
                if a is not None:
                    sub.append(models.FieldCondition(
                        key="age_group", match=models.MatchValue(value=a)))
                if sub:
                    should.append(models.Filter(must=sub))
            if should:
                conditions.append(models.Filter(should=should))
        if f.get("patterns"):
            conditions.append(models.FieldCondition(
                key="patterns", match=models.MatchAny(any=sorted(f["patterns"]))))
        if f.get("price_min") is not None or f.get("price_max") is not None:
            conditions.append(models.FieldCondition(
                key="sale_price",
                range=models.Range(gte=f.get("price_min"), lte=f.get("price_max"))))
        return conditions

    def _scroll_by_filters(self, f, limit):
        """Sorgu metni olmadan yalnız filtrelere uyan ürünleri Qdrant'tan tarar."""
        scroll = getattr(self.client, "scroll", None)
        if not callable(scroll):
            return []
        must = [models.FieldCondition(
            key="availability", match=models.MatchValue(value="in stock"))]
        must.extend(self._manual_must(f))
        records, _ = scroll(
            collection_name=config.COLLECTION,
            scroll_filter=models.Filter(must=must),
            limit=limit,
            with_payload=True,
        )
        return records

    @staticmethod
    def _facet_hits(facet_fn, key, qfilter, limit):
        try:
            return facet_fn(collection_name=config.COLLECTION, key=key,
                            facet_filter=qfilter, limit=limit).hits
        except Exception:
            return []

    def _compute_facets(self, intent, f) -> dict:
        """Gerçek katalog üzerinden facet: kategori varlığı + renk sayıları.
        Her facet kendi filtresini hariç tutar (standart faceted arama). Facet
        API/istemci yoksa boş döner; ön yüz eski (sonuç tabanlı) yola düşer."""
        client = getattr(self, "client", None)
        facet_fn = getattr(client, "facet", None) if client is not None else None
        if not callable(facet_fn):
            return {}
        # Sorgudan gelen (intent) kısıt facet'e DAHİL edilir -> "tişört" ya da
        # "mavi çizgili tişört" gibi sorgular ilgili tek değere daralır (disabled).
        # Manuel seçim ise HARİÇ tutulur -> kullanıcı başka değer ekleyip
        # çıkarabilir (renk çoklu seçimi ve sayıların kalıcılığı korunur).
        facets = {}
        f_no_cat = {k: v for k, v in f.items() if k != "category"}
        cat_filter = self._build_filter(intent, set(), f_no_cat)
        facets["category"] = [
            h.value for h in self._facet_hits(facet_fn, "category", cat_filter, 200)]
        f_no_color = {k: v for k, v in f.items() if k != "color"}
        color_filter = self._build_filter(intent, set(), f_no_color)
        facets["color"] = {
            h.value: h.count
            for h in self._facet_hits(facet_fn, "color", color_filter, 60)}
        has_category = bool(intent.category or f.get("category"))
        has_gender = bool(intent.gender or f.get("gender") or f.get("audience"))
        facets["show_color_counts"] = has_category and has_gender
        return facets

    @staticmethod
    def _to_product(pt) -> dict:
        p = pt.payload
        score = getattr(pt, "score", None)
        return {
            "id": p.get("id"),
            "title": p.get("title"),
            "category": p.get("category"),
            "color": p.get("color"),
            "patterns": p.get("patterns", []),
            "price": p.get("price"),
            "sale_price": p.get("sale_price"),
            "gender": p.get("gender"),
            "age_group": p.get("age_group"),
            "sizes": p.get("in_stock_sizes", []),
            "image": p.get("image"),
            "link": p.get("link"),
            "score": round(score, 4) if score is not None else None,
        }
