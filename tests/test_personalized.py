from types import SimpleNamespace
import unittest
import uuid

from src.personalized import AudienceLane, PersonalizedRecommender


def product(**overrides):
    base = {
        "id": "SEED",
        "title": "Siyah Basic Tişört",
        "category": "Tişört",
        "category_path": "Kadın > Giyim > Tişört",
        "color": "Siyah",
        "patterns": [],
        "fabrics": ["pamuk"],
        "fits": ["regular fit"],
        "price": 400.0,
        "sale_price": 300.0,
        "gender": "Kadın",
        "age_group": "Yetişkin",
        "availability": "in stock",
        "total_stock": 4,
        "in_stock_sizes": ["M"],
        "weekly_sales": 2,
        "image": "https://example.com/product.jpg",
        "link": "https://example.com/product",
    }
    base.update(overrides)
    return base


class FakeClient:
    def __init__(self, products):
        self.products = {
            str(uuid.uuid5(uuid.NAMESPACE_URL, item["id"])): item
            for item in products
        }
        self.retrieve_calls = []

    def retrieve(self, *, ids, **kwargs):
        self.retrieve_calls.append({"ids": ids, **kwargs})
        return [
            SimpleNamespace(payload=self.products[point_id])
            for point_id in ids
            if point_id in self.products
        ]


class FakeSearchEngine:
    def __init__(self, products, candidates=()):
        self.client = FakeClient(products)
        self.candidates = list(candidates)
        self.hybrid_calls = []
        self.search_calls = 0

    def search(self, *_args, **_kwargs):
        self.search_calls += 1
        raise AssertionError("Kişisel öneri detaylı arama akışını çağırmamalı")

    def _hybrid_query(self, text, qfilter, limit):
        self.hybrid_calls.append((text, qfilter, limit))
        return [
            SimpleNamespace(payload=item, score=0.9 - index / 100)
            for index, item in enumerate(self.candidates[:limit])
        ]


class PersonalizedRulesTest(unittest.TestCase):
    def test_zero_or_missing_signals_return_honest_empty_without_query(self):
        engine = FakeSearchEngine([])
        recommender = PersonalizedRecommender(engine)

        empty = recommender.recommend(limit=8)
        missing = recommender.recommend(
            recent_product_ids=["MISSING"], limit=8)

        for result in (empty, missing):
            self.assertEqual(result["strategy"], "empty")
            self.assertFalse(result["personalized"])
            self.assertEqual(result["signal_count"], 0)
            self.assertEqual(result["products"], [])
        self.assertEqual(len(engine.client.retrieve_calls), 1)
        self.assertEqual(engine.hybrid_calls, [])
        self.assertEqual(engine.search_calls, 0)

    def test_bulk_retrieve_and_hard_candidate_safety(self):
        seed = product()
        older_other_lane = product(
            id="OLDER-MALE", gender="Erkek",
            category_path="Erkek > Giyim > Tişört",
        )
        valid = product(
            id="VALID", title="Siyah Pamuk Gömlek", category="Gömlek",
            category_path="Kadın > Giyim > Gömlek", in_stock_sizes=["S/M"],
        )
        unisex = product(
            id="UNISEX", title="Siyah Unisex Sweatshirt",
            category="Sweatshirt", category_path="Unisex > Giyim > Sweatshirt",
            gender="Unisex", in_stock_sizes=["M"],
        )
        rejected = [
            product(id="MALE", gender="Erkek", category_path="Erkek > Giyim"),
            product(id="CHILD", age_group="Çocuk",
                    category_path="Kız Çocuk > Giyim"),
            product(id="BABY-COHORT", category_path="Kız Bebek > Giyim"),
            product(id="OUT", availability="out of stock"),
            product(id="ZERO", total_stock=0),
            product(id="WRONG-SIZE", in_stock_sizes=["L"]),
            product(id="NO-PATH", category_path=""),
            seed,
        ]
        candidates = [valid, unisex, *rejected]
        engine = FakeSearchEngine(
            [seed, older_other_lane, *candidates], candidates)

        result = PersonalizedRecommender(engine).recommend(
            recent_product_ids=[seed["id"], older_other_lane["id"]],
            size_preferences={"alpha": "M"},
            exclude_product_ids=["EXPLICIT"],
            limit=8,
        )

        self.assertEqual({item["id"] for item in result["products"]}, {
            "VALID", "UNISEX",
        })
        self.assertEqual(result["strategy"], "similar_to_recent")
        self.assertEqual(result["signal_count"], 1)
        self.assertTrue(result["personalized"])
        self.assertEqual(len(engine.client.retrieve_calls), 1)
        self.assertEqual(engine.search_calls, 0)
        self.assertTrue(all(item["reason"] for item in result["products"]))
        qfilter = engine.hybrid_calls[0][1].model_dump(exclude_none=True)
        filter_keys = {
            condition["key"] for condition in qfilter["must"]
        }
        self.assertEqual(
            filter_keys,
            {"availability", "total_stock", "gender", "age_group"},
        )
        self.assertEqual(len(qfilter["must_not"][0]["has_id"]), 3)

    def test_saved_signals_are_stronger_and_make_profile_personalized(self):
        recent = product(id="RECENT", category="Tişört", color="Beyaz")
        saved_source = product(id="SAVED-A", category="Elbise", color="Kırmızı")
        saved_product = product(id="SAVED-B", category="Ceket", color="Kırmızı")
        candidate = product(id="CANDIDATE", title="Kırmızı Elbise",
                            category="Elbise", color="Kırmızı")
        products = [recent, saved_source, saved_product, candidate]
        engine = FakeSearchEngine(products, [candidate])
        recommender = PersonalizedRecommender(engine)

        result = recommender.recommend(
            recent_product_ids=[recent["id"]],
            saved_outfits=[{
                "source_product_id": saved_source["id"],
                "recommended_product_id": saved_product["id"],
            }],
            limit=1,
        )

        self.assertEqual(result["strategy"], "personalized")
        self.assertEqual(result["signal_count"], 3)
        self.assertIn("kırmızı", engine.hybrid_calls[0][0].lower())
        self.assertGreater(
            engine.hybrid_calls[0][0].lower().count("kırmızı"),
            engine.hybrid_calls[0][0].lower().count("beyaz"),
        )
        self.assertIn("kırmızı", result["products"][0]["reason"].lower())
        self.assertIn("kaydettiğin kombinler", result["summary"].lower())
        self.assertIn("son gezdiğin", result["summary"].lower())

    def test_multiple_recent_summary_does_not_claim_saved_outfits(self):
        newest = product(id="NEWEST", category="Tişört")
        older = product(id="OLDER", category="Gömlek")
        candidate = product(id="CANDIDATE", title="Pamuklu Tişört")
        engine = FakeSearchEngine([newest, older, candidate], [candidate])

        result = PersonalizedRecommender(engine).recommend(
            recent_product_ids=[newest["id"], older["id"]], limit=1)

        self.assertEqual(result["strategy"], "personalized")
        self.assertIn("son gezdiğin ürünlerdeki", result["summary"].lower())
        self.assertNotIn("kaydet", result["summary"].lower())

    def test_size_safety_uses_only_recognized_relevant_systems(self):
        adult = AudienceLane("Kadın", "Yetişkin", "adult")
        baby = AudienceLane("Kadın", "Çocuk", "baby")

        cases = [
            (product(in_stock_sizes=["S/M"]), adult, {"alpha": "M"}, True),
            (product(in_stock_sizes=["L"]), adult, {"alpha": "M"}, False),
            (product(in_stock_sizes=["Özel beden"]), adult, {"alpha": "M"}, True),
            (product(in_stock_sizes=["STD"]), adult, {"alpha": "M"}, True),
            (product(category="Yüzük", in_stock_sizes=["12"]),
             adult, {"numeric": "38"}, True),
            (product(category="Sneaker", in_stock_sizes=["38-40"]),
             adult, {"shoe": "39"}, True),
            (product(category="Sneaker", in_stock_sizes=["40"]),
             adult, {"shoe": "39"}, False),
            (product(in_stock_sizes=["Beden 30 - Boy 32"]), adult,
             {"jean_waist": "30", "jean_length": "32"}, True),
            (product(category_path="Kız Bebek > Giyim", age_group="Çocuk",
                     in_stock_sizes=["0-1 Ay (56cm)"]), baby,
             {"child": "month:0"}, True),
            (product(category_path="Kız Bebek > Giyim", age_group="Çocuk",
                     in_stock_sizes=["18-24 Ay"]), baby,
             {"child": "month:0"}, False),
            (product(category="Sneaker", category_path="Kız Çocuk > Ayakkabı",
                     age_group="Çocuk", in_stock_sizes=["29/34"]),
             AudienceLane("Kadın", "Çocuk", "child"), {"shoe": "31"}, True),
            (product(category="Sneaker", category_path="Kız Çocuk > Ayakkabı",
                     age_group="Çocuk", in_stock_sizes=["29/34"]),
             AudienceLane("Kadın", "Çocuk", "child"), {"shoe": "35"}, False),
            (product(category="Çorap", category_path="Kız Çocuk > Giyim",
                     age_group="Çocuk", in_stock_sizes=["4-5 YAŞ"]),
             AudienceLane("Kadın", "Çocuk", "child"),
             {"child": "age:4-5"}, True),
            (product(category_path="Kız Çocuk > Giyim", age_group="Çocuk",
                     in_stock_sizes=["5-6 YAŞ"]),
             AudienceLane("Kadın", "Çocuk", "child"),
             {"child": "age:9-2"}, True),
        ]

        for candidate, lane, preferences, expected in cases:
            with self.subTest(
                category=candidate["category"],
                sizes=candidate["in_stock_sizes"],
            ):
                allowed, _ = PersonalizedRecommender._size_is_safe(
                    candidate, lane, preferences)
                self.assertEqual(allowed, expected)

        self.assertFalse(PersonalizedRecommender._numeric_size_matches(
            "38", "geçersiz"))

    def test_profile_query_order_is_stable_for_equal_weight_features(self):
        first = product(
            id="FIRST", patterns=["çizgili", "baskılı"],
            fabrics=["pamuk", "keten"],
        )
        second = product(
            id="SECOND", patterns=["baskılı", "çiçekli"],
            fabrics=["keten", "saten"],
        )
        products = {item["id"]: item for item in (first, second)}
        weights = {"FIRST": 1.0, "SECOND": 1.0}

        queries = {
            PersonalizedRecommender._build_profile(products, weights).query_text
            for _ in range(10)
        }

        self.assertEqual(len(queries), 1)
        query = queries.pop()
        self.assertLess(query.index("desen baskılı"), query.index("desen çizgili"))
        self.assertLess(query.index("desen çizgili"), query.index("desen çiçekli"))

    def test_title_id_category_and_color_diversity_is_deterministic(self):
        seed = product()
        candidates = [
            product(
                id=f"P{index}",
                title="Aynı Ürün" if index < 2 else f"Ürün {index}",
                category="Tişört" if index < 4 else "Gömlek",
                color="Siyah" if index < 4 else "Beyaz",
            )
            for index in range(6)
        ]
        engine = FakeSearchEngine([seed, *candidates], candidates)

        result = PersonalizedRecommender(engine).recommend(
            recent_product_ids=[seed["id"]], limit=4)

        ids = [item["id"] for item in result["products"]]
        titles = [item["title"] for item in result["products"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(titles), len(set(titles)))
        self.assertEqual(len(ids), 4)


if __name__ == "__main__":
    unittest.main()
