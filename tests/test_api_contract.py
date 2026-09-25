from types import SimpleNamespace
import unittest
import warnings

from starlette.exceptions import StarletteDeprecationWarning

with warnings.catch_warnings():
    warnings.simplefilter("ignore", StarletteDeprecationWarning)
    from fastapi.testclient import TestClient

from src import api


class FakeSearchEngine:
    def search(self, query, limit):
        return SimpleNamespace(
            applied_filters={"category": "Tişört"},
            relaxed=[],
            products=[{"id": "P1", "title": "Beyaz Tişört"}],
        )


class FakeOutfitRecommender:
    def __init__(self):
        self.recommend_calls = []
        self.replacement_calls = []

    def recommend(self, product_id, limit, preferred_sizes=None):
        self.recommend_calls.append({
            "product_id": product_id,
            "limit": limit,
            "preferred_sizes": preferred_sizes,
        })
        if product_id == "missing":
            return None
        return {
            "source": {"id": product_id},
            "eligible": True,
            "title": "Bu üstü tamamlayan alt giyim önerileri",
            "summary": "Renk ve ürün türü değerlendirildi.",
            "target_categories": ["Pantolon"],
            "count": 1,
            "products": [{"id": "P2", "reason": "Renk dengesi uygundur."}],
        }

    def recommend_replacement(
        self, product_id, current_product_id, *, preference,
        excluded_product_ids, limit, preferred_sizes=None,
    ):
        self.replacement_calls.append({
            "product_id": product_id,
            "current_product_id": current_product_id,
            "preference": preference,
            "excluded_product_ids": excluded_product_ids,
            "limit": limit,
            "preferred_sizes": preferred_sizes,
        })
        if product_id == "missing" or current_product_id == "missing":
            return None
        return {
            "source": {"id": product_id},
            "eligible": True,
            "title": "Düzenlenen kombin",
            "summary": "Alternatif değerlendirildi.",
            "target_categories": ["Pantolon"],
            "count": 1,
            "products": [{"id": "P3", "reason": "Yeni alternatif."}],
        }


class FakePersonalizedRecommender:
    def __init__(self):
        self.calls = []

    def recommend(self, **kwargs):
        self.calls.append(kwargs)
        signal_ids = kwargs["recent_product_ids"]
        return {
            "personalized": bool(signal_ids),
            "strategy": "similar_to_recent" if signal_ids else "empty",
            "signal_count": len(signal_ids),
            "summary": "Kişisel öneri özeti.",
            "count": 1 if signal_ids else 0,
            "products": (
                [{"id": "P9", "title": "Önerilen ürün", "reason": "Benzer."}]
                if signal_ids else []
            ),
        }


class ApiContractTest(unittest.TestCase):
    def setUp(self):
        self.old_engine = api.engine
        self.old_recommender = api.outfit_recommender
        self.old_personalized_recommender = api.personalized_recommender
        api.engine = FakeSearchEngine()
        self.fake_recommender = FakeOutfitRecommender()
        api.outfit_recommender = self.fake_recommender
        self.fake_personalized_recommender = FakePersonalizedRecommender()
        api.personalized_recommender = self.fake_personalized_recommender
        self.client = TestClient(api.app)

    def tearDown(self):
        api.engine = self.old_engine
        api.outfit_recommender = self.old_recommender
        api.personalized_recommender = self.old_personalized_recommender

    def test_search_top_level_contract_is_exactly_preserved(self):
        response = api.search(q="beyaz tişört", limit=24)
        self.assertEqual(
            set(response),
            {"query", "understood", "relaxed", "count", "products", "facets"})
        self.assertEqual(response["query"], "beyaz tişört")
        self.assertEqual(response["count"], 1)

    def test_outfit_is_a_separate_endpoint(self):
        response = api.outfit_recommendations(product_id="P1", limit=4)
        self.assertEqual(response["source"]["id"], "P1")
        self.assertEqual(response["count"], 1)
        self.assertTrue(response["eligible"])

    def test_outfit_http_validation_and_not_found_semantics(self):
        self.assertEqual(
            self.client.get("/products/missing/outfit-recommendations").status_code,
            404,
        )
        self.assertEqual(
            self.client.get("/products/P1/outfit-recommendations?limit=0").status_code,
            422,
        )
        self.assertEqual(
            self.client.get("/products/P1/outfit-recommendations?limit=9").status_code,
            422,
        )

    def test_outfit_replacement_preferences_are_separate_and_stateless(self):
        response = self.client.get(
            "/products/P1/outfit-recommendations"
            "?limit=1&preference=cheaper&current_product_id=P2"
            "&exclude_product_id=P2&exclude_product_id=P4"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["products"][0]["id"], "P3")
        self.assertEqual(self.fake_recommender.replacement_calls, [{
            "product_id": "P1",
            "current_product_id": "P2",
            "preference": "cheaper",
            "excluded_product_ids": {"P2", "P4"},
            "limit": 1,
            "preferred_sizes": set(),
        }])

    def test_remembered_sizes_are_optional_repeated_and_forwarded(self):
        response = self.client.get(
            "/products/P1/outfit-recommendations"
            "?preferred_size=M&preferred_size=S%2FM"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.fake_recommender.recommend_calls[-1], {
            "product_id": "P1",
            "limit": 4,
            "preferred_sizes": {"M", "S/M"},
        })

        response = self.client.get(
            "/products/P1/outfit-recommendations"
            "?limit=1&preference=alternative&current_product_id=P2"
            "&preferred_size=38"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            self.fake_recommender.replacement_calls[-1]["preferred_sizes"],
            {"38"},
        )

        too_many = "&".join(
            f"preferred_size=size-{index}" for index in range(17)
        )
        self.assertEqual(
            self.client.get(
                f"/products/P1/outfit-recommendations?{too_many}"
            ).status_code,
            422,
        )

    def test_outfit_replacement_http_validation(self):
        self.assertEqual(
            self.client.get(
                "/products/P1/outfit-recommendations?preference=cheaper"
            ).status_code,
            422,
        )
        self.assertEqual(
            self.client.get(
                "/products/P1/outfit-recommendations"
                "?preference=unknown&current_product_id=P2"
            ).status_code,
            422,
        )
        allowed = "&".join(
            f"exclude_product_id=P{index}" for index in range(100))
        self.assertEqual(
            self.client.get(
                "/products/P1/outfit-recommendations"
                f"?preference=alternative&current_product_id=P2&{allowed}"
            ).status_code,
            200,
        )
        too_many = "&".join(
            f"exclude_product_id=P{index}" for index in range(101))
        self.assertEqual(
            self.client.get(
                "/products/P1/outfit-recommendations"
                f"?preference=alternative&current_product_id=P2&{too_many}"
            ).status_code,
            422,
        )

    def test_search_http_validation_contract(self):
        self.assertEqual(self.client.get("/search?q=&limit=24").status_code, 422)
        self.assertEqual(self.client.get("/search?q=test&limit=101").status_code, 422)

    def test_personalized_endpoint_forwards_only_validated_id_signals(self):
        response = self.client.post("/personalized-recommendations", json={
            "recent_product_ids": [" P1 ", "P2"],
            "saved_outfits": [{
                "source_product_id": "P3",
                "recommended_product_id": "P4",
            }],
            "size_preferences": {
                "alpha": "M",
                "numeric": "",
                "shoe": "39",
                "jean_waist": "30",
                "jean_length": "32",
                "child": None,
            },
            "exclude_product_ids": [" P8 "],
            "limit": 8,
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            set(response.json()),
            {"personalized", "strategy", "signal_count", "summary", "count",
             "products"},
        )
        self.assertEqual(self.fake_personalized_recommender.calls, [{
            "recent_product_ids": ["P1", "P2"],
            "saved_outfits": [{
                "source_product_id": "P3",
                "recommended_product_id": "P4",
            }],
            "size_preferences": {
                "alpha": "M", "shoe": "39",
                "jean_waist": "30", "jean_length": "32",
            },
            "exclude_product_ids": ["P8"],
            "limit": 8,
        }])

    def test_personalized_empty_request_and_unavailable_service_contract(self):
        response = self.client.post("/personalized-recommendations", json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["strategy"], "empty")
        self.assertFalse(response.json()["personalized"])

        api.personalized_recommender = None
        unavailable = self.client.post("/personalized-recommendations", json={})
        self.assertEqual(unavailable.status_code, 503)

    def test_personalized_request_validation_limits_and_formats(self):
        invalid_payloads = [
            {"recent_product_ids": [f"P{i}" for i in range(11)]},
            {"saved_outfits": [
                {"source_product_id": f"S{i}", "recommended_product_id": f"R{i}"}
                for i in range(21)
            ]},
            {"exclude_product_ids": [f"P{i}" for i in range(101)]},
            {"recent_product_ids": [" "]},
            {"recent_product_ids": ["P" * 101]},
            {"limit": 0},
            {"limit": 25},
            {"size_preferences": {"alpha": "INVALID"}},
            {"size_preferences": {"child": "0-1 Ay"}},
            {"size_preferences": {"child": "age:9-2"}},
            {"size_preferences": {"jean_waist": "30"}},
            {"unknown": True},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                response = self.client.post(
                    "/personalized-recommendations", json=payload)
                self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
