from types import SimpleNamespace
import unittest
import uuid

from src.query_parser import parse_query
from src.personalized import PersonalizedRecommender
from src.search import SearchEngine


EXPECTED_QUERY_FILTERS = {
    "puantiyeli siyah elbise": {
        "category": "Elbise", "color": "Siyah", "patterns": ["puantiye"],
    },
    "500 tl altı keten gömlek": {
        "category": "Gömlek", "fabrics": ["keten"], "price_max": 500.0,
    },
    "oversize siyah sweatshirt kadın": {
        "category": "Sweatshirt", "color": "Siyah", "gender": "Kadın",
        "fits": ["oversize"],
    },
    "erkek slim fit jean": {
        "category": "Jean", "gender": "Erkek", "age_group": "Yetişkin",
        "fabrics": ["denim"], "fits": ["slim fit"],
    },
    "çiçekli midi etek": {
        "category": "Etek", "patterns": ["çiçekli"], "lengths": ["midi"],
    },
    "v yaka triko kazak bordo": {
        "category": "Kazak", "color": "Bordo", "fabrics": ["triko"],
        "necklines": ["v yaka"],
    },
    "puantiyeli kırmızı elbise": {
        "category": "Elbise", "color": "Kırmızı", "patterns": ["puantiye"],
    },
}


def point(index):
    return SimpleNamespace(
        score=0.5 - index / 100,
        payload={
            "id": f"P{index}",
            "title": f"Ürün {index}",
            "category": "Elbise",
            "color": "Siyah",
            "patterns": ["puantiye"],
            "price": 499.99,
            "sale_price": 399.99,
            "gender": "Kadın",
            "age_group": "Yetişkin",
            "in_stock_sizes": ["M"],
            "image": "https://example.com/image.jpg",
            "link": "https://example.com/product",
        },
    )


class SearchRegressionTest(unittest.TestCase):
    def test_documented_query_parser_contract_is_unchanged(self):
        for query, expected in EXPECTED_QUERY_FILTERS.items():
            with self.subTest(query=query):
                self.assertEqual(parse_query(query).active_filters(), expected)

    def test_exact_search_keeps_existing_result_shape_and_order(self):
        engine = SearchEngine.__new__(SearchEngine)
        calls = []
        expected_points = [point(index) for index in range(5)]

        def fake_hybrid(text, qfilter, limit):
            calls.append((text, qfilter, limit))
            return expected_points

        engine._hybrid_query = fake_hybrid
        result = engine.search("puantiyeli siyah elbise", limit=24)

        self.assertEqual(result.relaxed, [])
        self.assertEqual([item["id"] for item in result.products], [f"P{i}" for i in range(5)])
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            set(result.products[0]),
            {"id", "title", "category", "color", "patterns", "price",
             "sale_price", "gender", "age_group", "sizes", "image", "link", "score"},
        )

    def test_existing_relaxation_order_and_category_filter_are_preserved(self):
        engine = SearchEngine.__new__(SearchEngine)
        calls = []
        responses = [[], [point(index) for index in range(5)]]

        def fake_hybrid(text, qfilter, limit):
            calls.append(qfilter.model_dump(exclude_none=True))
            return responses[len(calls) - 1]

        engine._hybrid_query = fake_hybrid
        result = engine.search("puantiyeli kırmızı elbise", limit=24)

        # Mevcut kod patterns'i color'dan önce gevşetiyor; kombin özelliği bu
        # davranışı bilerek değiştirmiyor.
        self.assertEqual(result.relaxed, ["patterns"])
        self.assertEqual(len(calls), 2)
        for qfilter in calls:
            values = {
                condition["key"]: condition.get("match", {}).get("value")
                for condition in qfilter["must"]
            }
            self.assertEqual(values["category"], "Elbise")
            self.assertEqual(values["color"], "Kırmızı")
        self.assertTrue(any(condition["key"] == "patterns" for condition in calls[0]["must"]))
        self.assertFalse(any(condition["key"] == "patterns" for condition in calls[1]["must"]))

    def test_personalized_recommender_never_calls_search_entrypoint(self):
        seed = {
            **point(0).payload,
            "id": "SEED", "title": "Siyah Elbise",
            "category_path": "Kadın > Giyim > Elbise",
            "fabrics": ["pamuk"], "fits": ["regular fit"],
            "availability": "in stock", "total_stock": 2,
            "weekly_sales": 1,
        }
        candidate = {
            **seed,
            "id": "CANDIDATE", "title": "Siyah Pamuklu Elbise",
        }

        class Client:
            def retrieve(self, *, ids, **_kwargs):
                seed_id = str(uuid.uuid5(uuid.NAMESPACE_URL, seed["id"]))
                return [SimpleNamespace(payload=seed)] if seed_id in ids else []

        engine = SearchEngine.__new__(SearchEngine)
        engine.client = Client()
        search_calls = []

        def forbidden_search(*args, **kwargs):
            search_calls.append((args, kwargs))
            raise AssertionError("Detaylı arama çağrılmamalı")

        engine.search = forbidden_search
        engine._hybrid_query = lambda *_args: [
            SimpleNamespace(payload=candidate, score=0.5)
        ]

        result = PersonalizedRecommender(engine).recommend(
            recent_product_ids=[seed["id"]], limit=1)

        self.assertEqual([item["id"] for item in result["products"]], ["CANDIDATE"])
        self.assertEqual(search_calls, [])


if __name__ == "__main__":
    unittest.main()
