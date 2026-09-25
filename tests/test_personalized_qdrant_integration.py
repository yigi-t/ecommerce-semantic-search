from types import SimpleNamespace
import unittest
import uuid

from qdrant_client import QdrantClient, models

import config
from src.personalized import PersonalizedRecommender


class ScrollBackedSearchEngine:
    def __init__(self, client):
        self.client = client

    def _hybrid_query(self, _text, qfilter, limit):
        records, _ = self.client.scroll(
            collection_name=config.COLLECTION,
            scroll_filter=qfilter,
            limit=limit,
            with_payload=True,
        )
        return [SimpleNamespace(payload=record.payload, score=0.5) for record in records]


class FixedHybridSearchEngine:
    def __init__(self, client, hybrid_products):
        self.client = client
        self.hybrid_products = hybrid_products

    def _hybrid_query(self, _text, _qfilter, limit):
        return [
            SimpleNamespace(payload=product, score=0.5)
            for product in self.hybrid_products[:limit]
        ]


class PersonalizedQdrantIntegrationTest(unittest.TestCase):
    @staticmethod
    def _client_with(products):
        client = QdrantClient(":memory:")
        client.create_collection(
            collection_name=config.COLLECTION,
            vectors_config=models.VectorParams(
                size=1, distance=models.Distance.COSINE,
            ),
        )
        client.upsert(
            collection_name=config.COLLECTION,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, item["id"])),
                    vector=[1.0],
                    payload=item,
                )
                for item in products
            ],
        )
        return client

    @staticmethod
    def _adult_product(**overrides):
        product = {
            "id": "SEED", "title": "Beyaz Basic Tişört",
            "category": "Tişört", "category_path": "Kadın > Giyim > Tişört",
            "color": "Beyaz", "patterns": [], "fabrics": ["pamuk"],
            "fits": ["regular fit"], "price": 400.0, "sale_price": 300.0,
            "gender": "Kadın", "age_group": "Yetişkin",
            "availability": "in stock", "total_stock": 5,
            "in_stock_sizes": ["M"], "weekly_sales": 2,
            "image": "https://example.com/image.jpg",
            "link": "https://example.com/product",
        }
        product.update(overrides)
        return product

    def test_real_qdrant_hard_filter_and_post_filter_safety(self):
        seed = self._adult_product()
        valid = self._adult_product(
            id="VALID", title="Siyah Pamuk Gömlek", category="Gömlek",
            category_path="Kadın > Giyim > Gömlek", color="Siyah",
            in_stock_sizes=["S/M"],
        )
        unisex = self._adult_product(
            id="UNISEX", title="Unisex Sweatshirt", category="Sweatshirt",
            category_path="Unisex > Giyim > Sweatshirt", gender="Unisex",
        )
        rejected = [
            self._adult_product(
                id="MALE", gender="Erkek", category_path="Erkek > Giyim"),
            self._adult_product(
                id="CHILD", age_group="Çocuk",
                category_path="Kız Çocuk > Giyim"),
            self._adult_product(
                id="BABY-COHORT", category_path="Kız Bebek > Giyim"),
            self._adult_product(id="OUT", availability="out of stock"),
            self._adult_product(id="ZERO", total_stock=0),
            self._adult_product(id="WRONG-SIZE", in_stock_sizes=["L"]),
            self._adult_product(id="EXPLICIT"),
        ]
        client = self._client_with([seed, valid, unisex, *rejected])

        result = PersonalizedRecommender(
            ScrollBackedSearchEngine(client)
        ).recommend(
            recent_product_ids=[seed["id"]],
            size_preferences={"alpha": "M"},
            exclude_product_ids=["EXPLICIT"],
            limit=8,
        )

        self.assertEqual(
            {item["id"] for item in result["products"]},
            {"VALID", "UNISEX"},
        )
        self.assertTrue(all(item["sizes"] for item in result["products"]))

    def test_baby_post_filter_shortfall_uses_bounded_scroll_without_duplicates(self):
        seed = {
            "id": "BABY-SEED", "title": "Beyaz Tişört Kız Bebek",
            "category": "Tişört",
            "category_path": "Kız Bebek > Giyim > Tişört",
            "color": "Beyaz", "patterns": [], "fabrics": ["pamuk"],
            "fits": [], "price": 250.0, "sale_price": 200.0,
            "gender": "Kadın", "age_group": "Çocuk",
            "availability": "in stock", "total_stock": 2,
            "in_stock_sizes": ["0 AY"], "weekly_sales": 1,
            "image": "https://example.com/image.jpg",
            "link": "https://example.com/product",
        }
        first_valid = {
            **seed, "id": "FIRST-VALID", "title": "Siyah Bebek Pantolon",
            "category": "Pantolon",
            "category_path": "Kız Bebek > Giyim > Pantolon",
            "color": "Siyah", "in_stock_sizes": ["0-1 Ay (56cm)"],
        }
        wrong_sizes = [
            {
                **first_valid,
                "id": f"WRONG-{index}",
                "title": f"Gri Bebek Pantolon {index}",
                "color": "Gri",
                "in_stock_sizes": ["18-24 Ay (92cm)"],
            }
            for index in range(180)
        ]
        wrong_cohort = {
            **first_valid,
            "id": "WRONG-CHILD",
            "title": "Siyah Çocuk Pantolon",
            "category_path": "Kız Çocuk > Giyim > Pantolon",
        }
        fallback_valid = {
            **first_valid,
            "id": "FALLBACK-VALID",
            "title": "Lacivert Bebek Jogger",
            "color": "Lacivert",
        }
        client = self._client_with([
            seed, first_valid, *wrong_sizes, wrong_cohort, fallback_valid,
        ])
        hybrid_products = [
            first_valid, wrong_cohort, *wrong_sizes[:158],
        ]

        result = PersonalizedRecommender(
            FixedHybridSearchEngine(client, hybrid_products)
        ).recommend(
            recent_product_ids=[seed["id"]],
            size_preferences={"child": "month:0"},
            limit=2,
        )

        ids = [item["id"] for item in result["products"]]
        self.assertEqual(set(ids), {"FIRST-VALID", "FALLBACK-VALID"})
        self.assertEqual(len(ids), len(set(ids)))
        self.assertNotIn("WRONG-CHILD", ids)


if __name__ == "__main__":
    unittest.main()
