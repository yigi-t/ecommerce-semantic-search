from types import SimpleNamespace
import unittest
import uuid

from qdrant_client import QdrantClient, models

import config
from src.outfit import OutfitRecommender


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

    def _hybrid_query(self, _text, _qfilter, _limit):
        return [
            SimpleNamespace(payload=product, score=0.5)
            for product in self.hybrid_products
        ]


class OutfitQdrantIntegrationTest(unittest.TestCase):
    @staticmethod
    def _client_with(products):
        client = QdrantClient(":memory:")
        client.create_collection(
            collection_name=config.COLLECTION,
            vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
        )
        for item in products:
            client.upsert(
                collection_name=config.COLLECTION,
                points=[models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, item["id"])),
                    vector=[1.0],
                    payload=item,
                )],
            )
        return client

    def test_uuid_retrieve_and_real_qdrant_filter_contract(self):
        source = {
            "id": "SRC25SP", "title": "Kırmızı Baskılı Tişört",
            "category": "Tişört", "category_path": "Kadın > Giyim > Tişört",
            "color": "Kırmızı", "gender": "Kadın", "age_group": "Yetişkin",
            "patterns": ["baskılı"], "price": 399.99, "sale_price": 299.99,
            "availability": "in stock", "total_stock": 5,
            "in_stock_sizes": ["S"], "weekly_sales": 1,
        }
        candidates = [
            {
                **source, "id": "VALID", "title": "Siyah Basic Pantolon",
                "category": "Pantolon", "category_path": "Kadın > Giyim > Pantolon",
                "color": "Siyah", "total_stock": 3,
            },
            {
                **source, "id": "NO-STOCK", "title": "Bej Pantolon",
                "category": "Pantolon", "category_path": "Kadın > Giyim > Pantolon",
                "color": "Bej", "total_stock": 0,
            },
            {
                **source, "id": "WRONG-GENDER", "title": "Erkek Pantolon",
                "category": "Pantolon", "category_path": "Erkek > Giyim > Pantolon",
                "color": "Siyah", "gender": "Erkek",
            },
            {
                **source, "id": "WRONG-CATEGORY", "title": "Siyah Tişört",
                "category": "Tişört", "category_path": "Kadın > Giyim > Tişört",
                "color": "Siyah",
            },
        ]

        client = self._client_with([source, *candidates])
        recommender = OutfitRecommender(ScrollBackedSearchEngine(client))
        result = recommender.recommend(source["id"], limit=4)

        self.assertEqual([item["id"] for item in result["products"]], ["VALID"])

    def test_remembered_adult_sizes_filter_real_qdrant_array_payloads(self):
        source = {
            "id": "SIZE-SOURCE", "title": "Beyaz Tişört",
            "category": "Tişört", "category_path": "Kadın > Giyim > Tişört",
            "color": "Beyaz", "gender": "Kadın", "age_group": "Yetişkin",
            "patterns": [], "price": 399.99, "sale_price": 299.99,
            "availability": "in stock", "total_stock": 5,
            "in_stock_sizes": ["S", "M"], "weekly_sales": 1,
        }
        matching = {
            **source, "id": "SIZE-M", "title": "Siyah Pantolon",
            "category": "Pantolon", "category_path": "Kadın > Giyim > Pantolon",
            "color": "Siyah", "in_stock_sizes": ["M"],
        }
        wrong = {
            **matching, "id": "SIZE-L", "title": "Bej Pantolon",
            "color": "Bej", "in_stock_sizes": ["L"],
        }
        standard = {
            **matching, "id": "SIZE-STD", "title": "Gri Pantolon",
            "color": "Gri", "in_stock_sizes": ["STD"],
        }
        client = self._client_with([source, matching, wrong, standard])

        result = OutfitRecommender(
            ScrollBackedSearchEngine(client)
        ).recommend(source["id"], limit=4, preferred_sizes={"M"})

        self.assertEqual(
            {item["id"] for item in result["products"]},
            {"SIZE-M", "SIZE-STD"},
        )

    def test_clothing_size_does_not_filter_accessory_size_system(self):
        source = {
            "id": "DRESS-SOURCE", "title": "Kırmızı Elbise",
            "category": "Elbise", "category_path": "Kadın > Giyim > Elbise",
            "color": "Kırmızı", "gender": "Kadın", "age_group": "Yetişkin",
            "patterns": [], "price": 799.99, "sale_price": 699.99,
            "availability": "in stock", "total_stock": 5,
            "in_stock_sizes": ["M"], "weekly_sales": 1,
        }
        belt = {
            **source, "id": "BELT-90", "title": "Siyah Kemer",
            "category": "Kemer", "category_path": "Kadın > Aksesuar > Kemer",
            "color": "Siyah", "in_stock_sizes": ["90"],
        }
        client = self._client_with([source, belt])

        result = OutfitRecommender(
            ScrollBackedSearchEngine(client)
        ).recommend(source["id"], limit=4, preferred_sizes={"M"})

        self.assertEqual([item["id"] for item in result["products"]], ["BELT-90"])

    def test_child_size_normalization_survives_qdrant_candidate_fetch(self):
        source = {
            "id": "CHILD-SOURCE", "title": "Kırmızı Tişört Kız Çocuk",
            "category": "Tişört",
            "category_path": "Kız Çocuk - Genç Kız > Giyim > Tişört",
            "color": "Kırmızı", "gender": "Kadın", "age_group": "Çocuk",
            "patterns": [], "price": 299.99, "sale_price": 249.99,
            "availability": "in stock", "total_stock": 2,
            "in_stock_sizes": ["4/5 Yaş (110cm)"], "weekly_sales": 1,
        }
        matching = {
            **source, "id": "MATCHING-SIZE", "title": "Siyah Tayt Kız Çocuk",
            "category": "Tayt",
            "category_path": "Kız Çocuk - Genç Kız > Giyim > Tayt",
            "color": "Siyah", "in_stock_sizes": ["4-5 YAŞ"],
        }
        other_size = {
            **matching, "id": "OTHER-SIZE", "title": "Gri Tayt Kız Çocuk",
            "color": "Gri", "in_stock_sizes": ["7/8 Yaş (128cm)"],
        }
        client = self._client_with([source, matching, other_size])
        result = OutfitRecommender(
            ScrollBackedSearchEngine(client)).recommend(source["id"], limit=4)

        self.assertEqual(
            [item["id"] for item in result["products"]], ["MATCHING-SIZE"])

    def test_remembered_baby_range_and_standard_size_survive_qdrant_fetch(self):
        source = {
            "id": "BABY-SOURCE", "title": "Kırmızı Tişört Kız Bebek",
            "category": "Tişört",
            "category_path": "Kız Bebek > Giyim > Tişört",
            "color": "Kırmızı", "gender": "Kadın", "age_group": "Çocuk",
            "patterns": [], "price": 299.99, "sale_price": 249.99,
            "availability": "in stock", "total_stock": 2,
            "in_stock_sizes": ["18-24 Ay (92cm)"], "weekly_sales": 1,
        }
        covering_range = {
            **source, "id": "BABY-RANGE", "title": "Siyah Pantolon Kız Bebek",
            "category": "Pantolon",
            "category_path": "Kız Bebek > Giyim > Pantolon",
            "color": "Siyah", "in_stock_sizes": ["12-24 Ay"],
        }
        standard = {
            **covering_range, "id": "BABY-STD", "title": "Gri Pantolon Kız Bebek",
            "color": "Gri", "in_stock_sizes": ["STD"],
        }
        client = self._client_with([source, covering_range, standard])

        result = OutfitRecommender(
            ScrollBackedSearchEngine(client)
        ).recommend(
            source["id"],
            limit=4,
            preferred_sizes={"18-24 Ay (92cm)"},
        )

        self.assertEqual(
            {item["id"] for item in result["products"]},
            {"BABY-RANGE", "BABY-STD"},
        )

    def test_initial_baby_recommendation_scrolls_after_post_filter_shortfall(self):
        source = {
            "id": "BABY-SOURCE", "title": "Beyaz Tişört Kız Bebek",
            "category": "Tişört",
            "category_path": "Kız Bebek > Giyim > Tişört",
            "color": "Beyaz", "gender": "Kadın", "age_group": "Çocuk",
            "patterns": [], "price": 299.99, "sale_price": 249.99,
            "availability": "in stock", "total_stock": 2,
            "in_stock_sizes": ["0 AY"], "weekly_sales": 1,
        }
        first_valid = {
            **source, "id": "FIRST-VALID", "title": "Siyah Bebek Pantolon",
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
            for index in range(120)
        ]
        fallback_valid = {
            **first_valid,
            "id": "FALLBACK-VALID",
            "title": "Lacivert Bebek Jogger",
            "color": "Lacivert",
        }
        client = self._client_with(
            [source, first_valid, *wrong_sizes, fallback_valid])
        recommender = OutfitRecommender(FixedHybridSearchEngine(
            client, [first_valid, *wrong_sizes[:99]]))

        result = recommender.recommend(
            source["id"], limit=2, preferred_sizes={"0 AY"})

        ids = [item["id"] for item in result["products"]]
        self.assertEqual(set(ids), {"FIRST-VALID", "FALLBACK-VALID"})
        self.assertEqual(len(ids), len(set(ids)))

    def test_edit_filters_work_with_real_qdrant_conditions(self):
        source = {
            "id": "SRC25SP", "title": "Beyaz Basic Tişört",
            "category": "Tişört", "category_path": "Kadın > Giyim > Tişört",
            "color": "Beyaz", "gender": "Kadın", "age_group": "Yetişkin",
            "patterns": [], "price": 399.99, "sale_price": 299.99,
            "availability": "in stock", "total_stock": 5,
            "in_stock_sizes": ["S"], "weekly_sales": 1,
        }
        current = {
            **source, "id": "CURRENT", "title": "Siyah Basic Pantolon",
            "category": "Pantolon", "category_path": "Kadın > Giyim > Pantolon",
            "color": "Siyah", "sale_price": 600,
        }
        cheaper = {
            **current, "id": "CHEAPER", "title": "Siyah Regular Pantolon",
            "sale_price": 450,
        }
        equal_price = {
            **current, "id": "EQUAL", "title": "Bej Basic Pantolon",
            "color": "Bej", "sale_price": 600,
        }
        different_color = {
            **current, "id": "BEIGE", "title": "Bej Basic Pantolon",
            "color": "Bej", "sale_price": 550,
        }
        client = self._client_with(
            [source, current, cheaper, equal_price, different_color])
        recommender = OutfitRecommender(ScrollBackedSearchEngine(client))

        cheaper_result = recommender.recommend_replacement(
            source["id"], current["id"], preference="cheaper",
            excluded_product_ids={different_color["id"]},
        )
        color_result = recommender.recommend_replacement(
            source["id"], current["id"], preference="different_color",
            excluded_product_ids={equal_price["id"]},
        )

        self.assertEqual(
            [item["id"] for item in cheaper_result["products"]], ["CHEAPER"])
        self.assertEqual(
            [item["id"] for item in color_result["products"]], ["BEIGE"])

    def test_cheaper_edit_falls_back_to_normal_price_when_sale_price_is_missing(self):
        source = {
            "id": "SOURCE", "title": "Beyaz Basic Tişört",
            "category": "Tişört", "category_path": "Kadın > Giyim > Tişört",
            "color": "Beyaz", "gender": "Kadın", "age_group": "Yetişkin",
            "patterns": [], "price": 400, "sale_price": 300,
            "availability": "in stock", "total_stock": 5,
            "in_stock_sizes": ["S"], "weekly_sales": 1,
        }
        current = {
            **source, "id": "CURRENT", "title": "Siyah Basic Pantolon",
            "category": "Pantolon", "category_path": "Kadın > Giyim > Pantolon",
            "color": "Siyah", "price": 600, "sale_price": None,
        }
        cheaper_without_sale_price = {
            **current, "id": "CHEAPER-NORMAL", "title": "Siyah Regular Pantolon",
            "price": 450, "sale_price": None,
        }
        client = self._client_with([source, current, cheaper_without_sale_price])

        result = OutfitRecommender(
            ScrollBackedSearchEngine(client)).recommend_replacement(
                source["id"], current["id"], preference="cheaper")

        self.assertEqual(
            [item["id"] for item in result["products"]], ["CHEAPER-NORMAL"])

    def test_repeated_edits_return_products_not_seen_in_that_slot(self):
        source = {
            "id": "SOURCE", "title": "Beyaz Basic Tişört",
            "category": "Tişört", "category_path": "Kadın > Giyim > Tişört",
            "color": "Beyaz", "gender": "Kadın", "age_group": "Yetişkin",
            "patterns": [], "price": 400, "sale_price": 300,
            "availability": "in stock", "total_stock": 5,
            "in_stock_sizes": ["S"], "weekly_sales": 1,
        }
        black = {
            **source, "id": "BLACK", "title": "Siyah Basic Pantolon",
            "category": "Pantolon", "category_path": "Kadın > Giyim > Pantolon",
            "color": "Siyah", "sale_price": 600,
        }
        beige = {
            **black, "id": "BEIGE", "title": "Bej Basic Pantolon",
            "color": "Bej", "sale_price": 550,
        }
        navy = {
            **black, "id": "NAVY", "title": "Lacivert Chino Pantolon",
            "color": "Lacivert", "sale_price": 500,
        }
        brown = {
            **black, "id": "BROWN", "title": "Kahverengi Keten Pantolon",
            "color": "Kahve", "sale_price": 450,
        }
        client = self._client_with([source, black, beige, navy, brown])
        recommender = OutfitRecommender(ScrollBackedSearchEngine(client))

        first_color = recommender.recommend_replacement(
            source["id"], black["id"], preference="different_color",
            excluded_product_ids={black["id"]},
        )
        first_color_id = first_color["products"][0]["id"]
        second_color = recommender.recommend_replacement(
            source["id"], first_color_id, preference="different_color",
            excluded_product_ids={black["id"], first_color_id},
        )
        second_color_id = second_color["products"][0]["id"]
        third_color = recommender.recommend_replacement(
            source["id"], second_color_id, preference="different_color",
            excluded_product_ids={black["id"], first_color_id, second_color_id},
        )
        first_cheaper = recommender.recommend_replacement(
            source["id"], black["id"], preference="cheaper",
            excluded_product_ids={black["id"], navy["id"], brown["id"]},
        )
        first_cheaper_id = first_cheaper["products"][0]["id"]
        second_cheaper = recommender.recommend_replacement(
            source["id"], first_cheaper_id, preference="cheaper",
            excluded_product_ids={black["id"], first_cheaper_id, brown["id"]},
        )
        second_cheaper_id = second_cheaper["products"][0]["id"]
        third_cheaper = recommender.recommend_replacement(
            source["id"], second_cheaper_id, preference="cheaper",
            excluded_product_ids={black["id"], first_cheaper_id, second_cheaper_id},
        )
        third_cheaper_id = third_cheaper["products"][0]["id"]
        no_cheaper = recommender.recommend_replacement(
            source["id"], third_cheaper_id, preference="cheaper",
            excluded_product_ids={
                black["id"], first_cheaper_id, second_cheaper_id, third_cheaper_id,
            },
        )

        third_color_id = third_color["products"][0]["id"]
        self.assertNotIn(second_color_id, {black["id"], first_color_id})
        self.assertNotIn(
            third_color_id, {black["id"], first_color_id, second_color_id})
        prices = {item["id"]: item["sale_price"] for item in [beige, navy, brown]}
        self.assertEqual(
            [first_cheaper_id, second_cheaper_id, third_cheaper_id],
            ["BEIGE", "NAVY", "BROWN"],
        )
        self.assertLess(prices[first_cheaper_id], black["sale_price"])
        self.assertLess(prices[second_cheaper_id], prices[first_cheaper_id])
        self.assertLess(prices[third_cheaper_id], prices[second_cheaper_id])
        self.assertEqual(no_cheaper["products"], [])

    def test_edit_falls_back_to_scroll_when_hybrid_pool_fails_child_sizes(self):
        source = {
            "id": "CHILD-SOURCE", "title": "Beyaz Tişört Kız Çocuk",
            "category": "Tişört",
            "category_path": "Kız Çocuk - Genç Kız > Giyim > Tişört",
            "color": "Beyaz", "gender": "Kadın", "age_group": "Çocuk",
            "patterns": [], "price": 300, "sale_price": 250,
            "availability": "in stock", "total_stock": 5,
            "in_stock_sizes": ["4/5 Yaş (110cm)"], "weekly_sales": 1,
        }
        current = {
            **source, "id": "CURRENT", "title": "Siyah Tayt Kız Çocuk",
            "category": "Tayt",
            "category_path": "Kız Çocuk - Genç Kız > Giyim > Tayt",
            "color": "Siyah", "in_stock_sizes": ["4-5 YAŞ"],
        }
        wrong_sizes = [
            {
                **current,
                "id": f"WRONG-{index}",
                "title": f"Gri Tayt Kız Çocuk {index}",
                "color": "Gri",
                "in_stock_sizes": ["7/8 Yaş (128cm)"],
            }
            for index in range(140)
        ]
        valid = {
            **current, "id": "VALID-NEXT", "title": "Lacivert Tayt Kız Çocuk",
            "color": "Lacivert", "in_stock_sizes": ["4-5 YAŞ"],
        }
        client = self._client_with([source, current, *wrong_sizes, valid])
        recommender = OutfitRecommender(
            FixedHybridSearchEngine(client, wrong_sizes[:100]))

        result = recommender.recommend_replacement(
            source["id"], current["id"], preference="alternative",
            excluded_product_ids={current["id"]},
        )

        self.assertEqual([item["id"] for item in result["products"]], ["VALID-NEXT"])

    def test_edit_fallback_fills_slots_removed_by_title_deduplication(self):
        source = {
            "id": "SOURCE", "title": "Beyaz Basic Tişört",
            "category": "Tişört", "category_path": "Kadın > Giyim > Tişört",
            "color": "Beyaz", "gender": "Kadın", "age_group": "Yetişkin",
            "patterns": [], "price": 400, "sale_price": 300,
            "availability": "in stock", "total_stock": 5,
            "in_stock_sizes": ["S"], "weekly_sales": 1,
        }
        current = {
            **source, "id": "CURRENT", "title": "Siyah Basic Pantolon",
            "category": "Pantolon", "category_path": "Kadın > Giyim > Pantolon",
            "color": "Siyah", "sale_price": 600,
        }
        same_title_a = {
            **current, "id": "SAME-A", "title": "Regular Fit Pantolon",
            "color": "Bej",
        }
        same_title_b = {
            **current, "id": "SAME-B", "title": "Regular Fit Pantolon",
            "color": "Gri",
        }
        distinct_title = {
            **current, "id": "DISTINCT", "title": "Keten Chino Pantolon",
            "color": "Lacivert",
        }
        client = self._client_with(
            [source, current, same_title_a, same_title_b, distinct_title])
        recommender = OutfitRecommender(FixedHybridSearchEngine(
            client, [same_title_a, same_title_b]))

        result = recommender.recommend_replacement(
            source["id"], current["id"], preference="alternative",
            excluded_product_ids={current["id"]}, limit=2,
        )

        ids = {item["id"] for item in result["products"]}
        self.assertEqual(result["count"], 2)
        self.assertIn("DISTINCT", ids)
        self.assertEqual(len(ids & {"SAME-A", "SAME-B"}), 1)


if __name__ == "__main__":
    unittest.main()
