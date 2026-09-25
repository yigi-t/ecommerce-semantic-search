from types import SimpleNamespace
import unittest
import uuid

from src.outfit import (
    OutfitRecommender,
    compatibility_score,
    get_outfit_plan,
    recommendation_reason,
)


def product(**overrides):
    base = {
        "id": "SOURCE25SP",
        "title": "Kırmızı Çiçekli Tişört",
        "category": "Tişört",
        "category_path": "Kadın > Giyim > Tişört",
        "color": "Kırmızı",
        "gender": "Kadın",
        "age_group": "Yetişkin",
        "patterns": ["çiçekli"],
        "price": 399.99,
        "sale_price": 299.99,
        "availability": "in stock",
        "total_stock": 5,
        "in_stock_sizes": ["S", "M"],
        "weekly_sales": 2,
        "image": "https://example.com/product.jpg",
        "link": "https://example.com/product",
    }
    base.update(overrides)
    return base


class FakeClient:
    def __init__(self, source, products=()):
        self.products = {
            str(uuid.uuid5(uuid.NAMESPACE_URL, item["id"])): item
            for item in (source, *products)
        }

    def retrieve(self, ids, **_kwargs):
        return [
            SimpleNamespace(payload=self.products[point_id])
            for point_id in ids
            if point_id in self.products
        ]


class FakeSearchEngine:
    def __init__(self, source, candidates):
        self.client = FakeClient(source, candidates)
        self.candidates = candidates
        self.calls = []

    def _hybrid_query(self, text, qfilter, limit):
        self.calls.append((text, qfilter, limit))
        return [SimpleNamespace(payload=item, score=0.5) for item in self.candidates]


class OutfitRulesTest(unittest.TestCase):
    def test_supported_groups_do_not_treat_ambiguous_jean_as_bottom(self):
        self.assertIsNotNone(get_outfit_plan("Tişört"))
        self.assertIsNotNone(get_outfit_plan("Pantolon"))
        self.assertIsNotNone(get_outfit_plan("Elbise"))
        self.assertIsNone(get_outfit_plan("Jean"))
        self.assertIsNotNone(get_outfit_plan("Jean", "Slim Fit Jean Pantolon"))
        self.assertIsNone(get_outfit_plan("Jean", "Jean Etek Kız Çocuk"))
        self.assertIsNone(get_outfit_plan("Vücut Spreyi"))

    def test_plain_black_bottom_outranks_patterned_accent_bottom(self):
        source = product()
        plan = get_outfit_plan(source["category"])
        balanced = product(
            id="PANTS25SP",
            title="Siyah Basic Pantolon",
            category="Pantolon",
            category_path="Kadın > Giyim > Pantolon",
            color="Siyah",
            patterns=[],
            sale_price=479.99,
        )
        busy = product(
            id="PANTS25SP2",
            title="Turuncu Leopar Desenli Pantolon",
            category="Pantolon",
            category_path="Kadın > Giyim > Pantolon",
            color="Turuncu",
            patterns=["leopar"],
            sale_price=479.99,
        )
        self.assertGreater(
            compatibility_score(source, balanced, plan, 0),
            compatibility_score(source, busy, plan, 0),
        )
        self.assertIn("desenini dengeler", recommendation_reason(source, balanced))

    def test_general_print_signal_is_treated_as_a_visible_pattern(self):
        source = product(title="Kırmızı Baskılı Tişört", patterns=["baskılı"])
        plan = get_outfit_plan(source["category"])
        plain = product(
            id="PLAIN", title="Siyah Basic Pantolon", category="Pantolon",
            category_path="Kadın > Giyim > Pantolon", color="Siyah",
        )
        patterned = product(
            id="PATTERNED", title="Siyah Desenli Pantolon", category="Pantolon",
            category_path="Kadın > Giyim > Pantolon", color="Siyah",
        )
        self.assertGreater(
            compatibility_score(source, plain, plan, 0),
            compatibility_score(source, patterned, plan, 0),
        )

    def test_price_score_accounts_for_top_to_bottom_price_ratio(self):
        source = product(title="Beyaz Basic Tişört", color="Beyaz", sale_price=300)
        plan = get_outfit_plan(source["category"])
        expected = product(
            id="PANTS1NS",
            title="Siyah Pantolon",
            category="Pantolon",
            category_path="Kadın > Giyim > Pantolon",
            color="Siyah",
            sale_price=480,
        )
        too_low = product(
            id="PANTS2NS",
            title="Siyah Pantolon 2",
            category="Pantolon",
            category_path="Kadın > Giyim > Pantolon",
            color="Siyah",
            sale_price=200,
        )
        self.assertGreater(
            compatibility_score(source, expected, plan, 0),
            compatibility_score(source, too_low, plan, 0),
        )

    def test_candidate_validation_preserves_stock_audience_and_child_size(self):
        source = product(
            category_path="Kız Çocuk - Genç Kız > Giyim > Tişört",
            age_group="Çocuk",
            in_stock_sizes=["7/8 Yaş (128cm)"],
        )
        plan = get_outfit_plan(source["category"])
        valid = product(
            id="BOTTOM",
            category="Pantolon",
            category_path="Kız Çocuk - Genç Kız > Giyim > Pantolon",
            age_group="Çocuk",
            in_stock_sizes=["7-8 YAŞ", "8/9 Yaş (134cm)"],
        )
        baby = product(
            id="BABY",
            category="Pantolon",
            category_path="Kız Bebek > Giyim > Pantolon",
            age_group="Çocuk",
            in_stock_sizes=["7/8 Yaş (128cm)"],
        )
        wrong_size = product(
            id="WRONG-SIZE",
            category="Pantolon",
            category_path="Kız Çocuk - Genç Kız > Giyim > Pantolon",
            age_group="Çocuk",
            in_stock_sizes=["11/12 Yaş (152cm)"],
        )
        no_stock = product(
            id="NO-STOCK",
            category="Pantolon",
            category_path="Kız Çocuk - Genç Kız > Giyim > Pantolon",
            age_group="Çocuk",
            total_stock=0,
        )
        no_sizes = product(
            id="NO-SIZES",
            category="Pantolon",
            category_path="Kız Çocuk - Genç Kız > Giyim > Pantolon",
            age_group="Çocuk",
            in_stock_sizes=[],
        )
        standard = product(
            id="STANDARD-SIZE",
            category="Pantolon",
            category_path="Kız Çocuk - Genç Kız > Giyim > Pantolon",
            age_group="Çocuk",
            in_stock_sizes=["STD"],
        )
        self.assertTrue(OutfitRecommender._is_valid_candidate(source, valid, plan))
        self.assertTrue(OutfitRecommender._is_valid_candidate(source, standard, plan))
        self.assertFalse(OutfitRecommender._is_valid_candidate(source, baby, plan))
        self.assertFalse(OutfitRecommender._is_valid_candidate(source, wrong_size, plan))
        self.assertFalse(OutfitRecommender._is_valid_candidate(source, no_stock, plan))
        self.assertFalse(OutfitRecommender._is_valid_candidate(source, no_sizes, plan))

    def test_remembered_size_filters_adult_candidates_and_allows_standard_size(self):
        source = product()
        matching = product(
            id="MATCHING", title="Siyah Pantolon", category="Pantolon",
            category_path="Kadın > Giyim > Pantolon", in_stock_sizes=["M"],
        )
        other_size = product(
            id="OTHER", title="Bej Pantolon", category="Pantolon",
            category_path="Kadın > Giyim > Pantolon", in_stock_sizes=["L"],
        )
        standard = product(
            id="STANDARD", title="Gri Pantolon", category="Pantolon",
            category_path="Kadın > Giyim > Pantolon", in_stock_sizes=["STD"],
        )
        search_engine = FakeSearchEngine(
            source, [matching, other_size, standard])

        result = OutfitRecommender(search_engine).recommend(
            source["id"], limit=4, preferred_sizes={"M"})

        self.assertEqual(
            {item["id"] for item in result["products"]},
            {"MATCHING", "STANDARD"},
        )
        self.assertNotIn("OTHER", [item["id"] for item in result["products"]])
        self.assertIn("hatırlanan bedenlerin", result["summary"])
        qfilter = search_engine.calls[0][1].model_dump(exclude_none=True)
        size_filter = next(
            condition for condition in qfilter["must"]
            if "should" in condition
        )
        size_condition = next(
            condition for condition in size_filter["should"]
            if condition.get("key") == "in_stock_sizes"
        )
        self.assertEqual(
            set(size_condition["match"]["any"]),
            {"M", "STD", "One Size"},
        )

    def test_remembered_size_accepts_covering_child_ranges(self):
        covering_age = product(
            category="Pantolon",
            in_stock_sizes=["5-9 Yaş"],
        )
        covering_month = product(
            category="Pantolon",
            in_stock_sizes=["12-24 Ay"],
        )
        self.assertTrue(OutfitRecommender._matches_preferred_size(
            covering_age, {"7-8 YAŞ"}))
        self.assertTrue(OutfitRecommender._matches_preferred_size(
            covering_month, {"18-24 Ay (92cm)"}))
        self.assertFalse(OutfitRecommender._matches_preferred_size(
            covering_age, {"10-11 YAŞ"}))

    def test_single_month_size_matches_covering_baby_range(self):
        self.assertEqual(
            OutfitRecommender._normalize_child_size("0 AY"),
            "month:0",
        )
        covering_month = product(
            id="BABY-BOTTOM",
            category="Pantolon",
            category_path="Kız Bebek > Giyim > Pantolon",
            age_group="Çocuk",
            in_stock_sizes=["0-1 Ay (56cm)"],
        )
        source = product(
            category_path="Kız Bebek > Giyim > Tişört",
            age_group="Çocuk",
            in_stock_sizes=["0 AY"],
        )
        plan = get_outfit_plan(source["category"])

        self.assertTrue(OutfitRecommender._matches_preferred_size(
            covering_month, {"0 AY"}))
        self.assertTrue(OutfitRecommender._is_valid_candidate(
            source, covering_month, plan, {"0 AY"}))

    def test_accessory_size_system_is_exempt_from_clothing_preference(self):
        source = product(
            category="Elbise",
            title="Kırmızı Elbise",
            category_path="Kadın > Giyim > Elbise",
        )
        plan = get_outfit_plan(source["category"])

        for category, size in (
            ("Kemer", "90"),
            ("Çanta", "STD"),
            ("Takı", "One Size"),
            ("Yüzük", "12"),
        ):
            with self.subTest(category=category):
                accessory = product(
                    id=f"ACCESSORY-{category}", title=f"Siyah {category}",
                    category=category,
                    category_path=f"Kadın > Aksesuar > {category}",
                    color="Siyah", in_stock_sizes=[size],
                )
                self.assertTrue(OutfitRecommender._matches_preferred_size(
                    accessory, {"M"}))
                if category in plan.target_categories:
                    self.assertTrue(OutfitRecommender._is_valid_candidate(
                        source, accessory, plan, {"M"}))

    def test_recommender_filters_and_diversifies_results(self):
        source = product()
        candidates = [
            product(
                id="P1", title="Siyah Basic Pantolon", category="Pantolon",
                category_path="Kadın > Giyim > Pantolon", color="Siyah",
            ),
            product(
                id="P2", title="Siyah Basic Pantolon", category="Pantolon",
                category_path="Kadın > Giyim > Pantolon", color="Bej",
            ),
            product(
                id="P3", title="Mom Jean Pantolon", category="Jean Pantolon",
                category_path="Kadın > Giyim > Jean Pantolon", color="İndigo",
            ),
            product(
                id="P4", title="Krem Keten Pantolon", category="Pantolon",
                category_path="Kadın > Giyim > Pantolon", color="Ekru",
            ),
            product(
                id="AMBIGUOUS", title="Jean Etek", category="Jean",
                category_path="Kadın > Giyim > Jean", color="Mavi",
            ),
            product(
                id="EMPTY", title="Gri Pantolon", category="Pantolon",
                category_path="Kadın > Giyim > Pantolon", total_stock=0,
            ),
        ]
        search_engine = FakeSearchEngine(source, candidates)
        result = OutfitRecommender(search_engine).recommend(source["id"], limit=4)

        ids = [item["id"] for item in result["products"]]
        self.assertEqual(len(ids), 3)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertNotIn("AMBIGUOUS", ids)
        self.assertNotIn("EMPTY", ids)
        self.assertTrue(all(item["reason"] for item in result["products"]))
        self.assertTrue(all("score" not in item for item in result["products"]))
        self.assertEqual(len(search_engine.calls), 1)

        qfilter = search_engine.calls[0][1].model_dump(exclude_none=True)
        keys = {condition["key"] for condition in qfilter["must"]}
        self.assertTrue({"availability", "category", "total_stock", "gender", "age_group"} <= keys)

    def test_unsupported_product_returns_empty_without_querying(self):
        source = product(category="Vücut Spreyi")
        search_engine = FakeSearchEngine(source, [])
        result = OutfitRecommender(search_engine).recommend(source["id"])
        self.assertFalse(result["eligible"])
        self.assertEqual(result["products"], [])
        self.assertEqual(search_engine.calls, [])

    def test_missing_audience_data_fails_closed_without_querying(self):
        source = product(gender="")
        search_engine = FakeSearchEngine(source, [])
        result = OutfitRecommender(search_engine).recommend(source["id"])
        self.assertFalse(result["eligible"])
        self.assertIn("hedef kitle", result["summary"])
        self.assertEqual(search_engine.calls, [])

    def test_child_source_without_sizes_fails_closed_without_querying(self):
        source = product(
            age_group="Çocuk",
            category_path="Kız Çocuk - Genç Kız > Giyim > Tişört",
            in_stock_sizes=[],
        )
        search_engine = FakeSearchEngine(source, [])
        result = OutfitRecommender(search_engine).recommend(source["id"])
        self.assertFalse(result["eligible"])
        self.assertIn("beden bilgisi", result["summary"])
        self.assertEqual(search_engine.calls, [])

    def test_alternative_replacement_excludes_current_and_seen_products(self):
        source = product()
        current = product(
            id="CURRENT", title="Siyah Basic Pantolon", category="Pantolon",
            category_path="Kadın > Giyim > Pantolon", color="Siyah",
        )
        already_seen = product(
            id="SEEN", title="İndigo Jean Pantolon", category="Jean Pantolon",
            category_path="Kadın > Giyim > Jean Pantolon", color="İndigo",
        )
        replacement = product(
            id="NEXT", title="Bej Keten Pantolon", category="Pantolon",
            category_path="Kadın > Giyim > Pantolon", color="Bej",
        )
        search_engine = FakeSearchEngine(
            source, [current, already_seen, replacement])

        result = OutfitRecommender(search_engine).recommend_replacement(
            source["id"],
            current["id"],
            preference="alternative",
            excluded_product_ids={already_seen["id"]},
        )

        self.assertEqual([item["id"] for item in result["products"]], ["NEXT"])
        qfilter = search_engine.calls[0][1].model_dump(exclude_none=True)
        excluded_point_ids = set(qfilter["must_not"][0]["has_id"])
        self.assertEqual(excluded_point_ids, {
            str(uuid.uuid5(uuid.NAMESPACE_URL, "CURRENT")),
            str(uuid.uuid5(uuid.NAMESPACE_URL, "SEEN")),
        })

    def test_replacement_keeps_the_remembered_size_filter(self):
        source = product()
        current = product(
            id="CURRENT", title="Siyah Pantolon", category="Pantolon",
            category_path="Kadın > Giyim > Pantolon", in_stock_sizes=["M"],
        )
        wrong_size = product(
            id="WRONG", title="Bej Pantolon", category="Pantolon",
            category_path="Kadın > Giyim > Pantolon", in_stock_sizes=["L"],
        )
        matching = product(
            id="MATCHING", title="Lacivert Pantolon", category="Pantolon",
            category_path="Kadın > Giyim > Pantolon", in_stock_sizes=["S/M"],
        )
        search_engine = FakeSearchEngine(source, [wrong_size, matching, current])

        result = OutfitRecommender(search_engine).recommend_replacement(
            source["id"],
            current["id"],
            preference="alternative",
            preferred_sizes={"M", "S/M"},
        )

        self.assertEqual([item["id"] for item in result["products"]], ["MATCHING"])

    def test_cheaper_replacement_is_strictly_cheaper_and_same_category(self):
        source = product()
        current = product(
            id="CURRENT", title="Siyah Basic Pantolon", category="Pantolon",
            category_path="Kadın > Giyim > Pantolon", color="Siyah",
            sale_price=600,
        )
        cheaper = product(
            id="CHEAPER", title="Siyah Regular Pantolon", category="Pantolon",
            category_path="Kadın > Giyim > Pantolon", color="Siyah",
            sale_price=450,
        )
        equal_price = product(
            id="EQUAL", title="Bej Pantolon", category="Pantolon",
            category_path="Kadın > Giyim > Pantolon", color="Bej",
            sale_price=600,
        )
        cheaper_wrong_category = product(
            id="SKIRT", title="Siyah Etek", category="Etek",
            category_path="Kadın > Giyim > Etek", color="Siyah",
            sale_price=300,
        )
        search_engine = FakeSearchEngine(
            source, [current, equal_price, cheaper_wrong_category, cheaper])

        result = OutfitRecommender(search_engine).recommend_replacement(
            source["id"], current["id"], preference="cheaper")

        self.assertEqual([item["id"] for item in result["products"]], ["CHEAPER"])
        self.assertIn("daha uygun fiyatlıdır", result["products"][0]["reason"])
        qfilter = search_engine.calls[0][1].model_dump(exclude_none=True)
        category_condition = next(
            item for item in qfilter["must"] if item["key"] == "category")
        price_condition = next(
            item for item in qfilter["must"] if "should" in item)
        self.assertEqual(category_condition["match"]["any"], ["Pantolon"])
        sale_price_branch = price_condition["should"][0]
        normal_price_branch = price_condition["should"][1]["must"]
        self.assertEqual(sale_price_branch["key"], "sale_price")
        self.assertEqual(sale_price_branch["range"]["lt"], 600.0)
        self.assertEqual(normal_price_branch[0]["is_empty"]["key"], "sale_price")
        self.assertEqual(normal_price_branch[1]["key"], "price")
        self.assertEqual(normal_price_branch[1]["range"]["lt"], 600.0)

    def test_different_color_replacement_keeps_the_current_category(self):
        source = product()
        current = product(
            id="CURRENT", title="Siyah Basic Pantolon", category="Pantolon",
            category_path="Kadın > Giyim > Pantolon", color="Siyah",
        )
        same_color = product(
            id="BLACK", title="Siyah Pantolon", category="Pantolon",
            category_path="Kadın > Giyim > Pantolon", color="Siyah",
        )
        wrong_category = product(
            id="SKIRT", title="Bej Etek", category="Etek",
            category_path="Kadın > Giyim > Etek", color="Bej",
        )
        different_color = product(
            id="BEIGE", title="Bej Basic Pantolon", category="Pantolon",
            category_path="Kadın > Giyim > Pantolon", color="Bej",
        )
        search_engine = FakeSearchEngine(
            source, [current, same_color, wrong_category, different_color])

        result = OutfitRecommender(search_engine).recommend_replacement(
            source["id"], current["id"], preference="different_color")

        self.assertEqual([item["id"] for item in result["products"]], ["BEIGE"])
        self.assertIn("farklı renkte benzer", result["products"][0]["reason"])
        self.assertIn("Aynı kategorideki", result["summary"])

    def test_replacement_validation_fails_closed_before_querying(self):
        source = product()
        current = product(
            id="CURRENT", title="Siyah Basic Pantolon", category="Pantolon",
            category_path="Kadın > Giyim > Pantolon", color=None,
        )
        search_engine = FakeSearchEngine(source, [current])
        recommender = OutfitRecommender(search_engine)

        with self.assertRaisesRegex(ValueError, "renk bilgisi"):
            recommender.recommend_replacement(
                source["id"], current["id"], preference="different_color")
        self.assertEqual(search_engine.calls, [])


if __name__ == "__main__":
    unittest.main()
