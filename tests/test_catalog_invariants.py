from collections import Counter
import json
from pathlib import Path
import unittest


CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "products.jsonl"


class CatalogInvariantTest(unittest.TestCase):
    def test_documented_exact_match_counts(self):
        counts = Counter()
        with CATALOG_PATH.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                if product.get("availability") != "in stock":
                    continue
                category = product.get("category")
                color = product.get("color")
                patterns = set(product.get("patterns") or [])
                fabrics = set(product.get("fabrics") or [])
                fits = set(product.get("fits") or [])
                necklines = set(product.get("necklines") or [])
                lengths = set(product.get("lengths") or [])

                if category == "Elbise" and color == "Siyah" and "puantiye" in patterns:
                    counts["black_polka_dress"] += 1
                if (category == "Gömlek" and "keten" in fabrics
                        and (product.get("sale_price") or 0) <= 500):
                    counts["linen_shirt_under_500"] += 1
                if (category == "Sweatshirt" and color == "Siyah"
                        and product.get("gender") == "Kadın" and "oversize" in fits):
                    counts["women_black_oversize_sweatshirt"] += 1
                if (category == "Jean" and product.get("gender") == "Erkek"
                        and "denim" in fabrics and "slim fit" in fits):
                    counts["men_slim_jean"] += 1
                if category == "Etek" and "çiçekli" in patterns and "midi" in lengths:
                    counts["floral_midi_skirt"] += 1
                if (category == "Kazak" and color == "Bordo" and "triko" in fabrics
                        and "v yaka" in necklines):
                    counts["burgundy_vneck_knit"] += 1
                if category == "Elbise" and color == "Kırmızı" and "puantiye" in patterns:
                    counts["red_polka_dress"] += 1

        self.assertEqual(counts["black_polka_dress"], 10)
        self.assertEqual(counts["linen_shirt_under_500"], 52)
        self.assertEqual(counts["women_black_oversize_sweatshirt"], 15)
        self.assertEqual(counts["men_slim_jean"], 17)
        self.assertEqual(counts["floral_midi_skirt"], 15)
        self.assertEqual(counts["burgundy_vneck_knit"], 2)
        self.assertEqual(counts["red_polka_dress"], 0)


if __name__ == "__main__":
    unittest.main()
