# -*- coding: utf-8 -*-
"""
Defacto Google Merchant XML feed'ini akış halinde (streaming) parse eder.

Feed ~180 MB olduğu için DOM'a yüklenmez; lxml.iterparse ile item item
okunur, her item işlendikten sonra bellekten atılır. Çıktı: JSONL
(satır başına bir zenginleştirilmiş ürün).

Kullanım:
    python -m src.parse_feed data/veri-seti.xml data/products.jsonl
"""

import html
import json
import re
import sys

from lxml import etree

from .enrich import enrich

G_NS = "{http://base.google.com/ns/1.0}"


def _txt(item, tag: str) -> str:
    el = item.find(tag)
    if el is None or el.text is None:
        return ""
    return html.unescape(el.text).strip()


def _clean_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_item(item) -> dict | None:
    title = _txt(item, "tr-tr_title")
    if not title:
        return None

    # Beden/stok varyantları
    sizes, total_stock = [], 0
    sizes_el = item.find(f"{G_NS}sizes")
    if sizes_el is not None:
        for s in sizes_el.findall(f"{G_NS}size"):
            name_el = s.find("name")
            qty_el = s.find("stockQuantity")
            qty = int(qty_el.text) if qty_el is not None and (qty_el.text or "").isdigit() else 0
            total_stock += qty
            if name_el is not None and name_el.text and qty > 0:
                sizes.append(name_el.text.strip())

    def _f(tag):
        v = _txt(item, tag)
        try:
            return float(v)
        except ValueError:
            return None

    product = {
        "id": _txt(item, f"{G_NS}id"),
        "title": title,
        "description": _clean_html(_txt(item, "tr-tr_description")),
        "category_path": _txt(item, "tr-tr_product_type"),   # Kadın > Giyim > Elbise
        "category": _txt(item, f"{G_NS}wawlabs_product_category"),  # Elbise
        "color_raw": _txt(item, "tr-tr_color"),
        "material": _txt(item, "tr-tr_material"),
        "gender_raw": _txt(item, "tr-tr_gender"),
        "age_group_raw": _txt(item, f"{G_NS}age_group"),
        "price": _f("TRY_price"),
        "sale_price": _f("TRY_sale_price"),
        "availability": _txt(item, f"{G_NS}availability"),
        "in_stock_sizes": sizes,
        "total_stock": total_stock,
        "weekly_sales": _f(f"{G_NS}weekly_sales") or 0,
        "link": _txt(item, "tr-tr_link"),
        "image": _txt(item, "global_image_link"),
    }
    return enrich(product)


def parse_feed(xml_path: str, out_path: str) -> dict:
    stats = {"total": 0, "written": 0, "skipped": 0}
    with open(out_path, "w", encoding="utf-8") as out:
        for _, item in etree.iterparse(xml_path, tag="item",
                                       recover=True, huge_tree=True):
            stats["total"] += 1
            try:
                product = parse_item(item)
                if product:
                    out.write(json.dumps(product, ensure_ascii=False) + "\n")
                    stats["written"] += 1
                else:
                    stats["skipped"] += 1
            finally:
                # Belleği serbest bırak: işlenen item'ı ve öncesini temizle
                item.clear()
                while item.getprevious() is not None:
                    del item.getparent()[0]
    return stats


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/veri-seti.xml"
    dst = sys.argv[2] if len(sys.argv) > 2 else "data/products.jsonl"
    print(json.dumps(parse_feed(src, dst), ensure_ascii=False))
