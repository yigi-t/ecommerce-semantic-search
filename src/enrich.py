# -*- coding: utf-8 -*-
"""
Nitelik zenginleştirme (index-time attribute extraction).

Feed'de desen/fit/yaka gibi nitelikler ayrı alan olarak YOK; başlık ve
açıklamanın içinde gömülü. Bu modül, Türkçe moda sözlükleri (lexicon)
ile bu nitelikleri çıkarıp yapılandırılmış payload alanlarına dönüştürür.
Aynı sözlükler sorgu tarafında da (query_parser) kullanılır — tek kaynak.
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# Normalizasyon
# ---------------------------------------------------------------------------

_TR_MAP = str.maketrans("ıİşŞğĞüÜöÖçÇ", "iisSgGuUoOcC")


def tr_fold(text: str) -> str:
    """Türkçe karakterleri ASCII'ye indirger, küçük harfe çevirir.

    Eşleştirme her zaman fold edilmiş metin üzerinde yapılır ki
    'puantiyeli' / 'PUANTİYELİ' / 'Puantiye' aynı şeye denk gelsin.
    """
    text = text.replace("I", "ı").replace("İ", "i").lower()
    text = text.translate(_TR_MAP)
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


# ---------------------------------------------------------------------------
# Sözlükler: kanonik_değer -> tetikleyici kelime kökleri (fold edilmiş)
# Kökler kullanılır ki 'puantiye', 'puantiyeli' hepsi yakalansın.
# ---------------------------------------------------------------------------

PATTERNS = {
    "puantiye":  ["puantiye"],
    "çizgili":   ["cizgili", "cizgi desen"],
    "ekose":     ["ekose"],
    "kareli":    ["kareli", "kare desen"],
    "çiçekli":   ["cicekli", "cicek desen", "floral"],
    "leopar":    ["leopar"],
    "zebra":     ["zebra"],
    "kamuflaj":  ["kamuflaj"],
    "batik":     ["batik", "tie dye", "tie-dye"],
    "baskılı":   ["baskili", "baski detayli"],
    "desenli":   ["desenli"],
    "nakışlı":   ["nakisli", "nakis detayli", "islemeli"],
    "dantelli":  ["dantel"],
    "payetli":   ["payet"],
    "jakarlı":   ["jakar"],
    "düz":       ["duz renk", "basic duz"],
}

FABRICS = {
    "pamuk":    ["pamuk", "koton", "cotton"],
    "keten":    ["keten", "linen"],
    "denim":    ["denim", "jean", "kot "],
    "saten":    ["saten"],
    "kadife":   ["kadife"],
    "triko":    ["triko"],
    "örme":     ["orme"],
    "dokuma":   ["dokuma"],
    "tül":      ["tul "],
    "şifon":    ["sifon"],
    "viskon":   ["viskon", "viskoz"],
    "polar":    ["polar"],
    "yün":      ["yun ", "yunlu"],
    "deri":     ["deri"],
    "suni deri": ["suni deri"],
    "fitilli":  ["fitilli"],
    "twill":    ["twill"],
    "gabardin": ["gabardin"],
    "poplin":   ["poplin"],
    "muslin":   ["muslin"],
}

FITS = {
    "slim fit":     ["slim fit", "slim-fit", "dar kalip", "dar kesim"],
    "regular fit":  ["regular fit", "regular kalip", "standart kalip"],
    "relax fit":    ["relax fit", "relaxed fit", "rahat kalip", "rahat kesim"],
    "oversize":     ["oversize", "over size", "bol kesim", "bol kalip", "salas"],
    "skinny":       ["skinny"],
    "straight":     ["straight fit", "duz paca", "boru paca"],
    "wide leg":     ["wide leg", "genis paca", "bol paca"],
    "mom fit":      ["mom fit", "mom jean"],
    "boyfriend":    ["boyfriend"],
    "bodycon":      ["bodycon", "vucuda oturan"],
    "a kesim":      ["a kesim", "a-form", "a form"],
}

NECKLINES = {
    "v yaka":       ["v yaka"],
    "bisiklet yaka": ["bisiklet yaka"],
    "hakim yaka":   ["hakim yaka"],
    "polo yaka":    ["polo yaka"],
    "gömlek yaka":  ["gomlek yaka"],
    "kayık yaka":   ["kayik yaka"],
    "halter yaka":  ["halter"],
    "balıkçı yaka": ["balikci yaka", "bogazli"],
    "kare yaka":    ["kare yaka"],
    "degaje yaka":  ["degaje"],
    "dik yaka":     ["dik yaka"],
    "kapüşonlu":    ["kapuson", "kapsonlu", "hoodie"],
    "geniş yaka":   ["genis yaka"],
}

SLEEVES = {
    "uzun kollu":  ["uzun kol"],
    "kısa kollu":  ["kisa kol"],
    "kolsuz":      ["kolsuz", "askili"],
    "balon kol":   ["balon kol"],
    "yarım kol":   ["yarim kol", "truvakar"],
}

LENGTHS = {
    "mini":  ["mini "],
    "midi":  ["midi "],
    "maxi":  ["maxi", "maksi", "uzun etek", "uzun elbise"],
    "crop":  ["crop"],
    "tunik": ["tunik"],
}

# Feed'in tr-tr_color alanındaki değerleri kanonik forma indirger
COLOR_CANON = {
    "siyah": "Siyah", "beyaz": "Beyaz", "ekru": "Ekru", "bej": "Bej",
    "kahve": "Kahve", "kahverengi": "Kahve", "mavi": "Mavi",
    "lacivert": "Lacivert", "gri": "Gri", "antrasit": "Antrasit",
    "pembe": "Pembe", "acik pembe": "Pembe", "pudra": "Pembe",
    "yesil": "Yeşil", "haki": "Haki", "sari": "Sarı", "hardal": "Sarı",
    "bordo": "Bordo", "kirmizi": "Kırmızı", "mor": "Mor", "lila": "Mor",
    "indigo": "İndigo", "turuncu": "Turuncu", "turkuaz": "Turkuaz",
    "denim": "İndigo", "coral": "Turuncu", "mercan": "Turuncu",
    "altin": "Altın", "gumus": "Gümüş", "seffaf": "Şeffaf",
    "cok renkli": "Çok Renkli", "karma": "Çok Renkli",
    "gold": "Altın", "silver": "Gümüş", "krem": "Ekru",
}

GENDER_CANON = {"female": "Kadın", "male": "Erkek", "unisex": "Unisex"}
AGE_CANON = {"adult": "Yetişkin", "kids": "Çocuk", "baby": "Bebek",
             "newborn": "Yenidoğan", "toddler": "Bebek"}


def _extract(lexicon: dict, folded_text: str) -> list:
    """Fold edilmiş metinde sözlük köklerini arar, kanonik değerleri döner."""
    found = []
    for canon, triggers in lexicon.items():
        for t in triggers:
            if t in folded_text:
                found.append(canon)
                break
    return found


def canon_color(raw: str) -> str:
    return COLOR_CANON.get(tr_fold(raw).strip(), raw.strip().title()) if raw else ""


def enrich(product: dict) -> dict:
    """Ürün sözlüğünü zenginleştirilmiş nitelik alanlarıyla döner.

    product: parse_feed.py'nin ürettiği ham kayıt.
    """
    text = " ".join(filter(None, [
        product.get("title", ""),
        product.get("description", ""),
        product.get("category", ""),
    ]))
    folded = tr_fold(text)

    attrs = {
        "patterns":  _extract(PATTERNS, folded),
        "fabrics":   _extract(FABRICS, folded),
        "fits":      _extract(FITS, folded),
        "necklines": _extract(NECKLINES, folded),
        "sleeves":   _extract(SLEEVES, folded),
        "lengths":   _extract(LENGTHS, folded),
    }

    # 'desenli'/'baskılı' genel; spesifik desen varsa gürültü olmasın diye at
    specific = set(attrs["patterns"]) - {"desenli", "baskılı", "düz"}
    if specific:
        attrs["patterns"] = sorted(specific)

    product["color"] = canon_color(product.get("color_raw", ""))
    product["gender"] = GENDER_CANON.get(
        tr_fold(product.get("gender_raw", "")), product.get("gender_raw", ""))
    product["age_group"] = AGE_CANON.get(
        tr_fold(product.get("age_group_raw", "")), product.get("age_group_raw", ""))

    product.update(attrs)

    # Embedding'e girecek tek metin: nitelikleri de içerir ki dense vektör
    # "puantiyeli" sinyalini kaybetmesin.
    attr_words = " ".join(
        attrs["patterns"] + attrs["fabrics"] + attrs["fits"]
        + attrs["necklines"] + attrs["sleeves"] + attrs["lengths"]
    )
    product["search_text"] = " | ".join(filter(None, [
        product.get("title", ""),
        product.get("category", ""),
        product["color"],
        product["gender"],
        attr_words,
        re.sub(r"\s+", " ", product.get("description", ""))[:400],
    ]))
    return product
