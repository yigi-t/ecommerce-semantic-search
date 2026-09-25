# -*- coding: utf-8 -*-
"""
Sorgu anlama katmanı (query understanding).

"puantiyeli kırmızı elbise 500 tl altı" gibi bir sorguyu:
    {category: Elbise, color: Kırmızı, patterns: [puantiye],
     price_max: 500, semantic_text: "puantiyeli kırmızı elbise"}
şeklinde yapılandırılmış bir QueryIntent'e çevirir.

Sözlükler enrich.py ile ORTAK — index tarafında ne çıkarılıyorsa sorgu
tarafında da aynısı tanınır; iki tarafın senkron kalmasını garanti eder.
"""

import re
import difflib
from dataclasses import dataclass, field

from .enrich import (COLOR_CANON, FABRICS, FITS, LENGTHS, NECKLINES,
                     PATTERNS, SLEEVES, tr_fold)

# Feed'deki kategori isimleri + kullanıcıların kullandığı eş anlamlılar.
# Değerler feed'in wawlabs_product_category alanındaki kanonik isimlerdir.
CATEGORY_SYNONYMS = {
    "Elbise": ["elbise"],
    "Tişört": ["tisort", "t-shirt", "tshirt", "t shirt"],
    "Gömlek": ["gomlek"],
    "Bluz": ["bluz"],
    "Pantolon": ["pantolon"],
    "Jean": ["jean", "kot pantolon", "jean pantolon"],
    "Etek": ["etek"],
    "Şort": ["sort"],
    "Ceket": ["ceket"],
    "Mont": ["mont"],
    "Kaban": ["kaban"],
    "Trençkot": ["trenckot", "trenc kot"],
    "Hırka": ["hirka"],
    "Kazak": ["kazak"],
    "Sweatshirt": ["sweatshirt", "sweat shirt", "swet"],
    "Eşofman Altı": ["esofman alti", "esofman"],
    "Tayt": ["tayt"],
    "Ayakkabı": ["ayakkabi"],
    "Sneaker": ["sneaker", "spor ayakkabi"],
    "Bot": ["bot"],
    "Sandalet": ["sandalet"],
    "Terlik": ["terlik"],
    "Çanta": ["canta"],
    "Cüzdan": ["cuzdan"],
    "Kemer": ["kemer"],
    "Çorap": ["corap"],
    "Şapka": ["sapka"],
    "Bere": ["bere"],
    "Atkı": ["atki"],
    "Pijama Takımı": ["pijama"],
    "Gecelik": ["gecelik"],
    "Mayo": ["mayo"],
    "Bikini": ["bikini"],
    "Parfüm": ["parfum"],
    "Tulum": ["tulum"],
    "Yelek": ["yelek"],
    "Bermuda": ["bermuda"],
    "Kimono": ["kimono"],
    "Blazer Ceket": ["blazer"],
    "Chino Pantolon": ["chino"],
    "Kargo Pantolon": ["kargo pantolon", "kargo"],
    "Boxer": ["boxer"],
}

GENDER_TERMS = {
    "Kadın": ["kadin", "bayan"],
    "Erkek": ["erkek", "bay "],
}
AGE_TERMS = {
    "Çocuk": ["cocuk", "kiz cocuk", "erkek cocuk", "genc kiz"],
    "Bebek": ["bebek"],
    "Yetişkin": ["yetiskin"],
}

# "500 tl altı", "300-700 tl", "en fazla 400", "1000 tl üzeri"
_PRICE_UNDER = re.compile(r"(\d+)\s*(?:tl|lira)?\s*(?:alti|altinda|asagi|"
                          r"gecmeyen|en fazla|max|maksimum)")
_PRICE_OVER = re.compile(r"(\d+)\s*(?:tl|lira)?\s*(?:uzeri|ustunde|"
                         r"en az|min|minimum)")
_PRICE_RANGE = re.compile(r"(\d+)\s*[-–]\s*(\d+)\s*(?:tl|lira)")

# İndirim niyeti: "indirimli erkek tişört" gibi sorgular indirimli ürünlere
# (sale_price < price) filtrelenir. "indirim", "indirimli", "indirimde",
# "indirimdeki", "outlet", "sale" hepsi yakalanır.
_DISCOUNT_RE = re.compile(r"\b(?:indirim\w*|outlet|sale)\b")


@dataclass
class QueryIntent:
    raw: str
    semantic_text: str = ""
    category: str | None = None
    color: str | None = None
    gender: str | None = None
    age_group: str | None = None
    patterns: list = field(default_factory=list)
    fabrics: list = field(default_factory=list)
    fits: list = field(default_factory=list)
    necklines: list = field(default_factory=list)
    sleeves: list = field(default_factory=list)
    lengths: list = field(default_factory=list)
    price_min: float | None = None
    price_max: float | None = None
    # "indirimli" arandıysa True: sonuçlar indirimli ürünlere daraltılır.
    # Not: active_filters() bu alanı DÖNDÜRMEZ; mevcut /search sözleşmesi korunur.
    discounted: bool = False
    # Yazım düzeltmede yapılan düzeltmeler: [{"from": "puantieli", "to": "puantiye"}]
    # Not: active_filters() bu alanı DÖNDÜRMEZ; mevcut /search sözleşmesi korunur.
    corrections: list = field(default_factory=list)

    def active_filters(self) -> dict:
        out = {}
        for k in ("category", "color", "gender", "age_group",
                  "patterns", "fabrics", "fits", "necklines",
                  "sleeves", "lengths", "price_min", "price_max"):
            v = getattr(self, k)
            if v:
                out[k] = v
        return out


def _match_longest(lexicon: dict, folded: str):
    """En uzun tetikleyiciyi önce dener ('kot pantolon' > 'kot')."""
    best, best_len = None, 0
    for canon, triggers in lexicon.items():
        for t in triggers:
            if t in folded and len(t) > best_len:
                best, best_len = canon, len(t)
    return best


def _match_all(lexicon: dict, folded: str) -> list:
    found = []
    for canon, triggers in lexicon.items():
        if any(t in folded for t in triggers):
            found.append(canon)
    return found


# ---------------------------------------------------------------------------
# Yazım düzeltme (sözlük tabanlı fuzzy eşleşme)
#
# Kesin (substring) eşleşme HER ZAMAN önce çalışır; fuzzy yalnızca kesin
# olarak eşleşmeyen ("açıklanamayan") sorgu kelimeleri için, boş kalan
# alanları doldurmak üzere devreye girer. Böylece doğru yazılmış sorguların
# mevcut davranışı bit-bit korunur; yalnızca "puantieli" gibi hatalar
# düzeltilir. Yeni bağımlılık yok: standart kütüphanedeki difflib kullanılır.
# ---------------------------------------------------------------------------

_FUZZY_MIN_LEN = 4       # bu uzunluğun altındaki kelimeler için fuzzy kapalı
_FUZZY_CUTOFF = 0.8      # difflib benzerlik eşiği (0-1); yüksek = temkinli

_SINGLE_FIELD_LEXICONS = (
    ("category", CATEGORY_SYNONYMS),
    ("gender", GENDER_TERMS),
    ("age_group", AGE_TERMS),
)
_LIST_FIELD_LEXICONS = (
    ("patterns", PATTERNS),
    ("fabrics", FABRICS),
    ("fits", FITS),
    ("necklines", NECKLINES),
    ("sleeves", SLEEVES),
    ("lengths", LENGTHS),
)


def _build_fuzzy_index() -> dict:
    """tetikleyici -> [(alan, kanonik, liste_mi), ...] eşlemesini kurar.

    Yalnızca tek kelimeli (boşluksuz), harften oluşan ve yeterince uzun
    tetikleyiciler alınır; kısa/çok anlamlı kökler yanlış düzeltmeye yol açmasın.
    """

    index: dict[str, list] = {}

    def add(trigger: str, field_name: str, canon: str, is_list: bool) -> None:
        key = trigger.strip()
        if not key.isalpha() or len(key) < _FUZZY_MIN_LEN:
            return
        entry = (field_name, canon, is_list)
        bucket = index.setdefault(key, [])
        if entry not in bucket:
            bucket.append(entry)

    for field_name, lexicon in _SINGLE_FIELD_LEXICONS:
        for canon, triggers in lexicon.items():
            for trigger in triggers:
                add(trigger, field_name, canon, False)
    for field_name, lexicon in _LIST_FIELD_LEXICONS:
        for canon, triggers in lexicon.items():
            for trigger in triggers:
                add(trigger, field_name, canon, True)
    for trigger, canon in COLOR_CANON.items():
        add(trigger, "color", canon, False)
    return index


_FUZZY_INDEX = _build_fuzzy_index()
_FUZZY_TRIGGERS = list(_FUZZY_INDEX)

# Kesin eşleşen ("açıklanan") kelimeleri artık metinden silmek için TÜM
# tetikleyiciler (çok kelimeli ve kısa olanlar dahil). Uzun olanlar önce
# silinsin ki "slim fit" kaldırıldığında geriye "slim"/"fit" kalmasın.
_ALL_TRIGGERS = sorted(
    {
        trigger.strip()
        for _field, lexicon in (*_SINGLE_FIELD_LEXICONS, *_LIST_FIELD_LEXICONS)
        for triggers in lexicon.values()
        for trigger in triggers
        if trigger.strip()
    }
    | {key.strip() for key in COLOR_CANON if key.strip()},
    key=len,
    reverse=True,
)


def _fuzzy_fill(intent: "QueryIntent", folded: str, price_stripped: str) -> None:
    """Kesin eşleşmeyen kelimeleri sözlükteki en yakın terimle tamamlar.

    Sadece boş kalan tekil alanları doldurur ve listeye yeni değer ekler;
    mevcut kesin eşleşmeleri asla ezmez.
    """

    residual = price_stripped
    for trigger in _ALL_TRIGGERS:
        if trigger in folded:
            residual = residual.replace(trigger, " ")

    for token in residual.split():
        if not token.isalpha() or len(token) < _FUZZY_MIN_LEN:
            continue
        matches = difflib.get_close_matches(
            token, _FUZZY_TRIGGERS, n=1, cutoff=_FUZZY_CUTOFF,
        )
        if not matches:
            continue
        trigger = matches[0]
        applied = False
        for field_name, canon, is_list in _FUZZY_INDEX[trigger]:
            if is_list:
                values = getattr(intent, field_name)
                if canon not in values:
                    values.append(canon)
                    applied = True
            elif getattr(intent, field_name) is None:
                setattr(intent, field_name, canon)
                applied = True
        if applied:
            intent.corrections.append({"from": token, "to": trigger})


def parse_query(q: str) -> QueryIntent:
    intent = QueryIntent(raw=q)
    folded = tr_fold(q)

    # Fiyat
    if m := _PRICE_RANGE.search(folded):
        intent.price_min, intent.price_max = float(m.group(1)), float(m.group(2))
    else:
        if m := _PRICE_UNDER.search(folded):
            intent.price_max = float(m.group(1))
        if m := _PRICE_OVER.search(folded):
            intent.price_min = float(m.group(1))

    # Fiyat ifadeleri çıkarılmış metin: hem yazım düzeltme hem anlamsal metin
    # aynı temiz gövdeyi kullanır (fiyat kelimeleri fuzzy'ye aday olmasın).
    price_stripped = _PRICE_OVER.sub(
        " ", _PRICE_UNDER.sub(" ", _PRICE_RANGE.sub(" ", folded)))

    # "indirimli" niyeti: sonuçları indirimli ürünlere (sale_price < price)
    # daraltır. Terim anlamsal metinden ve yazım düzeltme adayından çıkarılır.
    intent.discounted = bool(_DISCOUNT_RE.search(folded))
    if intent.discounted:
        price_stripped = _DISCOUNT_RE.sub(" ", price_stripped)

    # Kategori (en uzun eşleşme kazanır; Türkçe ekler için kök arama:
    # 'elbisesi', 'elbiseler' → 'elbise' kökünü içerir)
    intent.category = _match_longest(CATEGORY_SYNONYMS, folded)

    # Renk: kanonik renk sözlüğündeki anahtarlar fold edilmiş haldedir
    for key, canon in COLOR_CANON.items():
        if re.search(rf"\b{re.escape(key)}\b|{re.escape(key)}li\b", folded):
            intent.color = canon
            break

    # Cinsiyet / yaş
    intent.gender = _match_longest(GENDER_TERMS, folded)
    intent.age_group = _match_longest(AGE_TERMS, folded)
    if intent.age_group in ("Çocuk", "Bebek") and intent.gender:
        # "kız çocuk elbise" → gender filtresi feed'de Female ama yaş Kids;
        # yaş belirtilmişse cinsiyet filtresi yaş grubuyla birlikte uygulanır
        pass

    # Nitelikler
    intent.patterns = _match_all(PATTERNS, folded)
    intent.fabrics = _match_all(FABRICS, folded)
    intent.fits = _match_all(FITS, folded)
    intent.necklines = _match_all(NECKLINES, folded)
    intent.sleeves = _match_all(SLEEVES, folded)
    intent.lengths = _match_all(LENGTHS, folded)

    # Yazım düzeltme: kesin eşleşmeyen kelimeleri sözlükteki en yakın terime
    # bağla (yalnız boş alanları doldurur; mevcut kesin eşleşmeleri ezmez).
    _fuzzy_fill(intent, folded, price_stripped)

    # "erkek" araması "erkek çocuk" ürünlerini de getiriyordu: feed'de her
    # ikisinin cinsiyeti de "Erkek", ayrım yaş grubunda. Cinsiyet "Erkek" olup
    # yaş belirtilmediyse yetişkin varsay; "çocuk"/"bebek" yazıldıysa korunur.
    # (Kadın tarafında bu karışıklık olmadığı için yalnız Erkek'e uygulanır.)
    if intent.gender == "Erkek" and intent.age_group is None:
        intent.age_group = "Yetişkin"

    # 'desenli' tek başına anlamlı, ama spesifik desen varsa genel olanı at
    specific = set(intent.patterns) - {"desenli", "baskılı", "düz"}
    if specific:
        intent.patterns = sorted(specific)

    # Anlamsal metin: fiyat ifadeleri temizlenmiş ham sorgu.
    # Nitelik kelimelerini ÇIKARMAYIZ — dense vektör de bu sinyali kullansın.
    intent.semantic_text = re.sub(
        r"\s+",
        " ",
        q if price_stripped.strip() == folded.strip() else price_stripped,
    ).strip() or q
    return intent
