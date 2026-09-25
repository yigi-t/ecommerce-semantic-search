# Defacto Semantic Search

> Not: Gerçek ürün veri seti, gizlilik ve dosya boyutu nedeniyle bu depoya dahil edilmemiştir.

Defacto ürün feed'i (25.059 ürün) üzerinde çalışan, "puantiyeli kırmızı elbise" gibi çok kısıtlı Türkçe sorguları doğru anlayan hybrid arama motoru. Qdrant + çok dilli dense embedding + BM25 sparse + sorgu anlama katmanı. Arama sonuçlarındaki uygun giyim parçaları için ayrıca stok, kategori, hedef kitle, renk, desen, kalıp, sezon ve fiyat dengesini kullanan kombin önerileri sunar.

## Problem

Klasik kelime eşleşmeli arama "puantiyeli kırmızı elbise" sorgusundaki üç ayrı kısıtı (desen + renk + kategori) ayrıştıramaz; kırmızı olan her şeyi veya başlığında "elbise" geçen her ürünü döndürür. Saf vektör araması da tek başına yetmez: "puantiye" gibi nadir ama belirleyici terimler dense embedding'de sönümlenir ve alakasız desenli ürünler üst sıralara çıkar.

## Mimari

```
                      ┌──────────────────────────────────────┐
XML Feed (177 MB) ──▶ │ parse_feed.py (streaming, lxml)      │
                      │ enrich.py (nitelik çıkarımı)         │──▶ products.jsonl
                      └──────────────────────────────────────┘
                                        │
                      ┌──────────────────────────────────────┐
                      │ build_index.py                       │
                      │  dense: yapılandırılabilir çok dilli │──▶ Qdrant
                      │  sparse: BM25 (IDF sunucu tarafında) │    (hybrid koleksiyon
                      │  payload: kategori/renk/desen/fiyat… │     + payload indeksleri)
                      └──────────────────────────────────────┘

Sorgu ──▶ query_parser.py ──▶ search.py ─────────────────────▶ Sonuç
          "puantiyeli          1. kesin filtre (must)
           kırmızı elbise"     2. hybrid arama (RRF füzyonu)
           ↓                   3. az sonuçta kademeli gevşetme
          {kategori: Elbise,      + soft-boost yeniden sıralama
           renk: Kırmızı,
           desen: puantiye}
```

Kombin özelliği detaylı arama hattından ayrıdır:

```
Seçilen ürün ID'si ──▶ outfit.py ──▶ tamamlayıcı kategori filtresi
                                      + aynı cinsiyet (veya Unisex)/yaş kitlesi
                                      + gerçek stok ve çocuk beden uyumu
                                      + renk/desen/kalıp/sezon/fiyat skoru
                                      ──▶ en fazla 4, mümkün olduğunda çeşitli öneri
```

`/search`, `SearchEngine.search()`, sorgu ayrıştırma ve filtre gevşetme davranışı değişmez. Kombin isteği yalnızca kullanıcı bir sonuç kartındaki **Bununla kombinle** düğmesine bastığında yapılır; bu nedenle ana aramaya ek yük veya N+1 istek eklenmez.

### Üç katman, üç iş

1. **Index-time zenginleştirme** (`enrich.py`) — Feed'de desen, kumaş, fit, yaka, kol, boy bilgisi ayrı alan olarak yok; başlık/açıklama içinde gömülü. Türkçe moda sözlükleriyle bu nitelikler çıkarılıp yapılandırılmış payload alanlarına yazılır. Gerçek feed'de kapsam: ürünlerin %86,7'sinden en az bir nitelik çıkarılıyor (136 puantiyeli, 942 çizgili, 598 çiçekli...).

2. **Hybrid arama** — Dense vektör anlamsal yakınlığı ("kına gecesi için şık elbise" gibi serbest sorgular), sparse BM25 kelime hassasiyetini (nadir terimler) sağlar. İkisi Qdrant Query API'nin `prefetch` + RRF füzyonuyla birleştirilir. E5 ailesi seçildiğinde gereken `query:`/`passage:` biçimi kod tarafından uygulanır.

> Not: Mevcut arama davranışını korumak için `config.py` içindeki MiniLM varsayılanı değiştirilmedi; `.env.example` kalite odaklı E5-large seçeneğini gösterir. Dense modeli değiştirmek koleksiyonun yeniden indekslenmesini gerektirir.

3. **Sorgu anlama** (`query_parser.py`) — Sorgudaki kategori, renk, cinsiyet, yaş grubu, nitelikler ve fiyat ifadeleri ("500 tl altı", "300-700 tl") aynı sözlüklerle ayrıştırılır ve Qdrant'ta **kesin filtre** olur. Sözlükler enrich.py ile ortaktır; index ve sorgu tarafı asla ayrışmaz.

### Kademeli gevşetme (relaxation)

Gerçek katalog verisinde doğrulanan kritik durum: **puantiyeli + kırmızı elbise bu katalogda hiç yok** (32 puantiyeli elbise var, hiçbiri kırmızı değil). Kesin filtre boş dönerse motor kısıtları önem sırasına göre gevşetir (boy → kol → yaka → fit → kumaş → desen → renk; kategori asla gevşetilmez), gevşetilen kısıtı skorlamada soft-boost'a çevirir ve yanıtta hangi kısıtın esnetildiğini raporlar. Mevcut sırada önce desen gevşetildiği için kullanıcı boş sayfa yerine kırmızı elbiseleri en yakın alternatif olarak görür.

## Kurulum ve çalıştırma

```bash
# 1. Qdrant'ı başlat
docker compose up -d

# 2. Bağımlılıklar
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Feed'i parse et (data/products.jsonl bu repoda hazır geliyorsa atlanabilir)
python -m src.parse_feed data/veri-seti.xml data/products.jsonl

# 4. İndeksle (ilk çalıştırmada embedding modeli indirilir, ~2 GB;
#    25K ürün CPU'da ~20-40 dk, GPU'da birkaç dakika sürer.
#    Hızlı deneme için config.py'de DENSE_MODEL'i
#    intfloat/multilingual-e5-small yapabilirsiniz.)
python -m src.build_index data/products.jsonl

# 5. API + demo arayüz
uvicorn src.api:app --port 8000
# → http://localhost:8000        (demo arayüz)
# → http://localhost:8000/search?q=puantiyeli+siyah+elbise
# → http://localhost:8000/products/B0651AXNSWT34/outfit-recommendations?limit=4
# → http://localhost:6333/dashboard  (Qdrant paneli)
```

## API yanıt örneği

```json
{
  "query": "puantiyeli kırmızı elbise",
  "understood": {"category": "Elbise", "color": "Kırmızı", "patterns": ["puantiye"]},
  "relaxed": ["patterns"],
  "count": 24,
  "products": [{"title": "Kuşaklı Kısa Kollu Midi Elbise", "color": "Kırmızı", "sale_price": 399.99, "...": "..."}]
}
```

`understood` alanı arayüzde "Şunu aradınız: Elbise · Kırmızı · puantiye" chip'leri olarak, `relaxed` ise hangi kısıtın esnetildiğini açıklayan mesaj olarak gösterilir — demo arayüz ikisini de yapıyor.

## Kombin önerisi API'si

```json
{
  "source": {
    "id": "B0651AXNSWT34",
    "title": "Oversıze Pamuklu Kısa Kollu Tişört",
    "category": "Tişört",
    "color": "Beyaz"
  },
  "eligible": true,
  "title": "Bu üstü tamamlayan alt giyim önerileri",
  "summary": "Renk, ürün türü, yaş grubu ve fiyat dengesi birlikte değerlendirildi.",
  "target_slot": "Alt giyim",
  "target_categories": ["Pantolon", "Jean Pantolon", "Etek", "Şort"],
  "count": 4,
  "products": [
    {
      "id": "...",
      "category": "Pantolon",
      "color": "Siyah",
      "reason": "Siyah tonu, beyaz parçayla dengeli bir palet kurar."
    }
  ]
}
```

Kesin kurallar stokta olma, `total_stock > 0`, tamamlayıcı kategori, aynı cinsiyet (veya gerçek bir `Unisex` ürün) ve aynı yaş kitlesidir. Bebek ve büyük çocuk ürünleri `category_path` üzerinden ayrılır; çocuk ürünlerinde ortak, stoktaki beden aranır. Renk, başlıktaki desen/kalıp, SKU sezon kodu, ürün türüne göre beklenen fiyat oranı ve satış popülerliği yalnızca sıralama sinyalidir. Sonuçlar ürün kimliği ve başlığıyla tekilleştirilir; kategori/renk çeşitliliği katalog izin verdiği ölçüde uygulanır. `Jean` kategorisi etek ve ceket de içerdiği için yalnızca başlığında açıkça pantolon yazan kaynak ürünler alt giyim kabul edilir; kombin adayı olarak belirsiz `Jean` kayıtları kullanılmaz. Kozmetik ve benzeri desteklenmeyen kaynak ürünler `eligible: false` ve boş ürün listesi döndürür.

### Düzenlenebilir kombin

Her öneri kartı mevcut seçimi koruyarak üç şekilde değiştirilebilir:

- `preference=alternative`: o anda ekranda görünenleri ve aynı kartta daha önce gösterilenleri dışlayıp sıradaki uyumlu parçayı getirir.
- `preference=cheaper`: mevcut tamamlayıcıyla aynı kategoride, kesin olarak daha düşük fiyatlı bir ürün getirir.
- `preference=different_color`: aynı kategoride, mevcut renkten farklı en yakın ürünü getirir.

```text
GET /products/{kaynak_id}/outfit-recommendations
    ?limit=1
    &preference=cheaper
    &current_product_id={mevcut_tamamlayici_id}
    &exclude_product_id={daha_once_gosterilen_id}
```

Düzenleme çağrıları ilk öneri cache'ini değiştirmez. Alternatif bulunamazsa başarılı ve boş bir ürün listesi döner; arayüz mevcut kartı yerinde tutar. Varsayılan `limit=4` kombin çağrısı ve detaylı arama hattı değişmeden çalışmaya devam eder.

### Bedenimi hatırla

Arayüz harfli giyim, sayısal giyim, jean bel/boy, ayakkabı ve çocuk/bebek bedenlerini türleri karıştırmadan bu cihazın `localStorage` alanında saklar. Arama sonuçları gizlenmez veya yeniden sıralanmaz; kartlarda hatırlanan bedenin stok durumu gösterilir. Son gezilen ve kaydedilen kartlar yerel anlık görüntüler olduğu için bu alanlarda durum açıkça “son görüntülendiğinde” şeklinde belirtilir. Kombin isteklerinde uygun bedenler tekrarlanan `preferred_size` parametreleriyle gönderilir ve yalnız bu bedenlerden en az biri stokta olan tamamlayıcılar değerlendirilir. Kemer, çanta, takı ve yüzük kendi ölçü sistemleri nedeniyle giyim filtresinden muaftır; `STD` ve `One Size` ürünleri standart beden olarak kabul edilir.

```text
GET /products/{kaynak_id}/outfit-recommendations
    ?limit=4
    &preferred_size=M
    &preferred_size=S%2FM
    &preferred_size=38
```

Beden tercihi yokken kombin endpoint'inin önceki filtreleri, sonuçları ve sıralaması aynen korunur. Kullanıcı “Bedenlerimi unut” düğmesiyle yalnız beden kaydını temizleyebilir; son gezilen ürünler ve kaydedilen kombinler etkilenmez.

### Sana özel ürünler

`Sana özel` sekmesi yalnız kullanıcı açtığında çalışır. Tarayıcı, cihazda zaten tutulan son 10 ürünün ve en fazla 20 kaydedilmiş kombinin yalnız ürün kimliklerini yeni ve bağımsız endpoint'e gönderir; arama metinleri kaydedilmez ve sunucuda kalıcı kullanıcı profili oluşturulmaz.

```text
POST /personalized-recommendations
{
  "recent_product_ids": ["..."],
  "saved_outfits": [
    {"source_product_id": "...", "recommended_product_id": "..."}
  ],
  "size_preferences": {"alpha": "M", "shoe": "39"},
  "exclude_product_ids": ["..."],
  "limit": 12
}
```

Backend kimliklerin güncel ürün verilerini Qdrant'tan toplu olarak doğrular. Kaydedilmiş kombinler daha güçlü, son gezilen ürünler ise yeniliğine göre azalan sinyal olur. Adaylar mevcut düşük seviyeli hybrid aramayla bulunur; ardından özellik ve fiyat yakınlığı ile küçük bir satış-popülerliği katkısı kullanılarak yeniden sıralanır. Stok, `total_stock`, aktif cinsiyet/yaş kitlesi, çocuk-bebek ayrımı ve ilgili hatırlanan beden güvenli filtrelerdir. Görülmüş, kaydedilmiş veya açıkça dışlanmış ürünler tekrar önerilmez; başlık, kategori ve renk çeşitliliği korunur. Karttaki “neden önerildi” metni yalnız gerçekten eşleşen katalog özelliklerinden üretilir.

Hiç geçerli etkileşim yoksa popüler ürünler kişiselmiş gibi gösterilmez; arayüz kullanıcıyı birkaç ürüne göz atmaya veya kombin kaydetmeye yönlendirir. `Önerileri yenile` çağrısı ekrandaki ürünleri dışlar ve yeni alternatif arar. Bu akış `/search`, sorgu ayrıştırma, filtre gevşetme ve kombin endpoint'lerinden tamamen ayrıdır.

## Testler

Canlı model veya Qdrant gerektirmeyen hızlı regresyon paketi:

```bash
python -m unittest discover -s tests -v
```

Testler mevcut `/search` üst seviye sözleşmesini, README sorgularının ayrıştırılmasını ve gerçek JSONL katalog sayımlarını, mevcut filtre gevşetme sırasını; kombin ve kişiselleştirme tarafındaki stok/kategori/cinsiyet/yaş/beden/tekilleştirme kurallarını sabitler. Geçici bir in-memory Qdrant koleksiyonu ayrıca deterministik UUID ile ürün bulma ve gerçek payload filtre sözleşmesini doğrular.

## Doğrulanmış sorgu örnekleri (gerçek feed üzerinde)

| Sorgu | Anlaşılan | Kesin eşleşme |
|---|---|---|
| puantiyeli siyah elbise | Elbise · Siyah · puantiye | 10 ürün |
| 500 tl altı keten gömlek | Gömlek · keten · fiyat≤500 | 52 ürün |
| oversize siyah sweatshirt kadın | Sweatshirt · Siyah · Kadın · oversize | 15 ürün |
| erkek slim fit jean | Jean · Erkek · slim fit | 17 ürün |
| çiçekli midi etek | Etek · çiçekli · midi | 15 ürün |
| v yaka triko kazak bordo | Kazak · Bordo · triko · v yaka | 2 ürün |
| puantiyeli kırmızı elbise | Elbise · Kırmızı · puantiye | 0 → gevşetme devreye girer |

## Dosya yapısı

```
├── config.py              # model isimleri, Qdrant adresi, eşikler
├── docker-compose.yml     # Qdrant
├── requirements.txt
├── data/products.jsonl    # parse edilmiş + zenginleştirilmiş 25.059 ürün
├── static/index.html      # demo arayüz
├── tests/                 # arama sözleşmesi + kombin regresyon testleri
└── src/
    ├── parse_feed.py      # streaming XML → JSONL
    ├── enrich.py          # nitelik çıkarımı + normalizasyon (ortak sözlükler)
    ├── query_parser.py    # sorgu → yapılandırılmış niyet
    ├── build_index.py     # Qdrant hybrid koleksiyon + indeksleme
    ├── search.py          # hybrid arama + kademeli gevşetme
    ├── outfit.py          # tamamlayıcı kategori + uyumluluk sıralaması
    ├── personalized.py    # yerel etkileşimlerden stateless kişisel öneriler
    └── api.py             # FastAPI
```

## Üretime taşırken yol haritası

- **Feed güncelleme:** parse+index'i cron'a bağlayıp yeni koleksiyona indeksleyip alias swap yapın (Qdrant `update_collection_aliases`) — kesintisiz güncelleme.
- **Sıralama sinyalleri:** feed'deki `weekly_sales`, `productView`, stok derinliği zaten payload'da; RRF skoruna popülerlik boost'u eklenebilir.
- **Yazım düzeltme:** "puantieli" gibi hatalar için sözlük tabanlı fuzzy eşleşme (rapidfuzz) query_parser'a eklenebilir; sparse BM25 kısmen tolere eder ama kesin filtre kaçırır.
- **Görsel arama:** aynı Qdrant koleksiyonuna CLIP/SigLIP görsel vektörü üçüncü named vector olarak eklenerek "benzer ürünler" ve fotoğrafla arama açılır.
- **Ölçüm:** tıklama loglarından offline değerlendirme seti (sorgu → beğenilen ürün) kurup nDCG ile model/ağırlık değişikliklerini kıyaslayın.

Not: data/products.jsonl.gz dosyasini kullanmadan once acin:
  gunzip -k data/products.jsonl.gz
