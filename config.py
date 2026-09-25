# -*- coding: utf-8 -*-
"""Merkezi yapılandırma. Ortam değişkenleriyle ezilebilir."""

import os
from pathlib import Path

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION = os.getenv("COLLECTION", "defacto_products")

# Dense model: Türkçe'de güçlü, çok dilli. E5 ailesi "query: " / "passage: "
# önekleriyle eğitilmiştir — build_index ve search bunu otomatik uygular.
DENSE_MODEL = os.getenv("DENSE_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# Sparse model: BM25 (fastembed). Kelime düzeyinde hassasiyet sağlar;
# "puantiye" gibi nadir terimlerin dense vektörde sönümlenmesini telafi eder.
SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")

# FastEmbed'in varsayılanı macOS'un geçici klasörüdür. Model dosyalarını
# kalıcı kullanıcı önbelleğinde tutarak her açılışta yeniden indirilmesini önle.
FASTEMBED_CACHE_DIR = os.path.expanduser(
    os.getenv("FASTEMBED_CACHE_PATH", str(Path.home() / ".cache" / "fastembed"))
)

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "256"))

# Arama davranışı
PREFETCH_LIMIT = 100      # füzyon öncesi her koldan aday sayısı
MIN_RESULTS_BEFORE_RELAX = 5   # bundan az sonuç varsa filtre gevşetilir

# Kombin davranışı (ana arama sıralamasından tamamen bağımsız)
OUTFIT_CANDIDATE_POOL = int(os.getenv("OUTFIT_CANDIDATE_POOL", "80"))
