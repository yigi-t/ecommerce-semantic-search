# -*- coding: utf-8 -*-
"""
Qdrant indeksleme: dense (çok dilli E5) + sparse (BM25) hybrid koleksiyon.

Neden hybrid?
  - Dense vektör "kına gecesi için şık elbise" gibi serbest/anlamsal
    sorguları yakalar ama "puantiye" gibi nadir, ayırt edici terimleri
    sönümleyebilir.
  - Sparse BM25 tam tersini yapar: nadir terime yüksek ağırlık verir
    ama eş anlamlıları/niyet ifadesini anlamaz.
  - RRF füzyonu ikisinin sıralamalarını birleştirir; literatürde ve
    e-ticaret pratiğinde tek başına ikisinden de tutarlı şekilde iyidir.

Kullanım:
    python -m src.build_index data/products.jsonl
"""

import json
import sys
import uuid

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

import config


def make_client() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)


def create_collection(client: QdrantClient, dense_dim: int):
    if client.collection_exists(config.COLLECTION):
        client.delete_collection(config.COLLECTION)

    client.create_collection(
        collection_name=config.COLLECTION,
        vectors_config={
            "dense": models.VectorParams(
                size=dense_dim,
                distance=models.Distance.COSINE,
                on_disk=True,
            )
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF  # BM25 için IDF sunucu tarafında
            )
        },
        optimizers_config=models.OptimizersConfigDiff(default_segment_number=2),
    )

    # Filtrelenecek her payload alanına indeks — 25K üründe şart değil ama
    # katalog büyüdüğünde filtreli aramanın hızını korur.
    keyword_fields = ["category", "color", "gender", "age_group",
                      "patterns", "fabrics", "fits", "necklines",
                      "sleeves", "lengths", "availability",
                      "in_stock_sizes"]
    for f in keyword_fields:
        client.create_payload_index(config.COLLECTION, f,
                                    models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(config.COLLECTION, "sale_price",
                                models.PayloadSchemaType.FLOAT)
    client.create_payload_index(config.COLLECTION, "weekly_sales",
                                models.PayloadSchemaType.FLOAT)
    client.create_payload_index(config.COLLECTION, "total_stock",
                                models.PayloadSchemaType.INTEGER)


def index_products(jsonl_path: str):
    client = make_client()
    dense_model = TextEmbedding(
        config.DENSE_MODEL, cache_dir=config.FASTEMBED_CACHE_DIR
    )
    sparse_model = SparseTextEmbedding(
        config.SPARSE_MODEL, cache_dir=config.FASTEMBED_CACHE_DIR
    )

    dense_dim = len(next(dense_model.embed(["boyut testi"])))
    create_collection(client, dense_dim)

    batch, total = [], 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            batch.append(json.loads(line))
            if len(batch) >= config.BATCH_SIZE:
                total += _flush(client, dense_model, sparse_model, batch)
                print(f"\rindekslenen: {total}", end="", flush=True)
                batch = []
    if batch:
        total += _flush(client, dense_model, sparse_model, batch)
    print(f"\rindekslenen: {total} ürün — tamamlandı")


def _flush(client, dense_model, sparse_model, batch) -> int:
    # E5 pasaj öneki: model bu formatla eğitildi, atlanırsa kalite düşer
    texts = ["passage: " + p["search_text"] for p in batch]
    dense_vecs = list(dense_model.embed(texts))
    sparse_vecs = list(sparse_model.embed([p["search_text"] for p in batch]))

    points = []
    for p, dv, sv in zip(batch, dense_vecs, sparse_vecs):
        payload = {k: v for k, v in p.items() if k != "search_text"}
        points.append(models.PointStruct(
            # Ürün ID'si string; Qdrant için deterministik UUID'e çevrilir
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, p["id"])),
            vector={
                "dense": dv.tolist(),
                "bm25": models.SparseVector(
                    indices=sv.indices.tolist(), values=sv.values.tolist()),
            },
            payload=payload,
        ))
    client.upsert(config.COLLECTION, points=points, wait=True)
    return len(points)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/products.jsonl"
    index_products(path)
