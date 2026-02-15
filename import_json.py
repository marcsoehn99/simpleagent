"""
Fügt eine transformiert.json in die bestehende ChromaDB ein,
ohne vorhandene Collections zu löschen.

Usage:
    python import_json.py                          # Standard: transformiert.json
    python import_json.py meine_datei.json         # Beliebige JSON-Datei
"""

import json
import os
import sys
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# ChromaDB Setup
client = chromadb.PersistentClient(path="./chroma_db")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key,
    model_name="text-embedding-3-small"
)

# Bestehende Collections holen (oder erstellen falls nicht vorhanden)
chunk_collection = client.get_or_create_collection(
    name="chunks",
    embedding_function=openai_ef
)
prop_collection = client.get_or_create_collection(
    name="propositions",
    embedding_function=openai_ef
)

# JSON-Datei bestimmen
json_path = sys.argv[1] if len(sys.argv) > 1 else "transformiert.json"

print(f"Lese {json_path} ein...")
with open(json_path, "r", encoding="utf-8") as f:
    daten = json.load(f)

BATCH_SIZE = 50
total_chunks = 0
total_props = 0

for doc in daten:
    doc_id = doc.get("doc_id", "unknown")

    chunk_ids, chunk_docs, chunk_metas = [], [], []
    prop_ids, prop_docs, prop_metas = [], [], []

    for chunk in doc.get("chunks", []):
        chunk_id = chunk["chunk_id"]
        title = chunk.get("title", "")
        content = chunk.get("content", "")
        source_pages = str(chunk.get("source_pages", []))

        chunk_ids.append(chunk_id)
        chunk_docs.append(content)
        chunk_metas.append({
            "doc_id": doc_id,
            "title": title,
            "source_pages": source_pages
        })

        # Propositions einfügen, falls vorhanden
        for i, prop_text in enumerate(chunk.get("propositions", [])):
            prop_ids.append(f"{chunk_id}_p{i}")
            prop_docs.append(prop_text)
            prop_metas.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "title": title
            })

    # Batch-Insert: Chunks (upsert um Duplikate zu vermeiden)
    for start in range(0, len(chunk_ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        chunk_collection.upsert(
            ids=chunk_ids[start:end],
            documents=chunk_docs[start:end],
            metadatas=chunk_metas[start:end]
        )

    # Batch-Insert: Propositions
    for start in range(0, len(prop_ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        prop_collection.upsert(
            ids=prop_ids[start:end],
            documents=prop_docs[start:end],
            metadatas=prop_metas[start:end]
        )

    total_chunks += len(chunk_ids)
    total_props += len(prop_ids)
    print(f"  {doc_id}: {len(chunk_ids)} Chunks, {len(prop_ids)} Propositions eingefügt")

print(f"\nImport fertig! {total_chunks} Chunks + {total_props} Propositions hinzugefügt.")
print(f"Gesamt in DB: {chunk_collection.count()} Chunks, {prop_collection.count()} Propositions")
