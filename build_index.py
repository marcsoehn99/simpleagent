import json
import os
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

# Alte Collections aufräumen
for name in ["propositions", "chunks"]:
    try:
        client.delete_collection(name)
    except Exception:
        pass

# Zwei Collections: Propositions (Suche) + Chunks (Kontext)
prop_collection = client.create_collection(
    name="propositions",
    embedding_function=openai_ef
)
chunk_collection = client.create_collection(
    name="chunks",
    embedding_function=openai_ef
)

# Daten einlesen
print("Lese transformiert.json ein...")
with open("transformiert.json", "r", encoding="utf-8") as f:
    daten = json.load(f)

BATCH_SIZE = 50

for doc in daten:
    doc_id = doc.get("doc_id", "unknown")

    # Batches vorbereiten
    prop_ids, prop_docs, prop_metas = [], [], []
    chunk_ids, chunk_docs, chunk_metas = [], [], []

    for chunk in doc.get("chunks", []):
        chunk_id = chunk["chunk_id"]
        title = chunk.get("title", "")
        content = chunk.get("content", "")
        source_pages = str(chunk.get("source_pages", []))

        # Chunk in Kontext-Collection (wird nicht durchsucht, nur per ID abgerufen)
        chunk_ids.append(chunk_id)
        chunk_docs.append(content)
        chunk_metas.append({
            "doc_id": doc_id,
            "title": title,
            "source_pages": source_pages
        })

        # Propositions in Such-Collection
        for i, prop_text in enumerate(chunk.get("propositions", [])):
            prop_ids.append(f"{chunk_id}_p{i}")
            prop_docs.append(prop_text)
            prop_metas.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "title": title
            })

    # Batch-Insert: Chunks
    for start in range(0, len(chunk_ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        chunk_collection.add(
            ids=chunk_ids[start:end],
            documents=chunk_docs[start:end],
            metadatas=chunk_metas[start:end]
        )

    # Batch-Insert: Propositions
    for start in range(0, len(prop_ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        prop_collection.add(
            ids=prop_ids[start:end],
            documents=prop_docs[start:end],
            metadatas=prop_metas[start:end]
        )

    print(f"  {doc_id}: {len(chunk_ids)} Chunks, {len(prop_ids)} Propositions")

print(f"\nFertig! {prop_collection.count()} Propositions + {chunk_collection.count()} Chunks in ChromaDB")