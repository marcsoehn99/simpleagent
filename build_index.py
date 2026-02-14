import json
import os
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv

# 1. API Key laden
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# 2. ChromaDB Client Setup (speichert Daten lokal im Ordner "chroma_db")
client = chromadb.PersistentClient(path="./chroma_db")

# 3. Embedding Funktion von OpenAI festlegen
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key,
    model_name="text-embedding-3-small"
)

# 4. Collection erstellen (wie eine Tabelle in einer Datenbank)
collection = client.get_or_create_collection(
    name="dokumente",
    embedding_function=openai_ef
)

# 5. JSON einlesen
print("Lese ergebnisse.json ein...")
with open("data/ergebnisse.json", "r", encoding="utf-8") as f:
    daten = json.load(f)

# 6. Daten vorbereiten und in ChromaDB laden
for i, doc in enumerate(daten):
    filename = doc.get("filename", f"doc_{i}")
    doc_type = doc.get("type", "unknown")
    
    # Textinhalt intelligent extrahieren
    text_content = ""
    if doc_type in ["text", "pdf"]:
        text_content = str(doc.get("content", ""))
    elif doc_type == "image":
        content_obj = doc.get("content", {})
        
        # Wir bauen einen strukturierten Text aus allen Feldern zusammen
        text_content = (
            f"Firma/Unternehmen: {content_obj.get('unternehmen', 'Unbekannt')}\n"
            f"Ort: {content_obj.get('ort', 'Unbekannt')}\n"
            f"Datum: {content_obj.get('datum', 'Unbekannt')}\n"
            f"Unterzeichner / Aussteller: {content_obj.get('unterzeichner', 'Unbekannt')}\n"
            f"Inhalt/Beschreibung: {content_obj.get('beschreibung', '')}"
        )

    # Nur hinzufügen, wenn auch wirklich Text vorhanden ist
    if text_content.strip():
        collection.add(
            documents=[text_content],
            metadatas=[{"filename": filename, "type": doc_type}],
            ids=[filename] # Der Dateiname dient als eindeutige ID
        )
        print(f"✅ {filename} wurde vektorisiert und gespeichert!")

print("🎉 Setup abgeschlossen! Die Vektordatenbank ist bereit.")