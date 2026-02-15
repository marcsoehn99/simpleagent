import os
import asyncio
from dotenv import load_dotenv
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from openai import OpenAI # Wir brauchen den Standard-Client für die Query-Generierung

from agents import Agent, Runner, function_tool

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI(api_key=api_key)

# 1. Datenbank-Verbindung (lesend)
client_db = chromadb.PersistentClient(path="./chroma_db")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key,
    model_name="text-embedding-3-small"
)
collection = client_db.get_collection(name="dokumente", embedding_function=openai_ef)

@function_tool
def durchsuche_dokumente(user_frage: str) -> str:
    """Nutze dieses Tool, um interne Dokumente zu durchsuchen. Gibt Text mit Quellenangaben zurück."""
    print(f"   [Multi-Query Engine startet für: '{user_frage}']")
    
    # Schritt A: Aus einer Frage drei Suchbegriffe machen
    prompt = f"Generiere 3 kurze, unterschiedliche Suchbegriffe in Deutsch für eine Vektordatenbank, um diese Frage zu beantworten: '{user_frage}'. Gib nur die Begriffe kommasepariert aus."
    response = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    
    suchbegriffe = [s.strip() for s in response.choices[0].message.content.split(",")]
    suchbegriffe.append(user_frage)
    print(f"   [Suche nach (inkl. Original): {suchbegriffe}]")
    
    # Schritt B: Alle Begriffe gleichzeitig an ChromaDB schicken
    ergebnisse = collection.query(
        query_texts=suchbegriffe,
        n_results=2
    )
    
    kontext_mit_quellen = []
    
    # Wir iterieren über die Dokumente UND deren Metadaten
    for i in range(len(ergebnisse["documents"])):
        for j in range(len(ergebnisse["documents"][i])):
            text = ergebnisse["documents"][i][j]
            quelle = ergebnisse["metadatas"][i][j]["filename"] # Hier holen wir den Dateinamen
            
            # Wir formatieren den Schnipsel so, dass der Agent die Quelle sieht
            eintrag = f"[QUELLE: {quelle}]\nINHALT: {text}"
            
            if eintrag not in kontext_mit_quellen:
                kontext_mit_quellen.append(eintrag)
    
    if not kontext_mit_quellen:
        return "Keine relevanten Informationen gefunden."
        
    return "\n\n---\n\n".join(kontext_mit_quellen)

# 3. Agenten-Konfiguration
agent = Agent(
    name="Profi-RAG-Assistent",
    instructions=(
        "Du bist ein präziser Assistent. Beantworte Fragen nur basierend auf dem gefundenen Kontext. "
        "WICHTIG: Gib am Ende deiner Antwort oder direkt im Satz immer die Quelle an (z.B. 'Laut Dokument [Dateiname]'). "
        "Wenn du Informationen aus verschiedenen Quellen kombinierst, nenne alle beteiligten Dateinamen."
    ),
    tools=[durchsuche_dokumente]
)

async def main():
    # Teste es mit einer komplexeren Frage
    #frage = "Gibt es Dokumente von der Zimmerei Masch und was wird dort angeboten?"
    #frage = "wer hat das zwischenzeugnis für marc söhn erstellt?"
    frage = "auf welches konto im skr04 werden fremdleistungen gebucht?"


    print(f"🗣️ User: {frage}")
    result = await Runner.run(agent, input=frage)
    
    print("\n✅ Finale Antwort:")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())