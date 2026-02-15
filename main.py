import os
import asyncio
from dotenv import load_dotenv
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal

from agents import Agent, Runner, function_tool


class Quelle(BaseModel):
    datei: str
    seite: int
    absatz: str

class ResearcherErgebnis(BaseModel):
    userprompt: str
    generierte_antwort: str
    citations: list[Quelle]
    confidence: float

class CriticErgebnis(BaseModel):
    reasoning: str
    verwendete_quellen: list[str]
    decision: Literal["bestaetigt", "korrigiert"]
    confidence: float
    gepruefte_antwort: str

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI(api_key=api_key)

# 1. Datenbank-Verbindung (lesend)
client_db = chromadb.PersistentClient(path="./chroma_db")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key,
    model_name="text-embedding-3-small"
)
prop_collection = client_db.get_collection(name="propositions", embedding_function=openai_ef)
chunk_collection = client_db.get_collection(name="chunks", embedding_function=openai_ef)

@function_tool
def durchsuche_dokumente(user_frage: str) -> str:
    """Nutze dieses Tool, um interne Dokumente zu durchsuchen. Gibt Text mit Quellenangaben zurück."""
    print(f"   [Multi-Query Engine startet für: '{user_frage}']")
    
    prompt = f"Generiere 3 kurze, unterschiedliche Suchbegriffe in Deutsch für eine Vektordatenbank, um diese Frage zu beantworten: '{user_frage}'. Gib nur die Begriffe kommasepariert aus."
    response = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    
    suchbegriffe = [s.strip() for s in response.choices[0].message.content.split(",")]
    suchbegriffe.append(user_frage)
    print(f"   [Suche nach (inkl. Original): {suchbegriffe}]")
    
    # SCHRITT 1: Suche in den atomaren Propositionen (Laser-Präzision)
    prop_ergebnisse = prop_collection.query(
        query_texts=suchbegriffe,
        n_results=3 # Wir holen die 3 besten Fakten pro Suchbegriff
    )
    
    # SCHRITT 2: Chunk-IDs sammeln und deduplizieren
    gefundene_chunk_ids = set()
    for metas in prop_ergebnisse["metadatas"]:
        for meta in metas:
            if meta and "chunk_id" in meta:
                gefundene_chunk_ids.add(meta["chunk_id"])
                
    if not gefundene_chunk_ids:
        return "Keine relevanten Informationen gefunden."
        
    print(f"   [Small-to-Big] Lade volle Kontexte für Chunks: {list(gefundene_chunk_ids)}")

    # SCHRITT 3: Vollständigen Kontext aus der Chunk-Collection holen
    chunk_ergebnisse = chunk_collection.get(
        ids=list(gefundene_chunk_ids)
    )
    
    # SCHRITT 4: Für den LLM aufbereiten
    kontext_mit_quellen = []
    for i, doc_text in enumerate(chunk_ergebnisse["documents"]):
        meta = chunk_ergebnisse["metadatas"][i]
        quelle = meta.get("doc_id", "Unbekannt")
        titel = meta.get("title", "")
        seiten = meta.get("source_pages", "")
        
        eintrag = f"[QUELLE: {quelle} | ABSCHNITT: {titel} | SEITE(N): {seiten}]\nINHALT:\n{doc_text}"
        kontext_mit_quellen.append(eintrag)
        
    return "\n\n---\n\n".join(kontext_mit_quellen)


critic_agent = Agent(
    name="csa_auditor",
    instructions=(
        "Du bist ein extrem strenger Auditor für Cloud Security. "
        "Du erhältst eine vorgeschlagene Antwort und die dazugehörigen Quellen. "
        "PRÜFAUFTRAG: "
        "1. Lies die Quellen aufmerksam durch. "
        "2. Prüfe, ob die generierte Antwort zu 100% durch diese spezifischen Quellen belegt ist. "
        "3. Wenn auch nur ein Detail der Antwort nicht in den Quellen steht, MUSST du 'korrigiert' wählen und den Fehler im reasoning benennen. "
        "4. Wenn alles exakt belegt ist, wähle 'bestaetigt'. "
        "5. BEFÜLLUNG DES FELDES 'gepruefte_antwort': "
        "   - Wenn decision='bestaetigt': Kopiere die Antwort des Researchers EXAKT in dieses Feld. "
        "   - Wenn decision='korrigiert': Schreibe in dieses Feld eine eigene, korrekte Antwort, die NUR auf den Fakten der Quellen basiert. "
        "Nutze niemals externes Wissen!"
    ),
    output_type=CriticErgebnis
)


researcher_agent = Agent(
    name="csa_researcher",
    instructions=(
        "Du bist ein technischer Researcher für Cloud Security Alliance (CSA) Compliance. "
        "Deine einzige Aufgabe ist es, Fakten zu einer spezifischen CSA-Anforderung zu finden. "
        "1. Nutze IMMER dein Tool 'durchsuche_dokumente'. "
        "2. Beantworte die Frage AUSSCHLIESSLICH basierend auf den gefundenen Texten. "
        "3. Erfinde niemals eigenes Wissen. "
        "4. Bereite die gefundenen Quellen und deine Antwort präzise für den nachfolgenden Auditor vor."
        "5. Übergebe nach den Schritten 1-4 an den csa_auditor um die finale Antwort zu erhalten."
    ),
    tools=[durchsuche_dokumente],
    output_type=ResearcherErgebnis,
    handoff_description="Nutze dies, sobald du alle Fakten gefunden und deine Antwort formuliert hast, um sie zur strengen Prüfung an den Auditor zu übergeben.",
    handoffs=[critic_agent]
)

async def main():
    # Teste es mit einer komplexeren Frage
    #frage = "Gibt es Dokumente von der Zimmerei Masch und was wird dort angeboten?"
    #frage = "wer hat das zwischenzeugnis für marc söhn erstellt?"
    frage = "auf welches konto im skr04 werden fremdleistungen gebucht?"


    print(f"🗣️ User: {frage}")
    result = await Runner.run(researcher_agent, input=frage)
    
    print("\n✅ Finale Antwort:")
    print(result.final_output)
    print("***********************************************************")
    print("***********************************************************")
    print("***********************************************************")
    print("***********************************************************")
    print(f"Entscheidung: {result.final_output.decision}")
    print(f"Antwort: {result.final_output.gepruefte_antwort}")

if __name__ == "__main__":
    asyncio.run(main())