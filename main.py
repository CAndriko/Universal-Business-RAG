import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# --- KONFIGURATION ---
# Lädt Umgebungsvariablen für Sicherheit (Google-Standard)
load_dotenv()

# Prüfen, ob der API Key existiert
if not os.getenv("OPENAI_API_KEY"):
    print("FATAL ERROR: Kein API-Key gefunden. Bitte .env Datei prüfen.")
    sys.exit(1)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Liest Text professionell aus PDF-Dateien.
    Ignoriert leere Seiten und fängt Lesefehler ab.
    """
    try:
        reader = PdfReader(pdf_path)
        text = []
        for page_num, page in enumerate(reader.pages):
            content = page.extract_text()
            if content:
                text.append(content)
            else:
                print(f"Warnung: Seite {page_num + 1} ist leer oder nicht lesbar.")
        return "\n".join(text)
    except Exception as e:
        print(f"Error beim PDF-Lesen: {e}")
        return None

def ask_business_bot(context: str, question: str) -> str:
    """
    Core-Logik: Verbindet Dokumenten-Wissen mit GPT-4o Intelligenz.
    """
    system_prompt = (
        "Du bist ein hochspezialisierter Business-Analyst. "
        "Deine Aufgabe ist es, Fragen basierend auf dem folgenden Dokumenten-Kontext präzise zu beantworten. "
        "Wenn die Antwort nicht im Text steht, sage höflich, dass das Dokument dazu keine Infos enthält. "
        "Antworte professionell und auf den Punkt."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"KONTEXT:\n{context}\n\nFRAGE DES KUNDEN:\n{question}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API Connection Error: {e}"

def main():
    print("\n" + "="*50)
    print("   UNIVERSAL BUSINESS BOT (RAG SYSTEM) v1.0")
    print("   Enterprise Edition - Ready for Analysis")
    print("="*50 + "\n")

    pdf_file = "daten.pdf"

    if not os.path.exists(pdf_file):
        print(f"❌ Setup-Fehler: Datei '{pdf_file}' fehlt im Verzeichnis.")
        print("Bitte PDF einfügen und Neustarten.")
        return

    print(f"📂 Lade Dokument: {pdf_file}...")
    document_content = extract_text_from_pdf(pdf_file)

    if not document_content:
        print("❌ Abbruch: Dokument konnte nicht gelesen werden.")
        return

    print("✅ System bereit. Integrierte KI: GPT-4o.\n")

    while True:
        try:
            user_input = input("Frage stellen (oder 'exit'): ")
            if user_input.lower() in ['exit', 'quit', 'ende']:
                print("System wird heruntergefahren...")
                break
            
            print("⏳ Analysiere...")
            answer = ask_business_bot(document_content, user_input)
            print(f"\n🤖 ANTWORT:\n{answer}\n")
            print("-" * 30)
            
        except KeyboardInterrupt:
            print("\nAbbruch durch Benutzer.")
            break

if __name__ == "__main__":
    main()