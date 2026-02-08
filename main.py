import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# --- CONFIGURATION ---
# Load environment variables for security (Industry Standard)
load_dotenv()

# Check for API Key
if not os.getenv("OPENAI_API_KEY"):
    print("FATAL ERROR: No API key found. Please check your .env file.")
    sys.exit(1)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from PDF files with error handling.
    Skips empty pages and manages reading errors.
    """
    try:
        reader = PdfReader(pdf_path)
        text = []
        for page_num, page in enumerate(reader.pages):
            content = page.extract_text()
            if content:
                text.append(content)
            else:
                print(f"Warning: Page {page_num + 1} is empty or unreadable.")
        return "\n".join(text)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def ask_business_bot(context: str, question: str) -> str:
    """
    Core Logic: RAG (Retrieval-Augmented Generation) using GPT-4o.
    """
    system_prompt = (
        "You are a highly specialized Business Analyst. "
        "Your task is to answer questions precisely based on the provided document context. "
        "If the answer is not in the text, politely state that the document does not contain this information. "
        "Keep answers professional and concise."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}"}
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

    # Expecting a file named 'data.pdf'
    pdf_file = "data.pdf"

    if not os.path.exists(pdf_file):
        print(f"❌ Setup Error: File '{pdf_file}' missing in directory.")
        print("Please place a PDF file named 'data.pdf' in this folder and restart.")
        return

    print(f"📂 Loading document: {pdf_file}...")
    document_content = extract_text_from_pdf(pdf_file)

    if not document_content:
        print("❌ Aborted: Could not read document.")
        return

    print("✅ System Ready. Integrated Model: GPT-4o.\n")

    while True:
        try:
            user_input = input("Enter question (or 'exit'): ")
            if user_input.lower() in ['exit', 'quit', 'stop']:
                print("Shutting down system...")
                break
            
            print("⏳ Analyzing...")
            answer = ask_business_bot(document_content, user_input)
            print(f"\n🤖 ANSWER:\n{answer}\n")
            print("-" * 30)
            
        except KeyboardInterrupt:
            print("\nAborted by user.")
            break

if __name__ == "__main__":
    main()