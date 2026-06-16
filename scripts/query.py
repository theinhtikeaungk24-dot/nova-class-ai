import os
import subprocess
import html as html_lib
from google import genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from db import get_chroma

load_dotenv()
client_ai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
HTML_FILE = "./kmate_output.html"

# ---- HTML browser output ----
def save_to_html(question, answer):
    answer_html = html_lib.escape(answer).replace("\n", "<br>").replace("**", "")
    question_html = html_lib.escape(question)
    block = f"""
    <div class="qa-block">
        <div class="question">🧑 {question_html}</div>
        <div class="answer">🤖 K.Mate<br><br>{answer_html}</div>
    </div>"""
    if not os.path.exists(HTML_FILE):
        with open(HTML_FILE, "w", encoding="utf-8") as f:
            f.write("""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<title>K.Mate Chat</title>
<style>
  body { font-family: 'Pyidaungsu','Padauk',Arial,sans-serif; background:#1e1e2e; color:#cdd6f4; padding:20px; max-width:900px; margin:auto; }
  .qa-block { background:#313244; border-radius:12px; padding:20px; margin-bottom:20px; }
  .question { color:#89b4fa; font-size:18px; font-weight:bold; margin-bottom:12px; }
  .answer { color:#cdd6f4; line-height:1.8; font-size:16px; }
  h1 { color:#cba6f7; text-align:center; }
</style></head>
<body><h1>K.Mate — TOPIK AI Tutor</h1></body></html>""")
    with open(HTML_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace("</body></html>", block + "\n</body></html>")
    with open(HTML_FILE, "w", encoding="utf-8") as f:
        f.write(content)
    subprocess.Popen(["open", HTML_FILE])

# ---- System load ----
print("⏳ Loading system...")
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
collection = get_chroma()
print(f"✅ Ready! ({collection.count()} chunks loaded)\n")

# ---- Ask function ----
def ask(question: str) -> str:
    query_vector = embedding_model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_vector, n_results=5)

    context_parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_parts.append(f"[{meta.get('source')}, Page {meta.get('page')}]\n{doc}")
    context = "\n\n".join(context_parts)

    prompt = f"""You are K.Mate, an expert Korean language tutor for TOPIK II exam preparation.
You can speak English, Korean (한국어), and Burmese (မြန်မာဘာသာ).

AUTO LANGUAGE DETECTION (STRICTLY FOLLOW):
- Analyze the student's question language carefully.
- If the question is written in Burmese script (မြန်မာ) OR mentions "burmese" OR says "in burmese" → reply ENTIRELY in Burmese (မြန်မာဘာသာ Unicode).
- If the question is written in Korean (한국어) → reply ENTIRELY in Korean.
- If the question is written in English → reply ENTIRELY in English.
- NEVER mix languages. Reply in ONE language only.

Reference material from TOPIK II official exam papers:
{context}

Student's question:
{question}

Instructions:
1. Detect the question language first, then reply in that SAME language.
2. READ and UNDERSTAND the question content — do NOT just match by question numbers.
3. Use Korean language expertise (grammar rules, vocabulary, context clues) to reason through it.
4. Identify what skill is being tested (grammar pattern, vocabulary, reading comprehension).
5. Explain step by step WHY the correct answer is right and why other options are wrong.
6. If the question has a blank ( ), analyze the sentence structure and grammar to find the best fit.
7. If you cannot determine the answer, say so honestly in the detected language."""

    response = client_ai.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )
    return response.text

# ---- Chat loop ----
print("=" * 50)
print("  K.Mate — TOPIK AI Tutor")
print("  Auto-detects: English / 한국어 / မြန်မာ")
print("  Type 'quit' to exit")
print("=" * 50 + "\n")

while True:
    question = input("You: ").strip()
    if not question:
        continue
    if question.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    print("K.Mate: thinking...", flush=True)
    try:
        answer = ask(question)
        print(answer)
        save_to_html(question, answer)
    except Exception as e:
        err = str(e)
        if "429" in err:
            print("⛔ Rate limit exceeded. Please wait and try again.")
        elif "503" in err or "UNAVAILABLE" in err:
            print("⚠️ Gemini server busy. Please try again in 10 seconds.")
        else:
            print(f"❌ Error: {e}")
    print()
