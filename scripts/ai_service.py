from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, chromadb, json, re
from google import genai
from google.genai import types
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from typing import List, Optional

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load models
client_ai      = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
client_db      = chromadb.HttpClient(host=os.getenv("CHROMA_HOST"), port=int(os.getenv("CHROMA_PORT")))
collection     = client_db.get_collection("topik_collection")

print(f"✅ AI Service ready ({collection.count()} chunks loaded)")

class AskRequest(BaseModel):
    question: str

@app.get("/")
def health():
    return {"status": "K.Mate AI Service running"}

@app.post("/ask")
def ask(req: AskRequest):
    try:
        # ChromaDB ထဲ ရှာ
        vector = embedding_model.encode([req.question]).tolist()
        results = collection.query(query_embeddings=vector, n_results=5)
        context = "\n\n".join(results["documents"][0])

        # Gemini နဲ့ ဖြေ
        prompt = f"""You are K.Mate, an expert Korean language tutor specializing in TOPIK II exam preparation.
You have deep knowledge of Korean grammar, vocabulary, TOPIK exam structure, scoring, and study strategies.
Auto-detect the language of the question and reply in the SAME language (English / 한국어 / မြန်မာဘာသာ).

Reference material from official TOPIK II exam papers (use this for specific exam questions):
{context}

Student question: {req.question}

Instructions:
1. AUTO-DETECT language → reply in that language ONLY.
2. If the reference material contains the answer → use it and explain step by step.
3. If the reference material does NOT contain the answer → use YOUR OWN expert knowledge about TOPIK, Korean language, grammar rules, vocabulary, scoring system, exam tips etc. NEVER say "I cannot answer."
4. For grammar questions: identify the pattern being tested and explain WHY the answer is correct.
5. For vocabulary: give meaning, example sentence, and usage tips.
6. For TOPIK general info (scores, levels, sections): answer from your expert knowledge.
7. Always give a helpful, complete answer.
8. IMPORTANT: Do NOT use HTML tags like <strong>, <br>, <b> etc. Use plain text only. For bold use **text**. For new lines use actual line breaks."""

        response = client_ai.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        return {"answer": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- Quiz Generate ----
class QuizGenRequest(BaseModel):
    topic: str = "mixed"
    count: int = 5

@app.post("/quiz/generate")
def generate_quiz(req: QuizGenRequest):
    try:
        prompt = f"""You are a TOPIK II exam creator. Generate exactly {req.count} TOPIK II style multiple choice questions about {req.topic} Korean language.

Return ONLY a valid JSON array, no extra text, no markdown. Example format:
[
  {{
    "q": "책을 많이 ( ) 지식을 쌓을 수 있다",
    "opts": ["읽으면", "읽어서", "읽지만", "읽는데"],
    "ans": 1,
    "section": "grammar"
  }}
]

STRICT RULES:
- ALL questions must be written in KOREAN (한국어) only — NO English sentences
- "q" = Korean sentence with blank ( ) for grammar, or 다음 중 올바른 것은? for vocabulary
- "opts" = exactly 4 Korean answer choices
- "ans" = correct answer index (1-based: 1, 2, 3, or 4)
- "section" = grammar / vocabulary / reading
- Make questions authentic TOPIK II difficulty level
- Use real Korean grammar patterns like -아/어야 하다, -ㄹ수록, -는 바람에, 만큼, etc."""

        response = client_ai.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)
        text = response.text.strip()

        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)

        # Extract JSON array
        start = text.find('[')
        end = text.rfind(']') + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON array found in response")

        questions = json.loads(text[start:end])
        return {"questions": questions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- Quiz Check ----
class QuizCheckRequest(BaseModel):
    questions: list
    answers: list
    language: str = "English"

@app.post("/quiz/check")
def check_quiz(req: QuizCheckRequest):
    try:
        score = 0
        results = []

        for i, (q, ua) in enumerate(zip(req.questions, req.answers)):
            correct = q["ans"]
            is_correct = (ua == correct)
            if is_correct:
                score += 1
            results.append({
                "question": q["q"],
                "options": q["opts"],
                "correct_answer": correct,
                "user_answer": ua,
                "is_correct": is_correct
            })

        # Build explanation prompt
        qa_lines = []
        for i, r in enumerate(results):
            opts_text = " / ".join([f"{j+1}.{o}" for j, o in enumerate(r["options"])])
            correct_opt = r["options"][r["correct_answer"] - 1]
            user_opt = r["options"][r["user_answer"] - 1] if r["user_answer"] > 0 else "No answer"
            status = "✅ Correct" if r["is_correct"] else "❌ Wrong"
            qa_lines.append(f"Q{i+1}: {r['question']}\nOptions: {opts_text}\nCorrect: {r['correct_answer']}. {correct_opt}\nStudent answered: {r['user_answer']}. {user_opt} → {status}")

        qa_text = "\n\n".join(qa_lines)

        prompt = f"""You are K.Mate, a friendly TOPIK II tutor. A student finished a practice quiz. Reply in {req.language}.

Quiz Results (Score: {score}/{len(results)}):

{qa_text}

For each question:
- State ✅ Correct or ❌ Wrong
- Explain WHY the correct answer is right (grammar rule / vocabulary meaning)
- For wrong answers: give a helpful study tip
- Be encouraging!

Format your response clearly with Q1:, Q2:, etc."""

        response = client_ai.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)

        return {
            "score": score,
            "total": len(results),
            "results": results,
            "explanation": response.text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)
