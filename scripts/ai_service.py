from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, chromadb, json, re, random
from google import genai
from google.genai import types
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import sys
sys.path.insert(0, os.path.dirname(__file__))
from db import get_mysql

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

def parse_json_questions(text):
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    start = text.find('[')
    end = text.rfind(']') + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON array found in response")
    return json.loads(text[start:end])

def verify_answers(questions):
    """Step 2: Verify ALL answers in ONE single API call to save quota."""
    questions_text = ""
    for i, q in enumerate(questions):
        opts_text = " / ".join([f"{j+1}.{o}" for j, o in enumerate(q["opts"])])
        questions_text += f"Q{i+1}: {q['q']}\nOptions: {opts_text}\n\n"

    verify_prompt = f"""You are a Korean language expert and TOPIK II specialist.

For each question below, insert each option into the blank ( ) and find which one is GRAMMATICALLY CORRECT and NATURAL in Korean.

{questions_text}

Reply with ONLY a comma-separated list of correct answer numbers (1-based).
Example for 3 questions: 2,1,4
Your answer:"""

    try:
        vres = client_ai.models.generate_content(model="gemini-2.5-flash-lite", contents=verify_prompt)
        digits = [d.strip() for d in vres.text.strip().split(",")]
        for i, q in enumerate(questions):
            if i < len(digits) and digits[i] in ["1", "2", "3", "4"]:
                q["ans"] = int(digits[i])
    except Exception:
        pass  # keep original answers if verification fails
    return questions

@app.post("/quiz/generate")
def generate_quiz(req: QuizGenRequest):
    try:
        conn = get_mysql()
        cursor = conn.cursor()

        # Fetch from question_bank (unlimited — no API call!)
        if req.topic == "mixed":
            cursor.execute(
                "SELECT * FROM question_bank ORDER BY RAND() LIMIT %s",
                (req.count,)
            )
        else:
            cursor.execute(
                "SELECT * FROM question_bank WHERE section = %s ORDER BY RAND() LIMIT %s",
                (req.topic, req.count)
            )

        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if not rows:
            raise ValueError("No questions found in question bank")

        questions = []
        for row in rows:
            opts = [row["opt1"], row["opt2"], row["opt3"], row["opt4"]]
            correct_answer = opts[row["correct_ans"] - 1]  # save correct text before shuffle
            random.shuffle(opts)                            # randomize option order
            new_ans = opts.index(correct_answer) + 1       # find new 1-based index
            questions.append({
                "q":       row["question"],
                "opts":    opts,
                "ans":     new_ans,
                "section": row["section"]
            })

        return {"questions": questions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- Quiz Check ----
class QuizCheckRequest(BaseModel):
    questions: list
    answers: list
    language: str = "English"

def score_questions(questions, answers):
    """Score a list of questions against user answers. Returns (score, results)."""
    score = 0
    results = []
    for q, ua in zip(questions, answers):
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
    return score, results

def build_explanation(results, score, language):
    """Generate K.Mate's language-aware explanation for a set of scored results,
    with a Gemini-free fallback when the API is rate-limited."""
    qa_lines = []
    for i, r in enumerate(results):
        opts_text = " / ".join([f"{j+1}.{o}" for j, o in enumerate(r["options"])])
        correct_opt = r["options"][r["correct_answer"] - 1]
        user_opt = r["options"][r["user_answer"] - 1] if r["user_answer"] > 0 else "No answer"
        status = "✅ Correct" if r["is_correct"] else "❌ Wrong"
        qa_lines.append(f"Q{i+1}: {r['question']}\nOptions: {opts_text}\nCorrect: {r['correct_answer']}. {correct_opt}\nStudent answered: {r['user_answer']}. {user_opt} → {status}")

    qa_text = "\n\n".join(qa_lines)

    lang_map = {
        "Burmese": ("မြန်မာဘာသာ", "Write your ENTIRE response in Burmese (မြန်မာဘာသာ) using Myanmar Unicode script. Every single word — greetings, labels, explanations, tips — must be in Burmese. Do NOT write in English or Korean at all."),
        "Korean":  ("한국어",      "Write your ENTIRE response in Korean (한국어). Every single word must be in Korean. Do NOT write in English or Burmese."),
        "English": ("English",    "Write your ENTIRE response in English. Every single word must be in English. Do NOT write in Korean or Burmese."),
    }
    lang_name, lang_rule = lang_map.get(language, (language, f"Write EVERYTHING in {language} only."))

    prompt = f"""LANGUAGE RULE (HIGHEST PRIORITY): {lang_rule}

You are K.Mate, a TOPIK II tutor explaining quiz results to a student.
Output language: {lang_name} ONLY.

Quiz data (questions are in Korean — your explanations must be in {lang_name}):

{qa_text}

Score: {score}/{len(results)}

Instructions:
- For each question (Q1, Q2, ...): say ✅ Correct or ❌ Wrong, then explain WHY the correct answer is right.
- For wrong answers: give a study tip.
- Be encouraging and friendly.
- ALL your text must be in {lang_name}. The Korean questions above are DATA only — do not let them change your output language."""

    try:
        response = client_ai.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)
        return response.text
    except Exception:
        # Fallback: build basic explanation without Gemini (when rate limited)
        lines = [f"📊 Quiz Results: {score}/{len(results)}\n"]
        for i, r in enumerate(results):
            correct_opt = r["options"][r["correct_answer"] - 1]
            user_opt = r["options"][r["user_answer"] - 1] if r["user_answer"] > 0 else "No answer"
            if r["is_correct"]:
                lines.append(f"Q{i+1}: ✅ Correct! — {correct_opt}")
            else:
                lines.append(f"Q{i+1}: ❌ Wrong — You chose: {user_opt} | Correct: {correct_opt}")
                lines.append(f"   💡 Study tip: Review this pattern and try again!")
        pct = int(score / len(results) * 100) if results else 0
        if pct >= 80:
            lines.append("\n🎉 Excellent work! Keep it up!")
        elif pct >= 60:
            lines.append("\n👍 Good effort! Keep practicing!")
        else:
            lines.append("\n💪 Don't give up! Review and try again!")
        return "\n".join(lines)

@app.post("/quiz/check")
def check_quiz(req: QuizCheckRequest):
    try:
        score, results = score_questions(req.questions, req.answers)
        explanation = build_explanation(results, score, req.language)
        return {
            "score": score,
            "total": len(results),
            "results": results,
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- Full TOPIK Exam Mode (Listening / Reading / Writing) ----
EXAM_SECTION_COUNTS = {"listening": 50, "reading": 50, "writing": 4}

class ExamGenerateRequest(BaseModel):
    sections: List[str]

@app.post("/exam/generate")
def generate_exam(req: ExamGenerateRequest):
    try:
        conn = get_mysql()
        cursor = conn.cursor()
        out = {}

        for section in req.sections:
            if section not in EXAM_SECTION_COUNTS:
                continue
            count = EXAM_SECTION_COUNTS[section]
            cursor.execute(
                "SELECT * FROM topik_questions WHERE section = %s ORDER BY RAND() LIMIT %s",
                (section, count)
            )
            rows = cursor.fetchall()
            questions = []
            for row in rows:
                opts = [row["option_1"], row["option_2"], row["option_3"], row["option_4"]]
                correct_answer = opts[row["answer"] - 1]
                random.shuffle(opts)
                new_ans = opts.index(correct_answer) + 1
                questions.append({
                    "q":         row["question"],
                    "opts":      opts,
                    "ans":       new_ans,
                    "audio_url": row["audio_url"],
                    "section":   section
                })
            out[section] = questions

        cursor.close()
        conn.close()

        if not out:
            raise ValueError("No sections requested or no questions found")

        return out

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ExamCheckRequest(BaseModel):
    sections: dict  # { "listening": {"questions": [...], "answers": [...]}, ... }
    language: str = "English"

@app.post("/exam/check")
def check_exam(req: ExamCheckRequest):
    try:
        overall_score = 0
        overall_total = 0
        section_results = {}

        for section, data in req.sections.items():
            questions = data.get("questions", [])
            answers = data.get("answers", [])
            score, results = score_questions(questions, answers)
            explanation = build_explanation(results, score, req.language)

            section_results[section] = {
                "score": score,
                "total": len(results),
                "results": results,
                "explanation": explanation
            }
            overall_score += score
            overall_total += len(results)

        return {
            "overall": {"score": overall_score, "total": overall_total},
            "sections": section_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- Real Exam Papers (authentic, unshuffled, official content) ----
@app.get("/exam/real/list")
def list_real_exams():
    try:
        conn = get_mysql()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT exam_no FROM topik_questions ORDER BY exam_no")
        exam_nos = [row["exam_no"] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return {"exam_nos": exam_nos}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RealExamGenerateRequest(BaseModel):
    exam_no: int
    sections: List[str]

@app.post("/exam/real/generate")
def generate_real_exam(req: RealExamGenerateRequest):
    try:
        conn = get_mysql()
        cursor = conn.cursor()
        out = {}

        for section in req.sections:
            if section == "writing":
                cursor.execute(
                    "SELECT * FROM topik_writing_questions WHERE exam_no = %s ORDER BY question_no",
                    (req.exam_no,)
                )
                rows = cursor.fetchall()
                out["writing"] = [{
                    "question_no": row["question_no"],
                    "prompt": row["prompt"],
                    "blank_count": row["blank_count"],
                    "min_chars": row["min_chars"],
                    "max_chars": row["max_chars"],
                    "points": row["points"],
                } for row in rows]
            elif section in ("listening", "reading"):
                cursor.execute(
                    "SELECT * FROM topik_questions WHERE exam_no = %s AND section = %s ORDER BY question_no",
                    (req.exam_no, section)
                )
                rows = cursor.fetchall()
                # No shuffle — authentic paper, options stay as printed
                out[section] = [{
                    "question_no": row["question_no"],
                    "q":           row["question"],
                    "opts":        [row["option_1"], row["option_2"], row["option_3"], row["option_4"]],
                    "ans":         row["answer"],
                    "audio_url":   row["audio_url"],
                    "section":     section,
                } for row in rows]

        cursor.close()
        conn.close()

        if not out:
            raise ValueError("No sections requested or no questions found")

        return out

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def grade_writing_answer(question_no, prompt, model_answer, student_answer, points, language):
    """Grade a single writing answer via Gemini against the real model answer/rubric."""
    if question_no in (51, 52):
        rubric = "Grade for grammatical correctness and contextual fit of the filled-in blank(s), comparing against the model answer. Minor wording differences that preserve the same meaning/grammar should still score well."
    else:
        rubric = "Grade for content relevance to the prompt, structure/organization, grammar accuracy, and adherence to the required character length. Use the model answer as a reference for expected content quality, not for exact wording."

    lang_map = {
        "Burmese": "မြန်မာဘာသာ", "Korean": "한국어", "English": "English",
    }
    lang_name = lang_map.get(language, language)

    prompt_text = f"""You are a strict but fair TOPIK II writing examiner.

Question prompt:
{prompt}

Model answer (reference only, do not require exact match):
{model_answer}

Student's answer:
{student_answer}

Maximum points: {points}
Grading rubric: {rubric}

Respond in {lang_name}. Output EXACTLY in this format:
SCORE: <integer 0-{points}>
FEEDBACK: <2-4 sentences of feedback in {lang_name}, encouraging but honest>"""

    try:
        response = client_ai.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt_text)
        text = response.text.strip()
        score_match = re.search(r"SCORE:\s*(\d+)", text)
        feedback_match = re.search(r"FEEDBACK:\s*(.+)", text, re.DOTALL)
        score = int(score_match.group(1)) if score_match else 0
        feedback = feedback_match.group(1).strip() if feedback_match else text
        return min(score, points), feedback
    except Exception:
        return 0, "⚠️ Automatic grading is temporarily unavailable (rate limited) — manual review needed."


class RealExamCheckRequest(BaseModel):
    exam_no: int
    sections: dict
    language: str = "English"

@app.post("/exam/real/check")
def check_real_exam(req: RealExamCheckRequest):
    try:
        overall_score = 0
        overall_total = 0
        section_results = {}

        for section, data in req.sections.items():
            if section == "writing":
                conn = get_mysql()
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM topik_writing_questions WHERE exam_no = %s ORDER BY question_no",
                    (req.exam_no,)
                )
                wq_rows = {row["question_no"]: row for row in cursor.fetchall()}
                cursor.close()
                conn.close()

                answers = data.get("answers", {})
                writing_results = []
                section_score = 0
                section_total = 0
                for qno_str, student_answer in answers.items():
                    qno = int(qno_str)
                    wq = wq_rows.get(qno)
                    if not wq:
                        continue
                    score, feedback = grade_writing_answer(
                        qno, wq["prompt"], wq["model_answer"], student_answer, wq["points"], req.language
                    )
                    writing_results.append({
                        "question_no": qno,
                        "score": score,
                        "max_points": wq["points"],
                        "feedback": feedback,
                    })
                    section_score += score
                    section_total += wq["points"]

                section_results["writing"] = {
                    "score": section_score,
                    "total": section_total,
                    "results": writing_results,
                }
                overall_score += section_score
                overall_total += section_total

            else:
                questions = data.get("questions", [])
                answers = data.get("answers", [])
                score, results = score_questions(questions, answers)
                explanation = build_explanation(results, score, req.language)

                section_results[section] = {
                    "score": score,
                    "total": len(results),
                    "results": results,
                    "explanation": explanation
                }
                overall_score += score
                overall_total += len(results)

        return {
            "overall": {"score": overall_score, "total": overall_total},
            "sections": section_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)
