"""
Seed Full TOPIK II Exam Mode question pool into `topik_questions`.
Generates Listening (50), Reading (50), Writing (4, MCQ substitute) via Gemini,
batch-verifies answers in one call per batch (rate-limit friendly), and inserts.

Run once: python scripts/seed_full_exam.py
"""
import os, json, re, sys, time

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
from google import genai
from db import get_mysql

load_dotenv()
client_ai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EXAM_NO = 1  # bookkeeping only — not used in selection queries


def call_gemini(prompt, retries=4, base_delay=8):
    """Gemini call with retry/backoff — the API intermittently returns 503 under high demand."""
    last_err = None
    for attempt in range(retries):
        try:
            return client_ai.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)
        except Exception as e:
            last_err = e
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                wait = base_delay * (attempt + 1)
                print(f"    ⏳ Gemini busy (503), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise last_err

# (section, total_count, batch_size)
PLAN = [
    ("listening", 50, 10),
    ("reading",   50, 10),
    ("writing",    4,  4),
]


def parse_questions(text):
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    start = text.find('[')
    end   = text.rfind(']') + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON array found")
    return json.loads(text[start:end])


def verify_batch(questions):
    """Verify all answers in ONE API call (same pattern as seed_questions.py)."""
    lines = ""
    for i, q in enumerate(questions):
        opts = " / ".join([f"{j+1}.{o}" for j, o in enumerate(q["opts"])])
        lines += f"Q{i+1}: {q['q']}\nOptions: {opts}\n\n"

    prompt = f"""You are a Korean language expert and TOPIK II specialist.
For each question, find the correct answer by inserting each option into the blank ( ) or evaluating the dialogue/passage context.
Choose the option that is GRAMMATICALLY CORRECT and CONTEXTUALLY NATURAL in Korean.

{lines}
Reply ONLY with comma-separated correct answer numbers (1-based).
Example for 4 questions: 2,1,4,3
Your answer:"""

    res = call_gemini(prompt)
    digits = [d.strip() for d in res.text.strip().split(",")]
    for i, q in enumerate(questions):
        if i < len(digits) and digits[i] in ["1", "2", "3", "4"]:
            q["ans"] = int(digits[i])
    return questions


def generate_batch(section, count):
    if section == "listening":
        spec = f"""Generate exactly {count} TOPIK II LISTENING-style questions.
Each "q" field must be a short Korean DIALOGUE TRANSCRIPT (2-4 lines, like a real listening script between two speakers, e.g. "남자: ... \\n여자: ...") followed by a question about it on a new line (e.g. "다음 대화를 듣고 남자의 중심 생각으로 가장 알맞은 것을 고르십시오.").
There is no actual audio yet — the "q" text IS the transcript students will read instead."""
    elif section == "reading":
        spec = f"""Generate exactly {count} TOPIK II READING questions.
Each "q" field must be a short Korean passage (2-4 sentences) followed by a question about it on a new line (e.g. "윗글의 내용과 같은 것을 고르십시오." or a ( ) blank-fill sentence)."""
    else:  # writing
        spec = f"""Generate exactly {count} TOPIK II WRITING-related multiple-choice questions (grammar/sentence-completion style, since this substitutes for free-form essay writing).
Each "q" field is a Korean sentence with a ( ) blank testing a grammar pattern commonly needed in writing (connectives, formal endings, etc.)."""

    prompt = f"""You are a TOPIK II exam creator. {spec}

Return ONLY a valid JSON array:
[
  {{
    "q": "Korean text (transcript/passage/sentence as described above)",
    "opts": ["option1", "option2", "option3", "option4"],
    "ans": 1,
    "section": "{section}"
  }}
]

STRICT RULES:
- ALL content in KOREAN only — NO English sentences
- "opts" = exactly 4 Korean options
- "ans" = correct answer index (1-based, can be a placeholder, will be re-verified)
- Authentic TOPIK II difficulty
- Make all 4 options plausible but only ONE correct"""

    res = call_gemini(prompt)
    return parse_questions(res.text.strip())


def save_batch(section, questions, start_no):
    conn = get_mysql()
    cursor = conn.cursor()
    saved = 0
    for i, q in enumerate(questions):
        try:
            opts = q.get("opts", [])
            while len(opts) < 4:
                opts.append("")
            cursor.execute("""
                INSERT INTO topik_questions
                    (exam_no, section, question_no, question, option_1, option_2, option_3, option_4, audio_url, answer)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NULL, %s)
            """, (
                EXAM_NO, section, start_no + i,
                q["q"], opts[0], opts[1], opts[2], opts[3],
                q.get("ans", 1),
            ))
            saved += 1
        except Exception as e:
            print(f"  ⚠️ Skip: {e}")
    conn.commit()
    cursor.close()
    conn.close()
    return saved


if __name__ == "__main__":
    grand_total = 0

    for section, total, batch_size in PLAN:
        print(f"\n📝 Section: {section} ({total} questions)")
        question_no = 1

        for offset in range(0, total, batch_size):
            n = min(batch_size, total - offset)
            print(f"  Generating batch of {n} (questions {offset+1}-{offset+n})...")
            try:
                batch = generate_batch(section, n)
                print(f"  ✅ Generated {len(batch)}")

                print(f"  🔍 Verifying answers (1 API call)...")
                batch = verify_batch(batch)

                saved = save_batch(section, batch, question_no)
                question_no += saved
                grand_total += saved
                print(f"  💾 Saved {saved} to topik_questions")

                time.sleep(1)  # small buffer between batches
            except Exception as e:
                print(f"  ❌ Error on batch (offset {offset}): {e}")

    print(f"\n🎉 Done! Total {grand_total} questions seeded into topik_questions (exam_no={EXAM_NO})")
