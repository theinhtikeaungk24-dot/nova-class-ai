"""
Seed 50 TOPIK II questions into MySQL question_bank.
Run once: python scripts/seed_questions.py
"""
import os, json, re, sys
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
from google import genai
from db import get_mysql

load_dotenv()
client_ai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

BATCHES = [
    {"topic": "grammar",    "count": 20},
    {"topic": "vocabulary", "count": 20},
    {"topic": "reading",    "count": 10},
]

def parse_questions(text):
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    start = text.find('[')
    end   = text.rfind(']') + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON array found")
    return json.loads(text[start:end])

def verify_all(questions):
    """Verify all answers in ONE API call."""
    lines = ""
    for i, q in enumerate(questions):
        opts = " / ".join([f"{j+1}.{o}" for j,o in enumerate(q["opts"])])
        lines += f"Q{i+1}: {q['q']}\nOptions: {opts}\n\n"

    prompt = f"""You are a Korean language expert and TOPIK II specialist.
For each question, find the correct answer by inserting each option into the blank ( ).
Choose the option that is GRAMMATICALLY CORRECT and NATURAL in Korean.

{lines}
Reply ONLY with comma-separated correct answer numbers (1-based).
Example for 5 questions: 2,1,4,3,1
Your answer:"""

    res = client_ai.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)
    digits = [d.strip() for d in res.text.strip().split(",")]
    for i, q in enumerate(questions):
        if i < len(digits) and digits[i] in ["1","2","3","4"]:
            q["ans"] = int(digits[i])
    return questions

def save_to_db(questions):
    conn = get_mysql()
    cursor = conn.cursor()
    saved = 0
    for q in questions:
        try:
            opts = q.get("opts", [])
            while len(opts) < 4:
                opts.append("")
            cursor.execute("""
                INSERT INTO question_bank (question, opt1, opt2, opt3, opt4, correct_ans, section)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                q["q"], opts[0], opts[1], opts[2], opts[3],
                q.get("ans", 1),
                q.get("section", "grammar")
            ))
            saved += 1
        except Exception as e:
            print(f"  ⚠️ Skip: {e}")
    conn.commit()
    cursor.close()
    conn.close()
    return saved

def generate_batch(topic, count):
    prompt = f"""You are a TOPIK II exam creator. Generate exactly {count} TOPIK II multiple choice questions about {topic} Korean language.

Return ONLY a valid JSON array:
[
  {{
    "q": "Korean sentence with ( ) blank",
    "opts": ["option1", "option2", "option3", "option4"],
    "ans": 1,
    "section": "{topic}"
  }}
]

STRICT RULES:
- ALL questions in KOREAN only — NO English sentences
- "q" = Korean sentence with ( ) blank
- "opts" = exactly 4 Korean options
- "ans" = correct answer index (1-based)
- Authentic TOPIK II difficulty
- Use patterns: -아/어야 하다, -ㄹ수록, -는 바람에, 만큼, -(으)므로, -도록, -는 한, etc.
- Make all 4 options plausible but only ONE correct"""

    res = client_ai.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)
    return parse_questions(res.text.strip())


if __name__ == "__main__":
    total_saved = 0

    for batch in BATCHES:
        topic = batch["topic"]
        count = batch["count"]
        print(f"\n📝 Generating {count} {topic} questions...")

        try:
            questions = generate_batch(topic, count)
            print(f"  ✅ Generated {len(questions)} questions")

            print(f"  🔍 Verifying answers (1 API call)...")
            questions = verify_all(questions)

            saved = save_to_db(questions)
            total_saved += saved
            print(f"  💾 Saved {saved} to MySQL")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    print(f"\n🎉 Done! Total {total_saved} questions in question_bank")
