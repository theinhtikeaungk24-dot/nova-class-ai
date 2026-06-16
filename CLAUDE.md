# NOVA CLASS — Python AI Service (K.MATE Brain)
> Part of the NOVA CLASS AI Education Platform

This is the AI microservice powering K.MATE. See `nova-class-frontend/CLAUDE.md` for full project overview.

## Quick Start
```bash
source .venv/bin/activate
python scripts/ai_service.py
# → http://localhost:8082
# First start: waits ~30s while SentenceTransformer model downloads/loads
```

## .env required
```
GEMINI_API_KEY=your_key
CHROMA_HOST=217.142.136.244
CHROMA_PORT=8001
```

## Endpoints
```
GET  /                   → health check
POST /ask                → { question } → { answer }
POST /quiz/generate      → { topic, count } → { questions[] }
POST /quiz/check         → { questions[], answers[], language } → { score, explanation }
```

## RAG Pipeline
```
User question
  → SentenceTransformer.encode()          (multilingual embedding)
  → ChromaDB.query(n_results=5)           (semantic vector search)
  → top 5 TOPIK chunks as context
  → Gemini 2.5 Flash Lite prompt
  → answer (auto-detected language: EN / KO / MY)
```

## ChromaDB Collection
- Name: `topik_collection`
- Contains: 83rd, 91st, 96th, 102nd TOPIK II official exam papers
- Host: `217.142.136.244:8001`

## Key Scripts
```
scripts/ai_service.py   ← MAIN: FastAPI server (run this)
scripts/embed_store.py  ← PDF → chunks → embed → ChromaDB (run once to load data)
scripts/load_data.py    ← helper for data loading
scripts/create_tables.py← MySQL table setup (run once)
```

## Models Used
| Model | Purpose |
|-------|---------|
| `gemini-2.5-flash-lite` | LLM (answering, quiz gen, explanations) |
| `paraphrase-multilingual-MiniLM-L12-v2` | Text embeddings for RAG |

## Quiz Question Format
```json
{
  "q": "책을 많이 ( ) 지식을 쌓을 수 있다",
  "opts": ["읽으면", "읽어서", "읽지만", "읽는데"],
  "ans": 1,
  "section": "grammar"
}
```
> `ans` is 1-based index (1 = first option)
> All questions generated in Korean only (한국어만)
