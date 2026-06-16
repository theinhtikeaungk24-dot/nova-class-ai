import os
import json
import chromadb
import fitz  # PyMuPDF
import easyocr
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PROGRESS_FILE = "./database/extracted_text.json"

# ---- ရပြီးသား data load လုပ်တယ် ----
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"📂 ရပြီးသား {len(data)} pages ရှိပြီ")
        return data
    return []

def save_progress(data):
    os.makedirs("./database", exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ---- 1. EasyOCR နဲ့ PDF text ထုတ်တယ် ----
def extract_text_with_easyocr(data_path, all_texts):
    print("⏳ EasyOCR loading (Korean + English)... ခဏစောင့်ပါ")
    # Korean နဲ့ English နှစ်မျိုးလုံး ဖတ်နိုင်တယ်
    reader = easyocr.Reader(["ko", "en"], gpu=False)
    print("✅ EasyOCR ready!\n")

    done = {(item["source"], item["page"]) for item in all_texts}

    for filename in sorted(os.listdir(data_path)):
        if not filename.endswith(".pdf"):
            continue

        pdf_path = os.path.join(data_path, filename)
        doc = fitz.open(pdf_path)
        total = len(doc)
        print(f"📄 {filename} ({total} pages)")

        for page_num in range(total):
            key = (filename, page_num + 1)

            if key in done:
                print(f"  Page {page_num+1}/{total} ⏭️  skip")
                continue

            # PDF page → numpy image array
            pix = doc[page_num].get_pixmap(dpi=150)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )

            print(f"  Page {page_num+1}/{total} reading...", end=" ", flush=True)

            results = reader.readtext(img_array, detail=0, paragraph=True)
            text = "\n".join(results).strip()

            if text:
                item = {"text": text, "source": filename, "page": page_num + 1}
                all_texts.append(item)
                done.add(key)
                save_progress(all_texts)
                print(f"✅ ({len(text)} chars)")
            else:
                print("⚠️ empty")

        doc.close()

    return all_texts

# ---- 2. Chunks ခွဲတယ် ----
def chunk_texts(text_list):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = []
    for item in text_list:
        splits = splitter.split_text(item["text"])
        for split in splits:
            chunks.append({
                "text": split,
                "source": item["source"],
                "page": item["page"]
            })
    print(f"✅ {len(chunks)} chunks created")
    return chunks

# ---- 3. ChromaDB ထဲ သိမ်းတယ် ----
def embed_and_store(chunks):
    print("\n⏳ Embedding model loading...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
    client_db = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    try:
        client_db.delete_collection("topik_collection")
    except:
        pass
    collection = client_db.create_collection("topik_collection")

    texts     = [c["text"]    for c in chunks]
    ids       = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": c["source"], "page": str(c["page"])} for c in chunks]

    print(f"⏳ Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print(f"✅ ChromaDB ထဲ {len(texts)} chunks သိမ်းပြီးပြီ!")

# ---- Main ----
if __name__ == "__main__":
    all_texts = load_progress()
    all_texts = extract_text_with_easyocr("./data", all_texts)

    if not all_texts:
        print("❌ Text မရသေးဘူး")
    else:
        print(f"\n📊 စုစုပေါင်း {len(all_texts)} pages extracted")
        chunks = chunk_texts(all_texts)
        embed_and_store(chunks)
        print("\n🎉 Database ready! query.py နဲ့ မေးလို့ရပြီ။")
