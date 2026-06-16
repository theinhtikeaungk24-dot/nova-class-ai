from langchain_community.document_loaders import PyPDFLoader
import os

def load_all_pdfs(data_folder):
    all_documents = []

    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(data_folder, filename)
            print(f"Loading: {filename}")

            loader = PyPDFLoader(filepath)
            documents = loader.load()
            all_documents.extend(documents)

            print(f"  → {len(documents)} မျက်နှာ load ပြီး")

    print(f"\n✅ စုစုပေါင်း {len(all_documents)} မျက်နှာ")
    return all_documents

# Run ပါ
docs = load_all_pdfs("./data")
print(f"\nနမူနာ:\n{docs[0].page_content[:300]}")