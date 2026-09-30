import datetime
import json
import json
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy  as np
import sys
from numpy.linalg import norm
from langchain_community.vectorstores import FAISS

load_dotenv()

CURRENT_DIR = os.path.dirname(__file__)
PREPARE_DIR = os.path.join(CURRENT_DIR, "..", "07")
sys.path.insert(0, PREPARE_DIR)

# 실행 위치(cwd)와 상관없이 이 파일 기준으로 경로를 잡음
DOC_PATH = os.path.join(CURRENT_DIR, "..", "..", "data", "manual.pdf")
INDEX_PATH = os.path.join(CURRENT_DIR, "faiss_index")
EMBED_MODEL = "text-embedding-3-small"

from prepare import prepare_chunks

def get_store(rebuild=False):
    if rebuild:
        chunks = prepare_chunks(DOC_PATH)
        print(f"Number of chunks created(분할완료): {len(chunks)}")
        embedding = OpenAIEmbeddings(model = EMBED_MODEL)
        print("....embedding 오래걸림")
        store =  FAISS.from_documents(chunks, embedding)  # Print embedding for first
        #store.save_local(INDEX_PATH)
        save_with_meta(store, INDEX_PATH, 
                       {"doc_path": DOC_PATH, "embed_model": EMBED_MODEL,
                         "chunk_size": 200, "chunk_overlap": 50})
        print("....indexing 완료")
    else:
        embedding = OpenAIEmbeddings(model = EMBED_MODEL)
        store = FAISS.load_local(INDEX_PATH, embedding, allow_dangerous_deserialization=True)
        print("....indexing load 완료")
    return store

def save_with_meta(store, path, info):
    store.save_local(path)
    info["created_at"] = datetime.datetime.now().isoformat()
    meta_path = os.path.join(path, "build_info.json")
    with open(meta_path, "w", encoding="utf-8") as file:
        json.dump(
            info,
            file,
            ensure_ascii=False,
            indent=2
        )
        
if __name__ == "__main__":
    store = get_store(rebuild=True)
    query = "환불 규정"
    docs = store.similarity_search(query, k=2)
    print(f"Similarity search results for '{query}':")
    for index, doc in enumerate(docs):
        similarity_score = np.dot(store.embeddings.embed_query(query), 
                                  store.embeddings.embed_query(doc.page_content)) /  (norm(store.embeddings.embed_query(query)) * norm(store.embeddings.embed_query(doc.page_content)))
        print(f"Similarity score: {similarity_score:.4f}")
        print(f"순서{index + 1}. {doc.page_content}")
        print("================================")