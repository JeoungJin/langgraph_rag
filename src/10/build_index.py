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

DOC_PATH = "../../data/manual.pdf"
INDEX_PATH = "faiss_index"
EMBED_MODEL = "text-embedding-3-small"

from prepare import prepare_chunks
chunks = prepare_chunks(DOC_PATH)
print(f"Number of chunks created(분할완료): {len(chunks)}")
embedding = OpenAIEmbeddings(model = "text-embedding-3-small")
print("....embedding 오래걸림")
store =  FAISS.from_documents(chunks, embedding)  # Print embedding for first
store.save_local(INDEX_PATH)
print("....indexing 완료")

FAISS.load_local(INDEX_PATH, embedding, allow_dangerous_deserialization=True)
print("....indexing load 완료")

query = "환불 규정"
docs = store.similarity_search(query, k=2)
print(f"Similarity search results for '{query}':")
for index, doc in enumerate(docs):
    similarity_score = np.dot(embedding.embed_query(query), embedding.embed_query(doc.page_content)) / (norm(embedding.embed_query(query)) * norm(embedding.embed_query(doc.page_content)))
    print(f"Similarity score: {similarity_score:.4f}")
    print(f"순서{index + 1}. {doc.page_content}")
    print("================================")
