from langchain_community.document_loaders import PyPDFLoader
import os
import re
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter


CURRENT_DIR = os.path.dirname(__file__)
INGEST_DIR = os.path.join(CURRENT_DIR, "..", "06")

sys.path.insert(0, INGEST_DIR)

from ingest import load_documents


docs = load_documents("../../data/manual.pdf")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20, separators=["\n\n", "\n", ".", " ", ""], 
    length_function=len
)
chunks = splitter.split_documents(docs)
print(f"Number of splits created: {len(chunks)}")
print(chunks[0].page_content[:200])  
print(chunks[0].metadata)  # Print first 200 characters of the second  spli
print(chunks[1].page_content[:200])  

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(chunk.page_content[:200])  # Print first 200 characters of the chunk
    #print(chunk.metadata)
    print("-" * 50)