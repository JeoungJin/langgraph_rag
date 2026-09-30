from langchain_community.document_loaders import PyPDFLoader
import os
import re
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter


CURRENT_DIR = os.path.dirname(__file__)
INGEST_DIR = os.path.join(CURRENT_DIR, "..", "06")

sys.path.insert(0, INGEST_DIR)

from ingest import load_documents



CHUNK_SIZE = 200
CHUNK_OVERLAP = 50


def prepare_chunks(path):
    """
    Prepare chunks from the loaded documents using RecursiveCharacterTextSplitter.
    """
    docs = load_documents(path)

    splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP, 
    separators=["\n\n", "\n", ".", " ", ""], 
    length_function=len
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i + 1
        #print(f"Chunk {i+1}:")
        #print(chunk.page_content[:200])  # Print first 200 characters of the chunk
        #print("-" * 50)

    return chunks



 