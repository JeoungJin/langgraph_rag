
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 스크립트 위치 기준 절대경로 (C:\RAG_Project\data\notice.txt)
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "notice.pdf"

loader = PyPDFLoader(str(DATA_PATH) )
documents = loader.load()

 
print(f"Number of documents loaded: {len(documents)}")
#print(f"First document content: {documents[0].page_content[:500]}")  # Print first 500 characters of the first document

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)
splits = splitter.split_documents(documents)
print(f"Number of splits created: {len(splits)}")
print(f"First split content0: {splits[0].page_content}")  # Print first 500 characters of the first split
print(f"First split content1: {splits[1].page_content}")  # Print first 500 characters of the first split
