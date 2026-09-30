import re
import os
import sys
from pathlib import Path

from pypdf.errors import PdfReadError
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 스크립트 위치 기준 절대경로 (C:\RAG_Project\data\notice.txt)
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "notice.pdf"

# 로드 전 파일 검증: 존재 여부 / 빈 파일(0바이트) 여부
# if not DATA_PATH.exists():
#     print(f"Error: 파일이 없습니다 -> {DATA_PATH}")
#     sys.exit(1)
# if DATA_PATH.stat().st_size == 0:
#     print(f"Error: 빈 파일입니다 (0 bytes) -> {DATA_PATH}")
#     sys.exit(1)

loader = PyPDFLoader(str(DATA_PATH))
NOISE = ["대외비"]  
try:
    documents = loader.load()
except PdfReadError as e:  # 손상되었거나 PDF 형식이 아닌 파일
    print(f"Error: PDF를 읽을 수 없습니다 -> {DATA_PATH} ({e})")
    sys.exit(1)

 
print(f"Number of documents loaded: {len(documents)}")
#print(f"First document content: {documents[0].page_content[:500]}")  # Print first 500 characters of the first document

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)
splits = splitter.split_documents(documents)




def load_documents(path):
    """
    Load documents from the specified PDF file and split them into chunks.
    """
    loader = PyPDFLoader(str(path))
    try:
        documents = loader.load()
    except PdfReadError as e:  # 손상되었거나 PDF 형식이 아닌 파일
        print(f"Error: PDF를 읽을 수 없습니다 -> {path} ({e})")
        sys.exit(1)

    print(f"Number of documents loaded: {len(documents)}")
    for d in documents[:3]:  # Print first 3 documents for verification
        print(f"Document (Page {d.metadata.get('page', 'Unknown')}): {d.page_content[:200]}...")  # Print first 200 characters  
        text  = d.page_content.strip()
        for noise in NOISE:
            text = text.replace(noise, "")

        text = re.sub(r'\n{3}', '\n\n', text)  # Replace three consecutive newlines with two
        #메타데이터
        d.metadata["filename"] = os.path.basename(path)
        d.metadata["page_number"] = d.metadata.get("page", "0")+1

    validate_splits(splits)




def validate_splits(splits):
    """
    Validate the splits to ensure they are not empty and do not exceed the chunk size.
    """
    empty_pages = []
   
    for i, split in enumerate(splits):
        text = split.page_content.strip()
        print(f"Split {i} (Page {split.metadata.get('page', 'Unknown')}): Length = {len(text)}")
        if not text: 
            page = split.metadata.get("page", "Unknown")
            print(f"Warning: Split {i} (Page {page}) is empty.")         
            empty_pages.append(i)
    
    
    if empty_pages:
        print(f"스캔할 페이지 없음: {empty_pages}")
    else:
        print("All splits are valid (non-empty).")
        print(f"Total number of splits: {len(splits)}")


if __name__ == "__main__":
    
    load_documents(DATA_PATH)
 

 