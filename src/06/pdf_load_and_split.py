
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 스크립트 위치 기준 절대경로 (C:\RAG_Project\data)
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PDF_FILES = ["notice.pdf", "manual.pdf"]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)

for file_name in PDF_FILES:
    pdf_path = DATA_DIR / file_name
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()  # 페이지 단위로 Document 생성

    print("=" * 60)
    print(f"[{file_name}]")
    print(f"Number of pages loaded: {len(documents)}")

    splits = splitter.split_documents(documents)
    print(f"Number of splits created: {len(splits)}")

    # 앞쪽 청크 3개 확인 (page 메타데이터 포함)
    for i, split in enumerate(splits[:3]):
        print(f"--- split {i} (page {split.metadata.get('page')}) ---")
        print(split.page_content)
