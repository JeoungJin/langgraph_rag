 
from langchain_community.document_loaders import TextLoader

 

loader = TextLoader("../../data/notice.txt", encoding="utf-8")
documents = loader.load()
print(f"Number of documents loaded: {len(documents)}")
print(f"First document content: {documents[0].page_content[:500]}")  # Print first 500 characters of the first document
