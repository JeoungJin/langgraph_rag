from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
import re
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter


CURRENT_DIR = os.path.dirname(__file__)
INGEST_DIR = os.path.join(CURRENT_DIR, "..", "07")

sys.path.insert(0, INGEST_DIR)

from prepare import prepare_chunks

load_dotenv()
DATA_DIR = os.path.join(CURRENT_DIR, "..", "..", "data")
PDF_FILES = ["manual.pdf", "notice.pdf"]

# 여러 PDF의 청크를 하나의 리스트로 합침
chunks = []
for pdf in PDF_FILES:
    file_chunks = prepare_chunks(os.path.join(DATA_DIR, pdf))
    print(f"{pdf}: {len(file_chunks)}개 청크")
    chunks.extend(file_chunks)

print(f"Number of chunks created(분할완료): {len(chunks)}")

embedding = OpenAIEmbeddings(model = "text-embedding-3-small")
print("....embedding 오래걸림")
#print(f"Embedding for first chunk: {embedding.embed_query(chunks[0].page_content[:200])}")  
store =  FAISS.from_documents(chunks, embedding)  # Print embedding for first 200 characters of the first chunk
#print(store )

print("인덱싱 완료")

# 답변을 생성할 LLM 준비
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

print("준비 완료\n")

def build_context(docs):
    parts = []

    for i, doc in enumerate(docs, 1):
        text = (
            f"[{i}] "
            f"({doc.metadata['filename']} "
            f"p.{doc.metadata['page_no']})\n"
            f"{doc.page_content}"
        )

        parts.append(text)

    context = "\n\n".join(parts)

    return context

def ask(question, k=3):
    found = store.similarity_search(question, k=k)

    if not found:
        return "관련 자료를 찾지 못했습니다.", []

    context = build_context(found)

    prompt = (
        "아래 자료만 근거로 답하세요.\n"
        "자료에 없는 내용은 '자료에서 확인할 수 없습니다'라고 답하세요.\n"
        "추측하지 마세요.\n\n"
        f"[자료]\n{context}\n\n"
        f"[질문] {question}"
    )

    response = llm.invoke(prompt)

    answer = response.content

    # 근거를 찾지 못한 답변이면 출처를 비워서 반환
    if "자료에서 확인할 수 없습니다" in answer:
        return answer, []

    return answer, found


TESTS = [
    "환불은 며칠 이내에 신청해야 하나요?",          # 두 문서 모두에 있음
    "AP-3300의 소비전력과 무게는?",                 # manual.pdf에만 있음
    "우리 회사 대표이사 이름이 뭔가요?",            # 어느 문서에도 없음
    "필터 교체 알림이 뜨면 어떻게 하나요?",          # manual.pdf에만 있음
    "고객센터 운영 시간과 무상 A/S 기간을 알려주세요", # 두 문서에 걸친 질문
    # 잘못된 전제의 질문 (문서 내용과 다름)
    "환불은 14일 이내에 가능하니까 10일째인 지금 신청해도 되죠?",   # 실제 7일
    "무상 A/S 기간이 2년이니까 필터도 무료로 교체해 주나요?",       # 실제 1년, 필터는 소모품
    "고객센터는 주말에도 운영하니까 토요일에 전화하면 되나요?",       # 실제 평일만
]

if __name__ == "__main__":
    for test in TESTS:
        answer, sources = ask(test)

        print("질문 : ", test)
        print("답변 : ", answer)

        if sources:
            print("출처 :")
            # 같은 파일·페이지가 여러 청크로 잡히면 한 번만 출력
            seen = set()
            for doc in sources:
                key = (doc.metadata['filename'], doc.metadata['page_no'])
                if key in seen:
                    continue
                seen.add(key)
                print(f"  - {key[0]} p.{key[1]}")

        print("----------")