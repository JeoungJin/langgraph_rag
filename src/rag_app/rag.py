# ============================================================
# 금융 질문에 답하는 Tool 사용 에이전트입니다. (LangChain create_agent / LangGraph)
# 에이전트가 스스로 계획하고 Tool을 여러 번 호출하며, 결과가 부족하면
# 검색어를 바꿔 다시 시도한 뒤 최종 답변을 만듭니다.
#   - 우리은행 문서 : search_company_docs (FAISS RAG)
#   - 주가 / 환율   : market_tools
#   - 숫자 계산     : calc_tools.calculate
#   - 일반 금융 지식: Tool 없이 직접 답변
# 문서 기반 답변은 출처와 인용 번호가 올바른지도 함께 확인합니다.
# ============================================================

import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from contextvars import ContextVar

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

import config
from indexer import get_store
from calc_tools import calculate
from market_tools import get_exchange_rate, get_stock_price
from prompts import AGENT_SYSTEM_PROMPT


# ============================================================
# 1. 기본 환경 준비
# ============================================================

# .env 파일의 OPENAI_API_KEY를 환경변수로 불러옵니다.
load_dotenv()


# FAISS 벡터 저장소를 준비합니다.
store = get_store()


# OpenAI LLM을 준비합니다.
llm = ChatOpenAI(
    model=config.LLM_MODEL,
    temperature=config.TEMPERATURE
)


# 에이전트가 한 질문에서 밟을 수 있는 최대 단계 수입니다. (모델 호출 + Tool 호출 합계)
RECURSION_LIMIT = 15

# 이번 질문에서 검색된 문서를 모아두는 목록입니다. (ask()가 매번 새로 만듭니다)
# ContextVar라서 웹에서 여러 사용자가 동시에 질문해도 서로 섞이지 않습니다.
_retrieved_var = ContextVar("retrieved", default=None)


# ============================================================
# 2. 검색 함수
# ============================================================
# 질문과 관련된 문서를 벡터 저장소에서 검색합니다.
# MIN_SCORE 이상의 문서만 최종 검색 결과로 사용합니다.
# ============================================================

def _search(question, k=None):

    # 검색 개수가 지정되지 않으면 기본값을 사용합니다.
    if k is None:
        k = config.TOP_K

    # 질문과 유사한 문서를 검색합니다.
    results = store.similarity_search_with_relevance_scores(
        question,
        k=k
    )

    # 기준 점수를 통과한 문서만 저장합니다.
    docs = []

    for doc, score in results:

        if score >= config.MIN_SCORE:
            docs.append(doc)

    return docs


# ============================================================
# 3. 컨텍스트 생성 함수
# ============================================================
# 검색된 문서들을 LLM에게 전달하기 좋은 문자열로 만듭니다.
# 각 문서 앞에 번호, 파일명, 페이지 번호를 표시합니다.
# ============================================================

def _build_context(docs, start=1):

    # 문서별 문자열을 저장할 리스트입니다.
    parts = []

    # 검색된 문서를 하나씩 처리합니다.
    for number, doc in enumerate(docs, start):

        # 원본 파일 이름을 가져옵니다.
        filename = doc.metadata.get(
            "filename",
            "unknown"
        )

        # 페이지 번호를 가져옵니다.
        page_no = doc.metadata.get("page_no")

        # page_no가 없으면 기본 page 값을 사용합니다.
        if page_no is None:
            page = doc.metadata.get("page", 0)
            page_no = page + 1

        # 출처 정보와 문서 내용을 하나의 문자열로 만듭니다.
        part = (
            f"[{number}] {filename} p.{page_no}\n"
            f"{doc.page_content}"
        )

        parts.append(part)

    # 문서 사이에 구분선을 넣어 하나의 문자열로 만듭니다.
    context = "\n\n---\n\n".join(parts)

    return context


# ============================================================
# 3-1. 사내 문서 검색 Tool
# ============================================================

@tool
def search_company_docs(query: str) -> str:
    """우리은행 사내 문서(WON플러스 예금 상품설명서 등)에서 상품의 금리, 한도,
    가입 조건, 중도해지, 만기, 예금담보대출 등 구체적인 내용을 검색합니다."""

    docs = _search(query)

    if not docs:
        return config.MSG_NO_DOC

    # 같은 질문에서 여러 번 검색해도 인용 번호가 겹치지 않게 이어서 붙입니다.
    retrieved = _retrieved_var.get()
    start = len(retrieved) + 1
    retrieved.extend(docs)

    return _build_context(docs, start)


# ============================================================
# 3-2. 에이전트 만들기
# ============================================================

TOOLS = [search_company_docs, get_stock_price, get_exchange_rate, calculate]

agent = create_agent(
    model=llm,
    tools=TOOLS,
    system_prompt=AGENT_SYSTEM_PROMPT,
)


# ============================================================
# 4. 질문 처리 함수
# ============================================================
# 에이전트가 Tool 호출 여부와 횟수를 스스로 판단합니다.
# 대화 기록(memory)이 있으면 함께 전달하고, 답변 후 기록에 저장합니다.
# ============================================================

def ask(question: str, k: int = None, memory=None) -> dict:

    # 질문이 없거나 공백만 있으면 종료합니다.
    if not question or not question.strip():

        return {
            "answer": "질문을 입력해주세요.",
            "sources": [],
            "ok": False
        }

    # 이번 질문의 검색 결과를 담을 목록을 준비합니다.
    docs = []
    _retrieved_var.set(docs)

    # (요약 + 최근 대화) → 이번 질문 순서로 전달합니다.
    history = memory.messages() if memory else []

    try:

        result = agent.invoke(
            {"messages": [*history, HumanMessage(question)]},
            config={"recursion_limit": RECURSION_LIMIT},
        )

    except Exception as error:

        print("[에이전트 오류]", error)

        return {
            "answer": config.MSG_ERROR,
            "sources": [],
            "ok": False
        }

    messages = result["messages"]
    answer = messages[-1].content

    # 이번 질문 이후에 호출된 Tool 이름을 순서대로 모읍니다.
    start = max(i for i, m in enumerate(messages) if m.type == "human")
    used_tools = [
        call["name"]
        for m in messages[start:]
        for call in (getattr(m, "tool_calls", None) or [])
    ]

    # --------------------------------------------------------
    # 출처 정리 (문서 검색을 사용한 경우에만)
    # --------------------------------------------------------

    sources = []

    for doc in docs:

        page_no = doc.metadata.get("page_no")

        if page_no is None:
            page_no = doc.metadata.get("page", 0) + 1

        sources.append({
            "file": doc.metadata.get("filename", "unknown"),
            "page": page_no
        })

    # --------------------------------------------------------
    # 인용 번호 확인 (문서 검색을 사용한 경우에만)
    # --------------------------------------------------------

    # 문서를 쓰지 않은 답변은 인용 대상이 아니므로 None으로 둡니다.
    cited = None

    if docs:

        numbers = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]

        cited = len(numbers) > 0 and all(
            1 <= n <= len(docs) for n in numbers
        )

    # 대화 기록에 저장합니다. (5턴 초과 시 오래된 턴이 자동 압축됩니다)
    if memory is not None:
        memory.add(question, answer, llm)

    return {
        "answer": answer,
        "sources": sources,
        "cited": cited,
        "tools": used_tools,
        "ok": True
    }
