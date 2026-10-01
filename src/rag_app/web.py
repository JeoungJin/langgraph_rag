import streamlit as st

from memory import ConversationMemory
from rag import ask

st.set_page_config(page_title="금융 Q&A", page_icon=":material/account_balance:")

TOOL_LABELS = {
    "search_company_docs": "사내 문서 검색",
    "get_stock_price": "주가 조회",
    "get_exchange_rate": "환율 조회",
}

SUGGESTIONS = [
    "예금담보대출 얼마나되는 거야",
    "중도해지하면 이자 어떻게 돼?",
    "삼성전자 주가 알려줘",
    "달러 환율 얼마야?",
]

# ── 세션 상태 ──
if "history" not in st.session_state:
    st.session_state.history = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()


def render_result(result):
    """답변 아래에 출처와 사용한 Tool을 작은 글씨로 보여줍니다."""
    pages = sorted({s["page"] for s in result["sources"]})
    if pages:
        file = result["sources"][0]["file"]
        st.caption(f":material/description: {file} {', '.join(map(str, pages))}페이지")

    tools = [TOOL_LABELS.get(t, t) for t in dict.fromkeys(result.get("tools", []))]
    if tools:
        st.caption(f":material/build: {', '.join(tools)}")

    if result.get("cited") is False:
        st.caption(":material/warning: 인용 표시가 확인되지 않았습니다.")


# ── 헤더 ──
title_col, clear_col = st.columns([5, 1], vertical_alignment="center")
with title_col:
    st.title("금융 Q&A")
with clear_col:
    if st.session_state.history and st.button(
        "지우기", icon=":material/delete:", width="stretch"
    ):
        st.session_state.history = []
        st.session_state.memory.clear()
        st.rerun()

st.caption("우리은행 상품 문서 · 주가 · 환율 · 일반 금융 지식을 물어보세요.")

# ── 지난 대화 (위에서 아래로 쌓입니다) ──
for question, result in st.session_state.history:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write(result["answer"])
        render_result(result)

# ── 입력 ──
# height=136: 기본(68px)의 2배. 글이 길어지면 자동으로 더 늘어납니다.
q = st.chat_input("궁금한 것을 물어보세요", height=136)

# 첫 화면에서만 예시 질문을 보여줍니다.
if not st.session_state.history and not q:
    picked = st.pills(
        "예시 질문",
        SUGGESTIONS,
        label_visibility="collapsed",
        key="suggestion",
    )
    if picked:
        q = picked

# ── 새 질문 처리: 기존 대화 바로 아래에 이어서 표시합니다 ──
if q:
    with st.chat_message("user"):
        st.write(q)

    with st.chat_message("assistant"):
        with st.spinner("답변을 만드는 중...", show_time=True):
            result = ask(q, memory=st.session_state.memory)
        st.write(result["answer"])
        render_result(result)

    st.session_state.history.append((q, result))
    st.session_state.pop("suggestion", None)

# ── 대화 기록 확인 (사이드바) ──
# 스크립트 맨 끝에서 그려야 방금 처리한 질문까지 반영됩니다.
with st.sidebar:
    st.subheader("대화 기록")
    mem = st.session_state.memory

    st.caption(f"원문 {len(mem.turns)}턴 (최대 5턴) · 요약 {'있음' if mem.summary else '없음'}")

    with st.expander("이전 대화 요약", icon=":material/compress:"):
        st.write(mem.summary or "아직 압축된 대화가 없습니다. (6턴째부터 생성)")

    with st.expander("최근 원문 대화", icon=":material/forum:"):
        if mem.turns:
            for i, (mq, ma) in enumerate(mem.turns, 1):
                st.markdown(f"**{i}. {mq}**")
                st.caption(ma)
        else:
            st.write("기록이 없습니다.")
