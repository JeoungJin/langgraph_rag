# ============================================================
# 대화 기록을 관리합니다.
#   - 최근 MAX_TURNS턴은 원문 그대로 LLM에게 전달합니다.
#   - 그보다 오래된 턴은 LLM으로 요약해 summary 한 덩어리로 압축합니다.
# ============================================================

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from prompts import SUMMARY_PROMPT

# 원문으로 유지할 최근 대화 턴 수 (질문+답변 = 1턴)
MAX_TURNS = 5


def _strip_citations(text):
    """이전 답변의 [1] 같은 근거 번호는 새 질문에서는 의미가 없으므로 제거합니다."""
    return re.sub(r"\s*\[\d+\]", "", text)


class ConversationMemory:

    def __init__(self):
        self.turns = []      # [(질문, 답변), ...] 최근 대화 원문
        self.summary = ""    # 오래된 대화의 압축본

    # ----------------------------------------------------
    # LLM 입력용 메시지 만들기
    # ----------------------------------------------------
    def messages(self):

        msgs = []

        if self.summary:
            msgs.append(SystemMessage(
                "[이전 대화 요약]\n" + self.summary + "\n"
                "(요약 속 시세는 당시 값이므로 최신 값이 필요하면 Tool을 다시 호출하십시오.)"
            ))

        for question, answer in self.turns:
            msgs.append(HumanMessage(question))
            msgs.append(AIMessage(answer))

        return msgs

    # ----------------------------------------------------
    # 한 턴 추가 후, 5턴을 넘으면 오래된 턴을 압축
    # ----------------------------------------------------
    def add(self, question, answer, llm):

        self.turns.append((question, _strip_citations(answer)))

        overflow = len(self.turns) - MAX_TURNS

        if overflow <= 0:
            return

        old = self.turns[:overflow]

        dialogue = "\n".join(f"사용자: {q}\n도우미: {a}" for q, a in old)

        try:
            self.summary = (
                SUMMARY_PROMPT | llm | StrOutputParser()
            ).invoke({
                "summary": self.summary or "(없음)",
                "dialogue": dialogue,
            }).strip()

        except Exception as error:
            # 요약에 실패하면 원문을 유지해 정보 손실을 막습니다.
            print("[요약 오류]", error)
            return

        self.turns = self.turns[overflow:]

    def clear(self):
        self.turns = []
        self.summary = ""
