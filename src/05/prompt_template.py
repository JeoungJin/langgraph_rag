from  langchain_core.prompts import PromptTemplate

template = """
다음 문서를 읽고 아래 질문에 답해줘.
Context: {doc}
Question: {question}
Answer:
"""


prompt_template = PromptTemplate(
    input_variables=["doc", "question"],
    template=template
)

prompt = prompt_template.format(doc="제품색상은 빨간색입니다.", question="제품 색상은 무엇인가요?")
print(prompt)

