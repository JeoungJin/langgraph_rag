import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
INDEXER_DIR = os.path.join(CURRENT_DIR, "..", "10")

sys.path.insert(0, INDEXER_DIR)
from indexer2 import get_store

store = get_store(rebuild=False)
QUESTION = "환불과 교환은 어떻게 다른가요?"
def show_result(name,retriever):
    print(f"=== {name} ===")
    docs = retriever.invoke(QUESTION)
    for i, doc in enumerate(docs, 1):
        print(f"[{i}] ({doc.metadata['filename']} 페이지번호 p.{doc.metadata['page_no']})\n{doc.page_content}\n"   )
        print("-" * 50)
similarity_retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
#mmr_retriever = store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5})
show_result("Similarity Retriever", similarity_retriever)
#show_result("MMR Retriever", mmr_retriever)


MIN_SCORE = 0.0

def compare_k(question, ks=(1, 3, 5)):
    print(f"질문: {question}")
    print("=" * 58)
    for k in ks:
        pairs = store.similarity_search_with_relevance_scores(
            question,
            k=k
        )
        scores = []
        low_count = 0
        for doc, score in pairs:
            scores.append(score)
            if score < MIN_SCORE:
                low_count = low_count + 1
        rounded_scores = []
        for score in scores:
            rounded_score = round(float(score), 3)
            rounded_scores.append(rounded_score)

        print(f"k={k:<3} 점수={rounded_scores}")
        print(f"     기준 미달({MIN_SCORE}) 조각: {low_count}개")
        print()

#compare_k("환불은 며칠 이내인가요?", ks=(1, 3, 5))
#compare_k(QUESTION, ks=(1, 3, 5))

#w
import warnings
warnings.filterwarnings("ignore")

manual_retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 3},
      filter=lambda doc: doc.metadata["filename"] == "manual.pdf")
show_result("Manual Retriever", manual_retriever)