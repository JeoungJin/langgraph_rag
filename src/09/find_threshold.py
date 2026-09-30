from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import os
import re
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy  as np
from numpy.linalg import norm

load_dotenv()

embedding = OpenAIEmbeddings(model = "text-embedding-3-small")

BASE = "환불은 상품 수령 후 7일 이내에 신청할 수 있습니다"
RELATED = [
    "반품하고 싶어요",
    "돈 돌려받을 수 있나요?",
    "구매 취소 기간이 어떻게 되나요?",
    "물건 안 마음에 들면 어떡하죠?",
    "환불 신청 기한 알려주세요",
]

UNRELATED = [
    "회사 주차장은 어디인가요?",
    "채용 공고 보고 싶어요",
    "오늘 날씨 어때요?",
    "대표이사가 누구인가요?",
    "직원 복지 제도 알려주세요",
]

def sim(sentence1, sentence2):
    vector1 = embedding.embed_query(sentence1)
    vector2 = embedding.embed_query(sentence2)

    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    dot_product = np.dot(vector1, vector2)
    length1 = np.linalg.norm(vector1)
    length2 = np.linalg.norm(vector2)
    similarity = dot_product / (length1 * length2)
    return round(float(similarity), 4)


related_similarities = [sim(BASE, sentence) for sentence in RELATED]
unrelated_similarities = [sim(BASE, sentence) for sentence in UNRELATED]

print("관련 문장 유사도:")
for sentence, similarity in zip(RELATED, related_similarities):
    print(f"{sentence}: {similarity}")      
print("\n관련 없는 문장 유사도:")
for sentence, similarity in zip(UNRELATED, unrelated_similarities):
    print(f"{sentence}: {similarity}")  

#관련있는 문장의 최소 --- 
#관련없는 문장의 최대 
#임계값 계산 
min_related_similarity = min(related_similarities)
max_unrelated_similarity = max(unrelated_similarities)

threshold = (min_related_similarity + max_unrelated_similarity) / 2
print(f"\n임계값 후보: {round(threshold, 4)}")

#그러면 누가 관련있는 문장인지 없는 문장인지 판단할 수 있는 임계값을 찾을 수 있음
#임계값을 찾고나면 어떻게해야하지? 임계값 이상만 선택 
