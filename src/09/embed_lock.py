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

vector = embedding.embed_query("환불")
vector2 = embedding.embed_query("반품")

print(f"Embedding  차원수:{len(vector)}")

# first_vector = vector[:10]
# first_vector2 = vector2[:10]
# print(f"Embedding  첫 10개:{first_vector}")
# print(f"Embedding  첫 10개:{first_vector2}")
# similarity = np.dot(vector, vector2) / (norm(vector) * norm(vector2))
# print(f"Embedding  유사도:{similarity}")


def sim(sentence1, sentence2):
    vector1 = embedding.embed_query(sentence1)
    vector2 = embedding.embed_query(sentence2)

    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    dot_product = np.dot(vector1, vector2)
    length1 = np.linalg.norm(vector1)
    length2 = np.linalg.norm(vector2)
    similarity = dot_product / (length1 * length2)
    return float(similarity)

PAIRS = [
    ("환불 규정", "반품 절차"),
    ("비밀번호 변경", "패스워드 재설정"),
    ("제품이 고장났어요", "기기 불량 신고"),
    ("환불 규정", "배송 안내"),
    ("환불 규정", "오늘 날씨 어때?"),
]

for sentence1, sentence2 in PAIRS:
    print(f"{sentence1} vs {sentence2}: {sim(sentence1, sentence2)}")