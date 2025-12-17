# ======================================================
# 대학 · 학과 추천 챗봇 (Embedding + GPT-4-mini)
# ======================================================

import os
import pandas as pd
import numpy as np
import ast
import re
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv


# ======================================================
# 1. .env 로드 & OpenAI 설정
# ======================================================
load_dotenv()  # 🔥 중요

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# ======================================================
# 2. 텍스트 정규화
# ======================================================
def normalize_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"[,\u00b7・]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ======================================================
# 3. 간단한 의도 추출
# ======================================================
def extract_intent(text):
    regions = ["서울", "경기", "부산", "대구", "인천", "광주", "대전", "울산"]
    majors = ["컴퓨터", "소프트웨어", "AI", "인공지능", "정보", "데이터"]

    region = next((r for r in regions if r in text), None)
    major = next((m for m in majors if m.lower() in text.lower()), None)

    return region, major


# ======================================================
# 4. CSV 로드 (🔥 핵심)
# ======================================================
print("📂 CSV 로딩 중...")

# 1️⃣ 학과 메타데이터
df_info = pd.read_csv("language.csv").fillna("")
df_info.columns = df_info.columns.str.strip()

# 컬럼명 깨짐 교정
df_info.rename(
    columns={
        "표준분 류계열(소)": "표준분류계열(소)"
    },
    inplace=True
)

# 2️⃣ 임베딩
df_embed = pd.read_csv("test_language.csv")
df_embed["embedding"] = df_embed["embedding"].apply(
    lambda x: np.array(ast.literal_eval(x))
)

corpus_embeddings = np.vstack(df_embed["embedding"].values)

model = SentenceTransformer("intfloat/multilingual-e5-base")

print("✅ 로딩 완료\n")


# ======================================================
# 5. 학과 검색
# ======================================================
def search_major(user_query, top_k=3):
    query = "query: " + normalize_text(user_query)

    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = np.dot(corpus_embeddings, query_embedding)

    df_result = df_info.copy()
    df_result["score"] = scores

    region, major = extract_intent(user_query)

    if region:
        df_result = df_result[df_result["소재지"].str.contains(region, na=False)]

    if major:
        df_result = df_result[
            df_result["학과명"].str.contains(major, case=False, na=False)
        ]

    return df_result.sort_values("score", ascending=False).head(top_k)


# ======================================================
# 6. GPT 프롬프트 생성
# ======================================================
def build_gpt_prompt(user_query, results_df):
    context = ""

    for _, row in results_df.iterrows():
        context += f"""
대학명: {row['대학명']}
단과대학: {row['단과대학']}
학과명: {row['학과명']}
소재지: {row['소재지']}
학과특성: {row.get('학과특성', '')}
표준계열(중): {row.get('표준분류계열(중)', '')}
---
"""

    return f"""
너는 한국의 대학 입시 및 진로 전문 상담 챗봇이다.

[사용자 질문]
{user_query}

[추천 가능한 학과 정보]
{context}

요청사항:
1. 사용자 질문에 맞는 학과를 추천해라
2. 학과 특징을 쉽게 설명해라
3. 졸업 후 진로와 전망을 알려라
4. 자연스럽고 친절한 한국어로 답변해라
5. 제공된 정보 외의 내용은 지어내지 마라
"""


# ======================================================
# 7. GPT-4-mini 호출
# ======================================================
def call_gpt4_mini(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "너는 대학 입시 전문 상담 챗봇이다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    return response.choices[0].message.content


# ======================================================
# 8. 챗봇 실행
# ======================================================
print("🤖 대학 · 학과 추천 챗봇")
print("예: 나는 서울에 있는 컴퓨터공학과에 가고 싶어")
print("종료: exit / 종료\n")

while True:
    user_input = input("🙋 사용자: ")

    if user_input.lower() in ["exit", "종료", "quit"]:
        print("👋 챗봇 종료")
        break

    results = search_major(user_input, top_k=3)

    if results.empty:
        print("🤖 조건에 맞는 학과를 찾지 못했어요.\n")
        continue

    prompt = build_gpt_prompt(user_input, results)
    answer = call_gpt4_mini(prompt)

    print("\n🤖 챗봇 답변:\n")
    print(answer)
    print("\n" + "=" * 60 + "\n")
