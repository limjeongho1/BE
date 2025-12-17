# from sentence_transformers import SentenceTransformer
# import pandas as pd
# import numpy as np
# import re


# def normalize_text(text: str) -> str:
#     """
#     SBERT 검색 안정화를 위한 텍스트 정규화
#     """
#     text = str(text)

#     # 의미 없는 구분자 통일
#     text = re.sub(r"[,\u00b7・]", " ", text)

#     # 다중 공백 제거
#     text = re.sub(r"\s+", " ", text)

#     return text.strip()


# def generate_and_save_embeddings(input_csv_path, output_csv_path, target_columns, encoding_type='utf-8'):
#     # --- 1. 파일 로드 ---
#     try:
#         df = pd.read_csv(input_csv_path, encoding=encoding_type)
#         print(f"✅ '{input_csv_path}' 파일 로드 완료. 총 {len(df)}개 데이터.")
#     except Exception as e:
#         print(f"❌ CSV 로드 오류: {e}")
#         return

#     # --- 2. 모델 로드 ---
#     model = SentenceTransformer("all-MiniLM-L6-v2")

#     # --- 3. 코퍼스 텍스트 생성 (🔥 전처리 포함) ---
#     df = df.fillna('')

#     df['corpus_text'] = df.apply(
#         lambda row: normalize_text(
#             " ".join([str(row[col]) for col in target_columns if col in df.columns])
#         ),
#         axis=1
#     )

#     corpus = df['corpus_text'].tolist()
#     print(f"➡️ 전처리된 코퍼스 생성 완료 (예시):\n{corpus[0]}")

#     # --- 4. SBERT 임베딩 계산 ---
#     print("➡️ SBERT 임베딩 계산 중...")
#     embeddings = model.encode(
#         corpus,
#         convert_to_numpy=True,
#         normalize_embeddings=True,
#         show_progress_bar=True
#     )

#     # --- 5. 임베딩만 문자열로 변환 ---
#     embedding_str = [
#         np.array2string(vec, separator=',', max_line_width=np.inf)
#         for vec in embeddings
#     ]

#     # --- 6. 임베딩만 DataFrame으로 저장 ---
#     df_embeddings = pd.DataFrame({
#         'sbert_embedding': embedding_str
#     })

#     try:
#         df_embeddings.to_csv(output_csv_path, index=False, encoding=encoding_type)
#         print(f"🎉 임베딩만 저장 완료: '{output_csv_path}'")
#     except Exception as e:
#         print(f"❌ 저장 오류: {e}")


# # --- 실행 설정 ---
# INPUT_CSV = 'language.csv'
# OUTPUT_CSV = 'test_language.csv'
# CSV_ENCODING = 'utf-8'

# TARGET_COLUMNS_TO_EMBED = [
#     '대학명', '단과대학', '학과명', '폐과', '소재지',
#     '소재지(상세)', '주야구분', '학과특징',
#     '표준분류계열(대)', '표준분류계열(중)', '표준분류계열(소)', '관심'
# ]

# # 실행
# generate_and_save_embeddings(
#     INPUT_CSV,
#     OUTPUT_CSV,
#     TARGET_COLUMNS_TO_EMBED,
#     CSV_ENCODING
# )

from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import re


def normalize_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"[,\u00b7・]", " ", text)   # , · ・ 제거
    text = re.sub(r"\s+", " ", text)          # 공백 정리
    return text.strip()


def generate_and_save_embeddings(input_csv, output_csv, target_columns):
    df = pd.read_csv(input_csv).fillna("")

    # corpus 생성 + 전처리
    df["corpus_text"] = df.apply(
        lambda row: normalize_text(
            " ".join([str(row[col]) for col in target_columns if col in df.columns])
        ),
        axis=1
    )

    # 🔥 E5는 prefix가 핵심
    corpus = ["passage: " + text for text in df["corpus_text"].tolist()]

    model = SentenceTransformer("intfloat/multilingual-e5-base")

    embeddings = model.encode(
        corpus,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    df_out = pd.DataFrame({
        "embedding": [vec.tolist() for vec in embeddings]
    })

    df_out.to_csv(output_csv, index=False)
    print(f"✅ 임베딩 저장 완료: {output_csv}")


# 실행
TARGET_COLUMNS = [
    '대학명', '단과대학', '학과명', '폐과', '소재지',
    '소재지(상세)', '주야구분', '학과특징',
    '표준분류계열(대)', '표준분류계열(중)', '표준분류계열(소)', '관심'
]

generate_and_save_embeddings(
    input_csv="language.csv",
    output_csv="test_language.csv",
    target_columns=TARGET_COLUMNS
)
