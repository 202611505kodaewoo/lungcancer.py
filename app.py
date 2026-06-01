# app.py - 폐암 위험도 예측 및 분석 대시보드
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 페이지 설정
# -------------------------------
st.set_page_config(page_title="🫁 폐암 위험 예측기", layout="wide")

# 커스텀 CSS (선택)
st.markdown("""
<style>
    .stApp { background-color: #f5f7fa; }
    .big-font { font-size:20px !important; font-weight: bold; }
    .pred-card {
        background-color: #1e2a3e;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 1. 데이터 로드
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("lung.csv")
    return df

# 2. 모델 및 스케일러 로드
@st.cache_resource
def load_model():
    with open("lung_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_scaler():
    with open("lung_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler

# -------------------------------
# 3. 주요 기능
# -------------------------------
st.title("🫁 폐암 위험도 예측 대시보드")
st.markdown("임상 데이터를 기반으로 **폐암 발생 가능성**을 예측합니다.")

# 데이터 로드 시도
try:
    df = load_data()
    st.success("✅ 데이터 로드 완료")
except:
    st.warning("⚠️ lung.csv 파일을 찾을 수 없습니다. 샘플 데이터로 데모를 실행합니다.")
    # 샘플 데이터 생성 (실제 데이터가 없을 경우 대비)
    df = pd.DataFrame({
        '나이': np.random.randint(30, 80, 100),
        '흡연': np.random.choice([0,1,2], 100),
        '기침': np.random.choice([1,2,3], 100),
        '호흡곤란': np.random.choice([1,2,3], 100),
        '목표변수': np.random.choice([0,1], 100)
    })

try:
    model = load_model()
    scaler = load_scaler()
    st.success("✅ 모델 및 스케일러 로드 완료")
except:
    st.error("❌ 모델 파일(lung_model.pkl) 또는 스케일러 파일(lung_scaler.pkl)이 없습니다. 먼저 학습하세요.")
    st.stop()

# -------------------------------
# 4. 데이터 탐색 (EDA)
# -------------------------------
st.subheader("📊 데이터 미리보기 및 통계")
col1, col2 = st.columns(2)
with col1:
    st.dataframe(df.head(10), use_container_width=True)
with col2:
    st.write("### 기술 통계")
    st.dataframe(df.describe(), use_container_width=True)

# 상관관계 히트맵 (숫자 컬럼만)
st.subheader("🔍 특징 간 상관관계")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) > 1:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)
else:
    st.info("상관관계를 표시할 수치형 컬럼이 부족합니다.")

# -------------------------------
# 5. 예측 입력 폼 (사용자 입력)
# -------------------------------
st.subheader("🧑‍⚕️ 환자 정보 입력")
st.markdown("아래 항목을 입력하면 폐암 위험도를 예측해드립니다.")

# 실제 사용하는 feature 이름은 모델 학습 시 사용된 컬럼과 동일해야 함.
# 여기서는 일반적인 폐암 위험 요인들을 예시로 작성 (사용자 lung.csv에 맞게 수정 필요)
# 실제로는 lung.csv의 독립변수 컬럼명을 확인하여 아래를 맞춰주세요.

# 예시 feature set (일반적)
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("나이", 20, 90, 55)
        smoking = st.selectbox("흡연 여부", ["비흡연", "과거 흡연", "현재 흡연"])
        yellow_fingers = st.selectbox("손가락 변색", ["없음", "약간", "심함"])
    with col2:
        anxiety = st.selectbox("불안감", ["낮음", "보통", "높음"])
        peer_pressure = st.selectbox("주변 흡연 압력", ["없음", "약간", "심함"])
        chronic_disease = st.selectbox("만성질환", ["없음", "있음"])
    with col3:
        fatigue = st.selectbox("피로감", ["낮음", "보통", "높음"])
        allergy = st.selectbox("알레르기", ["없음", "있음"])
        wheezing = st.selectbox("천명음", ["없음", "가끔", "자주"])
    
    submitted = st.form_submit_button("🩺 위험도 예측하기")

# 매핑 (수치화)
smoking_map = {"비흡연":0, "과거 흡연":1, "현재 흡연":2}
yellow_map = {"없음":0, "약간":1, "심함":2}
anxiety_map = {"낮음":0, "보통":1, "높음":2}
peer_map = {"없음":0, "약간":1, "심함":2}
chronic_map = {"없음":0, "있음":1}
fatigue_map = {"낮음":0, "보통":1, "높음":2}
allergy_map = {"없음":0, "있음":1}
wheezing_map = {"없음":0, "가끔":1, "자주":2}

if submitted:
    input_data = np.array([[
        age,
        smoking_map[smoking],
        yellow_map[yellow_fingers],
        anxiety_map[anxiety],
        peer_map[peer_pressure],
        chronic_map[chronic_disease],
        fatigue_map[fatigue],
        allergy_map[allergy],
        wheezing_map[wheezing]
    ]])
    
    # 스케일링
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0] if hasattr(model, "predict_proba") else [0,0]
    
    # 결과 표시
    st.markdown("---")
    if prediction == 1:
        st.error("## 🔴 위험도: 높음 (폐암 가능성 의심)")
        st.warning(f"예측 확률: {proba[1]*100:.1f}%")
        st.info("의료 전문가와 상담하시길 권장합니다.")
    else:
        st.success("## 🟢 위험도: 낮음 (정상 범위)")
        st.info(f"예측 확률: {proba[0]*100:.1f}% (정상)")
    
    # feature importance 시각화 (모델이 트리 기반일 경우)
    if hasattr(model, "feature_importances_"):
        st.subheader("📌 주요 영향 요인")
        features = ["나이", "흡연", "손가락변색", "불안감", "주변압력", "만성질환", "피로감", "알레르기", "천명음"]
        importances = model.feature_importances_
        fig, ax = plt.subplots()
        ax.barh(features, importances, color='coral')
        ax.set_xlabel("중요도")
        st.pyplot(fig)

# -------------------------------
# 6. 추가 정보 (주의사항)
# -------------------------------
st.markdown("---")
st.caption("⚠️ 본 예측은 통계적 모델에 기반한 참고용입니다. 정확한 진단은 의사와 상담하세요.")
