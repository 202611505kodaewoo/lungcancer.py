# app.py - 폐암 위험 예측 (모델 파일 없으면 자동 학습)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------
# 페이지 설정
# -------------------------------
st.set_page_config(page_title="🫁 폐암 위험 예측기", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f5f7fa; }
    .pred-card { background-color: #1e2a3e; border-radius: 15px; padding: 1.5rem; text-align: center; color: white; }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 1. 데이터 로드 및 모델 학습 (파일 없으면)
# -------------------------------
@st.cache_resource
def get_model_and_scaler():
    # lung.csv 존재 확인
    if not os.path.exists("lung.csv"):
        st.error("❌ lung.csv 파일이 없습니다. 샘플 데이터로 데모 모드를 실행합니다.")
        # 샘플 데이터 생성 (데모)
        df = pd.DataFrame({
            '나이': np.random.randint(30, 80, 200),
            '흡연': np.random.choice([0,1,2], 200),
            '손가락변색': np.random.choice([0,1,2], 200),
            '불안감': np.random.choice([0,1,2], 200),
            '주변압력': np.random.choice([0,1,2], 200),
            '만성질환': np.random.choice([0,1], 200),
            '피로감': np.random.choice([0,1,2], 200),
            '알레르기': np.random.choice([0,1], 200),
            '천명음': np.random.choice([0,1,2], 200),
            '목표변수': np.random.choice([0,1], 200)
        })
    else:
        df = pd.read_csv("lung.csv")
    
    # 특성과 타겟 분리 (마지막 컬럼이 target이라고 가정)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 스케일러 학습
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 모델 학습 (랜덤포레스트)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # 학습된 모델과 스케일러를 파일로 저장 (다음 실행부터 바로 불러오기 위함)
    with open("lung_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("lung_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    return model, scaler, X.columns.tolist()

# 모델 로드 또는 학습
try:
    # 이미 파일이 있으면 불러오기
    with open("lung_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("lung_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    # feature 이름은 별도로 저장하지 않았으므로, lung.csv에서 읽어옴
    if os.path.exists("lung.csv"):
        df_temp = pd.read_csv("lung.csv")
        feature_names = df_temp.iloc[:, :-1].columns.tolist()
    else:
        feature_names = ['나이', '흡연', '손가락변색', '불안감', '주변압력', '만성질환', '피로감', '알레르기', '천명음']
    st.success("✅ 저장된 모델과 스케일러를 불러왔습니다.")
except:
    st.warning("⚠️ 모델 파일이 없습니다. lung.csv로부터 새로 학습합니다...")
    model, scaler, feature_names = get_model_and_scaler()
    st.success("✅ 학습 완료! 모델과 스케일러가 저장되었습니다.")

# -------------------------------
# 2. 데이터 탐색
# -------------------------------
st.title("🫁 폐암 위험도 예측 대시보드")
st.markdown("임상 데이터 기반 **폐암 가능성** 예측 (랜덤포레스트 모델)")

# CSV가 있으면 원본 데이터 표시
if os.path.exists("lung.csv"):
    df_raw = pd.read_csv("lung.csv")
    st.subheader("📊 원본 데이터 미리보기")
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df_raw.head(10), use_container_width=True)
    with col2:
        st.dataframe(df_raw.describe(), use_container_width=True)
    
    # 상관관계 히트맵
    st.subheader("🔍 특성 간 상관관계")
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_raw[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# -------------------------------
# 3. 예측 입력 폼
# -------------------------------
st.subheader("🧑‍⚕️ 환자 정보 입력")
st.markdown("아래 항목을 모두 입력해주세요.")

# feature_names에 따라 동적으로 입력 필드 생성
input_data = []
with st.form("pred_form"):
    cols = st.columns(3)
    for i, feat in enumerate(feature_names):
        col_idx = i % 3
        if feat == '나이':
            val = cols[col_idx].number_input(feat, min_value=20, max_value=100, value=55, step=1)
        elif feat in ['흡연', '손가락변색', '불안감', '주변압력', '피로감', '천명음']:
            val = cols[col_idx].selectbox(feat, options=[0,1,2], format_func=lambda x: {0:"없음/낮음",1:"보통/약간",2:"심함/자주"}.get(x,str(x)))
        elif feat in ['만성질환', '알레르기']:
            val = cols[col_idx].selectbox(feat, options=[0,1], format_func=lambda x: "없음" if x==0 else "있음")
        else:
            val = cols[col_idx].number_input(feat, value=0.0)
        input_data.append(val)
    
    submitted = st.form_submit_button("🩺 위험도 예측하기")

if submitted:
    # 입력값을 numpy 배열로 변환
    input_array = np.array([input_data])
    input_scaled = scaler.transform(input_array)
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    
    st.markdown("---")
    if pred == 1:
        st.error(f"## 🔴 위험도: 높음 (폐암 가능성 {proba[1]*100:.1f}%)")
        st.warning("의료 전문가와 상담하시길 권장합니다.")
    else:
        st.success(f"## 🟢 위험도: 낮음 (정상 확률 {proba[0]*100:.1f}%)")
    
    # 특성 중요도 시각화
    if hasattr(model, "feature_importances_"):
        st.subheader("📌 영향력 높은 요인")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        fig, ax = plt.subplots()
        ax.barh(range(len(feature_names)), importances[indices], color='coral')
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel("중요도")
        st.pyplot(fig)

st.caption("⚠️ 본 예측은 참고용입니다. 정확한 진단은 의사와 상담하세요.")
