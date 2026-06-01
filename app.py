# app.py - 폐암 위험 예측 (자동 전처리 + 숫자 열만 사용)
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
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 1. 데이터 로드 및 전처리 + 모델 학습 (파일 없으면 자동)
# -------------------------------
@st.cache_resource
def get_model_and_scaler():
    # lung.csv 존재 확인
    if not os.path.exists("lung.csv"):
        st.error("❌ lung.csv 파일이 없습니다. 샘플 데이터로 데모 모드를 실행합니다.")
        # 샘플 데이터 생성 (데모) - 모두 숫자
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
    
    # 1) 숫자형 컬럼만 선택 (문자열 등 제거)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        st.error("❌ 숫자형 특성이 하나도 없습니다. 데이터를 확인하세요.")
        st.stop()
    X = X[numeric_cols]
    
    # 2) 결측치 처리 (간단히 해당 행 제거)
    before = len(X)
    X = X.dropna()
    y = y.loc[X.index]  # 같은 인덱스만 유지
    after = len(X)
    if before != after:
        st.warning(f"⚠️ 결측치 {before-after}개 행을 제거했습니다.")
    
    # 특성 이름 저장 (나중에 입력 폼에 사용)
    feature_names = numeric_cols
    
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
    
    return model, scaler, feature_names

# 모델 로드 또는 학습
try:
    # 이미 파일이 있으면 불러오기
    with open("lung_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("lung_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    # feature 이름은 저장 안 되어 있으므로 다시 CSV에서 읽거나 기본 사용
    if os.path.exists("lung.csv"):
        df_temp = pd.read_csv("lung.csv")
        X_temp = df_temp.iloc[:, :-1]
        feature_names = X_temp.select_dtypes(include=[np.number]).columns.tolist()
        if not feature_names:
            feature_names = ['특성1', '특성2']  # fallback
    else:
        feature_names = ['나이', '흡연', '손가락변색', '불안감', '주변압력', '만성질환', '피로감', '알레르기', '천명음']
    st.success("✅ 저장된 모델과 스케일러를 불러왔습니다.")
except:
    st.warning("⚠️ 모델 파일이 없습니다. lung.csv로부터 새로 학습합니다...")
    model, scaler, feature_names = get_model_and_scaler()
    st.success("✅ 학습 완료! 모델과 스케일러가 저장되었습니다.")

# -------------------------------
# 2. 데이터 탐색 (CSV 존재 시)
# -------------------------------
st.title("🫁 폐암 위험도 예측 대시보드")
st.markdown("숫자형 임상 데이터 기반 **폐암 가능성** 예측 (랜덤포레스트)")

if os.path.exists("lung.csv"):
    df_raw = pd.read_csv("lung.csv")
    st.subheader("📊 원본 데이터 미리보기 (숫자 열만 사용)")
    # 숫자 열만 표시
    numeric_df = df_raw.select_dtypes(include=[np.number])
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(numeric_df.head(10), use_container_width=True)
    with col2:
        st.dataframe(numeric_df.describe(), use_container_width=True)
    
    # 상관관계 히트맵 (숫자 열만)
    if numeric_df.shape[1] > 1:
        st.subheader("🔍 특성 간 상관관계")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# -------------------------------
# 3. 예측 입력 폼 (동적으로 생성)
# -------------------------------
st.subheader("🧑‍⚕️ 환자 정보 입력")
st.markdown("아래 항목을 모두 입력해주세요 (숫자 값만 허용).")

input_data = []
with st.form("pred_form"):
    cols = st.columns(3)
    for i, feat in enumerate(feature_names):
        col_idx = i % 3
        # 기본값: 각 특성의 중앙값을 샘플 데이터에서 추정하거나 0
        default_val = 0.0
        if os.path.exists("lung.csv"):
            try:
                default_val = float(pd.read_csv("lung.csv")[feat].median())
            except:
                default_val = 0.0
        val = cols[col_idx].number_input(feat, value=default_val, step=0.1, format="%.2f")
        input_data.append(val)
    
    submitted = st.form_submit_button("🩺 위험도 예측하기")

if submitted:
    # 입력값을 numpy 배열로 변환
    input_array = np.array([input_data])
    # 스케일러는 이미 학습된 것 사용
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
