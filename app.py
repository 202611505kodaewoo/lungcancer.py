import pandas as pd
import pickle
import sys

def main():
    # 1. 저장된 모델과 스케일러 로드
    try:
        with open('lung_model.pkl', 'rb') as f:
            kmeans_model = pickle.load(f)
        with open('lung_scaler.pkl', 'rb') as f:
            scaler_cluster = pickle.load(f)
        print("모델 로드 완료.")
    except FileNotFoundError as e:
        print(f"모델 파일을 찾을 수 없습니다: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        sys.exit(1)

    # 2. 사용자 입력
    print("\n새로운 환자의 정보를 입력해주세요:")
    try:
        age_input = float(input("나이 입력: "))
        smoking_input = float(input("흡연 (1: YES, 2: NO) 입력: "))
        alcohol_input = float(input("음주 (1: YES, 2: NO) 입력: "))
    except ValueError:
        print("숫자를 입력해주세요.")
        sys.exit(1)

    # 3. 입력 데이터를 DataFrame으로 변환
    new_patient_data = pd.DataFrame([[age_input, smoking_input, alcohol_input]],
                                     columns=['나이', '흡연', '음주'])

    # 4. 스케일링 및 군집 예측
    new_patient_scaled = scaler_cluster.transform(new_patient_data)
    predicted_cluster = kmeans_model.predict(new_patient_scaled)[0]

    # 5. 결과 출력
    print(f"\n이 환자는 {predicted_cluster}번 군집에 속합니다.")

if __name__ == "__main__":
    main()
