# Scikit-Learn 머신러닝 프로젝트 (sklearn-est15th-realfinal)

이 저장소는 **Scikit-Learn** 및 최신 ML 프레임워크를 활용한 다양한 머신러닝 실험, 주피터 노트북, 그리고 웹 애플리케이션을 포함하고 있습니다.

## 🚀 주요 기능

-   **타이타닉 생존자 예측**: Gradio 기반의 인터페이스를 통해 실시간 생존 예측 서비스를 제공합니다.
-   **자동화된 데이터 과학 API**: CSV 업로드, 데이터 분석(EDA), 전처리, 동적 모델 학습을 지원하는 FastAPI 백엔드입니다.
-   **심화 머신러닝 기법**: 앙상블 학습(Voting, Stacking), 하이퍼파라미터 최적화(Optuna), 그리고 다양한 알고리즘(SVM, GBM, XGBoost, CatBoost)을 다룹니다.
-   **데이터 시각화**: 상관계수 히트맵 및 특성 분포 분석 도구가 내장되어 있습니다.

## 📂 프로젝트 구조

-   `web_app.py`: Gradio 기반 타이타닉 생존 예측 서비스
-   `api.py`: 자동화된 데이터 처리를 위한 FastAPI 백엔드
-   **Jupyter Notebooks**:
    -   `1_sklearn_start.ipynb` ~ `13_unsupervisedLearning.ipynb`: Scikit-Learn 핵심 커리큘럼
    -   `Plus_*`: 레드 와인 품질 및 캘리포니아 주택 가격 분석 사례 연구
    -   `10_ensemble.ipynb` & `11_ensemble_Optuna.ipynb`: 고급 앙상블 모델링
-   `data/`: 데이터셋 저장소 (Titanic, Wine 등)
-   `static/` & `templates/`: 웹 서비스용 정적 자원 및 템플릿
-   `submission/`: 모델을 통해 생성된 Kaggle 제출용 파일

## 🛠 사용 기술 (Tech Stack)

-   **언어**: Python
-   **머신러닝**: Scikit-Learn, XGBoost, CatBoost, Optuna
-   **웹 프레임워크**: FastAPI (Backend), Gradio (ML Interface), Flask
-   **데이터 처리**: Pandas, NumPy
-   **시각화**: Matplotlib, Seaborn

## ⚙️ 설치 및 실행 방법

### 1. 환경 설정
Python이 설치된 환경에서 아래 명령어를 통해 필요한 패키지를 설치합니다:
```bash
pip install -r requirements.txt
```

### 2. 타이타닉 예측 UI 실행
```bash
python web_app.py
```
접속 주소: `http://127.0.0.1:7860`

### 3. 데이터 과학 API 실행
```bash
python api.py
```
API 문서 확인: `http://127.0.0.1:8000/docs`

### 4. 대화형 분석
VS Code나 Jupyter Notebook에서 각 `.ipynb` 파일을 열어 데이터 전처리 및 모델링 과정을 확인할 수 있습니다.

## 📊 데이터셋 정보
-   **Titanic**: 승객 정보를 바탕으로 한 생존 여부 예측
-   **Red Wine Quality**: 와인 특성에 따른 품질 분류 및 회귀
-   **California Housing**: 선형 회귀 및 다항 특성을 활용한 주택 가격 예측

---
*Scikit-Learn Estimation 15기 프로젝트의 일환으로 제작되었습니다.*
