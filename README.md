# Scikit-Learn 머신러닝 프로젝트 (sklearn-est15th-realfinal)

이 저장소는 **Antigravity AI**와 함께 개발된 프로젝트로, **Scikit-Learn** 및 최신 ML 프레임워크를 활용한 다양한 머신러닝 실험, 데이터 분석 코드, 그리고 웹 애플리케이션을 포함하고 있습니다.

## 🚀 주요 기능

-   **Antigravity 기반 개발**: Antigravity AI 코딩 어시스턴트를 활용하여 모델링, 분석 및 서비스 구현을 자동화하고 가속화했습니다.
-   **타이타닉 생존자 예측**: Gradio 기반의 인터페이스를 통해 실시간 생존 예측 서비스를 제공합니다.
-   **자동화된 데이터 과학 API**: CSV 업로드, 데이터 분석(EDA), 전처리, 동적 모델 학습을 지원하는 FastAPI 백엔드입니다.
-   **심화 머신러닝 기법**: 앙상블 학습(Voting, Stacking), 하이퍼파라미터 최적화(Optuna), 그리고 다양한 알고리즘(SVM, GBM, XGBoost, CatBoost)을 다룹니다.
-   **데이터 시각화**: 상관계수 히트맵 및 특성 분포 분석 도구가 내장되어 있습니다.

## 📂 프로젝트 구조

-   `web_app.py`: Gradio 기반 타이타닉 생존 예측 서비스
-   `api.py`: 자동화된 데이터 처리를 위한 FastAPI 백엔드
-   **Data Science Workflows**:
    -   `1_sklearn_start.ipynb` ~ `13_unsupervisedLearning.ipynb`: Scikit-Learn 핵심 커리큘럼 기반 분석
    -   `Plus_*`: Antigravity로 고도화된 레드 와인 품질 및 캘리포니아 주택 가격 분석 사례
    -   `10_ensemble.ipynb` & `11_ensemble_Optuna.ipynb`: 고급 앙상블 모델링 및 최적화
-   `data/`: 데이터셋 저장소 (Titanic, Wine 등)
-   `static/` & `templates/`: 웹 서비스용 정적 자원 및 템플릿
-   `submission/`: 모델을 통해 생성된 Kaggle 제출용 파일

## 🛠 사용 기술 (Tech Stack)

-   **AI Assistant**: **Antigravity (Advanced Agentic Coding)**
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

### 2. Antigravity를 통한 분석 및 개발
이 프로젝트는 Antigravity와 함께 상호작용하며 고도화되었습니다. 분석이나 추가 기능 구현이 필요할 때 Antigravity에게 다음과 같이 요청할 수 있습니다:
- "기존 앙상블 모델에 새로운 알고리즘을 추가해줘"
- "현재 데이터셋에 대한 시각화 보고서를 생성해줘"

### 3. 웹 서비스 실행
- **타이타닉 예측 UI**: `python web_app.py` (접속: `http://127.0.0.1:7860`)
- **데이터 과학 API**: `python api.py` (문서: `http://127.0.0.1:8000/docs`)

---
*Antigravity와 함께하는 Scikit-Learn Estimation 15기 프로젝트.*
