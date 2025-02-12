# 한국 부동산 데이터 시각화

### 1. 분석 목적
  - 배경 : 한국의 부동산 변화를 연구하여 사회 전반에 미치는 영향을 파악
  - 목표
    + EDA를 활용해 외부데이터 활용 탐색
    + 외부데이터를 활용해 아파트 실거래 예측 모델링 구현

### 2. 분석 과정
  - 분석 도구
    + python
  - 분석 데이터
    + dacon 제공 데이터
    + 코스피 지수
    + 서울 아파트거래량
    + 출생아 수
    + 자동차 등록대수
  - 분석 방법
    + 데이터 전처리 및 변수 정제
    + EDA
    + LSTM 모델 사용

### 3. 분석 결과
  - 외부데이터 탐색 후 유의미한 데이터인지 EDA를 통해 확인
  - 모델 구축
    + LSTM
       +  train set의 loss와 test set의 loss 모두 0에 가까운 값으로 수렴, 예측 성능 또한 좋음
       +  변수를 하나만 사용하더라도 데이터를 많이 확보한 모델이 성능이 더 좋음

### 4. 회고
  - 외부 데이터 활용이 어려웠음

[notion](https://www.notion.so/2021-3cc0c9353e094e32bd908eab53ff4ff6?pvs=4)
