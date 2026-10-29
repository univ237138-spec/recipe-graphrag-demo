# 레시피·알레르기 GraphRAG 데모

레시피 속 숨은 알레르기를 온톨로지 지식그래프로 찾아, 근거 경로와 함께 답하는 챗봇 데모입니다.
이 저장소에는 **데모 앱 실행에 필요한 파일만** 들어 있습니다.

- 바로 보기: https://recipe-graphrag-3jjwwebqeycq2xjwdg6m2t.streamlit.app/
  (오래 접속이 없으면 앱이 잠듭니다. "Yes, get this app back up!"을 누르고 1분쯤 기다리세요.)

## 내 PC에서 실행 (Docker)

[Docker Desktop](https://www.docker.com/products/docker-desktop/)을 켜고 터미널에서:

```bash
git clone https://github.com/univ237138-spec/recipe-graphrag-demo.git
cd recipe-graphrag-demo
docker build -t recipe-graphrag .
docker run --rm -p 8501:8501 recipe-graphrag
```

브라우저에서 http://localhost:8501 을 엽니다. 처음 build만 몇 분 걸립니다.
Git이 없으면 이 페이지의 **Code → Download ZIP**으로 받아 압축을 푼 폴더에서 `docker build`부터 하면 됩니다.

Docker 없이 Python(3.12 이상)으로 실행하려면:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 쓰는 법

- 기본은 **데모 모드(오프라인)** 입니다. 메모리에 올린 지식그래프로 답하므로 Neo4j·OpenAI 키가 필요 없습니다.
- 왼쪽 **예시 질문**을 누르면 챗봇이 결론과 근거 경로(재료 → 분류 → 알레르기 → 법령 조항)를 보여 줍니다.
- **📜 근거 그래프 · 법령** 탭에서 레시피별 그래프와 법적 근거 원문을 볼 수 있습니다.
- 자유 질문과 LLM 설명은 왼쪽 "LLM 세부 설정"에 본인 OpenAI 키를 넣었을 때만 동작합니다 (키는 저장하지 않음).

## 데이터 출처

- 레시피: 식품의약품안전처 식품안전나라 「조리식품의 레시피 DB」 (COOKRCP01) 오픈API
- 알레르기 법적 근거: 「식품 등의 표시·광고에 관한 법률 시행규칙」 [별표 2] (법제처 국가법령정보센터)

※ 포트폴리오용 데모이며 의료·영양 조언이 아닙니다. 레시피 원문 기준 판정이므로 실제 제품의 알레르기 표시를 꼭 확인하세요.
