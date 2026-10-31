"""레시피·알레르기 GraphRAG 화면 (Streamlit)

    streamlit run app.py

탭: 질문하기(챗봇) · 근거 그래프 · 법령
면접 시연이 끊기지 않게:
  - Neo4j가 안 되면 '데모 모드'(메모리 그래프, 3중 검증으로 Neo4j와 같은 결과)
  - OpenAI가 안 되면 예시 질문은 미리 정한 조건으로 동작한다
"""

import base64
import csv
import html
import json
import os
import re
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import answer  # noqa: E402
import graph_search  # noqa: E402
import llm  # noqa: E402
import load_neo4j  # noqa: E402
import local_graph  # noqa: E402
import query_parser  # noqa: E402
import recipe_graph  # noqa: E402
import rerank  # noqa: E402

PROVISIONS_PATH = ROOT / "data" / "regulation" / "provisions.json"
CATEGORIES_PATH = ROOT / "ontology" / "categories.csv"
DEMO, NEO4J = "데모 모드 (오프라인)", "Neo4j (AuraDB)"

# 예시 질문: 프로젝트의 장점별로 묶는다 (10/30). 모두 LLM 없이 미리 정한 조건으로 동작하고, 데모 모드 결과를 확인했다.
EXAMPLE_GROUPS = [
    ("🔍 숨은 알레르기 찾기", [
        {"key": "B2", "label": "🦐 갑각류 · 석류 보쌈김치 먹어도 돼?",
         "question": "갑각류 알레르기가 있어요. 석류 보쌈김치 먹어도 되나요?",
         "conditions": {"intent": "check_safety", "recipe_name": "석류 보쌈김치", "exclude_allergens": ["게", "새우"]}},
        {"key": "K1", "label": "🧈 우유 · 한우굴라쉬 먹어도 돼?",
         "question": "우유 알레르기인데 한우굴라쉬 먹어도 돼?",
         "conditions": {"intent": "check_safety", "recipe_name": "한우굴라쉬", "exclude_allergens": ["우유"]}},
        {"key": "L1", "label": "📋 부대된장찌개 알레르기 성분은 몇 개?",
         "question": "부대된장찌개에 알레르기 성분이 몇 개 있어?",
         "conditions": {"intent": "lookup", "recipe_name": "부대된장찌개"}},
        {"key": "B5", "label": "🥛 우유 · 고등어된장구이 먹어도 돼?",
         "question": "우유 알레르기인데 고등어된장구이 먹어도 돼?",
         "conditions": {"intent": "check_safety", "recipe_name": "고등어된장구이", "exclude_allergens": ["우유"]}},
    ]),
    ("🏷️ 이름에 속지 않기", [
        {"key": "T1", "label": "🥜 땅콩 · 땅콩호박 타르트 먹어도 돼?",
         "question": "땅콩 알레르기인데 땅콩호박 타르트 먹어도 돼?",
         "conditions": {"intent": "check_safety", "recipe_name": "땅콩호박 타르트", "exclude_allergens": ["땅콩"]}},
        {"key": "T2", "label": "🐙 오징어 · 문어핫바 먹어도 돼?",
         "question": "오징어 알레르기인데 문어핫바 먹어도 돼?",
         "conditions": {"intent": "check_safety", "recipe_name": "문어핫바", "exclude_allergens": ["오징어"]}},
    ]),
    ("⚖️ 확정과 주의 구분", [
        {"key": "E2", "label": "🌾 밀 · 부추 콩가루 찜은 왜 '주의'?",
         "question": "부추 콩가루 찜이 밀 알레르기에 '주의'로 표시되는 이유가 뭐야?",
         "conditions": {"intent": "explain", "recipe_name": "부추 콩가루 찜", "exclude_allergens": ["밀"]}},
        {"key": "W1", "label": "🍷 아황산 · 와인배숙 먹어도 돼?",
         "question": "아황산 알레르기인데 와인배숙 먹어도 돼?",
         "conditions": {"intent": "check_safety", "recipe_name": "와인배숙", "exclude_allergens": ["아황산류"]}},
    ]),
    ("📋 메뉴 전체에서 찾기 · 세기", [
        {"key": "N1", "label": "🫘 대두가 들어간 국·찌개는 몇 개?",
         "question": "대두가 들어간 국이나 찌개는 몇 개야?",
         "conditions": {"intent": "count", "include_allergens": ["대두"], "dish_type": "국&찌개"}},
        {"key": "A1", "label": "🥩 소고기가 들어간 국·찌개 모두",
         "question": "소고기가 들어간 국이나 찌개를 모두 알려줘",
         "conditions": {"intent": "list_all", "include_allergens": ["쇠고기"], "dish_type": "국&찌개"}},
    ]),
    ("🍽️ 조건에 맞게 추천", [
        {"key": "C2", "label": "🦀 갑각류 · 반찬 3개 추천",
         "question": "갑각류 알레르기가 있어. 반찬 3개 추천해줘",
         "conditions": {"intent": "recommend", "dish_type": "반찬", "exclude_allergens": ["게", "새우"], "limit": 3}},
        # 리랭킹 비교용: 조건은 위 C2와 똑같고 말만 다르다 (리랭킹이 꺼져 있으면 셋 다 같은 요리)
        {"key": "C3", "label": "🌶️ 갑각류 · 매콤한 반찬 3개 추천",
         "question": "갑각류 알레르기가 있는데, 매콤한 반찬 3개 추천해줘",
         "conditions": {"intent": "recommend", "dish_type": "반찬", "exclude_allergens": ["게", "새우"], "limit": 3}},
        {"key": "C4", "label": "🧒 갑각류 · 아이가 먹기 좋은 반찬 3개 추천",
         "question": "갑각류 알레르기 있는 아이가 먹기 좋은 반찬 3개 추천해줘",
         "conditions": {"intent": "recommend", "dish_type": "반찬", "exclude_allergens": ["게", "새우"], "limit": 3}},
        {"key": "R1", "label": "🍰 우유·달걀·밀·대두 · 후식 3개 추천",
         "question": "우유, 달걀, 밀, 대두 알레르기가 있어. 후식 3개 추천해줘",
         "conditions": {"intent": "recommend", "dish_type": "후식", "exclude_allergens": ["우유", "알류", "밀", "대두"], "limit": 3}},
    ]),
]
EXAMPLES = [example for _, group in EXAMPLE_GROUPS for example in group]
BASE_CONDITIONS = {"intent": "recommend", "recipe_name": "", "exclude_allergens": [], "include_categories": [],
                   "dish_type": "", "cooking_method": "", "max_kcal": None, "max_sodium_mg": None, "limit": 3}
VERDICT = {"unsafe": (st.error, "🚫 먹으면 안 됨"), "caution": (st.warning, "⚠️ 주의 필요 (제품에 따라 들어 있을 수 있음)"),
           "safe": (st.success, "✅ 먹어도 됨 (레시피 원문 기준)")}

# 10/30 프로젝트 엠블럼 (방패 + 그릇 + 지식그래프 경로): 브라우저 탭 아이콘, 제목, 챗봇 아이콘에 쓴다
EMBLEM = ROOT / "assets" / "emblem.png"
st.set_page_config(page_title="레시피·알레르기 GraphRAG", page_icon=str(EMBLEM), layout="wide")

# 채팅 말풍선 (10/30): 내 질문은 오른쪽 초록 말풍선, 챗봇 답은 왼쪽 흰 말풍선. 색은 .streamlit/config.toml 테마와 같다.
# Streamlit은 질문(기본 아바타)에 stChatMessageAvatarUser, 이미지 아바타(엠블럼 챗봇)는 이름표 없는 <img>로 그린다.
# 그래서 챗봇 답은 '질문이 아닌 메시지'로 고른다.
CHAT_CSS = """
<style>
[data-testid="stChatMessage"] { background: transparent; padding: 0.25rem 0; gap: 0.6rem; width: 100%; }
/* Streamlit 기본 글 상자의 아래 여백(-1rem) 때문에 말풍선 글자가 잘리지 않게 */
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] { margin-bottom: 0; }
/* 내 질문: 오른쪽 */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { flex-direction: row-reverse; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    flex: 0 1 auto; max-width: 70%; margin-left: auto; margin-right: 0; background: #2E7D5B; color: #FFFFFF;
    padding: 0.6rem 1rem; border-radius: 18px 18px 4px 18px; box-shadow: 0 1px 2px rgba(30, 42, 36, 0.12);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p {
    color: #FFFFFF; margin: 0;
}
/* 챗봇 답: 왼쪽 */
[data-testid="stChatMessage"]:not(:has([data-testid="stChatMessageAvatarUser"])) [data-testid="stChatMessageContent"] {
    flex: 0 1 auto; max-width: 88%; margin-left: 0; margin-right: auto;  /* 기본 margin:auto를 풀어 아이콘 옆에 붙임 */
    background: #FFFFFF; border: 1px solid #E1E6E3;
    padding: 0.75rem 1.1rem; border-radius: 18px 18px 18px 4px; box-shadow: 0 1px 2px rgba(30, 42, 36, 0.06);
}
/* 아이콘: 두 쪽 같은 크기, 챗봇은 연초록 원, 질문은 차분한 회녹색 (빨강은 '확정 알레르기' 색이라 피함) */
[data-testid="stChatMessage"] > :first-child { width: 2.25rem; height: 2.25rem; flex: 0 0 2.25rem; margin-top: 0.15rem; }
[data-testid="stChatMessage"]:not(:has([data-testid="stChatMessageAvatarUser"])) > :first-child {
    background: #DCEFE4; border-radius: 50%; padding: 0.3rem; box-sizing: border-box; object-fit: contain;
}
[data-testid="stChatMessageAvatarUser"] { background: #8A948C; border-radius: 50%; }
/* 간격: 말풍선 안 요소 사이와 메시지 사이를 좁힘 */
[data-testid="stChatMessageContent"] [data-testid="stVerticalBlock"] { gap: 0.6rem; }
.st-key-chatbox [data-testid="stVerticalBlock"] { gap: 0.5rem; }
.st-key-chatbox { gap: 0.5rem; }
</style>
"""
st.markdown(CHAT_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- 데이터

@st.cache_resource(show_spinner="지식그래프를 불러오는 중…")
def load_data():
    graph, recipes, mapping, mentions = recipe_graph.load_all()
    return {"graph": graph, "recipes": recipes, "mapping": mapping, "mentions": mentions,
            "runner": local_graph.LocalRunner(graph, recipes, mapping, mentions),
            "reranker": rerank.load(recipes=recipes),  # 레시피 임베딩 파일이 없으면 None → 추천은 번호순
            "expected": recipe_graph.expected_allergens(mentions, mapping, graph)}


DATA = load_data()
RECIPES = DATA["recipes"]


# ---------------------------------------------------------------- 사이드바

def use_example(example):
    """예시 질문 버튼: 대화창에 바로 질문을 올린다 (다음 실행에서 처리)."""
    st.session_state.pending = {"question": example["question"], "example": example["key"]}


with st.sidebar:
    # 예시 질문을 맨 위에 (시연할 때 가장 먼저 누르는 곳)
    st.subheader("💬 예시 질문")
    for title, group in EXAMPLE_GROUPS:
        st.caption(f"**{title}**")
        for example in group:
            st.button(example["label"], key=f"ex_{example['key']}", on_click=use_example, args=(example,),
                      width="stretch")

    st.divider()
    st.subheader("⚙️ 설정")
    # 그래프: 토글 하나 (끄면 Neo4j, 켜면 메모리 그래프)
    demo_on = st.toggle("데모 모드 (오프라인)", value=True, key="demo",  # 기본값: 데모 모드 (비밀번호 없이 바로 시연)
                        help="메모리에 올린 같은 그래프로 답합니다 (Neo4j 결과와 1,156개 레시피 모두 일치 확인). "
                             "끄면 Neo4j (AuraDB)에 연결합니다.")
    mode = DEMO if demo_on else NEO4J
    if mode == NEO4J:
        settings = load_neo4j.connection_settings()
        if st.session_state.get("driver"):
            st.caption("🟢 Neo4j 연결됨")
        else:
            with st.expander("🔌 Neo4j 연결", expanded=True):
                st.caption(f"주소: `{settings['uri'] or '없음'}`")
                password = st.text_input("Neo4j 비밀번호", type="password", value=os.environ.get("NEO4J_PASSWORD", ""))
                if st.button("연결", disabled=not password):
                    try:
                        st.session_state.driver, st.session_state.database = load_neo4j.open_driver(password, settings)
                        st.rerun()
                    except load_neo4j.ConnectionFailed as error:
                        st.error(str(error))

    # LLM: 토글만 보이고, 키·모델·effort는 접어 둔다 (키 입력이 토글보다 먼저 처리돼야 해서 자리만 먼저 잡음)
    llm_slot = st.empty()
    with st.expander("🔑 LLM 세부 설정 (API 키 · 모델)"):
        if not os.environ.get("OPENAI_API_KEY"):
            key = st.text_input("OpenAI API 키", type="password", help="이 실행 중에만 쓰고 저장하지 않습니다.")
            if key:
                os.environ["OPENAI_API_KEY"] = key.strip()
        model = st.text_input("모델", value=os.environ.get("OPENAI_MODEL") or "gpt-5-mini")
        effort = st.selectbox("reasoning effort", ["low", "minimal", "medium", "high", "지정 안 함"])
        effort = None if effort == "지정 안 함" else effort
    has_llm = bool(os.environ.get("OPENAI_API_KEY"))
    with llm_slot.container():
        use_llm = st.toggle("LLM 사용 (질문 해석·설명)", value=has_llm, disabled=not has_llm, key="use_llm",
                            help=None if has_llm else "OpenAI 키가 없으면 예시 질문만 동작합니다 (LLM 설명 없음).")
        if not has_llm:
            st.caption("키 없음 → 예시 질문만 동작")
    has_reranker = DATA["reranker"] is not None
    use_rerank = st.toggle("리랭킹 (추천 순서: 질문 유사도 + 다양성)", value=has_reranker, disabled=not has_reranker,
                           key="use_rerank",
                           help="추천 질문에서, 그래프가 거른 안전한 후보 안에서 순서만 다시 정합니다. "
                                "끄면 경고 없는 요리 먼저, 그다음 레시피 번호순. 안전 판단은 켜든 끄든 그래프가 합니다.")


# ---------------------------------------------------------------- 질문 처리

def find_example(key, question):
    return next((e for e in EXAMPLES if e["key"] == key and e["question"] == question), None)


def run_graphrag(run, question, example):
    """질문 해석 → 그래프 조회 → 안전 검사 → 설명. 단계마다 진행 상황을 보여 준다."""
    with st.status("질문을 처리하는 중…", expanded=False) as status:
        if use_llm:
            status.write("① 질문 해석 (LLM → 조건 JSON, 온톨로지 값만 허용)")
            cond = query_parser.parse(question, DATA["graph"], model, effort)
        elif example:
            status.write("① 질문 해석 (예시 질문: 미리 정한 조건 사용, LLM 없음)")
            cond = query_parser.validate({**BASE_CONDITIONS, **example["conditions"]}, DATA["graph"])
        else:
            raise llm.LLMError("LLM 없이는 예시 질문만 처리할 수 있습니다. 왼쪽에서 예시 질문을 고르거나 OpenAI 키를 넣으세요.")
        status.write("② 그래프 조회 (Cypher: 확정 알레르기 경로가 있으면 제외)")
        ranker = None
        if use_rerank and DATA["reranker"] and cond["intent"] == "recommend":
            # 취향 확장·임베딩을 못 만들면 None → 번호순 (예시 질문은 미리 저장해 둬서 키 없이도 된다)
            ranker = DATA["reranker"].for_question(question, cond["exclude_allergens"], model)
        result = graph_search.search(run, cond, RECIPES, ranker)
        if result.get("reranked"):
            status.write("②+ 리랭킹 (안전한 후보 안에서 질문 유사도 + 다양성으로 순서 정하기)")
        status.write("③ 안전 검사 (추천 요리를 별도 쿼리로 다시 확인)")
        notes = answer.safety_check(run, result) if result.get("not_found") is None else []
        paragraphs, error = None, None
        if use_llm and result.get("not_found") is None:
            status.write("④ 설명 (LLM, 근거 ID가 없는 문단은 거부)")
            evidence = answer.build_evidence(result)
            try:
                reply = llm.call(answer.build_request(question, result, evidence, model, effort))
                paragraphs = answer.validate_paragraphs(reply, evidence)
            except llm.LLMError as exc:
                error = str(exc)
        status.update(label="완료: ① 질문 해석 → ② 그래프 조회" + (" → 리랭킹" if result.get("reranked") else "")
                            + " → ③ 안전 검사" + (" → ④ 설명" if paragraphs else ""),
                      state="complete")
    return {"conditions": cond, "result": result, "paragraphs": paragraphs, "notes": notes, "llm_error": error}


# ---------------------------------------------------------------- 출력

@st.cache_data
def provision_tooltip():
    """법적 근거에 마우스를 올리면 보일 조항 원문 ([별표 2] PDF에서 추출한 data/regulation/provisions.json)."""
    if not PROVISIONS_PATH.exists():
        return ""
    data = json.loads(PROVISIONS_PATH.read_text(encoding="utf-8"))
    reg = data.get("regulation", {})
    lines = [f"{reg.get('법령명', '')} {reg.get('별표', '')}", f"(개정 {reg.get('별표_개정일', '')} · 시행 {reg.get('시행일자', '')})", ""]
    for prov in data.get("provisions", []):
        label = prov["id"].replace("별표2-", "").replace("-", ".")
        if prov.get("allergens"):  # 가목: PDF 줄바꿈("복숭\n아")이 섞이지 않게 추출한 항목으로 잇는다
            body = [", ".join(a["term"] for a in prov["allergens"])]
        else:
            body = prov.get("items") or [" ".join(prov["text"].split())]
        lines += [f"[{label}] {prov['title']}", *body, ""]
    return "\n".join(lines).strip()


def with_tooltip(label, tip):
    """마우스를 올리면 tip이 보이는 글자 (브라우저 기본 툴팁)."""
    if not tip:
        return html.escape(label)
    tip = html.escape(tip, quote=True).replace("\n", "&#10;")
    return (f'<span title="{tip}" style="text-decoration: underline dotted; text-underline-offset: 3px; '
            f'cursor: help;">{html.escape(label)}</span>')


def show_paths(paths):
    for p in paths:
        icon = "🔴" if p["certainty"] == "확정" else "🟠"
        legal = answer.legal_text(p)
        if legal:  # 10/30: 법적 근거에 마우스를 올리면 조항 원문이 보인다
            legal = f" · 📜 법적 근거: {with_tooltip(legal, provision_tooltip())}" + (
                f" ({html.escape(p['legal_condition'])})" if p.get("legal_condition") else "")
        st.markdown(f"{icon} {html.escape(chain_text(p['path'][1:]))} &nbsp; `{p['certainty']}` · "
                    f"원문: {html.escape(p['raw'])}{legal}", unsafe_allow_html=True)


def show_graphrag(output):
    result = output["result"]
    if result.get("not_found") is not None:
        st.warning(f"'{result['not_found']}' 요리를 찾지 못했습니다.")
        if result["suggestions"]:
            st.write("비슷한 요리: " + ", ".join(result["suggestions"]))
        return
    exclude = result["conditions"]["exclude_allergens"]
    if result["intent"] in ("count", "list_all"):
        st.markdown(f"{answer.condition_text(result['conditions'])}: **{result['count']}개** (조건에 맞는 요리 {result['scope']}개 중)")
        for n, item in enumerate(result["items"], 1):
            reason = f" — {answer.path_text(item['paths'][0])}" if item["paths"] else ""
            st.caption(f"{n}. {item['name']} · {answer.recipe_summary(item)}{reason}")
        if result["caution_only"]:
            st.caption(f"⚠️ 제품에 따라 들어 있을 수 있는(주의) 요리 {len(result['caution_only'])}개는 세지 않았습니다: "
                       + ", ".join(c["name"] for c in result["caution_only"][:10]))
    elif result["intent"] == "recommend":
        st.markdown(f"조건에 맞는 요리 **{result['matched']}개** 중 제외 알레르기(확정)가 없는 요리 **{result['allowed']}개**")
        st.caption(f"↕️ 순서: {answer.order_text(result)}")
        for n, item in enumerate(result["recommended"], 1):
            with st.container(border=True):
                score = f" · 질문 유사도 `{item['relevance']:.2f}`" if "relevance" in item else ""
                if item.get("matched_terms"):  # 하이브리드 리랭킹의 근거: 확장 말 중 이 레시피에 실제로 있는 것
                    score += " · 레시피에 있는 말: " + ", ".join(item["matched_terms"])
                st.markdown(f"**{n}. {item['name']}** · {answer.recipe_summary(item)}{score}")
                for path in item["paths"]:
                    st.caption(f"⚠️ 주의: {answer.path_text(path)}")
                for unknown in item["unknown_ingredients"]:
                    st.caption(f"⚠️ 확인 필요: '{unknown['raw']}'의 구성을 알 수 없음")
        if not result["recommended"]:
            st.info("조건에 맞는 요리가 없습니다.")
        if result["excluded_examples"]:
            with st.expander(f"제외한 요리 예시 (모두 {result['excluded_count']}개)"):
                for item in result["excluded_examples"]:
                    st.markdown(f"✗ **{item['name']}**")
                    show_paths(item["paths"][:2])
    else:
        recipe = result["recipe"]
        st.markdown(f"#### {recipe['name']}")
        st.caption(f"{answer.recipe_summary(recipe)} · 조리법: {recipe.get('method')}")
        if result.get("verdict"):
            box, text = VERDICT[result["verdict"]]
            box(f"{', '.join(exclude)} 알레르기 → {text}")
        summary = answer.allergen_summary(result["paths"])
        icons = {"확정": "🔴", "주의": "🟠"}
        parts = [f"{icons[k]} **{k} {len(v)}종**: {' · '.join(v)}" for k, v in summary.items() if v]
        st.markdown("**이 요리의 알레르기** (코드로 센 값) &nbsp; "
                    + (" &nbsp;&nbsp; ".join(parts) or "알레르기 성분이 들어간 재료가 없습니다"))
        shown =[p for p in result["paths"] if not exclude or p["allergen"] in exclude]
        st.markdown("**근거 경로** (재료 → 분류 → 알레르기 → 법적 근거)")
        if shown:
            show_paths(shown)
        else:
            st.caption(f"재료에서 {', '.join(exclude)} 성분을 찾지 못했습니다." if exclude else "알레르기 성분이 없습니다.")
        for unknown in result["unknown_ingredients"]:
            st.caption(f"⚠️ 확인 필요: '{unknown['raw']}'의 구성을 알 수 없음")
        if result["intent"] == "lookup":
            st.markdown("**재료:** " + ", ".join(i["raw"] for i in result["ingredients"] if i.get("raw")))
    if output["paragraphs"]:
        st.markdown("**설명** (LLM · 근거 ID 확인됨)")
        for p in output["paragraphs"]:
            st.markdown(f"{p['text']} " + " ".join(f"`{i}`" for i in p["evidence_ids"]))
    if output["llm_error"]:
        st.caption(f"설명 생략: {output['llm_error']}")
    for note in output["notes"]:
        st.error(f"안전 검사: {note}")


# ---------------------------------------------------------------- 대화

def josa(word, with_final, without_final):
    """조사를 붙인다: 받침이 있으면 with_final(은·이·과·으로·이라서), 없으면 without_final(는·가·와·로·라서).
    '으로'는 ㄹ 받침 뒤에서 '로'가 된다. 끝 글자가 한글이 아니면 받침 없음으로 본다."""
    last = word[-1] if word else ""
    final = (ord(last) - 0xAC00) % 28 if "가" <= last <= "힣" else 0
    if with_final == "으로" and final == 8:  # ㄹ 받침
        return word + "로"
    return word + (with_final if final else without_final)


def chain_text(path):
    """경로 표시: 재료 이름과 분류 이름이 같으면 한 번만 (버터 → 버터 → 유제품 → 우유 → 버터 → 유제품 → 우유)."""
    chain = [name for i, name in enumerate(path) if i == 0 or name != path[i - 1]]
    return " → ".join(chain)


@st.cache_data
def category_table():
    with open(CATEGORIES_PATH, encoding="utf-8-sig") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def trap_notes(recipe_id, allergens):
    """'먹어도 됨'일 때 헷갈리기 쉬운 재료를 온톨로지 분류로 설명한다 (코드가 만든다).
    - 이름 함정: 재료 이름에 알레르기 단어가 있지만 분류상 그 알레르기와 이어지지 않음 (땅콩호박 → 채소·버섯)
    - 분류 함정: 그 알레르기 분류와 같은 상위 분류의 다른 분류 (문어 → 연체류 > 기타 연체류, 오징어는 오징어류)"""
    cats, runner = category_table(), DATA["runner"]
    notes = []
    for allergen in allergens:
        own = [c for c in cats.values() if f"{allergen}(" in c["알레르기(물려받은 것 포함)"]]
        parents = {c["상위 분류"] for c in own}
        for m in runner.by_recipe.get(recipe_id, []):
            row = cats.get(runner.mapping.get(m["ingredient"], {}).get("category_id"))
            if not row or allergen in row["알레르기(물려받은 것 포함)"]:
                continue
            name = m["ingredient"]
            name_trap = allergen in name
            if name_trap:
                note = (f"이름에 ‘{allergen}’{josa(allergen, '이', '가')[len(allergen):]} 들어 있지만, "
                        f"{josa(name, '은', '는')} ‘{row['분류']}’{josa(row['분류'], '으로', '로')[len(row['분류']):]} "
                        f"분류돼 {josa(allergen, '과', '와')} 이어지지 않아요.")
            elif len(parents) == 1 and row["상위 분류"] in parents:  # 알레르기 분류가 한 곳에만 있을 때만 (오탐 방지)
                note = (f"{josa(name, '은', '는')} {josa(allergen, '과', '와')} 같은 ‘{row['상위 분류']}’지만 "
                        f"‘{row['분류']}’{josa(row['분류'], '이라서', '라서')[len(row['분류']):]}, "
                        f"법령의 {josa(allergen, '과', '와')} 이어지지 않아요.")
            else:
                continue
            notes.append((0 if name_trap else 1, note))  # 이름 함정을 먼저
    return [note for _, note in sorted(notes)[:1]]  # 가장 헷갈리는 재료 하나만


def chat_lead(output):
    """챗봇의 첫 문장: 그래프가 낸 결과를 대화체로 바꾼다 (LLM 없이 코드가 만든다)."""
    r = output["result"]
    if r.get("not_found") is not None:
        return f"'{r['not_found']}' 요리를 찾지 못했어요." + (
            f" 혹시 {', '.join(r['suggestions'][:3])} 중 하나인가요?" if r.get("suggestions") else "")
    cond = r["conditions"]
    exclude = ", ".join(cond["exclude_allergens"])
    if r["intent"] in ("count", "list_all"):
        include = "·".join(cond.get("include_allergens") or [])
        dish = (cond.get("dish_type") or "").replace("&", "·")
        what = f"{josa(include, '이', '가')} 확정으로 들어간 {dish} 요리" if include else f"{answer.condition_text(cond)} 요리"
        scope = f"{dish} {r['scope']}개 중" if dish else f"조건에 맞는 요리 {r['scope']}개 중"
        tail = " 근거 보기에서 전체 목록을 볼 수 있어요." if r["intent"] == "list_all" else ""
        return f"{what}는 **{r['count']}개**예요 ({scope}).{tail}"
    if r["intent"] == "recommend":
        n = len(r["recommended"])
        if not n:
            return "조건에 맞는 요리를 찾지 못했어요."
        head = f"{exclude} 알레르기가 확정으로 든 요리는 빼고, " if exclude else ""
        order = ""
        if r.get("reranked"):
            order = (f" 안전한 요리 {r['allowed']}개 중 ‘{r['preference']}’에 가까운 요리부터, 서로 겹치지 않게 골랐어요."
                     if r.get("preference") else f" 안전한 요리 {r['allowed']}개 중 서로 겹치지 않게 골랐어요.")
        return f"{head}조건에 맞는 요리 **{n}개**를 추천해 드릴게요.{order} 추천 전에 그래프로 한 번 더 확인했어요."
    recipe = r["recipe"]["name"]
    if r.get("verdict"):
        shown = [p for p in r["paths"] if p["allergen"] in cond["exclude_allergens"]]
        # 굵게(**…**) 뒤에 바로 '에서'가 붙으면 마크다운이 굵게로 읽지 않아, 따옴표로 감싼다
        why = (f" 재료 ‘{shown[0]['raw']}’에서 {chain_text(shown[0]['path'][1:])} 길이 나와요."
               if shown else "")
        return {"unsafe": f"{exclude} 알레르기가 있다면 **{recipe}**, 먹으면 안 돼요 🚫{why}",
                "caution": f"{exclude} 알레르기가 있다면 **{recipe}**, 드실 때 주의가 필요해요 ⚠️{why} "
                           "제품에 따라 들어 있을 수 있어요.",
                "safe": f"{exclude} 알레르기가 있어도 **{recipe}**, 먹어도 돼요 ✅ "
                        "레시피 원문 재료에서 해당 성분을 찾지 못했어요."
                        + "".join(f" {n}" for n in trap_notes(r["recipe"]["id"], cond["exclude_allergens"]))}[r["verdict"]]
    summary = answer.allergen_summary(r["paths"])
    parts = [f"{k} **{len(v)}종**({'·'.join(v)})" for k, v in summary.items() if v]
    if not parts:
        return f"**{recipe}**에는 알레르기 성분이 들어간 재료가 없어요 (레시피 원문 기준)."
    return f"**{recipe}**의 알레르기 성분은 {', '.join(parts)}이에요."


def show_reply(reply, latest):
    """챗봇 답: 결론 한두 문장 → 근거(판정·경로·법적 근거). 이전 답의 근거는 접어 둔다."""
    if reply.get("error"):
        st.error(reply["error"])
        return
    output = reply["output"]
    st.markdown(chat_lead(output))
    with st.expander("🔎 근거 보기 (그래프가 찾은 길)", expanded=latest):
        st.caption(f"그래프: {reply['mode']}")
        with st.popover("① 질문 해석 결과 (조건)"):
            st.json(output["conditions"])
        show_graphrag(output)


def show_message(message, latest):
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar=str(EMBLEM)):
            show_reply(message, latest)


# ---------------------------------------------------------------- 근거 그래프

@st.cache_data
def provisions():
    return json.loads(PROVISIONS_PATH.read_text(encoding="utf-8")) if PROVISIONS_PATH.exists() else {}


def show_law_text(paths):
    """그래프 아래 법적 근거 원문: [별표 2] Ⅰ.1.가(알레르기 유발물질)·나(표시 대상).
    이 레시피에서 찾은 알레르기는 원문 목록에서 색으로 표시한다 (확정 = 빨강, 주의만 = 주황)."""
    data = provisions()
    if not data or not any(p.get("provision") for p in paths):
        return
    level = answer.allergen_summary(paths)
    definite, caution = set(level["확정"]), set(level["주의"])
    reg = data["regulation"]
    with st.container(border=True):
        st.markdown("#### 📜 법적 근거 원문")
        st.caption(f"{reg['법령명']} {reg['별표']} · 개정 {reg['별표_개정일']} · 시행 {reg['시행일자']} · "
                   f"[법제처 원문 파일]({reg['파일_출처']})")
        for prov in data["provisions"]:
            label = prov["id"].replace("별표2-", "").replace("-", ".")
            st.markdown(f"**[{label}] {prov['title']}**")
            if prov.get("allergens"):
                terms = [f":red[**{a['term']}**]" if a["name"] in definite else
                         f":orange[**{a['term']}**]" if a["name"] in caution else a["term"] for a in prov["allergens"]]
                st.markdown(", ".join(terms))
                if definite or caution:
                    st.caption("색으로 표시한 것이 이 레시피에서 찾은 알레르기입니다 (빨강 = 확정, 주황 = 주의).")
            else:
                # '1)'로 시작하면 마크다운이 번호 목록으로 바꾸므로 괄호를 이스케이프하고, 줄바꿈으로만 나눈다
                st.markdown("  \n".join(re.sub(r"^(\d+)\)", r"\1\\)", item) for item in prov.get("items", [])))
                st.caption("2)·3)이 원재료의 원재료까지 표시 대상으로 정합니다. "
                           "그래프가 재료 → 분류 → 알레르기로 끝까지 따라가는 법적 근거입니다.")



def quote(text):
    return '"' + str(text).replace('"', '\\"') + '"'


# 근거 그래프 색 (.streamlit/config.toml 테마와 같은 색): 초록 = 레시피, 빨강 = 확정, 주황 = 주의
COLOR = {"primary": "#2E7D5B", "recipe": "#DCEFE4", "ingredient": "#F0F2EC", "ingredient_line": "#C9CFC4",
         "category": "#8A948C", "text": "#1E2A24", "edge": "#B4BBB2",
         "definite": "#C62828", "definite_fill": "#FBE3E1", "caution": "#D97706", "caution_fill": "#FDF0D3"}


def recipe_dot(recipe_id, focus=(), show_all=False):
    runner = DATA["runner"]
    paths = [p for p in runner.paths(recipe_id) if not focus or p["allergen"] in focus]
    lines = ["digraph G {", 'rankdir=LR; nodesep=0.25; ranksep=0.5; bgcolor="transparent";',
             f'node [fontsize=11, fontname="Malgun Gothic", fontcolor="{COLOR["text"]}", penwidth=1.2]; '
             f'edge [fontsize=9, color="{COLOR["edge"]}", fontcolor="{COLOR["category"]}", fontname="Malgun Gothic"];',
             f'r [label={quote(RECIPES[recipe_id]["name"])}, shape=box, style="rounded,filled,bold", '
             f'fillcolor="{COLOR["recipe"]}", color="{COLOR["primary"]}"];']
    nodes, edges = set(), set()

    def node(kind, name, **attrs):
        key = f"{kind}:{name}"
        if key not in nodes:
            nodes.add(key)
            line = attrs.get("line", COLOR["definite"])
            style = {"i": f'shape=ellipse, style=filled, fillcolor="{COLOR["ingredient"]}", color="{COLOR["ingredient_line"]}"',
                     "c": f'shape=box, style=rounded, color="{COLOR["category"]}"',
                     "a": f'shape=octagon, style=filled, fillcolor="{attrs.get("fill", COLOR["definite_fill"])}", '
                          f'color="{line}", fontcolor="{line}"',
                     "p": f'shape=note, fontsize=9, color="{COLOR["category"]}", fontcolor="{COLOR["category"]}", '
                          f'tooltip={quote(provision_tooltip()).replace(chr(10), "&#10;")}'}[kind]
            lines.append(f"{quote(key)} [label={quote(name)}, {style}];")
        return quote(key)

    def edge(a, b, label="", style=""):
        if (a, b, label) not in edges:
            edges.add((a, b, label))
            lines.append(f"{a} -> {b} [label={quote(label)}{', ' + style if style else ''}];")

    definite = {p["allergen"] for p in paths if p["certainty"] == "확정"}
    for p in paths:
        chain = p["path"]  # [레시피, 재료, 분류…, 알레르기]
        ingredient = node("i", chain[1])
        edge("r", ingredient)
        previous = ingredient
        for category in chain[2:-1]:
            current = node("c", category)
            edge(previous, current, "isA")
            previous = current
        sure = chain[-1] in definite
        allergen = node("a", chain[-1], fill=COLOR["definite_fill" if sure else "caution_fill"],
                        line=COLOR["definite" if sure else "caution"])
        red = p["certainty"] == "확정"
        edge(previous, allergen, p["certainty"],
             f'color="{COLOR["definite"]}", fontcolor="{COLOR["definite"]}", penwidth=1.6' if red
             else f'color="{COLOR["caution"]}", fontcolor="{COLOR["caution"]}", style=dashed')
        if p.get("provision"):  # 법적 근거 조항 (모든 알레르기가 같은 조항을 가리킨다)
            edge(allergen, node("p", answer.legal_text(p).replace("식품표시광고법 시행규칙 ", "시행규칙\\n")),
                 "법적 근거", 'style=dotted, arrowhead=none')
    if show_all:
        for m in runner.by_recipe.get(recipe_id, []):
            ingredient = node("i", m["ingredient"])
            edge("r", ingredient)
            edge(ingredient, node("c", runner.label[runner.mapping[m["ingredient"]]["category_id"]]), "isA")
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------- 화면

st.markdown(  # 제목 앞 엠블럼 (st.title은 이미지를 못 넣어 HTML로)
    '<div style="display:flex;align-items:center;gap:0.75rem;margin:0.4rem 0 0.2rem">'
    f'<img src="data:image/png;base64,{base64.b64encode(EMBLEM.read_bytes()).decode()}" '
    'alt="엠블럼" style="height:3.1rem">'
    '<h1 style="margin:0;padding:0">레시피·알레르기 GraphRAG</h1></div>', unsafe_allow_html=True)
st.caption("식약처 레시피 1,156개 · OWL 온톨로지(분류 74개, 알레르기 19개) · Neo4j 지식그래프 · "
           "**LLM은 말을 다루고, 안전 판단은 그래프가 한다**")

tab_ask, tab_graph = st.tabs(["💬 질문하기", "📜 근거 그래프 · 법령"])  # 10/30 평가 결과 탭은 뺌 (평가는 eval/ 보고서로)

with tab_ask:
    # 10/30 챗봇 형태: 예시 질문 버튼이나 아래 입력창으로 물으면 대화로 답한다
    _, top_right = st.columns([3, 1])  # 10/30 '벡터 RAG와 나란히 비교'는 뺌 (비교는 eval/ 평가 보고서로)
    if top_right.button("대화 지우기", key="clear_chat", width="stretch"):
        st.session_state.messages = []
        st.session_state.pop("last", None)
    messages = st.session_state.setdefault("messages", [])

    chat_box = st.container(key="chatbox")  # 대화는 입력창 위에 쌓인다 (key는 CSS 간격용 .st-key-chatbox)
    prompt = st.chat_input("예: 우유랑 새우 알레르기가 있어. 반찬 3개 추천해줘", key="chat")
    pending = st.session_state.pop("pending", None)
    if prompt:
        pending = {"question": prompt, "example": None}

    with chat_box:
        if not messages and not pending:
            with st.chat_message("assistant", avatar=str(EMBLEM)):
                st.markdown("안녕하세요! 알레르기와 먹고 싶은 요리를 말씀해 주세요. "
                            "**왼쪽 예시 질문**을 누르면 바로 시작할 수 있어요.")
        for n, message in enumerate(messages):
            show_message(message, latest=(n == len(messages) - 1 and not pending))
        if pending:
            user = {"role": "user", "content": pending["question"]}
            with st.chat_message("user"):
                st.markdown(user["content"])
            reply = {"role": "assistant", "mode": mode}
            with st.chat_message("assistant", avatar=str(EMBLEM)):
                example = find_example(pending["example"], pending["question"])
                try:
                    if mode == NEO4J:
                        if not st.session_state.get("driver"):
                            raise load_neo4j.ConnectionFailed("왼쪽에서 Neo4j에 연결하거나 데모 모드를 켜세요.")
                        with st.session_state.driver.session(database=st.session_state.database) as session:
                            reply["output"] = run_graphrag(graph_search.session_runner(session), pending["question"], example)
                    else:
                        reply["output"] = run_graphrag(DATA["runner"], pending["question"], example)
                    st.session_state.last = {"question": pending["question"], **reply}
                except (llm.LLMError, load_neo4j.ConnectionFailed) as error:
                    reply["error"] = str(error)
                show_reply(reply, latest=True)
            messages += [user, reply]
    st.caption(answer.DISCLAIMER)

with tab_graph:
    st.markdown("레시피를 고르면 **레시피 → 재료 → 분류 → 알레르기 → 법령 조항** 경로를 그립니다. "
                "🔴 실선 = 확정, 🟠 점선 = 주의. 분류는 OWL 온톨로지(`rdfs:subClassOf`)를 따라 올라가고, "
                "알레르기의 법적 근거는 식품표시광고법 시행규칙 [별표 2] PDF에서 추출했습니다.")
    names = sorted(RECIPES.values(), key=lambda r: r["name"])
    last = st.session_state.get("last")
    default_id = (last or {}).get("output", {}).get("result", {}).get("recipe", {}).get("id") or "159"
    index = next((i for i, r in enumerate(names) if r["id"] == default_id), 0)
    chosen = st.selectbox("레시피", names, index=index, format_func=lambda r: f"{r['name']} ({r['id']})")
    available = sorted({p["allergen"] for p in DATA["runner"].paths(chosen["id"])})
    c1, c2 = st.columns([3, 1])
    focus = c1.multiselect("알레르기만 보기", available, placeholder="알레르기를 고르세요 (비우면 전부)")
    show_all = c2.toggle("모든 재료 보기")
    if available or show_all:
        st.graphviz_chart(recipe_dot(chosen["id"], focus, show_all), width="stretch")
    else:
        st.info("이 레시피에는 알레르기 성분이 없습니다. '모든 재료 보기'를 켜면 재료와 분류를 볼 수 있습니다.")
    show_law_text([p for p in DATA["runner"].paths(chosen["id"]) if not focus or p["allergen"] in focus])
    with st.expander("같은 경로를 Neo4j에서 보는 Cypher"):
        st.code(f"MATCH path = (r:Recipe {{id: '{chosen['id']}'}})-[:HAS_INGREDIENT]->()-[:IS_A]->()"
                "-[:IS_A*0..]->()-[:CONTAINS]->(a:Allergen)\n      -[:DEFINED_IN]->(:Provision)-[:PART_OF]->(:Regulation)\n"
                "RETURN path", language="cypher")
