"""검색 조건 → Neo4j 조회 → 근거 경로가 붙은 결과.

어떤 요리가 안전한지는 여기(Cypher)에서 결정한다. LLM은 결정에 관여하지 않는다.
    - 확정 알레르기가 있는 요리는 추천 후보에서 뺀다.
    - 주의 알레르기·구성 불명 재료가 있는 요리는 경고를 붙이고 뒤로 보낸다.

조회 함수는 run(query, **params) -> [dict] 를 받는다 (테스트에서 가짜 run을 넣을 수 있게).
"""

import difflib
import re

PATH_TO_ALLERGEN = "(r)-[h:HAS_INGREDIENT]->(i:Ingredient)-[:IS_A]->(:Category)-[:IS_A*0..]->(c:Category)-[x:CONTAINS]->(a:Allergen)"

# 근거 경로 + 알레르기의 법적 근거 조항 (판정에는 쓰지 않고 근거로만 붙인다)
RECIPE_PATHS = f"""
MATCH path = (r:Recipe {{id: $id}})-[h:HAS_INGREDIENT]->(i:Ingredient)-[:IS_A]->(:Category)
             -[:IS_A*0..]->(c:Category)-[x:CONTAINS]->(a:Allergen)
OPTIONAL MATCH (a)-[:DEFINED_IN]->(p:Provision)
RETURN a.name AS allergen, x.certainty AS certainty, h.raw AS raw, h.section AS section,
       [n IN nodes(path) | coalesce(n.name, n.id)] AS path, p.id AS provision,
       CASE WHEN a.legalCondition = '' THEN null ELSE a.legalCondition END AS legal_condition
ORDER BY h.position
"""

RECIPE_UNKNOWN = """
MATCH (r:Recipe {id: $id})-[h:HAS_INGREDIENT]->(i:Ingredient)-[:IS_A]->(:Category)
      -[:IS_A*0..]->(:Category {requiresConfirmation: true})
RETURN DISTINCT i.name AS ingredient, h.raw AS raw
"""

RECIPE_DETAIL = """
MATCH (r:Recipe {id: $id})
OPTIONAL MATCH (r)-[h:HAS_INGREDIENT]->(i:Ingredient)
WITH r, h, i ORDER BY h.position
RETURN r {.id, .name, .dishType, .method, .kcal, .sodium_mg, .protein_g, .fat_g, .carbohydrate_g, .tip} AS recipe,
       collect({ingredient: i.name, raw: h.raw, section: h.section}) AS ingredients
"""

# 분류 조건으로 고른 요리의 근거: 재료 → 분류 → … → 물어본 분류
RECIPE_CATEGORY_PATHS = """
MATCH path = (r:Recipe {id: $id})-[h:HAS_INGREDIENT]->(i:Ingredient)-[:IS_A]->(:Category)-[:IS_A*0..]->(c:Category)
WHERE c.id IN $categories
RETURN c.id AS category, h.raw AS raw, h.section AS section, [n IN nodes(path) | coalesce(n.name, n.id)] AS path
ORDER BY h.position
"""

# 조건(요리종류·조리법·영양·포함 분류)에 맞는 요리 + 제외 알레르기가 확정으로 들어 있는지
CANDIDATES = f"""
MATCH (r:Recipe)
WHERE ($dish IS NULL OR r.dishType = $dish)
  AND ($method IS NULL OR r.method = $method)
  AND ($max_kcal IS NULL OR r.kcal <= $max_kcal)
  AND ($max_sodium IS NULL OR r.sodium_mg <= $max_sodium)
  AND ALL(cat IN $include WHERE EXISTS {{
        MATCH (r)-[:HAS_INGREDIENT]->(:Ingredient)-[:IS_A]->(:Category)-[:IS_A*0..]->(:Category {{id: cat}}) }})
OPTIONAL MATCH {PATH_TO_ALLERGEN}
WHERE a.name IN $exclude
WITH r, collect(DISTINCT CASE WHEN x.certainty = '확정' THEN a.name END) AS definite,
        collect(DISTINCT CASE WHEN x.certainty = '주의' THEN a.name END) AS caution
OPTIONAL MATCH (r)-[:HAS_INGREDIENT]->(u:Ingredient)-[:IS_A]->(:Category)-[:IS_A*0..]->(:Category {{requiresConfirmation: true}})
RETURN r.id AS id, r.name AS name, r.dishType AS dishType, r.kcal AS kcal, r.sodium_mg AS sodium_mg,
       definite, caution, collect(DISTINCT u.name) AS unknown
ORDER BY toInteger(r.id)
"""


# ---------------------------------------------------------------- 요리 이름 찾기 (Neo4j 없이)

def normalize(text):
    return re.sub(r"\s+", "", text or "")


def resolve_recipe(name, recipes):
    """질문 속 요리 이름 → (레시피 id 또는 None, 비슷한 후보 이름들)."""
    target = normalize(name)
    names = {normalize(r["name"]): r["id"] for r in recipes.values()}
    if target in names:
        return names[target], []
    contains = [key for key in names if target and (target in key or key in target)]
    if len(contains) == 1:
        return names[contains[0]], []
    close = difflib.get_close_matches(target, list(names), n=3, cutoff=0.6)
    if close and not contains:
        return names[close[0]], []
    suggestions = [recipes[names[k]]["name"] for k in (contains or close)[:5]]
    return None, suggestions


# ---------------------------------------------------------------- 조회

def recipe_evidence(run, recipe_id, allergens=None):
    """요리의 알레르기 근거 경로. allergens를 주면 그 알레르기만."""
    paths = [row for row in run(RECIPE_PATHS, id=recipe_id) if not allergens or row["allergen"] in allergens]
    unknown = run(RECIPE_UNKNOWN, id=recipe_id)
    return paths, unknown


def verdict(paths, unknown, allergens):
    """먹어도 되는지: 'unsafe'(확정 있음) / 'caution'(주의·구성 불명만) / 'safe'."""
    relevant = [p for p in paths if p["allergen"] in allergens]
    if any(p["certainty"] == "확정" for p in relevant):
        return "unsafe"
    if relevant or unknown:
        return "caution"
    return "safe"


def select(candidates, limit, ranker=None):
    """추천 요리 고르기: 확정 알레르기가 있는 요리는 빼고(안전 필터), 남은 요리에서 limit개.
    ranker가 없으면 경고 없는 요리 먼저 → 레시피 번호순. 있으면 ranker(안전한 후보, limit)가 순서를 정한다
    (rerank.Reranker: 질문 유사도 + 다양성). ranker는 안전 필터를 통과한 후보만 받는다."""
    allowed = sorted((c for c in candidates if not c["definite"]), key=lambda c: int(c["id"]))
    excluded = [c for c in candidates if c["definite"]]
    if ranker is None:
        allowed.sort(key=lambda c: bool(c["caution"]) or bool(c["unknown"]))  # 안정 정렬이라 번호순은 유지
        return allowed[:limit], excluded, len(allowed)
    return ranker(allowed, limit), excluded, len(allowed)


def find_all(run, conditions):
    """세기·전부 찾기: 조건에 맞고, 포함 알레르기가 모두 '확정'으로 있고, 제외 알레르기는 확정으로 없는 요리 전부.
    CANDIDATES를 그대로 쓰되, 포함·제외 알레르기를 함께 넘겨 확정/주의를 받은 뒤 나눈다."""
    include = set(conditions.get("include_allergens", []))
    exclude = set(conditions["exclude_allergens"])
    candidates = run(
        CANDIDATES,
        dish=conditions["dish_type"] or None, method=conditions["cooking_method"] or None,
        max_kcal=conditions["max_kcal"], max_sodium=conditions["max_sodium_mg"],
        include=conditions["include_categories"], exclude=sorted(include | exclude),
    )
    matched, caution_only = [], []
    for c in candidates:
        definite = set(c["definite"])
        if definite & exclude:
            continue
        if include <= definite:
            matched.append(c)
        elif include and include <= definite | set(c["caution"]):
            caution_only.append(c)  # 포함 알레르기가 '주의'로만 있는 요리 (세지는 않고 따로 알린다)
    return candidates, matched, caution_only


LIST_EVIDENCE_LIMIT = 30  # 전부 찾기에서 근거 경로를 붙일 최대 요리 수


def search(run, conditions, recipes, ranker=None):
    """조건에 따라 조회한다. 결과는 answer.py가 근거와 답변으로 만든다.
    ranker: 추천 질문에서만 쓰는 순서 정하기 (rerank.Reranker.for_question). 없으면 레시피 번호순."""
    intent = conditions["intent"]
    exclude = conditions["exclude_allergens"]
    if intent in ("count", "list_all"):
        candidates, matched, caution_only = find_all(run, conditions)
        include = conditions.get("include_allergens", [])
        categories = conditions.get("include_categories", [])
        items = []
        for n, item in enumerate(matched):
            paths = []
            if include and n < LIST_EVIDENCE_LIMIT:
                paths, _ = recipe_evidence(run, item["id"], include)
                paths = [p for p in paths if p["certainty"] == "확정"]
            elif categories and n < LIST_EVIDENCE_LIMIT:  # 분류로 고른 요리도 왜 골랐는지 근거로 남긴다
                seen = set()
                paths = [p for p in run(RECIPE_CATEGORY_PATHS, id=item["id"], categories=categories)
                         if not (p["category"] in seen or seen.add(p["category"]))]  # 분류마다 첫 경로 하나
            items.append({**item, "paths": paths})
        return {"intent": intent, "conditions": conditions, "count": len(matched), "items": items,
                "caution_only": caution_only, "scope": len(candidates)}
    if intent == "recommend":
        candidates = run(
            CANDIDATES,
            dish=conditions["dish_type"] or None, method=conditions["cooking_method"] or None,
            max_kcal=conditions["max_kcal"], max_sodium=conditions["max_sodium_mg"],
            include=conditions["include_categories"], exclude=exclude,
        )
        chosen, excluded, total = select(candidates, conditions["limit"], ranker)
        recommended = []
        for item in chosen:
            paths, unknown = recipe_evidence(run, item["id"], exclude) if (item["caution"] or item["unknown"]) else ([], [])
            recommended.append({**item, "paths": paths, "unknown_ingredients": unknown})
        excluded_examples = []
        for item in excluded[:3]:
            paths, _ = recipe_evidence(run, item["id"], exclude)
            excluded_examples.append({**item, "paths": [p for p in paths if p["certainty"] == "확정"]})
        return {"intent": intent, "conditions": conditions, "recommended": recommended,
                "excluded_examples": excluded_examples, "matched": len(candidates),
                "allowed": total, "excluded_count": len(excluded), "reranked": ranker is not None,
                "preference": getattr(ranker, "preference", None), "preference_terms": getattr(ranker, "terms", None)}

    recipe_id, suggestions = resolve_recipe(conditions["recipe_name"], recipes)
    if recipe_id is None:
        return {"intent": intent, "conditions": conditions, "not_found": conditions["recipe_name"],
                "suggestions": suggestions}
    detail = run(RECIPE_DETAIL, id=recipe_id)[0]
    paths, unknown = recipe_evidence(run, recipe_id)
    result = {"intent": intent, "conditions": conditions, "recipe": detail["recipe"],
              "ingredients": detail["ingredients"], "paths": paths, "unknown_ingredients": unknown}
    if intent in ("check_safety", "explain") and exclude:
        result["verdict"] = verdict(paths, unknown, exclude)
    return result


def session_runner(session):
    return lambda query, **params: [record.data() for record in session.run(query, **params)]
