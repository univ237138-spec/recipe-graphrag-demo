"""레시피 지식그래프 데이터를 모으고, 레시피별 알레르기를 계산한다 (Neo4j 없이).

그래프 구조 (Neo4j에 그대로 적재된다):
    (:Recipe)-[:HAS_INGREDIENT {amount, raw, section, note, position}]->(:Ingredient)
    (:Ingredient)-[:IS_A]->(:Category)-[:IS_A*]->(:Category)
    (:Category)-[:CONTAINS {certainty: '확정'|'주의'}]->(:Allergen)

레시피 알레르기를 두 방법으로 계산한다 (Neo4j Cypher 결과와 대조하기 위해):
    expected_allergens(): 재료 → 분류 → 온톨로지(SPARQL)로 물려받은 알레르기
    owl_allergens():      같은 데이터를 RDF 개체로 만들어 OWL RL 추론기로 추론
"""

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ontology  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RECIPES_PATH = ROOT / "data" / "raw" / "recipes.json"
MENTIONS_PATH = ROOT / "data" / "raw" / "ingredient_mentions.csv"
MAP_PATH = ROOT / "data" / "ingredient_map.csv"
NOT_INGREDIENT = "NOT_INGREDIENT"
UNKNOWN = "확인 필요"


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clean_step(text):
    text = (text or "").strip()
    return re.sub(r"(?<=[.!?])[a-z]$", "", text)  # 원본 끝의 "…건진다.a" 같은 이미지 표시 글자 제거


def load_recipes(path=None):
    rows = json.loads((path or RECIPES_PATH).read_text(encoding="utf-8"))
    recipes = {}
    for row in rows:
        steps = [clean_step(row.get(f"MANUAL{i:02d}")) for i in range(1, 21)]
        recipes[row["RCP_SEQ"]] = {
            "id": row["RCP_SEQ"],
            "name": row["RCP_NM"].strip(),
            "dishType": row.get("RCP_PAT2", ""),
            "method": row.get("RCP_WAY2", ""),
            "kcal": number(row.get("INFO_ENG")),
            "carbohydrate_g": number(row.get("INFO_CAR")),
            "protein_g": number(row.get("INFO_PRO")),
            "fat_g": number(row.get("INFO_FAT")),
            "sodium_mg": number(row.get("INFO_NA")),
            "ingredientsText": (row.get("RCP_PARTS_DTLS") or "").strip(),
            "steps": [step for step in steps if step],
            "tip": (row.get("RCP_NA_TIP") or "").strip(),
        }
    return recipes


def load_map(path=None):
    with (path or MAP_PATH).open(encoding="utf-8-sig") as file:
        return {row["name"]: row for row in csv.DictReader(file)}


def load_mentions(mapping, path=None):
    """재료 행 중 '재료 아님'을 뺀 것. 같은 레시피에 같은 재료가 두 번 나오면 두 행 그대로 둔다 (평가 질문 결정 사항)."""
    mentions, position = [], defaultdict(int)
    with (path or MENTIONS_PATH).open(encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            name = row["base_name"].strip()
            mapped = mapping.get(name)
            if not name or mapped is None or mapped["category_id"] == NOT_INGREDIENT:
                continue
            position[row["recipe_id"]] += 1
            mentions.append({
                "recipe_id": row["recipe_id"],
                "ingredient": name,
                "position": position[row["recipe_id"]],
                "section": row["section"],
                "raw": row["raw"],
                "amount": row["amount"],
                "note": row["note"],
            })
    return mentions


def category_allergens(graph):
    """{분류 id: {알레르기: 확정/주의}}, 그리고 '확인 필요' 분류 id 집합."""
    allergens, unknown = {}, set()
    for node in ontology.category_nodes(graph):
        node_id = node.split("#")[-1]
        allergens[node_id] = ontology.effective_allergens(graph, node)
        if ontology.requires_confirmation(graph, node):
            unknown.add(node_id)
    return allergens, unknown


def merge(result, allergen, certainty):
    if result.get(allergen) != "확정":
        result[allergen] = certainty


def expected_allergens(mentions, mapping, graph):
    """{레시피 id: {알레르기: 확정/주의, (구성 불명 재료가 있으면) '확인 필요': '주의'}}."""
    by_category, unknown = category_allergens(graph)
    result = defaultdict(dict)
    for mention in mentions:
        category = mapping[mention["ingredient"]]["category_id"]
        for allergen, certainty in by_category.get(category, {}).items():
            merge(result[mention["recipe_id"]], allergen, certainty)
        if category in unknown:
            result[mention["recipe_id"]][UNKNOWN] = "주의"
    return dict(result)


def owl_allergens(mentions, mapping, graph, recipe_ids=None):
    """레시피·재료를 RDF 개체로 만들고 OWL RL 추론기로 레시피 알레르기를 추론한다.
    property chain: rg:hasIngredient ∘ rg:definitelyContains → rg:recipeDefinitelyContains"""
    import owlrl
    from rdflib import Graph, URIRef
    from rdflib.namespace import RDF

    rg = ontology.RG
    world = Graph()
    for triple in graph:
        world.add(triple)
    wanted = set(recipe_ids) if recipe_ids else None
    ingredient_iri = {}
    for mention in mentions:
        if wanted and mention["recipe_id"] not in wanted:
            continue
        name = mention["ingredient"]
        if name not in ingredient_iri:
            ingredient_iri[name] = URIRef(f"{rg}ingredient/{len(ingredient_iri)}")
            world.add((ingredient_iri[name], RDF.type, rg[mapping[name]["category_id"]]))
        recipe = URIRef(f"{rg}recipe/{mention['recipe_id']}")
        world.add((recipe, RDF.type, rg.Recipe))
        world.add((recipe, rg.hasIngredient, ingredient_iri[name]))
    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(world)

    _, unknown = category_allergens(graph)
    unknown_iris = {rg[c] for c in unknown}
    result = defaultdict(dict)
    for recipe in set(world.subjects(RDF.type, rg.Recipe)):
        if not str(recipe).startswith(f"{rg}recipe/"):
            continue
        recipe_id = str(recipe).rsplit("/", 1)[-1]
        for allergen in world.objects(recipe, rg.recipeMayContain):
            merge(result[recipe_id], ontology.label(graph, allergen), "주의")
        for allergen in world.objects(recipe, rg.recipeDefinitelyContains):
            result[recipe_id][ontology.label(graph, allergen)] = "확정"
        for ingredient in world.objects(recipe, rg.hasIngredient):
            if set(world.objects(ingredient, RDF.type)) & unknown_iris:
                result[recipe_id][UNKNOWN] = "주의"
    return {key: value for key, value in result.items() if value}


def compare(left, right, recipe_ids):
    """두 계산 결과가 다른 레시피 목록: [(레시피 id, left, right)]."""
    return [(rid, left.get(rid, {}), right.get(rid, {})) for rid in sorted(recipe_ids, key=int)
            if left.get(rid, {}) != right.get(rid, {})]


def load_all():
    graph = ontology.load()
    mapping = load_map()
    return graph, load_recipes(), mapping, load_mentions(mapping)
