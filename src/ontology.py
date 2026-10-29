"""온톨로지(ontology/recipe.ttl)를 읽어 검사하고, 다음 단계에서 쓸 표를 만든다.

하는 일:
  1. 검사: 분류 계층이 식재료까지 이어지는지, 순환이 없는지, 알레르기 19개가 맞는지 등
  2. 알레르기 계산: 분류가 물려받는 알레르기를 SPARQL(rdfs:subClassOf*)로 계산
  3. 교차 검증: 같은 결과를 OWL RL 추론기(owlrl)로도 계산해 둘이 일치하는지 확인
  4. 내보내기: ontology/categories.csv, ontology/allergen_map.csv

사용법:
    python src/ontology.py
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS

ROOT = Path(__file__).resolve().parent.parent
TTL_PATH = ROOT / "ontology" / "recipe.ttl"
REGULATION_TTL_PATH = ROOT / "ontology" / "regulation.ttl"  # 자동 생성 (src/regulation.py)
RG = Namespace("http://example.org/recipe-graphrag#")
AL = Namespace("http://example.org/recipe-graphrag/allergen#")

CERTAINTY = {RG.definitelyContains: "확정", RG.mayContain: "주의"}
EXPECTED_ALLERGEN_COUNT = 19


def load(path=TTL_PATH, with_regulation=True):
    """기준 온톨로지(recipe.ttl) + 규정 문서에서 추출한 법적 근거(regulation.ttl, 있으면)."""
    graph = Graph()
    graph.parse(path, format="turtle")
    if with_regulation and REGULATION_TTL_PATH.exists():
        graph.parse(REGULATION_TTL_PATH, format="turtle")
    return graph


def legal_basis(graph):
    """{알레르기 이름: {"provision": 조항 번호, "condition": 법령 조건}} — regulation.ttl이 없으면 빈 dict."""
    result = {}
    for allergen in graph.subjects(RDF.type, RG.Allergen):
        provision = graph.value(allergen, RG.definedIn)
        if provision is not None:
            result[label(graph, allergen)] = {"provision": str(graph.value(provision, RG.provisionId)),
                                              "condition": str(graph.value(allergen, RG.legalCondition) or "")}
    return result


def label(graph, node):
    for value in graph.objects(node, RDFS.label):
        if getattr(value, "language", None) == "ko":
            return str(value)
    return node.split("#")[-1]


def category_nodes(graph):
    """rg:Ingredient 아래의 모든 분류 (Ingredient 자신은 제외)."""
    query = """
        SELECT DISTINCT ?c WHERE {
            ?c rdfs:subClassOf+ rg:Ingredient .
            FILTER(isIRI(?c))
        }"""
    return sorted({row.c for row in graph.query(query, initNs={"rdfs": RDFS, "rg": RG})}, key=str)


def named_parents(graph, node):
    return [p for p in graph.objects(node, RDFS.subClassOf) if isinstance(p, URIRef)]


def direct_links(graph, node):
    """분류에 직접 붙은 알레르기: [(알레르기 IRI, 확정/주의)]."""
    links = []
    for restriction in graph.objects(node, RDFS.subClassOf):
        if not isinstance(restriction, BNode):
            continue
        prop = graph.value(restriction, OWL.onProperty)
        value = graph.value(restriction, OWL.hasValue)
        if prop in CERTAINTY and value is not None:
            links.append((value, CERTAINTY[prop]))
    return links


def effective_allergens(graph, node):
    """상위 분류에서 물려받은 것까지 포함한 알레르기: {알레르기 이름: 확정/주의}.
    같은 알레르기가 확정과 주의로 모두 오면 확정이 이긴다."""
    query = """
        SELECT ?allergen ?prop WHERE {
            ?node rdfs:subClassOf* ?ancestor .
            ?ancestor rdfs:subClassOf ?r .
            ?r owl:onProperty ?prop ; owl:hasValue ?allergen .
        }"""
    result = {}
    for row in graph.query(query, initNs={"rdfs": RDFS, "owl": OWL}, initBindings={"node": node}):
        name, certainty = label(graph, row.allergen), CERTAINTY.get(row.prop)
        if certainty and result.get(name) != "확정":
            result[name] = certainty
    return result


def requires_confirmation(graph, node):
    query = "ASK { ?node rdfs:subClassOf* ?a . ?a rg:requiresConfirmation true }"
    return bool(graph.query(query, initNs={"rdfs": RDFS, "rg": RG}, initBindings={"node": node}).askAnswer)


def category_of_example(graph, name):
    """skos:example 목록에 이 재료명이 있는 분류. 없으면 None."""
    for node in category_nodes(graph):
        examples = str(graph.value(node, SKOS.example) or "")
        if name in [item.strip() for item in examples.replace("(", ",").replace(")", ",").split(",")]:
            return node
    return None


def path_to_root(graph, node):
    names = [label(graph, node)]
    while True:
        parents = named_parents(graph, node)
        if not parents or parents[0] == RG.Ingredient or len(names) > 20:
            return " > ".join(reversed(names))
        node = parents[0]
        names.append(label(graph, node))


def validate(graph):
    """문제 목록을 돌려준다. 비어 있으면 통과."""
    errors = []
    allergens = set(graph.subjects(RDF.type, RG.Allergen))
    if len(allergens) != EXPECTED_ALLERGEN_COUNT:
        errors.append(f"알레르기 개수가 {len(allergens)}개다 (기대값 {EXPECTED_ALLERGEN_COUNT}).")

    labels = defaultdict(list)
    for node in category_nodes(graph):
        name = label(graph, node)
        labels[name].append(node)
        if name == node.split("#")[-1]:
            errors.append(f"{node}: 한국어 이름(rdfs:label@ko)이 없다.")
        parents = named_parents(graph, node)
        if len(parents) != 1:
            errors.append(f"{name}: 상위 분류가 {len(parents)}개다 (하나여야 한다).")
        for allergen, _ in direct_links(graph, node):
            if allergen not in allergens:
                errors.append(f"{name}: 정의되지 않은 알레르기 {allergen}를 가리킨다.")
        for restriction in graph.objects(node, RDFS.subClassOf):
            if isinstance(restriction, BNode) and graph.value(restriction, OWL.onProperty) not in CERTAINTY:
                errors.append(f"{name}: 확정/주의가 아닌 속성의 제약이 있다.")
        # 순환 검사: 상위로 올라가다 자기 자신을 다시 만나면 안 된다
        seen, current = {node}, node
        while True:
            parents = named_parents(graph, current)
            if not parents or parents[0] == RG.Ingredient:
                break
            current = parents[0]
            if current in seen:
                errors.append(f"{name}: 분류 계층에 순환이 있다.")
                break
            seen.add(current)
    for name, nodes in labels.items():
        if len(nodes) > 1:
            errors.append(f"분류 이름 '{name}'이 {len(nodes)}번 쓰였다 (LLM 매핑 때 헷갈림).")
    return errors


def owl_rl_allergens(graph):
    """OWL RL 추론기로 분류별 알레르기를 계산한다 (교차 검증용).
    분류마다 가상의 재료 1개와 그 재료만 쓴 가상의 레시피 1개를 만들고,
    property chain(rg:recipeDefinitelyContains 등)으로 레시피의 알레르기가 추론되는지 본다."""
    import owlrl

    world = Graph()
    for triple in graph:
        world.add(triple)
    nodes = category_nodes(graph)
    for index, node in enumerate(nodes):
        ingredient, recipe = RG[f"_ingredient{index}"], RG[f"_recipe{index}"]
        world.add((ingredient, RDF.type, node))
        world.add((recipe, RDF.type, RG.Recipe))
        world.add((recipe, RG.hasIngredient, ingredient))
    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(world)

    result = {}
    for index, node in enumerate(nodes):
        recipe = RG[f"_recipe{index}"]
        found = {label(graph, a): "주의" for a in world.objects(recipe, RG.recipeMayContain)}
        found.update({label(graph, a): "확정" for a in world.objects(recipe, RG.recipeDefinitelyContains)})
        result[node] = found
    return result


def cross_check(graph):
    """SPARQL 계산과 OWL RL 추론의 결과가 다른 분류 목록."""
    by_owl = owl_rl_allergens(graph)
    return [
        (label(graph, node), effective_allergens(graph, node), by_owl[node])
        for node in category_nodes(graph)
        if effective_allergens(graph, node) != by_owl[node]
    ]


def export(graph, out_dir=ROOT / "ontology"):
    categories_path, allergen_path = out_dir / "categories.csv", out_dir / "allergen_map.csv"
    with categories_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "분류", "상위 분류", "경로", "알레르기(물려받은 것 포함)", "확인 필요", "예시"])
        for node in category_nodes(graph):
            parents = named_parents(graph, node)
            allergens = effective_allergens(graph, node)
            writer.writerow([
                node.split("#")[-1],
                label(graph, node),
                label(graph, parents[0]) if parents else "",
                path_to_root(graph, node),
                ", ".join(f"{name}({certainty})" for name, certainty in sorted(allergens.items())),
                "예" if requires_confirmation(graph, node) else "",
                str(graph.value(node, SKOS.example) or ""),
            ])
    with allergen_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["분류", "알레르기", "확정/주의", "근거"])
        for node in category_nodes(graph):
            note = str(graph.value(node, RG.allergenNote) or "")
            for allergen, certainty in sorted(direct_links(graph, node), key=lambda x: label(graph, x[0])):
                writer.writerow([label(graph, node), label(graph, allergen), certainty, note])
    return categories_path, allergen_path


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    graph = load()
    nodes = category_nodes(graph)
    links = sum(len(direct_links(graph, node)) for node in nodes)
    print(f"분류 {len(nodes)}개, 알레르기 연결 {links}개, 트리플 {len(graph)}개")

    errors = validate(graph)
    print("\n[검사] " + ("통과" if not errors else f"문제 {len(errors)}개"))
    for error in errors:
        print("  - " + error)

    mismatches = cross_check(graph)
    print("[교차 검증: SPARQL vs OWL RL 추론] " + ("일치" if not mismatches else f"불일치 {len(mismatches)}개"))
    for name, by_sparql, by_owl in mismatches:
        print(f"  - {name}: SPARQL={by_sparql} / OWL RL={by_owl}")

    for path in export(graph):
        print(f"저장: {path}")
    if errors or mismatches:
        sys.exit(1)


if __name__ == "__main__":
    main()
