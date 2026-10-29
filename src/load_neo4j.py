"""5단계: 레시피 지식그래프를 Neo4j에 적재하고, 알레르기 계산을 3중으로 교차 검증한다.

    (:Recipe)-[:HAS_INGREDIENT]->(:Ingredient)-[:IS_A]->(:Category)-[:IS_A*0..]->(:Category)-[:CONTAINS]->(:Allergen)
    (:Allergen)-[:DEFINED_IN]->(:Provision)-[:PART_OF]->(:Regulation)     법적 근거 (regulation.ttl, 규정 문서 PDF에서 추출)

교차 검증: 레시피별 알레르기를
    ① Python (재료 → 분류 → 온톨로지 SPARQL)
    ② OWL RL 추론기 (property chain)
    ③ Neo4j Cypher (가변 길이 경로)
로 각각 계산해 모두 같은지 확인한다.

사용법:
    python src/load_neo4j.py            # 적재(이 프로젝트의 노드만 지우고 다시 넣음) + 검증
    python src/load_neo4j.py --verify   # 검증만
접속: 주소·아이디는 Aura 접속 정보 파일(문서\\api-keys\\Neo4j-*.txt)에서 읽고, 비밀번호는 실행할 때 입력한다.
     환경변수 NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD가 있으면 그것을 먼저 쓴다.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ontology  # noqa: E402
import recipe_graph  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT / "data" / "allergen_crosscheck.json"
LABELS = ["Recipe", "Ingredient", "Category", "Allergen", "Provision", "Regulation"]
RELATIONS = ["HAS_INGREDIENT", "IS_A", "CONTAINS", "DEFINED_IN", "PART_OF"]
BATCH = 2000

RECIPE_ALLERGENS_CYPHER = """
MATCH (r:Recipe)-[:HAS_INGREDIENT]->(:Ingredient)-[:IS_A]->(:Category)
      -[:IS_A*0..]->(:Category)-[c:CONTAINS]->(a:Allergen)
RETURN r.id AS recipe, a.name AS allergen, collect(DISTINCT c.certainty) AS certainties
"""
UNKNOWN_CYPHER = """
MATCH (r:Recipe)-[:HAS_INGREDIENT]->(:Ingredient)-[:IS_A]->(:Category)
      -[:IS_A*0..]->(:Category {requiresConfirmation: true})
RETURN DISTINCT r.id AS recipe
"""
EVIDENCE_CYPHER = """
MATCH path = (r:Recipe {id: $recipe})-[h:HAS_INGREDIENT]->(:Ingredient)-[:IS_A]->(:Category)
             -[:IS_A*0..]->(:Category)-[c:CONTAINS]->(a:Allergen {name: $allergen})
OPTIONAL MATCH (a)-[:DEFINED_IN]->(p:Provision)-[:PART_OF]->(g:Regulation)
RETURN [n IN nodes(path) | coalesce(n.name, n.id)] AS names, h.raw AS raw, c.certainty AS certainty,
       p.id AS provision, g.name AS regulation
"""
# 알레르기마다 법적 근거 조항이 정확히 1개여야 한다
LEGAL_CYPHER = """
MATCH (a:Allergen)
OPTIONAL MATCH (a)-[:DEFINED_IN]->(p:Provision)-[:PART_OF]->(:Regulation)
WITH a, count(p) AS n WHERE n <> 1
RETURN a.name AS allergen, n
"""


# ---------------------------------------------------------------- 적재할 행 만들기 (Neo4j 없이 테스트 가능)

def build_rows(graph, recipes, mapping, mentions):
    RG = ontology.RG
    allergens = [
        {"id": node.split("#")[-1], "name": ontology.label(graph, node),
         "legalTerm": str(graph.value(node, RG.legalTerm) or ""),
         "legalCondition": str(graph.value(node, RG.legalCondition) or "")}
        for node in graph.subjects(ontology.RDF.type, RG.Allergen)
    ]
    regulations = [
        {"id": node.split("#")[-1], "name": ontology.label(graph, node), "revised": str(graph.value(node, RG.revised)),
         "effective": str(graph.value(node, RG.effective)), "source": str(graph.value(node, ontology.RDFS.seeAlso))}
        for node in graph.subjects(ontology.RDF.type, RG.Regulation)
    ]
    provisions = [
        {"id": str(graph.value(node, RG.provisionId)), "title": ontology.label(graph, node),
         "text": str(graph.value(node, RG.provisionText)), "regulation": graph.value(node, RG.partOf).split("#")[-1]}
        for node in graph.subjects(ontology.RDF.type, RG.Provision)
    ]
    defined_in = [
        {"allergen": allergen.split("#")[-1], "provision": str(graph.value(provision, RG.provisionId))}
        for allergen, provision in graph.subject_objects(RG.definedIn)
    ]
    categories, category_edges, contains = [], [], []
    for node in ontology.category_nodes(graph):
        node_id = node.split("#")[-1]
        categories.append({
            "id": node_id,
            "name": ontology.label(graph, node),
            "path": ontology.path_to_root(graph, node),
            "examples": str(graph.value(node, ontology.SKOS.example) or ""),
            "allergenNote": str(graph.value(node, ontology.RG.allergenNote) or ""),
            "requiresConfirmation": any(True for _ in graph.objects(node, ontology.RG.requiresConfirmation)),
        })
        for parent in ontology.named_parents(graph, node):
            if parent != ontology.RG.Ingredient:
                category_edges.append({"child": node_id, "parent": parent.split("#")[-1]})
        for allergen, certainty in ontology.direct_links(graph, node):
            contains.append({"category": node_id, "allergen": allergen.split("#")[-1], "certainty": certainty})

    used = {m["ingredient"] for m in mentions}
    ingredients = [
        {
            "name": name,
            "lookupName": row["lookup_name"],
            "canonicalName": row["canonical_name"],
            "category": row["category_id"],
            "mappingSource": row["source"],
            "confidence": row["confidence"],
        }
        for name, row in mapping.items() if name in used
    ]
    return {
        "allergens": allergens,
        "categories": categories,
        "category_edges": category_edges,
        "contains": contains,
        "ingredients": ingredients,
        "recipes": list(recipes.values()),
        "mentions": mentions,
        "regulations": regulations,
        "provisions": provisions,
        "defined_in": defined_in,
    }


# ---------------------------------------------------------------- Neo4j

def credentials_file():
    """Neo4j Aura에서 내려받은 접속 정보 파일(Neo4j-*.txt)을 찾는다. 비밀번호 줄은 읽지 않는다."""
    if os.environ.get("NEO4J_CREDENTIALS_FILE"):
        return Path(os.environ["NEO4J_CREDENTIALS_FILE"])
    home = Path.home()
    for folder in [home / "OneDrive" / "문서" / "api-keys", home / "Documents" / "api-keys",
                   home / "OneDrive" / "Documents" / "api-keys", home / "Downloads"]:
        found = sorted(folder.glob("Neo4j-*.txt")) if folder.exists() else []
        if found:
            return found[-1]
    return None


def read_settings(path):
    """접속 정보 파일에서 주소·아이디·DB 이름만 읽는다."""
    settings = {}
    if path and path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_DATABASE"):
                settings[key.strip()] = value.strip()
    return settings


def driver_args(uri):
    """neo4j+s:// 주소를 neo4j://로 바꾼다. 암호화와 인증서 검증은 driver_options에서 켠다."""
    for secure, plain in (("neo4j+s://", "neo4j://"), ("bolt+s://", "bolt://")):
        if uri.startswith(secure):
            return (plain + uri[len(secure):],)
    return (uri,)


def driver_options(uri):
    """Aura 인증서(SSL.com 루트)가 Windows 인증서 저장소에 없으면 Python이 연결을 거부한다
    ("Unable to retrieve routing information"). 그래서 Mozilla 공인 인증서 목록(certifi)으로 검증한다.
    검증을 끄는 것이 아니라 신뢰할 인증서 목록만 바꾸는 것이다."""
    if not uri.startswith(("neo4j+s://", "bolt+s://")):
        return {}
    import certifi
    from neo4j import TrustCustomCAs

    return {"encrypted": True, "trusted_certificates": TrustCustomCAs(certifi.where())}


class ConnectionFailed(RuntimeError):
    pass


def connection_settings():
    """주소·아이디·DB 이름: 환경변수 > 접속 정보 파일."""
    path = credentials_file()
    settings = read_settings(path)
    return {
        "uri": os.environ.get("NEO4J_URI") or settings.get("NEO4J_URI"),
        "user": os.environ.get("NEO4J_USERNAME") or settings.get("NEO4J_USERNAME") or "neo4j",
        "database": os.environ.get("NEO4J_DATABASE") or settings.get("NEO4J_DATABASE") or None,
        "file": path.name if path else None,
    }


def open_driver(password, settings=None):
    """비밀번호로 연결한다. 실패하면 ConnectionFailed (화면·터미널 모두에서 쓴다)."""
    from neo4j import GraphDatabase
    from neo4j.exceptions import AuthError, ServiceUnavailable

    settings = settings or connection_settings()
    if not settings["uri"]:
        raise ConnectionFailed("Neo4j 주소를 찾지 못했습니다. 접속 정보 파일 위치나 NEO4J_URI를 확인하세요.")
    uri = settings["uri"]
    driver = GraphDatabase.driver(*driver_args(uri), auth=(settings["user"], password), **driver_options(uri))
    try:
        driver.verify_connectivity()
    except AuthError:
        driver.close()
        raise ConnectionFailed("비밀번호가 맞지 않습니다. 접속 정보 파일의 NEO4J_PASSWORD 값을 다시 복사해 보세요.") from None
    except ServiceUnavailable:
        driver.close()
        raise ConnectionFailed("Neo4j에 연결할 수 없습니다. 콘솔(console.neo4j.io)에서 인스턴스가 RUNNING인지 확인하세요 (멈춰 있으면 Resume).") from None
    return driver, settings["database"]


def connect():
    """터미널용: 환경변수가 없으면 비밀번호를 직접 입력받는다."""
    import getpass

    settings = connection_settings()
    if not settings["uri"]:
        settings["uri"] = input("Neo4j 주소 (neo4j+s://…): ").strip()
    print(f"접속: {settings['uri']} (아이디 {settings['user']}){' · 정보 파일: ' + settings['file'] if settings['file'] else ''}")
    password = os.environ.get("NEO4J_PASSWORD") or getpass.getpass(
        "Neo4j 비밀번호를 붙여 넣고 Enter (입력해도 화면에 안 보이는 게 정상): ").strip()
    try:
        driver, database = open_driver(password, settings)
    except ConnectionFailed as error:
        sys.exit(str(error))
    print("연결 성공")
    return driver, database


def run(session, query, **params):
    return list(session.run(query, **params))


def batches(rows, size=BATCH):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def reset(session):
    """이 프로젝트의 노드(레시피·재료·분류·알레르기)만 지운다. 다른 데이터는 건드리지 않는다."""
    where = " OR ".join(f"n:{label}" for label in LABELS)
    while True:
        deleted = run(session, f"MATCH (n) WHERE {where} WITH n LIMIT 5000 DETACH DELETE n RETURN count(*) AS c")[0]["c"]
        if deleted == 0:
            return


def load(session, rows):
    for label, key in [("Recipe", "id"), ("Ingredient", "name"), ("Category", "id"), ("Allergen", "id"),
                       ("Provision", "id"), ("Regulation", "id")]:
        run(session, f"CREATE CONSTRAINT {label.lower()}_{key} IF NOT EXISTS FOR (n:{label}) REQUIRE n.{key} IS UNIQUE")

    run(session, """UNWIND $rows AS row MERGE (a:Allergen {id: row.id})
                    SET a.name = row.name, a.legalTerm = row.legalTerm, a.legalCondition = row.legalCondition""",
        rows=rows["allergens"])
    run(session, "UNWIND $rows AS row MERGE (g:Regulation {id: row.id}) SET g += row", rows=rows["regulations"])
    run(session, """UNWIND $rows AS row
                    MERGE (p:Provision {id: row.id}) SET p.title = row.title, p.text = row.text
                    WITH p, row MATCH (g:Regulation {id: row.regulation})
                    MERGE (p)-[:PART_OF]->(g)""", rows=rows["provisions"])
    run(session, """UNWIND $rows AS row
                    MATCH (a:Allergen {id: row.allergen}), (p:Provision {id: row.provision})
                    MERGE (a)-[:DEFINED_IN]->(p)""", rows=rows["defined_in"])
    run(session, "UNWIND $rows AS row MERGE (c:Category {id: row.id}) SET c += row", rows=rows["categories"])
    run(session, """UNWIND $rows AS row
                    MATCH (child:Category {id: row.child}), (parent:Category {id: row.parent})
                    MERGE (child)-[:IS_A]->(parent)""", rows=rows["category_edges"])
    run(session, """UNWIND $rows AS row
                    MATCH (c:Category {id: row.category}), (a:Allergen {id: row.allergen})
                    MERGE (c)-[x:CONTAINS]->(a) SET x.certainty = row.certainty""", rows=rows["contains"])
    for chunk in batches(rows["ingredients"]):
        run(session, """UNWIND $rows AS row
                        MERGE (i:Ingredient {name: row.name})
                        SET i.lookupName = row.lookupName, i.canonicalName = row.canonicalName,
                            i.mappingSource = row.mappingSource, i.confidence = row.confidence
                        WITH i, row MATCH (c:Category {id: row.category})
                        MERGE (i)-[:IS_A]->(c)""", rows=chunk)
    for chunk in batches(rows["recipes"]):
        run(session, "UNWIND $rows AS row MERGE (r:Recipe {id: row.id}) SET r += row", rows=chunk)
    for chunk in batches(rows["mentions"]):
        run(session, """UNWIND $rows AS row
                        MATCH (r:Recipe {id: row.recipe_id}), (i:Ingredient {name: row.ingredient})
                        CREATE (r)-[:HAS_INGREDIENT {position: row.position, section: row.section,
                                raw: row.raw, amount: row.amount, note: row.note}]->(i)""", rows=chunk)


def counts(session):
    result = {label: run(session, f"MATCH (n:{label}) RETURN count(n) AS c")[0]["c"] for label in LABELS}
    for rel in RELATIONS:
        result[rel] = run(session, f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c")[0]["c"]
    return result


def cypher_allergens(session):
    result = {}
    for record in run(session, RECIPE_ALLERGENS_CYPHER):
        certainty = "확정" if "확정" in record["certainties"] else "주의"
        recipe_graph.merge(result.setdefault(record["recipe"], {}), record["allergen"], certainty)
    for record in run(session, UNKNOWN_CYPHER):
        result.setdefault(record["recipe"], {})[recipe_graph.UNKNOWN] = "주의"
    return result


def evidence(session, recipe_id, allergen):
    return [dict(record) for record in run(session, EVIDENCE_CYPHER, recipe=recipe_id, allergen=allergen)]


# ---------------------------------------------------------------- 실행

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Neo4j 적재와 알레르기 3중 교차 검증")
    parser.add_argument("--verify", action="store_true", help="적재 없이 검증만")
    args = parser.parse_args()

    graph, recipes, mapping, mentions = recipe_graph.load_all()
    driver, database = connect()
    with driver, driver.session(database=database) as session:
        if not args.verify:
            started = time.time()
            print("적재 중… (이 프로젝트 노드를 지우고 다시 넣습니다)")
            reset(session)
            load(session, build_rows(graph, recipes, mapping, mentions))
            print(f"적재 완료 ({time.time() - started:.0f}초)")
        print("그래프 크기:", counts(session))

        print("\n[교차 검증] 레시피별 알레르기를 세 방법으로 계산합니다…")
        by_python = recipe_graph.expected_allergens(mentions, mapping, graph)
        by_cypher = cypher_allergens(session)
        by_owl = recipe_graph.owl_allergens(mentions, mapping, graph)
        checks = {
            "python_vs_cypher": recipe_graph.compare(by_python, by_cypher, recipes),
            "python_vs_owl": recipe_graph.compare(by_python, by_owl, recipes),
        }
        for name, mismatches in checks.items():
            print(f"  {name}: 레시피 {len(recipes)}개 중 불일치 {len(mismatches)}개")
            for rid, left, right in mismatches[:5]:
                print(f"    - {rid} {recipes[rid]['name']}: {left} / {right}")
        unlinked = [[r["allergen"], r["n"]] for r in run(session, LEGAL_CYPHER)]
        checks["allergen_without_one_provision"] = unlinked
        print(f"  법적 근거: 알레르기 중 조항 연결이 1개가 아닌 것 {len(unlinked)}개" + (f" {unlinked}" if unlinked else ""))

        print("\n[근거 경로 예시] 석류 보쌈김치(159) → 새우")
        for row in evidence(session, "159", "새우"):
            legal = f" → {row['provision']}" if row["provision"] else ""
            print(f"  {' → '.join(row['names'])}{legal}  ({row['certainty']}, 원문: {row['raw']})")

    report = {
        "recipes": len(recipes),
        "mismatches": {name: [list(item) for item in m] for name, m in checks.items()},
        "allergens_by_recipe": by_cypher,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {REPORT_PATH}")
    if any(checks.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
