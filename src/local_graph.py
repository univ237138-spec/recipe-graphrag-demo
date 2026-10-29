"""데모 모드: Neo4j 없이 메모리에 올린 그래프로 graph_search의 쿼리에 답한다.

Neo4j가 멈춰 있거나 네트워크가 안 될 때도 시연할 수 있게 한다.
같은 데이터(재료 → 분류 → 알레르기)로 계산하며, 5단계 3중 교차 검증에서 Neo4j Cypher 결과와
레시피 1,156개 모두 일치함을 확인했다. 테스트(tests/test_graphrag.py)도 이것을 쓴다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph_search  # noqa: E402
import ontology  # noqa: E402


class LocalRunner:
    def __init__(self, graph, recipes, mapping, mentions):
        self.recipes, self.mapping = recipes, mapping
        self.by_recipe = {}
        for m in mentions:
            self.by_recipe.setdefault(m["recipe_id"], []).append(m)
        nodes = ontology.category_nodes(graph)
        self.label = {n.split("#")[-1]: ontology.label(graph, n) for n in nodes}
        self.parent, self.links, self.unknown_ids = {}, {}, set()
        for n in nodes:
            nid = n.split("#")[-1]
            parents = [p for p in ontology.named_parents(graph, n) if p != ontology.RG.Ingredient]
            self.parent[nid] = parents[0].split("#")[-1] if parents else None
            self.links[nid] = [(ontology.label(graph, a), c) for a, c in ontology.direct_links(graph, n)]
            if any(True for _ in graph.objects(n, ontology.RG.requiresConfirmation)):
                self.unknown_ids.add(nid)
        self.legal = ontology.legal_basis(graph)  # 알레르기 → 법령 조항 (regulation.ttl)

    def chain(self, category):
        """분류 → 상위 분류들 (자기 자신 포함)."""
        result = []
        while category:
            result.append(category)
            category = self.parent[category]
        return result

    def ingredient_chain(self, ingredient):
        return self.chain(self.mapping[ingredient]["category_id"])

    def paths(self, recipe_id):
        rows = []
        for m in self.by_recipe.get(recipe_id, []):
            chain = self.ingredient_chain(m["ingredient"])
            for k, ancestor in enumerate(chain):
                for allergen, certainty in self.links[ancestor]:
                    legal = self.legal.get(allergen, {})
                    rows.append({"allergen": allergen, "certainty": certainty, "raw": m["raw"], "section": m["section"],
                                 "path": [self.recipes[recipe_id]["name"], m["ingredient"]]
                                 + [self.label[c] for c in chain[:k + 1]] + [allergen],
                                 "provision": legal.get("provision"), "legal_condition": legal.get("condition") or None})
        return rows

    def unknown(self, recipe_id):
        seen, rows = set(), []
        for m in self.by_recipe.get(recipe_id, []):
            if set(self.ingredient_chain(m["ingredient"])) & self.unknown_ids and m["ingredient"] not in seen:
                seen.add(m["ingredient"])
                rows.append({"ingredient": m["ingredient"], "raw": m["raw"]})
        return rows

    def __call__(self, query, **p):
        if query is graph_search.RECIPE_PATHS:
            return self.paths(p["id"])
        if query is graph_search.RECIPE_UNKNOWN:
            return self.unknown(p["id"])
        if query is graph_search.RECIPE_DETAIL:
            r = self.recipes[p["id"]]
            keys = ["id", "name", "dishType", "method", "kcal", "sodium_mg", "protein_g", "fat_g", "carbohydrate_g", "tip"]
            return [{"recipe": {k: r[k] for k in keys},
                     "ingredients": [{"ingredient": m["ingredient"], "raw": m["raw"], "section": m["section"]}
                                     for m in self.by_recipe.get(p["id"], [])]}]
        if query is graph_search.CANDIDATES:
            return self.candidates(**p)
        raise ValueError("데모 모드가 모르는 쿼리입니다")

    def candidates(self, dish, method, max_kcal, max_sodium, include, exclude):
        rows = []
        for rid, r in sorted(self.recipes.items(), key=lambda x: int(x[0])):
            if dish and r["dishType"] != dish or method and r["method"] != method:
                continue
            if max_kcal is not None and (r["kcal"] is None or r["kcal"] > max_kcal):
                continue
            if max_sodium is not None and (r["sodium_mg"] is None or r["sodium_mg"] > max_sodium):
                continue
            chains = [set(self.ingredient_chain(m["ingredient"])) for m in self.by_recipe.get(rid, [])]
            if not all(any(cat in c for c in chains) for cat in include):
                continue
            relevant = [x for x in self.paths(rid) if x["allergen"] in exclude]
            rows.append({"id": rid, "name": r["name"], "dishType": r["dishType"], "kcal": r["kcal"],
                         "sodium_mg": r["sodium_mg"],
                         "definite": sorted({x["allergen"] for x in relevant if x["certainty"] == "확정"}),
                         "caution": sorted({x["allergen"] for x in relevant if x["certainty"] == "주의"}),
                         "unknown": [u["ingredient"] for u in self.unknown(rid)]})
        return rows
