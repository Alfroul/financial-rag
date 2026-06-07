from __future__ import annotations

import logging
import os
import pickle
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx

from src.graph.triple import Triple

if TYPE_CHECKING:
    from neo4j import Driver

    from src.config import GraphConfig

logger = logging.getLogger(__name__)


class GraphStore(ABC):
    @abstractmethod
    def add_triples(self, triples: list[Triple]) -> int: ...

    @abstractmethod
    def query_neighbors(self, entity: str, max_depth: int = 1) -> list[Triple]: ...

    @abstractmethod
    def query_path(self, entity_a: str, entity_b: str) -> list[list[Triple]]: ...

    @abstractmethod
    def get_entities(self) -> list[str]: ...

    @abstractmethod
    def delete_by_source(self, source: str) -> int: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def stats(self) -> dict: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...


class NetworkxGraphStore(GraphStore):
    def __init__(self) -> None:
        self._graph = nx.MultiDiGraph()

    def add_triples(self, triples: list[Triple]) -> int:
        added = 0
        for t in triples:
            if self._graph.has_edge(t.head, t.tail):
                existing = self._graph[t.head][t.tail]
                if any(d.get("relation") == t.relation for d in existing.values()):
                    continue
            self._graph.add_node(t.head, type="entity")
            self._graph.add_node(t.tail, type="entity")
            self._graph.add_edge(t.head, t.tail, relation=t.relation, source=t.source)
            added += 1
        return added

    def query_neighbors(self, entity: str, max_depth: int = 1) -> list[Triple]:
        if entity not in self._graph:
            return []
        triples: list[Triple] = []
        visited: set[str] = {entity}
        queue: deque[tuple[str, int]] = deque([(entity, 0)])
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for _, tail, data in self._graph.out_edges(current, data=True):
                triples.append(Triple(
                    head=current,
                    relation=data.get("relation", ""),
                    tail=tail,
                    source=data.get("source", ""),
                ))
                if tail not in visited:
                    visited.add(tail)
                    queue.append((tail, depth + 1))
            for head, _, data in self._graph.in_edges(current, data=True):
                triples.append(Triple(
                    head=head,
                    relation=data.get("relation", ""),
                    tail=current,
                    source=data.get("source", ""),
                ))
                if head not in visited:
                    visited.add(head)
                    queue.append((head, depth + 1))
        return triples

    def query_path(self, entity_a: str, entity_b: str) -> list[list[Triple]]:
        if entity_a not in self._graph or entity_b not in self._graph:
            return []
        try:
            nodes = nx.shortest_path(self._graph.to_undirected(), entity_a, entity_b)
        except nx.NetworkXNoPath:
            return []
        path_triples: list[Triple] = []
        for i in range(len(nodes) - 1):
            u, v = nodes[i], nodes[i + 1]
            if self._graph.has_edge(u, v):
                data = list(self._graph[u][v].values())[0]
                path_triples.append(Triple(
                    head=u,
                    relation=data.get("relation", ""),
                    tail=v,
                    source=data.get("source", ""),
                ))
            elif self._graph.has_edge(v, u):
                data = list(self._graph[v][u].values())[0]
                path_triples.append(Triple(
                    head=v,
                    relation=data.get("relation", ""),
                    tail=u,
                    source=data.get("source", ""),
                ))
        return [path_triples] if path_triples else []

    def get_entities(self) -> list[str]:
        return list(self._graph.nodes)

    def delete_by_source(self, source: str) -> int:
        edges_to_remove = [
            (u, v, k)
            for u, v, k, data in self._graph.edges(data=True, keys=True)
            if data.get("source") == source
        ]
        self._graph.remove_edges_from(edges_to_remove)
        self._graph.remove_nodes_from(list(nx.isolates(self._graph)))
        return len(edges_to_remove)

    def clear(self) -> None:
        self._graph.clear()

    def stats(self) -> dict:
        sources = {
            data.get("source", "")
            for _, _, data in self._graph.edges(data=True)
        }
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "sources": len(sources),
        }

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self._graph, f)
        logger.info(
            "Graph saved to %s (%d nodes, %d edges)",
            path, self._graph.number_of_nodes(), self._graph.number_of_edges(),
        )

    def load(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            logger.info("Graph file not found at %s, starting with empty graph", path)
            return
        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, OSError) as e:
            logger.warning("Failed to load graph from %s: %s, starting with empty graph", path, e)
            return
        if not isinstance(data, nx.MultiDiGraph):
            logger.warning(
                "Graph file %s contains %s instead of MultiDiGraph, starting with empty graph",
                path, type(data).__name__,
            )
            return
        self._graph = data
        logger.info(
            "Graph loaded from %s (%d nodes, %d edges)",
            path, self._graph.number_of_nodes(), self._graph.number_of_edges(),
        )


class Neo4jGraphStore(GraphStore):
    def __init__(self, uri: str, user: str, password: str) -> None:
        from neo4j import GraphDatabase

        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        self._ensure_index()

    def _ensure_index(self) -> None:
        with self._driver.session() as session:
            session.run(
                "CREATE INDEX FOR (e:Entity) ON (e.name) IF NOT EXISTS"
            )

    def close(self) -> None:
        self._driver.close()

    def add_triples(self, triples: list[Triple]) -> int:
        if not triples:
            return 0
        params: list[dict[str, str]] = [
            {"head": t.head, "relation": t.relation, "tail": t.tail, "source": t.source}
            for t in triples
        ]
        with self._driver.session() as session:
            result = session.run(
                """
                UNWIND $batch AS row
                MERGE (h:Entity {name: row.head})
                MERGE (t:Entity {name: row.tail})
                MERGE (h)-[r:RELATES {type: row.relation}]->(t)
                ON CREATE SET r.source = row.source
                RETURN count(r) AS created
                """,
                batch=params,
            )
            return result.single()["created"]

    def query_neighbors(self, entity: str, max_depth: int = 1) -> list[Triple]:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (start:Entity {name: $entity})
                CALL {
                    WITH start
                    MATCH path = (start)-[:RELATES*1..2]-(other:Entity)
                    RETURN path
                }
                UNWIND relationships(path) AS rel
                WITH DISTINCT startNode(rel) AS h, rel, endNode(rel) AS t
                RETURN h.name AS head, rel.type AS relation, t.name AS tail,
                       coalesce(rel.source, '') AS source
                """,
                entity=entity,
                max_depth=max_depth,
            )
            triples: list[Triple] = []
            for record in result:
                triples.append(Triple(
                    head=record["head"],
                    relation=record["relation"],
                    tail=record["tail"],
                    source=record["source"],
                ))
            return triples[:100]

    def query_path(self, entity_a: str, entity_b: str) -> list[list[Triple]]:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (a:Entity {name: $entity_a}), (b:Entity {name: $entity_b})
                MATCH path = shortestPath((a)-[:RELATES*]-(b))
                RETURN [rel in relationships(path) | {
                    head: startNode(rel).name,
                    relation: rel.type,
                    tail: endNode(rel).name,
                    source: coalesce(rel.source, '')
                }] AS triples
                """,
                entity_a=entity_a,
                entity_b=entity_b,
            )
            record = result.single()
            if not record:
                return []
            path_triples = [
                Triple(head=t["head"], relation=t["relation"], tail=t["tail"], source=t["source"])
                for t in record["triples"]
            ]
            return [path_triples] if path_triples else []

    def get_entities(self) -> list[str]:
        with self._driver.session() as session:
            result = session.run("MATCH (e:Entity) RETURN e.name AS name")
            return [record["name"] for record in result]

    def delete_by_source(self, source: str) -> int:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH ()-[r:RELATES {source: $source}]->()
                DELETE r
                RETURN count(r) AS deleted
                """,
                source=source,
            )
            deleted = result.single()["deleted"]
            session.run(
                """
                MATCH (e:Entity)
                WHERE NOT (e)-[:RELATES]-()
                DELETE e
                """
            )
            return deleted

    def clear(self) -> None:
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def stats(self) -> dict:
        with self._driver.session() as session:
            nodes = session.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
            edges = session.run("MATCH ()-[r:RELATES]->() RETURN count(r) AS c").single()["c"]
            sources = session.run(
                "MATCH ()-[r:RELATES]->() RETURN count(DISTINCT r.source) AS c"
            ).single()["c"]
        return {"nodes": nodes, "edges": edges, "sources": sources}

    def save(self, path: str) -> None:
        logger.info("Neo4j backend: save() is a no-op (data persists in Neo4j)")

    def load(self, path: str) -> None:
        logger.info("Neo4j backend: load() is a no-op (data persists in Neo4j)")


def create_graph_store(config: GraphConfig) -> GraphStore:
    if config.backend == "neo4j":
        password = os.environ.get(config.neo4j_password_env, "")
        if not password:
            logger.warning(
                "Neo4j password not set (env: %s), falling back to NetworkX",
                config.neo4j_password_env,
            )
            return NetworkxGraphStore()
        try:
            store: GraphStore = Neo4jGraphStore(
                uri=config.neo4j_uri,
                user=config.neo4j_user,
                password=password,
            )
            logger.info("Using Neo4j graph store at %s", config.neo4j_uri)
            return store
        except Exception as e:
            logger.warning("Neo4j connection failed (%s), falling back to NetworkX", e)
            return NetworkxGraphStore()

    store = NetworkxGraphStore()
    persist = Path(config.persist_path)
    if persist.exists():
        store.load(str(persist))
    return store
