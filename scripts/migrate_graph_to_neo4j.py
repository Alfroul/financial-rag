"""将 NetworkX pickle 图谱迁移到 Neo4j。

用法:
    python -m scripts.migrate_graph_to_neo4j
"""
from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.graph.graph_store import Neo4jGraphStore
from src.graph.triple import Triple

logger = logging.getLogger(__name__)

PICKLE_PATH = "data/graph_store.pkl"


def load_networkx_graph(path: str) -> nx.MultiDiGraph:
    p = Path(path)
    if not p.exists():
        logger.error("Pickle file not found: %s", path)
        sys.exit(1)
    with open(p, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, nx.MultiDiGraph):
        logger.error("Expected MultiDiGraph, got %s", type(data).__name__)
        sys.exit(1)
    return data


def graph_to_triples(g: nx.MultiDiGraph) -> list[Triple]:
    triples: list[Triple] = []
    for u, v, data in g.edges(data=True):
        triples.append(Triple(
            head=u,
            relation=data.get("relation", ""),
            tail=v,
            source=data.get("source", ""),
        ))
    return triples


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = Config()
    graph_config = config.graph

    logger.info("Loading NetworkX graph from %s ...", PICKLE_PATH)
    nx_graph = load_networkx_graph(PICKLE_PATH)
    logger.info("NetworkX graph: %d nodes, %d edges", nx_graph.number_of_nodes(), nx_graph.number_of_edges())

    triples = graph_to_triples(nx_graph)
    logger.info("Extracted %d triples", len(triples))

    import os
    password = os.environ.get(graph_config.neo4j_password_env, "")
    if not password:
        logger.error("Neo4j password not set (env: %s)", graph_config.neo4j_password_env)
        sys.exit(1)

    neo4j_store = Neo4jGraphStore(
        uri=graph_config.neo4j_uri,
        user=graph_config.neo4j_user,
        password=password,
    )

    try:
        neo4j_store.clear()
        added = neo4j_store.add_triples(triples)
        logger.info("Inserted %d triples into Neo4j", added)

        stats = neo4j_store.stats()
        logger.info("Neo4j stats: %s", stats)

        nx_nodes = nx_graph.number_of_nodes()
        nx_edges = nx_graph.number_of_edges()
        if stats["nodes"] == nx_nodes and stats["edges"] == nx_edges:
            logger.info("Migration verified: node/edge counts match")
        else:
            logger.warning(
                "Mismatch! NetworkX(%d nodes, %d edges) vs Neo4j(%d nodes, %d edges)",
                nx_nodes, nx_edges, stats["nodes"], stats["edges"],
            )
    finally:
        neo4j_store.close()


if __name__ == "__main__":
    main()
