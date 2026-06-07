"""Knowledge Graph visualization tab — interactive entity-relation explorer."""

from __future__ import annotations

import html
import json
import logging

import gradio as gr

from src.config import Config
from src.graph.graph_store import create_graph_store

logger = logging.getLogger(__name__)
config = Config()


def _build_graph_json(entity: str, max_depth: int = 2) -> str:
    """Query graph store and return JSON for vis.js network visualization."""
    if not config.graph.enabled:
        return json.dumps({"nodes": [], "edges": [], "error": "Graph not enabled in config.yaml"})

    try:
        store = create_graph_store(config.graph)
    except Exception as e:
        return json.dumps({"nodes": [], "edges": [], "error": str(e)})

    if not entity.strip():
        # Show overall stats
        stats = store.stats()
        return json.dumps({"nodes": [], "edges": [], "stats": stats, "error": ""})

    triples = store.query_neighbors(entity, max_depth=max_depth)

    if not triples:
        return json.dumps({"nodes": [], "edges": [], "error": f"No triples found for '{entity}'"})

    # Build vis.js compatible JSON
    nodes_set: dict[str, dict] = {}
    edges: list[dict] = []

    # Add the query entity as root node
    nodes_set[entity] = {"id": entity, "label": entity, "group": "root"}

    for t in triples:
        if t.head not in nodes_set:
            nodes_set[t.head] = {"id": t.head, "label": t.head, "group": "entity"}
        if t.tail not in nodes_set:
            nodes_set[t.tail] = {"id": t.tail, "label": t.tail, "group": "value"}

        edges.append({
            "from": t.head,
            "to": t.tail,
            "label": t.relation,
            "arrows": "to",
        })

    return json.dumps({
        "nodes": list(nodes_set.values()),
        "edges": edges,
        "error": "",
        "count": len(triples),
    })


def _render_graph(entity: str, max_depth: int) -> str:
    """Return HTML with embedded vis.js graph."""
    graph_data = _build_graph_json(entity, max_depth)

    # Stats-only response
    try:
        parsed = json.loads(graph_data)
    except json.JSONDecodeError:
        return '<div style="color:var(--error);">Failed to parse graph data</div>'

    if parsed.get("error"):
        err = html.escape(parsed["error"])
        if parsed.get("stats"):
            s = parsed["stats"]
            return (
                f'<div style="color:var(--warning);margin-bottom:8px;">'
                f'Graph enabled but no entity specified. Current stats:</div>'
                f'<div style="font-size:0.85rem;">'
                f'Entities: {s.get("num_entities", 0)} | '
                f'Triples: {s.get("num_triples", 0)} | '
                f'Sources: {s.get("num_sources", 0)}</div>'
            )
        return f'<div style="color:var(--error);">{err}</div>'

    nodes = parsed.get("nodes", [])

    if not nodes:
        return '<div style="color:var(--text-muted);">No data to visualize</div>'

    count = parsed.get("count", 0)

    vis_html = f"""
    <div style="margin-bottom:8px;font-size:0.8rem;color:var(--text-muted);">
        Found {count} triples | {len(nodes)} entities | Depth: {max_depth}
    </div>
    <div id="kg-graph" style="width:100%;height:500px;background:#0D1421;
        border:1px solid var(--border);border-radius:6px;"></div>
    <script type="text/javascript">
    (function() {{
        if (typeof vis === 'undefined') {{
            var s = document.createElement('script');
            s.src = 'https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js';
            s.onload = function() {{ initGraph(); }};
            document.head.appendChild(s);
        }} else {{
            initGraph();
        }}
        function initGraph() {{
            var container = document.getElementById('kg-graph');
            if (!container) return;
            var data = {graph_data};
            var nodes = new vis.DataSet(data.nodes.map(function(n) {{
                var isRoot = n.group === 'root';
                return {{
                    id: n.id,
                    label: n.label.length > 12 ? n.label.substring(0, 12) + '...' : n.label,
                    title: n.label,
                    color: {{
                        background: isRoot ? '#C9A84C' : (n.group === 'value' ? '#1a73e8' : '#22C55E'),
                        border: isRoot ? '#C9A84C' : '#1E293B',
                        highlight: {{ background: isRoot ? '#DAB85C' : '#2a83f8', border: '#fff' }}
                    }},
                    font: {{ color: '#E8E6E3', size: isRoot ? 14 : 11 }},
                    shape: 'dot',
                    size: isRoot ? 20 : 12,
                    borderWidth: isRoot ? 2 : 1,
                }};
            }}));
            var edges = new vis.DataSet(data.edges.map(function(e) {{
                return {{
                    from: e.from,
                    to: e.to,
                    label: e.label,
                    arrows: 'to',
                    color: {{ color: '#4B5563', highlight: '#1a73e8' }},
                    font: {{ color: '#9CA3AF', size: 9, strokeWidth: 0 }},
                    smooth: {{ type: 'continuous' }},
                }};
            }}));
            var network = new vis.Network(container, {{ nodes: nodes, edges: edges }}, {{
                physics: {{
                    solver: 'forceAtlas2Based',
                    forceAtlas2Based: {{ gravitationalConstant: -80, springLength: 120 }},
                    stabilization: {{ iterations: 100 }}
                }},
                interaction: {{ hover: true, tooltipDelay: 200, zoomView: true }},
            }});
        }}
    }})();
    </script>
    """
    return vis_html


def create_graph_tab() -> None:
    """Create knowledge graph explorer tab."""
    gr.HTML('<div style="text-align:center;margin-bottom:8px;">'
            '<span style="color:var(--accent-gold);font-weight:600;">'
            'Knowledge Graph Explorer</span></div>')

    with gr.Row():
        entity_input = gr.Textbox(
            label="Entity Name",
            placeholder="Enter entity to explore, e.g. 贵州茅台",
            scale=4,
        )
        depth = gr.Slider(
            minimum=1, maximum=3, value=2, step=1,
            label="Depth",
        )
        search_btn = gr.Button("Explore", variant="primary", scale=1)

    graph_output = gr.HTML(value="<div style='color:var(--text-muted);text-align:center;padding:40px;'>"
                                  "Enter an entity name and click Explore</div>")

    def on_search(entity, d):
        return _render_graph(entity, int(d))

    search_btn.click(on_search, [entity_input, depth], [graph_output])
    entity_input.submit(on_search, [entity_input, depth], [graph_output])
