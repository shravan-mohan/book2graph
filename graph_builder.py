"""
graph_builder.py
----------------
Build a directed dependency graph from Gemini-extracted theorem data and
render an interactive HTML visualisation with pyvis.

Quick start
-----------
    from graph_builder import build_graph, visualize_graph

    G = build_graph("funcana_dependencies.json")
    visualize_graph(G, "theorem_graph.html")   # open in any browser
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
from pyvis.network import Network

# ── node styling by kind ──────────────────────────────────────────────────────

_KIND_STYLE: dict[str, dict] = {
    "Theorem":     {"color": "#4A90D9", "shape": "ellipse",   "size": 28},
    "Lemma":       {"color": "#27AE60", "shape": "ellipse",   "size": 22},
    "Corollary":   {"color": "#8EC6E6", "shape": "ellipse",   "size": 20},
    "Proposition": {"color": "#9B59B6", "shape": "ellipse",   "size": 22},
    "Definition":  {"color": "#E67E22", "shape": "box",       "size": 20},
    "Remark":      {"color": "#95A5A6", "shape": "dot",       "size": 14},
    "Example":     {"color": "#F1C40F", "shape": "dot",       "size": 14},
    "Unknown":     {"color": "#BDC3C7", "shape": "dot",       "size": 12},
}
_DEFAULT_STYLE = _KIND_STYLE["Unknown"]


def _style(kind: str) -> dict:
    return _KIND_STYLE.get(kind, _DEFAULT_STYLE)


# ── graph construction ────────────────────────────────────────────────────────

def build_graph(dependencies_json: str | Path) -> nx.DiGraph:
    """
    Build a directed graph from a dependency JSON file produced by
    :func:`dependency_extractor.extract_dependencies_from_sections`.

    Nodes
    -----
    One node per block label (e.g. ``"Theorem 1.4"``).
    Node attributes: ``kind``, ``number``, ``title``, ``section``,
    ``has_proof``.

    Edges
    -----
    A directed edge  A → B  means "A's proof depends on B".
    """
    path = Path(dependencies_json)
    with open(path, encoding="utf-8") as fh:
        data: list[dict] = json.load(fh)

    G: nx.DiGraph = nx.DiGraph()

    # ── pass 1: add all nodes ──────────────────────────────────────────────
    for section in data:
        section_title = section.get("section_title") or ""
        for block in section.get("blocks", []):
            label = block.get("label", "").strip()
            if not label:
                continue
            kind = block.get("kind", "Unknown")
            G.add_node(
                label,
                kind=kind,
                number=block.get("number", ""),
                title=block.get("title") or "",
                section=section_title,
                has_proof=block.get("has_proof", False),
                **_style(kind),
            )

    # ── pass 2: add edges ──────────────────────────────────────────────────
    for section in data:
        for block in section.get("blocks", []):
            source = block.get("label", "").strip()
            if not source:
                continue
            for dep in block.get("depends_on", []):
                dep = dep.strip()
                if not dep:
                    continue
                # Add stub node if the dependency was not found as a block
                # (can happen when it comes from a different chapter/section).
                if dep not in G:
                    kind = dep.split()[0] if dep else "Unknown"
                    G.add_node(dep, kind=kind, number="", title="",
                               section="(external)", has_proof=False,
                               **_style(kind))
                G.add_edge(source, dep)

    return G


# ── statistics ────────────────────────────────────────────────────────────────

def graph_summary(G: nx.DiGraph) -> dict:
    """Return a dict of basic graph statistics."""
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "nodes_by_kind": {
            k: sum(1 for _, d in G.nodes(data=True) if d.get("kind") == k)
            for k in _KIND_STYLE
        },
        "most_depended_on": sorted(
            ((n, G.in_degree(n)) for n in G.nodes),
            key=lambda x: x[1], reverse=True
        )[:10],
        "most_dependencies": sorted(
            ((n, G.out_degree(n)) for n in G.nodes),
            key=lambda x: x[1], reverse=True
        )[:10],
        "weakly_connected_components": nx.number_weakly_connected_components(G),
        "has_cycle": not nx.is_directed_acyclic_graph(G),
    }


# ── layout engines ────────────────────────────────────────────────────────────

# Canonical left-to-right column order for the partite view.
_PARTITION_ORDER = [
    "Definition", "Lemma", "Theorem", "Corollary",
    "Proposition", "Remark", "Example",
]


def _sort_key(label: str) -> tuple[int, ...]:
    """Sort block labels numerically by their trailing number (e.g. '1.4' → (1,4))."""
    parts = label.rsplit(None, 1)
    try:
        return tuple(int(x) for x in parts[-1].split("."))
    except ValueError:
        return (999,)


def _spring_layout(G: nx.DiGraph, seed: int = 42, scale: float = 3000.0) -> dict:
    UG = G.to_undirected()
    if G.number_of_nodes() <= 300:
        return nx.kamada_kawai_layout(UG, scale=scale)
    return nx.spring_layout(UG, seed=seed, scale=scale, k=2.5)


def _partite_layout(
    G: nx.DiGraph,
    col_gap: float = 520.0,
    row_gap: float = 55.0,
) -> tuple[dict[str, tuple[float, float]], list[tuple[float, str, str]]]:
    """
    Arrange nodes in vertical columns grouped by kind.

    Returns
    -------
    pos        : dict mapping node_id → (x, y)
    col_labels : list of (x, kind_name, color) for drawing column headers
    """
    columns: dict[str, list[str]] = {k: [] for k in _PARTITION_ORDER}
    others: list[str] = []

    for node_id, attrs in G.nodes(data=True):
        kind = attrs.get("kind", "Unknown")
        if kind in columns:
            columns[kind].append(node_id)
        else:
            others.append(node_id)

    for kind in columns:
        columns[kind].sort(key=_sort_key)

    pos: dict[str, tuple[float, float]] = {}
    col_labels: list[tuple[float, str, str]] = []

    for col_idx, kind in enumerate(_PARTITION_ORDER):
        nodes = columns[kind]
        x = col_idx * col_gap
        n = len(nodes)
        y_top = -(n - 1) * row_gap / 2.0
        for row_idx, node_id in enumerate(nodes):
            pos[node_id] = (x, y_top + row_idx * row_gap)
        color = _KIND_STYLE.get(kind, _DEFAULT_STYLE)["color"]
        col_labels.append((x, kind, color))

    # unknown-kind nodes in a right-most column
    if others:
        x = len(_PARTITION_ORDER) * col_gap
        y_top = -(len(others) - 1) * row_gap / 2.0
        for row_idx, node_id in enumerate(others):
            pos[node_id] = (x, y_top + row_idx * row_gap)

    return pos, col_labels


# ── shared rendering core ─────────────────────────────────────────────────────

def _build_network(
    G: nx.DiGraph,
    pos: dict[str, tuple[float, float]],
    height: str,
    width: str,
) -> Network:
    """Create a pyvis Network with pre-computed fixed positions."""
    net = Network(
        height=height,
        width=width,
        directed=True,
        bgcolor="#1a1a2e",
        font_color="#ecf0f1",
        notebook=False,
        cdn_resources="in_line",
    )

    for node_id, attrs in G.nodes(data=True):
        title_text = attrs.get("title", "")
        section    = attrs.get("section", "")
        tooltip = (
            f"<b>{node_id}</b>"
            + (f"<br><i>{title_text}</i>" if title_text else "")
            + (f"<br>Section: {section}"  if section    else "")
            + f"<br>Cited by: {G.in_degree(node_id)}"
            + f"<br>Depends on: {G.out_degree(node_id)}"
        )
        x, y = pos.get(node_id, (0.0, 0.0))
        net.add_node(
            node_id,
            label=node_id,
            title=tooltip,
            color=attrs.get("color", _DEFAULT_STYLE["color"]),
            shape=attrs.get("shape", _DEFAULT_STYLE["shape"]),
            size=attrs.get("size",  _DEFAULT_STYLE["size"]),
            font={"size": 13, "color": "#ecf0f1"},
            borderWidth=2,
            x=float(x),
            y=float(y),
            physics=False,
        )

    for src, dst in G.edges():
        net.add_edge(
            src, dst,
            color={"color": "#7f8c8d", "highlight": "#e74c3c"},
            arrows="to",
            smooth={"type": "curvedCW", "roundness": 0.15},
            width=1.2,
        )

    net.set_options("""
    {
      "physics": { "enabled": false },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true,
        "multiselect": true,
        "zoomView": true
      },
      "edges": { "smooth": true }
    }
    """)
    return net


def _inject_column_headers(
    html_path: Path,
    col_labels: list[tuple[float, str, str]],
) -> None:
    """
    Inject a floating row of column-kind labels as a fixed HTML bar above the graph.
    The labels are evenly spaced to mirror the column positions.
    """
    if not col_labels:
        return
    xs = [x for x, _, _ in col_labels]
    x_min, x_range = min(xs), max(xs) - min(xs) or 1.0
    items_html = ""
    for x, kind, color in col_labels:
        pct = (x - x_min) / x_range * 88 + 6   # map to 6%–94% of bar width
        items_html += (
            f'<span style="position:absolute;left:{pct:.1f}%;transform:translateX(-50%);'
            f'color:{color};font-weight:700;font-size:13px;white-space:nowrap;">'
            f'{kind}</span>\n'
        )
    bar = f"""
<div id="col-headers" style="
    position:fixed; top:0; left:0; width:100%; height:30px;
    background:rgba(20,20,40,0.93); border-bottom:1px solid #2c3e50;
    z-index:9998; font-family:sans-serif; pointer-events:none;">
{items_html}</div>
<style>
  #mynetwork {{ margin-top: 30px; }}
  canvas {{ margin-top: 0 !important; }}
</style>
"""
    content = html_path.read_text(encoding="utf-8")
    content = content.replace("<body>", "<body>\n" + bar, 1)
    html_path.write_text(content, encoding="utf-8")


# ── kind-toggle panel (injected into the rendered HTML) ───────────────────────

def _inject_kind_toggles(html_path: Path, G: nx.DiGraph) -> None:
    """
    Inject an interactive toggle panel so the user can show/hide each node
    kind directly in the browser — no re-run needed.

    Implementation
    --------------
    * Builds a JS map  nodeKinds = { "Theorem 1.4": "Theorem", … }
    * Adds a fixed panel of labelled checkboxes (one per kind, pre-checked).
    * On change, calls  network.body.data.nodes.update(…)  to flip the
      ``hidden`` flag on every node of that kind (and their incident edges).
    """
    # Build node → kind mapping
    node_kinds: dict[str, str] = {}
    for node_id, attrs in G.nodes(data=True):
        node_kinds[node_id] = attrs.get("kind", "Unknown")

    present_kinds = sorted(
        {k for k in node_kinds.values() if k != "Unknown"},
        key=lambda k: _PARTITION_ORDER.index(k) if k in _PARTITION_ORDER else 99,
    )

    # JS object literal
    js_map = json.dumps(node_kinds, ensure_ascii=False)

    # Checkbox HTML for each kind
    checkboxes_html = ""
    for kind in present_kinds:
        color = _KIND_STYLE.get(kind, _DEFAULT_STYLE)["color"]
        checkboxes_html += (
            f'<label style="display:flex;align-items:center;gap:6px;cursor:pointer;">'
            f'<input type="checkbox" checked onchange="toggleKind(\'{kind}\',this.checked)"'
            f' style="accent-color:{color};width:15px;height:15px;">'
            f'<span style="color:{color};font-weight:600;">{kind}</span>'
            f'</label>\n'
        )

    panel_and_script = f"""
<div id="kind-toggles" style="
    position:fixed; top:40px; right:16px; z-index:9999;
    background:rgba(20,20,40,0.94); border:1px solid #2c3e50;
    border-radius:10px; padding:14px 18px; font-family:sans-serif;
    font-size:13px; color:#ecf0f1; min-width:140px;
    display:flex; flex-direction:column; gap:8px;">
  <b style="margin-bottom:4px;font-size:14px;">Show / Hide</b>
{checkboxes_html}</div>

<script>
var _nodeKinds = {js_map};

function toggleKind(kind, show) {{
    var updates = [];
    Object.keys(_nodeKinds).forEach(function(nodeId) {{
        if (_nodeKinds[nodeId] === kind) {{
            updates.push({{id: nodeId, hidden: !show}});
        }}
    }});
    network.body.data.nodes.update(updates);

    // also hide/show edges whose both endpoints are now hidden
    var hiddenNodes = new Set();
    network.body.data.nodes.forEach(function(n) {{
        if (n.hidden) hiddenNodes.add(n.id);
    }});
    var edgeUpdates = [];
    network.body.data.edges.forEach(function(e) {{
        var hide = hiddenNodes.has(e.from) || hiddenNodes.has(e.to);
        edgeUpdates.push({{id: e.id, hidden: hide}});
    }});
    network.body.data.edges.update(edgeUpdates);
}}
</script>
"""
    content = html_path.read_text(encoding="utf-8")
    content = content.replace("</body>", panel_and_script + "\n</body>")
    html_path.write_text(content, encoding="utf-8")


# ── public visualisation functions ────────────────────────────────────────────

def _filter_kinds(G: nx.DiGraph, exclude_kinds: set[str]) -> nx.DiGraph:
    """Return a subgraph with nodes of the given kinds removed."""
    keep = [n for n, d in G.nodes(data=True) if d.get("kind", "") not in exclude_kinds]
    return G.subgraph(keep).copy()


def visualize_graph(
    G: nx.DiGraph,
    output_html: str | Path = "theorem_graph.html",
    *,
    height: str = "920px",
    width: str = "100%",
    layout_seed: int = 42,
    layout_scale: float = 3000.0,
    exclude_kinds: set[str] | None = None,
) -> Path:
    """
    Render an interactive spring/force-directed HTML graph.

    Positions are pre-computed with networkx so the graph appears instantly.
    A toggle panel lets you show/hide each kind live in the browser.

    Parameters
    ----------
    exclude_kinds : set of kind names to permanently remove before rendering,
                   e.g. ``{"Example", "Remark"}``.
    """
    out = Path(output_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    view = _filter_kinds(G, exclude_kinds or set())
    pos  = _spring_layout(view, seed=layout_seed, scale=layout_scale)
    net  = _build_network(view, pos, height, width)
    net.save_graph(str(out))
    _inject_kind_toggles(out, view)
    return out


def visualize_partite_graph(
    G: nx.DiGraph,
    output_html: str | Path = "theorem_graph_partite.html",
    *,
    height: str = "960px",
    width: str = "100%",
    col_gap: float = 520.0,
    row_gap: float = 55.0,
    exclude_kinds: set[str] | None = None,
) -> Path:
    """
    Render a seven-partite HTML graph with one vertical column per node kind.

    Columns (left → right): Definition | Lemma | Theorem | Corollary |
                             Proposition | Remark | Example

    Nodes within each column are sorted numerically.  Edges cross freely
    between columns.  A fixed header bar labels each column.
    A toggle panel lets you show/hide individual columns live in the browser.

    Parameters
    ----------
    exclude_kinds : set of kind names to permanently remove before rendering,
                   e.g. ``{"Example", "Remark"}``.
    """
    out = Path(output_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    view = _filter_kinds(G, exclude_kinds or set())
    pos, col_labels = _partite_layout(view, col_gap=col_gap, row_gap=row_gap)
    net = _build_network(view, pos, height, width)
    net.save_graph(str(out))
    _inject_column_headers(out, col_labels)
    _inject_kind_toggles(out, view)
    return out


# ── legend helper ─────────────────────────────────────────────────────────────

def _inject_legend(html_path: Path) -> None:
    """Inject a colour-coded kind legend into the generated HTML."""
    legend_html = """
<div id="legend" style="
    position:fixed; bottom:20px; left:20px; z-index:9999;
    background:rgba(26,26,46,0.92); border:1px solid #34495e;
    border-radius:8px; padding:12px 16px; font-family:sans-serif;
    font-size:13px; color:#ecf0f1; line-height:1.8;">
  <b style="display:block;margin-bottom:6px;">Node types</b>
"""
    for kind, style in _KIND_STYLE.items():
        if kind == "Unknown":
            continue
        shape_symbol = "&#9632;" if style["shape"] == "box" else "&#9679;"
        legend_html += (
            f'  <span style="color:{style["color"]};font-size:16px;">'
            f'{shape_symbol}</span> {kind}<br>\n'
        )
    legend_html += "</div>\n"

    content = html_path.read_text(encoding="utf-8")
    content = content.replace("</body>", legend_html + "</body>")
    html_path.write_text(content, encoding="utf-8")


# ── export helpers ────────────────────────────────────────────────────────────

def export_graphml(G: nx.DiGraph, output_path: str | Path) -> Path:
    """Save graph as GraphML (importable into Gephi, Cytoscape, etc.)."""
    out = Path(output_path)
    nx.write_graphml(G, str(out))
    return out


def export_json(G: nx.DiGraph, output_path: str | Path) -> Path:
    """Save graph as node-link JSON."""
    out = Path(output_path)
    data = nx.node_link_data(G)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    return out


# ── __main__ ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import subprocess

    deps_json = Path.home() / "Downloads" / "funcana_dependencies.json"
    if not deps_json.exists():
        print(f"Dependency file not found: {deps_json}")
        print("Run dependency_extractor.py first.")
        sys.exit(1)

    print(f"Loading dependencies from {deps_json} …")
    G = build_graph(deps_json)

    summary = graph_summary(G)
    print(f"\nGraph summary")
    print(f"  Nodes : {summary['nodes']}")
    print(f"  Edges : {summary['edges']}")
    print(f"  Weakly connected components : {summary['weakly_connected_components']}")
    print(f"  Has cycle : {summary['has_cycle']}")
    print(f"\n  Nodes by kind:")
    for kind, count in summary["nodes_by_kind"].items():
        if count:
            print(f"    {kind:15s}: {count}")
    print(f"\n  Most cited (in-degree):")
    for label, deg in summary["most_depended_on"][:5]:
        print(f"    {label:30s}: cited by {deg} blocks")
    print(f"\n  Most dependencies (out-degree):")
    for label, deg in summary["most_dependencies"][:5]:
        print(f"    {label:30s}: depends on {deg} blocks")

    out_dir = Path.home() / "Downloads"

    html_out = out_dir / "theorem_graph.html"
    print(f"\nRendering spring layout → {html_out} …")
    visualize_graph(G, html_out)
    _inject_legend(html_out)

    partite_out = out_dir / "theorem_graph_partite.html"
    print(f"Rendering partite layout → {partite_out} …")
    visualize_partite_graph(G, partite_out)
    _inject_legend(partite_out)

    print("Done. Opening partite view in browser …")
    subprocess.run(["open", str(partite_out)])

    graphml_out = out_dir / "theorem_graph.graphml"
    export_graphml(G, graphml_out)
    print(f"GraphML export → {graphml_out}")
