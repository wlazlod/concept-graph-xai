"""ConceptGraph: a tree of business concepts with feature leaves.

Tree-only in v1 (one parent per node). DAG support is reserved for v2 via
optional per-edge weights.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import networkx as nx

NodeKind = Literal["concept", "feature"]
ROOT_PARENT: str = ""


@dataclass(frozen=True, slots=True)
class NodeView:
    """Read-only view of a single node in the graph."""

    name: str
    kind: NodeKind
    parent: str | None
    path: tuple[str, ...]
    metadata: Mapping[str, Any]


class ConceptGraph:
    """Tree of business concepts with feature leaves.

    Internally backed by a ``networkx.DiGraph`` where edges point from parent to
    child. Every node carries the attributes ``kind`` and ``metadata``.

    Invariants enforced at construction:

    * Exactly one root (a concept node with no parent).
    * Every leaf is a ``feature`` node; every internal node is a ``concept``.
    * Every feature node is a leaf.
    * Node names are unique across the graph.
    """

    def __init__(self, graph: nx.DiGraph, root: str) -> None:
        self._graph = graph
        self._root = root
        self._validate()
        self._order: list[str] = list(nx.dfs_preorder_nodes(self._graph, source=self._root))
        self._path_cache: dict[str, tuple[str, ...]] = {}

    @property
    def root(self) -> str:
        return self._root

    @property
    def graph(self) -> nx.DiGraph:
        """Return a snapshot copy of the underlying NetworkX DiGraph.

        The copy preserves all node attributes (``kind``, ``metadata``) and
        edges, but mutations to the returned object never reach back into
        the ConceptGraph — preventing accidental cache / order corruption.
        """

        return self._graph.copy()

    def __len__(self) -> int:
        return int(self._graph.number_of_nodes())

    def __contains__(self, name: object) -> bool:
        return name in self._graph

    def __iter__(self) -> Iterator[str]:
        return iter(self._order)

    def nodes_in_order(self) -> list[str]:
        """Deterministic depth-first preorder list of node names."""

        return list(self._order)

    def features(self) -> list[str]:
        return [n for n in self._order if self._graph.nodes[n]["kind"] == "feature"]

    def concepts(self) -> list[str]:
        return [n for n in self._order if self._graph.nodes[n]["kind"] == "concept"]

    def kind(self, name: str) -> NodeKind:
        self._require(name)
        return self._graph.nodes[name]["kind"]  # type: ignore[no-any-return]

    def parent_of(self, name: str) -> str | None:
        self._require(name)
        if name == self._root:
            return None
        preds = list(self._graph.predecessors(name))
        return preds[0] if preds else None

    def children_of(self, name: str) -> list[str]:
        self._require(name)
        return list(self._graph.successors(name))

    def descendants_of(self, name: str) -> list[str]:
        self._require(name)
        descendants = set(nx.descendants(self._graph, name))
        return [n for n in self._order if n in descendants]

    def descendant_features(self, name: str) -> list[str]:
        self._require(name)
        if self.kind(name) == "feature":
            return [name]
        descendants = set(nx.descendants(self._graph, name))
        return [n for n in self._order if n in descendants and self.kind(n) == "feature"]

    def path(self, name: str) -> tuple[str, ...]:
        self._require(name)
        cached = self._path_cache.get(name)
        if cached is not None:
            return cached
        parts: list[str] = []
        cursor: str | None = name
        while cursor is not None:
            parts.append(cursor)
            cursor = self.parent_of(cursor)
        path = tuple(reversed(parts))
        self._path_cache[name] = path
        return path

    def view(self, name: str) -> NodeView:
        self._require(name)
        return NodeView(
            name=name,
            kind=self.kind(name),
            parent=self.parent_of(name),
            path=self.path(name),
            metadata=dict(self._graph.nodes[name].get("metadata", {})),
        )

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | Mapping[str, list[str] | Mapping[str, Any]],
        *,
        root: str | None = None,
    ) -> ConceptGraph:
        """Build a ConceptGraph from a nested-dict representation.

        ``data`` is a mapping with exactly one top-level key (the root concept)
        whose value describes the tree:

        * a list of strings: a concept whose children are all features;
        * a mapping ``name -> subtree``: a concept whose children are concepts
          (recursive) or, when the leaf value is a list, features.
        """

        if root is not None:
            payload = data
        else:
            if len(data) != 1:
                raise ValueError(
                    "from_dict requires either a single top-level key or an explicit root="
                )
            root = next(iter(data))
            payload = {root: data[root]}

        graph = nx.DiGraph()
        graph.add_node(root, kind="concept", metadata={})
        cls._build_subtree(graph, root, payload[root])
        return cls(graph, root)

    @classmethod
    def from_networkx(cls, graph: nx.DiGraph, root: str) -> ConceptGraph:
        """Wrap an existing NetworkX DiGraph (must already have ``kind`` attrs)."""

        return cls(graph.copy(), root)

    @classmethod
    def from_yaml(cls, path: str) -> ConceptGraph:
        from concept_graph_xai.io.yaml import load_yaml

        return cls.from_dict(load_yaml(path))

    @staticmethod
    def _build_subtree(graph: nx.DiGraph, parent: str, value: Any) -> None:
        if isinstance(value, list):
            for leaf in value:
                if not isinstance(leaf, str):
                    raise ValueError(f"Feature leaf under {parent!r} must be a string, got {leaf!r}")
                if leaf in graph:
                    raise ValueError(f"Duplicate node name: {leaf!r}")
                graph.add_node(leaf, kind="feature", metadata={})
                graph.add_edge(parent, leaf)
            return
        if isinstance(value, Mapping):
            for name, sub in value.items():
                if name in graph:
                    raise ValueError(f"Duplicate node name: {name!r}")
                graph.add_node(name, kind="concept", metadata={})
                graph.add_edge(parent, name)
                ConceptGraph._build_subtree(graph, name, sub)
            return
        raise ValueError(f"Subtree under {parent!r} must be a mapping or a list, got {type(value).__name__}")

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def _require(self, name: str) -> None:
        if name not in self._graph:
            raise KeyError(f"Unknown node: {name!r}")

    def _validate(self) -> None:
        if self._root not in self._graph:
            raise ValueError(f"root {self._root!r} is not a node")
        if not nx.is_tree(self._graph):
            raise ValueError("ConceptGraph must be a tree (got a non-tree DiGraph)")
        if self._graph.in_degree(self._root) != 0:
            raise ValueError(f"root {self._root!r} must have no incoming edges")
        kinds: dict[str, NodeKind] = {n: self._graph.nodes[n].get("kind") for n in self._graph}
        if any(k not in {"concept", "feature"} for k in kinds.values()):
            bad = [n for n, k in kinds.items() if k not in {"concept", "feature"}]
            raise ValueError(f"nodes missing valid 'kind' attribute: {bad!r}")
        for node in self._graph:
            out = self._graph.out_degree(node)
            if kinds[node] == "feature" and out != 0:
                raise ValueError(f"feature {node!r} must be a leaf (has children)")
            if kinds[node] == "concept" and out == 0 and node != self._root:
                raise ValueError(f"concept {node!r} has no children (orphan concept)")

    # ------------------------------------------------------------------ #
    # Convenience iteration
    # ------------------------------------------------------------------ #
    def iter_nodes(self) -> Iterable[NodeView]:
        for name in self._order:
            yield self.view(name)
