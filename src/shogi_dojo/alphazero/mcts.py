"""PUCT Monte Carlo Tree Search for AlphaZero-style play and self-play.

This module implements the search algorithm at the heart of AlphaZero:
best-first tree search guided by a neural network's policy (used as a
prior) and value (used to bootstrap leaf evaluations, replacing the
random rollouts of classical MCTS).

See ``doc/alphazero/AlphaZero_Reference.md`` §7 for the algorithm
walkthrough and ``doc/alphazero/AlphaZero_Training_Reference.md`` §4
for hyperparameter discussion.

Design choices for this implementation:

- **Game-agnostic interface.** The MCTS class is not hard-coded to any
  Shogi library. It takes four callables (``evaluate_fn``,
  ``legal_moves_fn``, ``apply_fn``, ``is_terminal_fn``) that together
  expose whatever game rules the caller wants to use. This keeps
  ``mcts.py`` free of shogi-rs / python-shogi dependencies and makes
  the algorithm testable against toy games.

- **Readable over fast.** Edges store their per-action statistics
  (``N``, ``W``, ``P``, child node) in a plain ``dict`` keyed by the
  move. A production implementation would use parallel NumPy arrays
  indexed by action id, but for a research-survey reference the dict
  form is much easier to read and debug. Expect a ~2-5× speed penalty
  that does not matter for the scale of search we run here.

- **No subtree reuse.** Each call to :meth:`MCTS.search` builds a fresh
  tree from the given root. Reusing the portion of the tree reachable
  after the actual played move is a standard optimization in
  production self-play engines, but it adds code complexity we do not
  need for the survey.

- **Constant ``c_puct``.** AlphaZero's PUCT formula is implemented with
  a single exploration constant, configurable via the ``c_puct``
  argument (default ``1.5``). Some open-source projects (LCZero,
  KataGo) use a log-scaling variant; we note that in the docstring but
  keep the simpler form here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


# ============================================================================
# Type aliases for the game-interface callables
# ============================================================================
#
# The MCTS class is written in terms of opaque "state" and "move" values;
# the caller decides what those types are. In Shogi these will typically
# be (Position, Move) from :mod:`shogi_dojo.alphazero.encoding` or the
# native types of a move-generation library.

State = Any
MoveType = Any

# evaluate_fn(state) -> (prior policy over legal moves, value estimate)
#   - prior: dict mapping each legal move at state to its network-predicted
#     probability. Should sum to ~1 across legal moves; it is not required
#     that the caller renormalize perfectly (small numerical drift is OK).
#   - value: scalar in [-1, +1], from the perspective of the side-to-move
#     at ``state``. +1 = current side wins, -1 = current side loses.
EvaluateFn = Callable[[State], tuple[dict[MoveType, float], float]]

# legal_moves_fn(state) -> list of legal moves at state.
LegalMovesFn = Callable[[State], list[MoveType]]

# apply_fn(state, move) -> new_state resulting from applying move.
# Must be pure (not mutate state).
ApplyFn = Callable[[State, MoveType], State]

# is_terminal_fn(state) -> (is_terminal, terminal_value)
# - is_terminal: True if the game ends at ``state``.
# - terminal_value: if terminal, a scalar in [-1, +1] from the perspective
#   of the side-to-move at ``state``. By convention this is typically
#   -1 when the previous mover just won (the side-to-move now loses),
#   +1 when they just lost, 0 for a draw. If not terminal, None.
IsTerminalFn = Callable[[State], tuple[bool, Optional[float]]]


# ============================================================================
# Tree data structures
# ============================================================================


@dataclass
class Edge:
    """Per-action statistics on an edge from a parent Node to a (lazily
    constructed) child.

    Attributes:
        prior: The network's prior probability ``P(s, a)`` for this move
            at the parent state. Set once at expansion; optionally mixed
            with Dirichlet noise at the root.
        visit_count: How many simulations have selected this edge.
            Denoted ``N(s, a)`` in the PUCT formula.
        total_value: Cumulative value observed through this edge during
            backup, from the parent's side-to-move perspective. Denoted
            ``W(s, a)`` in the paper.
        child: The ``Node`` reached by taking this move from the parent.
            Built lazily on the first backup through this edge; ``None``
            until then.
    """

    prior: float
    visit_count: int = 0
    total_value: float = 0.0
    child: "Optional[Node]" = None

    @property
    def q_value(self) -> float:
        """Mean value ``Q(s, a) = W / N``, or 0 if the edge is unvisited.

        The "0 for unvisited" convention is the AlphaZero paper's choice
        (technically undefined and sometimes replaced with an FPU —
        "First Play Urgency" — estimate in sophisticated engines). For
        the survey we use the simple form.
        """
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


@dataclass
class Node:
    """One position in the MCTS search tree.

    A node is created when it is first evaluated (expanded). Terminal
    nodes have ``is_terminal=True`` and an empty ``edges`` dict — their
    value is fixed by ``terminal_value`` and does not require a network
    call.

    Attributes:
        state: The opaque game state at this node. Passed through to
            the caller's ``legal_moves_fn`` / ``apply_fn`` /
            ``is_terminal_fn`` / ``evaluate_fn`` callbacks.
        edges: Mapping from legal move → ``Edge``. Empty for terminal
            nodes and before expansion.
        is_terminal: True if the game ends at this node.
        terminal_value: If terminal, the value from the node's STM
            perspective. ``None`` otherwise.
    """

    state: State
    edges: dict[MoveType, Edge] = field(default_factory=dict)
    is_terminal: bool = False
    terminal_value: Optional[float] = None

    @property
    def total_visits(self) -> int:
        """Sum of all edges' visit counts (used in PUCT's ``√Σ N`` term)."""
        return sum(e.visit_count for e in self.edges.values())

    @property
    def is_expanded(self) -> bool:
        """True if the node has had its legal moves enumerated and priors
        set (i.e. it's been evaluated at least once). Terminal nodes are
        trivially "expanded" — they just have no edges.
        """
        return self.is_terminal or bool(self.edges)


# ============================================================================
# PUCT selection
# ============================================================================


def puct_score(edge: Edge, parent_total_visits: int, c_puct: float) -> float:
    """Compute the PUCT selection score for a single edge.

    .. math::

        U(s, a) = Q(s, a) + c_{\\text{puct}} \\cdot P(s, a)
                  \\cdot \\frac{\\sqrt{\\sum_b N(s, b)}}{1 + N(s, a)}

    The first term exploits (prefers moves with good observed value);
    the second term explores (prefers moves with high prior and low
    visit count relative to siblings).

    Args:
        edge: The edge to score.
        parent_total_visits: ``Σ_b N(s, b)`` — summed visit count over
            all sibling edges (including this one). MCTS uses the raw
            sum, not ``sum + 1``; some implementations add 1 to avoid
            zero at the root first-expansion, but we keep the literal
            formula from the paper.
        c_puct: Exploration constant. AlphaZero uses ~1.0; LCZero ~1.5.

    Returns:
        The PUCT score. Higher is better during selection.
    """
    exploration = (
        c_puct
        * edge.prior
        * math.sqrt(parent_total_visits)
        / (1 + edge.visit_count)
    )
    return edge.q_value + exploration


def select_child(node: Node, c_puct: float) -> tuple[MoveType, Edge]:
    """Pick the child edge with the highest PUCT score.

    Raises:
        RuntimeError: if the node has no edges (terminal or
            unexpanded). Callers should never reach selection on such
            a node.
    """
    if not node.edges:
        raise RuntimeError(
            "select_child called on a node with no edges "
            "(terminal or unexpanded)"
        )
    parent_total = node.total_visits
    best_move = None
    best_score = -math.inf
    best_edge = None
    for move, edge in node.edges.items():
        score = puct_score(edge, parent_total, c_puct)
        if score > best_score:
            best_score = score
            best_move = move
            best_edge = edge
    # ``best_edge`` is guaranteed non-None since ``node.edges`` is non-empty.
    assert best_move is not None and best_edge is not None
    return best_move, best_edge


# ============================================================================
# Dirichlet noise
# ============================================================================


def apply_dirichlet_noise(
    node: Node,
    alpha: float,
    epsilon: float,
    rng: np.random.Generator | None = None,
) -> None:
    """Mix Dirichlet noise into the priors of a node's edges in place.

    .. math::

        P(a) \\leftarrow (1 - \\varepsilon) P(a) + \\varepsilon \\eta_a,
        \\qquad \\eta \\sim \\text{Dirichlet}(\\alpha, \\ldots, \\alpha)

    This is applied **only at the root** of a self-play search, to
    force the network to explore moves its policy might otherwise
    underweight. Evaluation matches (where we want the network's
    cleanest judgment) skip this step.

    Args:
        node: The node whose priors should be perturbed. Typically the
            root of the search tree.
        alpha: Dirichlet concentration. Paper uses 0.03 (Go), 0.3
            (chess), 0.15 (Shogi). Smaller α → flatter noise.
        epsilon: Mixture weight in ``[0, 1]``. Paper uses 0.25 across
            all games.
        rng: Optional NumPy random generator. Use a seeded generator
            for reproducible tests.
    """
    if not node.edges:
        return   # nothing to perturb on a terminal/unexpanded node
    rng = rng if rng is not None else np.random.default_rng()
    moves = list(node.edges.keys())
    noise = rng.dirichlet([alpha] * len(moves))
    for move, eta in zip(moves, noise):
        old = node.edges[move].prior
        node.edges[move].prior = (1.0 - epsilon) * old + epsilon * float(eta)


# ============================================================================
# MCTS driver
# ============================================================================


class MCTS:
    """Run PUCT-guided Monte Carlo Tree Search from a root state.

    The caller supplies four callbacks that together describe the
    game; the class is otherwise agnostic to what the game is.

    Args:
        evaluate_fn: ``state → (prior dict, value in [-1, +1])``. The
            prior dict is keyed by legal move; the value is from the
            STM's perspective.
        legal_moves_fn: ``state → list[move]``.
        apply_fn: ``(state, move) → new_state``. Must be pure.
        is_terminal_fn: ``state → (is_terminal, terminal_value_or_None)``.
            Terminal value is from STM's perspective (so ``-1`` when
            the previous mover just won).
        c_puct: PUCT exploration constant. Default 1.5 is reasonable for
            most games; 1.0 is the paper's chess/Shogi choice.
    """

    def __init__(
        self,
        evaluate_fn: EvaluateFn,
        legal_moves_fn: LegalMovesFn,
        apply_fn: ApplyFn,
        is_terminal_fn: IsTerminalFn,
        c_puct: float = 1.5,
    ):
        self.evaluate_fn = evaluate_fn
        self.legal_moves_fn = legal_moves_fn
        self.apply_fn = apply_fn
        self.is_terminal_fn = is_terminal_fn
        self.c_puct = c_puct

    # ---- public entry point ---------------------------------------------

    def search(
        self,
        root_state: State,
        num_simulations: int,
        dirichlet_alpha: float | None = None,
        dirichlet_epsilon: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> Node:
        """Run ``num_simulations`` MCTS simulations from ``root_state``.

        The tree is built fresh each call — no subtree carryover. After
        expanding the root, optional Dirichlet noise is mixed into the
        root priors before any simulations run, which is the standard
        AlphaZero self-play configuration.

        Args:
            root_state: Starting state for the search.
            num_simulations: Number of MCTS simulations to perform.
            dirichlet_alpha: Dirichlet concentration for root-noise
                injection. Omit or set to ``None`` to skip noise
                (e.g. during evaluation matches).
            dirichlet_epsilon: Mixture weight for the Dirichlet noise.
                Ignored when ``dirichlet_alpha is None``.
            rng: Optional NumPy RNG for reproducibility in tests.

        Returns:
            The fully populated root :class:`Node`. Caller can read
            visit counts via ``visit_counts(root)`` or sample a move
            via :func:`sample_move`.
        """
        root = Node(state=root_state)

        # Expand the root first so Dirichlet noise has priors to perturb.
        # If the root is terminal, return it immediately — no search to do.
        root_is_terminal, root_value = self.is_terminal_fn(root_state)
        if root_is_terminal:
            root.is_terminal = True
            root.terminal_value = root_value
            return root

        self._expand(root)
        if dirichlet_alpha is not None:
            apply_dirichlet_noise(root, dirichlet_alpha, dirichlet_epsilon, rng=rng)

        for _ in range(num_simulations):
            self._simulate(root)

        return root

    # ---- internals ------------------------------------------------------

    def _expand(self, node: Node) -> float:
        """Expand ``node`` by querying the network and creating child edges.

        Returns the network's value estimate at ``node`` (in the node's
        STM perspective), which the caller uses as the leaf value for
        backup.
        """
        prior, value = self.evaluate_fn(node.state)
        # Only create edges for moves the caller considered legal. If
        # ``prior`` accidentally contains illegal moves they are dropped;
        # if it is missing legal moves their prior defaults to 0.0.
        legal = self.legal_moves_fn(node.state)
        for move in legal:
            node.edges[move] = Edge(prior=float(prior.get(move, 0.0)))
        return value

    def _simulate(self, root: Node) -> None:
        """One MCTS simulation: SELECT → EXPAND/EVALUATE → BACKUP.

        Walks down from ``root`` via PUCT until it reaches either a
        terminal node or a leaf that has not yet been expanded; expands
        (or reads terminal value); then backs the value up the path
        flipping sign at every ply.
        """
        node = root
        path: list[tuple[Node, MoveType]] = []

        # SELECT: descend until we hit a leaf or terminal.
        while node.is_expanded and not node.is_terminal:
            move, edge = select_child(node, self.c_puct)
            path.append((node, move))
            if edge.child is None:
                # First descent through this edge: create the child but
                # don't expand it yet — that happens below.
                edge.child = Node(state=self.apply_fn(node.state, move))
                node = edge.child
                break
            node = edge.child

        # EVALUATE / EXPAND.
        if node.is_terminal:
            # Already known terminal (reached via subtree replay).
            value = node.terminal_value
        else:
            # Check terminality at the leaf first — if the game has
            # ended, the value comes from is_terminal_fn, not the
            # network.
            is_term, term_value = self.is_terminal_fn(node.state)
            if is_term:
                node.is_terminal = True
                node.terminal_value = term_value
                value = term_value
            else:
                value = self._expand(node)

        # BACKUP: walk back up the path, flipping sign each step because
        # the STM alternates between plies.
        assert value is not None
        for parent, move in reversed(path):
            edge = parent.edges[move]
            edge.visit_count += 1
            # The value we're backing up was measured from ``node``'s STM
            # view; the parent's STM is the other side, so flip sign.
            value = -value
            edge.total_value += value


# ============================================================================
# Helpers for extracting results from a finished root
# ============================================================================


def visit_counts(root: Node) -> dict[MoveType, int]:
    """Return a dict of ``{move: N(root, move)}``.

    This is the basis for both the action actually played in self-play
    (sampled from a temperature-controlled softmax of the visit counts)
    and the policy training target ``π`` (the visit-count distribution
    at temperature 1).
    """
    return {move: edge.visit_count for move, edge in root.edges.items()}


def policy_target(root: Node, temperature: float = 1.0) -> dict[MoveType, float]:
    """Return the MCTS-improved policy ``π`` at temperature ``τ``.

    .. math::

        \\pi(a) \\propto N(\\text{root}, a)^{1/\\tau}

    At ``τ = 1`` this is the raw visit-count distribution, which is the
    standard training target for the policy head. For ``τ → 0`` it
    concentrates on the most-visited move; at ``τ → ∞`` it approaches
    uniform over legal moves.

    Args:
        root: The expanded search tree's root.
        temperature: Temperature ``τ > 0``. Must be positive; if you
            want argmax behavior, use :func:`sample_move` with
            ``greedy=True`` instead of passing a tiny τ.

    Returns:
        Dict of ``{move: probability}``, summing to 1 over legal moves.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0; got {temperature}")
    counts = visit_counts(root)
    total = sum(counts.values())
    if total == 0:
        # No simulations were run (or root is terminal). Fall back to
        # a uniform-over-legal distribution if we have edges, else
        # empty dict.
        n = len(counts)
        return {m: 1.0 / n for m in counts} if n else {}
    weighted = {m: n ** (1.0 / temperature) for m, n in counts.items()}
    z = sum(weighted.values())
    return {m: w / z for m, w in weighted.items()}


def sample_move(
    root: Node,
    temperature: float = 1.0,
    greedy: bool = False,
    rng: np.random.Generator | None = None,
) -> MoveType:
    """Sample the next move to play from the search tree.

    In self-play, early moves (typically the first ~30 plies) are
    sampled with ``temperature=1.0`` for diversity, and later moves
    use ``greedy=True`` for sharpness. This is independent of the
    policy training target, which is always recorded at τ=1.

    Args:
        root: Expanded search-tree root.
        temperature: Sampling temperature. Ignored when ``greedy=True``.
        greedy: If True, return the most-visited move (argmax). If ties
            exist, the earliest one encountered in iteration order is
            returned.
        rng: Optional NumPy RNG for reproducibility.
    """
    if not root.edges:
        raise RuntimeError("sample_move called on a root with no edges")

    if greedy:
        return max(root.edges.items(), key=lambda kv: kv[1].visit_count)[0]

    probs = policy_target(root, temperature=temperature)
    moves = list(probs.keys())
    weights = np.array([probs[m] for m in moves], dtype=np.float64)
    # Guard against tiny numerical drift making ``weights`` not sum to 1.
    weights = weights / weights.sum()
    rng = rng if rng is not None else np.random.default_rng()
    idx = rng.choice(len(moves), p=weights)
    return moves[idx]
