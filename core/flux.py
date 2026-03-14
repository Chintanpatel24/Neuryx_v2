"""
core/flux.py
Reverse-mode automatic differentiation over scalar computation graphs.

Public surface: Flux — the only class you need.

Key fix vs previous versions
─────────────────────────────
The backward pass (diffuse) uses a fully ITERATIVE topological sort with
an explicit Python stack instead of recursive calls.  The recursive version
raises RecursionError on deep transformer graphs (4+ blocks, dim >= 64)
because Python's default call-stack limit is ~1000 frames and a single
training step builds graphs with tens of thousands of nodes.
"""

import math as _math


class Flux:
    """
    A single node in a directed-acyclic computation graph.

    Attributes
    ----------
    val   : float  — forward-pass scalar value
    delta : float  — accumulated gradient (∂loss / ∂self), filled by diffuse()
    src   : tuple  — parent Flux nodes that produced this node
    jac   : tuple  — local partial derivatives  ∂self/∂src[i]
    """

    __slots__ = ("val", "delta", "src", "jac")

    def __init__(self, val, src=(), jac=()):
        self.val   = float(val)
        self.delta = 0.0
        self.src   = src
        self.jac   = jac

    # ── Arithmetic ────────────────────────────────────────────────────────────

    def __add__(self, rhs):
        rhs = rhs if isinstance(rhs, Flux) else Flux(rhs)
        return Flux(self.val + rhs.val, (self, rhs), (1.0, 1.0))

    def __mul__(self, rhs):
        rhs = rhs if isinstance(rhs, Flux) else Flux(rhs)
        return Flux(self.val * rhs.val, (self, rhs), (rhs.val, self.val))

    def __pow__(self, exp):
        out_val = self.val ** exp
        slope   = exp * self.val ** (exp - 1)
        return Flux(out_val, (self,), (slope,))

    # ── Activations ───────────────────────────────────────────────────────────

    def loge(self):
        safe = max(self.val, 1e-12)
        return Flux(_math.log(safe), (self,), (1.0 / safe,))

    def expe(self):
        clamped = min(self.val, 88.72)
        result  = _math.exp(clamped)
        return Flux(result, (self,), (result,))

    def thresh(self):                          # ReLU
        return Flux(max(0.0, self.val), (self,), (float(self.val > 0),))

    def swish(self):                           # x · sigmoid(x)
        sig = 1.0 / (1.0 + _math.exp(-self.val))
        return Flux(self.val * sig, (self,), (sig + self.val * sig * (1.0 - sig),))

    # ── Operator aliases ──────────────────────────────────────────────────────

    def __neg__(self):           return self * -1
    def __radd__(self, lhs):     return self + lhs
    def __sub__(self, rhs):      return self + (-rhs)
    def __rsub__(self, lhs):     return Flux(lhs) + (-self)
    def __rmul__(self, lhs):     return self * lhs
    def __truediv__(self, rhs):  return self * rhs ** -1
    def __rtruediv__(self, lhs): return Flux(lhs) * self ** -1

    def __repr__(self):
        return f"Flux(val={self.val:.6g}, delta={self.delta:.6g})"

    # ── Backward pass — ITERATIVE (no recursion, no stack-overflow) ───────────

    def diffuse(self):
        """
        Back-propagate gradients through the full computation graph.

        Uses an iterative post-order DFS with an explicit stack instead of
        Python recursion.  This is critical: transformer graphs built for a
        single training step can be 50,000+ nodes deep; recursive traversal
        would immediately hit Python's ~1000-frame call-stack limit and raise
        RecursionError.

        Algorithm
        ---------
        1. Iterative post-order DFS  →  topological order list
        2. Seed self.delta = 1.0
        3. Walk in reverse topo order, accumulate chain-rule gradients
        """
        # ── Step 1: iterative post-order topological sort ──────────────────
        order: list["Flux"] = []
        seen:  set[int]     = set()

        # Stack entries: (node, parents_pushed)
        # parents_pushed=False  → first visit; push children, then re-push self
        # parents_pushed=True   → all parents done; append self to order
        stack: list[tuple["Flux", bool]] = [(self, False)]

        while stack:
            node, parents_pushed = stack.pop()
            nid = id(node)

            if parents_pushed:
                # All parents already in order — safe to add this node
                order.append(node)
                continue

            if nid in seen:
                continue
            seen.add(nid)

            # Re-push self with flag=True so it gets appended after its parents
            stack.append((node, True))

            # Push parents (unseen only) — they need to come BEFORE node in order
            for parent in node.src:
                if id(parent) not in seen:
                    stack.append((parent, False))

        # ── Step 2: seed gradient at the loss node ─────────────────────────
        self.delta = 1.0

        # ── Step 3: reverse-mode gradient accumulation ────────────────────
        for node in reversed(order):
            for parent, local_slope in zip(node.src, node.jac):
                parent.delta += local_slope * node.delta
