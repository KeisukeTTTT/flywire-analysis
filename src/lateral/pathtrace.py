"""Signed connectivity path tracing: mono- vs poly-synaptic inhibition.

The textbook definition of lateral inhibition is *interneuron-mediated* -- an
interneuron pools a neighborhood of feedforward cells and feeds inhibition back,
producing a center-surround field. So separating **direct** (monosynaptic) from
**mediated** (di-/poly-synaptic) inhibition is essential, not optional. This module
provides:

* :class:`RadialKernels` -- id-level spatial kernels in Δcolumn space, faithful
  reimplementations of ``direct_kernel`` and ``disyn_inh_kernel`` from
  ``notebook/lateral_inhibition_extended.py`` (the regression targets), built on the
  consolidated :mod:`src.lateral.hexgeom` geometry so any path order can be binned.
* :class:`SignedConnGraph` -- a compact type-level signed graph for topology:
  :meth:`~SignedConnGraph.trace_paths` enumerates signed paths of arbitrary length,
  generalizing the disynaptic exc->inh->T motif to e.g. inh->inh->T disinhibition.
* :func:`ffi_motifs` -- feedforward-inhibition census (reproduces ``ffi_for``).
* :func:`net_sign_to_target`, :func:`disinhibition_onto`, :func:`path_length_distribution`
  -- the new generalizations the headline metrics lacked.

The net sign of a path is the product of its edge signs (exc=+1, inh=-1); any path
through a modulatory ("other") edge has net sign 0 and is dropped from net-sign sums.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .hexgeom import hex_distance, interior_cells, pq_hemi_maps
from .nt_sign import SIGN_VALUE, add_sign


# --- id-level spatial kernels (Δcolumn) ----------------------------------------
@dataclass
class RadialKernels:
    """Δcolumn radial kernels around target cells (id-level, needs column coords).

    ``conn`` must carry ``sign`` (added if missing), ``syn_count``, ``pre_root_id``,
    ``post_root_id`` and ``pre_primary_type``. ``col_assign`` is the column table
    (see :func:`src.lateral.hexgeom.load_column_assignment`).

    NOTE -- :meth:`direct_kernel` (sign ``"inh"``) and :meth:`disyn_kernel` are **not a
    partition** of a target's inhibition. Both start from the same inhibitory partners
    ``J`` of ``T`` and reuse the same ``J->T`` weight; they differ only in where that
    weight is *placed* in Δcolumn space -- the direct kernel at ``J``'s own column, the
    disynaptic kernel redistributed across the columns of ``J``'s excitatory inputs.
    The same ``J->T`` mass therefore appears in both, so the two curves are meant for a
    **shape / RMS-radius (σ) comparison** ("direct is home-concentrated, mediated is
    broad"), not for adding their masses. (A columnar ``J`` contributes to both; a
    wide-field ``J`` with no column drops out of the direct kernel entirely yet still
    builds the mediated surround.)
    """

    conn: pd.DataFrame
    col_assign: pd.DataFrame
    P: dict
    Q: dict
    HM: dict

    @classmethod
    def from_data(cls, conn, col_assign):
        if "sign" not in conn.columns:
            conn = add_sign(conn)
        P, Q, HM = pq_hemi_maps(col_assign)
        return cls(conn=conn, col_assign=col_assign, P=P, Q=Q, HM=HM)

    def _target_home(self, target_type, hemi, min_nbrs=5):
        cells = interior_cells(self.col_assign, target_type, hemi, min_nbrs=min_nbrs)
        return cells, cells["p"].to_dict(), cells["q"].to_dict(), set(cells.index)

    def direct_kernel(self, target_type, sign, *, hemi="right", min_nbrs=5):
        """Monosynaptic input of a given ``sign`` onto ``target_type``, summed by
        Δcolumn (hex) distance between the source's column and the target's home
        column. Reproduces ``direct_kernel`` in lateral_inhibition_extended.py.
        """
        _, hp, hq, cids = self._target_home(target_type, hemi, min_nbrs)
        inc = self.conn[self.conn["post_root_id"].isin(cids) & (self.conn["sign"] == sign)].copy()
        inc["sp"] = inc["pre_root_id"].map(self.P)
        inc["sq"] = inc["pre_root_id"].map(self.Q)
        inc["sh"] = inc["pre_root_id"].map(self.HM)
        inc = inc.dropna(subset=["sp", "sq"])
        inc = inc[inc["sh"] == hemi]
        inc["d"] = hex_distance(
            inc["sp"] - inc["post_root_id"].map(hp),
            inc["sq"] - inc["post_root_id"].map(hq),
            integer=True,
        )
        return inc.groupby("d")["syn_count"].sum()

    def disyn_kernel(self, target_type, *, hemi="right", top_inh_types=12, min_nbrs=5):
        """Disynaptic inhibitory surround exc->inh(J)->T: the column footprint of the
        excitatory inputs that the inhibitory partners ``J`` of ``T`` pool, expressed
        relative to ``T``'s home column. Reproduces ``disyn_inh_kernel``.
        """
        _, hp, hq, cids = self._target_home(target_type, hemi, min_nbrs)
        inh_to_t = self.conn[
            self.conn["post_root_id"].isin(cids) & (self.conn["sign"] == "inh")
        ].copy()
        top_types = (
            inh_to_t.groupby("pre_primary_type")["syn_count"].sum()
            .sort_values(ascending=False)
            .head(top_inh_types)
            .index
        )
        inh_to_t = inh_to_t[inh_to_t["pre_primary_type"].isin(top_types)][
            ["pre_root_id", "post_root_id", "syn_count"]
        ].rename(columns={"syn_count": "w1"})
        J = set(inh_to_t["pre_root_id"])

        jin = self.conn[self.conn["post_root_id"].isin(J) & (self.conn["sign"] == "exc")].copy()
        jin["sp"] = jin["pre_root_id"].map(self.P)
        jin["sq"] = jin["pre_root_id"].map(self.Q)
        jin["sh"] = jin["pre_root_id"].map(self.HM)
        jin = jin.dropna(subset=["sp", "sq"])
        jin = jin[jin["sh"] == hemi]
        jin["w2n"] = jin["syn_count"] / jin.groupby("post_root_id")["syn_count"].transform("sum")
        jin = jin.rename(columns={"post_root_id": "j"})[["j", "sp", "sq", "w2n"]]

        mg = inh_to_t.merge(jin, left_on="pre_root_id", right_on="j", how="inner")
        mg["d"] = hex_distance(
            mg["sp"] - mg["post_root_id"].map(hp),
            mg["sq"] - mg["post_root_id"].map(hq),
            integer=True,
        )
        mg["w"] = mg["w1"] * mg["w2n"]
        return mg.groupby("d")["w"].sum()


def rms_radius(kern, maxd=8):
    """RMS radius ``sigma = sqrt(sum f(d) d^2 / sum f(d))`` of a Δcolumn kernel.

    Returns ``(sigma, normalized_series)`` over ``d`` in ``[0, maxd]``. Reproduces
    ``rms_radius`` in lateral_inhibition_extended.py.
    """
    s = kern.reindex(range(0, maxd + 1), fill_value=0).astype(float)
    s = s / s.sum() if s.sum() > 0 else s
    d = np.arange(0, maxd + 1)
    return float(np.sqrt((s.values * d ** 2).sum())), s


# --- type-level signed graph & generic path tracing ----------------------------
class SignedConnGraph:
    """Compact type-level signed connectivity graph.

    Aggregates a connection table to ``(pre_type, post_type, sign) -> summed weight``.
    Small (~10^3 node types) so multi-hop signed path enumeration is cheap and exact.
    """

    def __init__(
        self,
        conn,
        *,
        pre_col="pre_primary_type",
        post_col="post_primary_type",
        sign_col="sign",
        weight_col="syn_count",
    ):
        if sign_col not in conn.columns:
            conn = add_sign(conn)
        e = (
            conn.dropna(subset=[pre_col, post_col])
            .groupby([pre_col, post_col, sign_col])[weight_col]
            .sum()
            .reset_index()
        )
        e.columns = ["pre", "post", "sign", "w"]
        self.edges = e

    def trace_paths(self, sign_pattern, *, targets=None, sources=None, min_w=0.0):
        """Enumerate type-paths whose successive edge signs equal ``sign_pattern``.

        ``sign_pattern`` is ordered source -> ... -> target (e.g. ``["exc", "inh"]``
        for a disynaptic feedforward-inhibition surround, ``["inh", "inh"]`` for
        disinhibition). Returns a tidy frame with columns ``n0..nL`` (the type chain),
        ``w`` (product of edge weights), ``net_sign`` (product of edge signs) and
        ``length``. ``targets`` / ``sources`` optionally restrict the last / first node.
        """
        if not sign_pattern:
            raise ValueError("sign_pattern must have at least one sign")
        first = self.edges[self.edges["sign"] == sign_pattern[0]]
        if sources is not None:
            first = first[first["pre"].isin(set(sources))]
        path = first.rename(columns={"pre": "n0", "post": "n1", "w": "w"})[["n0", "n1", "w"]].copy()
        for k, sign in enumerate(sign_pattern[1:], start=1):
            nxt = self.edges[self.edges["sign"] == sign][["pre", "post", "w"]].rename(
                columns={"pre": f"n{k}", "post": f"n{k+1}", "w": "w_next"}
            )
            path = path.merge(nxt, on=f"n{k}", how="inner")
            path["w"] = path["w"] * path["w_next"]
            path = path.drop(columns=["w_next"])
        last = f"n{len(sign_pattern)}"
        if targets is not None:
            path = path[path[last].isin(set(targets))]
        if min_w:
            path = path[path["w"] >= min_w]
        path["net_sign"] = int(np.prod([SIGN_VALUE[s] for s in sign_pattern]))
        path["length"] = len(sign_pattern)
        return path.reset_index(drop=True)

    def signed_patterns(self, length, signs=("exc", "inh")):
        """All sign patterns of a given ``length`` over ``signs``."""
        return [list(p) for p in itertools.product(signs, repeat=length)]


def net_sign_to_target(graph, targets, *, max_len=2, signs=("exc", "inh")):
    """Net excitatory vs inhibitory drive reaching each target, by path length.

    Sums signed path weight over every sign pattern up to ``max_len`` (excluding
    modulatory edges). Returned tidy per ``(target, length, net_sign)`` so different
    hop counts -- whose raw weight magnitudes are not comparable -- are not collapsed
    together. Use it as a topology census (how much excitatory vs inhibitory, direct
    vs mediated, structure converges on a target), not as a magnitude estimate.
    """
    tset = set(targets)
    rows = []
    for length in range(1, max_len + 1):
        for pattern in graph.signed_patterns(length, signs=signs):
            paths = graph.trace_paths(pattern, targets=tset)
            if paths.empty:
                continue
            last = f"n{length}"
            grp = paths.groupby(last).agg(total_w=("w", "sum"), n_paths=("w", "size"))
            net = int(np.prod([SIGN_VALUE[s] for s in pattern]))
            for tgt, r in grp.iterrows():
                rows.append(
                    {
                        "target": tgt,
                        "length": length,
                        "net_sign": net,
                        "pattern": "->".join(pattern),
                        "total_w": float(r["total_w"]),
                        "n_paths": int(r["n_paths"]),
                    }
                )
    return pd.DataFrame(rows)


def disinhibition_onto(graph, targets):
    """Disynaptic disinhibition (inh->inh->T) reaching each target: net-positive."""
    paths = graph.trace_paths(["inh", "inh"], targets=targets)
    if paths.empty:
        return paths
    return (
        paths.groupby("n2")
        .agg(total_w=("w", "sum"), n_paths=("w", "size"))
        .reset_index()
        .rename(columns={"n2": "target"})
        .sort_values("total_w", ascending=False)
    )


def path_length_distribution(graph, targets, *, max_len=3, signs=("exc", "inh")):
    """For each target, the number of distinct source types whose net drive is
    excitatory vs inhibitory at each path length -- the signed reachability profile.
    """
    rows = []
    tset = set(targets)
    for length in range(1, max_len + 1):
        for pattern in graph.signed_patterns(length, signs=signs):
            paths = graph.trace_paths(pattern, targets=tset)
            if paths.empty:
                continue
            net = int(np.prod([SIGN_VALUE[s] for s in pattern]))
            last = f"n{length}"
            grp = paths.groupby(last)["n0"].nunique()
            for tgt, n_src in grp.items():
                rows.append({"target": tgt, "length": length, "net_sign": net, "n_source_types": int(n_src)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return (
        out.groupby(["target", "length", "net_sign"])["n_source_types"].sum().reset_index()
    )


def ffi_motifs(conn, targets, *, topn=6, min_xz=200):
    """Feedforward-inhibition census: common driver X -> inhibitor Z -> target T.

    Reproduces ``ffi_for`` from lateral_inhibition_extended.py at the type level: for
    each target, the top excitatory drivers X and top inhibitory sources Z, kept when
    X also excites Z with ``>= min_xz`` synapses. ``targets`` may be a single type or
    an iterable.
    """
    if "sign" not in conn.columns:
        conn = add_sign(conn)
    if isinstance(targets, str):
        targets = [targets]
    exc_te = (
        conn[conn["sign"] == "exc"].groupby(["pre_primary_type", "post_primary_type"])["syn_count"].sum()
    )
    frames = []
    for target in targets:
        inc = conn[conn["post_primary_type"] == target]
        exc_drv = (
            inc[inc["sign"] == "exc"].groupby("pre_primary_type")["syn_count"].sum()
            .sort_values(ascending=False).head(topn)
        )
        inh_src = (
            inc[inc["sign"] == "inh"].groupby("pre_primary_type")["syn_count"].sum()
            .sort_values(ascending=False).head(topn)
        )
        rows = []
        for x in exc_drv.index:
            for z in inh_src.index:
                xz = exc_te.get((x, z), 0)
                if xz >= min_xz:
                    rows.append(
                        dict(
                            target=target,
                            driver_X=x,
                            inhibitor_Z=z,
                            X_to_Z_exc=int(xz),
                            Z_to_T_inh=int(inh_src[z]),
                        )
                    )
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame(columns=["target", "driver_X", "inhibitor_Z", "X_to_Z_exc", "Z_to_T_inh"])
    return pd.concat(frames, ignore_index=True).sort_values("X_to_Z_exc", ascending=False)
