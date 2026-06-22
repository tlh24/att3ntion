from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


_WS_RE = re.compile(r"\s+")
_AND_RE = re.compile(r"\s+AND\s+")


def _normalize_spaces(text: str) -> str:
    return _WS_RE.sub(" ", text.strip())


def _canonicalize_variables(lf: str) -> str:
    """
    Renumber variable indices by first occurrence.
    ReCOGS logical forms use integer variable ids in declarations/relations.
    """
    tokens = _normalize_spaces(lf).split(" ")
    remap: dict[str, str] = {}
    next_id = 0
    out: list[str] = []
    for tok in tokens:
        if tok.isdigit():
            if tok not in remap:
                remap[tok] = str(next_id)
                next_id += 1
            out.append(remap[tok])
        else:
            out.append(tok)
    return " ".join(out)


def _split_atoms(lf: str) -> list[str]:
    """
    Split LF into atomic facts, invariant to ';' and 'AND' ordering.
    """
    lf = _normalize_spaces(lf)
    if not lf:
        return []

    atoms: list[str] = []
    semicolon_parts = [p.strip() for p in lf.split(";") if p.strip()]
    for part in semicolon_parts:
        and_parts = [a.strip() for a in _AND_RE.split(part) if a.strip()]
        atoms.extend(and_parts)
    return atoms


def canonical_atoms(lf: str) -> list[str]:
    canon = _canonicalize_variables(lf)
    atoms = _split_atoms(canon)
    return sorted(_normalize_spaces(a) for a in atoms)


def _parse_atoms(lf: str) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Parse an LF into (shape, variables) per atom.

    ``shape`` is the atom's token sequence with every variable id replaced by
    ``#`` (so ``agent ( 1 , 0 )`` -> ``('agent','(','#',',','#',')')``); ``variables``
    is the ordered tuple of the original variable ids in that atom. Two atoms can
    only correspond if their shapes are identical and a consistent variable
    renaming maps one's variables onto the other's.
    """
    atoms: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for atom in _split_atoms(_normalize_spaces(lf)):
        toks = atom.split(" ")
        shape = tuple("#" if t.isdigit() else t for t in toks)
        variables = tuple(t for t in toks if t.isdigit())
        atoms.append((shape, variables))
    return atoms


def semantic_exact_match(pred_lf: str, gold_lf: str) -> bool:
    """ReCOGS semantic exact match.

    Two logical forms match iff their atom sets are equal under some *bijection*
    between their variable ids — i.e. invariant to both conjunct ordering and
    variable renaming. We find such a bijection with a backtracking search that
    matches pred atoms to gold atoms, picking the most-constrained pred atom
    first (MRV) so connected ReCOGS structures resolve with little branching.
    """
    pred = _parse_atoms(pred_lf)
    gold = _parse_atoms(gold_lf)
    if len(pred) != len(gold):
        return False
    # Cheap reject: atom shape multisets must agree before any variable matching.
    if Counter(s for s, _ in pred) != Counter(s for s, _ in gold):
        return False

    n = len(gold)
    used = [False] * n
    matched = [False] * len(pred)
    fwd: dict[str, str] = {}  # pred var -> gold var
    bwd: dict[str, str] = {}  # gold var -> pred var

    def candidates(p_shape: tuple[str, ...], p_vars: tuple[str, ...]) -> list[int]:
        res: list[int] = []
        for j in range(n):
            if used[j]:
                continue
            g_shape, g_vars = gold[j]
            if g_shape != p_shape:
                continue
            ok = True
            for a, b in zip(p_vars, g_vars):
                fa = fwd.get(a)
                if fa is not None and fa != b:
                    ok = False
                    break
                rb = bwd.get(b)
                if rb is not None and rb != a:
                    ok = False
                    break
            if ok:
                res.append(j)
        return res

    def backtrack(count: int) -> bool:
        if count == len(pred):
            return True
        # Most-constrained-variable: pick the unmatched pred atom with the
        # fewest viable gold candidates; bail immediately on a dead end.
        best_i = -1
        best_cands: list[int] | None = None
        for i in range(len(pred)):
            if matched[i]:
                continue
            cands = candidates(*pred[i])
            if not cands:
                return False
            if best_cands is None or len(cands) < len(best_cands):
                best_i, best_cands = i, cands
                if len(cands) == 1:
                    break

        i = best_i
        _, p_vars = pred[i]
        matched[i] = True
        for j in best_cands or []:
            _, g_vars = gold[j]
            added: list[tuple[str, str]] = []
            for a, b in zip(p_vars, g_vars):
                if a not in fwd:
                    fwd[a] = b
                    bwd[b] = a
                    added.append((a, b))
            used[j] = True
            if backtrack(count + 1):
                return True
            used[j] = False
            for a, b in added:
                del fwd[a]
                del bwd[b]
        matched[i] = False
        return False

    return backtrack(0)


@dataclass(frozen=True)
class ScoreResult:
    exact_match: float
    total: int
    correct: int


def score_predictions(preds: list[str], golds: list[str]) -> ScoreResult:
    if len(preds) != len(golds):
        raise ValueError(f"preds and golds must have same length, got {len(preds)} vs {len(golds)}")
    total = len(preds)
    correct = sum(semantic_exact_match(p, g) for p, g in zip(preds, golds))
    exact_match = 0.0 if total == 0 else correct / total
    return ScoreResult(exact_match=exact_match, total=total, correct=correct)

