from pathlib import Path
import sys

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECOGS_DIR = PROJECT_ROOT / "experiments" / "recogs"
if str(RECOGS_DIR) not in sys.path:
    sys.path.insert(0, str(RECOGS_DIR))

from metrics import semantic_exact_match
from evaluate import greedy_decode_one, greedy_decode_batch
from model import RecogsDecoderLM


# Prompts of varying length so batched decoding exercises ragged prefixes (the
# short rows carry masked pad columns the long rows do not).
_PROMPTS = [[1, 5, 2], [1, 7, 8, 9, 2], [1, 2], [1, 4, 4, 4, 4, 2]]
_EOS_ID, _PAD_ID = 3, 0


def _assert_batch_matches_single(attn_impl: str, device: torch.device) -> None:
    torch.manual_seed(0)
    vocab, d_model, heads, layers, msl = 40, 256, 4, 2, 64
    model = RecogsDecoderLM(vocab, d_model, heads, layers, attn_impl, msl).to(device).eval()
    for prefix_lm in (False, True):
        singles = [
            greedy_decode_one(model, p, _EOS_ID, 12, device, prefix_lm=prefix_lm)
            for p in _PROMPTS
        ]
        batched = greedy_decode_batch(
            model, _PROMPTS, _EOS_ID, _PAD_ID, 12, device, prefix_lm=prefix_lm
        )
        assert batched == singles, f"{attn_impl} prefix_lm={prefix_lm}: {batched} != {singles}"


def test_greedy_decode_batch_matches_single_graph_flash():
    """Batched decode must reproduce per-example greedy decode token-for-token
    (graph_flash, CPU). This gates the eval-speedup rewrite."""
    _assert_batch_matches_single("graph_flash", torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required (hypergraph kernel)")
def test_greedy_decode_batch_matches_single_hypergraph_cuda():
    """Same equivalence for the O(N^3) hypergraph CUDA arm."""
    _assert_batch_matches_single("hypergraph_cuda", torch.device("cuda"))


def test_semantic_exact_match_identical():
    gold = "Emma ( 0 ) ; dog ( 2 ) ; help ( 1 ) AND agent ( 1 , 0 ) AND theme ( 1 , 2 )"
    pred = "Emma ( 0 ) ; dog ( 2 ) ; help ( 1 ) AND agent ( 1 , 0 ) AND theme ( 1 , 2 )"
    assert semantic_exact_match(pred, gold)


def test_semantic_exact_match_reordered_and_renamed():
    gold = "Emma ( 0 ) ; dog ( 2 ) ; help ( 1 ) AND agent ( 1 , 0 ) AND theme ( 1 , 2 )"
    pred = "dog ( 8 ) ; Emma ( 7 ) ; help ( 4 ) AND theme ( 4 , 8 ) AND agent ( 4 , 7 )"
    assert semantic_exact_match(pred, gold)


def test_semantic_exact_match_mismatch():
    gold = "Emma ( 0 ) ; dog ( 2 ) ; help ( 1 ) AND agent ( 1 , 0 ) AND theme ( 1 , 2 )"
    pred = "Emma ( 0 ) ; dog ( 2 ) ; help ( 1 ) AND agent ( 1 , 2 ) AND theme ( 1 , 0 )"
    assert not semantic_exact_match(pred, gold)

