"""N-gram overlap metrics: BLEU-1/2/3/4, ROUGE-1/2/L/W, ChrF, ChrF++, METEOR.

Each function expects **pre-cleaned** text (the public API handles cleaning).
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score as _nltk_meteor
from rouge_score import rouge_scorer
import sacrebleu

logger = logging.getLogger(__name__)

# NLTK's BLEU warning message starts with \n, so use [\s\S] to match across newlines.
warnings.filterwarnings("ignore", message=r"[\s\S]*hypothesis contains 0 counts[\s\S]*")


def _calculate_bleu(gt: str, hyp: str, weights: tuple) -> Optional[float]:
    if not gt and not hyp:
        return None
    if not gt or not hyp:
        return 0.0
    gt_tokens = [gt.split()]
    hyp_tokens = hyp.split()
    if not hyp_tokens:
        return 0.0
    return sentence_bleu(gt_tokens, hyp_tokens, weights=weights)


def calculate_bleu_1(gt: str, hyp: str) -> Optional[float]:
    return _calculate_bleu(gt, hyp, weights=(1.0,))


def calculate_bleu_2(gt: str, hyp: str) -> Optional[float]:
    return _calculate_bleu(gt, hyp, weights=(0.5, 0.5))


def calculate_bleu_3(gt: str, hyp: str) -> Optional[float]:
    return _calculate_bleu(gt, hyp, weights=(1 / 3, 1 / 3, 1 / 3))


def calculate_bleu_4(gt: str, hyp: str) -> Optional[float]:
    return _calculate_bleu(gt, hyp, weights=(0.25, 0.25, 0.25, 0.25))


def calculate_rouge_1(gt: str, hyp: str) -> Optional[float]:
    if not gt and not hyp:
        return None
    if not gt or not hyp:
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    return scorer.score(gt, hyp)["rouge1"].fmeasure


def calculate_rouge_2(gt: str, hyp: str) -> Optional[float]:
    if not gt and not hyp:
        return None
    if not gt or not hyp:
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)
    return scorer.score(gt, hyp)["rouge2"].fmeasure


def calculate_rouge_l(gt: str, hyp: str) -> Optional[float]:
    if not gt and not hyp:
        return None
    if not gt or not hyp:
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(gt, hyp)["rougeL"].fmeasure


def calculate_rouge_w(gt: str, hyp: str) -> Optional[float]:
    """ROUGE-W (weighted LCS). Uses rougeLsum as proxy (rouge-score library)."""
    if not gt and not hyp:
        return None
    if not gt or not hyp:
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
    return scorer.score(gt, hyp)["rougeLsum"].fmeasure


def calculate_chrf(gt: str, hyp: str) -> Optional[float]:
    if not gt and not hyp:
        return None
    if not gt or not hyp:
        return 0.0
    try:
        score = sacrebleu.sentence_chrf(hyp, [gt])
        return score.score / 100.0
    except Exception as e:
        logger.warning(f"chrF calculation failed: {e}")
        return None


def calculate_chrf_plus_plus(gt: str, hyp: str) -> Optional[float]:
    if not gt and not hyp:
        return None
    if not gt or not hyp:
        return 0.0
    try:
        score = sacrebleu.sentence_chrf(hyp, [gt], word_order=2)
        return score.score / 100.0
    except Exception as e:
        logger.warning(f"chrF++ calculation failed: {e}")
        return None


def calculate_meteor(gt: str, hyp: str) -> Optional[float]:
    if not gt and not hyp:
        return None
    if not gt or not hyp:
        return 0.0
    try:
        gt_tokens = gt.split()
        hyp_tokens = hyp.split()
        if not gt_tokens or not hyp_tokens:
            return 0.0
        return _nltk_meteor([gt_tokens], hyp_tokens)
    except Exception as e:
        logger.warning(f"METEOR calculation failed: {e}")
        return None
