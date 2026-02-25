"""BARTScore — negative log-likelihood scoring with ParaBank2-finetuned weights.

Base architecture: facebook/bart-large-cnn (~1.6GB, auto-downloaded from HuggingFace).
Finetuned weights: ParaBank2 checkpoint (.pth), must be downloaded manually and pointed
to via the BARTSCORE_CHECKPOINT env var.

Download the checkpoint from: https://github.com/neulab/BARTScore
(Google Drive link in their README → save as e.g. bart_score.pth)

Scores are unidirectional (ref→hyp).
"""

from __future__ import annotations

import os
import traceback
from typing import List

import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartTokenizer

from metrics.model_cache import get_device, models

_MODEL_KEY = "bart_large_cnn"


class BARTScorer:
    """BARTScore scorer — ported from colleague's bart_score.py.

    Kept largely intact (including internal batching) for faithful reproduction.
    """

    def __init__(self, device: str = "cpu", max_length: int = 1024,
                 checkpoint: str = "facebook/bart-large-cnn") -> None:
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        self.loss_fct = nn.NLLLoss(
            reduction="none", ignore_index=self.model.config.pad_token_id
        )
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self, path: str) -> None:
        """Load finetuned weights (e.g. ParaBank2) on top of the base model."""
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )

    def score(self, srcs: List[str], tgts: List[str], batch_size: int = 4) -> List[float]:
        """Score a batch of (source, target) pairs. Returns negative NLL per pair."""
        score_list: List[float] = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    src_tokens = encoded_src["input_ids"].to(self.device)
                    src_mask = encoded_src["attention_mask"].to(self.device)

                    tgt_tokens = encoded_tgt["input_ids"].to(self.device)
                    tgt_mask = encoded_tgt["attention_mask"]
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens,
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                raise RuntimeError(
                    f"BARTScore forward pass failed. source={src_list}, target={tgt_list}"
                )
        return score_list


def _load_bart_scorer() -> BARTScorer:
    checkpoint_path = os.environ.get("BARTSCORE_CHECKPOINT")
    if not checkpoint_path:
        raise RuntimeError(
            "BARTSCORE_CHECKPOINT env var is not set. "
            "Download the ParaBank2-finetuned checkpoint from "
            "https://github.com/neulab/BARTScore and set "
            "BARTSCORE_CHECKPOINT=/path/to/bart_score.pth in your .env file."
        )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"BARTSCORE_CHECKPOINT points to '{checkpoint_path}' which does not exist."
        )
    device = get_device()
    scorer = BARTScorer(device=device, checkpoint="facebook/bart-large-cnn")
    scorer.load(checkpoint_path)
    return scorer


models.register_loader(_MODEL_KEY, _load_bart_scorer)


def calculate_bart_score(gt: str, hyp: str, **kwargs) -> float:
    """BARTScore: negative log-likelihood of hyp given gt (unidirectional, ref→hyp)."""
    if not gt or not hyp:
        return 0.0

    scorer: BARTScorer = models.get(_MODEL_KEY)
    scores = scorer.score([gt], [hyp])
    return scores[0]
