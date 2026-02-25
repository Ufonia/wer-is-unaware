"""BERTScore utility functions — tokenisation, embedding, greedy matching."""

from __future__ import annotations

import sys
from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer

# Layer counts tuned on WMT16 correlation data — only need the one we use.
model2layers = {
    "roberta-large": 17,
}


def get_tokenizer(model_type: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_type, use_fast=False, add_prefix_space=True)


def get_model(model_type: str, num_layers: int) -> AutoModel:
    """Load model and truncate to num_layers."""
    model = AutoModel.from_pretrained(model_type)
    model.eval()

    assert hasattr(model, "encoder") and hasattr(model.encoder, "layer"), (
        f"Expected model.encoder.layer for {model_type}"
    )
    assert 0 <= num_layers <= len(model.encoder.layer), (
        f"Invalid num_layers={num_layers} for {model_type} "
        f"(max {len(model.encoder.layer)})"
    )
    model.encoder.layer = torch.nn.ModuleList(
        list(model.encoder.layer[:num_layers])
    )
    return model


def sent_encode(tokenizer: AutoTokenizer, sent: str) -> list[int]:
    """Encode a sentence with special tokens, truncated to model max length."""
    sent = sent.strip()
    if sent == "":
        return tokenizer.build_inputs_with_special_tokens([])
    max_len = min(tokenizer.model_max_length, 512)
    return tokenizer.encode(
        sent,
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
    )


def padding(arr: list[list[int]], pad_token: int, dtype=torch.long):
    """Pad a list of token-id lists to the same length."""
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, : lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, : lens[i]] = 1
    return padded, lens, mask


def collate_idf(
    arr: list[str],
    tokenizer: AutoTokenizer,
    idf_dict: dict,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenise, pad, and build IDF weight tensors."""
    arr_encoded = [sent_encode(tokenizer, a) for a in arr]
    idf_weights = [[idf_dict[i] for i in a] for a in arr_encoded]

    pad_token = tokenizer.pad_token_id
    padded, lens, mask = padding(arr_encoded, pad_token, dtype=torch.long)
    padded_idf, _, _ = padding(idf_weights, 0, dtype=torch.float)

    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lens = lens.to(device=device)
    return padded, padded_idf, lens, mask


def bert_encode(model: AutoModel, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Forward pass through the model, return last hidden state."""
    model.eval()
    with torch.no_grad():
        out = model(x, attention_mask=attention_mask, output_hidden_states=False)
    return out[0]


def get_bert_embedding(
    all_sens: list[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    idf_dict: dict,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute BERT embeddings in batches."""
    padded_sens, padded_idf, lens, mask = collate_idf(
        all_sens, tokenizer, idf_dict, device=device,
    )

    if batch_size == -1:
        batch_size = len(all_sens)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_sens), batch_size):
            batch_embedding = bert_encode(
                model,
                padded_sens[i: i + batch_size],
                attention_mask=mask[i: i + batch_size],
            )
            embeddings.append(batch_embedding)
            del batch_embedding

    total_embedding = torch.cat(embeddings, dim=0)
    return total_embedding, mask, padded_idf


def greedy_cos_idf(
    ref_embedding: torch.Tensor,
    ref_masks: torch.Tensor,
    ref_idf: torch.Tensor,
    ref_cls: torch.Tensor,
    hyp_embedding: torch.Tensor,
    hyp_masks: torch.Tensor,
    hyp_idf: torch.Tensor,
    hyp_cls: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedy cosine matching → Precision, Recall, F1."""
    ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

    batch_size = ref_embedding.size(0)
    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
    masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim)
    masks = masks.float().to(sim.device)
    sim = sim * masks

    word_precision = sim.max(dim=2)[0]
    word_recall = sim.max(dim=1)[0]

    hyp_idf.div_(hyp_idf.sum(dim=1, keepdim=True))
    ref_idf.div_(ref_idf.sum(dim=1, keepdim=True))
    precision_scale = hyp_idf.to(word_precision.device)
    recall_scale = ref_idf.to(word_recall.device)

    P = (word_precision * precision_scale).sum(dim=1)
    R = (word_recall * recall_scale).sum(dim=1)
    F = 2 * P * R / (P + R)

    hyp_zero_mask = hyp_masks.sum(dim=1).eq(0)
    ref_zero_mask = ref_masks.sum(dim=1).eq(0)

    if torch.any(hyp_zero_mask):
        print(
            "Warning: Empty candidate sentence detected; setting raw BERTscores to 0.",
            file=sys.stderr,
        )
        P = P.masked_fill(hyp_zero_mask, 0.0)
        R = R.masked_fill(hyp_zero_mask, 0.0)

    if torch.any(ref_zero_mask):
        print(
            "Warning: Empty reference sentence detected; setting raw BERTScores to 0.",
            file=sys.stderr,
        )
        P = P.masked_fill(ref_zero_mask, 0.0)
        R = R.masked_fill(ref_zero_mask, 0.0)

    F = F.masked_fill(torch.isnan(F), 0.0)
    return P, R, F


def bert_cos_score_idf(
    model: AutoModel,
    refs: list[str],
    hyps: list[str],
    tokenizer: AutoTokenizer,
    idf_dict: dict,
    device: str,
    batch_size: int = 64,
) -> torch.Tensor:
    """Compute BERTScore (P, R, F1) for lists of refs and hyps."""
    preds = []

    def dedup_and_sort(lst: list[str]) -> list[str]:
        return sorted(set(lst), key=lambda x: len(x.split(" ")), reverse=True)

    sentences = dedup_and_sort(refs + hyps)

    # Embed all unique sentences
    stats_dict: dict = {}
    for batch_start in range(0, len(sentences), batch_size):
        sen_batch = sentences[batch_start: batch_start + batch_size]
        embs, masks, padded_idf = get_bert_embedding(
            sen_batch, model, tokenizer, idf_dict,
            batch_size=batch_size, device=device,
        )
        embs = embs.cpu()
        masks = masks.cpu()
        padded_idf = padded_idf.cpu()

        for i, sen in enumerate(sen_batch):
            sequence_len = masks[i].sum().item()
            emb = embs[i, :sequence_len]
            idf = padded_idf[i, :sequence_len]
            # Strip [CLS] and [SEP] tokens
            stats_dict[sen] = (emb[1:-1], idf[1:-1], emb[0])

    def pad_batch_stats(sen_batch: list[str], stats: dict, dev: str):
        batch_stats = [stats[s] for s in sen_batch]
        emb, idf, cls_token = zip(*batch_stats)
        cls_token = [c.to(dev) for c in cls_token]
        cls_token = pad_sequence(cls_token, batch_first=True)
        emb = [e.to(dev) for e in emb]
        idf = [i.to(dev) for i in idf]
        lens = [e.size(0) for e in emb]
        emb_pad = pad_sequence(emb, batch_first=True, padding_value=2.0)
        idf_pad = pad_sequence(idf, batch_first=True)

        def length_to_mask(lens_list):
            lens_t = torch.tensor(lens_list, dtype=torch.long)
            max_len = max(lens_list)
            base = torch.arange(max_len, dtype=torch.long).expand(len(lens_list), max_len)
            return base < lens_t.unsqueeze(1)

        pad_mask = length_to_mask(lens).to(dev)
        return emb_pad, pad_mask, idf_pad, cls_token

    device = next(model.parameters()).device
    device_str = str(device)

    with torch.no_grad():
        for batch_start in range(0, len(refs), batch_size):
            batch_refs = refs[batch_start: batch_start + batch_size]
            batch_hyps = hyps[batch_start: batch_start + batch_size]
            ref_stats = pad_batch_stats(batch_refs, stats_dict, device_str)
            hyp_stats = pad_batch_stats(batch_hyps, stats_dict, device_str)
            P, R, F1 = greedy_cos_idf(*ref_stats, *hyp_stats)
            preds.append(torch.stack((P, R, F1), dim=-1).cpu())

    preds = torch.cat(preds, dim=0)
    return preds
