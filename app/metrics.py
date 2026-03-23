from __future__ import annotations

from typing import Dict, Iterable, List


def norm_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _lcs_length(ref_tokens: List[str], cand_tokens: List[str]) -> int:
    if not ref_tokens or not cand_tokens:
        return 0
    dp = [[0] * (len(cand_tokens) + 1) for _ in range(len(ref_tokens) + 1)]
    for i, ref_token in enumerate(ref_tokens, start=1):
        for j, cand_token in enumerate(cand_tokens, start=1):
            if ref_token == cand_token:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l_f1(reference: str, prediction: str) -> float:
    ref_tokens = norm_text(reference).lower().split()
    pred_tokens = norm_text(prediction).lower().split()
    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0
    lcs = _lcs_length(ref_tokens, pred_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def tok_f1(ref: str, cand: str) -> float:
    r = norm_text(ref).lower().split()
    c = norm_text(cand).lower().split()
    if not r and not c:
        return 1.0
    if not r or not c:
        return 0.0
    r_counts: Dict[str, int] = {}
    c_counts: Dict[str, int] = {}
    for token in r:
        r_counts[token] = r_counts.get(token, 0) + 1
    for token in c:
        c_counts[token] = c_counts.get(token, 0) + 1
    overlap = sum(min(r_counts.get(token, 0), c_counts.get(token, 0)) for token in set(r_counts) | set(c_counts))
    precision = overlap / max(1, len(c))
    recall = overlap / max(1, len(r))
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def ngram_precision(ref: str, cand: str, n: int = 1) -> float:
    r = norm_text(ref).lower().split()
    c = norm_text(cand).lower().split()
    if len(c) < n:
        return 0.0

    def ngrams(items: List[str], size: int) -> List[str]:
        return [" ".join(items[idx : idx + size]) for idx in range(len(items) - size + 1)]

    ref_set = set(ngrams(r, n))
    cand_grams = ngrams(c, n)
    if not cand_grams:
        return 0.0
    return sum(1 for gram in cand_grams if gram in ref_set) / len(cand_grams)


def compute_text_metrics(reference: str, prediction: str) -> Dict[str, float]:
    return {
        "rougeL_f": rouge_l_f1(reference, prediction),
        "tok_f1": tok_f1(reference, prediction),
        "uni_prec": ngram_precision(reference, prediction, 1),
        "bi_prec": ngram_precision(reference, prediction, 2),
    }


def mean_metric(rows: Iterable[Dict[str, float]], key: str) -> float:
    values = [row[key] for row in rows]
    return float(sum(values) / len(values)) if values else 0.0
