# metric.py
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import numpy as np
import string
import re
# Exact Match (EM)
def exact_match(prediction, ground_truth):
    return int(prediction == ground_truth)

# F1 Score
def f1(prediction, ground_truth):
    pred_tokens = word_tokenize(prediction.lower())
    truth_tokens = word_tokenize(ground_truth.lower())
    
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    
    return 2 * (precision * recall) / (precision + recall)

# BLEU Score
def bleu_score(prediction, ground_truth):
    pred_tokens = word_tokenize(prediction.lower())
    truth_tokens = word_tokenize(ground_truth.lower())
    return sentence_bleu([truth_tokens], pred_tokens)

# ROUGE Score
def rouge_score(prediction, ground_truth):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ground_truth, prediction)
    return scores

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
def f1_score(prediction, ground_truth):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)
def F1_scorer(predictions, answers):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        # for ground_truth in ground_truths:
            # score = max(score, qa_f1_score(prediction, ground_truth))
        # print(f'{prediction}--{ground_truth}\n')
        score = qa_f1_score(prediction, ground_truths)  # 只取一个 ground truth
        total_score += score
    return round(100 * total_score / len(predictions), 2)
# 计算数据集的平均分数
def compute_average_metrics(predictions, ground_truths):
    # 计算每个问题的分数
    exact_matches = [exact_match(pred, gt) for pred, gt in zip(predictions, ground_truths)]
    f1_scores = [F1_scorer(pred, gt) for pred, gt in zip(predictions, ground_truths)]
    bleu_scores = [bleu_score(pred, gt) for pred, gt in zip(predictions, ground_truths)]
    rouge_scores = [rouge_score(pred, gt) for pred, gt in zip(predictions, ground_truths)]

    # 计算数据集的平均分数
    em_avg = np.mean(exact_matches)
    f1_avg = np.mean(f1_scores)
    bleu_avg = np.mean(bleu_scores)
    rouge_avg = {key: np.mean([score[key].fmeasure for score in rouge_scores]) for key in rouge_scores[0]}

    return {
        'exact_match': em_avg,
        'f1': f1_avg,
        'bleu': bleu_avg,
        'rouge': rouge_avg
    }
