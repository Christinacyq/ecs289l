import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from collections import Counter
import math
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor import Meteor
from pycocoevalcap.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

# Download required NLTK data
nltk.download('punkt', quiet=True)

def compute_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation
    :param gts: Dictionary with the image ids and their gold captions
    :param res: Dictionary with the image ids and their generated captions
    :return: Dictionary with the scores
    """
    # Check if gts or res is empty
    if not gts or not res:
        print("Warning: Empty ground truth or results")
        return {
            'BLEU_1': 0.0,
            'BLEU_2': 0.0,
            'BLEU_3': 0.0,
            'BLEU_4': 0.0,
            'METEOR': 0.0,
            'ROUGE_L': 0.0,
            'CIDEr': 0.0
        }

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res)
            if isinstance(method, list):
                for sc, m in zip(score, method):
                    eval_res[m] = sc
            else:
                eval_res[method] = score
        except Exception as e:
            print(f"Warning: Error computing {method} score: {str(e)}")
            if isinstance(method, list):
                for m in method:
                    eval_res[m] = 0.0
            else:
                eval_res[method] = 0.0
    
    return eval_res

def compute_metrics(preds, targets):
    """
    Compute metrics for contrastive learning evaluation.
    Args:
        preds: Tensor of shape (batch_size,) containing predicted indices
        targets: Tensor of shape (batch_size,) containing target indices
    Returns:
        Dictionary of metrics
    """
    # Convert tensors to numpy arrays
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Compute accuracy
    accuracy = (preds == targets).mean()
    
    # Compute top-k accuracy
    k = 5
    top_k_accuracy = sum(1 for p, t in zip(preds, targets) if abs(p - t) < k) / len(preds)
    
    return {
        'accuracy': accuracy,
        f'top_{k}_accuracy': top_k_accuracy
    }

def calculate_bleu(preds, targets, n=1):
    """
    Calculate BLEU score for n-grams
    """
    smoothie = SmoothingFunction().method1
    scores = []
    for pred, target in zip(preds, targets):
        score = sentence_bleu([target], pred, weights=tuple([1.0/n] * n), smoothing_function=smoothie)
        scores.append(score)
    return np.mean(scores)

def calculate_cider(preds, targets, n=4):
    """
    Calculate CIDEr score
    """
    def compute_tfidf(sentences):
        # Compute term frequency
        tf = Counter()
        for sent in sentences:
            tf.update(sent)
        
        # Compute IDF
        N = len(sentences)
        idf = {}
        for term in tf:
            df = sum(1 for sent in sentences if term in sent)
            idf[term] = math.log((N + 1.0) / (df + 1.0))
        
        return tf, idf
    
    def compute_cider_score(pred, target, tf, idf):
        # Compute TF-IDF vectors
        pred_vec = np.zeros(len(tf))
        target_vec = np.zeros(len(tf))
        
        for i, term in enumerate(tf):
            pred_vec[i] = pred.count(term) * idf[term]
            target_vec[i] = target.count(term) * idf[term]
        
        # Compute cosine similarity
        norm_pred = np.linalg.norm(pred_vec)
        norm_target = np.linalg.norm(target_vec)
        if norm_pred == 0 or norm_target == 0:
            return 0
        return np.dot(pred_vec, target_vec) / (norm_pred * norm_target)
    
    # Compute TF-IDF for all sentences
    all_sentences = preds + targets
    tf, idf = compute_tfidf(all_sentences)
    
    # Compute CIDEr scores
    scores = []
    for pred, target in zip(preds, targets):
        score = compute_cider_score(pred, target, tf, idf)
        scores.append(score)
    
    return np.mean(scores)

def calculate_rouge(preds, targets):
    """
    Calculate ROUGE score using rouge_score library
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = []
    
    for pred, target in zip(preds, targets):
        # Convert to strings if they're not already
        pred = ' '.join(pred) if isinstance(pred, list) else str(pred)
        target = ' '.join(target) if isinstance(target, list) else str(target)
        
        # Calculate ROUGE scores
        score = scorer.score(target, pred)
        # Use ROUGE-L F1 score
        scores.append(score['rougeL'].fmeasure)
    
    return np.mean(scores) if scores else 0.0

def calculate_metrics(preds, targets):
    """
    Calculate metrics using pycocoevalcap
    """
    # Convert predictions and targets to COCO format
    gts = {}
    res = {}
    
    for i, (pred, target) in enumerate(zip(preds, targets)):
        # Convert to strings if they're not already
        pred = ' '.join(pred) if isinstance(pred, list) else str(pred)
        target = ' '.join(target) if isinstance(target, list) else str(target)
        
        # Add to dictionaries
        gts[i] = [target]
        res[i] = [pred]
    
    # Compute scores using pycocoevalcap
    scores = compute_scores(gts, res)
    
    # Convert scores to percentages
    return {
        'bleu1': scores['BLEU_1'] * 100,  # Convert to percentage
        'bleu2': scores['BLEU_2'] * 100,
        'bleu3': scores['BLEU_3'] * 100,
        'bleu4': scores['BLEU_4'] * 100,
        'rouge': scores['ROUGE_L'] * 100,
        'cider': scores['CIDEr'] * 100
    }