from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import medspacy

nlp = medspacy.load()

def extract_entities(text):
    doc = nlp(text)
    return set(ent.text.lower() for ent in doc.ents)

def compute_medspacy_f1(preds, targets):
    tp = 0
    fp = 0
    fn = 0

    for pred, target in zip(preds, targets):
        pred_ents = extract_entities(pred)
        target_ents = extract_entities(target)

        tp += len(pred_ents & target_ents)
        fp += len(pred_ents - target_ents)
        fn += len(target_ents - pred_ents)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Print examples from the first sample
    print("Example GT entities:")
    print(extract_entities(targets[0]))

    print("Example RES entities:")
    print(extract_entities(preds[0]))

    return {
        "medspacy_precision": precision,
        "medspacy_recall": recall,
        "medspacy_f1": f1
    }


def compute_scores(gts, res):
    """
    Performs the MS COCO + MedSpaCy evaluation.
    :param gts: Dictionary with image ids and gold captions
    :param res: Dictionary with image ids and generated captions
    :return: Dictionary with all scores
    """
    # Check if gts or res is empty
    if not gts or not res:
        print("Warning: Empty ground truth or results")
        return {
            'BLEU_1': 0.0,
            'BLEU_2': 0.0,
            'BLEU_3': 0.0,
            'BLEU_4': 0.0,
            'ROUGE_L': 0.0,
            'CIDEr': 0.0,
            'medspacy_precision': 0.0,
            'medspacy_recall': 0.0,
            'medspacy_f1': 0.0
        }

    # Standard metrics
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    
    eval_res = {}
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

    # Clinical NLP metrics (MedSpaCy F1)
    preds = [res[k][0] for k in res if k in gts]
    targets = [gts[k][0] for k in res if k in gts]
    medspacy_scores = compute_medspacy_f1(preds, targets)
    eval_res.update(medspacy_scores)

    return eval_res
