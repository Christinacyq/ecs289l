from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor import Meteor
from pycocoevalcap.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


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
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    # print(f"gts: {gts}")
    # print(res)
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