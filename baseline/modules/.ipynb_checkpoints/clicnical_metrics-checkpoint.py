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

    return {
        "medspacy_precision": precision,
        "medspacy_recall": recall,
        "medspacy_f1": f1
    }
