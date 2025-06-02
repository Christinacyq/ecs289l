import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


def compute_loss(output, reports_ids, reports_masks):
    criterion = LanguageModelCriterion()
    loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, outputs, reports_ids, reports_masks):
        # Get logits from outputs
        logits_v2t = outputs['logits_v2t']
        logits_t2v = outputs['logits_t2v']
        logits_v2t_bert = outputs['logits_v2t_bert']
        logits_t2v_bert = outputs['logits_t2v_bert']
        
        # Create labels (diagonal elements are positive pairs)
        batch_size = logits_v2t.size(0)
        labels = torch.arange(batch_size, device=logits_v2t.device)
        
        # Compute losses for regular features
        loss_v2t = F.cross_entropy(logits_v2t, labels)
        loss_t2v = F.cross_entropy(logits_t2v, labels)
        
        # Compute losses for BERT features
        loss_v2t_bert = F.cross_entropy(logits_v2t_bert, labels)
        loss_t2v_bert = F.cross_entropy(logits_t2v_bert, labels)
        
        # Combine losses
        total_loss = (loss_v2t + loss_t2v + loss_v2t_bert + loss_t2v_bert) / 4.0
        
        return total_loss