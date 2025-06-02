import torch
import torch.nn as nn
import torch.nn.functional as F

class MedCLIPModel(nn.Module):
    def __init__(self, visual_extractor, text_extractor):
        super(MedCLIPModel, self).__init__()
        self.visual_extractor = visual_extractor
        self.text_extractor = text_extractor
        self.temperature = 0.07

    def forward(self, images, reports_ids, reports_masks, reports_ids_bert, reports_masks_bert):
        # Extract visual features
        patch_embeddings, cls_token, visual_features = self.visual_extractor(images)
        
        # Extract text features
        text_features, text_features_bert = self.text_extractor(
            reports_ids,
            reports_masks,
            reports_ids_bert,
            reports_masks_bert
        )
        
        # Normalize features
        visual_features = F.normalize(visual_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        text_features_bert = F.normalize(text_features_bert, dim=-1)
        
        # Compute similarity matrices
        logits_v2t = torch.matmul(visual_features, text_features.t()) / self.temperature
        logits_t2v = logits_v2t.t()
        
        logits_v2t_bert = torch.matmul(visual_features, text_features_bert.t()) / self.temperature
        logits_t2v_bert = logits_v2t_bert.t()
        
        return {
            'logits_v2t': logits_v2t,
            'logits_t2v': logits_t2v,
            'logits_v2t_bert': logits_v2t_bert,
            'logits_t2v_bert': logits_t2v_bert,
            'patch_embeddings': patch_embeddings,
            'cls_token': cls_token
        } 