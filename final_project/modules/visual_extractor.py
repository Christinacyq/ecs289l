import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel

class VisualExtractor_medvit(nn.Module):
    def __init__(self, args):
        super(VisualExtractor_medvit, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        
        # Initialize PubMedCLIP model
        self.processor = CLIPProcessor.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
        self.feature_dim = self.model.vision_model.config.hidden_size  # 768 for ViT-Base
        
        if self.pretrained:
            print('################ loading pretrained weights for PubMedCLIP')
            # The model is already pretrained from HuggingFace
        
        # Projection head to match the expected dimension
        self.projection_head = nn.Linear(self.feature_dim, 512, bias=False)

    def forward(self, images):
        # Reshape images to process all at once
        batch_size = images.size(0)
        images = images.view(-1, *images.shape[2:])
        
        # Process images with CLIP processor
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(images.device) for k, v in inputs.items()}
        
        # Get image features
        outputs = self.model.vision_model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        
        # Reshape back to batch format
        last_hidden_states = last_hidden_states.view(batch_size, -1, last_hidden_states.size(-1))
        
        # Split into patch embeddings and CLS token
        patch_embeddings = last_hidden_states[:, 1:, :]  # Remove CLS token
        cls_token = last_hidden_states[:, 0, :]  # Get CLS token
        
        # Project features
        projected_features = self.projection_head(cls_token)
        
        return patch_embeddings, cls_token, projected_features
