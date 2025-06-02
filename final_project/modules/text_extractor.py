import torch
import torch.nn as nn
from transformers import AutoModel

class MedCLIPTextModel(nn.Module):
    def __init__(self,
        bert_type="emilyalsentzer/Bio_ClinicalBERT",
        proj_dim = 512,
        proj_bias = False) -> None:
        super().__init__()
        self.bert_type = bert_type
        self.last_n_layer = 4
        self.model = AutoModel.from_pretrained(self.bert_type, output_hidden_states=True)
        self.projection_head = nn.Linear(768, proj_dim, bias=proj_bias)

    def forward(self, input_ids, attention_mask, input_ids_bert, attention_mask_bert):
        # Process regular input
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2], output['hidden_states'][-1]])
        embed = last_hidden_states.permute(1,0,2,3).mean(2).mean(1)
        embed = self.projection_head(embed)
        
        # Process BERT input
        output_bert = self.model(input_ids=input_ids_bert, attention_mask=attention_mask_bert)
        last_hidden_states_bert = torch.stack([output_bert['hidden_states'][1], output_bert['hidden_states'][2], output_bert['hidden_states'][-1]])
        embed_bert = last_hidden_states_bert.permute(1,0,2,3).mean(2).mean(1)
        embed_bert = self.projection_head(embed_bert)
        
        return embed, embed_bert

