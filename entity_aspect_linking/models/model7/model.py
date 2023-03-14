import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class BertEmbedding(nn.Module):
    def __init__(self, pretrained: str) -> None:

        super(BertEmbedding, self).__init__()
        self.pretrained = pretrained
        self.config = AutoConfig.from_pretrained(self.pretrained)
        self.bert = AutoModel.from_pretrained(self.pretrained, config=self.config)

    def forward(self,input_ids, attention_mask = None, token_type_ids = None) -> torch.Tensor:
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return output[0][:, 0, :]


class AspectLinkModel(nn.Module):
    def __init__(self, pretrained: str):
        super(AspectLinkModel, self).__init__()
        self.bert_emb = BertEmbedding(pretrained=pretrained)
        self.score = nn.Linear(in_features=768+10, out_features=1)


    def forward(self, input_ids, attention_mask, token_type_ids, features):
        bert_emb = self.bert_emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        concat_embedding = torch.cat((bert_emb, features), dim=1)
        final_score = self.score(concat_embedding)
        return final_score.squeeze(dim=1)
