import torch
print(torch.__version__)
import torch.nn as nn
from transformers import AutoConfig, AutoModel

class BertEmbedding(nn.Module):
    def __init__(self, pretrained: str) -> None:

        super(BertEmbedding, self).__init__()
        self.pretrained = pretrained
        self.config = AutoConfig.from_pretrained(self.pretrained)
        self.bert = AutoModel.from_pretrained(self.pretrained, config=self.config)

    def forward(self,input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return output[0][:, 0, :]


class AspectLinkModel(nn.Module):
    def __init__(self, pretrained: str, emb_dim: int):
        super(AspectLinkModel, self).__init__()
        self.bert_emb = BertEmbedding(pretrained=pretrained)
        self.fc = nn.Linear(in_features=768+10, out_features=768)
        self.score = nn.CosineSimilarity()


    def forward(
            self,
            context_input_ids: torch.Tensor, context_attention_mask: torch.Tensor,
            aspect_input_ids: torch.Tensor, aspect_attention_mask: torch.Tensor,
            aspect_features: torch.Tensor
    ):

        context_emb = self.bert_emb(input_ids=context_input_ids, attention_mask=context_attention_mask)
        aspect_bert_emb = self.bert_emb(input_ids=aspect_input_ids, attention_mask=aspect_attention_mask)

        concat_embedding = torch.cat((aspect_bert_emb, aspect_features), dim=1)
        aspect_emb = self.fc(concat_embedding)

        final_score = self.score(context_emb, aspect_emb)

        return final_score
