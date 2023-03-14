import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class TextEmbedding(nn.Module):
    def __init__(self, pretrained: str) -> None:

        super(TextEmbedding, self).__init__()
        self.pretrained = pretrained
        self.config = AutoConfig.from_pretrained(self.pretrained)
        self.bert = AutoModel.from_pretrained(self.pretrained, config=self.config)

    def forward(self,input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return output[0][:, 0, :]

class TextBasedSimilarity(nn.Module):
    def __init__(self, pretrained: str, score_method: str):
        super(TextBasedSimilarity, self).__init__()
        self.context_encoder = TextEmbedding(pretrained=pretrained)
        self.aspect_encoder = TextEmbedding(pretrained=pretrained)
        self.score_method = score_method

        if score_method == 'linear':
            self.score = nn.Linear(in_features=768 * 2, out_features=1)
        elif score_method == 'bilinear':
            self.fc = nn.Linear(in_features=768, out_features=100)
            self.score = nn.Bilinear(in1_features=100, in2_features=100, out_features=1)
        elif score_method == 'cosine':
            self.score = nn.CosineSimilarity()


    def forward(
            self,
            context_input_ids: torch.Tensor,
            context_attention_mask: torch.Tensor,
            aspect_input_ids: torch.Tensor,
            aspect_attention_mask: torch.Tensor
    ) -> torch.Tensor:

        context_embedding = self.context_encoder(input_ids=context_input_ids, attention_mask=context_attention_mask)
        aspect_embedding = self.aspect_encoder(input_ids=aspect_input_ids, attention_mask=aspect_attention_mask)
        if self.score_method == 'linear':
            concat_embedding: torch.Tensor = torch.cat((context_embedding, aspect_embedding), dim=1)
            return self.score(concat_embedding)
        elif self.score_method == 'bilinear':
            context_embedding = self.fc(context_embedding)
            aspect_embedding = self.fc(aspect_embedding)
            return self.score(context_embedding, aspect_embedding)
        elif self.score_method == 'cosine':
            return self.score(context_embedding, aspect_embedding)


class AspectLinkModel(nn.Module):
    def __init__(self, pretrained: str, score_method: str):
        super(AspectLinkModel, self).__init__()
        self.text_similarity_model = TextBasedSimilarity(pretrained=pretrained, score_method=score_method)

    def forward(
            self,
            context_input_ids: torch.Tensor, context_attention_mask: torch.Tensor,
            aspect_input_ids: torch.Tensor, aspect_attention_mask: torch.Tensor,
    ):
        score = self.text_similarity_model(
            context_input_ids=context_input_ids, context_attention_mask=context_attention_mask,
            aspect_input_ids=aspect_input_ids, aspect_attention_mask=aspect_attention_mask
        )

        return score.squeeze(dim=1)







