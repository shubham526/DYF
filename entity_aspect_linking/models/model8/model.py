import torch
import torch.nn.functional as F
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


class EntityEmbedding(nn.Module):
    def __init__(self, pretrained: str) -> None:
        super(EntityEmbedding, self).__init__()
        self.pretrained = pretrained
        self.config = AutoConfig.from_pretrained(self.pretrained)
        self.bert = AutoModel.from_pretrained(self.pretrained, config=self.config)

    def forward(self, inputs_embeds:torch.Tensor) -> torch.Tensor:
        output = self.bert(inputs_embeds=inputs_embeds)
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


class EntityBasedSimilarity(nn.Module):
    def __init__(self, pretrained: str, entity_emb_dim: int, score_method: str):
        super(EntityBasedSimilarity, self).__init__()
        self.context_encoder = EntityEmbedding(pretrained=pretrained)
        self.aspect_encoder = EntityEmbedding(pretrained=pretrained)
        self.entity_emb_dim = entity_emb_dim
        self.score_method = score_method
        if entity_emb_dim != 768:
            self.fc1 = nn.Linear(entity_emb_dim, 768)

        if score_method == 'linear':
            self.score = nn.Linear(in_features=768 * 2, out_features=1)
        elif score_method == 'bilinear':
            self.fc2 = nn.Linear(in_features=768, out_features=100)
            self.score = nn.Bilinear(in1_features=100, in2_features=100, out_features=1)
        elif score_method == 'cosine':
            self.score = nn.CosineSimilarity()

    def forward(
            self,
            context_inputs_embeds: torch.Tensor,
            aspect_inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:

        if self.entity_emb_dim != 768:
            context_embedding = self.context_encoder(inputs_embeds=self.fc1(context_inputs_embeds))
            aspect_embedding = self.aspect_encoder(inputs_embeds=self.fc1(aspect_inputs_embeds))
        else:
            context_embedding = self.context_encoder(inputs_embeds=context_inputs_embeds)
            aspect_embedding = self.aspect_encoder(inputs_embeds=aspect_inputs_embeds)

        if context_embedding.size(1) < aspect_embedding.size(1):
            target_len = aspect_embedding.size(1)
            padding = (0, 0, 0, (target_len - context_embedding.size(1)))
            context_embedding = F.pad(context_embedding, padding, "constant", 0)
        elif context_embedding.size(1) > aspect_embedding.size(1):
            target_len = context_embedding.size(1)
            padding = (0, 0, 0, (target_len - aspect_embedding.size(1)))
            aspect_embedding = F.pad(aspect_embedding, padding, "constant", 0)
        else:
            pass

        if self.score_method == 'linear':
            concat_embedding: torch.Tensor = torch.cat((context_embedding, aspect_embedding), dim=1)
            return self.score(concat_embedding)
        elif self.score_method == 'bilinear':
            context_embedding = self.fc2(context_embedding)
            aspect_embedding = self.fc2(aspect_embedding)
            return self.score(context_embedding, aspect_embedding)
        elif self.score_method == 'cosine':
            return self.score(context_embedding, aspect_embedding)

class AspectLinkModel(nn.Module):
    def __init__(self, pretrained: str, entity_emb_dim: int, score_method: str):
        super(AspectLinkModel, self).__init__()
        self.feature_score = nn.Linear(10,1)
        self.text_similarity_model = TextBasedSimilarity(pretrained=pretrained, score_method=score_method)
        self.entity_similarity_model = EntityBasedSimilarity(pretrained=pretrained, entity_emb_dim=entity_emb_dim, score_method=score_method)


    def forward(
            self,
            context_input_ids: torch.Tensor, context_attention_mask: torch.Tensor,
            aspect_input_ids: torch.Tensor, aspect_attention_mask: torch.Tensor,
            context_inputs_embeds: torch.Tensor, aspect_inputs_embeds: torch.Tensor,
            aspect_features: torch.Tensor
    ):
        text_similarity_score = self.text_similarity_model(
            context_input_ids=context_input_ids, context_attention_mask=context_attention_mask,
            aspect_input_ids=aspect_input_ids, aspect_attention_mask=aspect_attention_mask
        )
        entity_similarity_score = self.entity_similarity_model(
            context_inputs_embeds=context_inputs_embeds, aspect_inputs_embeds=aspect_inputs_embeds
        )
        feature_score = self.feature_score(aspect_features)
        final_score = text_similarity_score + entity_similarity_score + feature_score
        return final_score.squeeze(dim=1)







