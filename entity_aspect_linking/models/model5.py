from typing import Tuple, List
import torch
import torch.nn.functional as F
print(torch.__version__)
import math
import numpy as np
import torch.nn as nn
from transformers import AutoConfig, AutoModel, DistilBertModel, AutoTokenizer, get_linear_schedule_with_warmup
import warnings
import json

class EntityEmbedding(nn.Module):
    def __init__(self, pretrained: str) -> None:
        super(EntityEmbedding, self).__init__()
        self.pretrained = pretrained
        self.config = AutoConfig.from_pretrained(self.pretrained, output_hidden_states=True)
        self.bert = AutoModel.from_pretrained(self.pretrained, config=self.config)

    def forward(self, inputs_embeds:torch.Tensor) -> torch.Tensor:
        output = self.bert(inputs_embeds=inputs_embeds)
        hidden_states = output.hidden_states
        # get last four layers
        last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]

        # cast layers to a tuple and concatenate over the last dimension
        cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)

        return cat_hidden_states


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
    def __init__(self, pretrained: str, entity_emb_dim: int):
        super(AspectLinkModel, self).__init__()
        self.encoder = EntityEmbedding(pretrained=pretrained)
        self.entity_emb_dim = entity_emb_dim
        # self.score_method = score_method
        if entity_emb_dim != 768:
            self.fc1 = nn.Linear(entity_emb_dim, 768)
        self.score = nn.CosineSimilarity()

        # if score_method == 'bilinear':
        #     self.score = nn.Bilinear(in1_features=emb_dim, in2_features=emb_dim, out_features=1)
        # elif score_method == 'cosine':
        #     self.score = nn.CosineSimilarity()



    def forward(self,context_inputs_embeds: torch.Tensor, aspect_inputs_embeds: torch.Tensor):

        if self.entity_emb_dim != 768:
            context_entity_embeddings = self.encoder(inputs_embeds=self.fc1(context_inputs_embeds))
            aspect_entity_embeddings = self.encoder(inputs_embeds=self.fc1(aspect_inputs_embeds))
        else:
            context_entity_embeddings = self.context_encoder(inputs_embeds=context_inputs_embeds)
            aspect_entity_embeddings = self.aspect_encoder(inputs_embeds=aspect_inputs_embeds)

        # The tensors are of shape (batch_size, seq_len, hidden_dim) where seq_len = number of query entities.
        # First, we change the tensors to shape (seq_len, batch_size, hidden_dim) for easier manipulation.
        context_entity_embeddings = torch.permute(context_entity_embeddings, (1, 0, 2))
        aspect_entity_embeddings = torch.permute(aspect_entity_embeddings, (1, 0, 2))
        aspect_score = 0.0

        aspect_entity_scores = torch.stack([
            self.score(emb1, emb2) for emb1 in aspect_entity_embeddings for emb2 in context_entity_embeddings
        ])

        # aspect_entity_scores = torch.sum(torch.squeeze(aspect_entity_scores, dim=0), dim=0)
        aspect_entity_scores = torch.sum(aspect_entity_scores, dim=0)
        return aspect_entity_scores

        # for emb1 in aspect_entity_embeddings:
        #     aspect_entity_score = sum([self.score(emb1, emb2) for emb2 in context_entity_embeddings])
        #     aspect_score += aspect_entity_score

        # return aspect_score







