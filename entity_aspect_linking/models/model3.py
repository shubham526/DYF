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
from torch.nn.utils.rnn import pad_sequence

def pad(emb1: torch.Tensor, emb2: torch.Tensor):

    if len(emb1) < len(emb2):
        emb1 = F.pad(
            input=emb1.squeeze(dim=1),
            pad=(0, 0, 0, (len(emb2) - len(emb1))),
            mode="constant",
            value=0
        )
    elif len(emb2) < len(emb1):
        emb2 = F.pad(
            input=emb2.squeeze(dim=1),
            pad=(0, 0, 0, (len(emb1) - len(emb2))),
            mode="constant",
            value=0
        )
    return emb1, emb2

class TextEmbedding(nn.Module):
    def __init__(self, pretrained: str, out_dim: int) -> None:
        super(TextEmbedding, self).__init__()
        self.pretrained = pretrained
        self.config = AutoConfig.from_pretrained(self.pretrained)
        self.bert = AutoModel.from_pretrained(self.pretrained, config=self.config)
        self.out_dim = out_dim
        self.fc = nn.Linear(in_features=768, out_features=out_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc(output[0][:, 0, :])


class EntityEmbedding(nn.Module):
    def __init__(self, pretrained: str, out_dim: int) -> None:
        super(EntityEmbedding, self).__init__()
        self.desc_encoder = TextEmbedding(pretrained=pretrained, out_dim=out_dim)
        self.embedding = nn.Bilinear(in1_features=out_dim, in2_features=out_dim, out_features=out_dim)

    def forward(
            self,
            desc_input_ids: List[torch.Tensor],
            desc_attention_mask: List[torch.Tensor],
            wiki2vec_embeddings: List[torch.Tensor]
    ) -> List[torch.Tensor]:

        desc_embeddings = [
            self.desc_encoder(input_ids, attention_mask)
            for input_ids, attention_mask in zip(desc_input_ids, desc_attention_mask)
        ]
        embeddings = [
            self.embedding(desc_emb, wiki2vec_emb)
            for desc_emb, wiki2vec_emb in zip(desc_embeddings, wiki2vec_embeddings)
        ]

        return embeddings

class ContextualAspectTextEmbedding(nn.Module):
    def __init__(self, pretrained: str, out_dim: int) -> None:
        super(ContextualAspectTextEmbedding, self).__init__()
        self.encoder = TextEmbedding(pretrained=pretrained, out_dim=out_dim)
        self.embedding = nn.Bilinear(in1_features=out_dim, in2_features=out_dim, out_features=out_dim)

    def forward(
            self,
            context_input_ids: torch.Tensor,
            context_attention_mask: torch.Tensor,
            aspect_input_ids: torch.Tensor,
            aspect_attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        context_embedding: torch.Tensor = self.encoder(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask
        )
        aspect_embedding: torch.Tensor = self.encoder(
            input_ids=aspect_input_ids,
            attention_mask=aspect_attention_mask
        )
        embedding: torch.Tensor = self.embedding(context_embedding, aspect_embedding)
        return embedding

class ContextualAspectEntityEmbedding(nn.Module):
    def __init__(self, pretrained: str, out_dim: int) -> None:
        super(ContextualAspectEntityEmbedding, self).__init__()
        self.encoder = EntityEmbedding(pretrained=pretrained, out_dim=out_dim)
        self.embedding = nn.Bilinear(in1_features=out_dim, in2_features=out_dim, out_features=out_dim)

    def forward(
            self,
            context_desc_input_ids: List[torch.Tensor],
            context_desc_attention_mask: List[torch.Tensor],
            context_wiki2vec_embeddings: List[torch.Tensor],
            aspect_desc_input_ids: List[torch.Tensor],
            aspect_desc_attention_mask: List[torch.Tensor],
            aspect_wiki2vec_embeddings: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        context_embedding = self.encoder(
            desc_input_ids=context_desc_input_ids,
            desc_attention_mask=context_desc_attention_mask,
            wiki2vec_embeddings=context_wiki2vec_embeddings
        )
        aspect_embedding = self.encoder(
            desc_input_ids=aspect_desc_input_ids,
            desc_attention_mask=aspect_desc_attention_mask,
            wiki2vec_embeddings=aspect_wiki2vec_embeddings
        )
        embeddings = []
        for context_emb, aspect_emb in zip(context_embedding, aspect_embedding):
            context_emb, aspect_emb = pad(context_emb, aspect_emb)
            embeddings.append(self.embedding(context_emb, aspect_emb))

        embeddings = pad_sequence(embeddings)

        # The entity embedding is of size (seq_len, batch_size, hidden_dim)
        # Take mean over entity dimension
        return torch.mean(embeddings, dim=0)

class AspectLinkModel(nn.Module):
    def __init__(self, pretrained: str, out_dim: int, device: str) -> None:
        super(AspectLinkModel, self).__init__()
        self.text_encoder = ContextualAspectTextEmbedding(pretrained=pretrained, out_dim=out_dim)
        self.entity_encoder = ContextualAspectEntityEmbedding(pretrained=pretrained, out_dim=out_dim)
        self.score = nn.Bilinear(in1_features=out_dim, in2_features=out_dim, out_features=1)
        self.device = device

    def forward(
            self,
            context_text_input_ids: torch.Tensor,
            context_text_attention_mask: torch.Tensor,
            aspect_text_input_ids: torch.Tensor,
            aspect_text_attention_mask: torch.Tensor,
            context_entity_desc_input_ids: List[torch.Tensor],
            context_entity_desc_attention_mask: List[torch.Tensor],
            context_entity_wiki2vec_embeddings: List[torch.Tensor],
            aspect_entity_desc_input_ids: List[torch.Tensor],
            aspect_entity_desc_attention_mask: List[torch.Tensor],
            aspect_entity_wiki2vec_embeddings: List[torch.Tensor],
    ) -> torch.Tensor:

        context_desc_input_ids = [input_ids.to(self.device) for input_ids in context_entity_desc_input_ids]
        context_desc_attention_mask = [attention_mask.to(self.device) for attention_mask in context_entity_desc_attention_mask]
        context_wiki2vec_embeddings = [wiki2vec_emb.to(self.device) for wiki2vec_emb in context_entity_wiki2vec_embeddings]

        aspect_desc_input_ids = [input_ids.to(self.device) for input_ids in aspect_entity_desc_input_ids]
        aspect_desc_attention_mask = [attention_mask.to(self.device) for attention_mask in aspect_entity_desc_attention_mask]
        aspect_wiki2vec_embeddings = [wiki2vec_emb.to(self.device) for wiki2vec_emb in aspect_entity_wiki2vec_embeddings]

        text_embedding: torch.Tensor = self.text_encoder(
            context_input_ids=context_text_input_ids.to(self.device),
            context_attention_mask=context_text_attention_mask.to(self.device),
            aspect_input_ids=aspect_text_input_ids.to(self.device),
            aspect_attention_mask=aspect_text_attention_mask.to(self.device),
        )

        entity_embedding: torch.Tensor = self.entity_encoder(
            context_desc_input_ids=context_desc_input_ids,
            context_desc_attention_mask=context_desc_attention_mask,
            context_wiki2vec_embeddings=context_wiki2vec_embeddings,
            aspect_desc_input_ids=aspect_desc_input_ids,
            aspect_desc_attention_mask=aspect_desc_attention_mask,
            aspect_wiki2vec_embeddings=aspect_wiki2vec_embeddings,
        )

        score: torch.Tensor = self.score(text_embedding, entity_embedding)
        return score.squeeze(dim=1)


