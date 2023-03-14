from typing import List
import torch
print(torch.__version__)
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from torch.nn.utils.rnn import pad_sequence

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

class WordEmbedding(nn.Module):
    def __init__(self, pretrained: str, out_dim: int) -> None:
        super(WordEmbedding, self).__init__()
        self.pretrained = pretrained
        self.config = AutoConfig.from_pretrained(self.pretrained, output_hidden_states=True)
        self.bert = AutoModel.from_pretrained(self.pretrained, config=self.config)
        self.out_dim = out_dim
        self.fc = nn.Linear(in_features=768, out_features=out_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = output.hidden_states
        # get last four layers
        last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]

        # cast layers to a tuple and concatenate over the last dimension
        cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)

        return cat_hidden_states

        # return self.fc(output[0][:, 0, :])

class EntityEmbedding(nn.Module):
    def __init__(self, pretrained: str, out_dim: int) -> None:
        super(EntityEmbedding, self).__init__()
        self.desc_encoder = TextEmbedding(pretrained=pretrained, out_dim=out_dim)
        self.embedding = nn.Bilinear(in1_features=768, in2_features=100, out_features=out_dim)

    def forward(
            self,
            desc_input_ids: torch.Tensor,
            desc_attention_mask: torch.Tensor,
            wiki2vec_embedding: torch.Tensor
    ) -> torch.Tensor:

        desc_embedding: torch.Tensor = self.desc_encoder(desc_input_ids, desc_attention_mask)
        embedding: torch.Tensor = self.embedding(desc_embedding, wiki2vec_embedding)
        return embedding

class EntityScore(nn.Module):
    def __init__(self, score_method: str, emb_dim: int) -> None:
        super(EntityScore, self).__init__()
        self.score_method = score_method
        self.emb_dim = emb_dim
        if score_method == 'linear':
            self.score = nn.Linear(in_features=emb_dim * 2, out_features=1)
        elif score_method == 'bilinear':
            self.score = nn.Bilinear(in1_features=emb_dim, in2_features=emb_dim, out_features=1)
        elif score_method == 'cosine':
            self.score = nn.CosineSimilarity()

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        if self.score_method == 'linear':
            combined_embedding: torch.Tensor = torch.cat((emb1, emb2), dim=1)
            return self.score(combined_embedding)

        return self.score(emb1, emb2)

class AspectLinkModel(nn.Module):
    def __init__(self, pretrained: str, out_dim: int, device: str) -> None:
        super(AspectLinkModel, self).__init__()
        self.context_word_encoder = WordEmbedding(pretrained=pretrained, out_dim=out_dim)
        self.aspect_word_encoder = WordEmbedding(pretrained=pretrained, out_dim=out_dim)
        self.context_entity_encoder = EntityEmbedding(pretrained=pretrained, out_dim=out_dim)
        self.aspect_entity_encoder = EntityEmbedding(pretrained=pretrained, out_dim=out_dim)
        self.score = nn.Bilinear(in1_features=out_dim, in2_features=out_dim, out_features=1)
        self.device = device

    def forward(
            self,
            context_entity_desc_input_ids: List[torch.Tensor],
            context_entity_desc_attention_mask: List[torch.Tensor],
            context_entity_wiki2vec_embeddings: List[torch.Tensor],
            aspect_entity_desc_input_ids: List[torch.Tensor],
            aspect_entity_desc_attention_mask: List[torch.Tensor],
            aspect_entity_wiki2vec_embeddings: List[torch.Tensor],
    ) -> torch.Tensor:

        context_desc_input_ids: List[torch.Tensor] = [input_ids.to(self.device) for input_ids in context_entity_desc_input_ids]
        context_desc_attention_mask: List[torch.Tensor] = [attention_mask.to(self.device) for attention_mask in context_entity_desc_attention_mask]
        context_wiki2vec_embeddings: List[torch.Tensor] = [wiki2vec_emb.to(self.device) for wiki2vec_emb in context_entity_wiki2vec_embeddings]

        aspect_desc_input_ids: List[torch.Tensor] = [input_ids.to(self.device) for input_ids in aspect_entity_desc_input_ids]
        aspect_desc_attention_mask: List[torch.Tensor] = [attention_mask.to(self.device) for attention_mask in aspect_entity_desc_attention_mask]
        aspect_wiki2vec_embeddings: List[torch.Tensor] = [wiki2vec_emb.to(self.device) for wiki2vec_emb in aspect_entity_wiki2vec_embeddings]

        context_entity_embeddings: torch.Tensor = pad_sequence([
            self.context_entity_encoder(input_ids, attention_mask, wiki2vec_embedding)
            for input_ids, attention_mask, wiki2vec_embedding in
            zip(context_desc_input_ids, context_desc_attention_mask, context_wiki2vec_embeddings)
        ])

        aspect_entity_embeddings: torch.Tensor = pad_sequence([
            self.aspect_entity_encoder(input_ids, attention_mask, wiki2vec_embedding)
            for input_ids, attention_mask, wiki2vec_embedding in
            zip(aspect_desc_input_ids, aspect_desc_attention_mask, aspect_wiki2vec_embeddings)
        ])

        aspect_entity_scores = torch.stack([
            self.score(emb1, emb2) for emb1 in aspect_entity_embeddings for emb2 in context_entity_embeddings
        ])

        aspect_entity_scores = torch.mean(torch.squeeze(aspect_entity_scores), dim=0)

        return aspect_entity_scores




















