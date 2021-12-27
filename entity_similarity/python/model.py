from typing import Tuple, List
import torch
print(torch.__version__)
import math
import numpy as np
import torch.nn as nn
from transformers import AutoConfig, AutoModel, DistilBertModel, AutoTokenizer, get_linear_schedule_with_warmup
import warnings
import json


class BertEmbedding(nn.Module):
    def __init__(self, pretrained: str) -> None:

        super(BertEmbedding, self).__init__()
        self.pretrained = pretrained
        self.config = AutoConfig.from_pretrained(self.pretrained)
        self.bert = AutoModel.from_pretrained(self.pretrained, config=self.config)

    def forward(self,input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state


class NGramEmbedding(nn.Module):
    def __init__(
            self,
            embedding_dimension: int,
            kernel_dim: int,
            kernel_sizes: List[int] = None
    ):
        super(NGramEmbedding, self).__init__()

        if kernel_sizes is None:
            kernel_sizes = [1, 3, 5]

        self.kernel_dim = kernel_dim
        self.kernel_sizes = kernel_sizes
        self.embedding_dimension = kernel_dim * len(kernel_sizes)
        self.encoder = nn.ModuleList()

        for kernel_size in kernel_sizes:
            conv = nn.Conv1d(
                in_channels=embedding_dimension,
                out_channels=kernel_dim,
                kernel_size=kernel_size,
                stride=1,
                padding='same'
            )
            self.encoder.append(conv)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:

        # Conv1D needs channel dimension in 2nd position so we transpose
        embeddings: torch.Tensor = embeddings.transpose(1,-1)

        # Pass through each Conv1D
        vectors = [encoder(embeddings) for encoder in self.encoder]

        # Concatenate vectors over the sequence dimension, then transpose back
        out = torch.cat(vectors, dim=1).transpose(1, -1)
        return out

class EmbeddingCombination(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(EmbeddingCombination, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dense = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        # Concatenate over the last dimension
        combined_embedding: torch.Tensor = torch.cat(embeddings, dim=2)

        # Now the embedding_dim of tokens is larger
        # So we convert to a fixed length embedding
        combined_embedding = self.dense(combined_embedding)

        # Mean over the token dimension
        combined_embedding = torch.mean(combined_embedding, dim=1)
        return combined_embedding


class TypeEmbedding(nn.Module):
    def __init__(self, pretrained_bert_model: str):
        super(TypeEmbedding, self).__init__()
        self.embedding = BertEmbedding(pretrained=pretrained_bert_model)

    def forward(self, type_input_ids: torch.Tensor, type_attention_mask: torch.Tensor) -> torch.Tensor:

        type_embedding: torch.Tensor = self.embedding(
                input_ids=type_input_ids,
                attention_mask=type_attention_mask
            )

        # BERT gives a Tensor of shape (batch_size, sequence_length, hidden_dim)
        # We create a type embedding by taking mean over the token embeddings of each token in the type
        type_embedding = torch.mean(type_embedding, dim=1)
        # Then concatenate the type embeddings and take the mean
        #type_embedding: torch.Tensor = torch.mean(torch.cat(type_embeddings_list, dim=0), dim=0).unsqueeze(dim=0)

        return type_embedding


class TextEmbedding(nn.Module):
    def __init__(
            self,
            pretrained_bert_model: str,
            out_dim: int,
            n_gram_sizes: List[int] = None,
    ):
        super(TextEmbedding, self).__init__()
        self.embedding = BertEmbedding(pretrained=pretrained_bert_model)
        self.encoder = NGramEmbedding(embedding_dimension=768, kernel_dim=out_dim, kernel_sizes=n_gram_sizes)
        self.combined_embedding_dim = self.encoder.embedding_dimension + 768
        self.output_embedding_dim = out_dim
        self.combination = EmbeddingCombination(input_dim=self.combined_embedding_dim, output_dim=self.output_embedding_dim)

    def forward(self,input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        bert_embedding: torch.Tensor = self.embedding(input_ids, attention_mask)
        ngram_embedding: torch.Tensor = self.encoder(bert_embedding)
        combined_embedding: torch.Tensor = self.combination([bert_embedding, ngram_embedding])
        return combined_embedding


class EntityEmbedding(nn.Module):
    def __init__(
            self,
            pretrained_bert_model: str,
            out_dim: int,
            n_gram_sizes: List[int] = None
    ):
        super(EntityEmbedding, self).__init__()
        self.desc_encoder = TextEmbedding(pretrained_bert_model=pretrained_bert_model, n_gram_sizes=n_gram_sizes, out_dim=out_dim)
        self.name_encoder = BertEmbedding(pretrained=pretrained_bert_model)
        self.type_encoder = TypeEmbedding(pretrained_bert_model=pretrained_bert_model)
        self.combined_embedding_dim = out_dim + 768 + 768
        self.output_embedding_dim = out_dim
        self.combination = nn.Linear(self.combined_embedding_dim, self.output_embedding_dim)

    def forward(
            self,
            desc_input_ids,
            desc_attention_mask,
            name_input_ids,
            name_attention_mask,
            type_input_ids,
            type_attention_mask
    ) -> torch.Tensor :


        # Unpack the description tensors
        #desc_input_ids, desc_attention_mask, desc_token_type_ids = description_inputs
        # Encode the description
        desc_embedding: torch.Tensor = self.desc_encoder(
            input_ids=desc_input_ids,
            attention_mask=desc_attention_mask
        )


        # Unpack the name tensors
        #name_input_ids, name_attention_mask, name_token_type_ids = name_inputs
        # Encode the name
        name_embedding: torch.Tensor = self.name_encoder(
            input_ids=name_input_ids,
            attention_mask=name_attention_mask
        )
        # BERT gives a Tensor of shape (batch_size, sequence_length, hidden_dim)
        # We create a name embedding by taking a mean over the token embeddings of each token in the name
        name_embedding = torch.mean(name_embedding, dim=1)

        # Encode the entity types
        type_embedding: torch.Tensor = self.type_encoder(
            type_input_ids=type_input_ids,
            type_attention_mask=type_attention_mask
        )

        # Finally, concatenate the three embeddings
        concat_embedding: torch.Tensor = torch.cat((desc_embedding, name_embedding, type_embedding), dim=1)

        # The final entity embedding is obtained after passing through a Linear layer
        entity_embedding: torch.Tensor = self.combination(concat_embedding)
        return entity_embedding


class ContextualEntityEmbedding(nn.Module):
    def __init__(
            self,
            pretrained_bert_model: str,
            out_dim: int,
            n_gram_sizes: List[int] = None):
        super(ContextualEntityEmbedding, self).__init__()
        self.context_encoder = TextEmbedding(
            pretrained_bert_model=pretrained_bert_model,
            out_dim=out_dim,
            n_gram_sizes=n_gram_sizes,
        )
        self.entity_encoder = EntityEmbedding(
            pretrained_bert_model=pretrained_bert_model,
            out_dim=out_dim,
            n_gram_sizes=n_gram_sizes,
        )
        self.combination = nn.Linear(out_dim * 2, out_dim)


    def forward(
            self,
            context_input_ids,
            context_attention_mask,
            desc_input_ids,
            desc_attention_mask,
            name_input_ids,
            name_attention_mask,
            type_input_ids,
            type_attention_mask
    ) -> torch.Tensor:


        # Unpack the context tensors
        #context_input_ids, context_attention_mask, context_token_type_ids = context_inputs
        # Encode the description
        context_embedding: torch.Tensor = self.context_encoder(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
        )

        entity_embedding: torch.Tensor = self.entity_encoder(
            desc_input_ids=desc_input_ids,
            desc_attention_mask=desc_attention_mask,
            name_input_ids=name_input_ids,
            name_attention_mask=name_attention_mask,
            type_input_ids=type_input_ids,
            type_attention_mask=type_attention_mask
        )

        out_embedding: torch.Tensor = torch.cat((context_embedding, entity_embedding), dim=1)
        out_embedding = self.combination(out_embedding)
        return out_embedding


if __name__ == '__main__':
    main()










