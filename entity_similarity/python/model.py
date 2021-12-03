from typing import Tuple, List
import torch
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


    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None,
                token_type_ids: torch.Tensor = None) -> torch.Tensor:

        if token_type_ids is not None:
            output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        return output.last_hidden_state


class NGramEncoder(nn.Module):
    def __init__(
            self,
            embedding_dimension: int,
            kernel_dim: int,
            kernel_sizes: List[int] = None
    ):
        super(NGramEncoder, self).__init__()

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
    def __init__(self, pretrained_bert_model: str, num_types: int, out_dim: int):
        super(TypeEmbedding, self).__init__()
        self.embedding = BertEmbedding(pretrained=pretrained_bert_model)
        self.combined_embedding_dim = 768 * num_types
        self.output_embedding_dim = out_dim
        self.combination = nn.Linear(self.combined_embedding_dim, self.output_embedding_dim)

    def forward(self, type_inputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> torch.Tensor:

        # First create a BERT embedding of each entity type
        type_embeddings_list: List[torch.Tensor] = [
            self.embedding(
                input_ids=type_input_ids,
                attention_mask=type_attention_mask,
                token_type_ids=type_token_type_ids
            )
            for type_input_ids, type_attention_mask, type_token_type_ids in type_inputs
        ]

        # BERT gives a Tensor of shape (batch_size, sequence_length, hidden_dim)
        # We create a type embedding by taking mean over the token embeddings of each token in the type
        type_embeddings_list = [torch.mean(type_embedding, dim=1) for type_embedding in type_embeddings_list]
        # Then concatenate the type embeddings
        type_embedding: torch.Tensor = torch.cat(type_embeddings_list, dim=1)
        # The final type embedding is obtained after passing through  a Linear layer
        type_embedding = self.combination(type_embedding)
        return type_embedding


class TextEncoder(nn.Module):
    def __init__(
            self,
            pretrained_bert_model: str,
            out_dim: int,
            n_gram_sizes: List[int] = None,
    ):
        super(TextEncoder, self).__init__()
        self.embedding = BertEmbedding(pretrained=pretrained_bert_model)
        self.encoder = NGramEncoder(embedding_dimension=768, kernel_dim=out_dim, kernel_sizes=n_gram_sizes)
        self.combined_embedding_dim = self.encoder.embedding_dimension + 768
        self.output_embedding_dim = out_dim
        self.combination = EmbeddingCombination(input_dim=self.combined_embedding_dim, output_dim=self.output_embedding_dim)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor = None,
            token_type_ids: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor,torch.Tensor]:
        bert_embedding: torch.Tensor = self.embedding(input_ids, attention_mask, token_type_ids)
        ngram_embedding: torch.Tensor = self.encoder(bert_embedding)
        combined_embedding: torch.Tensor = self.combination([bert_embedding, ngram_embedding])
        return bert_embedding, ngram_embedding, combined_embedding


class EntityEncoder(nn.Module):
    def __init__(
            self,
            pretrained_bert_model: str,
            num_types: int,
            out_dim: int,
            n_gram_sizes: List[int] = None
    ):
        super(EntityEncoder, self).__init__()
        self.desc_encoder = TextEncoder(pretrained_bert_model=pretrained_bert_model, n_gram_sizes=n_gram_sizes, out_dim=out_dim)
        self.name_encoder = BertEmbedding(pretrained=pretrained_bert_model)
        self.type_encoder = TypeEmbedding(pretrained_bert_model=pretrained_bert_model, num_types=num_types, out_dim=out_dim)
        self.combined_embedding_dim = out_dim + 768 + out_dim
        self.output_embedding_dim = out_dim
        self.combination = nn.Linear(self.combined_embedding_dim, self.output_embedding_dim)

    def forward(
            self,
            description_inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            name_inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            type_inputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor :


        # Unpack the description tensors
        desc_input_ids, desc_attention_mask, desc_token_type_ids = description_inputs
        # Encode the description
        bert_desc_embedding , ngram_desc_embedding, _ = self.desc_encoder(
            input_ids=desc_input_ids,
            attention_mask=desc_attention_mask,
            token_type_ids=desc_token_type_ids
        )
        desc_embedding: torch.Tensor = torch.cat((bert_desc_embedding, ngram_desc_embedding), dim=1)

        # Unpack the name tensors
        name_input_ids, name_attention_mask, name_token_type_ids = name_inputs
        # Encode the name
        name_embedding: torch.Tensor = self.name_encoder(
            input_ids=name_input_ids,
            attention_mask=name_attention_mask,
            token_type_ids=name_token_type_ids
        )
        # BERT gives a Tensor of shape (batch_size, sequence_length, hidden_dim)
        # We create a name embedding by taking a mean over the token embeddings of each token in the name
        name_embedding = torch.mean(name_embedding, dim=1)

        # Encode the entity types
        type_embedding: torch.Tensor = self.type_encoder(type_inputs=type_inputs)

        # Finally, concatenate the three embeddings
        concat_embedding: torch.Tensor = torch.cat((desc_embedding, name_embedding, type_embedding), dim=1)

        # The final entity embedding is obtained after passing through a Linear layer
        entity_embedding: torch.Tensor = self.combination(concat_embedding)
        return entity_embedding


class EntitySimilarityModel(nn.Module):
    def __init__(
            self,
            pretrained_bert_model: str,
            num_types: int,
            out_dim: int,
            n_gram_sizes: List[int] = None):
        super(EntitySimilarityModel, self).__init__()
        self.context_encoder = TextEncoder(pretrained_bert_model=pretrained_bert_model, out_dim=out_dim, n_gram_sizes=n_gram_sizes)
        self.entity_encoder = EntityEncoder(pretrained_bert_model=pretrained_bert_model, num_types=num_types, out_dim=out_dim, n_gram_sizes=n_gram_sizes)


    def forward(
            self,
            context_inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            description_inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            name_inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            type_inputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:


        # Unpack the context tensors
        context_input_ids, context_attention_mask, context_token_type_ids = context_inputs
        # Encode the description
        context_embedding: torch.Tensor = self.context_encoder(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
            token_type_ids=context_token_type_ids
        )

        entity_embedding: torch.Tensor = self.entity_encoder(description_inputs, name_inputs, type_inputs)

        out_embedding: torch.Tensor = torch.cat((context_embedding, entity_embedding), dim=1)
        return out_embedding




def create_bert_input(sentence, tokenizer):
    encoded_dict = tokenizer.encode_plus(
        text=sentence,
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=15,  # Pad & truncate all sentences.
        padding='max_length',
        truncation=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_token_type_ids=True,  # Construct token type ids
        return_tensors='pt'
    )
    return encoded_dict['input_ids'], encoded_dict['attention_mask'], encoded_dict['token_type_ids']




def main():
    pretrain = 'bert-base-uncased'
    vocab = 'bert-base-uncased'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(vocab)

    data = '{"doc2":{"entity_desc":"alfr ronald took sport fly fish learn craft river trent blyth dove river blyth near todai creswel green ronald construct banksid fish hut design primarili observatori trout behaviour river hut elsewher hi home river ronald conduct experi formul idea eventu publish flyfish entomolog 1836","entity_name":"The Fly-fisher Entomology","entity_types":["1836 books","angling literature","fly fishing","british books","recreational fishing in the united kingdom"]},"doc1":{"entity_desc":"alfr ronald took sport fly fish learn craft river trent blyth dove river blyth near todai creswel green ronald construct banksid fish hut design primarili observatori trout behaviour river hut elsewher hi home river ronald conduct experi formul idea eventu publish flyfish entomolog 1836","entity_name":"Creswell, Staffordshire","entity_types":["stafford borough","villages in staffordshire"]},"context":"Recreational fishing","label":1}'
    data_dict = json.loads(data)

    context = data_dict['context']

    e1_desc = data_dict['doc1']['entity_desc']
    e1_name = data_dict['doc1']['entity_name']
    e1_types = data_dict['doc1']['entity_types']

    e2_desc = data_dict['doc2']['entity_desc']
    e2_name = data_dict['doc2']['entity_name']
    e2_types = data_dict['doc2']['entity_types']

    num_types = min(len(e1_types), len(e2_types))
    e1_types = e1_types[:num_types]
    e2_types = e2_types[:num_types]

    context_inputs = create_bert_input(context, tokenizer)

    e1_desc_inputs = create_bert_input(e1_desc, tokenizer)
    e1_name_inputs = create_bert_input(e1_name, tokenizer)
    e1_type_inputs = [create_bert_input(e1_type, tokenizer) for e1_type in e1_types]

    e2_desc_inputs = create_bert_input(e2_desc, tokenizer)
    e2_name_inputs = create_bert_input(e2_name, tokenizer)
    e2_type_inputs = [create_bert_input(e2_type, tokenizer) for e2_type in e2_types]

    model = EntitySimilarityModel(pretrained_bert_model=pretrain, num_types=num_types, out_dim=100, n_gram_sizes=[1, 2, 3])
    model.to(device)
    model.train()
    embedding1 = model(context_inputs=context_inputs, description_inputs=e1_desc_inputs,
                       name_inputs=e1_name_inputs, type_inputs=e1_type_inputs)
    embedding2 = model(context_inputs=context_inputs, description_inputs=e2_desc_inputs, name_inputs=e2_name_inputs,
                       type_inputs=e2_type_inputs)

    score = torch.cosine_similarity(embedding1, embedding2)
    print('Score = {}'.format(score))


if __name__ == '__main__':
    main()











