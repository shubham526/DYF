from typing import Tuple, List
import torch
import numpy as np
import torch.nn as nn
from transformers import AutoConfig, AutoModel, DistilBertModel, AutoTokenizer, get_linear_schedule_with_warmup
import warnings

class BertEmbeddingLayer(nn.Module):
    def __init__(self, pretrained: str) -> None:

        super(BertEmbeddingLayer, self).__init__()
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


class ConvolutionEncoderLayer(nn.Module):
    def __init__(
            self,
            embedding_dimension: int,
            kernel_dim: int,
            kernel_sizes: List[int] = None
    ):
        super(ConvolutionEncoderLayer, self).__init__()

        if kernel_sizes is None:
            kernel_sizes = [1, 3, 5]

        self.kernel_dim = kernel_dim
        self.kernel_sizes = kernel_sizes
        self.embedding_dimension = kernel_dim * len(kernel_sizes)
        self.encoder = nn.ModuleList()

        for kernel_size in kernel_sizes:
            padding_size = int((kernel_size - 1) / 2)
            conv = nn.Conv1d(
                in_channels=embedding_dimension,
                out_channels=kernel_dim,
                kernel_size=kernel_size,
                stride=1,
                padding=padding_size
            )
            self.encoder.append(conv)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings: torch.Tensor = embeddings.transpose(1,-1)
        vectors = [encoder(embeddings) for encoder in self.encoder]
        out = torch.cat(vectors, 1).transpose(1, -1)
        return out

class EmbeddingCombinationLayer(nn.Module):
    def __init__(self, mode: str, input_dim: int, output_dim: int = 100):
        super(EmbeddingCombinationLayer, self).__init__()
        self.mode = mode
        self.input_dim = int(input_dim / 2) if mode == 'weighted' else input_dim
        self.output_dim = output_dim
        self.dense = nn.Linear(self.input_dim, self.output_dim)
        self.alpha = nn.Parameter(torch.rand(1, requires_grad=True))

    def _weighted_average(self, embedding1: torch.Tensor, embedding2: torch.Tensor):
        combined_embedding: torch.Tensor = embedding1 * self.alpha + embedding2 * (1 - self.alpha)
        combined_embedding = self.dense(combined_embedding)
        combined_embedding = torch.mean(combined_embedding, dim=1).squeeze()
        return combined_embedding


    def _concat(self, embedding1: torch.Tensor, embedding2: torch.Tensor):
        combined_embedding: torch.Tensor = torch.cat((embedding1, embedding2), dim=2)
        combined_embedding = self.dense(combined_embedding)
        combined_embedding = torch.mean(combined_embedding, dim=1).squeeze()
        return combined_embedding

    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor):
        if self.mode == 'weighted':
            if embedding1.shape != embedding2.shape:
                print('Cannot do weighted average of two tensors with different shapes.\n'
                      'Shape of embedding1={}.\n'
                      'Shape of embedding2={}.\n'
                      'Defaulting to concatenation'.format(embedding1.shape, embedding2.shape)
                )
                return self._concat(embedding1=embedding1, embedding2=embedding2)
            else:
                return self._weighted_average(embedding1=embedding1, embedding2=embedding2)
        else:
            return self._concat(embedding1=embedding1, embedding2=embedding2)


class Encoder(nn.Module):
    def __init__(
            self,
            pretrained_bert_model: str,
            n_gram_sizes: List[int] = None,
            out_dim: int = 100,
            mode: str='concat'
    ):
        super(Encoder, self).__init__()
        self.embedding = BertEmbeddingLayer(pretrained=pretrained_bert_model)
        self.encoder = ConvolutionEncoderLayer(embedding_dimension=768, kernel_dim=out_dim, kernel_sizes=n_gram_sizes)
        self.input_embedding_dimension = self.encoder.embedding_dimension + 768
        self.output_embedding_dim = out_dim
        self.mode = mode
        self.combination = EmbeddingCombinationLayer(mode=mode,
                                                     input_dim=self.input_embedding_dimension,
                                                     output_dim=self.output_embedding_dim)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor = None,
            token_type_ids: torch.Tensor = None,
    ):
        bert_embedding: torch.Tensor = self.embedding(input_ids, attention_mask, token_type_ids)
        conv_embedding: torch.Tensor = self.encoder(bert_embedding)
        combined_embedding: torch.Tensor = self.combination(bert_embedding, conv_embedding)

        # concatenated_embedding: torch.Tensor = torch.cat((bert_embedding, conv_embedding), dim=2)
        # out_embedding: torch.Tensor = self.dense(concatenated_embedding)
        # out_embedding = torch.mean(out_embedding, dim=1).squeeze()
        return combined_embedding

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
    return encoded_dict['input_ids'], encoded_dict['attention_mask']


def main():
    pretrain = 'bert-base-uncased'
    vocab = 'bert-base-uncased'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(vocab)
    model = Encoder(pretrained_bert_model=pretrain, out_dim=100, mode='weighted', n_gram_sizes=[1,3,5,7])
    model.to(device)
    model.train()
    text = 'Today is a rainy day. I need an umbrella.'
    input_ids, attention_mask = create_bert_input(text, tokenizer)
    embedding = model(input_ids=input_ids,attention_mask=attention_mask)
    print('hi')


if __name__ == '__main__':
    main()










