from typing import List, Tuple, Dict, Any
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pad(lst: List[Any]):
    max_len = len(max(lst, key=lambda i: len(i)))
    padded = [
        F.pad(
            input=t.squeeze(dim=1),
            pad=(0, 0, 0, (max_len - len(t))),
            mode="constant",
            value=0
        )
        for t in lst if len(t) <= max_len
    ]
    return torch.stack(padded)

class AspectLinkDataset(Dataset):
    def __init__(
            self,
            dataset,
            tokenizer: AutoTokenizer,
            max_len: int,
            data_type: str,
            train: bool
    ):

        self._dataset = dataset
        self._tokenizer = tokenizer
        self._max_len = max_len
        self._data_type = data_type
        self._train = train
        self._read_data()
        self._count = len(self._examples)

    def _read_data(self):
        with open(self._dataset, 'r') as f:
            self._examples = [json.loads(line) for line in f]

    def _create_bert_input(self, text) -> Tuple[torch.Tensor, torch.Tensor]:
        # Tokenize all of the sentences and map the tokens to their word IDs.
        encoded_dict = self._tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self._max_len,  # Pad & truncate all sentences.
            padding='max_length',
            truncation=True,
            return_attention_mask=True,  # Construct attn. masks.
        )

        return encoded_dict['input_ids'], encoded_dict['attention_mask']

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]
        if self._train:
            if self._data_type == 'pairwise':
                context_entity_desc_input_ids_and_attention_masks = [
                    self._create_bert_input(entity['desc'])
                    for entity in example['query']['entities']
                ]
                context_entity_wiki2vec_embeddings = [
                    entity['wiki2vec']
                    for entity in example['query']['entities']
                ]

                aspect_entity_desc_input_ids_and_attention_masks_pos = [
                    self._create_bert_input(entity['desc'])
                    for entity in example['doc_pos']['entities']
                ]
                aspect_entity_wiki2vec_embeddings_pos = [
                    entity['wiki2vec']
                    for entity in example['doc_pos']['entities']
                ]

                aspect_entity_desc_input_ids_and_attention_masks_neg = [
                    self._create_bert_input(entity['desc'])
                    for entity in example['doc_neg']['entities']
                ]
                aspect_entity_wiki2vec_embeddings_neg = [
                    entity['wiki2vec']
                    for entity in example['doc_neg']['entities']
                ]

                return {
                    'context_entity_desc_input_ids_and_attention_masks': context_entity_desc_input_ids_and_attention_masks,
                    'context_entity_wiki2vec_embeddings': context_entity_wiki2vec_embeddings,
                    'aspect_entity_desc_input_ids_and_attention_masks_pos': aspect_entity_desc_input_ids_and_attention_masks_pos,
                    'aspect_entity_wiki2vec_embeddings_pos': aspect_entity_wiki2vec_embeddings_pos,
                    'aspect_entity_desc_input_ids_and_attention_masks_neg': aspect_entity_desc_input_ids_and_attention_masks_neg,
                    'aspect_entity_wiki2vec_embeddings_neg': aspect_entity_wiki2vec_embeddings_neg

                }
            elif self._data_type == 'pointwise':
                context_entity_desc_input_ids_and_attention_masks = [
                    self._create_bert_input(entity['desc'])
                    for entity in example['query']['entities']
                ]
                context_entity_wiki2vec_embeddings = [
                    entity['wiki2vec']
                    for entity in example['query']['entities']
                ]

                aspect_entity_desc_input_ids_and_attention_masks = [
                    self._create_bert_input(entity['desc'])
                    for entity in example['doc']['entities']
                ]
                aspect_entity_wiki2vec_embeddings = [
                    entity['wiki2vec']
                    for entity in example['doc']['entities']
                ]

                return {
                    'context_entity_desc_input_ids_and_attention_masks': context_entity_desc_input_ids_and_attention_masks,
                    'context_entity_wiki2vec_embeddings': context_entity_wiki2vec_embeddings,
                    'aspect_entity_desc_input_ids_and_attention_masks': aspect_entity_desc_input_ids_and_attention_masks,
                    'aspect_entity_wiki2vec_embeddings': aspect_entity_wiki2vec_embeddings,
                    'label': example['label']
                }
            else:
                raise ValueError('Model type must be `pairwise` or `pointwise`.')
        else:
            context_entity_desc_input_ids_and_attention_masks = [
                self._create_bert_input(entity['desc'])
                for entity in example['query']['entities']
            ]
            context_entity_wiki2vec_embeddings = [
                entity['wiki2vec']
                for entity in example['query']['entities']
            ]

            aspect_entity_desc_input_ids_and_attention_masks = [
                self._create_bert_input(entity['desc'])
                for entity in example['doc']['entities']
            ]
            aspect_entity_wiki2vec_embeddings = [
                entity['wiki2vec']
                for entity in example['doc']['entities']
            ]

            return {
                'query_id': example['query_id'],
                'doc_id': example['doc_id'],
                'label': example['label'],
                'context_entity_desc_input_ids_and_attention_masks': context_entity_desc_input_ids_and_attention_masks,
                'context_entity_wiki2vec_embeddings': context_entity_wiki2vec_embeddings,
                'aspect_entity_desc_input_ids_and_attention_masks': aspect_entity_desc_input_ids_and_attention_masks,
                'aspect_entity_wiki2vec_embeddings': aspect_entity_wiki2vec_embeddings,
            }

    def collate(self, batch):
        if self._train:
            if self._data_type == 'pairwise':
                context_entity_desc_input_ids = [
                    torch.tensor(
                        [inputs_ids for inputs_ids, _ in item['context_entity_desc_input_ids_and_attention_masks']])
                    for item in batch
                ]
                context_entity_desc_attention_mask = [
                    torch.tensor(
                        [attention_mask for _, attention_mask in
                         item['context_entity_desc_input_ids_and_attention_masks']])
                    for item in batch
                ]
                context_entity_wiki2vec_embeddings = [
                    torch.tensor(item['context_entity_wiki2vec_embeddings']) for item in batch
                ]

                aspect_entity_desc_input_ids_pos = [
                    torch.tensor(
                        [inputs_ids for inputs_ids, _ in item['aspect_entity_desc_input_ids_and_attention_masks_pos']])
                    for item in batch
                ]
                aspect_entity_desc_attention_mask_pos = [
                    torch.tensor(
                        [attention_mask for _, attention_mask in
                         item['aspect_entity_desc_input_ids_and_attention_masks_pos']])
                    for item in batch
                ]
                aspect_entity_wiki2vec_embeddings_pos = [
                    torch.tensor(item['aspect_entity_wiki2vec_embeddings_pos']) for item in batch
                ]

                aspect_entity_desc_input_ids_neg = [
                    torch.tensor(
                        [inputs_ids for inputs_ids, _ in item['aspect_entity_desc_input_ids_and_attention_masks_neg']])
                    for item in batch
                ]
                aspect_entity_desc_attention_mask_neg = [
                    torch.tensor(
                        [attention_mask for _, attention_mask in
                         item['aspect_entity_desc_input_ids_and_attention_masks_neg']])
                    for item in batch
                ]
                aspect_entity_wiki2vec_embeddings_neg = [
                    torch.tensor(item['aspect_entity_wiki2vec_embeddings_neg']) for item in batch
                ]

                return {
                    'context_entity_desc_input_ids': context_entity_desc_input_ids,
                    'context_entity_desc_attention_mask': context_entity_desc_attention_mask,
                    'context_entity_wiki2vec_embeddings': context_entity_wiki2vec_embeddings,
                    'aspect_entity_desc_input_ids_pos': aspect_entity_desc_input_ids_pos,
                    'aspect_entity_desc_attention_mask_pos': aspect_entity_desc_attention_mask_pos,
                    'aspect_entity_wiki2vec_embeddings_pos': aspect_entity_wiki2vec_embeddings_pos,
                    'aspect_entity_desc_input_ids_neg': aspect_entity_desc_input_ids_neg,
                    'aspect_entity_desc_attention_mask_neg': aspect_entity_desc_attention_mask_neg,
                    'aspect_entity_wiki2vec_embeddings_neg': aspect_entity_wiki2vec_embeddings_neg

                }
            elif self._data_type == 'pointwise':

                context_entity_desc_input_ids = [
                    torch.tensor(
                        [inputs_ids for inputs_ids, _ in item['context_entity_desc_input_ids_and_attention_masks']])
                        for item in batch
                ]
                context_entity_desc_attention_mask = [
                    torch.tensor(
                        [attention_mask for _, attention_mask in item['context_entity_desc_input_ids_and_attention_masks']])
                    for item in batch
                ]
                context_entity_wiki2vec_embeddings = [
                    torch.tensor(item['context_entity_wiki2vec_embeddings']) for item in batch
                ]


                aspect_entity_desc_input_ids = [
                    torch.tensor(
                        [inputs_ids for inputs_ids, _ in item['aspect_entity_desc_input_ids_and_attention_masks']])
                        for item in batch
                ]
                aspect_entity_desc_attention_mask = [
                    torch.tensor(
                        [attention_mask for _, attention_mask in item['aspect_entity_desc_input_ids_and_attention_masks']])
                        for item in batch
                ]
                aspect_entity_wiki2vec_embeddings = [
                    torch.tensor(item['aspect_entity_wiki2vec_embeddings']) for item in batch
                ]

                label = torch.tensor([item['label'] for item in batch if item], dtype=torch.float)

                return {
                    'context_entity_desc_input_ids': context_entity_desc_input_ids,
                    'context_entity_desc_attention_mask': context_entity_desc_attention_mask,
                    'context_entity_wiki2vec_embeddings': context_entity_wiki2vec_embeddings,
                    'aspect_entity_desc_input_ids': aspect_entity_desc_input_ids,
                    'aspect_entity_desc_attention_mask': aspect_entity_desc_attention_mask,
                    'aspect_entity_wiki2vec_embeddings': aspect_entity_wiki2vec_embeddings,
                    'label': label
                }
            else:
                raise ValueError('Model type must be `pairwise` or `pointwise`.')
        else:
            query_id = [item['query_id'] for item in batch if item]
            doc_id = [item['doc_id'] for item in batch if item]
            label = [item['label'] for item in batch if item]

            context_entity_desc_input_ids = [
                torch.tensor(
                    [inputs_ids for inputs_ids, _ in item['context_entity_desc_input_ids_and_attention_masks']])
                for item in batch
            ]
            context_entity_desc_attention_mask = [
                torch.tensor(
                    [attention_mask for _, attention_mask in item['context_entity_desc_input_ids_and_attention_masks']])
                for item in batch
            ]
            context_entity_wiki2vec_embeddings = [
                torch.tensor(item['context_entity_wiki2vec_embeddings']) for item in batch
            ]


            aspect_entity_desc_input_ids = [
                torch.tensor(
                    [inputs_ids for inputs_ids, _ in item['aspect_entity_desc_input_ids_and_attention_masks']])
                for item in batch
            ]
            aspect_entity_desc_attention_mask = [
                torch.tensor(
                    [attention_mask for _, attention_mask in item['aspect_entity_desc_input_ids_and_attention_masks']])
                for item in batch
            ]
            aspect_entity_wiki2vec_embeddings = [
                torch.tensor(item['aspect_entity_wiki2vec_embeddings']) for item in batch
            ]
            return {
                'query_id': query_id,
                'doc_id': doc_id,
                'label': label,
                'context_entity_desc_input_ids': context_entity_desc_input_ids,
                'context_entity_desc_attention_mask': context_entity_desc_attention_mask,
                'context_entity_wiki2vec_embeddings': context_entity_wiki2vec_embeddings,
                'aspect_entity_desc_input_ids': aspect_entity_desc_input_ids,
                'aspect_entity_desc_attention_mask': aspect_entity_desc_attention_mask,
                'aspect_entity_wiki2vec_embeddings': aspect_entity_wiki2vec_embeddings,
            }
