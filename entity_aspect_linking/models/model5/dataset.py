from typing import List, Tuple, Dict, Any
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def is_empty(batch):
    empty = True
    for item in batch:
        if item:
            empty = False
            break
    return empty

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

def _is_value_dict(d):
    for x in d.values():
        if isinstance(x, dict):
            return True
        else:
            return False

class AspectLinkDataset(Dataset):
    def __init__(
            self,
            dataset,
            tokenizer: AutoTokenizer,
            max_len: int,
            data_type: str,
            entity_embeddings,
            train: bool
    ):

        self._dataset = dataset
        self._tokenizer = tokenizer
        self._max_len = max_len
        self._data_type = data_type
        self._train = train
        self._entity_embeddings = entity_embeddings

        print('Reading data...')
        self._read_data()
        print('[Done].')

        self._count = len(self._examples)

    def _read_data(self):
        with open(self._dataset, 'r') as f:
            self._examples = [json.loads(line) for i, line in enumerate(f)]

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


    def collate(self, batch):
        if not is_empty(batch):
            if self._train:
                if self._data_type == 'pairwise':
                    context_entity_embeddings = pad([item['context_entity_embeddings'] for item in batch if item])
                    aspect_entity_embeddings_pos = pad([item['aspect_entity_embeddings_pos'] for item in batch if item])
                    aspect_entity_embeddings_neg = pad([item['aspect_entity_embeddings_neg'] for item in batch if item])

                    return {
                        'context_entity_embeddings': context_entity_embeddings,
                        'aspect_entity_embeddings_pos': aspect_entity_embeddings_pos,
                        'aspect_entity_embeddings_neg': aspect_entity_embeddings_neg
                    }
                elif self._data_type == 'pointwise':

                    context_entity_embeddings = pad([item['context_entity_embeddings'] for item in batch if item])
                    aspect_entity_embeddings = pad([item['aspect_entity_embeddings'] for item in batch if item])

                    label = torch.tensor([item['label'] for item in batch if item], dtype=torch.float)
                    return {
                        'context_entity_embeddings': context_entity_embeddings,
                        'aspect_entity_embeddings': aspect_entity_embeddings,
                        'label': label
                    }
                else:
                    raise ValueError('Model type must be `pairwise` or `pointwise`.')
            else:
                query_id = [item['query_id'] for item in batch if item]
                doc_id = [item['doc_id'] for item in batch if item]
                label = [item['label'] for item in batch if item]

                context_entity_embeddings = pad([item['context_entity_embeddings'] for item in batch if item])
                aspect_entity_embeddings = pad([item['aspect_entity_embeddings'] for item in batch if item])

                return {
                    'query_id': query_id,
                    'doc_id': doc_id,
                    'label': label,
                    'context_entity_embeddings': context_entity_embeddings,
                    'aspect_entity_embeddings': aspect_entity_embeddings,
                }

    def _get_embeddings(self, query_id, entities):

        if _is_value_dict(self._entity_embeddings) and query_id in self._entity_embeddings:
            return [
                self._entity_embeddings[query_id][entity_id]
                for entity_id in entities if entity_id in self._entity_embeddings[query_id]
            ]
        else:
            return [
                self._entity_embeddings[entity_id]
                for entity_id in entities if entity_id in self._entity_embeddings
            ]

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]
        if self._train:
            if self._data_type == 'pairwise':

                context_entity_embeddings = self._get_embeddings(
                    query_id=example['query_id'],
                    entities=example['query']['entities']
                )

                aspect_entity_embeddings_pos = self._get_embeddings(
                    query_id=example['query_id'],
                    entities=example['doc_pos']['entities']
                )

                aspect_entity_embeddings_neg = self._get_embeddings(
                    query_id=example['query_id'],
                    entities=example['doc_neg']['entities']
                )

                if len(context_entity_embeddings) == 0 or len(aspect_entity_embeddings_pos) == 0 or len(
                        aspect_entity_embeddings_neg) == 0:
                    return {}

                return {
                    'context_entity_embeddings': torch.stack(context_entity_embeddings),
                    'aspect_entity_embeddings_pos': torch.stack(aspect_entity_embeddings_pos),
                    'aspect_entity_embeddings_neg': torch.stack(aspect_entity_embeddings_neg),
                }
            elif self._data_type == 'pointwise':

                context_entity_embeddings = self._get_embeddings(
                    query_id=example['query_id'],
                    entities=example['query']['entities']
                )

                aspect_entity_embeddings = self._get_embeddings(
                    query_id=example['query_id'],
                    entities=example['doc']['entities']
                )

                if len(context_entity_embeddings) == 0 or len(aspect_entity_embeddings) == 0:
                    return {}



                return {
                    'context_entity_embeddings': torch.stack(context_entity_embeddings),
                    'aspect_entity_embeddings': torch.stack(aspect_entity_embeddings),
                    'label': example['label']
                }
            else:
                raise ValueError('Model type must be `pairwise` or `pointwise`.')
        else:
            context_entity_embeddings = self._get_embeddings(
                query_id=example['query_id'],
                entities=example['query']['entities']
            )


            aspect_entity_embeddings = self._get_embeddings(
                query_id=example['query_id'],
                entities=example['doc']['entities']
            )

            if len(context_entity_embeddings) == 0 or len(aspect_entity_embeddings) == 0:
                return {}


            return {
                'query_id': example['query_id'],
                'doc_id': example['doc_id'],
                'label': example['label'],
                'context_entity_embeddings': torch.stack(context_entity_embeddings),
                'aspect_entity_embeddings': torch.stack(aspect_entity_embeddings),
            }


