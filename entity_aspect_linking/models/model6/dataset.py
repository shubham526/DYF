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
            data_type: str,
            train: bool,
            max_len: int = 512,
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

    def collate(self, batch):
        if not is_empty(batch):
            if self._train:
                if self._data_type == 'pairwise':
                    context_input_ids = torch.tensor([item['context_input_ids'] for item in batch])
                    context_attention_mask = torch.tensor([item['context_attention_mask'] for item in batch])

                    aspect_input_ids_pos = torch.tensor([item['aspect_input_ids_pos'] for item in batch])
                    aspect_attention_mask_pos = torch.tensor([item['aspect_attention_mask_pos'] for item in batch])

                    aspect_input_ids_neg = torch.tensor([item['aspect_input_ids_neg'] for item in batch])
                    aspect_attention_mask_neg = torch.tensor([item['aspect_attention_mask_neg'] for item in batch])

                    aspect_features_pos = torch.tensor([item['aspect_features_pos'] for item in batch])
                    aspect_features_neg = torch.tensor([item['aspect_features_neg'] for item in batch])

                    return {
                        'context_input_ids': context_input_ids,
                        'context_attention_mask': context_attention_mask,
                        'aspect_input_ids_pos': aspect_input_ids_pos,
                        'aspect_attention_mask_pos': aspect_attention_mask_pos,
                        'aspect_input_ids_neg': aspect_input_ids_neg,
                        'aspect_attention_mask_neg': aspect_attention_mask_neg,
                        'aspect_features_pos': aspect_features_pos,
                        'aspect_features_neg': aspect_features_neg

                    }
                elif self._data_type == 'pointwise':

                    context_input_ids = torch.tensor([item['context_input_ids'] for item in batch])
                    context_attention_mask = torch.tensor([item['context_attention_mask'] for item in batch])


                    aspect_input_ids = torch.tensor([item['aspect_input_ids'] for item in batch])
                    aspect_attention_mask = torch.tensor([item['aspect_attention_mask'] for item in batch])


                    aspect_features = torch.tensor([item['aspect_features'] for item in batch])

                    label = torch.tensor([item['label'] for item in batch if item], dtype=torch.float)
                    return {
                        'context_input_ids': context_input_ids,
                        'context_attention_mask': context_attention_mask,
                        'aspect_input_ids': aspect_input_ids,
                        'aspect_attention_mask': aspect_attention_mask,
                        'aspect_features': aspect_features,
                        'label': label
                    }
                else:
                    raise ValueError('Model type must be `pairwise` or `pointwise`.')
            else:
                query_id = [item['query_id'] for item in batch if item]
                doc_id = [item['doc_id'] for item in batch if item]
                label = [item['label'] for item in batch if item]

                context_input_ids = torch.tensor([item['context_input_ids'] for item in batch])
                context_attention_mask = torch.tensor([item['context_attention_mask'] for item in batch])

                aspect_input_ids = torch.tensor([item['aspect_input_ids'] for item in batch])
                aspect_attention_mask = torch.tensor([item['aspect_attention_mask'] for item in batch])

                aspect_features = torch.tensor([item['aspect_features'] for item in batch])

                return {
                    'query_id': query_id,
                    'doc_id': doc_id,
                    'label': label,
                    'context_input_ids': context_input_ids,
                    'context_attention_mask': context_attention_mask,
                    'aspect_input_ids': aspect_input_ids,
                    'aspect_attention_mask': aspect_attention_mask,
                    'aspect_features': aspect_features
                }

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]
        if self._train:
            if self._data_type == 'pairwise':

                # Create the text inputs

                context_input_ids, context_attention_mask = self._create_bert_input(example['query']['text'])
                aspect_input_ids_pos, aspect_attention_mask_pos = self._create_bert_input(example['doc_pos']['text'])
                aspect_input_ids_neg, aspect_attention_mask_neg = self._create_bert_input(example['doc_neg']['text'])
                aspect_features_pos = json.loads(example['doc_pos']['feature'])
                aspect_features_neg = json.loads(example['doc_neg']['feature'])


                return {
                    'context_input_ids': context_input_ids,
                    'context_attention_mask': context_attention_mask,
                    'aspect_input_ids_pos': aspect_input_ids_pos,
                    'aspect_attention_mask_pos': aspect_attention_mask_pos,
                    'aspect_input_ids_neg': aspect_input_ids_neg,
                    'aspect_attention_mask_neg': aspect_attention_mask_neg,
                    'aspect_features_pos': aspect_features_pos,
                    'aspect_features_neg': aspect_features_neg

                }
            elif self._data_type == 'pointwise':

                context_input_ids, context_attention_mask = self._create_bert_input(example['query']['text'])
                aspect_input_ids, aspect_attention_mask = self._create_bert_input(example['doc']['text'])
                aspect_features = json.loads(example['doc']['feature'])

                return {
                    'context_input_ids': context_input_ids,
                    'context_attention_mask': context_attention_mask,
                    'aspect_input_ids': aspect_input_ids,
                    'aspect_attention_mask': aspect_attention_mask,
                    'aspect_features': aspect_features,
                    'label': example['label']
                }
            else:
                raise ValueError('Model type must be `pairwise` or `pointwise`.')
        else:

            context_input_ids, context_attention_mask = self._create_bert_input(example['query']['text'])
            aspect_input_ids, aspect_attention_mask = self._create_bert_input(example['doc']['text'])
            aspect_features = json.loads(example['doc']['feature'])


            return {
                'query_id': example['query_id'],
                'doc_id': example['doc_id'],
                'label': example['label'],
                'context_input_ids': context_input_ids,
                'context_attention_mask': context_attention_mask,
                'aspect_input_ids': aspect_input_ids,
                'aspect_attention_mask': aspect_attention_mask,
                'aspect_features': aspect_features
            }
