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

    def _create_bert_input(self, query, document) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Tokenize all of the sentences and map the tokens to their word IDs.
        encoded_dict = self._tokenizer.encode_plus(
            text=query,
            text_pair=document,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self._max_len,  # Pad & truncate all sentences.
            padding='max_length',
            truncation=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_token_type_ids=True  # Construct token type ids
        )

        return encoded_dict['input_ids'], encoded_dict['attention_mask'], encoded_dict['token_type_ids']

    def collate(self, batch):
        if not is_empty(batch):
            if self._train:
                if self._data_type == 'pairwise':
                    input_ids_pos = torch.tensor([item['input_ids_pos'] for item in batch])
                    token_type_ids_pos = torch.tensor([item['token_type_ids_pos'] for item in batch])
                    attention_mask_pos = torch.tensor([item['attention_mask_pos'] for item in batch])
                    input_ids_neg = torch.tensor([item['input_ids_neg'] for item in batch])
                    token_type_ids_neg = torch.tensor([item['token_type_ids_neg'] for item in batch])
                    attention_mask_neg = torch.tensor([item['attention_mask_neg'] for item in batch])
                    features_pos = torch.tensor([item['features_pos'] for item in batch])
                    features_neg = torch.tensor([item['features_neg'] for item in batch])

                    return {
                        'input_ids_pos': input_ids_pos,
                        'token_type_ids_pos': token_type_ids_pos,
                        'attention_mask_pos': attention_mask_pos,
                        'input_ids_neg': input_ids_neg,
                        'token_type_ids_neg': token_type_ids_neg,
                        'attention_mask_neg': attention_mask_neg,
                        'features_pos': features_pos,
                        'features_neg': features_neg

                    }
                elif self._data_type == 'pointwise':

                    input_ids = torch.tensor([item['input_ids'] for item in batch])
                    token_type_ids = torch.tensor([item['token_type_ids'] for item in batch])
                    attention_mask = torch.tensor([item['attention_mask'] for item in batch])
                    features = torch.tensor([item['features'] for item in batch])
                    label = torch.tensor([item['label'] for item in batch if item], dtype=torch.float)
                    return {
                        'input_ids': input_ids,
                        'token_type_ids': token_type_ids,
                        'attention_mask': attention_mask,
                        'features': features,
                        'label': label
                    }
                else:
                    raise ValueError('Model type must be `pairwise` or `pointwise`.')
            else:
                query_id = [item['query_id'] for item in batch if item]
                doc_id = [item['doc_id'] for item in batch if item]
                label = [item['label'] for item in batch if item]
                input_ids = torch.tensor([item['input_ids'] for item in batch])
                token_type_ids = torch.tensor([item['token_type_ids'] for item in batch])
                attention_mask = torch.tensor([item['attention_mask'] for item in batch])
                features = torch.tensor([item['features'] for item in batch])

                return {
                    'query_id': query_id,
                    'doc_id': doc_id,
                    'label': label,
                    'input_ids': input_ids,
                    'token_type_ids': token_type_ids,
                    'attention_mask': attention_mask,
                    'features': features
                }

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]
        if self._train:
            if self._data_type == 'pairwise':

                # Create the text inputs
                input_ids_pos, attention_mask_pos, token_type_ids_pos = self._create_bert_input(example['query']['text'],
                                                                                                example['doc_pos']['text'])
                input_ids_neg, attention_mask_neg, token_type_ids_neg = self._create_bert_input(example['query']['text'],
                                                                                                example['doc_neg']['text'])
                features_pos = json.loads(example['doc_pos']['feature'])
                features_neg = json.loads(example['doc_neg']['feature'])


                return {
                    'input_ids_pos': input_ids_pos,
                    'token_type_ids_pos': token_type_ids_pos,
                    'attention_mask_pos': attention_mask_pos,
                    'input_ids_neg': input_ids_neg,
                    'token_type_ids_neg': token_type_ids_neg,
                    'attention_mask_neg': attention_mask_neg,
                    'features_pos': features_pos,
                    'features_neg': features_neg

                }
            elif self._data_type == 'pointwise':

                input_ids, attention_mask, token_type_ids = self._create_bert_input(example['query']['text'],
                                                                                    example['doc']['text'])
                features = json.loads(example['doc']['feature'])

                return {
                    'input_ids': input_ids,
                    'token_type_ids': token_type_ids,
                    'attention_mask': attention_mask,
                    'features': features,
                    'label': example['label']
                }
            else:
                raise ValueError('Model type must be `pairwise` or `pointwise`.')
        else:

            input_ids, attention_mask, token_type_ids = self._create_bert_input(example['query']['text'],
                                                                                example['doc']['text'])
            features = json.loads(example['doc']['feature'])


            return {
                'query_id': example['query_id'],
                'doc_id': example['doc_id'],
                'label': example['label'],
                'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'features': features,
            }
