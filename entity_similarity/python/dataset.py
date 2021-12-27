import abc
from typing import List, Tuple, Dict, Any
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class Collator:
    def __init__(self, dataset):
        self._collate_type = 'pair' if isinstance(dataset, LabelledEntityPairDataset) else 'triplet'

    def collate_fn(self, batch):
        if self._collate_type == 'pair':
            context_input_ids = torch.tensor([item['context_input_ids'] for item in batch])
            context_attention_mask = torch.tensor([item['context_attention_mask'] for item in batch])

            e1_desc_input_ids = torch.tensor([item['e1_desc_input_ids'] for item in batch])
            e1_desc_attention_mask = torch.tensor([item['e1_desc_attention_mask'] for item in batch])

            e1_name_input_ids = torch.tensor([item['e1_name_input_ids'] for item in batch])
            e1_name_attention_mask = torch.tensor([item['e1_name_attention_mask'] for item in batch])

            e1_type_input_ids = torch.tensor([item['e1_type_input_ids'] for item in batch])
            e1_type_attention_mask = torch.tensor([item['e1_type_attention_mask'] for item in batch])

            e2_desc_input_ids = torch.tensor([item['e2_desc_input_ids'] for item in batch])
            e2_desc_attention_mask = torch.tensor([item['e2_desc_attention_mask'] for item in batch])

            e2_name_input_ids = torch.tensor([item['e2_name_input_ids'] for item in batch])
            e2_name_attention_mask = torch.tensor([item['e2_name_attention_mask'] for item in batch])

            e2_type_input_ids = torch.tensor([item['e2_type_input_ids'] for item in batch])
            e2_type_attention_mask = torch.tensor([item['e2_type_attention_mask'] for item in batch])

            label = torch.tensor([item['label'] for item in batch])

            return {
                'context_input_ids': context_input_ids,
                'context_attention_mask': context_attention_mask,
                'e1_desc_input_ids': e1_desc_input_ids,
                'e1_desc_attention_mask': e1_desc_attention_mask,
                'e1_name_input_ids': e1_name_input_ids,
                'e1_name_attention_mask': e1_name_attention_mask,
                'e1_type_input_ids': e1_type_input_ids,
                'e1_type_attention_mask': e1_type_attention_mask,
                'e2_desc_input_ids': e2_desc_input_ids,
                'e2_desc_attention_mask': e2_desc_attention_mask,
                'e2_name_input_ids': e2_name_input_ids,
                'e2_name_attention_mask': e2_name_attention_mask,
                'e2_type_input_ids': e2_type_input_ids,
                'e2_type_attention_mask': e2_type_attention_mask,
                'label': label
            }
        else:
            context_input_ids = torch.tensor([item['context_input_ids'] for item in batch])
            context_attention_mask = torch.tensor([item['context_attention_mask'] for item in batch])

            anchor_desc_input_ids = torch.tensor([item['anchor_desc_input_ids'] for item in batch])
            anchor_desc_attention_mask = torch.tensor([item['anchor_desc_attention_mask'] for item in batch])

            anchor_name_input_ids = torch.tensor([item['anchor_name_input_ids'] for item in batch])
            anchor_name_attention_mask = torch.tensor([item['anchor_name_attention_mask'] for item in batch])

            anchor_type_input_ids = torch.tensor([item['anchor_type_input_ids'] for item in batch])
            anchor_type_attention_mask = torch.tensor([item['anchor_type_attention_mask'] for item in batch])

            pos_desc_input_ids = torch.tensor([item['pos_desc_input_ids'] for item in batch])
            pos_desc_attention_mask = torch.tensor([item['pos_desc_attention_mask'] for item in batch])

            pos_name_input_ids = torch.tensor([item['pos_name_input_ids'] for item in batch])
            pos_name_attention_mask = torch.tensor([item['pos_name_attention_mask'] for item in batch])

            pos_type_input_ids = torch.tensor([item['pos_type_input_ids'] for item in batch])
            pos_type_attention_mask = torch.tensor([item['pos_type_attention_mask'] for item in batch])

            neg_desc_input_ids = torch.tensor([item['neg_desc_input_ids'] for item in batch])
            neg_desc_attention_mask = torch.tensor([item['neg_desc_attention_mask'] for item in batch])

            neg_name_input_ids = torch.tensor([item['neg_name_input_ids'] for item in batch])
            neg_name_attention_mask = torch.tensor([item['neg_name_attention_mask'] for item in batch])

            neg_type_input_ids = torch.tensor([item['neg_type_input_ids'] for item in batch])
            neg_type_attention_mask = torch.tensor([item['neg_type_attention_mask'] for item in batch])


            return {
                'context_input_ids': context_input_ids,
                'context_attention_mask': context_attention_mask,
                'anchor_desc_input_ids': anchor_desc_input_ids,
                'anchor_desc_attention_mask': anchor_desc_attention_mask,
                'anchor_name_input_ids': anchor_name_input_ids,
                'anchor_name_attention_mask': anchor_name_attention_mask,
                'anchor_type_input_ids': anchor_type_input_ids,
                'anchor_type_attention_mask': anchor_type_attention_mask,
                'pos_desc_input_ids': pos_desc_input_ids,
                'pos_desc_attention_mask': pos_desc_attention_mask,
                'pos_name_input_ids': pos_name_input_ids,
                'pos_name_attention_mask': pos_name_attention_mask,
                'pos_type_input_ids': pos_type_input_ids,
                'pos_type_attention_mask': pos_type_attention_mask,
                'neg_desc_input_ids': neg_desc_input_ids,
                'neg_desc_attention_mask': neg_desc_attention_mask,
                'neg_name_input_ids': neg_name_input_ids,
                'neg_name_attention_mask': neg_name_attention_mask,
                'neg_type_input_ids': neg_type_input_ids,
                'neg_type_attention_mask': neg_type_attention_mask,
            }


class EntityDataset(Dataset):
    def __init__(self, dataset, tokenizer: AutoTokenizer, max_len: int) -> None:

        self._dataset = dataset
        self._tokenizer = tokenizer
        self._max_len = max_len
        self._read_data()
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

    def __len__(self) -> int:
        return self._count

    @abc.abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
       pass


class LabelledEntityPairDataset(EntityDataset):
    def __init__( self, dataset, tokenizer: AutoTokenizer, max_len: int):
        super(LabelledEntityPairDataset, self).__init__(dataset=dataset, tokenizer=tokenizer, max_len=max_len)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]
        context_input_ids, context_attention_mask = self._create_bert_input(example['context'])

        e1_desc_input_ids, e1_desc_attention_mask = self._create_bert_input(example['doc1']['entity_desc'])
        e1_name_input_ids, e1_name_attention_mask = self._create_bert_input(example['doc1']['entity_name'])
        e1_type_input_ids, e1_type_attention_mask = self._create_bert_input(' '.join(example['doc1']['entity_types']))

        e2_desc_input_ids, e2_desc_attention_mask = self._create_bert_input(example['doc2']['entity_desc'])
        e2_name_input_ids, e2_name_attention_mask = self._create_bert_input(example['doc2']['entity_name'])
        e2_type_input_ids, e2_type_attention_mask = self._create_bert_input(' '.join(example['doc2']['entity_types']))

        return {
            'context_input_ids': context_input_ids,
            'context_attention_mask': context_attention_mask,
            'e1_desc_input_ids': e1_desc_input_ids,
            'e1_desc_attention_mask': e1_desc_attention_mask,
            'e1_name_input_ids': e1_name_input_ids,
            'e1_name_attention_mask': e1_name_attention_mask,
            'e1_type_input_ids': e1_type_input_ids,
            'e1_type_attention_mask': e1_type_attention_mask,
            'e2_desc_input_ids': e2_desc_input_ids,
            'e2_desc_attention_mask': e2_desc_attention_mask,
            'e2_name_input_ids': e2_name_input_ids,
            'e2_name_attention_mask': e2_name_attention_mask,
            'e2_type_input_ids': e2_type_input_ids,
            'e2_type_attention_mask': e2_type_attention_mask,
            'label': example['label']
        }

class TripletDataset(EntityDataset):
    def __init__( self, dataset, tokenizer: AutoTokenizer, max_len: int):
        super(TripletDataset, self).__init__(dataset=dataset, tokenizer=tokenizer, max_len=max_len)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]
        context_input_ids, context_attention_mask = self._create_bert_input(example['context'])

        anchor_desc_input_ids, anchor_desc_attention_mask = self._create_bert_input(example['anchor']['entity_desc'])
        anchor_name_input_ids, anchor_name_attention_mask = self._create_bert_input(example['anchor']['entity_name'])
        anchor_type_input_ids, anchor_type_attention_mask = self._create_bert_input(' '.join(example['anchor']['entity_types']))

        pos_desc_input_ids, pos_desc_attention_mask = self._create_bert_input(example['positive']['entity_desc'])
        pos_name_input_ids, pos_name_attention_mask = self._create_bert_input(example['positive']['entity_name'])
        pos_type_input_ids, pos_type_attention_mask = self._create_bert_input(' '.join(example['positive']['entity_types']))

        neg_desc_input_ids, neg_desc_attention_mask = self._create_bert_input(example['negative']['entity_desc'])
        neg_name_input_ids, neg_name_attention_mask = self._create_bert_input(example['negative']['entity_name'])
        neg_type_input_ids, neg_type_attention_mask = self._create_bert_input(' '.join(example['negative']['entity_types']))

        return {
            'context_input_ids': context_input_ids,
            'context_attention_mask': context_attention_mask,
            'anchor_desc_input_ids': anchor_desc_input_ids,
            'anchor_desc_attention_mask': anchor_desc_attention_mask,
            'anchor_name_input_ids': anchor_name_input_ids,
            'anchor_name_attention_mask': anchor_name_attention_mask,
            'anchor_type_input_ids': anchor_type_input_ids,
            'anchor_type_attention_mask': anchor_type_attention_mask,
            'pos_desc_input_ids': pos_desc_input_ids,
            'pos_desc_attention_mask': pos_desc_attention_mask,
            'pos_name_input_ids': pos_name_input_ids,
            'pos_name_attention_mask': pos_name_attention_mask,
            'pos_type_input_ids': pos_type_input_ids,
            'pos_type_attention_mask': pos_type_attention_mask,
            'neg_desc_input_ids': neg_desc_input_ids,
            'neg_desc_attention_mask': neg_desc_attention_mask,
            'neg_name_input_ids': neg_name_input_ids,
            'neg_name_attention_mask': neg_name_attention_mask,
            'neg_type_input_ids': neg_type_input_ids,
            'neg_type_attention_mask': neg_type_attention_mask,
        }