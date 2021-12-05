from typing import List, Tuple, Dict, Any
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def collate_fn(batch):

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


class EntitySimilarityDataset(Dataset):
    def __init__(
            self,
            dataset,
            tokenizer: AutoTokenizer,
            max_len: int = 512,
    ) -> None:

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



