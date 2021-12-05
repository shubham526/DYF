from typing import List, Dict, Any
import torch
from torch.utils.data import DataLoader

from dataset import EntitySimilarityDataset, collate_fn


class EntitySimilarityDataLoader(DataLoader):
    def __init__(self,
                 dataset: EntitySimilarityDataset,
                 batch_size: int,
                 shuffle: bool = False,
                 num_workers: int = 0
                 ) -> None:
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
