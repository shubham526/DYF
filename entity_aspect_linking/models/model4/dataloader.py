from typing import List, Dict, Any
import torch
from torch.utils.data import DataLoader

from dataset import AspectLinkDataset


class AspectLinkDataLoader(DataLoader):
    def __init__(self,
                 dataset: AspectLinkDataset,
                 batch_size: int,
                 shuffle: bool = False,
                 num_workers: int = 0
                 ) -> None:
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=dataset.collate
        )