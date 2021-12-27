import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, distance_function: str = 'cosine', margin: float = 5, reduction: str = 'mean'):
        super(TripletLoss, self).__init__()
        if distance_function == 'manhattan':
            self.distance_function =  lambda x, y: F.pairwise_distance(x, y, p=1)
        elif distance_function == 'euclidean':
            self.distance_function = lambda x, y: F.pairwise_distance(x, y, p=2)
        else:
            self.distance_function = lambda x, y: 1-F.cosine_similarity(x, y)
        self.margin = margin
        self.reduction = reduction

    def forward(self, anchor_embedding: torch.Tensor, pos_embedding: torch.Tensor, neg_embedding: torch.Tensor):
        distance_pos = self.distance_function(anchor_embedding, pos_embedding)
        distance_neg = self.distance_function(anchor_embedding, neg_embedding)
        losses = F.relu(distance_pos - distance_neg + self.margin)
        if self.reduction == 'none':
            return losses
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses.mean()

class ContrastiveLoss(nn.Module):
    def __init__(self, distance_function: str = 'cosine', margin: float = 0.5, reduction: str = 'mean'):
        super(ContrastiveLoss, self).__init__()
        if distance_function == 'manhattan':
            self.distance_function =  lambda x, y: F.pairwise_distance(x, y, p=1)
        elif distance_function == 'euclidean':
            self.distance_function = lambda x, y: F.pairwise_distance(x, y, p=2)
        else:
            self.distance_function = lambda x, y: 1-F.cosine_similarity(x, y)
        self.margin = margin
        self.reduction = reduction

    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor, labels: torch.Tensor):
        distances = self.distance_function(embedding1, embedding2)
        losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))
        if self.reduction == 'none':
            return losses
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses.mean()


