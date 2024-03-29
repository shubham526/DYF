import torch
import tqdm

class Trainer:
    def __init__(self, model, model_type, optimizer, criterion, scheduler, metric, data_loader, use_cuda, device):
        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._scheduler = scheduler
        self._metric = metric
        self._data_loader = data_loader
        self._model_type = model_type
        self._use_cuda = use_cuda
        self._device = device

    def make_train_step(self):
        # Builds function that performs a step in the train loop


        def train_step(train_batch):
            # Sets model to TRAIN mode
            self._model.train()

            # Zero the gradients
            self._optimizer.zero_grad()

            # Makes predictions and compute loss
            if self._model_type == 'pairwise':
                batch_score_pos = self._model(
                    context_input_ids=train_batch[ 'context_input_ids'].to(self._device),
                    context_attention_mask=train_batch['context_attention_mask'].to(self._device),
                    aspect_input_ids=train_batch['aspect_input_ids_pos'].to(self._device),
                    aspect_attention_mask=train_batch['aspect_attention_mask_pos'].to(self._device),
                )

                batch_score_neg = self._model(
                    context_input_ids=train_batch['context_input_ids'].to(self._device),
                    context_attention_mask=train_batch['context_attention_mask'].to(self._device),
                    aspect_input_ids=train_batch['aspect_input_ids_neg'].to(self._device),
                    aspect_attention_mask=train_batch['aspect_attention_mask_neg'].to(self._device),
                )

                batch_loss = self._criterion(
                    batch_score_pos.tanh(),
                    batch_score_neg.tanh(),
                    torch.ones(batch_score_pos.size()).to(self._device)
                )

            elif self._model_type == 'pointwise':
                batch_score = self._model(
                    context_input_ids=train_batch['context_input_ids'].to(self._device),
                    context_attention_mask=train_batch['context_attention_mask'].to(self._device),
                    aspect_input_ids=train_batch['aspect_input_ids'].to(self._device),
                    aspect_attention_mask=train_batch['aspect_attention_mask'].to(self._device),
                )

                batch_loss = self._criterion(batch_score, train_batch['label'].float().to(self._device))
            else:
                raise ValueError('Model type must be `pairwise` or `pointwise`.')

            # Computes gradients
            batch_loss.backward()

            # Updates parameters
            self._optimizer.step()
            self._scheduler.step()

            # Returns the loss
            return batch_loss.item()

        # Returns the function that will be called inside the train loop
        return train_step

    def train(self):
        train_step = self.make_train_step()
        epoch_loss = 0
        num_batch = len(self._data_loader)
        # print('Number of batches = {}'.format(num_batch))
        for _, batch in tqdm.tqdm(enumerate(self._data_loader), total=num_batch):
            if batch is not None:
                batch_loss = train_step(batch)
                epoch_loss += batch_loss

        return epoch_loss


