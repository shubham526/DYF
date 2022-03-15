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
                    context_text_input_ids=train_batch['context_text_input_ids'],
                    context_text_attention_mask=train_batch['context_text_attention_mask'],
                    aspect_text_input_ids=train_batch['aspect_text_input_ids_pos'],
                    aspect_text_attention_mask=train_batch['aspect_text_attention_mask_pos'],
                    context_entity_desc_input_ids=train_batch['context_entity_desc_input_ids'],
                    context_entity_desc_attention_mask=train_batch['context_entity_desc_attention_mask'],
                    context_entity_wiki2vec_embeddings=train_batch['context_entity_wiki2vec_embeddings'],
                    aspect_entity_desc_input_ids=train_batch['aspect_entity_desc_input_ids_pos'],
                    aspect_entity_desc_attention_mask=train_batch['aspect_entity_desc_attention_mask_pos'],
                    aspect_entity_wiki2vec_embeddings=train_batch['aspect_entity_wiki2vec_embeddings_pos'],
                )

                batch_score_neg = self._model(
                    context_text_input_ids=train_batch['context_text_input_ids'],
                    context_text_attention_mask=train_batch['context_text_attention_mask'],
                    aspect_text_input_ids=train_batch['aspect_text_input_ids_neg'],
                    aspect_text_attention_mask=train_batch['aspect_text_attention_mask_neg'],
                    context_entity_desc_input_ids=train_batch['context_entity_desc_input_ids'],
                    context_entity_desc_attention_mask=train_batch['context_entity_desc_attention_mask'],
                    context_entity_wiki2vec_embeddings=train_batch['context_entity_wiki2vec_embeddings'],
                    aspect_entity_desc_input_ids=train_batch['aspect_entity_desc_input_ids_neg'],
                    aspect_entity_desc_attention_mask=train_batch['aspect_entity_desc_attention_mask_neg'],
                    aspect_entity_wiki2vec_embeddings=train_batch['aspect_entity_wiki2vec_embeddings_neg'],
                )

                batch_loss = self._criterion(
                    batch_score_pos.tanh(),
                    batch_score_neg.tanh(),
                    torch.ones(batch_score_pos.size()).to(self._device)
                )

            elif self._model_type == 'pointwise':
                batch_score = self._model(
                    context_text_input_ids=train_batch['context_text_input_ids'],
                    context_text_attention_mask=train_batch['context_text_attention_mask'],
                    aspect_text_input_ids=train_batch['aspect_text_input_ids'],
                    aspect_text_attention_mask=train_batch['aspect_text_attention_mask'],
                    context_entity_desc_input_ids=train_batch['context_entity_desc_input_ids'],
                    context_entity_desc_attention_mask=train_batch['context_entity_desc_attention_mask'],
                    context_entity_wiki2vec_embeddings=train_batch['context_entity_wiki2vec_embeddings'],
                    aspect_entity_desc_input_ids=train_batch['aspect_entity_desc_input_ids'],
                    aspect_entity_desc_attention_mask=train_batch['aspect_entity_desc_attention_mask'],
                    aspect_entity_wiki2vec_embeddings=train_batch['aspect_entity_wiki2vec_embeddings'],
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

        for _, batch in tqdm.tqdm(enumerate(self._data_loader), total=num_batch):
            if batch is not None:
                batch_loss = train_step(batch)
                epoch_loss += batch_loss

        return epoch_loss


