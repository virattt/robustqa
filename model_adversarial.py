import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertForQuestionAnswering


class AdversarialModel(nn.Module):

    def __init__(self, args, hidden_size=768, num_classes=6, discriminator_lambda=0.01):
        super(AdversarialModel, self).__init__()

        # Load models
        self.qa_model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')
        self.discriminator_model = DiscriminatorModel()

        # Set fields
        self.num_classes = num_classes
        self.discriminator_lambda = discriminator_lambda

        # Create output layer
        self.qa_outputs = nn.Linear(hidden_size, 2)
        self.qa_outputs.weight.data.normal_(mean=0.0, std=0.02)
        self.qa_outputs.bias.data.zero_()

    def forward(self, input_ids, attention_mask,
                start_positions=None, end_positions=None, labels=None,
                model_type=None):
        """
        Parameters
        ----------
        input_ids is shape [16, 384] or [batch_size, max_embedding_length]
        attention_mask is shape [16, 384]
        start_positions is shape [16, ]
        end_positions is shape [16, ]
        """
        if model_type == 'qa_model':
            qa_loss = self.forward_qa(input_ids, attention_mask, start_positions, end_positions)
            return qa_loss
        elif model_type == 'discriminator_model':
            print('discriminator_model')

    def forward_qa(self, input_ids, attention_mask, start_positions, end_positions):
        # Do forward pass on DistilBERT
        outputs = self.qa_model(input_ids,
                                attention_mask=attention_mask,
                                start_positions=start_positions,
                                end_positions=end_positions,
                                output_hidden_states=True
                                )

        # Get final hidden state from DistilBERT output
        last_hidden_state = outputs["hidden_states"][-1]
        hidden = last_hidden_state[:, 0]  # same as cls_embedding

        # Use the final hidden state to get the targets from the discriminator model
        log_prob = self.discriminator_model(hidden)
        targets = torch.ones_like(log_prob) * (1 / self.num_classes)

        # Compute KL loss
        kl_criterion = nn.KLDivLoss(reduction="batchmean")
        kld = self.discriminator_lambda * kl_criterion(log_prob, targets)

        # Get output layer logits (start and end)
        logits = self.qa_outputs(last_hidden_state)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # Sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)

        # Compute total loss by combining QA loss with KLD loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        qa_loss = (start_loss + end_loss) / 2
        total_loss = qa_loss + kld
        return total_loss

    def forward_discriminator(self, input_ids, attention_mask, start_positions, end_positions, labels):
        with torch.no_grad():
            # Do forward pass on DistilBERT
            outputs = self.qa_model(input_ids,
                                    attention_mask=attention_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions,
                                    output_hidden_states=True
                                    )

            # Get final hidden state from DistilBERT output
            last_hidden_state = outputs["hidden_states"][-1]
            hidden = last_hidden_state[:, 0]  # same as cls_embedding

        log_prob = self.discriminator(hidden.detach())
        criterion = nn.NLLLoss()
        loss = criterion(log_prob, labels)
        return loss

    def save(self, path: str):
        # TODO - look at mrqa trainer.py's save_model method
        self.qa_model.save_pretrained(path)


class DiscriminatorModel(nn.Module):

    def __init__(self, num_classes=6, input_size=768, hidden_size=768, num_layers=3, dropout=0.1):
        super(DiscriminatorModel, self).__init__()
        hidden_layers = []

        # Create layers for NN
        for i in range(num_layers):
            if i == 0:
                input_dim = input_size
            else:
                input_dim = hidden_size
            hidden_layers.append(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ))
        # Append final layer for classification
        hidden_layers.append(
            nn.Linear(hidden_size, num_classes)
        )
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.num_layers = num_layers

    def forward(self, x):
        # forward pass
        for i in range(self.num_layers - 1):
            x = self.hidden_layers[i](x)
        logits = self.hidden_layers[-1](x)
        log_prob = F.log_softmax(logits, dim=1)
        return log_prob
