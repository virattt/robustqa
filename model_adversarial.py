import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertForQuestionAnswering
from transformers.modeling_outputs import QuestionAnsweringModelOutput


class AdversarialModel(nn.Module):

    def __init__(self, args, hidden_size=768, num_classes=6, discriminator_lambda=0.01, checkpoint_path: str = None, load_path: str = None):
        super(AdversarialModel, self).__init__()
        self.args = args
        # Load models
        if checkpoint_path:
            self.qa_model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
        else:
            self.qa_model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')

        self.discriminator_model = DiscriminatorModel()

        # Set fields
        self.num_classes = num_classes
        self.discriminator_lambda = discriminator_lambda

        # Create output layer
        self.qa_outputs = nn.Linear(hidden_size, 2)
        self.qa_outputs.weight.data.normal_(mean=0.0, std=0.02)
        self.qa_outputs.bias.data.zero_()

        if load_path is not None:
            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage))

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
            discriminator_loss = self.forward_discriminator(input_ids, attention_mask, start_positions, end_positions, labels)
            return discriminator_loss
        else:
            # For evaluation
            outputs = self.qa_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden_state = outputs["hidden_states"][-1]
            logits = self.qa_outputs(last_hidden_state)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            return QuestionAnsweringModelOutput(
                start_logits=start_logits,
                end_logits=end_logits
            )

    def forward_qa(self, input_ids, attention_mask, start_positions, end_positions):
        # Do forward pass on DistilBERT
        outputs = self.qa_model(input_ids,
                                attention_mask=attention_mask,
                                start_positions=start_positions,
                                end_positions=end_positions,
                                output_hidden_states=True)

        # Get final hidden state from DistilBERT output
        last_hidden_state = outputs["hidden_states"][-1]

        # Get output layer logits (start and end)
        logits = self.qa_outputs(last_hidden_state)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # Sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            # Use the final hidden state to get the targets from the discriminator model
            hidden = last_hidden_state[:, 0]  # same as cls_embedding
            log_prob = self.discriminator_model(hidden)
            targets = torch.ones_like(log_prob) * (1 / self.num_classes)

            # Compute KL loss
            kl_criterion = nn.KLDivLoss(reduction="batchmean")
            kld = self.discriminator_lambda * kl_criterion(log_prob, targets)

            # Compute total loss by combining QA loss with KLD loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            qa_loss = (start_loss + end_loss) / 2
            total_loss = qa_loss + kld
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

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

        log_prob = self.discriminator_model(hidden.detach())
        criterion = nn.NLLLoss()
        loss = criterion(log_prob, labels)
        return loss

    def save(self, path: str):
        self.qa_model.save_pretrained(path)

    def save_qa_output_model(self, path: str):
        print(f'BEFORE load: self.qa_outputs.state_dict(): {self.qa_outputs.state_dict()}')
        torch.save(self.qa_outputs.state_dict(), path)

    def load(self, path: str):
        self.qa_model.from_pretrained(path)

    def load_qa_output_model(self, path: str):
        self.qa_outputs.load_state_dict(torch.load(path))
        print(f'AFTER load: self.qa_outputs.state_dict(): {self.qa_outputs.state_dict()}')

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
