import json
import os

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import DistilBertTokenizerFast

import data_utils
import util
from args import get_train_test_args
from model_adversarial import AdversarialModel


class Trainer:
    def __init__(self, args, log):
        self.device = args.device
        self.eval_every: int = args.eval_every
        self.log = log
        self.lr: float = args.lr
        self.num_epochs: int = args.num_epochs
        self.num_visuals: int = args.num_visuals
        self.path: str = os.path.join(args.save_dir, 'checkpoint')
        self.save_dir: str = args.save_dir
        self.visualize_predictions: bool = args.visualize_predictions

        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def save(self, model: AdversarialModel):
        model.save(self.path)

    def train(self, model: AdversarialModel, train_dataloader, eval_dataloader, val_dict):
        device = self.device
        model.to(device)

        # Create optimizers
        optimizer_params = util.get_optimizer_grouped_parameters(model, self.lr)
        qa_optimizer = torch.optim.AdamW(optimizer_params, lr=self.lr, weight_decay=0)
        discriminator_optimizer = torch.optim.AdamW(optimizer_params, lr=self.lr, weight_decay=0)

        # Initialize training loop vars
        global_index = 0
        avg_qa_loss = 0
        avg_discriminator_loss = 0
        best_scores = {'F1': -1.0, 'EM': -1.0}
        tensorboard_writer = SummaryWriter(self.save_dir)

        for epoch in range(self.num_epochs):
            self.log.info("Epoch: {}".format(epoch))
            with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset), position=0, leave=True) as progress_bar:
                for batch in train_dataloader:
                    model.train()
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    start_positions = batch['start_positions'].to(device)
                    end_positions = batch['end_positions'].to(device)
                    qa_loss = model(input_ids,
                                    attention_mask=attention_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions,
                                    model_type='qa_model')

                    qa_loss = qa_loss.mean()
                    qa_loss.backward()
                    avg_qa_loss = self.cal_running_avg_loss(qa_loss.item(), avg_qa_loss)
                    print(f'avg_qa_loss: {avg_qa_loss}')

                    qa_optimizer.step()
                    qa_optimizer.zero_grad()

                    # Update discriminator model
                    discriminator_loss = model(input_ids,
                                    attention_mask=attention_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions,
                                    model_type='discriminator_model')


    @staticmethod
    def cal_running_avg_loss(loss, running_avg_loss, decay=0.99):
        if running_avg_loss == 0:
            return loss
        else:
            running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
            return running_avg_loss


def main():
    # Get command-line args and set seed
    args = get_train_test_args()
    util.set_seed(args.seed)

    # Load model
    model = AdversarialModel(args)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    if args.do_train:
        # Make /save directory
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)

        # Get logger
        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        log.info("Preparing Training Data...")

        # Set the device to cuda if GPU available
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Load training data
        train_dataset, train_dict = data_utils.get_dataset(args, args.train_datasets, args.train_dir, tokenizer,
                                                           'train')
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,  # batches the examples into groups of 16
                                  sampler=RandomSampler(train_dataset))
        # Load validation data
        val_dataset, val_dict = data_utils.get_dataset(args, args.train_datasets, args.val_dir, tokenizer, 'val')
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(val_dataset))

        # Train!!!
        trainer = Trainer(args, log)
        trainer.train(model, train_loader, val_loader, val_dict)

    if args.do_eval:
        print("do_eval")


# Evaluate


if __name__ == '__main__':
    main()
