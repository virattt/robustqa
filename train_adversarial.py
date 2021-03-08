import json
import os
from collections import OrderedDict

import torch
import csv
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
        avg_qa_loss = 0
        avg_discriminator_loss = 0
        global_idx = 0
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
                    labels = batch['labels'].to(device)
                    outputs = model(input_ids,
                                    attention_mask=attention_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions,
                                    model_type='qa_model')

                    # qa_loss = outputs[0].mean()
                    qa_loss = outputs[0]
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
                                               labels=labels,
                                               model_type='discriminator_model')

                    discriminator_loss = discriminator_loss.mean()
                    discriminator_loss.backward()

                    avg_discriminator_loss = self.cal_running_avg_loss(discriminator_loss.item(),
                                                                       avg_discriminator_loss)
                    print(f'avg_discriminator_loss: {avg_discriminator_loss}')

                    discriminator_optimizer.step()
                    discriminator_optimizer.zero_grad()

                    progress_bar.update(len(input_ids))
                    progress_bar.set_postfix(epoch=epoch, NLL=qa_loss.item())

                    if (global_idx % self.eval_every) == 0:
                        self.log.info(f'Evaluating at step {global_idx}...')
                        preds, curr_score = self.evaluate(model, eval_dataloader, val_dict, return_preds=True)
                        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                        self.log.info('Visualizing in TensorBoard...')
                        for k, v in curr_score.items():
                            tensorboard_writer.add_scalar(f'val/{k}', v, global_idx)
                        self.log.info(f'Eval {results_str}')
                        if self.visualize_predictions:
                            util.visualize(tensorboard_writer,
                                           pred_dict=preds,
                                           gold_dict=val_dict,
                                           step=global_idx,
                                           split='val',
                                           num_visuals=self.num_visuals)
                        if curr_score['F1'] >= best_scores['F1']:
                            best_scores = curr_score
                            self.save(model)
                    global_idx += 1
                return best_scores

    def evaluate(self, model, data_loader, data_dict, return_preds=False, split='validation'):
        device = self.device

        model.eval()
        pred_dict = {}
        all_start_logits = []
        all_end_logits = []
        with torch.no_grad(), \
             tqdm(total=len(data_loader.dataset), position=0, leave=True) as progress_bar:
            for batch in data_loader:
                # Setup for forward
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                # Forward
                outputs = model(input_ids, attention_mask=attention_mask, model_type='qa_model')
                start_logits, end_logits = outputs.start_logits, outputs.end_logits

                all_start_logits.append(start_logits)
                all_end_logits.append(end_logits)

                # Update progress
                batch_size = len(input_ids)
                progress_bar.update(batch_size)

        # Get F1 and EM scores
        start_logits = torch.cat(all_start_logits).cpu().numpy()
        end_logits = torch.cat(all_end_logits).cpu().numpy()
        preds = util.postprocess_qa_predictions(data_dict,
                                                data_loader.dataset.encodings,
                                                (start_logits, end_logits))
        if split == 'validation':
            results = util.eval_dicts(data_dict, preds)
            results_list = [('F1', results['F1']),
                            ('EM', results['EM'])]
        else:
            results_list = [('F1', -1.0),
                            ('EM', -1.0)]
        results = OrderedDict(results_list)
        if return_preds:
            return preds, results
        return results

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

        # Set the device to cuda if GPU available
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Load training data
        log.info("Preparing Training Data...")
        train_dataset, train_dict = data_utils.get_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train')
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,  # batches the examples into groups of 16
                                  sampler=RandomSampler(train_dataset))
        # Load validation data
        log.info("Preparing Validation Data...")
        val_dataset, val_dict = data_utils.get_dataset(args, args.train_datasets, args.val_dir, tokenizer, 'val')
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(val_dataset))

        # Train!!!
        trainer = Trainer(args, log)
        trainer.train(model, train_loader, val_loader, val_dict)

    if args.do_eval:
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        split_name = 'test' if 'test' in args.eval_dir else 'validation'
        log = util.get_logger(args.save_dir, f'log_{split_name}')
        checkpoint_path = os.path.join(args.save_dir, 'checkpoint')

        # Load model
        model = AdversarialModel(args, checkpoint_path=checkpoint_path)
        model.to(args.device)

        # Load eval data
        eval_dataset, eval_dict = data_utils.get_dataset(args, args.eval_datasets, args.eval_dir, tokenizer, split_name)
        eval_loader = DataLoader(eval_dataset,
                                 batch_size=args.batch_size,
                                 sampler=SequentialSampler(eval_dataset))

        # Evaluate!!!
        trainer = Trainer(args, log)
        eval_preds, eval_scores = trainer.evaluate(model,
                                                   data_loader=eval_loader,
                                                   data_dict=eval_dict,
                                                   return_preds=True,
                                                   split=split_name)
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in eval_scores.items())
        log.info(f'Eval {results_str}')

        # Write submission file
        sub_path = os.path.join(args.save_dir, split_name + '_' + args.sub_file)
        log.info(f'Writing submission file to {sub_path}...')
        with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
            csv_writer = csv.writer(csv_fh, delimiter=',')
            csv_writer.writerow(['Id', 'Predicted'])
            for uuid in sorted(eval_preds):
                csv_writer.writerow([uuid, eval_preds[uuid]])


if __name__ == '__main__':
    main()
