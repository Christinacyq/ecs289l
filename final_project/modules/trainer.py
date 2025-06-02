import os
from abc import abstractmethod
import numpy as np
import time
import torch
import pandas as pd
from numpy import inf
from tqdm import tqdm
from torch import nn
import wandb
from timm.utils import ModelEmaV2
import logging
import json


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        if self.args.use_ema:
            self.model_ema = ModelEmaV2(model, decay=0.999)
            self.model_ema.module.to(self.device)
            print('################################# using ema')
        else:
            self.model_ema = None

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        self._print_best()
        self._print_best_to_file()

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name + '.csv')
        
        # Create new records
        new_records = pd.DataFrame([self.best_recorder['val'], self.best_recorder['test']])
        
        # Load existing records if any
        if os.path.exists(record_path):
            record_table = pd.read_csv(record_path)
            record_table = pd.concat([record_table, new_records], ignore_index=True)
        else:
            record_table = new_records
            
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        if self.args.use_ema:
            state['state_dict_ema'] = self.model_ema.module.state_dict()
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.args.use_ema:
            self.model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        """
        Record the best model based on the validation metric.
        """
        # Get the base metric name without the 'val_' prefix
        base_metric = self.mnt_metric.replace('val_', '')
        current_metric = f'val_{base_metric}'
        
        improved = (self.mnt_mode == 'min' and log[current_metric] <= self.best_recorder['val'][self.mnt_metric]) or \
                   (self.mnt_mode == 'max' and log[current_metric] >= self.best_recorder['val'][self.mnt_metric])
        
        if improved:
            self.best_recorder['val'].update(log)
            self._save_best_model()
            self.logger.info(f"Saved best model: {current_metric} = {log[current_metric]:.4f}")
            
        # Save latest model
        self._save_checkpoint(-1)
        
        # Save metrics
        self._save_metrics(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, device,
                 train_dataloader, val_dataloader=None, test_dataloader=None,
                 lr_scheduler=None, tokenizer=None):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.logger = logging.getLogger()
        
        # Setup GPU device if available
        self.device = device
        self.model = model.to(self.device)
        
        # Initialize dataloaders
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        
        # Initialize optimizer and scheduler
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer
        
        # Initialize loss and metrics
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        
        # Initialize best model tracking
        self.mnt_best = np.inf if args.monitor_mode == 'min' else -np.inf
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        self.mnt_mode = args.monitor_mode
        
        # Initialize best model recorder
        self.best_recorder = {
            'val': {
                self.mnt_metric: self.mnt_best
            },
            'test': {
                self.mnt_metric_test: self.mnt_best
            }
        }
        
        # Create output directories
        self.save_dir = os.path.join(args.output_dir, 'checkpoints')
        self.record_dir = os.path.join(args.output_dir, 'records')
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.record_dir, exist_ok=True)
        
        # Initialize EMA model if enabled
        if args.use_ema:
            self.model_ema = ModelEmaV2(model)

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + image_loss) / 2.0

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def _eval_epoch(self, log, split, dataloader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (images_id, image_tags, images, reports_ids, reports_masks, reports_ids_bert, reports_masks_bert, seq_length, seq_length_bert) in enumerate(dataloader):
                # Move data to device
                images = images.to(self.device)
                reports_ids = reports_ids.to(self.device)
                reports_masks = reports_masks.to(self.device)
                reports_ids_bert = reports_ids_bert.to(self.device)
                reports_masks_bert = reports_masks_bert.to(self.device)
                
                # Forward pass
                outputs = self.model(images, reports_ids, reports_masks, reports_ids_bert, reports_masks_bert)
                
                # Compute loss
                loss = self.criterion(outputs, reports_ids, reports_masks)
                total_loss += loss.item()
                
                # Get predictions and targets for metrics
                preds = outputs['logits_v2t'].argmax(dim=-1)
                targets = torch.arange(preds.size(0), device=preds.device)
                
                # Convert predictions and targets to text
                pred_texts = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
                target_texts = self.tokenizer.batch_decode(targets, skip_special_tokens=True)
                
                all_preds.extend(pred_texts)
                all_targets.extend(target_texts)
        
        # Compute average loss
        avg_loss = total_loss / len(dataloader)
        log[f'{split}_loss'] = avg_loss
        
        # Compute metrics if available
        if self.metric_ftns is not None:
            for metric_name, metric_fn in self.metric_ftns.items():
                score = metric_fn(all_preds, all_targets)
                log[f'{split}_{metric_name}'] = score
        
        return log

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (images_id, image_tags, images, reports_ids, reports_masks, reports_ids_bert, reports_masks_bert, seq_length, seq_length_bert) in enumerate(
                self.train_dataloader):
            # Move data to device
            images = images.to(self.device)
            reports_ids = reports_ids.to(self.device)
            reports_masks = reports_masks.to(self.device)
            reports_ids_bert = reports_ids_bert.to(self.device)
            reports_masks_bert = reports_masks_bert.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, reports_ids, reports_masks, reports_ids_bert, reports_masks_bert)
            
            # Compute loss
            loss = self.criterion(outputs, reports_ids, reports_masks)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            
            # Update weights
            self.optimizer.step()
            
            # Update EMA model if enabled
            if self.args.use_ema:
                self.model_ema.update(self.model)
            
            # Update learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % self.args.logging_steps == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(images), len(self.train_dataloader.dataset),
                    100. * batch_idx / len(self.train_dataloader), loss.item()))
        
        # Compute average loss
        avg_loss = total_loss / len(self.train_dataloader)
        log = {'train_loss': avg_loss}
        
        # Evaluate on validation and test sets
        self._eval_epoch(log, 'val', self.val_dataloader)
        self._eval_epoch(log, 'test', self.test_dataloader)
        
        return log

    def _save_best_model(self):
        """
        Save the best model checkpoint.
        """
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'best_metric': self.best_recorder['val'][self.mnt_metric],
            'epoch': self.start_epoch
        }
        
        # Save the best model
        best_model_path = os.path.join(self.save_dir, 'best_model.pt')
        torch.save(state, best_model_path)
        self.logger.info(f"Saved best model to {best_model_path}")

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Save a checkpoint.
        """
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'epoch': epoch
        }
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint-epoch{epoch}.pt')
        torch.save(state, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model if requested
        if save_best:
            best_model_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(state, best_model_path)
            self.logger.info(f"Saved best model to {best_model_path}")

    def _save_metrics(self, log):
        """
        Save metrics to a JSON file.
        """
        metrics_path = os.path.join(self.record_dir, 'metrics.json')
        
        # Load existing metrics if file exists
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = []
        
        # Add current metrics
        metrics.append({
            'epoch': self.start_epoch,
            **log
        })
        
        # Save updated metrics
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

    def test(self):
        """
        Test the model on the test set.
        """
        self.model.eval()
        log = {}
        return self._eval_epoch(log, 'test', self.test_dataloader)
