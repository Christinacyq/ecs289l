import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, CLIPProcessor
from modules.visual_extractor import VisualExtractor_medvit
from modules.text_extractor import MedCLIPTextModel
from modules.trainer import Trainer
from modules.datasets import BaseDataset, MultiImageDataset, custom_collate_fn
from modules.utils import set_seed
from modules.loss import ContrastiveLoss
from modules.metrics import calculate_metrics
from modules.model import MedCLIPModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

def parse_args():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument('--visual_extractor', type=str, default='pubmedclip')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True)
    parser.add_argument('--text_extractor', type=str, default='emilyalsentzer/Bio_ClinicalBERT')
    parser.add_argument('--proj_dim', type=int, default=512)
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    
    # Data parameters
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--ann_path', type=str, required=True)
    parser.add_argument('--max_seq_length', type=int, default=100)
    parser.add_argument('--max_seq_length_bert', type=int, default=512)
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--use_ema', type=bool, default=False)
    parser.add_argument('--monitor_mode', type=str, default='max')
    parser.add_argument('--monitor_metric', type=str, default='val_loss')
    parser.add_argument('--save_period', type=int, default=1)
    parser.add_argument('--dataset_name', type=str, default='medical')
    parser.add_argument('--record_dir', type=str, default='./records')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume', type=str, default=None)
    
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.record_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.text_extractor)
    
    # Initialize datasets
    train_dataset = MultiImageDataset(args, tokenizer, 'train')
    val_dataset = MultiImageDataset(args, tokenizer, 'val')
    test_dataset = MultiImageDataset(args, tokenizer, 'test')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    # Initialize models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    visual_extractor = VisualExtractor_medvit(args).to(device)
    text_extractor = MedCLIPTextModel(
        bert_type=args.text_extractor,
        proj_dim=args.proj_dim
    ).to(device)
    
    # Create combined model
    model = MedCLIPModel(visual_extractor, text_extractor).to(device)
    
    # Initialize loss and metrics
    criterion = ContrastiveLoss()
    metric_ftns = {
        'metrics': lambda preds, targets: calculate_metrics(preds, targets)
    }
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    # Calculate total training steps
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    # Learning rate scheduler with warmup
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=args.learning_rate * 0.01
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        metric_ftns=metric_ftns,
        optimizer=optimizer,
        args=args,
        device=device,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        lr_scheduler=scheduler,
        tokenizer=tokenizer
    )
    
    # Start training
    log = trainer.train()
    
    # Run test evaluation
    test_log = trainer.test()
    test_metrics = {k: v for k, v in test_log.items()}
    log.update(**{'test_'+k: v for k, v in test_metrics.items()})
    
    # Save final metrics
    trainer._save_metrics(log)
    
    # Save final model
    trainer._save_checkpoint(trainer.start_epoch, save_best=True)
    
    return log

if __name__ == '__main__':
    main() 