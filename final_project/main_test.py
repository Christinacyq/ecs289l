import os
import argparse
import torch
import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, CLIPProcessor
from modules.visual_extractor import VisualExtractor_medvit
from modules.text_extractor import MedCLIPTextModel
from modules.tester import Tester
from modules.datasets import ImageTextDataset
from modules.utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument('--visual_extractor', type=str, default='pubmedclip')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True)
    parser.add_argument('--text_extractor', type=str, default='emilyalsentzer/Bio_ClinicalBERT')
    parser.add_argument('--proj_dim', type=int, default=512)
    
    # Testing parameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--min_length', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    
    # Data parameters
    parser.add_argument('--test_data_path', type=str, required=True)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--max_text_length', type=int, default=512)
    
    # Model checkpoint
    parser.add_argument('--checkpoint_path', type=str, required=True)
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./results')
    
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(args.text_extractor)
    processor = CLIPProcessor.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
    
    # Initialize test dataset
    test_dataset = ImageTextDataset(
        data_path=args.test_data_path,
        tokenizer=tokenizer,
        processor=processor,
        image_size=args.image_size,
        max_text_length=args.max_text_length,
        is_train=False
    )
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    visual_extractor = VisualExtractor_medvit(args).to(device)
    text_extractor = MedCLIPTextModel(
        bert_type=args.text_extractor,
        proj_dim=args.proj_dim
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    visual_extractor.load_state_dict(checkpoint['visual_extractor_state_dict'])
    text_extractor.load_state_dict(checkpoint['text_extractor_state_dict'])
    
    # Initialize tester
    tester = Tester(
        visual_extractor=visual_extractor,
        text_extractor=text_extractor,
        test_loader=test_loader,
        tokenizer=tokenizer,
        args=args,
        device=device
    )
    
    # Start testing
    results = tester.test()
    
    # Save results
    output_file = os.path.join(args.output_dir, 'test_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Test results saved to {output_file}")

if __name__ == '__main__':
    main() 