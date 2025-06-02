import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torchvision import transforms


def custom_collate_fn(batch):
    """
    Custom collate function to handle None values in the batch.
    Args:
        batch: List of tuples returned by __getitem__
    Returns:
        Collated batch with None values replaced with empty tensors
    """
    # Separate the batch into individual components
    images_id = [item[0] for item in batch]
    image_tags = [item[1] for item in batch]
    images = torch.stack([item[2] for item in batch])
    reports_ids = torch.stack([item[3] for item in batch])
    reports_masks = torch.stack([item[4] for item in batch])
    reports_ids_bert = torch.stack([item[5] for item in batch])
    reports_masks_bert = torch.stack([item[6] for item in batch])
    seq_length = torch.tensor([item[7] for item in batch])
    seq_length_bert = torch.tensor([item[8] for item in batch])
    
    # Replace None tags with empty strings
    image_tags = [tag if tag is not None else "" for tag in image_tags]
    
    return (images_id, image_tags, images, reports_ids, reports_masks, 
            reports_ids_bert, reports_masks_bert, seq_length, seq_length_bert)


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.max_seq_length_bert = args.max_seq_length_bert
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.tokenizer_bert = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            # Convert BERT tokenization to tensors
            bert_encoding = self.tokenizer_bert(
                self.examples[i]['report'],
                max_length=self.max_seq_length_bert,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            self.examples[i]['ids_bert'] = bert_encoding['input_ids'].squeeze(0)
            self.examples[i]['mask_bert'] = bert_encoding['attention_mask'].squeeze(0)
            
            # Convert regular tokenization to tensors
            encoding = self.tokenizer(
                self.examples[i]['report'],
                max_length=self.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            self.examples[i]['ids'] = encoding['input_ids'].squeeze(0)
            self.examples[i]['mask'] = encoding['attention_mask'].squeeze(0)

    def __len__(self):
        return len(self.examples)


class MultiImageDataset(BaseDataset):
    def __init__(self, args, tokenizer, split):
        super(MultiImageDataset, self).__init__(args, tokenizer, split)
        # Add default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # Remove normalization since CLIP processor will handle it
            ])

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        
        # Load and convert images to RGB
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        
        # Apply transforms to convert to tensors
        image_1 = self.transform(image_1)
        image_2 = self.transform(image_2)
        
        # Stack the tensors
        image = torch.stack((image_1, image_2), 0)
        
        report_ids = example['ids']
        report_masks = example['mask']
        report_ids_bert = example['ids_bert']
        report_masks_bert = example['mask_bert']
        
        seq_length = len(report_ids)
        seq_length_bert = len(report_ids_bert)
        
        # Always return image_tag, defaulting to None if not present
        image_tag = example.get('tag', None)
        
        return (image_id, image_tag, image, report_ids, report_masks, report_ids_bert, report_masks_bert, seq_length, seq_length_bert)


class ImageTextDataset(Dataset):
    def __init__(self, data_path, tokenizer, processor, image_size=224, max_text_length=512, is_train=True):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.is_train = is_train
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Image transforms for training augmentation
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and transform image
        image_path = os.path.join(os.path.dirname(self.data_path), item['image_path'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Process image with CLIP processor
        processed_image = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        
        # Tokenize text
        text = item['report']
        encoding = self.tokenizer(
            text,
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'image': processed_image,
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'report': text
        }