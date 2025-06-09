import os
import json
import torch
import faiss
import numpy as np
from tqdm import tqdm
from modules.visual_extractor import VisualExtractor_emebed


class Retriever:
    def __init__(self, args, tokenizer, split='train'):
        self.args = args
        self.tokenizer = tokenizer
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_seq_length = args.max_seq_length

        with open(self.ann_path, 'r') as f:
            all_data = json.load(f)
        
        self.annotations = {}
        for split in ['train', 'val', 'test']:
            for entry in all_data.get(split, []):
                sample_id = entry['id']
                report = entry['report']
                self.annotations[sample_id] = {
                    'report': report,
                    'img_path': entry['image_path'][0]  # only use 0.png for now
                }
        
        self.sample_ids = [k for k in self.annotations.keys() if os.path.isfile(os.path.join(self.image_dir, self.annotations[k]['img_path']))]
        
        print(f"[INFO] Found {len(self.sample_ids)} valid samples for retrieval.")

        



        self.model = VisualExtractor_emebed(args).to(self.device).eval()

        self.index, self.id_to_report = self._build_faiss_index()

    def _build_faiss_index(self):
        all_embeds = []
        id_to_report = {}
    
        skipped = 0
    
        for sample_id in tqdm(self.sample_ids, desc="Building FAISS index"):
            image_path = os.path.join(self.image_dir, self.annotations[sample_id]['img_path'])
            if not os.path.isfile(image_path):
                print(f"[WARN] Skipping: {image_path} does not exist")
                skipped += 1
                continue
    
            try:
                image = self._load_image_tensor(image_path).to(self.device).unsqueeze(0)
                _, _, img_embed = self.model(image)
                all_embeds.append(img_embed.detach().cpu().numpy())
                id_to_report[sample_id] = self.annotations[sample_id]['report']
            except Exception as e:
                print(f"[ERROR] Could not process {sample_id}: {e}")
                skipped += 1
    
        if not all_embeds:
            raise RuntimeError("No embeddings were extracted. FAISS index cannot be built.")
    
        embed_matrix = np.vstack(all_embeds).astype('float32')
        index = faiss.IndexFlatL2(embed_matrix.shape[1])
        index.add(embed_matrix)
    
        print(f"[INFO] Built FAISS index with {len(all_embeds)} samples. Skipped: {skipped}")
        return index, id_to_report


    def _load_image_tensor(self, path):
        from PIL import Image
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        img = Image.open(path).convert('RGB')
        return transform(img)

    def retrieve_report(self, image_tensor):
        with torch.no_grad():
            _, _, query_embed = self.model(image_tensor.to(self.device).unsqueeze(0))
            query_embed_np = query_embed.cpu().numpy().astype('float32')
    
            D, I = self.index.search(query_embed_np, k=3)  # top-3
            retrieved_ids = [self.sample_ids[idx] for idx in I[0]]
    
            sep_token = self.tokenizer.tokenizer.sep_token if hasattr(self.tokenizer, 'tokenizer') else '[SEP]'
    
            # Join top-3 reports with separator
            joined_report = f" {sep_token} ".join([self.id_to_report[rid] for rid in retrieved_ids])
    
            tokens = self.tokenizer(joined_report)
            tokens = tokens[:self.max_seq_length // 2]  # limit to avoid over-length
            return torch.tensor(tokens, dtype=torch.long).to(self.device)
    
