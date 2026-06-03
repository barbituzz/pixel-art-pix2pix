
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PixelArtDataset(Dataset):
    #Dataset

    def __init__(self, root: str, split: str = "train", img_size: int = 512):
        meta_path = Path(root) / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)

        self.samples = [s for s in meta["samples"] if s["split"] == split]

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),   # [-1, 1]
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_a = self.transform(Image.open(s["file_a"]).convert("RGB"))  # оригинал
        img_b = self.transform(Image.open(s["file_b"]).convert("RGB"))  # пиксель-арт
        return {"A": img_a, "B": img_b, "label": s["label"]}


# Пример использования:
if __name__ == "__main__":
    ds = PixelArtDataset("dataset", split="train")
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)
    batch = next(iter(dl))
    print("A shape:", batch["A"].shape)   # (4, 3, 512, 512)
    print("B shape:", batch["B"].shape)   # (4, 3, 512, 512)