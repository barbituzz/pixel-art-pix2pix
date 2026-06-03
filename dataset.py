import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms


class PixelArtDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        self.root_dir = Path(root_dir)
        self.mode = mode

        # Пути к папкам
        self.real_dir = self.root_dir / 'real'
        self.pixel_dir = self.root_dir / 'pixel'

        # список файлов
        self.files = sorted(list(self.real_dir.glob('*.png')) +
                            list(self.real_dir.glob('*.jpg')))

        # Аугментация для тренировки
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),  # ✅ 512 (совпадает с датасетом)
                # transforms.RandomHorizontalFlip(),  #
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),  # ✅ ИСПРАВЛЕНО: было 256, теперь 512
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        #  изображения
        real_path = self.files[idx]
        pixel_path = self.pixel_dir / real_path.name

        real_image = Image.open(real_path).convert('RGB')
        pixel_image = Image.open(pixel_path).convert('RGB')

        real = self.transform(real_image)
        pixel = self.transform(pixel_image)

        return real, pixel