import os
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# Импорты из твоих модулей
from models import Generator, Discriminator
from dataset import PixelArtDataset


class Pix2PixTrainer:

    def __init__(self,
                 dataset_dir='dataset',
                 batch_size=1,
                 lr=0.0004,
                 num_epochs=250,
                 img_size=512,
                 checkpoint_dir='checkpoints',
                 save_every=10):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.img_size = img_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_every = save_every

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Определяем устройство
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f" Устройство: {self.device}")
        print(f"📦 Датасет: {dataset_dir}")
        print(f"️  Размер изображений: {img_size}x{img_size}")
        print(f"⚙️  Batch size: {batch_size}")
        print(f"📈 Learning rate: {lr}")
        print(f"🔄 Эпох: {num_epochs}\n")

        # Инициализируем модели
        self.generator = Generator(in_channels=3, out_channels=3).to(self.device)
        self.discriminator = Discriminator(in_channels=6).to(self.device)

        # Оптимизаторы  (ПРОБЛЕМА!!!!!)
        self.optimizer_g = optim.Adam(
            self.generator.parameters(), lr=lr, betas=(0.5, 0.999)
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(), lr=lr * 1, betas=(0.5, 0.999)  # ← Увеличил в 4 раза, затем обратно
        )

        # Функция потерь
        self.criterion_gan = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()

        # Лямбда для L1 loss (ПРОБЛЕМА!!!!) УМЕНЬШИЛ СО СТА ДО 75
        self.lambda_l1 = 75.0

        # DataLoader
        self.train_dataset = PixelArtDataset(
            root_dir=dataset_dir,
            mode='train'
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True
        )

        self.val_dataset = PixelArtDataset(
            root_dir=dataset_dir,
            mode='val'
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )

        print(f"📊 Пар в train: {len(self.train_dataset)}")
        print(f"📊 Пар в val: {len(self.val_dataset)}\n")

        # Истории потерь
        self.g_losses = []
        self.d_losses = []

    def train_step(self, real, pixel):
        batch_size = real.size(0)

        # ===== ДИСКРИМИНАТОР  (ПРОБЛЕМА)
        for _ in range(1):  # ← Дискриминатор учился 2 раза на каждый шаг генератора, затем обратно
            self.optimizer_d.zero_grad()

            real_pair = torch.cat([real, pixel], dim=1)
            d_real = self.discriminator(real_pair)

            # Label smoothing
            real_labels = torch.ones_like(d_real).to(self.device) * 0.9
            fake_labels = torch.zeros_like(d_real).to(self.device)

            loss_d_real = self.criterion_gan(d_real, real_labels)

            fake_pixel = self.generator(real)
            fake_pair = torch.cat([real, fake_pixel.detach()], dim=1)
            d_fake = self.discriminator(fake_pair)
            loss_d_fake = self.criterion_gan(d_fake, fake_labels)

            loss_d = (loss_d_real + loss_d_fake) / 2
            loss_d.backward()
            self.optimizer_d.step()

        # ===== ГЕНЕРАТОР (1 шаг) =====
        self.optimizer_g.zero_grad()

        fake_pair_for_g = torch.cat([real, fake_pixel], dim=1)
        d_fake_for_g = self.discriminator(fake_pair_for_g)
        loss_g_gan = self.criterion_gan(d_fake_for_g, real_labels)

        # Уменьшаем lambda_l1, чтобы GAN loss работала сильнее
        loss_g_l1 = self.criterion_l1(fake_pixel, pixel) * self.lambda_l1

        loss_g = loss_g_gan + loss_g_l1
        loss_g.backward()
        self.optimizer_g.step()

        return loss_g.item(), loss_d.item()

    # ЧЕКПОИНТЫ
    def save_checkpoint(self, epoch):

        checkpoint = {
            'epoch': epoch,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
        }
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, path)
        print(f"  💾 Чекпоинт сохранён: {path}")

        torch.save(
            self.generator.state_dict(),
            self.checkpoint_dir / 'generator_final.pth'
        )

    #   ВИЗУАЛИЗАЦИЯ
    def visualize_progress(self, real, pixel, fake, epoch, step):

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        def to_pil(tensor):
            img = tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
            return (img * 255).astype(np.uint8)

        axes[0].imshow(to_pil(real[0]))
        axes[0].set_title('Original')
        axes[0].axis('off')

        axes[1].imshow(to_pil(pixel[0]))
        axes[1].set_title('Target Pixel Art')
        axes[1].axis('off')

        axes[2].imshow(to_pil(fake[0]))
        axes[2].set_title(f'Generated (epoch {epoch})')
        axes[2].axis('off')

        plt.tight_layout()
        save_path = self.checkpoint_dir / f'progress_epoch_{epoch}_step_{step}.png'
        plt.savefig(save_path, dpi=100)
        plt.close()

    # ЦИКЛ ОБУЧЕНИЯ
    def train(self, resume_from=None):

        print("=" * 60)
        print("🎯 НАЧАЛО ОБУЧЕНИЯ Pix2Pix")
        print("=" * 60)

        # Загрузка чекпоинта, если указан
        start_epoch = 1
        if resume_from:
            print(f"\n📥 Загрузка чекпоинта: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=self.device)

            self.generator.load_state_dict(checkpoint['generator'])
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])
            self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])

            start_epoch = checkpoint['epoch'] + 1
            print(f"✅ Продолжаем с эпохи {start_epoch}\n")

        self.generator.train()
        self.discriminator.train()

        for epoch in range(start_epoch, self.num_epochs + 1):  # ✅ 8 пробелов
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            num_batches = 0

            for step, (real, pixel) in enumerate(self.train_loader):
                real = real.to(self.device)
                pixel = pixel.to(self.device)

                g_loss, d_loss = self.train_step(real, pixel)

                epoch_g_loss += g_loss
                epoch_d_loss += d_loss
                num_batches += 1

                # Логируем каждый 10-й шаг
                if step % 10 == 0:
                    print(f"  Epoch [{epoch}/{self.num_epochs}] "
                          f"Step [{step}/{len(self.train_loader)}] "
                          f"G Loss: {g_loss:.4f} | D Loss: {d_loss:.4f}")

            # Средние потери за эпоху
            avg_g_loss = epoch_g_loss / max(num_batches, 1)
            avg_d_loss = epoch_d_loss / max(num_batches, 1)

            self.g_losses.append(avg_g_loss)
            self.d_losses.append(avg_d_loss)

            print(f"\n✅ Epoch {epoch} завершена | "
                  f"G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f}\n")

            # Визуализация каждые 5 эпох на 2 изображениях (увеличенный размер)
            if epoch % 5 == 0:
                try:
                    test_indices = [317, 346]

                    fig, axes = plt.subplots(len(test_indices), 3, figsize=(20, 10 * len(test_indices)))

                    if len(test_indices) == 1:
                        axes = axes.reshape(1, -1)

                    def to_pil(tensor):
                        img = tensor.detach().cpu().permute(1, 2, 0)
                        img = (img + 1) / 2
                        img = img.clamp(0, 1).numpy()
                        return (img * 255).astype(np.uint8)

                    for row_idx, img_idx in enumerate(test_indices):
                        try:
                            real_sample, pixel_sample = self.val_dataset[img_idx]
                            real_sample = real_sample.unsqueeze(0).to(self.device)
                            pixel_sample = pixel_sample.unsqueeze(0).to(self.device)

                            with torch.no_grad():
                                fake_sample = self.generator(real_sample)

                            # Увеличенный размер шрифта для заголовков
                            axes[row_idx, 0].imshow(to_pil(real_sample[0]))
                            axes[row_idx, 0].set_title(f'Original (#{img_idx})', fontsize=14, fontweight='bold', pad=15)
                            axes[row_idx, 0].axis('off')

                            axes[row_idx, 1].imshow(to_pil(pixel_sample[0]))
                            axes[row_idx, 1].set_title('Target Pixel Art', fontsize=14, fontweight='bold', pad=15)
                            axes[row_idx, 1].axis('off')

                            axes[row_idx, 2].imshow(to_pil(fake_sample[0]))
                            axes[row_idx, 2].set_title(f'Generated (epoch {epoch})', fontsize=14, fontweight='bold',
                                                       pad=15)
                            axes[row_idx, 2].axis('off')

                        except Exception as e:
                            print(f"  ⚠️ Ошибка обработки изображения #{img_idx}: {e}")
                            continue

                    plt.tight_layout(pad=3.0)  # Увеличенные отступы между картинками
                    save_path = self.checkpoint_dir / f'progress_epoch_{epoch}_multi.png'
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')  # Высокое DPI и обрезка лишних полей
                    plt.close()

                    print(f"  📸 Визуализация сохранена: {save_path} ({len(test_indices)} изображения)")

                except Exception as e:
                    print(f"  ⚠️ Ошибка визуализации: {e}")

            # Сохраняем чекпоинт каждые N эпох
            if epoch % self.save_every == 0 or epoch == self.num_epochs:
                self.save_checkpoint(epoch)

        print("=" * 60)
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print(f"📁 Чекпоинты: {self.checkpoint_dir}")
        print(f"📁 Финальная модель: {self.checkpoint_dir / 'generator_final.pth'}")
        print("=" * 60)

        # Сохраняем графики потерь
        self.plot_losses()
    def plot_losses(self):
        """График потерь за всё обучение."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.g_losses, label='Generator Loss', color='blue')
        plt.plot(self.d_losses, label='Discriminator Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = self.checkpoint_dir / 'losses.png'
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"📊 График потерь сохранён: {save_path}")


if __name__ == "__main__":
    trainer = Pix2PixTrainer(
        dataset_dir='dataset',
        batch_size=1,
        lr=0.0004,
        num_epochs=250,
        img_size=512,
        checkpoint_dir='checkpoints',
        save_every=5
    )
    trainer.train(resume_from='checkpoints/checkpoint_epoch_40.pth')
