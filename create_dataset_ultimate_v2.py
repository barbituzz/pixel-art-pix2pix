import os
import random
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from pathlib import Path


class FinalAugmentedDataset:


    def __init__(self, input_folder="your_photos", output_folder="dataset", num_samples=700):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.num_samples = num_samples

        (self.output_folder / 'real').mkdir(parents=True, exist_ok=True)
        (self.output_folder / 'pixel').mkdir(parents=True, exist_ok=True)

    def create_clean_pixel_art(self, image, pixel_size=4, max_colors=48):

        w, h = image.size
        grid_w = w // pixel_size
        grid_h = h // pixel_size

        # Уменьшаем до пиксельной сетки
        small = image.resize((grid_w, grid_h), Image.LANCZOS)

        # Квантование цветов
        quantized = small.quantize(colors=max_colors, method=Image.MEDIANCUT, dither=Image.NONE)
        clean = quantized.convert('RGB')
        clean_arr = np.array(clean)

        # Мягкие контуры на пиксельной сетке
        gray = cv2.cvtColor(clean_arr, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 1)
        edges = cv2.Canny(blurred, 80, 200)

        # Только сильные границы
        mask = edges > 150
        clean_arr[mask] = (clean_arr[mask] * 0.7).astype(np.uint8)

        #  Увеличиваем обратно
        final = Image.fromarray(clean_arr).resize((w, h), Image.NEAREST)

        return final

    def augment_pair(self, img_a, img_b, idx):

        # Зеркальная аугментация:

        pairs = []

        # 1. Оригинальная пара
        pairs.append((f"orig_{idx:05d}", img_a, img_b))

        # 2. Зеркальное отражение (синхронно для A и B)
        pairs.append((f"flip_{idx:05d}",
                      img_a.transpose(Image.FLIP_LEFT_RIGHT),
                      img_b.transpose(Image.FLIP_LEFT_RIGHT)))

        return pairs

    def create_dataset(self):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        image_files = sorted([
                                 f for f in self.input_folder.iterdir()
                                 if f.suffix.lower() in image_extensions
                             ][:self.num_samples])

        total_pairs = 0
        success_count = 0

        for i, img_path in enumerate(image_files):
            try:
                print(f"[{i + 1}/{len(image_files)}] {img_path.name}")

                # Загрузка оригинала и  к 512x512
                orig_img = Image.open(img_path).convert('RGB')
                orig_img = orig_img.resize((512, 512), Image.LANCZOS)

                # Пиксель-арт
                pixel_img = self.create_clean_pixel_art(orig_img)

                # Генерация аугментированных пар
                augmented_pairs = self.augment_pair(orig_img, pixel_img, i)

                # Сохраняем
                for name, img_a, img_b in augmented_pairs:
                    img_a.save(self.output_folder / 'real' / f"{name}.png", 'PNG')
                    img_b.save(self.output_folder / 'pixel' / f"{name}.png", 'PNG')
                    total_pairs += 1

                success_count += 1

            except Exception as e:
                print(f"  ❌ Ошибка с {img_path.name}: {e}")
                continue

if __name__ == "__main__":
    creator = FinalAugmentedDataset(
        input_folder="your_photos",
        output_folder="dataset",
        num_samples=709
    )
    creator.create_dataset()