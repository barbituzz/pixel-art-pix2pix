from dataset import PixelArtDataset
from PIL import Image
import os

# Загружаем валидационный датасет
val_dataset = PixelArtDataset(root_dir='dataset', mode='val')

print(f"Всего изображений в валидации: {len(val_dataset)}")
print("\nПримеры индексов:")

# Показываем первые 10 файлов
for i in range(min(10, len(val_dataset))):
    file_path = val_dataset.files[i]
    print(f"  [{i:3d}] {file_path.name}")

# Или выбери конкретные индексы для проверки
test_indices = [1, 22, 56, 178, 622]
print(f"\nПроверка конкретных индексов {test_indices}:")
for idx in test_indices:
    if idx < len(val_dataset):
        file_path = val_dataset.files[idx]
        print(f"  [{idx:3d}] {file_path.name}")