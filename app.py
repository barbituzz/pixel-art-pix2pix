import gradio as gr
import torch
from PIL import Image
import torchvision.transforms as transforms
from models import Generator
import numpy as np
import os
import tempfile


class PixelArtApp:
    def __init__(self, model_path='checkpoints/generator_final.pth'):
        # Проверяем наличие GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Загрузка модели на: {self.device}")

        # Загружаем архитектуру
        self.generator = Generator().to(self.device)

        # Загружаем веса
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.generator.load_state_dict(state_dict)
            self.generator.eval()
            print("Модель успешно загружена.")
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            raise


        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def convert_to_pixel_art(self, image):
        if image is None:
            return None, None, "️ Пожалуйста, загрузите изображение"

        try:
            original_size = image.size

            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output_tensor = self.generator(input_tensor)

            output_image = (output_tensor.cpu().squeeze() + 1) / 2
            output_image = output_image.permute(1, 2, 0).numpy()
            output_image = (output_image * 255).astype(np.uint8)
            pixel_art_img = Image.fromarray(output_image)


            pixel_art_img = pixel_art_img.resize(original_size, Image.NEAREST)


            temp_dir = tempfile.mkdtemp()
            output_path = os.path.join(temp_dir, "pixel_art.jpg")

            if pixel_art_img.mode == 'RGBA':
                pixel_art_img = pixel_art_img.convert('RGB')

            pixel_art_img.save(output_path, 'JPEG', quality=95)

            return pixel_art_img, output_path, "✅ Успешно преобразовано! Файл готов к скачиванию."

        except Exception as e:
            return None, None, f"❌ Ошибка: {str(e)}"


def create_gradio_interface():
    model_path = 'checkpoints/generator_final.pth'

    try:
        app = PixelArtApp(model_path)
    except Exception as e:
        print(f"⚠️ Не удалось загрузить модель: {e}")
        print("Убедитесь, что обучение завершено и файл существует.")
        app = None

    def process_image(input_img):
        if app is None:
            return None, None, "Ошибка: Модель не загружена. Сначала запустите train.py"
        result, file_path, message = app.convert_to_pixel_art(input_img)
        return result, file_path, message

    # CSS для четких пикселей
    with gr.Blocks(
            title="Pixel Art AI Converter",
            theme=gr.themes.Soft(),
            css=".output-image img { image-rendering: pixelated !important; }"
    ) as demo:

        gr.Markdown("""
        # 🎨 AI Pixel Art Converter
        Загрузите фотографию, и нейросеть превратит её в пиксель-арт.
        """)

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label=" Загрузите фото", type="pil")
                convert_btn = gr.Button("✨ Преобразовать", variant="primary", size="lg")

            with gr.Column():
                output_image = gr.Image(label="🖼️ Результат", type="pil")
                download_file = gr.File(label=" Скачать JPG", visible=False)
                status_text = gr.Textbox(label="Статус", interactive=False)

        convert_btn.click(
            fn=process_image,
            inputs=[input_image],
            outputs=[output_image, download_file, status_text]
        )

    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True)  # ← Было share=False, стало True