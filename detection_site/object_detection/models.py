import os
import cv2
import numpy as np
from django.db import models
from django.conf import settings
from django.contrib.auth import get_user_model


# Модель DetectedObject
class DetectedObject(models.Model):
    image_feed = models.ForeignKey('ImageFeed', related_name='detected_objects', on_delete=models.CASCADE)
    object_type = models.CharField(max_length=100)
    confidence = models.FloatField()
    location = models.JSONField()

    def __str__(self):
        return f"{self.object_type} ({self.confidence * 100:.2f}%) на {self.image_feed.image.name}"


# Модель ImageFeed
class ImageFeed(models.Model):
    PROCESSING_CHOICES = [
        ('detection', 'Model 1 CPU (Быстро)'),
        ('segmentation', 'Model 2 GPU (Немного медленнее)'),
    ]

    user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE)
    image = models.ImageField(upload_to='images/')
    processed_image = models.ImageField(upload_to='processed_images/', null=True, blank=True)
    processing_type = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username} - {self.image.name}"

    def process_image(self):
        print("Начало обработки изображения.")
        if self.processing_type == 'detection':
            self.process_image_detection()
        elif self.processing_type == 'segmentation':
            self.process_image_segmentation()
        else:
            print(f"Неизвестный тип обработки: {self.processing_type}")

    def process_image_detection(self):
        try:
            print("Обработка изображения для обнаружения объектов (MobileNet SSD).")

            # Путь к папке с моделями обнаружения
            model_dir = os.path.join(settings.BASE_DIR, 'object_detection')
            prototxt_path = os.path.join(model_dir, 'mobilenet_ssd_deploy.prototxt')
            model_path = os.path.join(model_dir, 'mobilenet_iter_73000.caffemodel')

            # Проверка наличия файлов модели
            if not os.path.isfile(prototxt_path):
                raise FileNotFoundError(f"Файл прототипа модели не найден по пути: {prototxt_path}")
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Файл модели не найден по пути: {model_path}")

            # Загрузка модели MobileNetSSD
            net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            print("Модель MobileNetSSD успешно загружена.")

            # Загрузка изображения
            image = cv2.imread(self.image.path)
            if image is None:
                raise ValueError(f"Не удалось загрузить изображение: {self.image.path}")
            print("Изображение успешно загружено.")

            (h, w) = image.shape[:2]

            # Подготовка блоба для модели. Специальный формат данных, представляющий собой многомерный массив (тензор),
            # подготовленный для подачи на вход нейронной сети.
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
                                         0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            print("Обработка модели завершена.")

            # Список классов для MobileNet SSD (Pascal VOC)
            VOC_CLASS_NAMES = [
                'background', 'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor'
            ]

            # Цвета для рисования рамок
            colors = np.random.uniform(0, 255, size=(len(VOC_CLASS_NAMES), 3))

            # Копия изображения для рисования
            result_image = image.copy()

            # Порог уверенности
            confidence_threshold = 0.5

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > confidence_threshold:
                    class_id = int(detections[0, 0, i, 1])
                    # Проверяем, что class_id не выходит за пределы списка
                    if class_id < len(VOC_CLASS_NAMES):
                        object_type = VOC_CLASS_NAMES[class_id]
                    else:
                        object_type = 'Unknown'

                    # Координаты ограничивающего прямоугольника
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")

                    # Рисуем рамку и метку на изображении
                    color = colors[class_id % len(colors)]
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                    label = f"{object_type}: {confidence * 100:.2f}%"
                    cv2.putText(result_image, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Сохраняем информацию об обнаруженном объекте
                    location = {'x': int(x1), 'y': int(y1),
                                'width': int(x2 - x1), 'height': int(y2 - y1)}
                    DetectedObject.objects.create(
                        image_feed=self,
                        object_type=object_type,
                        confidence=float(confidence),
                        location=location
                    )

            # Сохранение обработанного изображения
            processed_image_name = f"processed_{os.path.basename(self.image.name)}"
            processed_image_path = os.path.join(settings.MEDIA_ROOT, 'processed_images', processed_image_name)
            cv2.imwrite(processed_image_path, result_image)
            self.processed_image.name = f"processed_images/{processed_image_name}"
            self.save()

            print(f"Успешно обработано изображение с ID {self.id}")

        except Exception as e:
            print(f"Ошибка при обработке изображения (обнаружение объектов): {e}")
            import traceback
            traceback.print_exc()

    def process_image_segmentation(self):
        try:
            import torch
            from torchvision import models, transforms
            from PIL import Image
            from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

            print("Обработка изображения для сегментации объектов (Mask R-CNN с torchvision).")

            # Загрузка модели
            weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            model = models.detection.maskrcnn_resnet50_fpn(weights=weights)
            model.eval()
            print("Модель Mask R-CNN успешно загружена.")

            # Преобразования для входного изображения
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

            # Загрузка и преобразование изображения
            image = Image.open(self.image.path).convert("RGB")
            orig_width, orig_height = image.size
            image_tensor = transform(image)

            # Обработка изображения моделью
            with torch.no_grad():
                outputs = model([image_tensor])
            print("Модель успешно обработала изображение.")

            # Обработка результатов
            scores = outputs[0]['scores'].numpy()
            labels = outputs[0]['labels'].numpy()
            boxes = outputs[0]['boxes'].numpy()
            masks = outputs[0]['masks'].numpy()

            # Порог уверенности
            confidence_threshold = 0.5

            # Список классов COCO
            COCO_INSTANCE_CATEGORY_NAMES = [
                '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
                'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
                'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                'hair drier', 'toothbrush'
            ]

            # Создаем пустое изображение с альфа-каналом (прозрачный фон)
            result_image = Image.new("RGBA", image.size)

            print(f"Количество обнаруженных объектов: {len(scores)}")

            for i in range(len(scores)):
                if scores[i] > confidence_threshold:
                    box = boxes[i]
                    label_idx = labels[i]
                    mask = masks[i, 0]
                    mask = mask > 0.5

                    # Получаем имя класса
                    if label_idx < len(COCO_INSTANCE_CATEGORY_NAMES):
                        object_type = COCO_INSTANCE_CATEGORY_NAMES[label_idx]
                    else:
                        object_type = 'Unknown'

                    # Создаем маску изображения
                    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
                    # Выделяем объект на оригинальном изображении
                    object_image = Image.new("RGBA", image.size)
                    object_image.paste(image, mask=mask_image)

                    # Накладываем объект на результирующее изображение
                    result_image = Image.alpha_composite(result_image, object_image)

                    # Сохраняем информацию об обнаруженном объекте
                    location = {
                        'x': int(box[0]),
                        'y': int(box[1]),
                        'width': int(box[2] - box[0]),
                        'height': int(box[3] - box[1])
                    }
                    DetectedObject.objects.create(
                        image_feed=self,
                        object_type=object_type,
                        confidence=float(scores[i]),
                        location=location
                    )

            # Сохранение обработанного изображения в формате PNG
            processed_image_name = f"processed_{os.path.splitext(os.path.basename(self.image.name))[0]}.png"
            processed_image_path = os.path.join(settings.MEDIA_ROOT, 'processed_images', processed_image_name)
            result_image.save(processed_image_path)
            self.processed_image.name = f"processed_images/{processed_image_name}"
            self.save()

            print(f"Успешно обработано изображение с ID {self.id}")

        except Exception as e:
            print(f"Ошибка при обработке изображения (сегментация объектов): {e}")
            import traceback
            traceback.print_exc()
