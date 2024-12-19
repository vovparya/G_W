from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth import login, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from .models import ImageFeed
from .utils import process_image
from .forms import ImageFeedForm
import json


def home(request):
    return render(request, 'object_detection/home.html')


def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('object_detection:dashboard')
    else:
        form = UserCreationForm()
    return render(request, 'object_detection/register.html', {'form': form})


def user_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('object_detection:dashboard')
    else:
        form = AuthenticationForm()
    return render(request, 'object_detection/login.html', {'form': form})


def upload_image(request):
    if request.method == 'POST':
        form = ImageFeedForm(request.POST, request.FILES)
        if form.is_valid():
            image_feed = form.save(commit=False)
            image_feed.user = request.user
            image_feed.save()
            return redirect('image_detail', pk=image_feed.pk)
    else:
        form = ImageFeedForm()
    return render(request, 'upload_image.html', {'form': form})


def image_detail(request, pk):
    image_feed = ImageFeed.objects.get(pk=pk)
    return render(request, 'image_detail.html', {'image_feed': image_feed})


@login_required
def user_logout(request):
    logout(request)
    return redirect('object_detection:login')


@login_required
def dashboard(request):
    image_feeds = ImageFeed.objects.filter(user=request.user)
    return render(request, 'object_detection/dashboard.html', {'image_feeds': image_feeds})


@login_required
def process_image_feed(request, feed_id):
    image_feed = get_object_or_404(ImageFeed, id=feed_id, user=request.user)
    if not image_feed.processed_image:
        process_image(image_feed)  # Consider handling this asynchronously
    return redirect('object_detection:dashboard')


@login_required
def add_image_feed(request):
    # Передача информации о модели из представления. Управление информацией о моделях с сервера.
    model_info = {
        'detection': {
            'name': 'MobileNet SSD (Обнаружение объектов)',
            'description': (
                'Лёгкая и быстрая модель для обнаружения объектов, обученная на датасете PASCAL VOC.\n'
                '***Поддерживаемые форматы изображений:*** .jpg, .jpeg, .png, .bmp, .webp, .tif'
            ),
            'details': (
                '***Архитектура модели: MobileNet SSD (Обнаружение объектов)\n'
                '***Разработчики:\n'
                '*****SSD: Разработан Google и другими в 2016 году.\n'
                '*****MobileNet: Разработан Google в 2017 году.\n'
                '*****Комбинация MobileNet SSD: Широко используется сообществом с 2017 года.\n'
                '***Год выпуска модели: Около 2017 года.\n'
                '***Датасет обучения: PASCAL VOC.\n'
                '***Количество классов объектов:*** 20\n'
                '***Объекты для распознавания:*** '
                'aeroplane (самолёт), bicycle (велосипед), bird (птица), boat (лодка), bottle (бутылка), '
                'bus (автобус), car (автомобиль), cat (кошка), chair (стул), cow (корова), '
                'diningtable (обеденный стол), dog (собака), horse (лошадь), motorbike (мотоцикл), '
                'person (человек), pottedplant (комнатное растение), sheep (овца), sofa (диван), '
                'train (поезд), tvmonitor (телевизор/монитор).\n'
                '***Области применения:***\n'
                '- Реальное время, где важна скорость обработки.\n'
                '- Приложения с ограниченными вычислительными ресурсами.\n'
                '- Системы видеонаблюдения и мониторинга.'
            )
        },
        'segmentation': {
            'name': 'Mask R-CNN с ResNet50 FPN (Сегментация объектов)',
            'description': (
                'Мощная модель для сегментации объектов, обученная на датасете COCO.\n'
                '***Поддерживаемые форматы изображений:*** .jpg, .jpeg, .png, .bmp, .webp, .tif, .gif'
            ),
            'details': (
                '***Архитектура модели: Mask R-CNN с ResNet50 FPN (Сегментация объектов)\n'
                '***Разработчики:*** Facebook AI Research (FAIR)\n'
                '***Год выпуска модели:*** 2017\n'
                '***Датасет обучения: COCO\n'
                '***Количество классов объектов:*** 80\n'
                '***Объекты для распознавания:***\n'
                'person (человек), bicycle (велосипед), car (автомобиль), motorcycle (мотоцикл), airplane (самолёт), '
                'bus (автобус), train (поезд), truck (грузовик), boat (лодка), traffic light (светофор), '
                'fire hydrant (пожарный гидрант), stop sign (дорожный знак "стоп"), parking meter (парковочный счётчик), '
                'bench (скамейка), bird (птица), cat (кошка), dog (собака), horse (лошадь), sheep (овца), cow (корова), '
                'elephant (слон), bear (медведь), zebra (зебра), giraffe (жираф), backpack (рюкзак), umbrella (зонт), '
                'handbag (сумка), tie (галстук), suitcase (чемодан), frisbee (фрисби), skis (лыжи), snowboard (сноуборд), '
                'sports ball (мяч), kite (воздушный змей), baseball bat (бейсбольная бита), '
                'baseball glove (бейсбольная перчатка), skateboard (скейтборд), surfboard (серфборд), '
                'tennis racket (теннисная ракетка), bottle (бутылка), wine glass (бокал для вина), cup (чашка), '
                'fork (вилка), knife (нож), spoon (ложка), bowl (миска), banana (банан), apple (яблоко), '
                'sandwich (бутерброд), orange (апельсин), broccoli (брокколи), carrot (морковь), hot dog (хот-дог), '
                'pizza (пицца), donut (пончик), cake (торт), chair (стул), couch (диван), '
                'potted plant (комнатное растение), bed (кровать), dining table (обеденный стол), toilet (туалет), '
                'tv (телевизор), laptop (ноутбук), mouse (компьютерная мышь), remote (пульт дистанционного управления), '
                'keyboard (клавиатура), cell phone (мобильный телефон), microwave (микроволновка), oven (духовка), '
                'toaster (тостер), sink (раковина), refrigerator (холодильник), book (книга), clock (часы), vase (ваза), '
                'scissors (ножницы), teddy bear (плюшевый мишка), hair drier (фен), toothbrush (зубная щётка).\n'
                '***Области применения:***\n'
                '- Задачи с высокой точностью сегментации.\n'
                '- Медицинская визуализация, автономное вождение, робототехника.\n'
                '- Анализ изображений и видео, требующий детализации.'
            )
        }
    }

    # Сериализуем model_info в JSON
    model_info_json = json.dumps(model_info, ensure_ascii=False)

    if request.method == 'POST':
        form = ImageFeedForm(request.POST, request.FILES)
        if form.is_valid():
            image_feed = form.save(commit=False)
            image_feed.user = request.user
            image_feed.save()
            # Запуск обработки изображения в зависимости от выбранной модели
            image_feed.process_image()
            return redirect('object_detection:dashboard')
    else:
        form = ImageFeedForm()
    return render(request, 'object_detection/add_image_feed.html',
                  {'form': form, 'model_info_json': model_info_json})


@login_required
def delete_image(request, image_id):
    image = get_object_or_404(ImageFeed, id=image_id, user=request.user)  # Ensuring only the owner can delete
    image.delete()
    return redirect('object_detection:dashboard')
