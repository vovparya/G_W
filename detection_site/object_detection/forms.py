from django import forms
from .models import ImageFeed


class ImageFeedForm(forms.ModelForm):
    PROCESSING_CHOICES = [
        ('detection', 'Обнаружение объектов (MobileNet SSD)'),
        ('segmentation', 'Сегментация объектов (Mask R-CNN ResNet50 FPN)')
    ]

    processing_type = forms.ChoiceField(choices=PROCESSING_CHOICES, widget=forms.RadioSelect)

    class Meta:
        model = ImageFeed
        fields = ['image', 'processing_type']
        labels = {
            'image': 'Изображение:',
        }
