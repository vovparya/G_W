from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import ImageFeed


@receiver(post_save, sender=ImageFeed)
def image_feed_post_save(sender, instance, created, **kwargs):
    if created:
        print(f"Сохранен новый ImageFeed с ID {instance.id}, запускаем обработку...")
        instance.process_image()
