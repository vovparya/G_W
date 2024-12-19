from django.apps import AppConfig


class ObjectDetectionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'object_detection'

    def ready(self):
        import object_detection.signals
