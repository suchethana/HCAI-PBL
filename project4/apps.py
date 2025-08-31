from django.apps import AppConfig


class Project4Config(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'project4'

    def ready(self):
        pass
