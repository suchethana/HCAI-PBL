# pbl/apps.py
from django.apps import AppConfig

class PblAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'pbl'

    def ready(self):
        # Import the model training function here to ensure it's called on startup
        # This will load/train the model once when the Django app is ready
        from project4.views import train_matrix_factorization_model
        print("[PBL App Config] Calling train_matrix_factorization_model from ready method...")
        train_matrix_factorization_model()