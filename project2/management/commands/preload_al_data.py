import os
import pickle
import numpy as np
from django.core.management.base import BaseCommand
from django.conf import settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import save_npz # For efficient sparse matrix saving/loading

from project2.utils import get_full_imdb_data # Assuming this is accessible

# Define paths for preloaded data within your MODEL_DIR
PRELOAD_DIR = os.path.join(settings.BASE_DIR, 'al_preload_data')
os.makedirs(PRELOAD_DIR, exist_ok=True)

X_TRAIN_TRANSFORMED_PATH = os.path.join(PRELOAD_DIR, 'X_train_transformed.npz')
Y_TRAIN_ENCODED_PATH = os.path.join(PRELOAD_DIR, 'y_train_encoded.pkl')
X_TEST_TRANSFORMED_PATH = os.path.join(PRELOAD_DIR, 'X_test_transformed.npz')
Y_TEST_ENCODED_PATH = os.path.join(PRELOAD_DIR, 'y_test_encoded.pkl')
VECTORIZER_PATH = os.path.join(PRELOAD_DIR, 'tfidf_vectorizer_al.pkl')
LABEL_ENCODER_PATH = os.path.join(PRELOAD_DIR, 'label_encoder_al.pkl')
# --- NEW PATH FOR MODEL (initial model only) ---
INITIAL_MODEL_PATH = os.path.join(PRELOAD_DIR, 'initial_al_model.pkl')


class Command(BaseCommand):
    help = 'Preloads and processes IMDB data for faster Active Learning initialization.'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("Starting Active Learning data preloading..."))

        # 1. Load full raw data
        self.stdout.write("Loading full IMDB dataset (this may take a while)...")
        X_train_raw, y_train_raw, X_test_raw, y_test_raw = get_full_imdb_data(settings.IMDB_DATA_PATH)
        self.stdout.write(self.style.SUCCESS("IMDB data loaded."))

        # 2. Initialize and fit TF-IDF Vectorizer
        self.stdout.write("Fitting TF-IDF Vectorizer...")
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_transformed = vectorizer.fit_transform(X_train_raw)
        X_test_transformed = vectorizer.transform(X_test_raw)
        self.stdout.write(self.style.SUCCESS("TF-IDF Vectorizer fitted and data transformed."))

        # 3. Label encode target
        self.stdout.write("Encoding labels...")
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train_raw)
        y_test_encoded = le.transform(y_test_raw)
        self.stdout.write(self.style.SUCCESS("Labels encoded."))

        # 4. Save processed data and models
        self.stdout.write(f"Saving preloaded data to: {PRELOAD_DIR}...")
        try:
            save_npz(X_TRAIN_TRANSFORMED_PATH, X_train_transformed)
            save_npz(X_TEST_TRANSFORMED_PATH, X_test_transformed)

            with open(Y_TRAIN_ENCODED_PATH, 'wb') as f:
                pickle.dump(y_train_encoded, f)
            with open(Y_TEST_ENCODED_PATH, 'wb') as f:
                pickle.dump(y_test_encoded, f)
            with open(VECTORIZER_PATH, 'wb') as f:
                pickle.dump(vectorizer, f)
            with open(LABEL_ENCODER_PATH, 'wb') as f:
                pickle.dump(le, f)

            # --- OPTIONAL: Save a dummy initial model or placeholder (not strictly needed as it's retrained)
            # You could save a simple LogisticRegression object here if you wanted.
            # from sklearn.linear_model import LogisticRegression
            # dummy_model = LogisticRegression(random_state=42)
            # with open(INITIAL_MODEL_PATH, 'wb') as f:
            #     pickle.dump(dummy_model, f)

            self.stdout.write(self.style.SUCCESS("Active Learning data preloaded and saved successfully!"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error saving preloaded data: {e}"))
            import traceback
            traceback.print_exc()