import os
import pickle


def load_imdb_dataset(data_path, sentiment):
    reviews = []
    sentiment_path = os.path.join(data_path, sentiment)
    for filename in os.listdir(sentiment_path):
        if filename.endswith(".txt"):
            with open(os.path.join(sentiment_path, filename), "r", encoding="utf-8") as f:
                reviews.append(f.read())
    return reviews


def get_full_imdb_data(base_imdb_path, progress_callback=None):

    if progress_callback:
        progress_callback("Loading training data (positive reviews)...")
    train_pos_reviews = load_imdb_dataset(os.path.join(base_imdb_path, 'train'), 'pos')

    if progress_callback:
        progress_callback("Loading training data (negative reviews)...")
    train_neg_reviews = load_imdb_dataset(os.path.join(base_imdb_path, 'train'), 'neg')

    X_train = train_pos_reviews + train_neg_reviews
    y_train = [1] * len(train_pos_reviews) + [0] * len(train_neg_reviews)

    if progress_callback:
        progress_callback("Loading test data (positive reviews)...")
    test_pos_reviews = load_imdb_dataset(os.path.join(base_imdb_path, 'test'), 'pos')

    if progress_callback:
        progress_callback("Loading test data (negative reviews)...")
    test_neg_reviews = load_imdb_dataset(os.path.join(base_imdb_path, 'test'), 'neg')

    X_test = test_pos_reviews + test_neg_reviews
    y_test = [1] * len(test_pos_reviews) + [0] * len(test_neg_reviews)

    if progress_callback:
        progress_callback(f"Data loaded: {len(X_train)} training samples, {len(X_test)} test samples.")

    return X_train, y_train, X_test, y_test


def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)