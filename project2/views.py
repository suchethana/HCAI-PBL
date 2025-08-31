import os
import pickle
import matplotlib

matplotlib.use('Agg')

from django.shortcuts import render, HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.conf import settings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import clone
from sklearn.model_selection import train_test_split

import numpy as np

from .utils import get_full_imdb_data, save_model, load_model

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import traceback

import base64


MODEL_DIR = os.path.join(settings.BASE_DIR, 'models_project2')
os.makedirs(MODEL_DIR, exist_ok=True)

TFIDF_VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'logistic_regression_classifier.pkl')


AL_CURRENT_MODEL_PATH = os.path.join(MODEL_DIR, 'al_current_model.pkl')
MAX_QUERIES_ALLOWED = 100
TARGET_ACCURACY = 0.95
BATCH_SIZE_MIXTURE = 5
UNCERTAINTY_PORTION = 0.8


def index(request):

    context = {
        'accuracy': request.GET.get('accuracy'),
        'trained': request.GET.get('trained', 'false') == 'true',
        'message': request.GET.get('message'),
        'error': request.GET.get('error'),

        'active_status': request.GET.get('active_status'),
        'current_accuracy': request.GET.get('current_accuracy'),
        'queries': request.GET.get('queries'),
        'final_accuracy': request.GET.get('final_accuracy'),
    }

    context['al_initialized'] = 'active_learning_state' in request.session and \
                                request.session.get('active_learning_state', {}).get('labeled_indices') is not None

    if context['al_initialized']:
        al_state = request.session.get('active_learning_state', {})
        context['num_labeled'] = len(al_state.get('labeled_indices', []))
        context['num_unlabeled'] = len(al_state.get('unlabeled_indices', []))
        context['num_queries_made'] = al_state.get('num_queries_made', 0)
        context['active_learning_accuracy_history'] = al_state.get('active_learning_accuracy_history', [])

    return render(request, 'project2/index.html', context)


def train_baseline_classifier(request):
    if request.method == 'POST':
        try:
            def report_progress(message):
                print(f"[Project 2 Training Status] {message}")

            report_progress("Starting data loading...")
            X_train_raw_full, y_train_full, X_test_raw, y_test = get_full_imdb_data(settings.IMDB_DATA_PATH,
                                                                                    progress_callback=report_progress)

            report_progress("Initializing TF-IDF Vectorizer and transforming text...")
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            X_train_transformed = vectorizer.fit_transform(X_train_raw_full)
            X_test_transformed = vectorizer.transform(X_test_raw)

            report_progress("Training Logistic Regression classifier...")
            classifier = LogisticRegression(max_iter=1000, random_state=42)
            classifier.fit(X_train_transformed, y_train_full)

            report_progress("Calculating test accuracy...")
            y_pred = classifier.predict(X_test_transformed)
            accuracy = accuracy_score(y_test, y_pred)

            report_progress(f"Training complete. Test Accuracy: {accuracy:.4f}")

            report_progress("Saving trained models...")
            save_model(vectorizer, TFIDF_VECTORIZER_PATH)
            save_model(classifier, CLASSIFIER_PATH)
            report_progress("Models saved.")

            return HttpResponseRedirect(reverse('project2:index') + f'?accuracy={accuracy:.4f}&trained=true')

        except Exception as e:
            print(f"[Project 2 Training Error] {e}")
            return HttpResponseRedirect(reverse('project2:index') + f'?error={e}')

    return HttpResponseRedirect(reverse('project2:index'))


def load_pretrained_classifier(request):
    if request.method == 'POST':
        try:
            loaded_vectorizer = load_model(TFIDF_VECTORIZER_PATH)
            loaded_classifier = load_model(CLASSIFIER_PATH)

            message_lines = [
                "Pre-trained model (TF-IDF Vectorizer & Logistic Regression)",
                "loaded successfully."
            ]

            return render(request, 'project2/index.html', {'message_lines': message_lines})

        except FileNotFoundError:
            message = "No pre-trained model found. Please train the baseline classifier first."
            return render(request, 'project2/index.html', {'error': message})
        except Exception as e:
            message = f"Error loading pre-trained model: {e}"
            traceback.print_exc()
            return render(request, 'project2/index', {'error': message})

    return HttpResponseRedirect(reverse('project2:index'))


def active_learn_init(request):
    if request.method == 'POST':
        try:
            if not os.path.exists(TFIDF_VECTORIZER_PATH):
                raise FileNotFoundError("Baseline vectorizer not found. Please train baseline classifier first.")

            X_train_raw_full, y_train_full, X_test_raw, y_test = get_full_imdb_data(settings.IMDB_DATA_PATH)

            request.session['X_train_raw_full'] = X_train_raw_full
            request.session['y_train_full'] = y_train_full
            request.session['X_test_raw'] = X_test_raw
            request.session['y_test'] = y_test

            vectorizer = load_model(TFIDF_VECTORIZER_PATH)
            request.session['fitted_vectorizer_pkl_b64'] = base64.b64encode(pickle.dumps(vectorizer)).decode('utf-8')

            num_initial_labeled = 20

            if len(X_train_raw_full) < num_initial_labeled:
                raise ValueError(
                    f"Dataset has only {len(X_train_raw_full)} samples, cannot initialize with {num_initial_labeled} labeled samples.")

            all_train_indices = np.arange(len(X_train_raw_full))
            np.random.shuffle(all_train_indices)

            initial_indices = all_train_indices[:num_initial_labeled]
            unlabeled_indices = all_train_indices[num_initial_labeled:]

            al_state = {
                'labeled_indices': [int(i) for i in initial_indices],
                'unlabeled_indices': [int(i) for i in unlabeled_indices],
                'num_queries_made': int(0),
                'current_accuracy': None,
                'al_terminated': False,
                'termination_reason': '',
                'final_accuracy': None,
                'model_path': AL_CURRENT_MODEL_PATH,
                'active_learning_accuracy_history': []
            }
            request.session['active_learning_state'] = al_state
            request.session.modified = True

            X_initial_labeled_raw = [X_train_raw_full[i] for i in initial_indices]
            y_initial_labeled = [y_train_full[i] for i in initial_indices]


            if len(np.unique(y_initial_labeled)) < 2:
                raise ValueError(
                    "Initial labeled set does not have at least two classes. Cannot train a meaningful model.")


            X_initial_transformed = vectorizer.transform(X_initial_labeled_raw)


            classifier = LogisticRegression(max_iter=1000, random_state=42)
            classifier.fit(X_initial_transformed, y_initial_labeled)


            print(f"[Project 2 AL Init Status] Attempting to save initial AL model to: {AL_CURRENT_MODEL_PATH}")
            save_model(classifier, AL_CURRENT_MODEL_PATH)
            print(f"[Project 2 AL Init Status] Initial AL model saved successfully.")


            message = f"Active Learning initialized with {num_initial_labeled} initial labeled samples."
            return HttpResponseRedirect(reverse('project2:index') + f'?active_status={message}')

        except FileNotFoundError as e:
            return HttpResponseRedirect(
                reverse('project2:index') + f'?error=AL Init Error: {e}. Please train the baseline classifier first.')
        except Exception as e:
            traceback.print_exc()
            return HttpResponseRedirect(reverse('project2:index') + f'?error=AL Init Error: {e}')

    return HttpResponseRedirect(reverse('project2:index'))


def active_learn_query(request):
    if request.method == 'POST':
        query_strategy = request.POST.get('query_strategy', 'least_confidence')

        BATCH_SIZE_MIXTURE = 5
        UNCERTAINTY_PORTION = 0.8

        al_state = request.session.get('active_learning_state')
        if al_state is None:
            return HttpResponseRedirect(reverse('project2:index') + '?error=AL_Session_Expired_Please_Reinitialize')

        labeled_indices = al_state.get('labeled_indices', [])
        unlabeled_indices = al_state.get('unlabeled_indices', [])
        num_queries_made = al_state.get('num_queries_made', 0)
        accuracy_history = al_state.get('active_learning_accuracy_history', [])

        model_path = al_state.get('model_path')
        if model_path is None:
            raise ValueError("AL Session incomplete: model_path not found in state. Reinitialize.")

        X_train_raw_full = request.session.get('X_train_raw_full')
        y_train_full = request.session.get('y_train_full')
        X_test_raw = request.session.get('X_test_raw')
        y_test = request.session.get('y_test')

        fitted_vectorizer_pkl_b64 = request.session.get('fitted_vectorizer_pkl_b64')

        try:
            if fitted_vectorizer_pkl_b64 is None:
                raise ValueError("AL Session Incomplete. Please Initialize Active Learning again.")
            fitted_vectorizer = pickle.loads(base64.b64decode(fitted_vectorizer_pkl_b64))

            MIN_SAMPLES_FOR_TRAINING = 2
            if len(labeled_indices) < MIN_SAMPLES_FOR_TRAINING:
                if num_queries_made == 0:
                    X_labeled_current_raw = [X_train_raw_full[i] for i in labeled_indices]
                    y_labeled_current = [y_train_full[i] for i in labeled_indices]

                    if len(np.unique(y_labeled_current)) < 2:
                        raise ValueError("Not enough classes in initial labeled set. Cannot train model yet.")

                    classifier = LogisticRegression(max_iter=1000, random_state=42)
                    X_labeled_transformed = fitted_vectorizer.transform(X_labeled_current_raw)
                    classifier.fit(X_labeled_transformed, y_labeled_current)

                    X_test_transformed = fitted_vectorizer.transform(X_test_raw)
                    y_pred_test = classifier.predict(fitted_vectorizer.transform(X_test_raw))
                    current_accuracy = accuracy_score(y_test, y_pred_test)

                    al_state['current_accuracy'] = float(current_accuracy)
                    al_state['active_learning_accuracy_history'].append(
                        {"queries": int(num_queries_made), "accuracy": float(current_accuracy)})
                    request.session['active_learning_state'] = al_state
                    save_model(classifier, model_path)
                else:
                    raise ValueError(
                        f"Labeled samples fell below minimum for training: {len(labeled_indices)}/{MIN_SAMPLES_FOR_TRAINING}.")

            al_terminated = False
            termination_reason = ""
            final_accuracy = None

            if not unlabeled_indices:
                al_terminated = True
                termination_reason = "Unlabeled pool exhausted."
                final_accuracy = al_state['current_accuracy']
            elif num_queries_made >= settings.MAX_QUERIES_ALLOWED:
                al_terminated = True
                termination_reason = f"Max queries ({settings.MAX_QUERIES_ALLOWED}) reached."
                final_accuracy = al_state['current_accuracy']
            elif al_state['current_accuracy'] is not None and al_state['current_accuracy'] >= settings.TARGET_ACCURACY:
                al_terminated = True
                termination_reason = f"Target accuracy ({settings.TARGET_ACCURACY * 100:.0f}%) achieved."
                final_accuracy = al_state['current_accuracy']

            al_state['al_terminated'] = al_terminated
            al_state['termination_reason'] = termination_reason
            al_state['final_accuracy'] = final_accuracy
            request.session['active_learning_state'] = al_state
            request.session.modified = True

            if al_terminated:
                display_accuracy = al_state.get('current_accuracy', "N/A")
                if display_accuracy != "N/A":
                    display_accuracy = f"{display_accuracy:.4f}"

                message = f"Active Learning terminated: {termination_reason}"
                return HttpResponseRedirect(reverse(
                    'project2:index') + f'?active_status={message}&final_accuracy={display_accuracy}&queries={num_queries_made}')


            samples_to_query_indices = []

            num_to_query = 1

            classifier = load_model(model_path)
            if not hasattr(classifier, 'predict_proba'):
                raise ValueError(
                    "Classifier (Logistic Regression) must support 'predict_proba' for uncertainty/mixture sampling.")

            X_unlabeled_current_raw = [X_train_raw_full[i] for i in unlabeled_indices]
            X_unlabeled_transformed = fitted_vectorizer.transform(X_unlabeled_current_raw)

            if X_unlabeled_transformed.shape[0] == 0:
                raise ValueError("No more valid unlabeled samples after transformation.")

            if query_strategy == 'least_confidence':

                probas = classifier.predict_proba(X_unlabeled_transformed)
                least_confidence_scores = 1 - np.max(probas, axis=1)
                query_index_in_unlabeled_relative = np.argmax(least_confidence_scores)
                samples_to_query_indices.append(unlabeled_indices[query_index_in_unlabeled_relative])

            elif query_strategy == 'random':

                query_index_in_unlabeled_relative = np.random.randint(0, len(unlabeled_indices))
                samples_to_query_indices.append(unlabeled_indices[query_index_in_unlabeled_relative])

            elif query_strategy == 'uncertainty_random_mixture':

                num_to_query = min(BATCH_SIZE_MIXTURE, len(unlabeled_indices))
                if num_to_query == 0:
                    raise ValueError("No more samples to query in the mixture strategy.")

                probas = classifier.predict_proba(X_unlabeled_transformed)
                least_confidence_scores = 1 - np.max(probas, axis=1)

                uncertainty_sorted_indices_relative = np.argsort(least_confidence_scores)[::-1]

                num_uncertain_samples_in_batch = int(num_to_query * UNCERTAINTY_PORTION)
                num_random_samples_in_batch = num_to_query - num_uncertain_samples_in_batch

                queried_batch_indices_relative = list(
                    uncertainty_sorted_indices_relative[:num_uncertain_samples_in_batch])

                if num_random_samples_in_batch > 0:
                    all_unqueried_relative_indices = list(
                        set(range(len(unlabeled_indices))) - set(queried_batch_indices_relative))

                    if len(all_unqueried_relative_indices) < num_random_samples_in_batch:
                        random_indices_relative = np.random.choice(len(unlabeled_indices),
                                                                   size=num_random_samples_in_batch,
                                                                   replace=False).tolist()
                    else:
                        random_indices_relative = np.random.choice(all_unqueried_relative_indices,
                                                                   size=num_random_samples_in_batch,
                                                                   replace=False).tolist()

                    queried_batch_indices_relative.extend(random_indices_relative)
                    queried_batch_indices_relative = list(set(queried_batch_indices_relative))

                samples_to_query_indices = [unlabeled_indices[i] for i in queried_batch_indices_relative[:num_to_query]]

                samples_to_query_indices = list(np.unique(samples_to_query_indices))[:num_to_query]


            else:
                raise ValueError("Invalid query strategy selected.")

            num_queried_this_step = 0
            for queried_global_idx in samples_to_query_indices:
                if queried_global_idx in unlabeled_indices:
                    labeled_indices.append(int(queried_global_idx))
                    unlabeled_indices.remove(queried_global_idx)
                    num_queried_this_step += 1

            if num_queried_this_step == 0:
                raise ValueError("No new samples successfully queried from pool. Pool exhausted or selection issue.")

            al_state['num_queries_made'] = int(al_state['num_queries_made']) + int(num_queried_this_step)

            X_labeled_current_raw = [X_train_raw_full[i] for i in labeled_indices]
            y_labeled_current = [y_train_full[i] for i in labeled_indices]

            if len(np.unique(y_labeled_current)) < 2:
                if al_state['num_queries_made'] > 0 and len(labeled_indices) > 5:
                    raise ValueError(
                        "Not enough classes in labeled set for meaningful model training. Pool might be degenerate.")
                pass

            classifier_retrain = LogisticRegression(max_iter=1000, random_state=42)
            X_labeled_transformed = fitted_vectorizer.transform(X_labeled_current_raw)
            classifier_retrain.fit(X_labeled_transformed, y_labeled_current)

            save_model(classifier_retrain, model_path)

            X_test_transformed = fitted_vectorizer.transform(X_test_raw)
            y_pred_test = classifier_retrain.predict(X_test_transformed)
            current_accuracy = accuracy_score(y_test, y_pred_test)

            al_state['active_learning_accuracy_history'].append(
                {"queries": int(al_state['num_queries_made']), "accuracy": float(current_accuracy)})
            al_state['current_accuracy'] = float(current_accuracy)

            request.session['active_learning_state'] = al_state
            request.session.modified = True

            message = f"Queried {num_queried_this_step} samples. Total queries: {al_state['num_queries_made']}. Accuracy: {current_accuracy:.4f}"
            return HttpResponseRedirect(reverse(
                'project2:index') + f'?active_status={message}&current_accuracy={current_accuracy:.4f}&queries={al_state["num_queries_made"]}')

        except Exception as e:
            traceback.print_exc()
            return HttpResponseRedirect(reverse('project2:index') + f'?error=AL_Query_Error: {e}')

    return HttpResponseRedirect(reverse('project2:index'))


def active_learn_reset(request):
    if request.method == 'POST':
        session_keys_to_clear = [
            'X_train_raw_full', 'y_train_full', 'X_test_raw', 'y_test',
            'fitted_vectorizer_pkl_b64', 'active_learning_state'
        ]
        for key in session_keys_to_clear:
            if key in request.session:
                del request.session[key]

        if os.path.exists(AL_CURRENT_MODEL_PATH):
            os.remove(AL_CURRENT_MODEL_PATH)

        message = "Active Learning session reset."
        return HttpResponseRedirect(reverse('project2:index') + f'?active_status={message}')
    return HttpResponseRedirect(reverse('project2:index'))