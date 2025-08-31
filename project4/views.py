import matplotlib
matplotlib.use('Agg')

from django.shortcuts import render, HttpResponse, HttpResponseRedirect, redirect, get_object_or_404
from django.urls import reverse
from django.conf import settings
from django.utils.html import format_html

import os
import pandas as pd
import numpy as np
import random
import re
import joblib

from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD

from .models import UserStudyData

MOVIELENS_MOVIES_PATH = os.path.join(settings.MOVIELENS_DATA_PATH, 'movies.csv')
MOVIELENS_RATINGS_PATH = os.path.join(settings.MOVIELENS_DATA_PATH, 'ratings.csv')

MODEL_DIR = os.path.join(settings.BASE_DIR, 'model_data')
os.makedirs(MODEL_DIR, exist_ok=True)
ITEM_FACTORS_PATH = os.path.join(MODEL_DIR, 'item_factors_V.joblib')
USER_ID_MAP_PATH = os.path.join(MODEL_DIR, 'user_id_map.joblib')
MOVIE_ID_MAP_PATH = os.path.join(MODEL_DIR, 'movie_id_map.joblib')

_all_movies_df = None
_all_ratings_df = None
_V_matrix = None
_user_id_map = None
_movie_id_map = None
_K_latent_dimensions = 20
_lambda_reg = 0.1

MIN_RATING_SCALE = 0.5
MAX_RATING_SCALE = 5.0

TOTAL_MOVIES = 10

def format_movie_title(title):
    match = re.match(r'^(.*), (The|A|An) \((\d{4})\)$', title)
    if match:
        main_title, article, year = match.groups()
        return f"{article} {main_title} ({year})"
    match2 = re.match(r'^(.*), (The|A|An) \(([^)]+)\) \((\d{4})\)$', title)
    if match2:
        main_title, article, additional_info, year = match2.groups()
        return f"{article} {main_title} ({additional_info}) ({year})"
    return title

def load_movies_data():
    global _all_movies_df
    if _all_movies_df is None:
        try:
            _all_movies_df = pd.read_csv(MOVIELENS_MOVIES_PATH)
            _all_movies_df['formatted_title'] = _all_movies_df['title'].apply(format_movie_title)
        except FileNotFoundError:
            _all_movies_df = pd.DataFrame(columns=['movieId', 'title', 'genres', 'formatted_title'])
    return _all_movies_df

def load_ratings_data():
    global _all_ratings_df
    if _all_ratings_df is None:
        try:
            _all_ratings_df = pd.read_csv(MOVIELENS_RATINGS_PATH)
        except FileNotFoundError:
            _all_ratings_df = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])
    return _all_ratings_df

def train_matrix_factorization_model():
    global _V_matrix, _user_id_map, _movie_id_map, _all_movies_df, _all_ratings_df

    if os.path.exists(ITEM_FACTORS_PATH) and os.path.exists(USER_ID_MAP_PATH) and os.path.exists(MOVIE_ID_MAP_PATH):
        try:
            _V_matrix = joblib.load(ITEM_FACTORS_PATH)
            _user_id_map = joblib.load(USER_ID_MAP_PATH)
            _movie_id_map = joblib.load(MOVIE_ID_MAP_PATH)
            return
        except Exception:
            pass

    movies_df = load_movies_data()
    ratings_df = load_ratings_data()

    if movies_df.empty or ratings_df.empty:
        return

    unique_users = ratings_df['userId'].unique()
    unique_movies = movies_df['movieId'].unique()

    _user_id_map = {original_id: i for i, original_id in enumerate(unique_users)}
    _movie_id_map = {original_id: i for i, original_id in enumerate(unique_movies)}

    num_users = len(unique_users)
    num_movies = len(unique_movies)

    R = np.zeros((num_users, num_movies))
    for _, row in ratings_df.iterrows():
        user_idx = _user_id_map.get(row['userId'])
        movie_idx = _movie_id_map.get(row['movieId'])
        if user_idx is not None and movie_idx is not None:
            R[user_idx, movie_idx] = row['rating']

    try:
        svd = TruncatedSVD(n_components=_K_latent_dimensions, random_state=42)
        _ = svd.fit_transform(R)
        _V_matrix = svd.components_.T
        joblib.dump(_V_matrix, ITEM_FACTORS_PATH)
        joblib.dump(_user_id_map, USER_ID_MAP_PATH)
        joblib.dump(_movie_id_map, MOVIE_ID_MAP_PATH)
    except Exception:
        _V_matrix = None

def objective_function_user_vector(user_vector, known_ratings, item_factors, lambda_reg):
    error = 0
    for movie_id, rating in known_ratings.items():
        movie_idx = _movie_id_map.get(movie_id)
        if movie_idx is not None and movie_idx < item_factors.shape[0]:
            V_j = item_factors[movie_idx, :]
            predicted_rating = np.dot(user_vector, V_j)
            error += (rating - predicted_rating) ** 2
    regularization_term = lambda_reg * np.sum(user_vector ** 2)
    return error + regularization_term

def learn_new_user_latent_factor(user_ratings, item_factors, lambda_reg):
    if item_factors is None:
        return None
    known_ratings_dict = {item['movieId']: item['rating'] for item in user_ratings}
    initial_user_vector = np.random.rand(_K_latent_dimensions) * 0.1
    result = minimize(
        objective_function_user_vector,
        initial_user_vector,
        args=(known_ratings_dict, item_factors, lambda_reg),
        method='L-BFGS-B'
    )
    if result.success:
        return result.x
    return None

def scale_prediction(prediction, min_val, max_val, target_min=0.5, target_max=5.0):
    if max_val == min_val:
        return target_min + (target_max - target_min) / 2
    scaled = ((prediction - min_val) / (max_val - min_val)) * (target_max - target_min) + target_min
    return max(target_min, min(target_max, scaled))

def predict_recommendations(user_latent_factor, movies_df, item_factors, user_rated_movie_ids, num_recommendations=10):
    if user_latent_factor is None or item_factors is None:
        return []
    all_movie_ids = movies_df['movieId'].tolist()
    predictions = {}
    for movie_id in all_movie_ids:
        if movie_id in user_rated_movie_ids:
            continue
        movie_idx = _movie_id_map.get(movie_id)
        if movie_idx is not None and movie_idx < item_factors.shape[0]:
            V_j = item_factors[movie_idx, :]
            predicted_rating = np.dot(user_latent_factor, V_j)
            predictions[movie_id] = predicted_rating
    if not predictions:
        return []
    prediction_values = list(predictions.values())
    min_pred = min(prediction_values)
    max_pred = max(prediction_values)
    scaled_predictions = {mid: scale_prediction(pred, min_pred, max_pred, MIN_RATING_SCALE, MAX_RATING_SCALE)
                          for mid, pred in predictions.items()}
    sorted_predictions = sorted(scaled_predictions.items(), key=lambda item: item[1], reverse=True)
    top = []
    for movie_id, predicted_rating in sorted_predictions:
        if len(top) >= num_recommendations:
            break
        movie_info = movies_df[movies_df['movieId'] == movie_id].iloc[0]
        top.append({
            'movieId': movie_info['movieId'],
            'title': movie_info.get('formatted_title', movie_info['title']),
            'genres': movie_info['genres'],
            'predicted_rating': round(predicted_rating, 2)
        })
    return top

def make_pre_impact_hint(current_movie_row, movies_df, user_ratings_list):
    """
    Build the pre-submit hint under the input, based on whether the movie’s
    genres are new or familiar to the user. Always reminds to 'rate your interest'.
    """
    display_title = current_movie_row.get('formatted_title', current_movie_row['title'])
    genres_raw = current_movie_row.get('genres', '')
    current_movie_genres = set(genres_raw.split('|')) if genres_raw else set()

    rated_genres = set()
    if user_ratings_list:
        rated_movie_ids = [item['movieId'] for item in user_ratings_list]
        prev_rated = movies_df[movies_df['movieId'].isin(rated_movie_ids)]
        for _, row in prev_rated.iterrows():
            if row['genres']:
                rated_genres.update(row['genres'].split('|'))

    new_genres = current_movie_genres - rated_genres

    genres_text = ", ".join(sorted(current_movie_genres)) if current_movie_genres else "these genres"
    if new_genres:
        return format_html(
            'Your rating for this will give us your <b>first signal</b> about <b>{}</b>, '
            'so we can <b>explore</b> more like these. '
            'If you haven’t watched it, <b>rate your interest</b>.',
            genres_text
        )
    else:
        return format_html(
            'Your rating will help us <b>refine</b> your taste for <b>{}</b> and adjust similar titles. '
            'If you haven’t watched it, <b>rate your interest</b>.',
            genres_text
        )

def index(request):
    return render(request, 'project4/index.html')

def start_study(request):
    if request.method == 'POST':
        group = 'A' if random.random() < 0.5 else 'B'
        request.session['group'] = group
        request.session['current_user_ratings'] = []
        request.session['current_movie_index'] = 0

        movies_df = load_movies_data()
        all_movie_ids = movies_df['movieId'].tolist()
        num_movies_to_ask = TOTAL_MOVIES
        random_movie_ids = random.sample(all_movie_ids, min(len(all_movie_ids), num_movies_to_ask))
        request.session['movies_asked_ids'] = random_movie_ids

        request.session.modified = True
        return HttpResponseRedirect(reverse('project4:rate_movie'))

    return render(request, 'project4/consent.html')

def rate_movie(request):
    """
    GET: show rating form + dynamic pre-impact genre-aware hint.
    POST: store rating and advance. (No interstitial by default.)
    """
    movies_df = load_movies_data()

    if movies_df.empty:
        return render(request, 'project4/cold_start_study.html', {'error': 'Movie data not found.'})

    if _V_matrix is None or _movie_id_map is None:
        return render(request, 'project4/cold_start_study.html', {'error': 'Recommendation system not initialized.'})

    if request.method == 'POST':
        movie_id = request.POST.get('movie_id')
        rating = request.POST.get('rating')
        user_ratings = request.session.get('current_user_ratings', [])
        current_movie_index = request.session.get('current_movie_index', 0)

        if movie_id and rating:
            user_ratings.append({'movieId': int(movie_id), 'rating': float(rating)})
            request.session['current_user_ratings'] = user_ratings
            request.session['current_movie_index'] = current_movie_index + 1
            request.session.modified = True

        movies_asked_ids = request.session.get('movies_asked_ids', [])
        if request.session['current_movie_index'] >= len(movies_asked_ids):
            session_id = request.session.session_key
            group = request.session.get('group', 'B')
            data_entry, created = UserStudyData.objects.get_or_create(
                session_id=session_id,
                defaults={'group': group, 'ratings_data': user_ratings}
            )
            if not created:
                data_entry.group = group
                data_entry.ratings_data = user_ratings
                data_entry.save()
            return HttpResponseRedirect(reverse('project4:recommendations'))

        return HttpResponseRedirect(reverse('project4:rate_movie'))

    current_movie_index_in_session = request.session.get('current_movie_index', 0)
    movies_asked_ids_in_session = request.session.get('movies_asked_ids', [])

    if current_movie_index_in_session < len(movies_asked_ids_in_session):
        current_movie_id = movies_asked_ids_in_session[current_movie_index_in_session]
        current_movie_series = movies_df[movies_df['movieId'] == current_movie_id]

        if not current_movie_series.empty:
            current_movie = current_movie_series.iloc[0]
            display_title = current_movie.get('formatted_title', current_movie['title'])

            user_ratings_data = request.session.get('current_user_ratings', [])
            pre_hint = make_pre_impact_hint(current_movie, movies_df, user_ratings_data)

            context = {
                'current_movie': current_movie.to_dict(),
                'current_movie_index': current_movie_index_in_session,
                'total_movies_to_ask': len(movies_asked_ids_in_session),
                'current_movie_display_number': current_movie_index_in_session + 1,
                'pre_impact_hint': pre_hint,
                'impact_message': pre_hint,
                'display_title': display_title,
                'is_last_movie': current_movie_index_in_session + 1 >= len(movies_asked_ids_in_session),
            }
            return render(request, 'project4/cold_start_study.html', context)
        else:
            request.session['movies_asked_ids'] = []
            request.session.modified = True
            return render(
                request,
                'project4/cold_start_study.html',
                {'error': f'Movie data inconsistency for ID {current_movie_id}.'}
            )
    else:
        return HttpResponseRedirect(reverse('project4:recommendations'))

def show_recommendations(request):
    movies_df = load_movies_data()
    user_ratings = request.session.get('current_user_ratings', [])
    user_rated_movie_ids = {item['movieId'] for item in user_ratings}

    if not user_ratings:
        return render(request, 'project4/recommendations.html',
                      {'message': 'No ratings provided. Please go back to start the study.'})

    if _V_matrix is None:
        return render(request, 'project4/recommendations.html',
                      {'error': 'Recommendation model is not fully prepared.'})

    user_latent_factor = learn_new_user_latent_factor(user_ratings, _V_matrix, _lambda_reg)
    if user_latent_factor is None:
        return render(request, 'project4/recommendations.html',
                      {'error': 'Could not generate recommendations for you.'})

    recommendations = predict_recommendations(
        user_latent_factor, movies_df, _V_matrix, user_rated_movie_ids, num_recommendations=10
    )

    session_id = request.session.session_key
    group = request.session.get('group', 'B')
    data_entry, created = UserStudyData.objects.get_or_create(
        session_id=session_id,
        defaults={'group': group, 'ratings_data': user_ratings}
    )
    if not created:
        data_entry.group = group
        data_entry.ratings_data = user_ratings
        data_entry.save()

    context = {
        'recommendations': recommendations,
        'user_ratings_count': len(user_ratings),
        'message': 'Here are your personalized movie recommendations!' if recommendations
                   else 'No recommendations could be generated at this time.'
    }
    return render(request, 'project4/recommendations.html', context)

def questionnaire(request):
    if request.method == 'POST':
        session_id = request.session.session_key
        try:
            data_entry = UserStudyData.objects.get(session_id=session_id)
            data_entry.perceived_accuracy = int(request.POST.get('perceived_accuracy'))
            data_entry.trust = int(request.POST.get('trust'))
            data_entry.qualitative_feedback = request.POST.get('qualitative_feedback', '')
            data_entry.save()
        except UserStudyData.DoesNotExist:
            pass
        return HttpResponseRedirect(reverse('project4:debrief'))

    return render(request, 'project4/questionnaire.html')

def debrief(request):
    group = request.session.get('group', 'B')
    for key in ['group', 'current_user_ratings', 'movies_asked_ids', 'current_movie_index']:
        if key in request.session:
            del request.session[key]
    request.session.modified = True
    return render(request, 'project4/debrief.html', {'group': group})
