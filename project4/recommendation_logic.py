import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from scipy.optimize import minimize
import os


def load_movielens_data(data_path):
    try:
        ratings_df = pd.read_csv(os.path.join(data_path, 'ratings.csv'))
        movies_df = pd.read_csv(os.path.join(data_path, 'movies.csv'))
        return ratings_df, movies_df
    except FileNotFoundError:
        print(f"Error: Make sure your data files are in {data_path}")
        return None, None
    except Exception as e:
        print(f"An error occurred loading MovieLens data: {e}")
        return None, None


def train_item_embeddings(user_item_matrix, latent_features=50, reg_param=0.1, n_iterations=100):
    model = NMF(n_components=latent_features, init='random', random_state=42, max_iter=n_iterations, solver='cd',
                alpha_W=reg_param, alpha_H=reg_param)
    model.fit(user_item_matrix)
    V_matrix_transposed = model.components_
    V_matrix = V_matrix_transposed
    movie_id_to_col_idx = {movieId: idx for idx, movieId in enumerate(user_item_matrix.columns)}

    print(f"Shape of V_matrix (latent_features x num_movies): {V_matrix.shape}")
    return V_matrix, movie_id_to_col_idx


def learn_new_user_representation(user_ratings, V_matrix, movie_id_to_col_idx, K_dimension, lambda_reg):
    initial_U_i = np.random.rand(K_dimension)

    rated_movie_ids = list(user_ratings.keys())
    known_rated_movie_ids = [mid for mid in rated_movie_ids if mid in movie_id_to_col_idx]
    rated_col_indices = [movie_id_to_col_idx[mid] for mid in known_rated_movie_ids]

    if not rated_col_indices:
        return initial_U_i

    V_j_rated = V_matrix[:, rated_col_indices]
    R_ij_rated = np.array([user_ratings[mid] for mid in known_rated_movie_ids])

    def objective_function(U_i_current):
        predictions = np.dot(U_i_current.T, V_j_rated)
        loss = np.sum((R_ij_rated - predictions) ** 2)
        regularization = lambda_reg * np.sum(U_i_current ** 2)
        return loss + regularization

    result = minimize(objective_function, initial_U_i, method='L-BFGS-B')
    new_U_i = result.x
    return new_U_i


def get_unrated_movies(user_rated_movie_ids, all_movie_ids):
    return [mid for mid in all_movie_ids if mid not in user_rated_movie_ids]


def select_next_movie(user_U_i, V_matrix, unrated_movie_ids, movie_id_to_col_idx, movies_df):
    if not unrated_movie_ids:
        return None, "No more unrated movies."

    predictions = {}
    for movie_id in unrated_movie_ids:
        if movie_id in movie_id_to_col_idx:
            movie_col_idx = movie_id_to_col_idx[movie_id]
            V_j = V_matrix[:, movie_col_idx]
            predicted_rating = np.dot(user_U_i.T, V_j)
            predictions[movie_id] = predicted_rating

    if not predictions:
        return None, "No suitable movie found for prediction."

    sorted_predictions = sorted(predictions.items(), key=lambda item: item[1], reverse=True)
    next_movie_id = sorted_predictions[0][0]

    movie_title_series = movies_df[movies_df['movieId'] == next_movie_id]['title']
    next_movie_title = movie_title_series.iloc[
        0] if not movie_title_series.empty else f"Movie ID {next_movie_id} (Title Not Found)"

    return next_movie_id, next_movie_title


def calculate_influence_information(user_U_i, V_matrix, current_movie_id, movie_id_to_col_idx, movies_df, lambda_reg,
                                    K_dimension):
    if current_movie_id not in movie_id_to_col_idx:
        return {"error": f"Current movie ID {current_movie_id} not known by the model."}

    influence_data = {}

    current_movie_col_idx = movie_id_to_col_idx[current_movie_id]
    V_current_movie = V_matrix[:, current_movie_col_idx]
    predicted_rating_before = np.dot(user_U_i.T, V_current_movie)
    influence_data['predicted_rating_for_this_movie'] = round(predicted_rating_before, 2)

    example_popular_movies = [296, 356, 318, 50, 2571]

    simulated_impacts = {}
    for hypothetical_rating in [0.5, 5.0]:
        simulated_user_ratings_for_this_movie = {current_movie_id: hypothetical_rating}

        simulated_U_i = learn_new_user_representation(simulated_user_ratings_for_this_movie, V_matrix,
                                                      movie_id_to_col_idx, K_dimension, lambda_reg)

        simulated_predictions = {}
        for pop_movie_id in example_popular_movies:
            if pop_movie_id in movie_id_to_col_idx:
                pop_movie_col_idx = movie_id_to_col_idx[pop_movie_id]
                V_pop_movie = V_matrix[:, pop_movie_col_idx]
                sim_pred = np.dot(simulated_U_i.T, V_pop_movie)

                pop_movie_title_series = movies_df[movies_df['movieId'] == pop_movie_id]['title']
                pop_movie_title = pop_movie_title_series.iloc[
                    0] if not pop_movie_title_series.empty else f"Movie ID {pop_movie_id} (Title Not Found)"

                simulated_predictions[pop_movie_title] = round(sim_pred, 2)
        simulated_impacts[f'If rated {hypothetical_rating}'] = simulated_predictions

    influence_data['simulated_impact_on_other_movies'] = simulated_impacts

    return influence_data