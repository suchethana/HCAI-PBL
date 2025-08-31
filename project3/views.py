import matplotlib
import os

matplotlib.use('Agg')

from django.shortcuts import render, HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.conf import settings

import palmerpenguins
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import io
import base64

import matplotlib.pyplot as plt

try:
    from gosdt.standard import GOSDT
except ImportError:
    try:
        from gosdt import GOSDT
    except ImportError:
        GOSDT = None

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

MODEL_DIR_P3 = os.path.join(settings.BASE_DIR, 'models_project3')
os.makedirs(MODEL_DIR_P3, exist_ok=True)


def index(request):
    dt_results = request.session.pop('dt_results', None)
    sparse_dt_results = request.session.pop('sparse_dt_results', None)
    sparse_lr_results = request.session.pop('sparse_lr_results', None)
    counterfactual_results = request.session.pop('counterfactual_results', None)

    context = {
        'error': request.GET.get('error'),
        'dt_results': dt_results,
        'sparse_dt_results': sparse_dt_results,
        'sparse_lr_results': sparse_lr_results,
        'counterfactual_results': counterfactual_results
    }

    return render(request, 'project3/index.html', context)


def train_decision_tree_baseline(request):
    if request.method == 'POST':
        try:
            penguins = palmerpenguins.load_penguins()
            penguins = penguins.dropna()

            features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
            target = 'species'

            X = penguins[features]
            y_raw = penguins[target]

            le = LabelEncoder()
            y = le.fit_transform(y_raw)
            class_names = le.classes_.tolist()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

            dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=4)
            dt_classifier.fit(X_train, y_train)

            y_pred = dt_classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            num_leaves = dt_classifier.get_n_leaves()

            tree_rules = export_text(dt_classifier, feature_names=features)

            plt.figure(figsize=(20, 10))
            plot_tree(dt_classifier,
                      feature_names=features,
                      class_names=[str(c) for c in class_names],
                      filled=True,
                      rounded=True,
                      fontsize=10
                      )
            plt.title("Decision Tree Structure")

            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()

            tree_graph_url = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

            request.session['dt_results'] = {
                'accuracy': float(accuracy),
                'num_leaves': int(num_leaves),
                'tree_structure': tree_rules,
                'tree_graph_url': tree_graph_url,
            }
            request.session.modified = True

            return HttpResponseRedirect(reverse('project3:index'))

        except Exception as e:
            import traceback
            traceback.print_exc()
            return HttpResponseRedirect(reverse('project3:index') + f'?error={e}')
    return HttpResponseRedirect(reverse('project3:index'))


def train_sparse_dt_model(request):
    if request.method == 'POST':
        try:
            lambda_value = float(request.POST.get('lambda_value'))

            max_depth = 5 - round(lambda_value * 10)
            if max_depth < 1:
                max_depth = 1

            penguins = palmerpenguins.load_penguins()
            penguins = penguins.dropna()
            features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
            target = 'species'
            X = penguins[features]
            y_raw = penguins[target]

            le = LabelEncoder()
            y = le.fit_transform(y_raw)
            class_names = le.classes_.tolist()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

            sparse_dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
            sparse_dt_classifier.fit(X_train, y_train)

            y_pred = sparse_dt_classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            num_leaves = sparse_dt_classifier.get_n_leaves()

            tree_rules = export_text(sparse_dt_classifier, feature_names=features)

            plt.figure(figsize=(20, 10))
            plot_tree(sparse_dt_classifier,
                      feature_names=features,
                      class_names=[str(c) for c in class_names],
                      filled=True,
                      rounded=True,
                      fontsize=10
                      )
            plt.title(f"Sparse Decision Tree (λ={lambda_value}, Max Depth={max_depth})")

            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()

            tree_graph_url = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

            request.session.pop('sparse_dt_results', None)
            request.session['sparse_dt_results'] = {
                'accuracy': float(accuracy),
                'num_leaves': int(num_leaves),
                'tree_structure': tree_rules,
                'tree_graph_url': tree_graph_url,
                'lambda_value': float(lambda_value),
                'max_depth': int(max_depth),
            }
            request.session.modified = True

            return HttpResponseRedirect(reverse('project3:index') + '#sdt')

        except Exception as e:
            import traceback
            traceback.print_exc()
            return HttpResponseRedirect(reverse('project3:index') + f'?error={e}')
    return HttpResponseRedirect(reverse('project3:index'))


def train_sparse_lr_model(request):
    if request.method == 'POST':
        try:
            lambda_value = float(request.POST.get('lambda_value'))

            C = 1.0 / lambda_value

            penguins = palmerpenguins.load_penguins()
            penguins = penguins.dropna()
            features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
            target = 'species'
            X = penguins[features]
            y_raw = penguins[target]

            le = LabelEncoder()
            y = le.fit_transform(y_raw)
            class_names = le.classes_.tolist()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

            sparse_lr_classifier = LogisticRegression(
                random_state=42,
                penalty='l1',
                C=C,
                solver='liblinear',
                multi_class='ovr'
            )
            sparse_lr_classifier.fit(X_train, y_train)

            y_pred = sparse_lr_classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            coef_arr = sparse_lr_classifier.coef_
            num_features_used = int(np.count_nonzero(coef_arr))

            coefficients = coef_arr.tolist()

            feature_names = features

            ZERO_TOL = 1e-6
            is_zero = np.isclose(coef_arr, 0.0, atol=ZERO_TOL)

            removed_mask = is_zero.all(axis=0)
            removed_features = [feature_names[i] for i, m in enumerate(removed_mask) if m]

            partial_zero_mask = is_zero.any(axis=0)
            partially_pruned_features = [
                feature_names[i] for i in range(len(feature_names))
                if partial_zero_mask[i] and not removed_mask[i]
            ]

            coefficients_named = [
                {'feature': feature_names[i], 'coefs': coef_arr[:, i].tolist()}
                for i in range(len(feature_names))
            ]

            request.session['sparse_lr_results'] = {
                'accuracy': float(accuracy),
                'num_features_used': num_features_used,
                'coefficients': coefficients,
                'coefficients_named': coefficients_named,
                'removed_features': removed_features,
                'partially_pruned_features': partially_pruned_features,
                'feature_names': feature_names,
                'class_names': class_names,
                'lambda_value': float(lambda_value),
                'C_param': float(C),
            }
            request.session.modified = True

            return HttpResponseRedirect(reverse('project3:index') + '#lr')

        except Exception as e:
            import traceback
            traceback.print_exc()
            return HttpResponseRedirect(reverse('project3:index') + f'?error={e}')
    return HttpResponseRedirect(reverse('project3:index'))


def generate_counterfactuals(request):
    if request.method == 'POST':
        try:
            example_index = int(request.POST.get('example_index'))
            target_label = int(request.POST.get('target_label'))

            penguins = palmerpenguins.load_penguins()
            penguins = penguins.dropna()
            features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
            X = penguins[features]
            y_raw = penguins['species']

            le = LabelEncoder()
            y = le.fit_transform(y_raw)
            class_names = le.classes_.tolist()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )

            # model used to evaluate counterfactuals
            C = 1.0
            clf = LogisticRegression(
                random_state=42, penalty='l1', C=C, solver='liblinear', multi_class='ovr'
            )
            clf.fit(X_train, y_train)

            # safe index wrap
            n_test = len(X_test)
            example_index = example_index % max(1, n_test)

            example_x = X_test.iloc[[example_index]].to_numpy()
            original_pred = clf.predict(example_x)[0]
            original_label = class_names[original_pred]

            # MAD-weighted L1 distance
            X_ref = X_train.to_numpy()
            mad = np.median(np.abs(X_ref - np.median(X_ref, axis=0)), axis=0)
            mad = np.where(mad == 0, 1.0, mad)

            def mad_weighted_l1(a, b):
                return float(np.sum(np.abs(a - b) / mad))

            # random search with progressive noise scales
            k = 3
            counterfactuals = []
            rng = np.random.default_rng(42)

            for noise_std in (0.3, 0.6, 1.0, 1.5, 2.0):
                attempts = 0
                max_attempts = 6000
                while attempts < max_attempts and len(counterfactuals) < 5 * k:
                    noise = rng.normal(0.0, noise_std, size=example_x.shape)
                    candidate = example_x + noise
                    if clf.predict(candidate.reshape(1, -1))[0] == target_label:
                        d = mad_weighted_l1(example_x, candidate)
                        counterfactuals.append({'x': candidate, 'distance': d})
                    attempts += 1
                if len(counterfactuals) >= k:
                    break

            # fallback: nearest predicted-target point from training set
            if not counterfactuals:
                X_train_np = X_train.to_numpy()
                preds_train = clf.predict(X_train_np)
                mask = preds_train == target_label
                if not np.any(mask):
                    mask = y_train == target_label
                candidates = X_train_np[mask]
                if candidates.size == 0:
                    raise Exception("No counterfactuals found for the given example and target label.")
                dists = [mad_weighted_l1(example_x, c.reshape(1, -1)) for c in candidates]
                order = np.argsort(dists)[:k]
                for idx in order:
                    counterfactuals.append({'x': candidates[idx].reshape(1, -1), 'distance': float(dists[idx])})

            # sort and keep top-k; also attach verify chips (predicted class + prob)
            counterfactuals.sort(key=lambda cf: cf['distance'])
            best = counterfactuals[:k]

            cf_out = []
            for cf in best:
                probs = clf.predict_proba(np.array(cf['x']).reshape(1, -1))[0]
                pred_idx = int(np.argmax(probs))
                pred_label = class_names[pred_idx]
                pred_prob = float(probs[pred_idx])
                cf_out.append({
                    'x': np.array(cf['x']).tolist(),
                    'distance': float(cf['distance']),
                    'predicted_label': pred_label,
                    'predicted_prob': pred_prob
                })

            request.session['counterfactual_results'] = {
                'original_x_index': int(example_index),
                'original_x': example_x.tolist(),
                'original_label': original_label,
                'target_label': class_names[target_label],
                'feature_names': features,        # needed for diffs
                'counterfactuals': cf_out,        # now includes verify chip info
            }
            request.session.modified = True

            return HttpResponseRedirect(reverse('project3:index') + '#counterfactual_results')

        except Exception as e:
            import traceback
            traceback.print_exc()
            return HttpResponseRedirect(reverse('project3:index') + f'?error={e}')
    return HttpResponseRedirect(reverse('project3:index'))