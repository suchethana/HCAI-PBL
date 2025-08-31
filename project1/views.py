from django.shortcuts import render, HttpResponseRedirect
from django.urls import reverse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error
)
import warnings, traceback
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

HYPERPARAMETER_DEFS = {
    "logistic_regression": [
        {"name": "C (Inverse of regularization strength)", "param": "C", "type": "float", "default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01},
        {"name": "Solver", "param": "solver", "type": "str", "default": "liblinear", "options": ["liblinear", "lbfgs"],
         "info": "liblinear good for small datasets, lbfgs for larger."},
    ],
    "decision_tree_classifier": [
        {"name": "Max Depth", "param": "max_depth", "type": "int", "default": 5, "min": 1, "max": 20},
        {"name": "Min Samples Split", "param": "min_samples_split", "type": "int", "default": 2, "min": 2, "max": 20},
    ],
    "svc": [
        {"name": "C (Regularization parameter)", "param": "C", "type": "float", "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
        {"name": "Kernel", "param": "kernel", "type": "str", "default": "rbf", "options": ["linear", "poly", "rbf", "sigmoid"]},
    ],
    "random_forest_classifier": [
        {"name": "N Estimators (No. of trees)", "param": "n_estimators", "type": "int", "default": 100, "min": 10, "max": 500, "step": 10},
        {"name": "Max Depth", "param": "max_depth", "type": "int", "default": 10, "min": 1, "max": 30},
    ],
    "knn_classifier": [
        {"name": "N Neighbors", "param": "n_neighbors", "type": "int", "default": 5, "min": 1, "max": 20},
    ],
    "linear_regression": [
        {"name": "Fit Intercept", "param": "fit_intercept", "type": "bool", "default": True},
    ],
    "random_forest_regressor": [
        {"name": "N Estimators (No. of trees)", "param": "n_estimators", "type": "int", "default": 100, "min": 10, "max": 500, "step": 10},
        {"name": "Max Depth", "param": "max_depth", "type": "int", "default": 10, "min": 1, "max": 30},
    ],
    "knn_regressor": [
        {"name": "N Neighbors", "param": "n_neighbors", "type": "int", "default": 5, "min": 1, "max": 20},
    ],
}

def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

def smart_detect_problem_type(df: pd.DataFrame, target_col: str):
    target = df[target_col]
    nunq = target.nunique(dropna=True)
    if pd.api.types.is_numeric_dtype(target):
        is_int = pd.api.types.is_integer_dtype(target)
        if is_int and 2 <= nunq <= 5:
            return True, "Classification (Numeric Labels)"
        else:
            return False, "Regression"
    else:
        if nunq > 1:
            return True, "Classification (Categorical Labels)"
        return False, "Cannot determine problem type (single unique target value or invalid type)"

def get_manual_override(request):
    override = request.POST.get('force_problem_type') or request.GET.get('force_problem_type')
    if override in ("classification", "regression"):
        return override
    return None

def index(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('csv_file')
        if not uploaded_file:
            return render(request, 'project1/index.html', {'error': 'No file uploaded. Please select a CSV file.'})
        if not uploaded_file.name.endswith('.csv'):
            return render(request, 'project1/index.html', {'error': 'Invalid file type. Please upload a CSV file.'})
        try:
            data_io = io.StringIO(uploaded_file.read().decode('utf-8'))
            df = pd.read_csv(data_io)
            if len(df.columns) > 1 and (
                df.columns[0].lower() in ['id', 'idx', 'index'] or
                (pd.api.types.is_numeric_dtype(df.iloc[:, 0]) and
                 df.iloc[:, 0].nunique() == len(df) and
                 df.iloc[:, 0].is_monotonic_increasing)
            ):
                df = df.iloc[:, 1:]
            request.session['df_data'] = df.to_json()
            numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
            initial_context = {
                'selected_plot_type': 'scatter',
                'selected_x_feature': numerical_cols[0] if len(numerical_cols) >= 1 else (df.columns[0] if len(df.columns) else ''),
                'selected_y_feature': numerical_cols[1] if len(numerical_cols) >= 2 else '',
                'selected_group_by': '',
            }
            request.session['initial_viz_context'] = initial_context
            return HttpResponseRedirect(reverse('project1:visualize_data'))
        except pd.errors.EmptyDataError:
            return render(request, 'project1/index.html', {'error': 'The uploaded CSV file is empty.'})
        except pd.errors.ParserError:
            return render(request, 'project1/index.html', {'error': 'Could not parse CSV file. Please check its format.'})
        except Exception as e:
            return render(request, 'project1/index.html', {'error': f'An unexpected error occurred: {e}'})
    return render(request, 'project1/index.html')

def visualize_data(request):
    df_json = request.session.get('df_data')
    if not df_json:
        return HttpResponseRedirect(reverse('project1:index'))
    df = pd.read_json(df_json)

    plot_url = None
    plot_error = None
    training_error = None
    evaluation_results = None

    all_column_names = df.columns.tolist()
    numerical_column_names = df.select_dtypes(include=['number']).columns.tolist()
    categorical_column_names = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    target_column_name = df.columns[-1]

    override = get_manual_override(request)
    if override == "classification":
        is_classification_problem = True
        problem_type_display = "Classification (Manual Override)"
    elif override == "regression":
        is_classification_problem = False
        problem_type_display = "Regression (Manual Override)"
    else:
        is_classification_problem, problem_type_display = smart_detect_problem_type(df, target_column_name)

    initial_viz_context = request.session.pop('initial_viz_context', {})
    selected_plot_type = initial_viz_context.get('selected_plot_type', 'scatter')
    selected_x_feature = initial_viz_context.get('selected_x_feature',
                                                 numerical_column_names[0] if numerical_column_names else '')
    selected_y_feature = initial_viz_context.get('selected_y_feature',
                                                 numerical_column_names[1] if len(numerical_column_names) > 1 else '')
    selected_group_by = initial_viz_context.get('selected_group_by', '')

    selected_model_name = 'logistic_regression' if is_classification_problem else 'linear_regression'
    selected_test_size = '0.2'
    current_override_value = override or ("classification" if is_classification_problem else "regression")

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'generate_plot':
            selected_plot_type = request.POST.get('plot_type', 'scatter')
            selected_x_feature = request.POST.get('x_axis_feature', '')
            selected_y_feature = request.POST.get('y_axis_feature', '')
            selected_group_by = request.POST.get('group_by_feature', '')
            current_override_value = get_manual_override(request) or current_override_value

        elif action == 'train_model':
            selected_model_name = request.POST.get('model_name')
            selected_test_size = request.POST.get('test_size', '0.2')
            current_override_value = get_manual_override(request) or current_override_value

            model_hyperparameters = {}
            for param_def in HYPERPARAMETER_DEFS.get(selected_model_name, []):
                param_name = param_def['param']
                param_type = param_def['type']
                param_value_str = request.POST.get(f'hp_{param_name}')
                if param_value_str is not None:
                    try:
                        if param_type == 'int':
                            model_hyperparameters[param_name] = int(float(param_value_str))
                        elif param_type == 'float':
                            model_hyperparameters[param_name] = float(param_value_str)
                        elif param_type == 'bool':
                            model_hyperparameters[param_name] = (param_value_str.lower() == 'true')
                        else:
                            model_hyperparameters[param_name] = str(param_value_str)
                    except ValueError:
                        raise ValueError(f"Invalid value for hyperparameter '{param_def['name']}'. Expected {param_type}.")

            train_features = request.POST.getlist('train_features')
            train_target = request.POST.get('train_target_column')

            try:
                test_size = float(selected_test_size)
                if not (0.0 < test_size < 1.0):
                    raise ValueError("Test size must be between 0 and 1.")
                if not train_target or train_target not in all_column_names:
                    raise ValueError("Please select a valid target column for training.")
                if not train_features:
                    train_features = [col for col in numerical_column_names if col != train_target]
                    if not train_features:
                        raise ValueError("No numerical features found or selected for training.")
                for feature in train_features:
                    if feature not in numerical_column_names:
                        raise ValueError(f"Feature '{feature}' is not numerical and cannot be used for training.")

                X = df[train_features]
                y = df[train_target]
                combined = pd.concat([X, y], axis=1).dropna()
                if combined.empty or len(combined) < 2:
                    raise ValueError("Not enough valid data points for training after handling missing values.")
                X_cleaned = combined[train_features]
                y_cleaned = combined[train_target]

                if is_classification_problem:
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y_cleaned)
                    request.session['class_names'] = _json_safe(list(le.classes_))
                else:
                    y_encoded = y_cleaned

                X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_encoded, test_size=test_size, random_state=42)

                model = None
                if is_classification_problem:
                    if selected_model_name == 'logistic_regression':
                        model = LogisticRegression(random_state=42, **model_hyperparameters)
                    elif selected_model_name == 'decision_tree_classifier':
                        model = DecisionTreeClassifier(random_state=42, **model_hyperparameters)
                    elif selected_model_name == 'svc':
                        model = SVC(random_state=42, **model_hyperparameters)
                    elif selected_model_name == 'random_forest_classifier':
                        model = RandomForestClassifier(random_state=42, **model_hyperparameters)
                    elif selected_model_name == 'knn_classifier':
                        model = KNeighborsClassifier(**model_hyperparameters)
                else:
                    if selected_model_name == 'linear_regression':
                        model = LinearRegression(**model_hyperparameters)
                    elif selected_model_name == 'random_forest_regressor':
                        model = RandomForestRegressor(random_state=42, **model_hyperparameters)
                    elif selected_model_name == 'knn_regressor':
                        model = KNeighborsRegressor(**model_hyperparameters)

                if model is None:
                    raise ValueError("Selected model is not valid for the detected problem type.")

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                results = {
                    'Model': selected_model_name.replace('_', ' ').title(),
                    'Problem Type': problem_type_display,
                    'Test Size': f"{test_size * 100:.0f}%",
                    'Features Used': ", ".join(train_features),
                    'Target Column': train_target,
                }
                if is_classification_problem:
                    results.update({
                        'Accuracy': float(accuracy_score(y_test, y_pred)),
                        'Precision (Weighted)': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                        'Recall (Weighted)': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                        'F1 Score (Weighted)': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
                    })
                else:
                    results.update({
                        'Mean Squared Error': float(mean_squared_error(y_test, y_pred)),
                        'R2 Score': float(r2_score(y_test, y_pred)),
                        'Mean Absolute Error': float(mean_absolute_error(y_test, y_pred)),
                    })

                evaluation_results = results
                request.session['evaluation_results'] = _json_safe(evaluation_results)

            except ValueError as ve:
                training_error = str(ve)
            except Exception as e:
                training_error = f"An unexpected error occurred during training: {e}"
                traceback.print_exc()

    try:
        fig, ax = plt.subplots(figsize=(10, 7))
        if not selected_x_feature or selected_x_feature not in all_column_names:
            if numerical_column_names:
                selected_x_feature = numerical_column_names[0]
            else:
                raise ValueError("No numerical features available for plotting.")
        x_data = df[selected_x_feature]

        if selected_plot_type == 'scatter':
            if not selected_y_feature or selected_y_feature not in all_column_names:
                if len(numerical_column_names) > 1:
                    selected_y_feature = numerical_column_names[1]
                else:
                    raise ValueError("Scatter plot requires a Y-axis feature.")
            if selected_x_feature not in numerical_column_names or selected_y_feature not in numerical_column_names:
                raise ValueError("Scatter plot features must be numerical.")
            y_data = df[selected_y_feature]
            if selected_group_by and selected_group_by != "None" and selected_group_by in all_column_names:
                group_col_data = df[selected_group_by]
                unique_groups = group_col_data.unique()
                colors = plt.cm.get_cmap('tab10', len(unique_groups))
                for i, gv in enumerate(unique_groups):
                    subset = df[group_col_data == gv]
                    ax.scatter(subset[selected_x_feature], subset[selected_y_feature],
                               color=colors(i), label=str(gv), alpha=0.7)
                ax.legend(title=selected_group_by, bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout(rect=[0, 0, 0.85, 1])
            else:
                ax.scatter(x_data, y_data, alpha=0.7)
            ax.set_xlabel(selected_x_feature)
            ax.set_ylabel(selected_y_feature)
            ax.set_title(f'Scatter Plot: {selected_x_feature} vs {selected_y_feature}')

        elif selected_plot_type == 'histogram':
            if selected_x_feature not in numerical_column_names:
                raise ValueError("Histogram feature must be numerical.")
            xdn = df[selected_x_feature].dropna()
            ax.hist(xdn, bins=20, edgecolor='black', alpha=0.7)
            ax.set_xlabel(selected_x_feature)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Histogram of {selected_x_feature}')

        elif selected_plot_type == 'boxplot':
            if not selected_x_feature:
                raise ValueError("Box Plot requires a numerical feature.")
            if selected_x_feature not in numerical_column_names:
                raise ValueError("Box Plot feature must be numerical.")
            if selected_group_by and selected_group_by != "None" and selected_group_by in all_column_names:
                if pd.api.types.is_numeric_dtype(df[selected_group_by]) and df[selected_group_by].nunique() > 10:
                    raise ValueError(
                        f"Box Plot grouping by '{selected_group_by}' requires a categorical or low-cardinality numerical column."
                    )
                plot_df = df[[selected_x_feature, selected_group_by]].dropna()
                if plot_df.empty:
                    raise ValueError("No valid data points after dropping NaNs for Box Plot with grouping.")
                plot_df.boxplot(column=selected_x_feature, by=selected_group_by, ax=ax)
                ax.set_title(f'Box Plot of {selected_x_feature} by {selected_group_by}')
                plt.suptitle('')
                ax.set_xlabel(selected_group_by)
                ax.set_ylabel(selected_x_feature)
            else:
                ax.boxplot(df[selected_x_feature].dropna())
                ax.set_title(f'Box Plot of {selected_x_feature}')
                ax.set_xticks([1])
                ax.set_xticklabels([selected_x_feature])
        else:
            raise ValueError("Invalid plot type selected.")

        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_url = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
    except Exception as e:
        plot_error = f"Error generating plot: {e}"
        print(f"Plotting Error: {e}")

    df_head_html = df.head().to_html(classes='table table-bordered table-striped', index=False)
    if 'evaluation_results' in request.session:
        evaluation_results = request.session.pop('evaluation_results')

    context = {
        'plot_url': plot_url,
        'df_head': df_head_html,
        'all_column_names': all_column_names,
        'numerical_column_names': numerical_column_names,
        'categorical_column_names': categorical_column_names,
        'selected_plot_type': selected_plot_type,
        'selected_x_feature': selected_x_feature,
        'selected_y_feature': selected_y_feature,
        'selected_group_by': selected_group_by,
        'plot_error': plot_error,
        'training_error': training_error,
        'evaluation_results': evaluation_results,
        'problem_type': problem_type_display,
        'target_column_name': target_column_name,
        'is_classification_problem': is_classification_problem,
        'model_options': [
            {"name": "Logistic Regression", "value": "logistic_regression", "type": "classification"},
            {"name": "Decision Tree Classifier", "value": "decision_tree_classifier", "type": "classification"},
            {"name": "Support Vector Classifier (SVC)", "value": "svc", "type": "classification"},
            {"name": "Random Forest Classifier", "value": "random_forest_classifier", "type": "classification"},
            {"name": "K-Nearest Neighbors Classifier", "value": "knn_classifier", "type": "classification"},
            {"name": "Linear Regression", "value": "linear_regression", "type": "regression"},
            {"name": "Random Forest Regressor", "value": "random_forest_regressor", "type": "regression"},
            {"name": "K-Nearest Neighbors Regressor", "value": "knn_regressor", "type": "regression"},
        ],
        'hyperparameter_defs': HYPERPARAMETER_DEFS,
        'selected_model_name': selected_model_name,
        'selected_test_size': selected_test_size,
        'current_problem_override': current_override_value,
    }
    return render(request, 'project1/visualization.html', context)
