 # 📊 Human Centric Artificial Intelligence- PBL

### This repository contains 5 projects built as part of the Human-Centric AI

### course.

### Projects:

- **Project 1: Automated Machine Learning**
- **Project 2: Active Learning for Text Classification**
- **Project 3: Explainability**
- **Project 4: Influence of Future Predictions in Recommender Systems**
- **Project 5: Reinforcement Learning with Human Feedback**

## 🛠️ Tech Stack

| **Layer**             | **Technologies Used** |
|----------------------|----------------------|
| **Programming Language**    | Python 3.10+ |
| **Web Framework**    | Django (for Project 1 – Automated ML) |
| **Data Handling**    | pandas, NumPy |
| **Visualization**    | Matplotlib, Seaborn (for plots), SHAP/LIME (for explainability) |
| **Machine Learning** | scikit-learn (classification, regression, active learning loop), Surprise/implicit (matrix factorization for recommendation), custom RL agent code |
| **Model Evaluation** | scikit-learn metrics (accuracy, precision, recall, F1), confusion matrix, ROC curves |
| **Backend Utilities**| SQLite (default Django DB), joblib (model persistence) |
| **Frontend**         | HTML, CSS (custom responsive design), Django templates |
| **Environment**      | Virtualenv / venv for environment management |

---
## 🛠️ Setup & Installation (Run This First)

### 1. Clone the repository
```
- git clone https://github.com/suchethana/HCAI-PBL
- cd HCAI-PBL-main
```

### 2. Create & activate a virtual environment (recommended)

```
- python -m venv .venv
- source .venv/bin/activate # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Run migrations (important!)

```
- python manage.py makemigrations
- python manage.py migrate
```

### 5. Start the server
```
- python manage.py runserver
```

### 6. Open the app in browser

```
- Go to [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
```

---

# 📊 Project 1 – Dataset Visualization & Model

## Training

This project is a **Django web application** that allows users to:

- Upload a CSV dataset.
- Explore and visualize data (scatter, histogram, box plots).
- Detect problem type (classification/regression) automatically.
- Train machine learning models (e.g., Logistic Regression) with custom
    hyperparameters.
- View model evaluation metrics (Accuracy, Precision, Recall, F1-Score).

## 🚀 Features

- **Dataset Upload** – Upload any CSV file.
- **Interactive Visualization** – Choose plot type, features, and group by categories.
- **ML Model Training** – Select model, solver, test split, and features.
- **Metrics Display** – See Accuracy, Precision, Recall, and F1 in a modal.
- **Navigation** – Buttons to upload a new dataset or return to home page.

## 🛠️ Tech Stack

- **Backend:** Django (Python)
- **Frontend:** HTML, CSS (custom + Django templating)
- **Data/ML:** pandas, matplotlib, scikit-learn
- **Styling:** Custom responsive CSS with animated elements


## 📊 Example Workflow

1. Upload the Iris dataset.
2. Visualize sepal.length vs sepal.width as a scatter plot.
3. Train a **Logistic Regression** model (test size = 0.2).
4. View perfect classification metrics in the results modal.
5. Upload another dataset or return to home page.

# ⚠️ Notes for Reviewers

 Before running the project:

Please execute to initialize the database tables.

- python manage.py makemigrations
- python manage.py migrate

Without this, you may get a “no such table” error.


---
# 📚 Project 2 – Active Learning for Text

## Classification

This project is a **Django web application** that demonstrates how active learning can be
used to train a text classifier with fewer labeled samples.

It uses the IMDB 50k movie reviews dataset and compares a baseline supervised classifier
with active learning approaches.

## 🚀 Features

- **Baseline Classifier Training** – Trains a TF-IDF + Logistic Regression model on the
    full labeled dataset.
- **Pretrained Model Option** – Quickly load a pretrained TF-IDF vectorizer and
    classifier.
- **Pool-Based Active Learning** – Initialize a labeled/unlabeled pool and iteratively
    query new samples to label.
- **Query Strategies** – Supports:
     Least Confidence Sampling
     Random Sampling
     Uncertainty + Random Mixture
- **Accuracy Tracking** – Displays a live accuracy vs. number-of-queries chart after
    each iteration.
- **Reset & Re-run** – Easily reset the state to try different strategies.

## 🧠 How It Works

1. **Text Representation** – Uses TfidfVectorizer to convert raw reviews into feature
    vectors.


2. **Classifier** – A LogisticRegression model predicts sentiment (positive/negative).
3. **Active Learning Loop** –
    a. Start with a small set of labeled samples.
    b. Pick unlabeled samples based on the selected query strategy.
    c. Reveal their labels, retrain, and measure accuracy.
    d. Stop when:
       i. The unlabeled pool is empty
ii. 100 queries are made
iii. 95 % test accuracy is reached

## 🛠 Tech Stack

- **Backend:** Django (Python)
- **Frontend:** HTML + CSS (Django templating, custom styles)
- **ML & Data:** scikit-learn (TF-IDF, Logistic Regression)
- **Charts:** Chart.js for interactive accuracy progress plots
- **Styling:** Responsive CSS, custom UI cards, progress messages

## 🖼 Example Workflow

1. Launch baseline training and view test accuracy (target score).
2. Initialize Active Learning – creates labeled/unlabeled pools.
3. Choose a query strategy and click “Query Next Sample.”
4. Watch accuracy improve on the live chart.
5. Reset to try a different strategy.


---
# 🌳 Project 3 – Explainability

This project is a **Django web application** that allows users to:

- Train **Decision Trees** with configurable sparsity (via λ) and visualize their structure.
- Compare model complexity (number of leaves) and accuracy across different λ
    values.
- Train **Sparse Logistic Regression** models with true coefficient sparsity (L1 penalty)
    and display which features were pruned.
- Generate **Counterfactual Explanations** for individual examples, showing what
    feature changes lead to a different predicted class.
- View results side by side, with collapsible comparison cards and feature-level
    coefficient tables.

## 🚀 Features

- **Sparse Decision Tree Training** – Train a decision tree with adjustable regularization
    (λ) and visualize its graphical structure.
- **Comparison History** – Preserve and compare multiple runs (λ=0.1 vs λ=0.3, etc.)
    with collapsible side-by-side cards.
- **Sparse Logistic Regression** – Train with penalty='l1' and solver='liblinear' or saga to
    enforce sparsity. Show removed features and coefficients per class.
- **Counterfactual Explanations** – Generate top-k counterfactuals for a chosen
    sample and target class, with feature-level diffs and predicted probabilities.
- **UI Enhancements** – Sticky action bar for λ slider & training buttons, clickable
    anchors to deep-link sections, and ability to clear comparison history.

## 🛠 Tech Stack

- **Backend:** Django (Python)


- **Frontend:** HTML, CSS (custom + Django templating), JavaScript for dynamic UI
    updates
- **Data/ML:** scikit-learn (DecisionTreeClassifier, LogisticRegression), numpy, pandas,
    matplotlib
- **Visualization:** Matplotlib (for decision tree plots), collapsible cards for history
- **Styling:** Custom responsive CSS with animated elements, sticky action bars, and
    comparison UI

## 🧪 Example Workflow

1. Select a λ value (regularization strength) and click **Train Sparse Decision Tree**.
2. View tree accuracy, number of leaves, and a rendered tree plot.
3. Adjust λ (e.g., from 0.1 → 0.3) to see how complexity and accuracy trade off.
4. Train a **Sparse Logistic Regression** model and inspect which features were pruned.
5. Generate **Counterfactuals** for an individual test example — see which feature
    values need to change to flip its predicted class.

---
# 🎬 Project 4 – Influence of Future Predictions in

## Recommender Systems

This project is an **interactive Django web application** that allows users to participate in a
guided cold-start study for movie recommendations.

It combines a structured user study with machine-learning–powered collaborative filtering
to learn user preferences from very few ratings and generate personalized
recommendations.

## 🚀 Features

- **Interactive Rating Flow** – Users rate **10 randomly selected movies** , one at a time.
- **Dynamic Genre Hints** – Before each rating, the system displays a hint such as:

_“Your rating for this will give us your first signal about Drama, Thriller, so we can explore
more like these.”_

- **Impact Feedback (Group A)** – After each rating, participants in Group A see an
    explanation of how their rating influenced their profile (explore/refine effect).
- **Personalized Recommendations** – After 10 ratings, a **matrix-factorization model**
    generates top-N recommendations with predicted scores.
- **Final Questionnaire** – Users rate **perceived accuracy** , **trust** , and provide
    qualitative feedback.
- **Debrief Page** – Reveals the assigned experimental group (A/B) and closes the study.

## 🧠 Tech Stack

- **Backend:** Django (Python)
- **Frontend:** HTML, CSS (Django templating)


- **Machine Learning:**
     Data: MovieLens Dataset (movies.csv, ratings.csv)
     Model: Matrix Factorization using TruncatedSVD
     Optimization: SciPy’s minimize to learn new user latent vector
- **Libraries:** pandas, numpy, scikit-learn, joblib
- **Styling:** Responsive CSS with modern UI, progress counters, and study navigation
    buttons

## 🧩 Example Workflow

1. **Consent** – User agrees to participate and is randomly assigned to **Group A** (with
    impact feedback) or **Group B** (control).
2. **Rate Movies** – User rates 10 movies (0.5–5.0 scale) with genre hints guiding them.
3. **Impact View (Group A)** – After each rating, the system displays how this rating
    refined or broadened their taste profile.
4. **Recommendations** – After 10 ratings, the system generates a ranked list of
    recommended movies with predicted scores (0.5–5.0).
5. **Questionnaire & Debrief** – User rates accuracy/trust, provides open feedback, and
    sees which experimental group they were in.

(Optional) Pre-train item factor matrix:

- python manage.py shell
    from project4.views import train_matrix_factorization_model
    train_matrix_factorization_model()

Then start the development server:

- python manage.py runserver

---
# 🐭 Project 5 – Reinforcement Learning with

## Human Feedback (RLHF)

This project is a **Django web application** that demonstrates the use of reinforcement
learning with human feedback (RLHF) for training an agent (a mouse) to collect standard
cheese while avoiding traps and organic cheese.

The system allows users to:

- Train a baseline policy using the REINFORCE algorithm.
- View trajectories (step-by-step actions + rewards) taken by the trained policy.
- Provide human feedback by comparing two trajectories and selecting the preferred
    one.
- Retrain the policy using the learned reward model and KL-penalized REINFORCE
    fine-tuning.
- Visualize performance improvement (average rewards, histograms, and
    comparison charts).

## 🚀 Features

- **Baseline Training** – Train the initial policy network using REINFORCE and visualize
    results.
- **Trajectory Visualization** – See step-by-step actions, rewards, and grid states
    (mouse, cheese, traps).
- **Human Feedback Collection** – Compare two trajectories side by side and choose
    which is better.
- **Reward Model Training** – Learn a reward function using Bradley–Terry pairwise loss
    on trajectory sums.
- **RLHF Fine-Tuning** – Retrain the policy network using the learned reward + KL
    penalty to stay close to baseline.
- **Performance Summary & Graphs** – View average rewards, standard deviation, and
    reward distributions before vs. after retraining.


## 🛠 Tech Stack

- **Backend:** Django (Python)
- **RL & ML:** PyTorch (Policy Network, Reward Model), NumPy
- **Frontend:** HTML, CSS (responsive, translucent design with grid visualization)
- **Visualization:** Matplotlib (learning curves, histograms, comparison plots)

## 📊 Example Workflow

1. **Start Training**
    a. Click **Start REINFORCE Training** to train the baseline policy.
    b. View the learning curve and reward distribution for the trained policy.
2. **Provide Human Feedback**
    a. Click **Start Feedback Study** to see two trajectory rollouts.
    b. Choose which trajectory you prefer (e.g., one that avoids traps and collects
       more cheese).
3. **Retrain Policy**
    a. Click **Retrain Policy** to fine-tune the policy using your feedback.
    b. View performance summary (average rewards) and comparison graphs.
4. **Inspect Results**
    a. Scroll down to see **Baseline Trajectories** vs. **RLHF Trajectories** , total
       rewards, and detailed step-by-step actions.
