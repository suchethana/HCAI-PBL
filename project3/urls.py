# project3/urls.py
from django.urls import path
from . import views

app_name = 'project3'
urlpatterns = [
    path('', views.index, name='index'),
    path('train_dt_baseline/', views.train_decision_tree_baseline, name='train_dt_baseline'),
    path('train_sparse_dt_model/', views.train_sparse_dt_model, name='train_sparse_dt_model'),
    path('train_sparse_lr_model/', views.train_sparse_lr_model, name='train_sparse_lr_model'),
    path('generate_counterfactuals/', views.generate_counterfactuals, name='generate_counterfactuals'),
]