
from django.urls import path
from . import views

app_name = 'project2'
urlpatterns = [
    path('', views.index, name='index'),
    path('train_baseline_classifier/', views.train_baseline_classifier, name='train_baseline_classifier'),
    path('load_pretrained_classifier/', views.load_pretrained_classifier, name='load_pretrained_classifier'),
    path('active_learn_init/', views.active_learn_init, name='active_learn_init'),
    path('active_learn_query/', views.active_learn_query, name='active_learn_query'),
    path('active_learn_reset/', views.active_learn_reset, name='active_learn_reset'),
]