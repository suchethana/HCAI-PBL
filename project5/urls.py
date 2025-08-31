from django.urls import path
from . import views

app_name = 'project5'

urlpatterns = [
    path('', views.index, name='index'),
    path('train/', views.train_agent_view, name='train_agent'),
    path('feedback/study/', views.start_feedback_study_view, name='start_feedback_study'),
    path('feedback/collect/', views.collect_feedback_view, name='collect_feedback'),
    path('retrain/', views.retrain_with_feedback_view, name='retrain_with_feedback'),
]