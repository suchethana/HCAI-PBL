from django.urls import path
from . import views

app_name = 'project4'
urlpatterns = [
    path('', views.index, name='index'),
    path('start_study/', views.start_study, name='start_study'),
    path('rate_movie/', views.rate_movie, name='rate_movie'),
    path('recommendations/', views.show_recommendations, name='recommendations'),
    path('questionnaire/', views.questionnaire, name='questionnaire'),
    path('debrief/', views.debrief, name='debrief'),
]