
from django.urls import path
from . import views

app_name = 'project1'
urlpatterns = [
    path('', views.index, name='index'),
    path('visualize/', views.visualize_data, name='visualize_data')
]