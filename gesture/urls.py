from django.urls import path
from .views import collect_coordinates, train_model_api

urlpatterns = [
    path("collect/", collect_coordinates),
    path("train/", train_model_api),
]
