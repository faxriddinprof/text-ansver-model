from django.urls import path
from . import views

app_name = "esg_app"

urlpatterns = [
    path("analyze/",        views.analyze, name="analyze"),
    path("history/",        views.history, name="history"),
    path("history/<int:pk>/", views.detail, name="detail"),
]
