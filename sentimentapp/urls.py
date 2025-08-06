"""URL patterns for the sentimentapp API endpoints."""
from django.urls import path

from .views import NewsSentimentAPIView, AnalyzeTextAPIView, DashboardAPIView

urlpatterns = [
    path('news-sentiment/', NewsSentimentAPIView.as_view(), name='news-sentiment'),
    path('analyze-text/', AnalyzeTextAPIView.as_view(), name='analyze-text'),
    path('dashboard/', DashboardAPIView.as_view(), name='dashboard'),
]