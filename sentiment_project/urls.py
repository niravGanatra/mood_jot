"""URL configuration for the Django project."""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    # API routes for the sentiment app
    path('api/', include('sentimentapp.urls')),
    
]