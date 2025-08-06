from __future__ import annotations

from django.contrib import admin

from .models import NewsItem


@admin.register(NewsItem)
class NewsItemAdmin(admin.ModelAdmin):
    list_display = (
        'title',
        'published_at',
        'location',
        'sentiment_polarity',
        'sentiment_subjectivity',
    )
    search_fields = ('title', 'description', 'location')
    list_filter = ('location',)