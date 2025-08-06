from __future__ import annotations

from rest_framework import serializers

from .models import NewsItem


class NewsItemSerializer(serializers.ModelSerializer):
    """Serializer for the NewsItem model."""

    class Meta:
        model = NewsItem
        fields = [
            'id',
            'title',
            'description',
            'content',
            'url',
            'published_at',
            'location',
            'sentiment_polarity',
            'sentiment_subjectivity',
        ]


class TextAnalysisSerializer(serializers.Serializer):
    """Serializer for analysing arbitrary text."""

    text = serializers.CharField()