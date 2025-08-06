from __future__ import annotations

from django.db import models


class NewsItem(models.Model):
    """Model representing a news article and its sentiment analysis."""

    title = models.CharField(max_length=255)
    description = models.TextField(null=True, blank=True)
    content = models.TextField(null=True, blank=True)
    url = models.URLField(max_length=512)
    published_at = models.DateTimeField()
    location = models.CharField(max_length=255, default='', help_text="Detected location keyword, if any")
    sentiment_polarity = models.FloatField()
    sentiment_subjectivity = models.FloatField()

    def __str__(self) -> str:  # pragma: no cover
        return self.title