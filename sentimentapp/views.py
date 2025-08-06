from __future__ import annotations

import os
import re
from collections import Counter,defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

import requests
from bs4 import BeautifulSoup  # type: ignore
from django.conf import settings
from django.db.models import QuerySet
from django.utils import timezone
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView
from textblob import TextBlob  # type: ignore
from transformers import pipeline
from bs4 import BeautifulSoup
import requests
from datetime import date, datetime, timedelta
from serpapi import GoogleSearch




from .models import NewsItem
from .serializers import NewsItemSerializer, TextAnalysisSerializer
# from .services.news_service import fetch_and_analyze_news 


def fetch_article_content(url: str) -> str:
    """
    Scrapes the visible text content from a news article using BeautifulSoup.
    Returns a clean string.
    """

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code != 200:
            return ""

        soup = BeautifulSoup(resp.text, 'html.parser')

        # Remove scripts and styles
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()

        # Get text
        text = soup.get_text(separator=' ')
        text = ' '.join(text.split())  # Clean excessive spaces
        return text[:2000]  # truncate to avoid huge text

    except Exception as e:
        print(f"Error fetching article: {e}")
        return ""



# Load transformer model once
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def analyze_sentiment_transformer(text: str):
    result = sentiment_pipeline(text[:512])[0]  # truncate to avoid long-text issues
    label = result['label']  # POSITIVE / NEGATIVE
    score = result['score']
    polarity = score if label == "POSITIVE" else -score
    subjectivity = 0.5  # placeholder since model does not give subjectivity
    return polarity, subjectivity


def analyze_sentiment(text: str) -> Tuple[float, float]:
    """Return the sentiment polarity and subjectivity of the given text."""
    print(text)
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment.polarity, sentiment.subjectivity


def extract_location_from_text(text: str) -> str:
    """Very naive location detection based on keyword matching.

    Looks for predefined location names in the provided text and returns the
    first match.  If no location is found returns an empty string.
    """
    # List of example location keywords; expand as needed
    locations = [
        'India', 'USA', 'United States', 'China', 'UK', 'United Kingdom', 'Gujarat', 'Surat',
        'Delhi', 'New York', 'California', 'Europe', 'Asia', 'Africa',
    ]
    text_lower = text.lower()
    for loc in locations:
        if loc.lower() in text_lower:
            return loc
    return ''


def fetch_serper_news(query: str) -> List[Dict[str, Any]]:
    """Fetch news articles from Serper API.

    Returns a list of article dictionaries or an empty list if the request
    fails or the API key is not configured.
    """
    # api_key = getattr(settings, 'SERPER_API_KEY', '')
    # if not api_key:
        # return []
    url = "https://www.searchapi.io/api/v1/search"

    # headers = {
    #     'X-API-KEY': api_key,
    #     'Content-Type': 'application/json',
    # }
    # payload = {'q': query}
    params = {
        "engine": "google_news",
        "q": query,
        "location": "New Delhi,India",
        "api_key": "Uz8jKEySxcQqDfJRWqJwCJ2A"
        }
    print(params)
    try:
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            return []
        data = resp.json()
        # Extract the fields as a list of dictionaries
        results = [
            {
        "title": item.get("title"),
        "description": item.get("snippet"),
        "link": item.get("link"),
        "date": item.get("date")
        }
        for item in data.get("organic_results", [])
        ]

        return results
    except Exception:
        return []


def fetch_google_news_fallback(query: str) -> List[Dict[str, Any]]:
    """Fetch news articles via Google News RSS feed as a fallback.

    This method scrapes the Google News RSS feed for the given query.
    Returns up to 10 articles.
    """
    from urllib.parse import quote_plus

    rss_url = f'https://news.google.com/rss/search?q={quote_plus(query)}'
    articles: List[Dict[str, Any]] = []
    try:
        resp = requests.get(rss_url, timeout=10)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.content, 'xml')
        items = soup.find_all('item')[:10]
        for item in items:
            title = item.title.text if item.title else ''
            description = item.description.text if item.description else ''
            link = item.link.text if item.link else ''
            pub_date = item.pubDate.text if item.pubDate else ''
            # Convert pubDate to datetime; fallback to now on failure
            try:
                published_at = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z')
            except Exception:
                published_at = timezone.now()
            articles.append({
                'title': title,
                'description': description,
                'link': link,
                'published_at': published_at,
            })
    except Exception:
        pass
    return articles


class NewsSentimentAPIView(APIView):
    """API endpoint for news search with sentiment analysis."""

    def get(self, request, *args, **kwargs):
        query = request.GET.get('q')
        print("NewsSentimentAPIView called",query)
        if not query:
            return Response({'detail': 'Query parameter "q" is required.'}, status=status.HTTP_400_BAD_REQUEST)

        # Fetch articles from Serper API or fallback
        articles = fetch_serper_news(query)
        print(articles)
        if not articles:
            # Use fallback scraping when API call fails or API key is missing
            articles = fetch_google_news_fallback(query)
            # Convert keys to align with Serper's response format
            formatted_articles: List[Dict[str, Any]] = []
            for art in articles:
                formatted_articles.append({
                    'title': art.get('title'),
                    'description': art.get('description'),
                    'link': art.get('link'),
                    'date': art.get('published_at').isoformat() if isinstance(art.get('published_at'), datetime) else str(art.get('published_at')),
                })
            articles = formatted_articles
            print(articles)

        results: List[Dict[str, Any]] = []
        for art in articles:
            title = art.get('title') or ''
            description = art.get('description') or ''
            url = art.get('link') or art.get('url') or ''
            published_at_str = art.get('date') or art.get('published_date') or art.get('publishedAt') or ''
            content= fetch_article_content(url) or ''
            # Parse published date string
            try:
                published_at = datetime.fromisoformat(published_at_str.replace('Z', '+00:00'))
            except Exception:
                published_at = timezone.now()
            # Compute sentiment
            analyse=title + ' ' + description + ' ' + content
            polarity, subjectivity = analyze_sentiment_transformer(analyse)
            # Detect location
            location = extract_location_from_text(f"{title} {description}")
            results.append({
                'title': title,
                'description': description,
                'url': url,
                'published_at': published_at,
                'sentiment_polarity': polarity,
                'sentiment_subjectivity': subjectivity,
                'location': location,
            })
            # Persist to the database so that dashboard statistics remain meaningful
            NewsItem.objects.create(
                title=title,
                description=description,
                content=description,
                url=url,
                published_at=published_at,
                location=location,
                sentiment_polarity=polarity,
                sentiment_subjectivity=subjectivity,
            )

        return Response(results, status=status.HTTP_200_OK)


class AnalyzeTextAPIView(APIView):
    """API endpoint for free‑text sentiment analysis."""

    def post(self, request, *args, **kwargs):
        serializer = TextAnalysisSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        text = serializer.validated_data['text']
        polarity, subjectivity = analyze_sentiment_transformer(text)
        print(polarity, subjectivity)
        return Response({
            'text': text,
            'sentiment_polarity': polarity,
            'sentiment_subjectivity': subjectivity,
        }, status=status.HTTP_200_OK)


class DashboardAPIView(APIView):
    """
    GET /api/dashboard/
    → queries Google News via SerpAPI + falls back on BeautifulSoup,
      runs transformer sentiment on each snippet, aggregates everything.
    """

    def get(self, request):
        # --- 1) Fetch latest 20 Google News results via SerpAPI ---
        print("DashboardAPIView called")
        results = fetch_serper_news("India news")
        print(results[0])

        articles=[]

        try:
            # print(articles)
            for it in results:
                articles.append({
                    "title":     it.get("title", ""),
                    "link":      it.get("link", ""),
                    "published": it.get("published_date", ""),
                    "snippet":   it.get("snippet", ""),
                    "source":    it.get("source_id", ""),
                })
        except Exception:
            # Fallback scraping
            fb_url = "https://news.google.com/search?q=world%20news&hl=en-US&gl=US&ceid=US:en"
            try:
                page = requests.get(fb_url, timeout=10)
                soup = BeautifulSoup(page.text, "html.parser")
                for art in soup.select("article")[:20]:
                    a = art.find("a", href=True)
                    title = a.text.strip()
                    link = "https://news.google.com" + a["href"][1:]
                    pub   = art.select_one("time")["datetime"] if art.select_one("time") else ""
                    snip  = art.select_one(".HO8did").text if art.select_one(".HO8did") else ""
                    articles.append({
                        "title":   title,
                        "link":    link,
                        "published": pub,
                        "snippet": snip,
                        "source":  "",
                    })
            except Exception:
                articles = []

        # --- 2) Run sentiment pipeline on each snippet/title ---
        sentiment_counts = Counter()
        pos_buckets = defaultdict(list)
        neu_buckets = defaultdict(list)
        neg_buckets = defaultdict(list)
        location_counts  = Counter()
        recents          = []

        today = date.today()
        last_three_days = [today - timedelta(days=i) for i in range(2, -1, -1)]
        day_keys = [d.strftime("%b %d") for d in last_three_days]

        for art in articles:
            text_in = art["snippet"] or art["title"]
            try:
                out = sentiment_pipeline(text_in[:512])[0]
                lab = out["label"]      # "POSITIVE" or "NEGATIVE"
                score = out["score"]    # 0.00–1.00
            except Exception:
                # fallback neutral
                lab, score = "NEUTRAL", 0.0

            # convert to unified counts
            if lab == "POSITIVE":
                polarity = score
                sentiment_counts["positive"] += 1
            elif lab == "NEGATIVE":
                polarity = -score
                sentiment_counts["negative"] += 1
            else:
                polarity = 0.0
                sentiment_counts["neutral"]  += 1
            


            # trend by day
            try:
                dt = datetime.fromisoformat(art["published"])
                day = dt.strftime("%b %d")
            except Exception:
                day = today.strftime("%b %d")
            if polarity > 0:
                pos_buckets[day].append(polarity)
            elif polarity < 0:
                neg_buckets[day].append(abs(polarity))
            else:
                neu_buckets[day].append(0)

            # if day in three_days:
            #     trend_buckets[day].append(polarity)

            # location by source
            src = art.get("source") or ""
            if src:
                location_counts[src] += 1

            recents.append({
                "title": art["title"],
                "published_at": art["published"]
            })

        # --- 3) Build `sentiment_trend` list for last3 days ---
        sentiment_trend = []
        for day in day_keys:
            sentiment_trend.append({
                "date":    day,
                "positive": round(sum(pos_buckets.get(day, [])) / (len(pos_buckets.get(day, [])) or 1), 3),
                "negative": round(sum(neg_buckets.get(day, [])) / (len(neg_buckets.get(day, [])) or 1), 3),
                "neutral":  round(sum(neu_buckets.get(day, [])) / (len(neu_buckets.get(day, [])) or 1), 3),
            })

        # --- 4) Trending keywords (simple freq on titles) ---
        from collections import Counter as _C
        words = _C()
        stop = set(["the","and","news","world","says","say"])
        for art in articles:
            for w in art["title"].lower().split():
                w2 = "".join(ch for ch in w if ch.isalpha())
                if len(w2)>3 and w2 not in stop:
                    words[w2] += 1
        trending_keywords = [w for w,_ in words.most_common(10)]

        # --- 5) Country counts for map (example mapping) ---
        iso_map = {"BBC":"GB","CNN":"US","Al Jazeera":"QA"}
        country_counts = { iso_map.get(k,"UN"): v for k,v in location_counts.items() }

        payload = {
            "sentiment_distribution": {
                "positive": sentiment_counts["positive"],
                "neutral":  sentiment_counts["neutral"],
                "negative": sentiment_counts["negative"],
            },
            "sentiment_trend":    sentiment_trend,
            "location_distribution": dict(location_counts),
            "trending_keywords":  trending_keywords,
            "recent_articles":    recents[:5],
            "country_counts":     country_counts,
            "total_articles":     len(articles),
        }
        print(payload)
        return Response(payload)