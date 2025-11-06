from unittest.mock import AsyncMock

import pytest

from simple_narrative_scraper import SimpleNarrativeScraper


@pytest.mark.asyncio
async def test_get_game_narrative_uses_cache(monkeypatch):
    scraper = SimpleNarrativeScraper(cache_ttl=300)

    reddit_payload = {
        "narrative_strength": 0.8,
        "public_lean": 0.65,
        "vegas_bait": False,
        "media_hype": 0.4,
        "storylines": ["Playoff Implications"],
        "sentiment_counts": {"over": 4, "under": 2, "hype": 1},
    }
    youtube_payload = {
        "videos_analyzed": 2,
        "narratives": {"trap_game": 0.2},
        "narrative_strength": 0.5,
    }
    twitter_payload = {
        "tweet_count": 5,
        "sharp_mentions": 2,
        "public_mentions": 3,
        "sharp_vs_public": 0.4,
    }

    reddit_mock = AsyncMock(return_value=reddit_payload)
    youtube_mock = AsyncMock(return_value=youtube_payload)
    twitter_mock = AsyncMock(return_value=twitter_payload)

    monkeypatch.setattr(scraper, "_scrape_reddit_nfl", reddit_mock)
    monkeypatch.setattr(scraper, "_scrape_youtube_sentiment", youtube_mock)
    monkeypatch.setattr(scraper, "_scrape_twitter_sentiment", twitter_mock)

    first_result = await scraper.get_game_narrative("KC", "BUF")
    second_result = await scraper.get_game_narrative("KC", "BUF")

    assert first_result == second_result
    assert reddit_mock.await_count == 1
    assert youtube_mock.await_count == 1
    assert twitter_mock.await_count == 1
    assert first_result["confidence_weights"]["overall"] > 0

    await scraper.close()


def test_sparse_data_downweights_public_bias():
    scraper = SimpleNarrativeScraper()
    reddit_data = {
        "narrative_strength": 0.7,
        "public_lean": 0.9,
        "vegas_bait": True,
        "media_hype": 0.3,
        "storylines": [],
        "sentiment_counts": {"over": 1, "under": 0, "hype": 0},
    }
    youtube_data = {"videos_analyzed": 0, "narratives": {}, "narrative_strength": 0}
    twitter_data = {"tweet_count": 0, "sharp_vs_public": 0.5}

    combined = scraper._combine_narratives(reddit_data, youtube_data, twitter_data)
    assert combined["public_lean"] < 0.75
    assert combined["public_lean_confidence"] < 0.2
    assert combined["confidence_weights"]["overall"] < 0.2


def test_conspiracy_score_scales_with_signal_weight():
    scraper = SimpleNarrativeScraper()

    heavy_reddit = {
        "narrative_strength": 0.9,
        "public_lean": 0.8,
        "vegas_bait": True,
        "media_hype": 0.8,
        "storylines": ["Vegas Trap"],
        "sentiment_counts": {"over": 6, "under": 2, "hype": 3},
    }
    heavy_youtube = {
        "videos_analyzed": 6,
        "narratives": {"trap_game": 0.4, "public_favorite": 0.5},
        "narrative_strength": 0.7,
    }
    heavy_twitter = {
        "tweet_count": 45,
        "sharp_vs_public": 0.2,
    }

    sparse_reddit = {
        "narrative_strength": 0.6,
        "public_lean": 0.8,
        "vegas_bait": True,
        "media_hype": 0.6,
        "storylines": [],
        "sentiment_counts": {"over": 1, "under": 0, "hype": 0},
    }
    sparse_youtube = {
        "videos_analyzed": 1,
        "narratives": {"trap_game": 0.4, "public_favorite": 0.4},
        "narrative_strength": 0.4,
    }
    sparse_twitter = {
        "tweet_count": 0,
        "sharp_vs_public": 0.2,
    }

    strong_signals = scraper._combine_narratives(heavy_reddit, heavy_youtube, heavy_twitter)
    weak_signals = scraper._combine_narratives(sparse_reddit, sparse_youtube, sparse_twitter)

    assert strong_signals["conspiracy_score"] > 0.6
    assert weak_signals["conspiracy_score"] < strong_signals["conspiracy_score"]
    assert weak_signals["conspiracy_score"] < 0.5
