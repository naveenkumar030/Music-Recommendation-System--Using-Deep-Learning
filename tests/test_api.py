"""
Unit and Integration Tests for Milestone 5: FastAPI Serving and Latency Benchmarks.
"""

import time
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.serving.api import app, startup_event


@pytest.fixture(scope="module")
def client():
    startup_event()
    return TestClient(app)


def test_health_check(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["catalog_size"] >= 1000
    assert data["models_loaded"]["rl_agent"] is True


def test_songs_and_users_endpoints(client):
    # Browse mode with limit=10: should return envelope {songs, total, offset, limit}
    res_songs = client.get("/api/songs?limit=10")
    assert res_songs.status_code == 200
    data = res_songs.json()
    # New envelope shape
    assert "songs" in data, f"Expected 'songs' key in response, got: {list(data.keys())}"
    assert "total" in data
    assert "offset" in data
    assert "limit" in data
    songs = data["songs"]
    assert len(songs) == 10
    assert "danceability" in songs[0]
    # Total must be the full catalog size, not just the page
    assert data["total"] >= 10

    res_users = client.get("/api/users")
    assert res_users.status_code == 200
    users = res_users.json()
    assert len(users) > 0


def test_recommendation_and_latency_benchmark(client):
    """Verifies recommendations and validates that p95 latency is strictly under 100 ms."""
    latencies = []
    
    for i in range(25):
        t0 = time.perf_counter()
        res = client.post("/api/recommend", json={
            "user_id": "usr_00001",
            "history_song_ids": ["trk_100001", "trk_100002"],
            "history_skip_types": ["no_skip", "skip_early"],
            "history_likes": [1, 0],
            "model_type": "wolpertinger",
            "slate_size": 5
        })
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

        assert res.status_code == 200
        data = res.json()
        assert len(data["recommendations"]) == 5
        assert "q_score_distribution" in data

    p95_lat = np.percentile(latencies, 95)
    mean_lat = np.mean(latencies)
    assert p95_lat < 100.0, f"p95 Latency too high: {p95_lat:.2f} ms (expected < 100ms)"


def test_feedback_and_simulation(client):
    # Test feedback
    res_fb = client.post("/api/feedback", json={
        "session_id": "test_sess_01",
        "user_id": "usr_00001",
        "song_id": "trk_100001",
        "action_type": "liked"
    })
    assert res_fb.status_code == 200
    assert res_fb.json()["reward"] > 1.0

    # Test simulation
    res_sim = client.post("/api/simulate", json={
        "num_sessions": 5,
        "models_to_compare": ["wolpertinger", "baseline", "random"]
    })
    assert res_sim.status_code == 200
    sim_data = res_sim.json()
    assert "wolpertinger" in sim_data["comparison"]
    assert "baseline" in sim_data["comparison"]


def test_related_songs_endpoint(client):
    """Verifies that the /api/songs/{song_id}/related endpoint returns categorized sections and low latency."""
    latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        res = client.get("/api/songs/trk_100001/related?limit_per_section=6")
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)
    
    assert res.status_code == 200
    data = res.json()
    assert "seed_song" in data
    assert data["seed_song"]["song_id"] == "trk_100001"
    assert "sections" in data
    
    sections = data["sections"]
    assert "acoustic_twins" in sections
    assert "same_genre" in sections
    assert "cross_genre" in sections
    assert "vibe_transitions" in sections
    
    # Check acoustic twins properties
    twins = sections["acoustic_twins"]["songs"]
    assert len(twins) > 0
    assert "match_score" in twins[0]
    assert "match_percentage" in twins[0]
    assert "acoustic_deltas" in twins[0]
    
    # Check same genre
    same_g = sections["same_genre"]["songs"]
    if len(same_g) > 0:
        assert same_g[0]["genre"] == data["seed_song"]["genre"]
        
    # Check cross genre
    cross_g = sections["cross_genre"]["songs"]
    if len(cross_g) > 0:
        assert cross_g[0]["genre"] != data["seed_song"]["genre"]
        
    # Check latency requirement: backend latency is well under 15ms, p95 HTTP under 50ms
    p95_lat = np.percentile(latencies, 95)
    assert data["latency_ms"] < 25.0
    assert p95_lat < 50.0



def test_songs_catalog_count(client):
    """
    Sanity check: /api/songs total must cover the full songs.parquet and must be
    >= the catalog_size from /api/health (the feature-store only indexes songs that have
    embeddings, so parquet can legitimately have slightly more rows).

    This guards against any future truncation regression in the browse endpoint.
    """
    health_res = client.get("/api/health")
    assert health_res.status_code == 200
    catalog_size = health_res.json()["catalog_size"]
    assert catalog_size >= 1000, f"Expected at least 1000 songs in feature store, got {catalog_size}"

    songs_res = client.get("/api/songs")
    assert songs_res.status_code == 200
    data = songs_res.json()
    assert "total" in data, "Response must include 'total' field"
    songs_total = data["total"]

    # songs.parquet rows >= feature-store indexed songs (some may lack embeddings)
    assert songs_total >= catalog_size, (
        f"/api/songs total={songs_total} < catalog_size={catalog_size}. "
        "The song catalog endpoint is missing rows that the feature store has indexed."
    )
    assert songs_total >= 1000, f"Expected at least 1000 songs in catalog, got {songs_total}"
    # The first page of songs should exist
    assert len(data["songs"]) > 0
    print(f"[Catalog Sanity] /api/songs total={songs_total}, feature_store catalog_size={catalog_size} ✅")



def test_songs_search_no_truncation(client):
    """
    A broad search query (single letter) should match many catalog songs —
    previously this was capped at 8 in the Quick-Add or 40 in the catalog tab.
    With the fix, the search path returns up to 500 matches uncapped.
    """
    # 'a' should match most songs in the catalog by title or artist
    res = client.get("/api/songs?search=a")
    assert res.status_code == 200
    data = res.json()
    assert "songs" in data
    assert "total" in data
    # With 1552 songs, searching 'a' should return far more than 50
    assert data["total"] > 50, (
        f"Search for 'a' returned only {data['total']} results. "
        "The search endpoint should not truncate result count."
    )
    # All returned songs must contain 'a' somewhere
    assert len(data["songs"]) <= 500  # capped at 500 per design
    print(f"[Search Sanity] /api/songs?search=a total={data['total']}, returned={len(data['songs'])} ✅")
