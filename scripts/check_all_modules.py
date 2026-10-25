import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_all_modules():
    results = {}
    print("==================================================")
    print("STARTING FULL SYSTEM MODULE HEALTH CHECK")
    print("==================================================")

    # 1. Health & Model Architecture
    try:
        r = requests.get(f"{BASE_URL}/api/health", timeout=15)
        data = r.json()
        assert r.status_code == 200
        assert data["status"] == "healthy"
        results["Health & Models"] = f"PASS (Catalog: {data.get('catalog_size')}, Pipeline: {data.get('active_pipeline')})"
    except Exception as e:
        results["Health & Models"] = f"FAIL: {e}"

    # 2. Users & Personas
    try:
        r = requests.get(f"{BASE_URL}/api/users", timeout=15)
        users = r.json()
        assert r.status_code == 200 and len(users) >= 6
        results["Users & Personas"] = f"PASS ({len(users)} personas loaded)"
    except Exception as e:
        results["Users & Personas"] = f"FAIL: {e}"

    # 3. Catalog & Songs Index
    try:
        r = requests.get(f"{BASE_URL}/api/songs?limit=10", timeout=15)
        songs = r.json()
        assert r.status_code == 200 and len(songs.get("songs", [])) > 0
        results["Catalog & Index"] = f"PASS ({songs.get('total')} total songs indexable)"
    except Exception as e:
        results["Catalog & Index"] = f"FAIL: {e}"

    # 4. Hybrid Deep Learning Recommendation Engine
    try:
        payload = {
            "user_id": "usr_00002",
            "model_type": "deep_ranker",
            "history_song_ids": [],
            "history_skip_types": [],
            "history_likes": [],
            "top_k": 5
        }
        r = requests.post(f"{BASE_URL}/api/recommend/hybrid", json=payload, timeout=15)
        rec = r.json()
        assert r.status_code == 200
        assert len(rec.get("recommendations", [])) > 0
        results["DL Recommendations (Hybrid)"] = f"PASS ({len(rec['recommendations'])} recs, Pipeline: {rec.get('active_pipeline')})"
    except Exception as e:
        results["DL Recommendations (Hybrid)"] = f"FAIL: {e}"

    # 5. Online Deep Learning Feedback & Optimization Step
    try:
        fb_payload = {
            "session_id": "sess_test_001",
            "user_id": "usr_00002",
            "song_id": "trk_100001",
            "action_type": "liked",
            "listen_ms": 45000
        }
        r = requests.post(f"{BASE_URL}/api/feedback", json=fb_payload, timeout=15)
        fb = r.json()
        assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
        assert "reward" in fb
        results["DL Feedback & Online Learning"] = f"PASS (Utility: {fb.get('reward')}, State tracked: {list(fb.get('user_state', {}).keys())})"
    except Exception as e:
        results["DL Feedback & Online Learning"] = f"FAIL: {e}"

    # 6. Related Songs Discovery
    try:
        r = requests.get(f"{BASE_URL}/api/songs/trk_100001/related?limit_per_section=4", timeout=15)
        rel = r.json()
        assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
        sections = list(rel.get("sections", {}).keys())
        results["Related Songs & Acoustic Twins"] = f"PASS (Sections: {sections})"
    except Exception as e:
        results["Related Songs & Acoustic Twins"] = f"FAIL: {e}"

    # 7. Synchronized Real-Time Lyrics
    try:
        r = requests.get(f"{BASE_URL}/api/lyrics?song_id=trk_100001&title=Levitating&artist=Dua+Lipa", timeout=15)
        lyr = r.json()
        assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
        results["Synchronized Lyrics Engine"] = f"PASS (Source: {lyr.get('source')}, Lines: {len(lyr.get('synced_lyrics', []))})"
    except Exception as e:
        results["Synchronized Lyrics Engine"] = f"FAIL: {e}"

    # 8. OpenSpot Search & Streaming Charts
    try:
        r = requests.get(f"{BASE_URL}/api/openspot/charts?genre=Pop", timeout=15)
        chart = r.json()
        assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
        chart_tracks = chart.get('results', chart.get('tracks', []))
        assert len(chart_tracks) > 0, "No tracks returned in charts"
        results["OpenSpot Live Charts"] = f"PASS ({len(chart_tracks)} tracks returned for Pop)"
    except Exception as e:
        results["OpenSpot Live Charts"] = f"FAIL: {e}"

    # 9. Stream Resolver
    try:
        r = requests.get(f"{BASE_URL}/api/stream/resolve?song_id=trk_100001", timeout=15)
        stream = r.json()
        assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
        results["Audio Stream Resolver"] = f"PASS (Resolved: {bool(stream.get('audio_url'))}, ID: {stream.get('song_id')})"
    except Exception as e:
        results["Audio Stream Resolver"] = f"FAIL: {e}"

    # 10. A/B Simulation Lab Engine
    try:
        sim_payload = {"num_sessions": 3, "models_to_compare": ["deep_ranker", "baseline", "random"]}
        r = requests.post(f"{BASE_URL}/api/simulate", json=sim_payload, timeout=15)
        sim = r.json()
        assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
        results["A/B Simulation Lab"] = f"PASS (Models evaluated: {list(sim.get('comparison', {}).keys())})"
    except Exception as e:
        results["A/B Simulation Lab"] = f"FAIL: {e}"

    # 11. Offline Metrics & Guardrails
    try:
        r = requests.get(f"{BASE_URL}/api/metrics", timeout=15)
        met = r.json()
        assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
        results["Metrics & Guardrails Engine"] = f"PASS (Guardrails: {list(met.get('guardrails', {}).keys())})"
    except Exception as e:
        results["Metrics & Guardrails Engine"] = f"FAIL: {e}"

    # 12. Dynamic Behavior Store & Profile
    try:
        r = requests.get(f"{BASE_URL}/api/behavior/profile/usr_00002", timeout=15)
        prof = r.json()
        assert r.status_code == 200
        results["Behavior & Taste Profile"] = f"PASS (Mood: {prof.get('session_mood')}, Centroid: {bool(prof.get('acoustic_centroid'))})"
    except Exception as e:
        results["Behavior & Taste Profile"] = f"FAIL: {e}"

    # 13. OpenSpot Paginated Search Engine
    try:
        r1 = requests.get(f"{BASE_URL}/api/openspot/search?q=Taylor&limit=5&p=1", timeout=15)
        r2 = requests.get(f"{BASE_URL}/api/openspot/search?q=Taylor&limit=5&p=2", timeout=15)
        d1, d2 = r1.json(), r2.json()
        assert r1.status_code == 200 and r2.status_code == 200
        assert len(d1.get("results", [])) > 0 and len(d2.get("results", [])) > 0
        results["OpenSpot Paginated Search"] = f"PASS (Page 1: {len(d1['results'])} tracks, Page 2: {len(d2['results'])} tracks)"
    except Exception as e:
        results["OpenSpot Paginated Search"] = f"FAIL: {e}"

    # 14. Dynamic Neural Import to DL Catalog
    try:
        sample_track = {
            "song_id": "os_test_chk_99",
            "title": "Module Check Song",
            "artist_name": "Antigravity Test Artist",
            "album": "Deep Learning Album",
            "genre": "Pop",
            "image_url": "https://example.com/art.jpg",
            "audio_url": "https://example.com/audio.mp4",
            "duration_ms": 200000,
            "danceability": 0.70,
            "energy": 0.80,
            "valence": 0.75,
            "tempo": 124.0,
            "acousticness": 0.15,
            "instrumentalness": 0.01,
            "speechiness": 0.05,
            "loudness": -5.0,
            "popularity": 85
        }
        r_imp = requests.post(f"{BASE_URL}/api/openspot/import", json=sample_track, timeout=15)
        imp = r_imp.json()
        assert r_imp.status_code == 200 and imp.get("status") == "success"
        results["Dynamic Neural Import to DL"] = f"PASS (Catalog size: {imp.get('catalog_size')})"
    except Exception as e:
        results["Dynamic Neural Import to DL"] = f"FAIL: {e}"

    print("\n==================================================")
    print("MODULE CHECK RESULTS SUMMARY")
    print("==================================================")
    all_pass = True
    for mod, status in results.items():
        is_p = status.startswith("PASS")
        if not is_p:
            all_pass = False
        print(f"[{'PASS' if is_p else 'FAIL'}] {mod.ljust(35)}: {status}")
    print("==================================================")
    print("ALL MODULES FUNCTIONAL:", all_pass)
    print("==================================================")

if __name__ == "__main__":
    test_all_modules()
