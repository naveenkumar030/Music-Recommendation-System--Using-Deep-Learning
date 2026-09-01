"""
Spotify Web API Integration Client.

Provides:
- Generation of direct Spotify Web Player URLs and Search URLs.
- Optional Spotify Web API authentication (Client Credentials Flow) for track fetching and metadata.
- Spotify Embed URL generator for in-app iframe playback.
"""

import os
import urllib.parse
from typing import Dict, List, Optional
import requests


class SpotifyClient:
    """Helper client for Spotify linking and Web API integration."""

    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        self.client_id = client_id or os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("SPOTIFY_CLIENT_SECRET")
        self.access_token: Optional[str] = None

    def get_search_url(self, title: str, artist: str) -> str:
        """Generates a direct Spotify Web search URL."""
        query = f"{title} {artist}"
        encoded = urllib.parse.quote(query)
        return f"https://open.spotify.com/search/{encoded}"

    def get_embed_url(self, spotify_track_id: Optional[str] = None, title: Optional[str] = None, artist: Optional[str] = None) -> str:
        """Generates Spotify embed player URL."""
        if spotify_track_id:
            return f"https://open.spotify.com/embed/track/{spotify_track_id}?utm_source=generator&theme=0"
        return self.get_search_url(title or "", artist or "")

    def authenticate(self) -> bool:
        """Authenticates with Spotify Web API using Client Credentials flow if keys provided."""
        if not self.client_id or not self.client_secret:
            return False

        try:
            auth_url = "https://accounts.spotify.com/api/token"
            res = requests.post(
                auth_url,
                data={"grant_type": "client_credentials"},
                auth=(self.client_id, self.client_secret),
                timeout=5
            )
            if res.status_code == 200:
                self.access_token = res.json().get("access_token")
                return True
        except Exception:
            pass
        return False

    def search_spotify_track(self, title: str, artist: str) -> Optional[Dict]:
        """Searches Spotify Web API for track metadata and preview URL."""
        if not self.access_token:
            if not self.authenticate():
                return None

        try:
            query = f"track:{title} artist:{artist}"
            url = f"https://api.spotify.com/v1/search?q={urllib.parse.quote(query)}&type=track&limit=1"
            headers = {"Authorization": f"Bearer {self.access_token}"}
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code == 200:
                items = res.json().get("tracks", {}).get("items", [])
                if items:
                    track = items[0]
                    images = track.get("album", {}).get("images", [])
                    album_art = images[0]["url"] if images else None
                    return {
                        "spotify_id": track["id"],
                        "spotify_url": track["external_urls"]["spotify"],
                        "preview_url": track.get("preview_url"),
                        "album_art": album_art,
                        "album_name": track.get("album", {}).get("name")
                    }
        except Exception:
            pass
        return None
