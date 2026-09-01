"""
OpenSpot Music API Client & Stream Decryption Engine.

Based on OpenSpot (https://github.com/BlackHatDevX/openspot-music-app) architecture.
Provides:
- JioSaavn search & metadata retrieval.
- DES decryption of encrypted_media_url to direct HQ audio streaming URLs (320kbps / 160kbps MP4/AAC).
- High-resolution 500x500 album artwork extraction.
- Acoustic feature estimation & normalization for RL feature store compatibility.
"""

import base64
import html
import json
import logging
import re
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)

# Standard DES key used in OpenSpot / JioSaavn
DES_KEY = b"38346591"

# Base API URL
JIO_SAAVN_BASE_URL = "https://www.jiosaavn.com/api.php"

# Default User-Agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
]

GENRE_DEFAULTS = {
    "Pop": {"danceability": 0.68, "energy": 0.72, "valence": 0.65, "tempo": 122.0, "acousticness": 0.20, "instrumentalness": 0.001, "speechiness": 0.04, "loudness": -6.5, "popularity": 75},
    "Hip-Hop": {"danceability": 0.82, "energy": 0.74, "valence": 0.58, "tempo": 132.0, "acousticness": 0.12, "instrumentalness": 0.000, "speechiness": 0.18, "loudness": -5.5, "popularity": 78},
    "EDM / Electronic": {"danceability": 0.76, "energy": 0.89, "valence": 0.52, "tempo": 128.0, "acousticness": 0.07, "instrumentalness": 0.25, "speechiness": 0.06, "loudness": -4.8, "popularity": 70},
    "Rock": {"danceability": 0.48, "energy": 0.84, "valence": 0.52, "tempo": 126.0, "acousticness": 0.14, "instrumentalness": 0.05, "speechiness": 0.05, "loudness": -5.8, "popularity": 68},
    "R&B / Soul": {"danceability": 0.69, "energy": 0.54, "valence": 0.62, "tempo": 98.0, "acousticness": 0.38, "instrumentalness": 0.002, "speechiness": 0.07, "loudness": -7.2, "popularity": 72},
    "Lo-Fi / Chillhop": {"danceability": 0.58, "energy": 0.36, "valence": 0.46, "tempo": 84.0, "acousticness": 0.76, "instrumentalness": 0.65, "speechiness": 0.04, "loudness": -11.5, "popularity": 60},
    "Jazz": {"danceability": 0.52, "energy": 0.42, "valence": 0.58, "tempo": 108.0, "acousticness": 0.82, "instrumentalness": 0.55, "speechiness": 0.04, "loudness": -12.0, "popularity": 55},
    "Classical": {"danceability": 0.24, "energy": 0.28, "valence": 0.32, "tempo": 92.0, "acousticness": 0.94, "instrumentalness": 0.88, "speechiness": 0.03, "loudness": -16.0, "popularity": 50},
    "Indie / Alternative": {"danceability": 0.60, "energy": 0.66, "valence": 0.51, "tempo": 118.0, "acousticness": 0.42, "instrumentalness": 0.08, "speechiness": 0.04, "loudness": -7.5, "popularity": 65},
    "Metal": {"danceability": 0.36, "energy": 0.96, "valence": 0.32, "tempo": 142.0, "acousticness": 0.03, "instrumentalness": 0.15, "speechiness": 0.08, "loudness": -4.2, "popularity": 62},
    "Acoustic / Folk": {"danceability": 0.50, "energy": 0.39, "valence": 0.48, "tempo": 104.0, "acousticness": 0.86, "instrumentalness": 0.02, "speechiness": 0.04, "loudness": -9.8, "popularity": 58},
    "Synthwave": {"danceability": 0.67, "energy": 0.80, "valence": 0.56, "tempo": 116.0, "acousticness": 0.10, "instrumentalness": 0.45, "speechiness": 0.04, "loudness": -6.2, "popularity": 64}
}


def decrypt_media_url(encrypted_media_url: str, quality: str = "320kbps") -> Optional[str]:
    """
    Decrypts JioSaavn encrypted media url using DES (key: 38346591, ECB, PKCS7).
    Returns the direct CDN streaming audio link for the requested quality.
    """
    if not encrypted_media_url:
        return None
    try:
        from cryptography.hazmat.decrepit.ciphers.algorithms import TripleDES
        from cryptography.hazmat.primitives.ciphers import Cipher, modes
        from cryptography.hazmat.primitives import padding

        ciphertext = base64.b64decode(encrypted_media_url)
        cipher = Cipher(TripleDES(DES_KEY * 3), modes.ECB())
        decryptor = cipher.decryptor()
        decrypted_padded = decryptor.update(ciphertext) + decryptor.finalize()
        
        unpadder = padding.PKCS7(64).unpadder()
        decrypted = unpadder.update(decrypted_padded) + unpadder.finalize()
        url_str = decrypted.decode("utf-8")

        # Replace _96 with requested bitrate
        quality_map = {
            "320kbps": "_320.mp4",
            "160kbps": "_160.mp4",
            "96kbps": "_96.mp4",
            "48kbps": "_48.mp4",
            "12kbps": "_12.mp4"
        }
        target_suffix = quality_map.get(quality, "_320.mp4")
        stream_url = re.sub(r'_[0-9]+\.mp4$', target_suffix, url_str)
        if not stream_url.endswith(".mp4"):
            stream_url = stream_url.replace("_96", quality_map.get(quality, "_320").replace(".mp4", ""))
        return stream_url
    except Exception as e:
        logger.debug(f"Failed to decrypt media url: {e}")
        return None


def format_image_url(image_url: str, size: str = "500x500") -> str:
    """Formats JioSaavn image URL to high-resolution (500x500, 150x150, 50x50)."""
    if not image_url:
        return "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=500&auto=format&fit=crop&q=80"
    url = re.sub(r'150x150|50x50', size, image_url)
    if url.startswith("http://"):
        url = "https://" + url[7:]
    return url


def clean_text(text: Optional[str]) -> str:
    """Unescapes HTML entities and trims whitespace."""
    if not text:
        return ""
    return html.unescape(text).strip()


class OpenSpotClient:
    """Client for OpenSpot / JioSaavn music catalog and direct audio streaming."""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.headers = {
            "User-Agent": USER_AGENTS[0],
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
        }

    def _fetch_api(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Makes an HTTP GET request to JioSaavn API."""
        query_params = {
            "__call": endpoint,
            "_format": "json",
            "_marker": "0",
            "api_version": "4",
            "ctx": "web6dot0",
            **params
        }
        url = f"{JIO_SAAVN_BASE_URL}?{urllib.parse.urlencode(query_params)}"
        req = urllib.request.Request(url, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                content = response.read().decode("utf-8", errors="ignore")
                return json.loads(content)
        except Exception as e:
            logger.warning(f"OpenSpot API request failed for {endpoint}: {e}")
            return None

    def search_songs(self, query: str, limit: int = 20, p: int = 1) -> List[Dict[str, Any]]:
        """Searches songs with full audio streaming links, covers, and parsed metadata."""
        data = self._fetch_api("search.getResults", {"q": query, "p": p, "n": limit})
        if not data:
            return []

        results = data.get("results", [])
        parsed_tracks = []
        for r in results:
            parsed = self._parse_song_item(r)
            if parsed:
                parsed_tracks.append(parsed)
        return parsed_tracks

    def search_autocomplete(self, query: str) -> Dict[str, Any]:
        """Provides quick instant search across songs, albums, and artists."""
        data = self._fetch_api("autocomplete.get", {"query": query})
        if not data:
            return {"songs": [], "albums": [], "artists": []}

        songs_raw = data.get("songs", {}).get("data", [])
        parsed_songs = []
        for s in songs_raw:
            more = s.get("more_info", {})
            enc_url = more.get("encrypted_media_url", "")
            stream_url = decrypt_media_url(enc_url, "320kbps") if enc_url else None
            
            parsed_songs.append({
                "song_id": f"os_{s.get('id', '')}",
                "raw_id": s.get("id", ""),
                "title": clean_text(s.get("title")),
                "artist_name": clean_text(more.get("singers") or more.get("primary_artists") or s.get("description", "")),
                "album": clean_text(more.get("album", "")),
                "image_url": format_image_url(s.get("image", ""), "500x500"),
                "audio_url": stream_url,
                "encrypted_media_url": enc_url
            })

        return {
            "songs": parsed_songs,
            "albums": data.get("albums", {}).get("data", []),
            "artists": data.get("artists", {}).get("data", [])
        }

    def get_song_details(self, song_id: str) -> Optional[Dict[str, Any]]:
        """Gets detailed metadata for a single song."""
        raw_id = song_id.replace("os_", "").replace("trk_", "")
        data = self._fetch_api("song.getDetails", {"pids": raw_id})
        if not data or raw_id not in data:
            return None
        return self._parse_song_item(data[raw_id])

    def get_top_charts(self, genre: str = "Pop", limit: int = 20) -> List[Dict[str, Any]]:
        """Fetches top trending songs for a given genre or category."""
        query_map = {
            "Pop": "Top Pop Hits",
            "Hip-Hop": "Top Hip Hop Hits",
            "EDM / Electronic": "EDM Top Hits",
            "Rock": "Classic Rock Hits",
            "R&B / Soul": "R&B Hits",
            "Lo-Fi / Chillhop": "Lofi Chill Beats",
            "Jazz": "Jazz Classics",
            "Classical": "Best Classical Masterpieces",
            "Indie / Alternative": "Indie Alternative Hits",
            "Metal": "Heavy Metal Hits",
            "Acoustic / Folk": "Acoustic Hits",
            "Synthwave": "Synthwave Retrowave Hits"
        }
        q = query_map.get(genre, f"Top {genre} Songs")
        return self.search_songs(q, limit=limit)

    def _parse_song_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Transforms raw JioSaavn JSON item into standard OpenSpot/SoundSpace track format."""
        if not item:
            return None

        more_info = item.get("more_info", {})
        enc_url = more_info.get("encrypted_media_url", "")
        audio_320 = decrypt_media_url(enc_url, "320kbps")
        audio_160 = decrypt_media_url(enc_url, "160kbps")

        raw_id = str(item.get("id", ""))
        title = clean_text(item.get("title") or item.get("song"))
        artists = clean_text(
            more_info.get("singers") or 
            more_info.get("primary_artists") or 
            more_info.get("music") or 
            item.get("subtitle", "")
        )
        album = clean_text(more_info.get("album", "") or item.get("album", ""))
        img_url = format_image_url(item.get("image", ""), "500x500")
        
        duration_sec = 0
        try:
            duration_sec = int(more_info.get("duration", 0) or item.get("duration", 0))
        except (ValueError, TypeError):
            duration_sec = 210

        duration_ms = duration_sec * 1000 if duration_sec > 0 else 210000
        mins = duration_sec // 60
        secs = duration_sec % 60
        duration_formatted = f"{mins}:{secs:02d}"

        # Detect or assign genre
        genre = self._infer_genre(title, artists, album, more_info.get("language", ""))

        # Estimate calibrated continuous acoustic features
        acoustic = self.estimate_acoustic_features(
            title=title,
            artist=artists,
            genre=genre,
            duration_ms=duration_ms,
            explicit=more_info.get("explicit_content") == "1"
        )

        return {
            "song_id": f"os_{raw_id}" if not raw_id.startswith("os_") else raw_id,
            "raw_id": raw_id,
            "title": title,
            "artist_name": artists,
            "album": album or title,
            "genre": genre,
            "image_url": img_url,
            "audio_url": audio_320 or audio_160,
            "audio_url_320": audio_320,
            "audio_url_160": audio_160,
            "encrypted_media_url": enc_url,
            "duration_ms": duration_ms,
            "song_length_ms": duration_ms,
            "duration_formatted": duration_formatted,
            "language": more_info.get("language", "english"),
            "year": more_info.get("year", "2024"),
            "has_lyrics": more_info.get("has_lyrics") == "true",
            "spotify_url": f"https://open.spotify.com/search/{urllib.parse.quote(f'{title} {artists}')}",
            **acoustic
        }

    def _infer_genre(self, title: str, artists: str, album: str, language: str) -> str:
        """Infers musical genre from artist and title keywords."""
        combined = f"{title} {artists} {album}".lower()
        if any(k in combined for k in ["lo-fi", "lofi", "chillhop", "study", "nujabes", "idealism", "saib"]):
            return "Lo-Fi / Chillhop"
        if any(k in combined for k in ["calvin harris", "avicii", "daft punk", "marshmello", "garrix", "kygo", "tiesto", "david guetta", "remix", "club", "edm", "electro"]):
            return "EDM / Electronic"
        if any(k in combined for k in ["kendrick", "drake", "travis scott", "j. cole", "rap", "hip hop", "eminem", "future", "cardi", "21 savage"]):
            return "Hip-Hop"
        if any(k in combined for k in ["queen", "arctic monkeys", "rolling stones", "foo fighters", "coldplay", "nirvana", "rock", "rhcp", "pink floyd"]):
            return "Rock"
        if any(k in combined for k in ["sza", "frank ocean", "giveon", "daniel caesar", "r&b", "soul", "weeknd", "bryson tiller"]):
            return "R&B / Soul"
        if any(k in combined for k in ["miles davis", "coltrane", "jazz", "norah jones", "chet baker", "bill evans"]):
            return "Jazz"
        if any(k in combined for k in ["beethoven", "mozart", "bach", "chopin", "debussy", "orchestra", "symphony", "classical", "piano"]):
            return "Classical"
        if any(k in combined for k in ["metallica", "slipknot", "iron maiden", "rammstein", "metal", "heavy metal", "system of a down"]):
            return "Metal"
        if any(k in combined for k in ["tame impala", "the 1975", "phoebe bridgers", "indie", "alternative", "vampire weekend", "beach house"]):
            return "Indie / Alternative"
        if any(k in combined for k in ["bon iver", "lumineers", "acoustic", "folk", "sufjan stevens", "vance joy", "ed sheeran"]):
            return "Acoustic / Folk"
        if any(k in combined for k in ["synthwave", "kavinsky", "gunship", "the midnight", "retrowave", "timecop"]):
            return "Synthwave"
        return "Pop"

    def estimate_acoustic_features(
        self,
        title: str,
        artist: str,
        genre: str,
        duration_ms: int = 210000,
        explicit: bool = False
    ) -> Dict[str, float]:
        """
        Generates continuous acoustic feature values calibrated for the song
        to integrate smoothly with Two-Tower & Wolpertinger RL models.
        """
        base = GENRE_DEFAULTS.get(genre, GENRE_DEFAULTS["Pop"])
        seed = abs(hash(f"{title}_{artist}")) % (2**31 - 1)
        rng = np.random.RandomState(seed)

        dance = float(np.clip(rng.normal(base["danceability"], 0.08), 0.1, 0.98))
        energy = float(np.clip(rng.normal(base["energy"], 0.08), 0.1, 0.98))
        valence = float(np.clip(rng.normal(base["valence"], 0.09), 0.1, 0.98))
        tempo = float(np.clip(rng.normal(base["tempo"], 8.0), 60.0, 195.0))
        acoustic = float(np.clip(rng.normal(base["acousticness"], 0.08), 0.01, 0.99))
        instrumental = float(np.clip(rng.normal(base["instrumentalness"], 0.1), 0.0, 0.99))
        speech = float(np.clip(rng.normal(base["speechiness"], 0.03) + (0.1 if explicit else 0.0), 0.02, 0.85))
        loudness = float(np.clip(rng.normal(base["loudness"], 2.0), -28.0, -1.0))
        popularity = int(np.clip(rng.normal(base["popularity"], 10), 20, 99))

        return {
            "danceability": round(dance, 4),
            "energy": round(energy, 4),
            "valence": round(valence, 4),
            "tempo": round(tempo, 2),
            "acousticness": round(acoustic, 4),
            "instrumentalness": round(instrumental, 6),
            "speechiness": round(speech, 4),
            "loudness": round(loudness, 2),
            "popularity": popularity
        }


# Singleton instance
openspot_client = OpenSpotClient()
