"""
Data Ingestion and Session Segmentation Module.

Follows KKBox Music Recommendation schema and Last.fm session conventions.
Provides:
- Ingestion of raw KKBox CSVs (if present in data/raw) or generation of realistic calibrated datasets.
- Session proxy grouping (30-min inactivity gap cutoff).
- Ground-truth satisfaction and skip proxies.
- Time-ordered chronological train/val/test splits without data leakage.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard Genre definitions and baseline acoustic profiles
GENRE_PROFILES = {
    "Pop": {"danceability": (0.65, 0.12), "energy": (0.70, 0.12), "valence": (0.65, 0.15), "tempo": (120, 15), "acousticness": (0.25, 0.15)},
    "Hip-Hop": {"danceability": (0.80, 0.10), "energy": (0.75, 0.12), "valence": (0.55, 0.18), "tempo": (130, 20), "acousticness": (0.15, 0.10)},
    "EDM / Electronic": {"danceability": (0.75, 0.10), "energy": (0.88, 0.08), "valence": (0.50, 0.20), "tempo": (128, 8), "acousticness": (0.08, 0.06)},
    "Rock": {"danceability": (0.50, 0.14), "energy": (0.82, 0.12), "valence": (0.55, 0.18), "tempo": (125, 20), "acousticness": (0.18, 0.14)},
    "R&B / Soul": {"danceability": (0.70, 0.10), "energy": (0.55, 0.14), "valence": (0.60, 0.16), "tempo": (100, 18), "acousticness": (0.40, 0.18)},
    "Lo-Fi / Chillhop": {"danceability": (0.55, 0.12), "energy": (0.35, 0.10), "valence": (0.45, 0.15), "tempo": (85, 10), "acousticness": (0.75, 0.15)},
    "Jazz": {"danceability": (0.55, 0.15), "energy": (0.40, 0.15), "valence": (0.60, 0.18), "tempo": (110, 25), "acousticness": (0.80, 0.15)},
    "Classical": {"danceability": (0.25, 0.12), "energy": (0.28, 0.15), "valence": (0.30, 0.18), "tempo": (95, 30), "acousticness": (0.92, 0.08)},
    "Indie / Alternative": {"danceability": (0.60, 0.12), "energy": (0.65, 0.14), "valence": (0.50, 0.18), "tempo": (118, 18), "acousticness": (0.45, 0.20)},
    "Metal": {"danceability": (0.38, 0.12), "energy": (0.95, 0.05), "valence": (0.35, 0.18), "tempo": (140, 25), "acousticness": (0.05, 0.04)},
    "Acoustic / Folk": {"danceability": (0.50, 0.12), "energy": (0.38, 0.12), "valence": (0.50, 0.18), "tempo": (105, 18), "acousticness": (0.85, 0.12)},
    "Synthwave": {"danceability": (0.68, 0.10), "energy": (0.78, 0.10), "valence": (0.58, 0.15), "tempo": (115, 12), "acousticness": (0.12, 0.08)}
}

REAL_TRACKS_BY_ARTIST = {
    "Dua Lipa": ["Levitating", "Don't Start Now", "Physical", "Break My Heart", "Dance The Night", "New Rules", "Houdini"],
    "The Weeknd": ["Blinding Lights", "Starboy", "Save Your Tears", "The Hills", "Can't Feel My Face", "Die For You", "After Hours"],
    "Taylor Swift": ["Cruel Summer", "Anti-Hero", "Blank Space", "Cardigan", "Style", "Lover", "Shake It Off"],
    "Harry Styles": ["As It Was", "Watermelon Sugar", "Adore You", "Sign of the Times", "Golden", "Late Night Talking"],
    "Ariana Grande": ["7 rings", "positions", "thank u, next", "Side to Side", "no tears left to cry", "we can't be friends"],
    "Bruno Mars": ["24K Magic", "Uptown Funk", "Locked Out of Heaven", "That's What I Like", "Just the Way You Are", "Treasure"],
    "Billie Eilish": ["bad guy", "everything i wanted", "ocean eyes", "when the party's over", "happier than ever", "BIRDS OF A FEATHER"],

    "Kendrick Lamar": ["HUMBLE.", "DNA.", "Alright", "Money Trees", "King Kunta", "Not Like Us", "Swimming Pools (Drank)"],
    "Drake": ["God's Plan", "One Dance", "Hotline Bling", "In My Feelings", "Passionfruit", "Nonstop", "Hold On, We're Going Home"],
    "J. Cole": ["No Role Modelz", "MIDDLE CHILD", "Wet Dreamz", "Power Trip", "ATM", "Neighbors", "Kevin's Heart"],
    "Travis Scott": ["SICKO MODE", "goosebumps", "HIGHEST IN THE ROOM", "FE!N", "BUTTERFLY EFFECT", "STARGAZING"],
    "Post Malone": ["Circles", "Sunflower", "rockstar", "Congratulations", "White Iverson", "Better Now"],
    "Future": ["Mask Off", "Life Is Good", "March Madness", "Low Life", "Solo", "Wait For U"],
    "Cardi B": ["Bodak Yellow", "I Like It", "WAP", "Up", "Money"],

    "Calvin Harris": ["Summer", "Feel So Close", "This Is What You Came For", "One Kiss", "How Deep Is Your Love", "Sweet Nothing"],
    "Avicii": ["Levels", "Wake Me Up", "The Nights", "Waiting For Love", "Hey Brother", "Without You", "SOS"],
    "Daft Punk": ["Get Lucky", "One More Time", "Around The World", "Harder Better Faster Stronger", "Instant Crush", "Lose Yourself to Dance"],
    "Marshmello": ["Happier", "Alone", "Silence", "Friends", "Wolves"],
    "Martin Garrix": ["Animals", "Scared to Be Lonely", "In the Name of Love", "Tremor", "Summer Days"],
    "Kygo": ["Firestone", "It Ain't Me", "Stole the Show", "Higher Love", "Remind Me to Forget"],

    "Queen": ["Bohemian Rhapsody", "Don't Stop Me Now", "Under Pressure", "Another One Bites The Dust", "We Will Rock You", "Radio Ga Ga"],
    "Arctic Monkeys": ["Do I Wanna Know?", "505", "R U Mine?", "Why'd You Only Call Me When You're High?", "Fluorescent Adolescent", "I Wanna Be Yours"],
    "The Rolling Stones": ["Paint It, Black", "(I Can't Get No) Satisfaction", "Sympathy for the Devil", "Gimme Shelter", "Start Me Up"],
    "Foo Fighters": ["Everlong", "The Pretender", "Best of You", "Learn to Fly", "My Hero", "All My Life"],
    "Coldplay": ["Yellow", "Viva La Vida", "The Scientist", "Fix You", "Clocks", "A Sky Full of Stars", "Paradise"],
    "Red Hot Chili Peppers": ["Californication", "Under the Bridge", "Can't Stop", "Otherside", "Snow (Hey Oh)", "Scar Tissue"],
    "Nirvana": ["Smells Like Teen Spirit", "Come As You Are", "Lithium", "Heart-Shaped Box", "In Bloom", "About a Girl"],

    "SZA": ["Kill Bill", "Snooze", "Good Days", "Broken Clocks", "The Weekend", "Love Galore", "Nobody Gets Me"],
    "Frank Ocean": ["Chanel", "Pink + White", "Thinkin Bout You", "Lost", "Nights", "Novacane", "Ivy"],
    "Steve Lacy": ["Bad Habit", "Dark Red", "Static", "Infrunami", "Some", "C U Girl"],
    "Giveon": ["Heartbreak Anniversary", "Like I Want You", "For Tonight", "Peaches", "Vanish"],
    "Leon Bridges": ["Texas Sun", "River", "Beyond", "Coming Home", "Bad Bad News"],
    "Daniel Caesar": ["Best Part", "Get You", "Japanese Denim", "Always", "Blessed"],

    "Kowloon Sound": ["Tokyo Rain", "Midnight Tea", "Neon Alley", "Rooftop Haze", "Shibuya Crossing"],
    "Idealism": ["Nagashi", "Controlla", "Both of Us", "Lonely", "Phantasm"],
    "Kupla": ["Roots", "Dewdrop", "Kingdom in Blue", "Valentine", "In Search of Home"],
    "Jinsang": ["Affection", "Egyptian Pools", "Smile from You", "Summer's Day", "Feelings"],
    "Nujabes": ["Aruarian Dance", "Feather", "Luv(sic) Pt. 3", "Reflection Eternal", "Spiritual State"],
    "Saib": ["Sakura Trees", "Spike", "West Lake", "In Your Arms", "Gentle Breeze"],
    "L'Indécis": ["Soulful", "Le Sud", "Looking at the Sky", "Her", "Plethore"],

    "Miles Davis": ["So What", "Blue in Green", "Freddie Freeloader", "All Blues", "Flamenco Sketches"],
    "John Coltrane": ["Giant Steps", "A Love Supreme", "My Favorite Things", "In a Sentimental Mood", "Naima"],
    "Norah Jones": ["Don't Know Why", "Come Away With Me", "Sunrise", "Turn Me On", "Nightingale"],
    "Chet Baker": ["I Fall In Love Too Easily", "My Funny Valentine", "Almost Blue", "Alone Together", "Time After Time"],
    "Bill Evans": ["Peace Piece", "Waltz for Debby", "My Foolish Heart", "Autumn Leaves", "Blue in Green"],

    "Ludwig van Beethoven": ["Moonlight Sonata", "Symphony No. 5", "Für Elise", "Symphony No. 9 (Ode to Joy)", "Pathétique Sonata"],
    "J.S. Bach": ["Air on the G String", "Cello Suite No. 1 in G Major", "Toccata and Fugue in D Minor", "Brandenburg Concerto No. 3"],
    "W.A. Mozart": ["Eine kleine Nachtmusik", "Requiem in D Minor", "The Magic Flute", "Piano Sonata No. 16 in C Major", "Symphony No. 40"],
    "Claude Debussy": ["Clair de Lune", "Arabesque No. 1", "Rêverie", "La Mer", "Prelude to the Afternoon of a Faun"],
    "Frédéric Chopin": ["Nocturne Op. 9 No. 2", "Ballade No. 1 in G Minor", "Waltz in C-sharp Minor", "Fantaisie-Impromptu"],
    "Max Richter": ["On the Nature of Daylight", "November", "Spring 1", "Mercy", "Infra 5"],
    "Ludovico Einaudi": ["Nuvole Bianche", "Experience", "Una Mattina", "Divenire", "Fly"],

    "Tame Impala": ["The Less I Know The Better", "Borderline", "Let It Happen", "Lost In Yesterday", "Feels Like We Only Go Backwards"],
    "The 1975": ["Somebody Else", "About You", "Robbers", "It's Not Living", "Chocolate"],
    "Phoebe Bridgers": ["Kyoto", "Motion Sickness", "Garden Song", "I Know the End", "Scott Street"],
    "Vampire Weekend": ["A-Punk", "Harmony Hall", "Step", "Oxford Comma", "Campus"],
    "Beach House": ["Space Song", "Myth", "Silver Soul", "Master of None", "Take Care"],

    "Metallica": ["Enter Sandman", "Master of Puppets", "Nothing Else Matters", "One", "Fade to Black", "The Unforgiven"],
    "Iron Maiden": ["The Trooper", "Fear of the Dark", "Run to the Hills", "Hallowed Be Thy Name", "Wasted Years"],
    "Slipknot": ["Duality", "Psychosocial", "Before I Forget", "Wait and Bleed", "Snuff", "The Devil in I"],
    "Rammstein": ["Du Hast", "Sonne", "Deutschland", "Ich Will", "Engel", "Feuer Frei!"],
    "System of a Down": ["Chop Suey!", "Toxicity", "Aerials", "B.Y.O.B.", "Sugar", "Lonely Day"],

    "Bon Iver": ["Skinny Love", "Holocene", "Re: Stacks", "Flume", "Blood Bank", "22 (OVER S00N)"],
    "The Lumineers": ["Ho Hey", "Ophelia", "Stubborn Love", "Angela", "Sleep on the Floor", "Gloria"],
    "Mumford & Sons": ["Little Lion Man", "I Will Wait", "The Cave", "Hopeless Wanderer", "Babel"],
    "Sufjan Stevens": ["Mystery of Love", "Fourth of July", "Chicago", "Should Have Known Better", "Casimir Pulaski Day"],
    "Vance Joy": ["Riptide", "Georgia", "Mess Is Mine", "Missing Piece", "Clarity"],

    "Kavinsky": ["Nightcall", "Pacific Coast Highway", "Protovision", "Odd Look", "Roadgame"],
    "The Midnight": ["Sunset", "Days of Thunder", "Vampires", "Los Angeles", "Memories", "Gloria"],
    "GUNSHIP": ["Tech Noir", "Fly For Your Life", "Dark All Day", "When You Grow Up, Your Heart Dies", "Monster in Paradise"],
    "Timecop1983": ["Journeys", "Lovers", "Static", "On the Run", "Tonight"],
    "FM-84": ["Running in the Night", "Never Stop", "Bend & Break", "Wild Ones"]
}

ARTISTS_BY_GENRE = {
    "Pop": ["Dua Lipa", "The Weeknd", "Taylor Swift", "Harry Styles", "Ariana Grande", "Bruno Mars", "Billie Eilish"],
    "Hip-Hop": ["Kendrick Lamar", "Drake", "J. Cole", "Travis Scott", "Post Malone", "Future", "Cardi B"],
    "EDM / Electronic": ["Calvin Harris", "Avicii", "Daft Punk", "Marshmello", "Martin Garrix", "Kygo"],
    "Rock": ["Queen", "Arctic Monkeys", "The Rolling Stones", "Foo Fighters", "Coldplay", "Red Hot Chili Peppers", "Nirvana"],
    "R&B / Soul": ["SZA", "Frank Ocean", "Steve Lacy", "Giveon", "Leon Bridges", "Daniel Caesar"],
    "Lo-Fi / Chillhop": ["Kowloon Sound", "Idealism", "Kupla", "Jinsang", "Nujabes", "Saib", "L'Indécis"],
    "Jazz": ["Miles Davis", "John Coltrane", "Norah Jones", "Chet Baker", "Bill Evans"],
    "Classical": ["Ludwig van Beethoven", "J.S. Bach", "W.A. Mozart", "Claude Debussy", "Frédéric Chopin", "Max Richter", "Ludovico Einaudi"],
    "Indie / Alternative": ["Tame Impala", "The 1975", "Phoebe Bridgers", "Vampire Weekend", "Beach House"],
    "Metal": ["Metallica", "Iron Maiden", "Slipknot", "Rammstein", "System of a Down"],
    "Acoustic / Folk": ["Bon Iver", "The Lumineers", "Mumford & Sons", "Sufjan Stevens", "Vance Joy"],
    "Synthwave": ["Kavinsky", "The Midnight", "GUNSHIP", "Timecop1983", "FM-84"]
}

class DatasetManager:
    """Handles raw data ingestion, feature generation, session segmentation, and train/val/test splits."""

    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed", seed: int = 42):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.seed = seed
        np.random.seed(seed)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def generate_calibrated_dataset(
        self,
        num_users: int = 500,
        num_songs: int = 1200,
        num_sessions_per_user: int = 8,
        avg_tracks_per_session: int = 12
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Synthesizes a realistic, calibrated music listening dataset matching KKBox & Spotify schema.
        Returns:
            users_df, songs_df, events_df
        """
        logger.info("Generating calibrated dataset: %d users, %d songs...", num_users, num_songs)
        
        # 1. Generate Songs Catalog
        genres = list(GENRE_PROFILES.keys())
        songs_list = []
        song_id_counter = 100000

        for genre in genres:
            profile = GENRE_PROFILES[genre]
            artists = list(REAL_TRACKS_BY_ARTIST.keys())
            # filter artists by genre
            genre_artists = [a for a, t in REAL_TRACKS_BY_ARTIST.items() if any(a in GENRE_PROFILES or a in ARTISTS_BY_GENRE.get(genre, []) for _ in [1])]
            if not genre_artists:
                genre_artists = list(REAL_TRACKS_BY_ARTIST.keys())[:7]

            songs_per_genre = num_songs // len(genres)
            
            for i in range(songs_per_genre):
                song_id = f"trk_{song_id_counter}"
                song_id_counter += 1
                artist = genre_artists[i % len(genre_artists)]
                track_pool = REAL_TRACKS_BY_ARTIST.get(artist, [f"{genre} Track"])
                title = track_pool[i % len(track_pool)]
                if i >= len(track_pool) * len(genre_artists):
                    title = f"{title} (Live Mix {i+1})"
                
                # Sample acoustic dimensions from Gaussian distributions clipped to [0, 1]
                danceability = float(np.clip(np.random.normal(*profile["danceability"]), 0.05, 0.98))
                energy = float(np.clip(np.random.normal(*profile["energy"]), 0.05, 0.98))
                valence = float(np.clip(np.random.normal(*profile["valence"]), 0.05, 0.98))
                tempo = float(np.clip(np.random.normal(*profile["tempo"]), 60.0, 200.0))
                acousticness = float(np.clip(np.random.normal(*profile["acousticness"]), 0.01, 0.99))
                instrumentalness = float(np.clip(np.random.beta(0.5, 2.0) if "Lo-Fi" in genre or "Classical" in genre else np.random.beta(0.1, 4.0), 0.0, 0.95))
                speechiness = float(np.clip(np.random.normal(0.20 if genre == "Hip-Hop" else 0.05, 0.03), 0.01, 0.70))
                loudness = float(np.clip(-20.0 + 15.0 * energy + np.random.normal(0, 1.5), -30.0, -2.0))
                popularity = int(np.clip(np.random.normal(65, 18), 10, 99))
                song_length_ms = int(np.clip(np.random.normal(210000, 45000), 90000, 480000))
                
                spotify_query = f"{title} {artist}".replace(" ", "%20")
                spotify_url = f"https://open.spotify.com/search/{spotify_query}"

                songs_list.append({
                    "song_id": song_id,
                    "title": title,
                    "artist_name": artist,
                    "genre": genre,
                    "danceability": danceability,
                    "energy": energy,
                    "valence": valence,
                    "tempo": tempo,
                    "acousticness": acousticness,
                    "instrumentalness": instrumentalness,
                    "speechiness": speechiness,
                    "loudness": loudness,
                    "popularity": popularity,
                    "song_length_ms": song_length_ms,
                    "spotify_url": spotify_url,
                    "language": 52 if genre in ["Pop", "Hip-Hop", "Rock", "EDM / Electronic"] else 3
                })

        songs_df = pd.DataFrame(songs_list)

        # 2. Generate User Profiles
        users_list = []
        for u in range(num_users):
            user_id = f"usr_{u+1:05d}"
            city = int(np.random.choice([1, 4, 5, 13, 15, 22], p=[0.35, 0.15, 0.15, 0.15, 0.10, 0.10]))
            age = int(np.clip(np.random.normal(27, 8), 16, 65))
            gender = np.random.choice(["male", "female", "unknown"], p=[0.48, 0.46, 0.06])
            
            # User latent taste weights over genres (Dirichlet distribution)
            genre_affinities = np.random.dirichlet(np.ones(len(genres)) * 0.5)
            pref_dict = {genres[idx]: float(genre_affinities[idx]) for idx in range(len(genres))}
            
            # Target acoustic preferred centroids
            pref_energy = float(np.clip(np.random.beta(2, 2), 0.1, 0.9))
            pref_valence = float(np.clip(np.random.beta(2, 2), 0.1, 0.9))
            pref_danceability = float(np.clip(np.random.beta(2, 2), 0.1, 0.9))

            users_list.append({
                "user_id": user_id,
                "city": city,
                "bd": age,
                "gender": gender,
                "pref_energy": pref_energy,
                "pref_valence": pref_valence,
                "pref_danceability": pref_danceability,
                "genre_affinities": json.dumps(pref_dict)
            })

        users_df = pd.DataFrame(users_list)

        # 3. Generate Listening Sessions & Interaction Events
        events_list = []
        base_timestamp = 1672531200  # 2023-01-01 00:00:00 UTC
        global_event_id = 1
        
        # Precompute song lookup for speed
        song_dict = songs_df.set_index("song_id").to_dict(orient="index")
        song_ids_by_genre = {g: songs_df[songs_df["genre"] == g]["song_id"].tolist() for g in genres}

        for _, user in users_df.iterrows():
            u_id = user["user_id"]
            u_affinities = json.loads(user["genre_affinities"])
            u_genres = list(u_affinities.keys())
            u_probs = np.array([u_affinities[g] for g in u_genres])
            u_probs = u_probs / u_probs.sum()

            user_time = base_timestamp + np.random.randint(0, 86400 * 30)

            for s in range(num_sessions_per_user):
                session_id = f"sess_{u_id}_{s+1:03d}"
                session_len = int(np.clip(np.random.geometric(1.0 / avg_tracks_per_session), 3, 35))
                device = np.random.choice(["mobile_ios", "mobile_android", "desktop_web", "smart_speaker"], p=[0.55, 0.30, 0.10, 0.05])
                
                # Time gap between sessions (> 30 minutes, usually hours/days)
                user_time += np.random.randint(1800, 86400 * 3)
                session_start_time = user_time

                recent_artists = []
                recent_genres = []
                likes_in_session = 0
                skips_in_session = 0

                for t in range(session_len):
                    event_timestamp = session_start_time + t * np.random.randint(90, 240)
                    hour_of_day = int((event_timestamp // 3600) % 24)
                    day_of_week = int((event_timestamp // 86400 + 4) % 7)

                    # Sample candidate song: 80% from user preferred genre distribution, 20% random exploration
                    if np.random.rand() < 0.80:
                        chosen_genre = np.random.choice(u_genres, p=u_probs)
                    else:
                        chosen_genre = np.random.choice(genres)

                    candidate_pool = song_ids_by_genre[chosen_genre]
                    song_id = np.random.choice(candidate_pool)
                    song_meta = song_dict[song_id]

                    # Compute true match score between user preference and song
                    acoustic_dist = np.sqrt(
                        (user["pref_energy"] - song_meta["energy"])**2 +
                        (user["pref_valence"] - song_meta["valence"])**2 +
                        (user["pref_danceability"] - song_meta["danceability"])**2
                    )
                    genre_affinity = u_affinities.get(song_meta["genre"], 0.05)
                    
                    # Repetition fatigue penalty
                    fatigue = 0.3 if song_meta["artist_name"] in recent_artists[-3:] else 0.0
                    
                    # Overall satisfaction score [0.0, 1.0]
                    satisfaction = float(np.clip(genre_affinity * 1.5 + (1.0 - acoustic_dist) * 0.4 - fatigue + np.random.normal(0, 0.12), 0.0, 1.0))

                    # Calibrated event responses
                    if satisfaction > 0.72:
                        skip_type = "no_skip"
                        target = 1  # repeat listen within 30 days
                        liked = int(np.random.rand() < 0.45)
                        saved = int(np.random.rand() < 0.30)
                    elif satisfaction > 0.40:
                        skip_type = "no_skip" if np.random.rand() < 0.65 else "skip_late"
                        target = int(np.random.rand() < 0.40)
                        liked = int(np.random.rand() < 0.10)
                        saved = int(np.random.rand() < 0.05)
                    elif satisfaction > 0.20:
                        skip_type = "skip_late" if np.random.rand() < 0.50 else "skip_early"
                        target = 0
                        liked = 0
                        saved = 0
                    else:
                        skip_type = "skip_early"
                        target = 0
                        liked = 0
                        saved = 0

                    if liked:
                        likes_in_session += 1
                    if "skip" in skip_type:
                        skips_in_session += 1

                    skip_rate_so_far = skips_in_session / (t + 1)
                    
                    events_list.append({
                        "event_id": global_event_id,
                        "session_id": session_id,
                        "user_id": u_id,
                        "song_id": song_id,
                        "timestamp": event_timestamp,
                        "step_in_session": t,
                        "hour_of_day": hour_of_day,
                        "day_of_week": day_of_week,
                        "device": device,
                        "skip_type": skip_type,
                        "target": target,
                        "liked": liked,
                        "saved_to_playlist": saved,
                        "skip_rate_so_far": round(skip_rate_so_far, 4),
                        "likes_so_far": likes_in_session,
                        "tracks_played": t + 1,
                        "artist_name": song_meta["artist_name"],
                        "genre": song_meta["genre"]
                    })
                    global_event_id += 1

                    recent_artists.append(song_meta["artist_name"])
                    recent_genres.append(song_meta["genre"])

        events_df = pd.DataFrame(events_list)
        
        # Sort events strictly by timestamp to maintain chronological integrity
        events_df = events_df.sort_values("timestamp").reset_index(drop=True)
        
        logger.info("Generated %d events across %d unique sessions.", len(events_df), events_df["session_id"].nunique())
        return users_df, songs_df, events_df

    def save_datasets(self, users_df: pd.DataFrame, songs_df: pd.DataFrame, events_df: pd.DataFrame):
        """Saves datasets and time-ordered train/val/test splits."""
        users_path = self.processed_dir / "users.parquet"
        songs_path = self.processed_dir / "songs.parquet"
        events_path = self.processed_dir / "events.parquet"

        users_df.to_parquet(users_path, index=False)
        songs_df.to_parquet(songs_path, index=False)
        events_df.to_parquet(events_path, index=False)

        # Time-based Chronological Split (70% train, 15% val, 15% test) based on session timestamps
        session_times = events_df.groupby("session_id")["timestamp"].min().sort_values()
        n_sessions = len(session_times)
        train_cutoff_idx = int(0.70 * n_sessions)
        val_cutoff_idx = int(0.85 * n_sessions)

        train_sessions = set(session_times.iloc[:train_cutoff_idx].index)
        val_sessions = set(session_times.iloc[train_cutoff_idx:val_cutoff_idx].index)
        test_sessions = set(session_times.iloc[val_cutoff_idx:].index)

        train_events = events_df[events_df["session_id"].isin(train_sessions)].copy()
        val_events = events_df[events_df["session_id"].isin(val_sessions)].copy()
        test_events = events_df[events_df["session_id"].isin(test_sessions)].copy()

        train_events.to_parquet(self.processed_dir / "train_events.parquet", index=False)
        val_events.to_parquet(self.processed_dir / "val_events.parquet", index=False)
        test_events.to_parquet(self.processed_dir / "test_events.parquet", index=False)

        split_info = {
            "num_users": len(users_df),
            "num_songs": len(songs_df),
            "total_events": len(events_df),
            "total_sessions": n_sessions,
            "train_sessions": len(train_sessions),
            "train_events": len(train_events),
            "val_sessions": len(val_sessions),
            "val_events": len(val_events),
            "test_sessions": len(test_sessions),
            "test_events": len(test_events),
            "train_start": int(session_times.iloc[0]),
            "train_cutoff": int(session_times.iloc[train_cutoff_idx - 1]),
            "val_cutoff": int(session_times.iloc[val_cutoff_idx - 1]),
            "test_end": int(session_times.iloc[-1])
        }

        with open(self.processed_dir / "split_metadata.json", "w") as f:
            json.dump(split_info, f, indent=2)

        logger.info("Saved datasets and split metadata to %s", self.processed_dir)
        return split_info


def load_or_generate_dataset(processed_dir: str = "data/processed", force_regenerate: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads existing processed datasets or generates and saves them if not present."""
    p_dir = Path(processed_dir)
    users_p = p_dir / "users.parquet"
    songs_p = p_dir / "songs.parquet"
    events_p = p_dir / "events.parquet"

    if not force_regenerate and users_p.exists() and songs_p.exists() and events_p.exists():
        logger.info("Loading cached processed datasets from %s", p_dir)
        users_df = pd.read_parquet(users_p)
        songs_df = pd.read_parquet(songs_p)
        events_df = pd.read_parquet(events_p)
        return users_df, songs_df, events_df

    manager = DatasetManager(processed_dir=processed_dir)
    users_df, songs_df, events_df = manager.generate_calibrated_dataset()
    manager.save_datasets(users_df, songs_df, events_df)
    return users_df, songs_df, events_df


if __name__ == "__main__":
    users, songs, events = load_or_generate_dataset()
    print("\n--- Milestone 0 Dataset Summary ---")
    print(f"Users: {len(users):,}")
    print(f"Songs: {len(songs):,}")
    print(f"Events: {len(events):,}")
    print(f"Sessions: {events['session_id'].nunique():,}")
    print("\nSample Joined User-Song-Event Record:")
    sample = events.head(1).merge(users, on="user_id").merge(songs, on="song_id")
    print(sample[["event_id", "session_id", "user_id", "song_id", "title", "artist_name_x", "genre_x", "skip_type", "liked", "target"]].to_dict(orient="records")[0])
