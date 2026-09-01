/**
 * BehaviorCollector — Real-Time User Behavior Tracking Engine.
 *
 * Tracks and reports:
 * - Listening time (ms accurately measured via AudioContext/intervals)
 * - Replay detection (backward seek past 80% of song)
 * - Search behavior (queries + clicked song IDs)
 * - Playlist saves
 * - Context detection (time-of-day based activity inference)
 *
 * All events are batched and sent to /api/behavior/log and fed into
 * /api/recommend/hybrid calls as extra signals.
 */

class BehaviorCollector {
  constructor() {
    this.userId = 'usr_00001';
    this.currentTrack = null;

    // Listen time tracking
    this.listenStartTime = null;    // Date.now() when play started
    this.totalListenMs = 0;         // Accumulated ms for current song
    this.listenTimerInterval = null;

    // Per-session maps
    this.sessionListenHistoryMs = {};   // {song_id: total_ms}
    this.replaySongIds = new Set();
    this.searchHistory = [];
    this.playlistSavedIds = new Set();
    this.lastPlaybackPct = 0;           // For replay detection (0-100%)

    // Pending flush queue
    this._pendingEvents = [];
    this._flushTimer = null;
  }

  // ────────────────────────────────────────────────────────────────────────────
  // Configuration
  // ────────────────────────────────────────────────────────────────────────────

  setUser(userId) {
    this.userId = userId;
  }

  setCurrentTrack(track) {
    // Flush previous track's listen time before switching
    if (this.currentTrack && this.currentTrack.song_id !== track?.song_id) {
      this._flushListenTime();
    }
    this.currentTrack = track;
    this.totalListenMs = 0;
    this.listenStartTime = null;
    this.lastPlaybackPct = 0;
  }

  // ────────────────────────────────────────────────────────────────────────────
  // Listen Timer Control
  // ────────────────────────────────────────────────────────────────────────────

  onPlaybackStarted() {
    if (this.listenStartTime !== null) return; // Already counting
    this.listenStartTime = Date.now();

    // Update every 5s to accumulate time
    this.listenTimerInterval = setInterval(() => {
      if (this.listenStartTime !== null) {
        const now = Date.now();
        const delta = now - this.listenStartTime;
        this.listenStartTime = now;
        this.totalListenMs += delta;
        if (this.currentTrack) {
          this.sessionListenHistoryMs[this.currentTrack.song_id] =
            (this.sessionListenHistoryMs[this.currentTrack.song_id] || 0) + delta;
        }
      }
    }, 5000);
  }

  onPlaybackPaused() {
    if (this.listenStartTime !== null) {
      const delta = Date.now() - this.listenStartTime;
      this.totalListenMs += delta;
      if (this.currentTrack) {
        this.sessionListenHistoryMs[this.currentTrack.song_id] =
          (this.sessionListenHistoryMs[this.currentTrack.song_id] || 0) + delta;
      }
      this.listenStartTime = null;
    }
    clearInterval(this.listenTimerInterval);
    this.listenTimerInterval = null;
  }

  onPlaybackProgress(currentTimeSec, durationSec) {
    if (!this.currentTrack || durationSec <= 0) return;
    const pct = (currentTimeSec / durationSec) * 100;

    // Replay detection: user seeked backward past 80% threshold
    if (pct < this.lastPlaybackPct - 20 && this.lastPlaybackPct > 80) {
      this._onReplayDetected();
    }
    this.lastPlaybackPct = pct;
  }

  // ────────────────────────────────────────────────────────────────────────────
  // Event Recording
  // ────────────────────────────────────────────────────────────────────────────

  onSongFinished() {
    this._flushListenTime();
    const track = this.currentTrack;
    if (!track) return;
    this._logEvent({
      event_type: 'listen_end',
      song_id: track.song_id,
      listen_ms: this.totalListenMs,
    });
    this.totalListenMs = 0;
  }

  onFeedback(actionType, listenMs = 0) {
    const track = this.currentTrack;
    if (!track) return;

    // Accumulate any in-progress listen time
    this.onPlaybackPaused();
    const totalMs = listenMs || this.totalListenMs;

    this._logEvent({
      event_type: actionType,
      song_id: track.song_id,
      listen_ms: totalMs,
    });
  }

  onPlaylistSave(track) {
    if (!track) return;
    this.playlistSavedIds.add(track.song_id);
    this._logEvent({
      event_type: 'playlist_save',
      song_id: track.song_id,
      listen_ms: this.totalListenMs,
    });
  }

  onSearchClick(query, track) {
    if (!track) return;
    if (query && !this.searchHistory.includes(query)) {
      this.searchHistory.unshift(query);
      this.searchHistory = this.searchHistory.slice(0, 10);
    }
    this._logEvent({
      event_type: 'search_click',
      song_id: track.song_id,
      listen_ms: 0,
      search_query: query,
    });
  }

  // ────────────────────────────────────────────────────────────────────────────
  // Context Detection
  // ────────────────────────────────────────────────────────────────────────────

  detectContext() {
    const hour = new Date().getHours();
    if (hour >= 5 && hour < 10)  return 'morning_focus';
    if (hour >= 10 && hour < 17) return 'afternoon_energy';
    if (hour >= 17 && hour < 21) return 'evening_chill';
    return 'night_study';
  }

  getContextLabel() {
    const labels = {
      morning_focus:    '☕ Morning Focus',
      afternoon_energy: '⚡ Afternoon Energy',
      evening_chill:    '🌆 Evening Chill',
      night_study:      '🌙 Night Study',
      workout:          '🏋️ Workout',
      travel:           '✈️ Travel',
    };
    return labels[this.detectContext()] || '🎵 Listening';
  }

  // ────────────────────────────────────────────────────────────────────────────
  // Hybrid Recommendation Payload Builder
  // ────────────────────────────────────────────────────────────────────────────

  buildHybridPayload(basePayload = {}) {
    return {
      ...basePayload,
      context_type: this.detectContext(),
      listen_history_ms: { ...this.sessionListenHistoryMs },
      replay_song_ids: [...this.replaySongIds],
      search_history: this.searchHistory.slice(0, 5),
      pull_openspot: true,
      explore_slots: 2,
      diversity_window: 3,
    };
  }

  // ────────────────────────────────────────────────────────────────────────────
  // Internal
  // ────────────────────────────────────────────────────────────────────────────

  _onReplayDetected() {
    const track = this.currentTrack;
    if (!track) return;
    if (!this.replaySongIds.has(track.song_id)) {
      this.replaySongIds.add(track.song_id);
      this._logEvent({ event_type: 'replay', song_id: track.song_id, listen_ms: 0 });
    }
  }

  _flushListenTime() {
    if (this.listenStartTime !== null) {
      const delta = Date.now() - this.listenStartTime;
      this.totalListenMs += delta;
      if (this.currentTrack) {
        this.sessionListenHistoryMs[this.currentTrack.song_id] =
          (this.sessionListenHistoryMs[this.currentTrack.song_id] || 0) + delta;
      }
      this.listenStartTime = null;
    }
    clearInterval(this.listenTimerInterval);
    this.listenTimerInterval = null;
  }

  _logEvent(eventData) {
    const track = this.currentTrack;
    if (!eventData.song_id) return;

    const payload = {
      user_id:      this.userId,
      song_id:      eventData.song_id,
      event_type:   eventData.event_type,
      title:        track?.title || '',
      artist_name:  track?.artist_name || '',
      genre:        track?.genre || 'Pop',
      listen_ms:    eventData.listen_ms || 0,
      duration_ms:  track?.duration_ms || track?.song_length_ms || 210000,
      energy:       track?.energy ?? 0.65,
      valence:      track?.valence ?? 0.60,
      danceability: track?.danceability ?? 0.60,
      tempo:        track?.tempo ?? 120.0,
      search_query: eventData.search_query || null,
    };

    // Non-blocking fire-and-forget
    fetch('/api/behavior/log', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).catch(() => {}); // silently ignore network errors
  }
}

// Global singleton
window.behaviorCollector = new BehaviorCollector();
