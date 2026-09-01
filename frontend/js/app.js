/**
 * Main Application Controller for RL Music Recommendation Studio.
 * Includes OpenSpot Live Search, HQ Direct Streaming, Dynamic Ingestion,
 * Synchronized Real-Time Lyrics, Karaoke Theater, and Discovery Hub.
 */

const DEFAULT_COVER_ART = "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=500&auto=format&fit=crop&q=80";

class App {
  constructor() {
    this.currentUserId = 'usr_00002'; // Default Lo-Fi Coder
    this.currentModelType = 'wolpertinger';
    this.currentTrack = null;
    this.sessionStep = 0;
    this.sessionLikes = 0;
    this.sessionSkips = 0;
    this.sessionReward = 0.0;
    
    this.historySongIds = [];
    this.historySkipTypes = [];
    this.historyLikes = [];

    this.relatedData = null;
    this.currentRelatedFilter = 'all';
    this.searchDebounceTimer = null;
    this.importedTrackIds = new Set();

    // Lyrics State
    this.currentLyrics = null;
    this.activeLyricIndex = -1;
    this.lyricsAutoScroll = true;
    this.lyricsMode = 'sync'; // 'sync' | 'plain'
    this.isTheaterMode = false;
    this.lyricsCache = new Map();

    this.userProfile = {
      user_id: 'usr_00002',
      pref_energy: 0.35,
      pref_valence: 0.45,
      pref_danceability: 0.55,
      genre_affinities: { "Lo-Fi / Chillhop": 0.65, "Jazz": 0.25, "Pop": 0.10 }
    };

    // Queue State & Filter Controls
    this.currentQueueTracks = [];
    this.currentQDist = [];
    this.queueSearchQuery = '';
    this.queueFilter = 'all';
    this.quickAddDebounceTimer = null;

    // Trajectory History State & Filters
    this.sessionTrajectoryHistory = [];
    this.trajectoryFilter = 'all';

    this.initEventListeners();
    this.loadInitialData();
    this._initBehaviorCollector();
  }

  _initBehaviorCollector() {
    // Keep behavior collector in sync with user ID
    if (window.behaviorCollector) {
      window.behaviorCollector.setUser(this.currentUserId);
    }
    // Update context badge on load
    this._updateContextBadge();
    // Refresh context badge every 30 minutes
    setInterval(() => this._updateContextBadge(), 30 * 60 * 1000);
  }

  _updateContextBadge() {
    if (!window.behaviorCollector) return;
    const label = window.behaviorCollector.getContextLabel();
    const badge = document.getElementById('context-badge');
    if (badge) badge.innerText = label;
  }

  initEventListeners() {
    // Navigation Tabs
    document.querySelectorAll('.nav-item').forEach(btn => {
      btn.addEventListener('click', () => {
        const tabId = btn.getAttribute('data-tab');
        this.switchTab(tabId);
      });
    });

    // Model Selector Toggle
    document.querySelectorAll('.toggle-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.currentModelType = btn.getAttribute('data-model');
        this.fetchNextRecommendation();
      });
    });

    // Persona Selector
    const personaSel = document.getElementById('persona-select');
    if (personaSel) {
      personaSel.addEventListener('change', (e) => {
        this.currentUserId = e.target.value;
        if (window.behaviorCollector) window.behaviorCollector.setUser(this.currentUserId);
        this.updatePersonaDetails(this.currentUserId);
        this.resetSession();
      });
    }

    // Playlist Save Button
    const btnSave = document.getElementById('btn-playlist-save');
    if (btnSave) {
      btnSave.addEventListener('click', () => {
        if (this.currentTrack && window.behaviorCollector) {
          window.behaviorCollector.onPlaylistSave(this.currentTrack);
          this.handleFeedback('saved', 1.5);
          btnSave.classList.add('saved');
          setTimeout(() => btnSave.classList.remove('saved'), 2000);
          this.showToast(`💾 "${this.currentTrack.title}" saved to playlist!`);
        }
      });
    }

    // Audio Play / Pause
    const playBtn = document.getElementById('btn-audio-play');
    if (playBtn) {
      playBtn.addEventListener('click', () => {
        if (this.currentTrack) {
          window.audioSynth.toggle(this.currentTrack);
        }
      });
    }

    // Audio Scrubber input
    const scrubber = document.getElementById('audio-scrubber');
    if (scrubber) {
      scrubber.addEventListener('input', (e) => {
        window.audioSynth.seek(parseFloat(e.target.value));
      });
    }

    // Interactive Feedback Action Buttons
    const btnLike = document.getElementById('btn-feedback-like');
    if (btnLike) btnLike.addEventListener('click', () => this.handleFeedback('liked', 2.5));

    const btnFull = document.getElementById('btn-feedback-full');
    if (btnFull) btnFull.addEventListener('click', () => this.handleFeedback('no_skip', 1.0));

    const btnSkip = document.getElementById('btn-feedback-skip');
    if (btnSkip) btnSkip.addEventListener('click', () => this.handleFeedback('skip_early', -1.0));

    const btnDislike = document.getElementById('btn-feedback-dislike');
    if (btnDislike) btnDislike.addEventListener('click', () => this.handleFeedback('disliked', -2.0));

    const btnRefresh = document.getElementById('btn-refresh-recs');
    if (btnRefresh) {
      btnRefresh.addEventListener('click', () => {
        btnRefresh.classList.add('fa-spin');
        this.fetchNextRecommendation().finally(() => {
          setTimeout(() => btnRefresh.classList.remove('fa-spin'), 600);
        });
      });
    }

    // =========================================================================
    // Recommendations Queue & Search Event Listeners
    // =========================================================================
    const queueSearchInput = document.getElementById('queue-search-input');
    const btnQueueSearchClear = document.getElementById('btn-queue-search-clear');
    if (queueSearchInput) {
      queueSearchInput.addEventListener('input', (e) => {
        this.queueSearchQuery = e.target.value.trim().toLowerCase();
        if (btnQueueSearchClear) {
          btnQueueSearchClear.style.display = this.queueSearchQuery ? 'flex' : 'none';
        }
        this.renderQueue();
      });
    }

    if (btnQueueSearchClear) {
      btnQueueSearchClear.addEventListener('click', () => {
        if (queueSearchInput) queueSearchInput.value = '';
        this.queueSearchQuery = '';
        btnQueueSearchClear.style.display = 'none';
        this.renderQueue();
        if (queueSearchInput) queueSearchInput.focus();
      });
    }

    // Queue Category Filter Pills
    document.querySelectorAll('#queue-filter-pills .queue-pill').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#queue-filter-pills .queue-pill').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.queueFilter = btn.getAttribute('data-filter');
        this.renderQueue();
      });
    });

    // Queue Action Buttons (Add Song, Shuffle, Clear)
    const btnQuickAddQueue = document.getElementById('btn-quick-add-queue');
    const btnCloseQuickAdd = document.getElementById('btn-close-quick-add');
    const quickAddPanel = document.getElementById('quick-add-panel');
    const quickAddInput = document.getElementById('quick-add-search-input');

    if (btnQuickAddQueue) {
      btnQuickAddQueue.addEventListener('click', () => {
        if (quickAddPanel) {
          const isVisible = quickAddPanel.style.display !== 'none';
          quickAddPanel.style.display = isVisible ? 'none' : 'flex';
          if (!isVisible && quickAddInput) {
            quickAddInput.focus();
            if (quickAddInput.value.trim()) {
              this.searchCatalogForQuickAdd(quickAddInput.value.trim());
            }
          }
        }
      });
    }

    if (btnCloseQuickAdd && quickAddPanel) {
      btnCloseQuickAdd.addEventListener('click', () => {
        quickAddPanel.style.display = 'none';
      });
    }

    if (quickAddInput) {
      quickAddInput.addEventListener('input', (e) => {
        clearTimeout(this.quickAddDebounceTimer);
        const q = e.target.value.trim();
        if (q.length >= 2) {
          this.quickAddDebounceTimer = setTimeout(() => this.searchCatalogForQuickAdd(q), 300);
        } else {
          const resEl = document.getElementById('quick-add-results');
          if (resEl) resEl.innerHTML = '<div class="quick-add-prompt">Type 2 or more characters to search songs...</div>';
        }
      });
    }

    const btnShuffleQueue = document.getElementById('btn-shuffle-queue');
    if (btnShuffleQueue) {
      btnShuffleQueue.addEventListener('click', () => this.shuffleQueue());
    }

    const btnClearQueue = document.getElementById('btn-clear-queue');
    if (btnClearQueue) {
      btnClearQueue.addEventListener('click', () => this.clearQueue());
    }

    // Trajectory Filter Pills
    document.querySelectorAll('#trajectory-filter-pills .traj-pill').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#trajectory-filter-pills .traj-pill').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.trajectoryFilter = btn.getAttribute('data-filter');
        this.renderTrajectoryLog();
      });
    });

    const btnClearTraj = document.getElementById('btn-clear-trajectory');
    if (btnClearTraj) {
      btnClearTraj.addEventListener('click', () => this.clearTrajectoryLog());
    }

    // Global Keydown Shortcut ('/' to focus Queue Search)
    window.addEventListener('keydown', (e) => {
      if ((e.key === '/' || (e.ctrlKey && e.key === 'k')) && !['INPUT', 'TEXTAREA'].includes(e.target.tagName)) {
        e.preventDefault();
        const qi = document.getElementById('queue-search-input');
        if (qi) {
          qi.focus();
          qi.select();
        }
      }
    });

    // =========================================================================
    // Lyrics Event Listeners
    // =========================================================================
    const btnLyricsToggle = document.getElementById('btn-lyrics-toggle');
    const lyricsCard = document.getElementById('player-lyrics-card');
    if (btnLyricsToggle && lyricsCard) {
      btnLyricsToggle.addEventListener('click', () => {
        const isCollapsed = lyricsCard.classList.toggle('collapsed');
        btnLyricsToggle.classList.toggle('active', !isCollapsed);
      });
    }

    const btnTheater = document.getElementById('btn-lyrics-theater');
    const btnTheaterCard = document.getElementById('btn-lyrics-fullscreen-card');
    const btnTheaterClose = document.getElementById('btn-theater-close');

    if (btnTheater) btnTheater.addEventListener('click', () => this.toggleTheaterMode(true));
    if (btnTheaterCard) btnTheaterCard.addEventListener('click', () => this.toggleTheaterMode(true));
    if (btnTheaterClose) btnTheaterClose.addEventListener('click', () => this.toggleTheaterMode(false));

    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.isTheaterMode) {
        this.toggleTheaterMode(false);
      }
      if ((e.key === 'l' || e.key === 'L') && !['INPUT', 'TEXTAREA'].includes(e.target.tagName)) {
        this.toggleTheaterMode(!this.isTheaterMode);
      }
    });

    const btnModeSync = document.getElementById('btn-lyrics-mode-sync');
    const btnModePlain = document.getElementById('btn-lyrics-mode-plain');
    const containerSync = document.getElementById('lyrics-container');
    const containerPlain = document.getElementById('lyrics-plain-container');

    if (btnModeSync && btnModePlain) {
      btnModeSync.addEventListener('click', () => {
        btnModeSync.classList.add('active');
        btnModePlain.classList.remove('active');
        this.lyricsMode = 'sync';
        if (containerSync) containerSync.style.display = 'block';
        if (containerPlain) containerPlain.style.display = 'none';
      });

      btnModePlain.addEventListener('click', () => {
        btnModePlain.classList.add('active');
        btnModeSync.classList.remove('active');
        this.lyricsMode = 'plain';
        if (containerSync) containerSync.style.display = 'none';
        if (containerPlain) containerPlain.style.display = 'block';
      });
    }

    const btnAutoScroll = document.getElementById('btn-lyrics-autoscroll');
    if (btnAutoScroll) {
      btnAutoScroll.addEventListener('click', () => {
        this.lyricsAutoScroll = !this.lyricsAutoScroll;
        btnAutoScroll.classList.toggle('active', this.lyricsAutoScroll);
        btnAutoScroll.setAttribute('data-enabled', this.lyricsAutoScroll ? 'true' : 'false');
      });
    }

    // Theater Mode Playback Controls
    const theaterPlay = document.getElementById('btn-theater-play');
    const theaterLike = document.getElementById('btn-theater-like');
    const theaterSkip = document.getElementById('btn-theater-skip');
    const theaterScrubber = document.getElementById('theater-audio-scrubber');

    if (theaterPlay) {
      theaterPlay.addEventListener('click', () => {
        if (this.currentTrack) window.audioSynth.toggle(this.currentTrack);
      });
    }
    if (theaterLike) theaterLike.addEventListener('click', () => this.handleFeedback('liked', 2.5));
    if (theaterSkip) theaterSkip.addEventListener('click', () => this.handleFeedback('skip_early', -1.0));
    if (theaterScrubber) {
      theaterScrubber.addEventListener('input', (e) => {
        window.audioSynth.seek(parseFloat(e.target.value));
      });
    }

    // Simulation Button
    const btnSim = document.getElementById('btn-run-simulation');
    if (btnSim) {
      btnSim.addEventListener('click', () => {
        const count = parseInt(document.getElementById('sim-session-count').value, 10) || 25;
        window.benchmarkView.runSimulation(count);
      });
    }

    // Catalog Search & Filter
    const searchInp = document.getElementById('catalog-search');
    const genreSel = document.getElementById('catalog-genre-filter');
    if (searchInp) searchInp.addEventListener('input', () => this.loadCatalog());
    if (genreSel) genreSel.addEventListener('change', () => this.loadCatalog());

    // Related Discovery Hub Close Button
    const btnCloseHub = document.getElementById('btn-close-related-hub');
    if (btnCloseHub) {
      btnCloseHub.addEventListener('click', () => this.closeRelatedHub());
    }

    // Related Section Filter Buttons
    document.querySelectorAll('.sec-filter-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.sec-filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.currentRelatedFilter = btn.getAttribute('data-section');
        if (this.relatedData) {
          this.renderRelatedSections(this.relatedData, this.currentRelatedFilter);
        }
      });
    });

    // Explore Catalog Related from Live Player Shelf
    const btnExplore = document.getElementById('btn-explore-catalog-related');
    if (btnExplore) {
      btnExplore.addEventListener('click', () => {
        if (this.currentTrack) {
          this.switchTab('catalog-tab');
          this.openRelatedHub(this.currentTrack.song_id);
        }
      });
    }

    // OpenSpot Search Input & Button
    const openSpotInput = document.getElementById('openspot-search-input');
    const btnOpenSpotSearch = document.getElementById('btn-openspot-search');

    if (btnOpenSpotSearch) {
      btnOpenSpotSearch.addEventListener('click', () => {
        const q = openSpotInput ? openSpotInput.value.trim() : '';
        if (q) this.searchOpenSpot(q);
      });
    }

    if (openSpotInput) {
      openSpotInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          const q = openSpotInput.value.trim();
          if (q) this.searchOpenSpot(q);
        }
      });

      openSpotInput.addEventListener('input', (e) => {
        clearTimeout(this.searchDebounceTimer);
        const q = e.target.value.trim();
        if (q.length >= 3) {
          this.searchDebounceTimer = setTimeout(() => this.searchOpenSpot(q), 450);
        }
      });
    }

    // OpenSpot Genre Filter Pills
    document.querySelectorAll('.genre-pill').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.genre-pill').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const genre = btn.getAttribute('data-genre');
        this.loadOpenSpotCharts(genre);
      });
    });
    // Volume Slider
    const volumeSlider = document.getElementById('volume-slider');
    if (volumeSlider) {
      volumeSlider.addEventListener('input', (e) => {
        const vol = parseFloat(e.target.value) / 100;
        window.audioSynth.setVolume(vol);
        const volIcon = document.getElementById('volume-icon');
        const volPct = document.getElementById('volume-pct');
        if (volPct) volPct.innerText = `${e.target.value}%`;
        if (volIcon) {
          if (vol === 0) volIcon.className = 'fa-solid fa-volume-xmark';
          else if (vol < 0.4) volIcon.className = 'fa-solid fa-volume-low';
          else volIcon.className = 'fa-solid fa-volume-high';
        }
      });
    }

    // Mini Player Controls
    const miniPlayBtn = document.getElementById('mini-play-btn');
    const miniSkipBtn = document.getElementById('mini-skip-btn');
    const miniGoToPlayer = document.getElementById('mini-go-to-player');
    if (miniPlayBtn) {
      miniPlayBtn.addEventListener('click', () => {
        if (this.currentTrack) window.audioSynth.toggle(this.currentTrack);
        this.updateMiniPlayer();
      });
    }
    if (miniSkipBtn) {
      miniSkipBtn.addEventListener('click', () => this.handleFeedback('skip_early', -1.0));
    }
    if (miniGoToPlayer) {
      miniGoToPlayer.addEventListener('click', () => this.switchTab('player-tab'));
    }
  }

  async loadInitialData() {
    await this.updatePersonaDetails(this.currentUserId);
    await this.fetchNextRecommendation();
    await this.loadCatalog();
  }

  async updatePersonaDetails(userId) {
    try {
      const res = await fetch('/api/users');
      const users = await res.json();
      const u = users.find(x => x.user_id === userId) || users[0];
      if (u) {
        this.userProfile = u;
        document.getElementById('user-pref-energy').innerText = u.pref_energy.toFixed(2);
        document.getElementById('user-pref-valence').innerText = u.pref_valence.toFixed(2);
        
        const topG = Object.entries(u.genre_affinities).sort((a, b) => b[1] - a[1])[0];
        if (topG) {
          document.getElementById('user-top-genre').innerText = topG[0];
        }
      }
    } catch (e) {
      console.error('Error loading persona:', e);
    }
  }

  resetSession() {
    this.sessionStep = 0;
    this.sessionLikes = 0;
    this.sessionSkips = 0;
    this.sessionReward = 0.0;
    this.historySongIds = [];
    this.historySkipTypes = [];
    this.historyLikes = [];
    this.sessionTrajectoryHistory = [];

    this.updateCounters();
    this.renderTrajectoryLog();
    
    window.rlVisualizer.resetTrajectory();
    this.fetchNextRecommendation();
  }

  async fetchNextRecommendation() {
    try {
      // Build hybrid payload via BehaviorCollector
      const basePayload = {
        user_id: this.currentUserId,
        history_song_ids: this.historySongIds,
        history_skip_types: this.historySkipTypes,
        history_likes: this.historyLikes,
        slate_size: 10,
      };

      const hybridPayload = window.behaviorCollector
        ? window.behaviorCollector.buildHybridPayload(basePayload)
        : basePayload;

      const res = await fetch('/api/recommend/hybrid', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(hybridPayload)
      });

      const data = await res.json();

      if (data.recommendations && data.recommendations.length > 0) {
        this.currentTrack = data.recommendations[0];
        this.renderCurrentTrack(this.currentTrack);

        // Notify behavior collector of current track
        if (window.behaviorCollector) {
          window.behaviorCollector.setCurrentTrack(this.currentTrack);
        }

        // Context badge
        if (data.context_label) {
          const badge = document.getElementById('context-badge');
          if (badge) badge.innerText = data.context_label;
        }

        // Render enriched queue with hybrid reasons
        this.renderQueue(data.recommendations.slice(1), data.q_score_distribution || []);

        if (data.q_score_distribution) {
          window.rlVisualizer.updateQChart(data.q_score_distribution);
        }
        window.rlVisualizer.updateRadar(this.userProfile, this.currentTrack);

        const trajX = this.currentTrack.energy;
        const trajY = this.currentTrack.valence;
        window.rlVisualizer.addTrajectoryPoint(trajX, trajY, `#${this.sessionStep + 1}`);

        this.updatePlayerRelatedShelf(this.currentTrack.song_id);
        this.loadLyricsForTrack(this.currentTrack);
        this.updateMiniPlayer();

        if (window.audioSynth.isPlaying) {
          window.audioSynth.playTrack(this.currentTrack);
        }

        // Update user insights panel
        this.loadUserInsights();
      }
    } catch (e) {
      console.error('Hybrid recommendation fetch error:', e);
      // Fallback to original recommend endpoint
      try {
        const res = await fetch('/api/recommend', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: this.currentUserId,
            history_song_ids: this.historySongIds,
            history_skip_types: this.historySkipTypes,
            history_likes: this.historyLikes,
            model_type: this.currentModelType,
            slate_size: 5
          })
        });
        const data = await res.json();
        if (data.recommendations && data.recommendations.length > 0) {
          this.currentTrack = data.recommendations[0];
          this.renderCurrentTrack(this.currentTrack);
          this.renderQueue(data.recommendations.slice(1), data.q_score_distribution);
          if (data.q_score_distribution) window.rlVisualizer.updateQChart(data.q_score_distribution);
          window.rlVisualizer.updateRadar(this.userProfile, this.currentTrack);
          this.loadLyricsForTrack(this.currentTrack);
          this.updateMiniPlayer();
        }
      } catch(e2) {
        console.error('Fallback recommendation also failed:', e2);
      }
    }
  }

  renderCurrentTrack(track) {
    document.getElementById('current-title').innerText = track.title;
    document.getElementById('current-artist').innerText = track.artist_name;
    document.getElementById('current-genre').innerText = track.genre;

    // Album cover in turntable vinyl
    const albumArt = document.getElementById('player-album-art');
    if (albumArt) {
      albumArt.src = track.image_url || DEFAULT_COVER_ART;
    }

    document.getElementById('current-energy').innerText = track.energy.toFixed(2);
    document.getElementById('current-valence').innerText = track.valence.toFixed(2);
    document.getElementById('current-tempo').innerText = `${Math.round(track.tempo)} BPM`;
    document.getElementById('current-dance').innerText = track.danceability.toFixed(2);

    const durLabel = document.getElementById('player-time-dur');
    if (durLabel) {
      durLabel.innerText = track.duration_formatted || "3:30";
    }

    // Update Theater Mode metadata if open
    if (this.isTheaterMode) {
      const backdrop = document.getElementById('theater-backdrop');
      const theaterArt = document.getElementById('theater-album-art');
      const theaterVinylArt = document.getElementById('theater-vinyl-art');
      const theaterTitle = document.getElementById('theater-title');
      const theaterArtist = document.getElementById('theater-artist');
      const theaterGenre = document.getElementById('theater-genre');  // FIX: was undeclared 'genreEl'

      const cover = track.image_url || DEFAULT_COVER_ART;
      if (backdrop) backdrop.style.backgroundImage = `url("${cover}")`;
      if (theaterArt) theaterArt.src = cover;
      if (theaterVinylArt) theaterVinylArt.src = cover;
      if (theaterTitle) theaterTitle.innerText = track.title;
      if (theaterArtist) theaterArtist.innerText = track.artist_name;
      if (theaterGenre) theaterGenre.innerText = track.genre;  // FIX: was `if (genreEl)` which was undefined
    }
  }

  // =========================================================================
  // Lyrics Engine Implementation
  // =========================================================================

  onTrackLoaded(track) {
    this.currentTrack = track;
    this.renderCurrentTrack(track);
    this.loadLyricsForTrack(track);
  }

  async loadLyricsForTrack(track) {
    if (!track) return;
    const listEl = document.getElementById('lyrics-lines-list');
    const heroText = document.getElementById('active-hero-text');
    const loadingEl = document.getElementById('lyrics-loading-state');
    const sourceBadge = document.getElementById('lyrics-source-badge');
    const sourceName = document.getElementById('lyrics-source-name');
    const theaterScroll = document.getElementById('theater-lyrics-scroll');
    const theaterBadge = document.getElementById('theater-source-badge');
    const plainTextEl = document.getElementById('lyrics-plain-text');

    if (loadingEl) loadingEl.style.display = 'flex';
    if (listEl) listEl.innerHTML = '';
    if (theaterScroll) theaterScroll.innerHTML = '';
    if (heroText) heroText.innerText = `Preparing lyrics for "${track.title}"...`;
    this.activeLyricIndex = -1;

    const cacheKey = `${track.song_id}_${track.title}_${track.artist_name}`;
    let lyrics = this.lyricsCache.get(cacheKey);

    if (!lyrics) {
      try {
        const params = new URLSearchParams({
          song_id: track.song_id || '',
          title: track.title || '',
          artist: track.artist_name || '',
          genre: track.genre || 'Pop',
          energy: track.energy !== undefined ? track.energy : 0.65,
          valence: track.valence !== undefined ? track.valence : 0.60,
          tempo: track.tempo !== undefined ? track.tempo : 120.0,
          // FIX: songs from catalog use song_length_ms, OpenSpot imports use duration_ms
          duration_ms: track.duration_ms || track.song_length_ms || 210000
        });

        const res = await fetch(`/api/lyrics?${params.toString()}`);
        if (res.ok) {
          lyrics = await res.json();
          this.lyricsCache.set(cacheKey, lyrics);
        }
      } catch (e) {
        console.error('Error fetching lyrics:', e);
      }
    }

    if (loadingEl) loadingEl.style.display = 'none';

    if (lyrics && lyrics.synced_lyrics && lyrics.synced_lyrics.length > 0) {
      this.currentLyrics = lyrics;

      if (sourceBadge) {
        const badgeIcon = lyrics.source === 'lrclib' ? 'fa-circle-check' : 'fa-wand-magic-sparkles';
        const badgeText = lyrics.source === 'lrclib' ? 'Studio Master' : (lyrics.source === 'jiosaavn' ? 'JioSaavn Official' : 'AI Calibrated');
        sourceBadge.innerHTML = `<i class="fa-solid ${badgeIcon}"></i> ${badgeText}`;
      }
      if (sourceName) sourceName.innerText = lyrics.source_label || 'SoundSpace Synced';
      if (theaterBadge) theaterBadge.innerHTML = `<i class="fa-solid fa-microphone-lines"></i> ${lyrics.source_label || 'Synced Karaoke'}`;

      if (plainTextEl) {
        plainTextEl.innerText = lyrics.plain_lyrics || lyrics.synced_lyrics.map(l => l.text).join('\n');
      }

      // Render studio card lines
      if (listEl) {
        listEl.innerHTML = lyrics.synced_lyrics.map((line, idx) => `
          <div class="lyric-line" id="lyric-line-${idx}" data-index="${idx}" data-time="${line.time}" onclick="window.app.seekToLyric(${idx})">
            <span class="lyric-time-tag">${line.formatted_time || '0:00'}</span>
            <span class="lyric-text">${this.escapeHtml(line.text)}</span>
          </div>
        `).join('');
      }

      // Render theater stage lines
      if (theaterScroll) {
        theaterScroll.innerHTML = lyrics.synced_lyrics.map((line, idx) => `
          <div class="theater-lyric-line" id="theater-lyric-line-${idx}" data-index="${idx}" data-time="${line.time}" onclick="window.app.seekToLyric(${idx})">
            ${this.escapeHtml(line.text)}
          </div>
        `).join('');
      }

      if (heroText) {
        heroText.innerText = lyrics.synced_lyrics[0] ? lyrics.synced_lyrics[0].text : track.title;
      }

      // Immediately sync with current audio playback position
      const curTime = window.audioSynth.getCurrentTime();
      const durTime = window.audioSynth.getDuration();
      this.onPlaybackTimeUpdate(curTime, durTime);
    } else {
      if (heroText) heroText.innerText = `Instrumental track — "${track.title}"`;
      if (listEl) listEl.innerHTML = '<p style="color:var(--text-secondary); text-align:center; padding:30px; font-size:13px;"><i class="fa-solid fa-music"></i> Enjoy the instrumental groove</p>';
    }
  }

  onPlaybackTimeUpdate(currentTime, duration) {
    if (!this.currentLyrics || !this.currentLyrics.synced_lyrics || this.currentLyrics.synced_lyrics.length === 0) return;

    const lines = this.currentLyrics.synced_lyrics;
    let targetIndex = -1;

    for (let i = 0; i < lines.length; i++) {
      if (currentTime >= lines[i].time) {
        targetIndex = i;
      } else {
        break;
      }
    }

    if (targetIndex === -1 && lines.length > 0) targetIndex = 0;

    if (targetIndex !== this.activeLyricIndex) {
      this.activeLyricIndex = targetIndex;

      // Update active and passed CSS classes
      for (let i = 0; i < lines.length; i++) {
        const el = document.getElementById(`lyric-line-${i}`);
        const tEl = document.getElementById(`theater-lyric-line-${i}`);
        if (i < targetIndex) {
          if (el) { el.classList.remove('active'); el.classList.add('passed'); }
          if (tEl) { tEl.classList.remove('active'); tEl.classList.add('passed'); }
        } else if (i === targetIndex) {
          if (el) { el.classList.remove('passed'); el.classList.add('active'); }
          if (tEl) { tEl.classList.remove('passed'); tEl.classList.add('active'); }
        } else {
          if (el) { el.classList.remove('active', 'passed'); }
          if (tEl) { tEl.classList.remove('active', 'passed'); }
        }
      }

      // Update Hero display banner
      const heroText = document.getElementById('active-hero-text');
      if (heroText && lines[targetIndex]) {
        heroText.innerText = lines[targetIndex].text;
      }

      // Smooth auto-scroll for studio card
      if (this.lyricsAutoScroll) {
        const activeEl = document.getElementById(`lyric-line-${targetIndex}`);
        const container = document.getElementById('lyrics-container');
        if (activeEl && container) {
          const offset = activeEl.offsetTop - (container.clientHeight / 2) + (activeEl.clientHeight / 2);
          container.scrollTo({ top: Math.max(0, offset), behavior: 'smooth' });
        }

        // Smooth auto-scroll for fullscreen theater mode
        const activeTheaterEl = document.getElementById(`theater-lyric-line-${targetIndex}`);
        const theaterScroll = document.getElementById('theater-lyrics-scroll');
        if (activeTheaterEl && theaterScroll) {
          const tOffset = activeTheaterEl.offsetTop - (theaterScroll.clientHeight / 2) + (activeTheaterEl.clientHeight / 2);
          theaterScroll.scrollTo({ top: Math.max(0, tOffset), behavior: 'smooth' });
        }
      }
    }

    // Update Theater Mode timer and scrubber
    if (this.isTheaterMode) {
      const tCur = document.getElementById('theater-time-cur');
      const tDur = document.getElementById('theater-time-dur');
      const tScrub = document.getElementById('theater-audio-scrubber');
      if (tCur) tCur.innerText = window.audioSynth.formatTime(currentTime);
      if (tDur && duration) tDur.innerText = window.audioSynth.formatTime(duration);
      if (tScrub && !tScrub.matches(':active') && duration > 0) {
        tScrub.value = (currentTime / duration) * 100;
      }
    }
  }

  seekToLyric(index) {
    if (!this.currentLyrics || !this.currentLyrics.synced_lyrics) return;
    const line = this.currentLyrics.synced_lyrics[index];
    if (line && line.time !== undefined) {
      window.audioSynth.seekToTime(line.time);
      if (!window.audioSynth.isPlaying && this.currentTrack) {
        window.audioSynth.playTrack(this.currentTrack);
      }
    }
  }

  toggleTheaterMode(show) {
    this.isTheaterMode = show;
    const modal = document.getElementById('lyrics-theater-modal');
    if (!modal) return;

    modal.style.display = show ? 'flex' : 'none';

    if (show && this.currentTrack) {
      const backdrop = document.getElementById('theater-backdrop');
      const art = document.getElementById('theater-album-art');
      const vinylArt = document.getElementById('theater-vinyl-art');
      const titleEl = document.getElementById('theater-title');
      const artistEl = document.getElementById('theater-artist');
      const genreEl = document.getElementById('theater-genre');
      const cover = this.currentTrack.image_url || DEFAULT_COVER_ART;

      if (backdrop) backdrop.style.backgroundImage = `url("${cover}")`;
      if (art) art.src = cover;
      if (vinylArt) vinylArt.src = cover;
      if (titleEl) titleEl.innerText = this.currentTrack.title;
      if (artistEl) artistEl.innerText = this.currentTrack.artist_name;
      if (genreEl) genreEl.innerText = this.currentTrack.genre;

      const playBtn = document.getElementById('btn-theater-play');
      const vinyl = document.getElementById('theater-vinyl-disc');
      if (playBtn) playBtn.innerHTML = window.audioSynth.isPlaying ? '<i class="fa-solid fa-pause"></i>' : '<i class="fa-solid fa-play"></i>';
      if (vinyl) {
        if (window.audioSynth.isPlaying) vinyl.classList.remove('paused');
        else vinyl.classList.add('paused');
      }

      const cur = window.audioSynth.getCurrentTime();
      const dur = window.audioSynth.getDuration();
      this.onPlaybackTimeUpdate(cur, dur);
    }
  }

  escapeHtml(text) {
    if (!text) return '';
    const map = {
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
  }

  renderQueue(queueTracks, qDist) {
    const container = document.getElementById('queue-list-container');
    const countBadge = document.getElementById('queue-filter-count');
    if (!container) return;

    if (queueTracks !== undefined) {
      this.currentQueueTracks = Array.isArray(queueTracks) ? [...queueTracks] : [];
    }
    if (qDist !== undefined) {
      this.currentQDist = Array.isArray(qDist) ? [...qDist] : [];
    }

    const allTracks = this.currentQueueTracks || [];
    if (allTracks.length === 0) {
      if (countBadge) countBadge.innerText = '0 tracks';
      container.innerHTML = `
        <div class="openspot-empty-state" style="padding:24px 12px; text-align:center;">
          <i class="fa-solid fa-list-music" style="font-size:24px; color:var(--text-muted); margin-bottom:8px;"></i>
          <p style="color:var(--text-secondary); font-size:12px; margin:0 0 10px;">Queue is currently empty.</p>
          <button class="btn-small" onclick="window.app.fetchNextRecommendation()" style="margin:0 auto;">
            <i class="fa-solid fa-arrows-rotate"></i> Fetch RL Recommendations
          </button>
        </div>
      `;
      return;
    }

    // Build genre counts for diversity tag calculation
    const genreCounts = {};
    allTracks.forEach(t => { genreCounts[t.genre] = (genreCounts[t.genre] || 0) + 1; });
    const uniqueGenres = Object.keys(genreCounts).length;

    // Filter tracks by search query and category pill
    const filteredTracks = [];
    allTracks.forEach((t, origIdx) => {
      const qValNum = this.currentQDist && this.currentQDist[origIdx] ? this.currentQDist[origIdx].q_score : (2.5 - origIdx * 0.4);
      const qVal = qValNum.toFixed(2);
      const reason = this._getRecommendationReason(t, origIdx, uniqueGenres);
      const reasonText = reason.replace(/<[^>]*>/g, '').toLowerCase();
      
      // Filter Tag check
      let matchesFilter = true;
      if (this.queueFilter === 'top_q') {
        matchesFilter = origIdx === 0 || qValNum >= 3.0 || reasonText.includes('top q');
      } else if (this.queueFilter === 'energy') {
        matchesFilter = (t.energy || 0.5) > 0.65 || reasonText.includes('energy');
      } else if (this.queueFilter === 'chill') {
        matchesFilter = (t.energy || 0.5) < 0.45 || reasonText.includes('chill');
      } else if (this.queueFilter === 'mood') {
        matchesFilter = reasonText.includes('mood') || Math.abs((t.valence || 0.5) - (this.currentTrack ? this.currentTrack.valence || 0.5 : 0.5)) < 0.2;
      } else if (this.queueFilter === 'discovery') {
        matchesFilter = reasonText.includes('discovery') || (this.currentTrack && t.genre !== this.currentTrack.genre);
      }

      // Search Query check
      let matchesSearch = true;
      if (this.queueSearchQuery) {
        const titleMatch = (t.title || '').toLowerCase().includes(this.queueSearchQuery);
        const artistMatch = (t.artist_name || '').toLowerCase().includes(this.queueSearchQuery);
        const genreMatch = (t.genre || '').toLowerCase().includes(this.queueSearchQuery);
        const reasonMatch = reasonText.includes(this.queueSearchQuery);
        matchesSearch = titleMatch || artistMatch || genreMatch || reasonMatch;
      }

      if (matchesFilter && matchesSearch) {
        filteredTracks.push({ track: t, origIdx, qVal, qValNum, reason });
      }
    });

    // Update count badge
    if (countBadge) {
      countBadge.innerText = `${filteredTracks.length} of ${allTracks.length} tracks`;
    }

    if (filteredTracks.length === 0) {
      container.innerHTML = `
        <div class="openspot-empty-state" style="padding:20px 12px; text-align:center;">
          <i class="fa-solid fa-filter-circle-xmark" style="font-size:22px; color:var(--accent-amber); margin-bottom:6px;"></i>
          <p style="color:var(--text-secondary); font-size:12px; margin:0 0 10px;">No songs in queue match "<strong>${this.escapeHtml(this.queueSearchQuery || this.queueFilter)}</strong>"</p>
          <div style="display:flex; justify-content:center; gap:8px;">
            <button class="btn-small" onclick="document.getElementById('btn-queue-search-clear').click()">
              <i class="fa-solid fa-xmark"></i> Clear Filters
            </button>
            <button class="btn-small" onclick="document.getElementById('btn-quick-add-queue').click()">
              <i class="fa-solid fa-plus"></i> Search Full Library
            </button>
          </div>
        </div>
      `;
      return;
    }

    container.innerHTML = filteredTracks.map((item, displayIdx) => {
      const t = item.track;
      const origIdx = item.origIdx;
      const qVal = item.qVal;
      const qValNum = item.qValNum;
      const reason = item.reason;
      const coverUrl = t.image_url || DEFAULT_COVER_ART;
      const isCurrentPlaying = this.currentTrack && this.currentTrack.song_id === t.song_id && window.audioSynth.isPlaying;

      // Calculate Q fill percent (clamped 0 - 100)
      const qFillPct = Math.max(5, Math.min(100, (parseFloat(qVal) + 5) / 20 * 100));

      return `
        <div class="queue-item ${isCurrentPlaying ? 'is-playing' : ''}" style="animation-delay: ${displayIdx * 45}ms">
          <div class="queue-item-left">
            <span class="queue-index-badge">#${origIdx + 1}</span>
            <div class="queue-thumb-container">
              <img class="queue-thumb" src="${coverUrl}" alt="Cover">
              ${isCurrentPlaying ? `
                <div class="queue-wave-overlay">
                  <span class="queue-wave-bar"></span>
                  <span class="queue-wave-bar"></span>
                  <span class="queue-wave-bar"></span>
                </div>
              ` : ''}
            </div>
            <div class="queue-item-info">
              <span class="queue-item-title" title="${this.escapeHtml(t.title)}">${this.escapeHtml(t.title)}</span>
              <span class="queue-item-artist" title="${this.escapeHtml(t.artist_name)} • ${this.escapeHtml(t.genre)}">
                ${this.escapeHtml(t.artist_name)} • ${this.escapeHtml(t.genre)}
              </span>
              <div><span class="queue-reason-chip">${reason}</span></div>
            </div>
          </div>
          <div class="queue-item-meta">
            <div class="queue-item-actions">
              <button class="queue-action-btn btn-play-now" onclick="window.app.playSongDirectly('${t.song_id}')" title="Play Song Now">
                <i class="fa-solid fa-play"></i>
              </button>
              <button class="queue-action-btn" onclick="window.app.moveQueueItemToTop('${t.song_id}')" title="Move to Top (Play Next)">
                <i class="fa-solid fa-arrow-up"></i>
              </button>
              <button class="queue-action-btn" onclick="window.app.openRelatedHubFromSong('${t.song_id}')" title="Explore Acoustic Twins">
                <i class="fa-solid fa-wand-magic-sparkles"></i>
              </button>
              <button class="queue-action-btn btn-remove" onclick="window.app.removeQueueItem('${t.song_id}')" title="Remove from Queue">
                <i class="fa-solid fa-xmark"></i>
              </button>
            </div>
            <div class="q-score-badge" title="Wolpertinger Critic Q-value expected future reward (+${qVal})">
              <span class="q-val-text">Q: +${qVal}</span>
              <div class="q-score-bar">
                <div class="q-score-fill" style="width:${qFillPct}%"></div>
              </div>
            </div>
          </div>
        </div>
      `;
    }).join('');
  }

  /**
   * Generates a human-readable recommendation reason chip.
   * Uses rec_reason from hybrid API if available; falls back to heuristic.
   */
  _getRecommendationReason(track, idx, uniqueGenres) {
    // Use pre-computed reason from hybrid ranker if available
    if (track.rec_reason && track.rec_reason_icon) {
      const color = track.rec_reason_color || 'var(--accent-blue)';
      return `<i class="fa-solid ${track.rec_reason_icon}" style="color:${color}"></i> ${track.rec_reason}`;
    }

    if (!this.currentTrack) return '<i class="fa-solid fa-brain"></i> RL Pick';

    const energyDiff = (track.energy || 0.5) - (this.currentTrack.energy || 0.5);
    const isDifferentGenre = track.genre !== this.currentTrack.genre;
    const isHighQ = idx === 0;
    const isHighEnergy = (track.energy || 0) > 0.75;
    const isMoodMatch = Math.abs((track.valence || 0.5) - (this.currentTrack.valence || 0.5)) < 0.15;

    if (isDifferentGenre && uniqueGenres > 1) {
      return `<i class="fa-solid fa-compass" style="color:#45aaf2"></i> Explore New`;
    }
    if (isHighQ) {
      return `<i class="fa-solid fa-trophy"></i> Top Q-Value`;
    }
    if (energyDiff > 0.15) {
      return `<i class="fa-solid fa-bolt"></i> Energy Boost`;
    }
    if (energyDiff < -0.15) {
      return `<i class="fa-solid fa-wind"></i> Chill Down`;
    }
    if (isMoodMatch) {
      return `<i class="fa-solid fa-face-smile"></i> Mood Match`;
    }
    if (isHighEnergy) {
      return `<i class="fa-solid fa-fire"></i> High Energy`;
    }
    return `<i class="fa-solid fa-brain"></i> RL Pick`;
  }

  /**
   * Moves a queued song to the top of the queue.
   */
  moveQueueItemToTop(songId) {
    const idx = this.currentQueueTracks.findIndex(s => s.song_id === songId);
    if (idx > 0) {
      const [item] = this.currentQueueTracks.splice(idx, 1);
      this.currentQueueTracks.unshift(item);
      this.renderQueue();
    }
  }

  /**
   * Removes a song from the recommendation queue.
   */
  removeQueueItem(songId) {
    this.currentQueueTracks = this.currentQueueTracks.filter(s => s.song_id !== songId);
    this.renderQueue();
  }

  /**
   * Shuffles the current recommendation queue order.
   */
  shuffleQueue() {
    if (!this.currentQueueTracks || this.currentQueueTracks.length <= 1) return;
    for (let i = this.currentQueueTracks.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [this.currentQueueTracks[i], this.currentQueueTracks[j]] = [this.currentQueueTracks[j], this.currentQueueTracks[i]];
    }
    this.renderQueue();
  }

  /**
   * Clears the entire recommendations queue.
   */
  clearQueue() {
    this.currentQueueTracks = [];
    this.renderQueue();
  }

  /**
   * Adds a track to the queue either at the top or the end.
   */
  addTrackToQueue(track, playNext = false) {
    if (!track || !track.song_id) return;
    if (playNext) {
      this.currentQueueTracks.unshift(track);
    } else {
      this.currentQueueTracks.push(track);
    }
    this.renderQueue();

    // Flash confirmation
    const btn = document.getElementById('btn-quick-add-queue');
    if (btn) {
      const origText = btn.innerHTML;
      btn.innerHTML = '<i class="fa-solid fa-check"></i> Added!';
      setTimeout(() => btn.innerHTML = origText, 1200);
    }
  }

  /**
   * Searches the catalog for songs to add directly to the queue.
   */
  async searchCatalogForQuickAdd(query) {
    const resultsContainer = document.getElementById('quick-add-results');
    if (!resultsContainer) return;

    resultsContainer.innerHTML = '<div class="quick-add-prompt"><i class="fa-solid fa-spinner fa-spin"></i> Searching library...</div>';

    try {
      const res = await fetch(`/api/songs?search=${encodeURIComponent(query)}&limit=8`);
      const songs = await res.json();

      if (!songs || songs.length === 0) {
        resultsContainer.innerHTML = '<div class="quick-add-prompt">No matching songs found in library.</div>';
        return;
      }

      resultsContainer.innerHTML = songs.map(s => {
        const cover = s.image_url || DEFAULT_COVER_ART;
        const songJson = JSON.stringify(s).replace(/"/g, '&quot;');
        return `
          <div class="quick-add-item">
            <div style="display:flex; align-items:center; min-width:0; flex:1;">
              <img class="quick-add-thumb" src="${cover}" alt="Art">
              <div class="quick-add-item-info">
                <span class="quick-add-item-title">${this.escapeHtml(s.title)}</span>
                <span class="quick-add-item-meta">${this.escapeHtml(s.artist_name)} • ${this.escapeHtml(s.genre)}</span>
              </div>
            </div>
            <div class="quick-add-actions">
              <button class="btn-small" onclick="window.app.addTrackToQueue(${songJson}, true); document.getElementById('quick-add-panel').style.display='none';" title="Insert as Next Track">
                <i class="fa-solid fa-arrow-up"></i> Next
              </button>
              <button class="btn-small" onclick="window.app.addTrackToQueue(${songJson}, false); document.getElementById('quick-add-panel').style.display='none';" title="Add to End of Queue">
                <i class="fa-solid fa-plus"></i> Queue
              </button>
              <button class="btn-small" onclick="window.app.playSongDirectly('${s.song_id}'); document.getElementById('quick-add-panel').style.display='none';" title="Play Now">
                <i class="fa-solid fa-play"></i>
              </button>
            </div>
          </div>
        `;
      }).join('');
    } catch (e) {
      console.error('Quick add search error:', e);
      resultsContainer.innerHTML = '<div class="quick-add-prompt">Error querying songs catalog.</div>';
    }
  }

  async handleFeedback(actionType, rewardDelta) {
    if (!this.currentTrack) return;

    try {
      // Collect listen time from behavior tracker
      const listenMs = (window.behaviorCollector)
        ? (window.behaviorCollector.totalListenMs || 0)
        : 0;

      // Notify behavior collector
      if (window.behaviorCollector) {
        window.behaviorCollector.onFeedback(actionType, listenMs);
      }

      const res = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: 'session_demo_01',
          user_id: this.currentUserId,
          song_id: this.currentTrack.song_id,
          action_type: actionType,
          listen_ms: listenMs
        })
      });

      const data = await res.json();
      const r = data.reward !== undefined ? data.reward : rewardDelta;

      this.sessionStep++;
      this.sessionReward += r;
      if (actionType === 'liked') this.sessionLikes++;
      if (actionType.startsWith('skip')) this.sessionSkips++;

      this.historySongIds.push(this.currentTrack.song_id);
      this.historySkipTypes.push(actionType);
      this.historyLikes.push(actionType === 'liked' ? 1 : 0);

      this.updateCounters();
      this.addHistoryChip(this.currentTrack, actionType, r);
      this.updateMiniPlayer();

      await this.fetchNextRecommendation();
    } catch (e) {
      console.error('Feedback submission error:', e);
    }
  }

  updateCounters() {
    document.getElementById('stat-session-step').innerText = this.sessionStep;
    document.getElementById('stat-session-likes').innerText = this.sessionLikes;
    document.getElementById('stat-session-skips').innerText = this.sessionSkips;

    const rewardEl = document.getElementById('stat-session-reward');
    if (rewardEl) {
      rewardEl.innerText = (this.sessionReward >= 0 ? '+' : '') + this.sessionReward.toFixed(2);
      // Trigger pulse animation
      rewardEl.classList.remove('pulse-animate');
      void rewardEl.offsetWidth; // force reflow
      rewardEl.classList.add('pulse-animate');
    }
  }

  addHistoryChip(track, action, reward) {
    if (!track) return;
    
    // Store in structured trajectory history array
    this.sessionTrajectoryHistory.unshift({
      id: Date.now() + Math.random(),
      track: { ...track },
      action: action,
      reward: reward,
      step: this.sessionStep,
      timestamp: new Date()
    });

    this.renderTrajectoryLog();
  }

  /**
   * Renders the session trajectory log chips filtered by category.
   */
  renderTrajectoryLog() {
    const container = document.getElementById('history-chips-container');
    const cumRewardPill = document.getElementById('trajectory-cum-reward');
    const summaryStatsEl = document.getElementById('trajectory-stats-summary');
    if (!container) return;

    // Update cumulative reward pill
    if (cumRewardPill) {
      cumRewardPill.innerText = `${this.sessionReward >= 0 ? '+' : ''}${this.sessionReward.toFixed(2)} R`;
      cumRewardPill.classList.toggle('negative', this.sessionReward < 0);
    }

    // Update summary stats
    if (summaryStatsEl) {
      summaryStatsEl.innerText = `${this.sessionLikes} likes • ${this.sessionSkips} skips • ${this.sessionTrajectoryHistory.length} actions`;
    }

    if (!this.sessionTrajectoryHistory || this.sessionTrajectoryHistory.length === 0) {
      container.innerHTML = '<span class="chip-empty">Session started. Waiting for first interaction...</span>';
      return;
    }

    // Filter items
    const filtered = this.sessionTrajectoryHistory.filter(item => {
      if (this.trajectoryFilter === 'liked') return item.action === 'liked';
      if (this.trajectoryFilter === 'skip') return item.action.startsWith('skip');
      if (this.trajectoryFilter === 'listened') return item.action === 'no_skip';
      if (this.trajectoryFilter === 'disliked') return item.action === 'disliked';
      return true;
    });

    if (filtered.length === 0) {
      container.innerHTML = `<span class="chip-empty">No ${this.trajectoryFilter} events recorded in this session.</span>`;
      return;
    }

    container.innerHTML = filtered.map(item => {
      const action = item.action;
      const track = item.track;
      const reward = item.reward;
      const actionClass = action === 'liked' ? 'liked' : (action.startsWith('skip') ? 'skipped' : (action === 'disliked' ? 'disliked' : 'listened'));

      let icon = 'fa-check';
      let actionLabel = 'Listened';
      if (action === 'liked') { icon = 'fa-heart'; actionLabel = 'Liked'; }
      if (action === 'skip_early') { icon = 'fa-forward-step'; actionLabel = 'Skipped Early'; }
      if (action === 'skip_late') { icon = 'fa-forward'; actionLabel = 'Skipped Late'; }
      if (action === 'disliked') { icon = 'fa-thumbs-down'; actionLabel = 'Disliked'; }

      const timeStr = item.timestamp ? item.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '';
      const tooltip = `Step #${item.step}: ${actionLabel} "${track.title}" (${reward >= 0 ? '+' : ''}${reward.toFixed(1)} Reward) at ${timeStr}`;

      return `
        <div class="history-chip ${actionClass}" title="${this.escapeHtml(tooltip)}" onclick="window.app.playSongDirectly('${track.song_id}')">
          <i class="fa-solid ${icon}"></i>
          <span>${this.escapeHtml(track.title)}</span>
          <span class="chip-reward">${reward >= 0 ? '+' : ''}${reward.toFixed(1)}</span>
        </div>
      `;
    }).join('');
  }

  /**
   * Clears the session trajectory log.
   */
  clearTrajectoryLog() {
    this.sessionTrajectoryHistory = [];
    this.sessionStep = 0;
    this.sessionLikes = 0;
    this.sessionSkips = 0;
    this.sessionReward = 0.0;
    this.updateCounters();
    this.renderTrajectoryLog();
  }

  // OpenSpot Music Hub Methods
  async searchOpenSpot(query) {
    const grid = document.getElementById('openspot-results-grid');
    const titleEl = document.getElementById('openspot-section-title');
    const countEl = document.getElementById('openspot-results-count');
    if (!grid) return;

    this.lastOpenSpotQuery = query;
    grid.innerHTML = '<div class="openspot-empty-state"><i class="fa-solid fa-spinner fa-spin"></i><p>Searching OpenSpot catalog...</p></div>';

    try {
      const res = await fetch(`/api/openspot/search?q=${encodeURIComponent(query)}&limit=24`);
      const data = await res.json();
      const results = data.results || [];

      if (titleEl) titleEl.innerHTML = `<i class="fa-solid fa-magnifying-glass"></i> Search Results: "${query}"`;
      if (countEl) countEl.innerText = `${results.length} tracks`;

      this.renderOpenSpotGrid(results);
    } catch (e) {
      grid.innerHTML = `<div class="openspot-empty-state"><i class="fa-solid fa-triangle-exclamation"></i><p>Error searching OpenSpot: ${e.message}</p></div>`;
    }
  }

  async loadOpenSpotCharts(genre) {
    const grid = document.getElementById('openspot-results-grid');
    const titleEl = document.getElementById('openspot-section-title');
    const countEl = document.getElementById('openspot-results-count');
    if (!grid) return;

    grid.innerHTML = '<div class="openspot-empty-state"><i class="fa-solid fa-spinner fa-spin"></i><p>Loading OpenSpot chart tracks...</p></div>';

    try {
      const res = await fetch(`/api/openspot/charts?genre=${encodeURIComponent(genre)}&limit=24`);
      const data = await res.json();
      const results = data.results || [];

      if (titleEl) titleEl.innerHTML = `<i class="fa-solid fa-fire"></i> Top ${genre} Trending Hits`;
      if (countEl) countEl.innerText = `${results.length} tracks`;

      this.renderOpenSpotGrid(results);
    } catch (e) {
      grid.innerHTML = `<div class="openspot-empty-state"><i class="fa-solid fa-triangle-exclamation"></i><p>Error loading charts: ${e.message}</p></div>`;
    }
  }

  renderOpenSpotGrid(tracks) {
    const grid = document.getElementById('openspot-results-grid');
    if (!grid) return;

    if (!tracks || tracks.length === 0) {
      grid.innerHTML = '<div class="openspot-empty-state"><i class="fa-solid fa-circle-question"></i><p>No tracks found. Try another search query or genre!</p></div>';
      return;
    }

    grid.innerHTML = tracks.map((track, idx) => {
      const cover = track.image_url || DEFAULT_COVER_ART;
      const isImported = this.importedTrackIds.has(track.song_id);
      const jsonStr = encodeURIComponent(JSON.stringify(track));

      return `
        <div class="openspot-track-card" id="card-${track.song_id}">
          <div class="card-top-row">
            <img class="card-cover" src="${cover}" alt="${track.title}">
            <div class="card-content">
              <div class="card-title" title="${track.title}">${track.title}</div>
              <div class="card-artist" title="${track.artist_name}">${track.artist_name}</div>
              <div class="card-album" title="${track.album || ''}">${track.album || ''}</div>
            </div>
          </div>

          <div class="card-acoustic-row">
            <span title="Energy"><i class="fa-solid fa-bolt" style="color:var(--accent-amber);"></i> ${track.energy.toFixed(2)}</span>
            <span title="Danceability"><i class="fa-solid fa-music" style="color:var(--accent-green);"></i> ${track.danceability.toFixed(2)}</span>
            <span title="Tempo"><i class="fa-solid fa-person-walking"></i> ${Math.round(track.tempo)} BPM</span>
            <span title="Duration" style="margin-left:auto;"><i class="fa-solid fa-clock"></i> ${track.duration_formatted || '3:30'}</span>
          </div>

          <div class="card-actions-row">
            <button class="btn-preview-stream" onclick="window.app.previewOpenSpotTrack('${jsonStr}')">
              <i class="fa-solid fa-play"></i> Stream
            </button>
            <button class="btn-import-rl ${isImported ? 'imported' : ''}" id="btn-import-${track.song_id}" onclick="window.app.importOpenSpotTrack('${jsonStr}')">
              <i class="fa-solid ${isImported ? 'fa-check' : 'fa-plus'}"></i> ${isImported ? 'In RL Brain' : 'Import to RL'}
            </button>
          </div>
        </div>
      `;
    }).join('');
  }

  previewOpenSpotTrack(encodedJson) {
    try {
      const track = JSON.parse(decodeURIComponent(encodedJson));
      this.currentTrack = track;
      this.renderCurrentTrack(track);
      this.switchTab('player-tab');
      window.audioSynth.playTrack(track);
      this.loadLyricsForTrack(track);
      this.showToast(`Now Streaming: "${track.title}" by ${track.artist_name}`);
    } catch (e) {
      console.error('Error previewing track:', e);
    }
  }

  async importOpenSpotTrack(encodedJson) {
    try {
      const track = JSON.parse(decodeURIComponent(encodedJson));
      const res = await fetch('/api/openspot/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(track)
      });

      const data = await res.json();
      if (res.ok) {
        this.importedTrackIds.add(track.song_id);
        const btn = document.getElementById(`btn-import-${track.song_id}`);
        if (btn) {
          btn.className = 'btn-import-rl imported';
          btn.innerHTML = '<i class="fa-solid fa-check"></i> In RL Brain';
        }
        this.showToast(`✨ Imported "${track.title}" into active RL Recommendation Brain! Catalog size: ${data.catalog_size}`);
        this.loadCatalog();
      } else {
        alert(`Import failed: ${data.detail || 'Unknown error'}`);
      }
    } catch (e) {
      console.error('Error importing track:', e);
    }
  }

  showToast(message) {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = 'toast-message';
    toast.innerHTML = `<i class="fa-solid fa-circle-check"></i> <span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  async loadCatalog() {
    const searchInp = document.getElementById('catalog-search');
    const genreSel = document.getElementById('catalog-genre-filter');
    const tbody = document.getElementById('catalog-table-body');
    if (!tbody) return;

    const query = searchInp ? searchInp.value : '';
    const genre = genreSel ? genreSel.value : 'All';

    try {
      const res = await fetch(`/api/songs?genre=${encodeURIComponent(genre)}&search=${encodeURIComponent(query)}&limit=40`);
      const songs = await res.json();

      tbody.innerHTML = songs.map(s => {
        const cover = s.image_url || DEFAULT_COVER_ART;
        
        return `
          <tr>
            <td><img class="catalog-thumb" src="${cover}" alt="Cover"></td>
            <td>
              <strong>${s.title}</strong>
              <div style="font-size:11.5px; color:var(--text-secondary);">${s.artist_name}</div>
            </td>
            <td><span class="track-genre-badge">${s.genre}</span></td>
            <td>${s.energy.toFixed(2)}</td>
            <td>${s.danceability.toFixed(2)}</td>
            <td>${s.valence.toFixed(2)}</td>
            <td>${Math.round(s.tempo)} BPM</td>
            <td>
              <div style="display:flex; gap:6px;">
                <button class="btn-small" onclick="window.app.playCatalogSong('${s.song_id}')" title="Play Now">
                  <i class="fa-solid fa-play"></i> Play
                </button>
                <button class="btn-small" onclick="window.app.openRelatedHub('${s.song_id}')" title="Find Related Songs (Vector Search)">
                  <i class="fa-solid fa-wand-magic-sparkles"></i> Related
                </button>
              </div>
            </td>
          </tr>
        `;
      }).join('');
    } catch (e) {
      console.error('Catalog load error:', e);
    }
  }

  async openRelatedHub(songId) {
    try {
      const res = await fetch(`/api/songs/${songId}/related?limit_per_section=8`);
      if (!res.ok) throw new Error('Failed to fetch related songs');
      
      const data = await res.json();
      this.relatedData = data;

      const hub = document.getElementById('related-discovery-hub');
      if (hub) {
        hub.style.display = 'block';
        this.renderSeedSpotlight(data.seed_song);
        
        const twinsCount = data.sections.acoustic_twins.songs.length;
        const sameGCount = data.sections.same_genre.songs.length;
        const crossGCount = data.sections.cross_genre.songs.length;
        const vibeCount = data.sections.vibe_transitions.songs.length;
        const totalCount = twinsCount + sameGCount + crossGCount + vibeCount;

        document.getElementById('count-all').innerText = totalCount;
        document.getElementById('count-twins').innerText = twinsCount;
        document.getElementById('count-same-genre').innerText = sameGCount;
        document.getElementById('count-cross-genre').innerText = crossGCount;
        document.getElementById('count-vibe').innerText = vibeCount;

        this.renderRelatedSections(data, this.currentRelatedFilter);
        hub.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    } catch (e) {
      console.error('Error opening related hub:', e);
    }
  }

  openRelatedHubFromSong(songId) {
    this.switchTab('catalog-tab');
    this.openRelatedHub(songId);
  }

  closeRelatedHub() {
    const hub = document.getElementById('related-discovery-hub');
    if (hub) hub.style.display = 'none';
  }

  renderSeedSpotlight(seed) {
    const container = document.getElementById('seed-spotlight-card');
    if (!container) return;

    const cover = seed.image_url || DEFAULT_COVER_ART;

    container.innerHTML = `
      <div class="seed-main-info">
        <img class="catalog-thumb" style="width:54px; height:54px; border-radius:8px;" src="${cover}" alt="Cover">
        <div class="seed-details">
          <span class="badge-tag highlight"><i class="fa-solid fa-bullseye"></i> Seed Reference Track</span>
          <h3>${seed.title}</h3>
          <p>${seed.artist_name} • <span class="track-genre-badge">${seed.genre}</span></p>
        </div>
      </div>

      <div class="seed-features">
        <span class="pill"><i class="fa-solid fa-bolt"></i> Energy: <b>${seed.energy.toFixed(2)}</b></span>
        <span class="pill"><i class="fa-solid fa-face-smile"></i> Valence: <b>${seed.valence.toFixed(2)}</b></span>
        <span class="pill"><i class="fa-solid fa-music"></i> Dance: <b>${seed.danceability.toFixed(2)}</b></span>
        <span class="pill"><i class="fa-solid fa-person-walking"></i> ${Math.round(seed.tempo)} BPM</span>
      </div>

      <div class="seed-actions">
        <button class="btn-action btn-full" onclick="window.app.playSongDirectly('${seed.song_id}')" style="padding:8px 14px; font-size:12px;">
          <i class="fa-solid fa-play"></i> Play in Studio
        </button>
      </div>
    `;
  }

  renderRelatedSections(data, filter) {
    const container = document.getElementById('related-sections-container');
    if (!container) return;

    const sections = data.sections;
    const sectionKeys = filter === 'all' 
      ? ['acoustic_twins', 'same_genre', 'cross_genre', 'vibe_transitions']
      : [filter];

    let html = '';

    sectionKeys.forEach(secKey => {
      const sec = sections[secKey];
      if (!sec || sec.songs.length === 0) return;

      html += `
        <div class="related-section-block">
          <div class="related-section-header">
            <div class="related-section-header-left">
              <span class="badge-tag"><i class="fa-solid ${sec.icon}"></i> ${sec.badge}</span>
              <h5>${sec.title}</h5>
            </div>
            <span class="related-section-desc">${sec.description}</span>
          </div>

          <div class="related-cards-grid">
            ${sec.songs.map(song => this.renderRelatedCard(song)).join('')}
          </div>
        </div>
      `;
    });

    container.innerHTML = html || '<p style="color:var(--text-secondary); padding:20px;">No matching tracks found for this section.</p>';
  }

  renderRelatedCard(song) {
    const deltas = song.acoustic_deltas || {};
    const eSign = deltas.energy_diff >= 0 ? `+${deltas.energy_diff}` : `${deltas.energy_diff}`;
    const vSign = deltas.valence_diff >= 0 ? `+${deltas.valence_diff}` : `${deltas.valence_diff}`;
    const tSign = deltas.tempo_diff >= 0 ? `+${deltas.tempo_diff}` : `${deltas.tempo_diff}`;
    const cover = song.image_url || DEFAULT_COVER_ART;

    return `
      <div class="related-song-card">
        <div style="display:flex; gap:10px; align-items:flex-start;">
          <img class="catalog-thumb" src="${cover}" alt="Cover" style="width:42px; height:42px; flex-shrink:0;">
          <div style="flex:1; min-width:0;">
            <div class="related-card-top">
              <div style="min-width:0;">
                <div class="related-card-title" title="${song.title}">${song.title}</div>
                <div class="related-card-artist">${song.artist_name} • <span class="track-genre-badge" style="font-size:9px;">${song.genre}</span></div>
              </div>
              <span class="match-gauge-badge">${song.match_percentage}% Match</span>
            </div>

            <div class="acoustic-deltas-row" style="margin-top:6px;">
              <span class="delta-pill ${deltas.energy_diff >= 0 ? 'positive' : 'neutral'}">Energy: ${eSign}</span>
              <span class="delta-pill neutral">Mood: ${vSign}</span>
              <span class="delta-pill neutral">Tempo: ${tSign} BPM</span>
            </div>
          </div>
        </div>

        <div class="related-card-actions">
          <button class="btn-card-play" onclick="window.app.playSongDirectly('${song.song_id}')" title="Play in RL Live Studio">
            <i class="fa-solid fa-play"></i> Studio
          </button>
          <button class="btn-card-icon" onclick="window.app.openRelatedHub('${song.song_id}')" title="Explore Related Tracks">
            <i class="fa-solid fa-wand-magic-sparkles"></i>
          </button>
        </div>
      </div>
    `;
  }

  async updatePlayerRelatedShelf(songId) {
    try {
      const res = await fetch(`/api/songs/${songId}/related?limit_per_section=4`);
      if (!res.ok) return;

      const data = await res.json();
      const container = document.getElementById('player-related-cards');
      const genreBadge = document.getElementById('shelf-current-genre');
      
      if (genreBadge && data.seed_song) {
        genreBadge.innerText = data.seed_song.genre;
      }

      if (!container) return;

      const topItems = [
        ...data.sections.acoustic_twins.songs.slice(0, 3),
        ...data.sections.cross_genre.songs.slice(0, 3)
      ];

      container.innerHTML = topItems.map(song => {
        const cover = song.image_url || DEFAULT_COVER_ART;

        return `
          <div class="mini-related-card">
            <div class="mini-card-top">
              <img class="catalog-thumb" src="${cover}" alt="Cover" style="width:32px; height:32px; flex-shrink:0;">
              <div style="flex:1; min-width:0; margin-left:6px;">
                <div class="mini-title" title="${song.title}">${song.title}</div>
                <div class="mini-artist">${song.artist_name}</div>
              </div>
              <span class="mini-badge">${song.match_percentage}%</span>
            </div>
            <div style="display:flex; gap:6px; margin-top:6px;">
              <button class="btn-card-play" style="padding:4px 8px; font-size:10.5px;" onclick="window.app.playSongDirectly('${song.song_id}')">
                <i class="fa-solid fa-play"></i> Play
              </button>
              <button class="btn-card-icon" style="padding:4px 8px; font-size:10.5px;" onclick="window.app.openRelatedHubFromSong('${song.song_id}')" title="Explore Sections">
                <i class="fa-solid fa-wand-magic-sparkles"></i>
              </button>
            </div>
          </div>
        `;
      }).join('');
    } catch (e) {
      console.error('Error updating player related shelf:', e);
    }
  }

  async playSongDirectly(songId) {
    try {
      // FIX: Use song_id as direct search key; fallback to stream/resolve for audio URL
      const res = await fetch(`/api/songs?search=${encodeURIComponent(songId)}&limit=5`);
      const songs = await res.json();
      // Match by song_id exactly (search endpoint does text match on song_id field too)
      let track = songs.find(s => s.song_id === songId) || (songs.length > 0 ? songs[0] : null);

      if (!track) {
        // If text search failed to find exact match, use stream/resolve to get metadata
        const resolveRes = await fetch(`/api/stream/resolve?song_id=${encodeURIComponent(songId)}`);
        const resolveData = await resolveRes.json();
        if (resolveData && resolveData.song_id) {
          track = resolveData;
        }
      }

      if (track) {
        this.currentTrack = track;
        this.renderCurrentTrack(this.currentTrack);
        this.switchTab('player-tab');

        // Notify behavior collector
        if (window.behaviorCollector) {
          window.behaviorCollector.setCurrentTrack(this.currentTrack);
        }

        await window.audioSynth.playTrack(this.currentTrack);
        this.loadLyricsForTrack(this.currentTrack);

        window.rlVisualizer.updateRadar(this.userProfile, this.currentTrack);
        window.rlVisualizer.addTrajectoryPoint(this.currentTrack.energy, this.currentTrack.valence, `#${this.sessionStep + 1}`);
        this.updatePlayerRelatedShelf(this.currentTrack.song_id);

        this.historySongIds.push(this.currentTrack.song_id);
        this.historySkipTypes.push('no_skip');
        this.historyLikes.push(1);
        this.sessionLikes++;
        this.updateCounters();
        this.updateMiniPlayer();
      }
    } catch (e) {
      console.error('Error playing song directly:', e);
    }
  }

  async playCatalogSong(songId) {
    await this.playSongDirectly(songId);
  }

  // =========================================================================
  // Mini Player (visible when on non-player tabs)
  // =========================================================================

  updateMiniPlayer() {
    const mini = document.getElementById('mini-player-bar');
    if (!mini || !this.currentTrack) return;

    const t = this.currentTrack;
    const cover = t.image_url || DEFAULT_COVER_ART;

    document.getElementById('mini-cover').src = cover;
    document.getElementById('mini-title').innerText = t.title;
    document.getElementById('mini-artist').innerText = t.artist_name;

    const isPlaying = window.audioSynth.isPlaying;
    const playBtn = document.getElementById('mini-play-btn');
    if (playBtn) playBtn.innerHTML = isPlaying
      ? '<i class="fa-solid fa-pause"></i>'
      : '<i class="fa-solid fa-play"></i>';
  }

  switchTab(tabId) {
    document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

    const targetNav = document.querySelector(`.nav-item[data-tab="${tabId}"]`);
    if (targetNav) targetNav.classList.add('active');

    const pane = document.getElementById(tabId);
    if (pane) pane.classList.add('active');

    // Show mini-player when not on player tab
    const mini = document.getElementById('mini-player-bar');
    if (mini) {
      mini.classList.toggle('visible', tabId !== 'player-tab' && this.currentTrack !== null);
    }

    if (tabId === 'openspot-tab' && (!this.lastOpenSpotQuery || this.lastOpenSpotQuery === '')) {
      this.loadOpenSpotCharts('Pop');
    }

    // Load user insights when switching to player tab
    if (tabId === 'player-tab') {
      this.loadUserInsights();
    }
  }

  // =========================================================================
  // User Insights Panel
  // =========================================================================

  async loadUserInsights() {
    try {
      const res = await fetch(`/api/behavior/profile/${this.currentUserId}`);
      if (!res.ok) return;
      const profile = await res.json();
      this.renderUserInsightsPanel(profile);
    } catch (e) {
      // Silently ignore — insights are non-critical
    }
  }

  renderUserInsightsPanel(profile) {
    const panel = document.getElementById('user-insights-panel');
    if (!panel) return;

    const genreWeights = profile.genre_weights || {};
    const topGenres = Object.entries(genreWeights)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);

    const artistWeights = profile.artist_weights || {};
    const topArtists = Object.entries(artistWeights)
      .filter(([, w]) => w > 0)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 4);

    const mood = profile.session_mood || 'neutral';
    const energyTrend = profile.energy_trend || 'mixed';
    const ctx = profile.current_context || {};
    const centroid = profile.acoustic_centroid || {};

    const moodEmoji = { happy: '😄', melancholic: '😔', neutral: '😐' }[mood] || '🎵';
    const energyEmoji = { high: '⚡', low: '🌊', mixed: '🎭' }[energyTrend] || '🎵';

    // Genre heatmap bars
    const maxW = topGenres.length > 0 ? Math.max(...topGenres.map(([,w]) => Math.abs(w))) || 1 : 1;
    const genreHtml = topGenres.map(([genre, weight]) => {
      const pct = Math.round((Math.abs(weight) / maxW) * 100);
      const isPos = weight >= 0;
      const color = isPos ? '#26de81' : '#fc5c65';
      return `
        <div class="insight-genre-row">
          <span class="insight-genre-label">${genre}</span>
          <div class="insight-bar-track">
            <div class="insight-bar-fill" style="width:${pct}%; background:${color};"></div>
          </div>
          <span class="insight-genre-weight" style="color:${color}">${isPos ? '+' : ''}${(weight * 100).toFixed(0)}%</span>
        </div>
      `;
    }).join('');

    const artistHtml = topArtists.map(([artist, weight]) => {
      const pct = Math.round((Math.abs(weight) / maxW) * 80);
      return `<span class="insight-artist-chip" title="Affinity: ${(weight*100).toFixed(0)}%">${artist}</span>`;
    }).join('');

    const trajHtml = (profile.valence_trajectory || []).slice(-8).map((pt, i) => {
      const h = Math.round(pt.valence * 40) + 4;
      const color = pt.event_type === 'liked' ? '#26de81' : (pt.event_type?.startsWith('skip') ? '#fc5c65' : 'var(--accent-blue)');
      return `<div class="insight-traj-bar" style="height:${h}px; background:${color};" title="${pt.event_type}: valence ${pt.valence}"></div>`;
    }).join('');

    panel.innerHTML = `
      <div class="insights-header">
        <span class="insights-title"><i class="fa-solid fa-chart-simple"></i> Your Taste Profile</span>
        <span class="insights-context-badge">${ctx.label || '🎵 Listening'}</span>
      </div>

      <div class="insights-mood-row">
        <div class="insights-mood-chip">
          <span>${moodEmoji}</span>
          <span>Mood: <b>${mood}</b></span>
        </div>
        <div class="insights-mood-chip">
          <span>${energyEmoji}</span>
          <span>Energy: <b>${energyTrend}</b></span>
        </div>
        <div class="insights-mood-chip">
          <i class="fa-solid fa-heart" style="color:#fc5c65;"></i>
          <span><b>${profile.liked_count || 0}</b> likes</span>
        </div>
        <div class="insights-mood-chip">
          <i class="fa-solid fa-forward" style="color:var(--accent-amber);"></i>
          <span><b>${profile.skipped_count || 0}</b> skips</span>
        </div>
      </div>

      ${topGenres.length > 0 ? `
      <div class="insights-section">
        <div class="insights-section-title">Genre Heatmap</div>
        <div class="insight-genres">${genreHtml}</div>
      </div>
      ` : ''}

      ${topArtists.length > 0 ? `
      <div class="insights-section">
        <div class="insights-section-title">Liked Artists</div>
        <div class="insight-artists">${artistHtml}</div>
      </div>
      ` : ''}

      ${centroid.energy !== undefined ? `
      <div class="insights-section">
        <div class="insights-section-title">Your Sound Centroid</div>
        <div class="insight-centroid-row">
          <span class="insight-centroid-chip"><i class="fa-solid fa-bolt"></i> Energy <b>${(centroid.energy || 0).toFixed(2)}</b></span>
          <span class="insight-centroid-chip"><i class="fa-solid fa-face-smile"></i> Valence <b>${(centroid.valence || 0).toFixed(2)}</b></span>
          <span class="insight-centroid-chip"><i class="fa-solid fa-music"></i> Dance <b>${(centroid.danceability || 0).toFixed(2)}</b></span>
          <span class="insight-centroid-chip"><i class="fa-solid fa-person-walking"></i> ${Math.round(centroid.tempo || 120)} BPM</span>
        </div>
      </div>
      ` : ''}

      ${trajHtml ? `
      <div class="insights-section">
        <div class="insights-section-title">Mood Trajectory</div>
        <div class="insight-traj-chart">${trajHtml}</div>
        <div class="insight-traj-label">valence over recent interactions</div>
      </div>
      ` : ''}
    `;
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.app = new App();
});
