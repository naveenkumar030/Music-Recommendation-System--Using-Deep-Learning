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

    // Custom Playlists State
    const savedPl = localStorage.getItem('soundspace_custom_playlists');
    if (savedPl) {
      try {
        this.customPlaylists = JSON.parse(savedPl);
      } catch (e) {
        this.customPlaylists = [];
      }
    } else {
      this.customPlaylists = [
        {
          id: 'pl_lofi_study',
          name: 'Chill Study Beats',
          desc: 'Relaxing lo-fi hip hop and jazz beats for focus',
          tracks: []
        },
        {
          id: 'pl_workout_hype',
          name: 'Workout Energy 2026',
          desc: 'High BPM electronic and hip hop bangers',
          tracks: []
        }
      ];
      localStorage.setItem('soundspace_custom_playlists', JSON.stringify(this.customPlaylists));
    }
    this.pendingTrackForPlaylist = null;

    this.initEventListeners();
    this.loadInitialData();
    this.renderCustomPlaylists();
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
        if (this.currentTrack) {
          if (window.behaviorCollector) window.behaviorCollector.onPlaylistSave(this.currentTrack);
          this.handleFeedback('saved', 1.5);
          btnSave.classList.add('saved');
          setTimeout(() => btnSave.classList.remove('saved'), 2000);
          this.openAddToPlaylistModal(this.currentTrack);
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

    // Global Spotify Top Bar Search
    const globalSearch = document.getElementById('global-search-input');
    if (globalSearch) {
      globalSearch.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          const q = globalSearch.value.trim();
          if (q) {
            this.switchTab('openspot-tab');
            const openInput = document.getElementById('openspot-search-input');
            if (openInput) openInput.value = q;
            this.searchOpenSpot(q);
          }
        }
      });
      globalSearch.addEventListener('input', (e) => {
        clearTimeout(this.searchDebounceTimer);
        const q = e.target.value.trim();
        if (q.length >= 3) {
          this.searchDebounceTimer = setTimeout(() => {
            this.switchTab('openspot-tab');
            const openInput = document.getElementById('openspot-search-input');
            if (openInput) openInput.value = q;
            this.searchOpenSpot(q);
          }, 500);
        }
      });
    }

    // History Back / Forward navigation buttons
    const btnNavBack = document.getElementById('btn-nav-back');
    const btnNavFwd = document.getElementById('btn-nav-forward');
    if (btnNavBack) btnNavBack.addEventListener('click', () => this.switchTab('home-tab'));
    if (btnNavFwd) btnNavFwd.addEventListener('click', () => this.switchTab('player-tab'));

    // Persistent Spotify Player Bar Controls
    const spPlay = document.getElementById('sp-btn-play');
    const spPrev = document.getElementById('sp-btn-prev');
    const spNext = document.getElementById('sp-btn-next');
    const spShuffle = document.getElementById('sp-btn-shuffle');
    const spLike = document.getElementById('sp-btn-like');
    const spSave = document.getElementById('sp-btn-save');
    const spScrubber = document.getElementById('sp-audio-scrubber');
    const spVolSlider = document.getElementById('sp-volume-slider');
    const spVolIcon = document.getElementById('sp-btn-volume-icon');
    const spLyricsToggle = document.getElementById('sp-btn-lyrics-toggle');
    const spQueueToggle = document.getElementById('sp-btn-queue-toggle');
    const spTheater = document.getElementById('sp-btn-theater');

    if (spPlay) {
      spPlay.addEventListener('click', () => {
        if (this.currentTrack) window.audioSynth.toggle(this.currentTrack);
      });
    }
    if (spPrev) {
      spPrev.addEventListener('click', () => {
        window.audioSynth.seek(0);
        this.showToast('⏮ Replaying track from start');
      });
    }
    if (spNext) {
      spNext.addEventListener('click', () => this.handleFeedback('skip_early', -1.0));
    }
    if (spShuffle) {
      spShuffle.addEventListener('click', () => {
        this.currentModelType = this.currentModelType === 'wolpertinger' ? 'baseline' : 'wolpertinger';
        spShuffle.classList.toggle('active', this.currentModelType === 'wolpertinger');
        this.showToast(this.currentModelType === 'wolpertinger' ? '⚡ RL Smart Shuffle Enabled' : '📻 Two-Tower Baseline Mode');
        this.fetchNextRecommendation();
      });
    }
    if (spLike) {
      spLike.addEventListener('click', () => {
        this.handleFeedback('liked', 1.0);
        spLike.classList.add('liked');
        setTimeout(() => spLike.classList.remove('liked'), 2500);
      });
    }
    if (spSave) {
      spSave.addEventListener('click', () => {
        if (this.currentTrack) {
          this.openAddToPlaylistModal(this.currentTrack);
        }
      });
    }

    // Modal Listeners for Playlist Management
    const btnCreatePl = document.getElementById('btn-create-playlist');
    const btnCloseCreatePl = document.getElementById('btn-close-create-playlist');
    const btnCancelCreatePl = document.getElementById('btn-cancel-create-playlist');
    const btnSubmitCreatePl = document.getElementById('btn-submit-create-playlist');

    if (btnCreatePl) btnCreatePl.addEventListener('click', () => this.openCreatePlaylistModal());
    if (btnCloseCreatePl) btnCloseCreatePl.addEventListener('click', () => this.closeCreatePlaylistModal());
    if (btnCancelCreatePl) btnCancelCreatePl.addEventListener('click', () => this.closeCreatePlaylistModal());
    if (btnSubmitCreatePl) btnSubmitCreatePl.addEventListener('click', () => this.submitCreatePlaylist());

    const btnCloseAddPl = document.getElementById('btn-close-add-playlist');
    const btnCancelAddPl = document.getElementById('btn-cancel-add-playlist');
    const btnOpenCreateFromAdd = document.getElementById('btn-open-create-from-add');

    if (btnCloseAddPl) btnCloseAddPl.addEventListener('click', () => this.closeAddToPlaylistModal());
    if (btnCancelAddPl) btnCancelAddPl.addEventListener('click', () => this.closeAddToPlaylistModal());
    if (btnOpenCreateFromAdd) {
      btnOpenCreateFromAdd.addEventListener('click', () => {
        this.closeAddToPlaylistModal();
        this.openCreatePlaylistModal();
      });
    }
    if (spScrubber) {
      spScrubber.addEventListener('input', (e) => {
        window.audioSynth.seek(parseFloat(e.target.value));
      });
    }
    if (spVolSlider) {
      spVolSlider.addEventListener('input', (e) => {
        const v = parseFloat(e.target.value) / 100;
        window.audioSynth.setVolume(v);
        if (spVolIcon) {
          spVolIcon.innerHTML = v === 0 ? '<i class="fa-solid fa-volume-xmark"></i>' : (v < 0.4 ? '<i class="fa-solid fa-volume-low"></i>' : '<i class="fa-solid fa-volume-high"></i>');
        }
      });
    }
    if (spVolIcon) {
      spVolIcon.addEventListener('click', () => {
        const cur = window.audioSynth.gainNode ? window.audioSynth.gainNode.gain.value : 1;
        const newV = cur > 0 ? 0 : 0.8;
        window.audioSynth.setVolume(newV);
        if (spVolSlider) spVolSlider.value = newV * 100;
        spVolIcon.innerHTML = newV === 0 ? '<i class="fa-solid fa-volume-xmark"></i>' : '<i class="fa-solid fa-volume-high"></i>';
      });
    }
    if (spLyricsToggle) {
      spLyricsToggle.addEventListener('click', () => {
        this.switchTab('player-tab');
        const lyricsCard = document.getElementById('player-lyrics-card');
        if (lyricsCard && lyricsCard.classList.contains('collapsed')) {
          lyricsCard.classList.remove('collapsed');
        }
      });
    }
    if (spQueueToggle) {
      spQueueToggle.addEventListener('click', () => {
        this.switchTab('player-tab');
        const qContainer = document.getElementById('queue-list-container');
        if (qContainer) qContainer.scrollIntoView({ behavior: 'smooth' });
      });
    }
    if (spTheater) {
      spTheater.addEventListener('click', () => this.toggleTheaterMode(true));
    }

    // Library Action Buttons
    const btnLibPlayAll = document.getElementById('btn-lib-play-all');
    const btnLibRefresh = document.getElementById('btn-lib-refresh');
    if (btnLibPlayAll) {
      btnLibPlayAll.addEventListener('click', () => {
        const likedTracks = this.sessionTrajectoryHistory.filter(h => h.action === 'liked').map(h => h.track);
        if (likedTracks.length > 0) {
          this.playSongDirectly(likedTracks[0].song_id);
          this.showToast(`▶ Playing Liked Songs playlist (${likedTracks.length} tracks)!`);
        } else if (this.currentTrack) {
          this.playSongDirectly(this.currentTrack.song_id);
        }
      });
    }
    if (btnLibRefresh) {
      btnLibRefresh.addEventListener('click', () => {
        this.renderSpotifyLibrary();
        this.showToast('📚 Library updated!');
      });
    }

    // Interactive Feedback Action Buttons
    const btnLike = document.getElementById('btn-feedback-like');
    if (btnLike) btnLike.addEventListener('click', () => this.handleFeedback('liked', 1.0));

    const btnFull = document.getElementById('btn-feedback-full');
    if (btnFull) btnFull.addEventListener('click', () => this.handleFeedback('no_skip', 0.5));

    const btnSkip = document.getElementById('btn-feedback-skip');
    if (btnSkip) btnSkip.addEventListener('click', () => this.handleFeedback('skip_early', -1.0));

    const btnDislike = document.getElementById('btn-feedback-dislike');
    if (btnDislike) btnDislike.addEventListener('click', () => this.handleFeedback('disliked', -1.5));

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

    // Catalog Tab Genre Pills (Browse All Genres & Moods)
    document.querySelectorAll('.pill-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.pill-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const genre = btn.getAttribute('data-genre');
        const genreSel = document.getElementById('catalog-genre-filter');
        if (genreSel) genreSel.value = genre;
        this.loadCatalog();
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
    // Sanity check: log total catalog size vs /api/health
    try {
      const [healthRes, songsRes] = await Promise.all([
        fetch('/api/health'),
        fetch('/api/songs')
      ]);
      const health = await healthRes.json();
      const songsData = await songsRes.json();
      const catalogSize = health.catalog_size || 0;
      const totalInResponse = songsData.total || 0;
      console.log(`[Catalog Sanity] /api/health catalog_size=${catalogSize}, /api/songs total=${totalInResponse}`);
      if (catalogSize > 0 && totalInResponse !== catalogSize) {
        console.warn(`[Catalog Sanity] MISMATCH: catalog_size=${catalogSize} but /api/songs total=${totalInResponse}`);
      } else {
        console.log('[Catalog Sanity] ✅ Catalog size matches.');
      }
    } catch (e) {
      console.warn('[Catalog Sanity] Could not verify catalog size:', e);
    }
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

      // Throw on non-OK so the catch block triggers the fallback
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.detail || `Hybrid recommend failed (HTTP ${res.status})`);
      }

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

        // Show which pipeline is active
        const modeTag = document.getElementById('audio-mode-tag');
        if (modeTag && data.active_pipeline) {
          const isRL = data.active_pipeline === 'wolpertinger_rl';
          modeTag.innerHTML = isRL
            ? '<i class="fa-solid fa-brain"></i> Wolpertinger RL Active'
            : '<i class="fa-solid fa-tower-cell"></i> Hybrid Heuristic Mode';
          modeTag.title = isRL
            ? 'Trained Wolpertinger RL agent is scoring recommendations'
            : 'Using heuristic hybrid ranker (RL agent not loaded)';
        }

        // Render enriched queue with hybrid reasons
        this.renderQueue(data.recommendations.slice(1), data.q_score_distribution || []);

        if (data.q_score_distribution && data.q_score_distribution.length > 0) {
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

        // Update Spotify Home Shelves & Library
        this.renderSpotifyHomeShelves(data.recommendations, data.q_score_distribution || []);
        this.renderSpotifyLibrary();
        this.updateSpotifyPlayerBar();

        // Update interactive RL State loop
        if (window.rlVisualizer && window.rlVisualizer.updateRLStateLoop) {
          const topG = this.userProfile?.genre_affinities ? Object.keys(this.userProfile.genre_affinities)[0] : 'Lo-Fi';
          window.rlVisualizer.updateRLStateLoop(
            {
              user_id: this.currentUserId,
              genre: topG,
              artist: this.currentTrack.artist_name,
              mood: { energy: this.currentTrack.energy, valence: this.currentTrack.valence },
              current_track: this.currentTrack,
              total_interactions: this.sessionStep
            },
            data.q_score_distribution || []
          );
        }
      }
    } catch (e) {
      console.error('Hybrid recommendation fetch error:', e);
      // Fallback to original recommend (real RL) endpoint
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
            slate_size: 10
          })
        });

        if (!res.ok) {
          const errBody = await res.json().catch(() => ({}));
          throw new Error(errBody.detail || `RL recommend also failed (HTTP ${res.status})`);
        }

        const data = await res.json();
        if (data.recommendations && data.recommendations.length > 0) {
          this.currentTrack = data.recommendations[0];
          this.renderCurrentTrack(this.currentTrack);
          this.renderQueue(data.recommendations.slice(1), data.q_score_distribution || []);
          if (data.q_score_distribution && data.q_score_distribution.length > 0) {
            window.rlVisualizer.updateQChart(data.q_score_distribution);
          }
          window.rlVisualizer.updateRadar(this.userProfile, this.currentTrack);
          this.loadLyricsForTrack(this.currentTrack);
          this.updateMiniPlayer();
        }
      } catch(e2) {
        console.error('Fallback RL recommendation also failed:', e2);
        this.showToast(`⚠️ Recommendation engine error: ${e2.message}. Check server logs.`, 'error');
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
   * Searches the ENTIRE local catalog for songs to add directly to the queue.
   * Uses limit=200 so search-to-select covers the full catalog for any reasonable query.
   */
  async searchCatalogForQuickAdd(query) {
    const resultsContainer = document.getElementById('quick-add-results');
    if (!resultsContainer) return;

    resultsContainer.innerHTML = '<div class="quick-add-prompt"><i class="fa-solid fa-spinner fa-spin"></i> Searching library...</div>';

    try {
      // No genre filter, no limit cap — returns all matches up to 500 via search path
      const res = await fetch(`/api/songs?search=${encodeURIComponent(query)}`);
      const data = await res.json();
      const songs = data.songs || data;  // backward-compat if still bare array
      const total = data.total != null ? data.total : songs.length;

      if (!songs || songs.length === 0) {
        resultsContainer.innerHTML = '<div class="quick-add-prompt">No matching songs found in library.</div>';
        return;
      }

      const countEl = `<div class="quick-add-prompt" style="margin-bottom:4px; font-size:11px; color:var(--text-muted);"><i class="fa-solid fa-compact-disc"></i> ${total} match${total !== 1 ? 'es' : ''} in catalog</div>`;

      const rows = songs.map(s => {
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
      });

      resultsContainer.innerHTML = countEl + rows.join('');
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

      // Update interactive RL closed loop visualizer
      if (window.rlVisualizer && window.rlVisualizer.updateRLStateLoop) {
        const uState = data.user_state || {
          user_id: this.currentUserId,
          current_track: this.currentTrack,
          total_interactions: this.sessionStep
        };
        uState.current_track = this.currentTrack;
        window.rlVisualizer.updateRLStateLoop(
          uState,
          this.currentQDist,
          { action_type: actionType, reward: r },
          {
            policy_updated: data.policy_updated,
            loss_info: data.loss_info,
            buffer_size: data.buffer_size
          }
        );
      }

      if (data.policy_updated) {
        const lossVal = data.loss_info?.critic_loss ? data.loss_info.critic_loss.toFixed(4) : '';
        this.showToast(`🧠 RL Policy Updated via TD-Step ${lossVal ? `(Loss: ${lossVal})` : ''}`, 'success');
      }

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
      if (!res.ok) throw new Error(`Search failed (${res.status})`);
      const data = await res.json();
      const results = data.results || [];

      if (titleEl) titleEl.innerHTML = `<i class="fa-solid fa-magnifying-glass"></i> Search Results: "${query}"`;
      if (countEl) countEl.innerText = `${results.length} tracks`;

      this.openSpotCurrentResults = results;
      this.selectedOpenSpotIds.clear();
      this.renderOpenSpotGrid(results);
      this._updateOpenSpotBulkActions();
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
      if (!res.ok) throw new Error(`Charts failed (${res.status})`);
      const data = await res.json();
      const results = data.results || [];

      if (titleEl) titleEl.innerHTML = `<i class="fa-solid fa-fire"></i> Top ${genre} Trending Hits`;
      if (countEl) countEl.innerText = `${results.length} tracks`;

      this.openSpotCurrentResults = results;
      this.selectedOpenSpotIds.clear();
      this.renderOpenSpotGrid(results);
      this._updateOpenSpotBulkActions();
    } catch (e) {
      grid.innerHTML = `<div class="openspot-empty-state"><i class="fa-solid fa-triangle-exclamation"></i><p>Error loading charts: ${e.message}</p></div>`;
    }
  }

  /** Updates the bulk-action bar in the OpenSpot results panel based on selection state. */
  _updateOpenSpotBulkActions() {
    const bar = document.getElementById('openspot-bulk-bar');
    if (!bar) return;
    const total = this.openSpotCurrentResults.length;
    const selected = this.selectedOpenSpotIds.size;
    bar.style.display = total > 0 ? 'flex' : 'none';

    const selCountEl = bar.querySelector('#openspot-sel-count');
    if (selCountEl) selCountEl.innerText = selected > 0 ? `${selected} selected` : `${total} tracks`;

    const btnImportSel = bar.querySelector('#btn-import-selected');
    if (btnImportSel) {
      btnImportSel.disabled = selected === 0;
      btnImportSel.innerHTML = `<i class="fa-solid fa-brain"></i> Import Selected${selected > 0 ? ` (${selected})` : ''}`;
    }
  }

  /** Toggles a track's checkbox selection in the OpenSpot grid. */
  toggleOpenSpotSelection(songId) {
    if (this.selectedOpenSpotIds.has(songId)) {
      this.selectedOpenSpotIds.delete(songId);
    } else {
      this.selectedOpenSpotIds.add(songId);
    }
    // Reflect on card UI
    const card = document.getElementById(`card-${songId}`);
    if (card) card.classList.toggle('selected', this.selectedOpenSpotIds.has(songId));
    this._updateOpenSpotBulkActions();
  }

  /** Selects or deselects all visible OpenSpot tracks. */
  toggleSelectAllOpenSpot() {
    const allSelected = this.selectedOpenSpotIds.size === this.openSpotCurrentResults.length;
    this.selectedOpenSpotIds.clear();
    if (!allSelected) {
      this.openSpotCurrentResults.forEach(t => this.selectedOpenSpotIds.add(t.song_id));
    }
    // Re-render to sync checkboxes
    this.renderOpenSpotGrid(this.openSpotCurrentResults);
    this._updateOpenSpotBulkActions();
  }

  /** Adds all currently-visible OpenSpot results to the player queue. */
  addAllOpenSpotToQueue() {
    if (!this.openSpotCurrentResults || this.openSpotCurrentResults.length === 0) {
      this.showToast('No tracks loaded — search or pick a genre first.');
      return;
    }
    let added = 0;
    this.openSpotCurrentResults.forEach(track => {
      if (!this.currentQueueTracks.find(t => t.song_id === track.song_id)) {
        this.currentQueueTracks.push(track);
        added++;
      }
    });
    this.renderQueue();
    this.showToast(`✅ Added ${added} OpenSpot tracks to queue!`);
    this.switchTab('player-tab');
  }

  /** Plays the first OpenSpot result and queues the rest. */
  playAllOpenSpot() {
    if (!this.openSpotCurrentResults || this.openSpotCurrentResults.length === 0) {
      this.showToast('No tracks loaded — search or pick a genre first.');
      return;
    }
    const [first, ...rest] = this.openSpotCurrentResults;
    // Queue the rest
    rest.forEach(track => {
      if (!this.currentQueueTracks.find(t => t.song_id === track.song_id)) {
        this.currentQueueTracks.push(track);
      }
    });
    this.renderQueue();
    this.previewOpenSpotTrack(encodeURIComponent(JSON.stringify(first)));
    this.showToast(`▶ Playing "${first.title}" + ${rest.length} more queued!`);
  }

  /** Batch-imports all checkbox-selected OpenSpot tracks into the RL catalog. */
  async importSelectedOpenSpot() {
    const selectedTracks = this.openSpotCurrentResults.filter(t => this.selectedOpenSpotIds.has(t.song_id));
    if (selectedTracks.length === 0) {
      this.showToast('Select at least one track to import.');
      return;
    }

    const btn = document.getElementById('btn-import-selected');
    if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Importing...'; }

    try {
      const res = await fetch('/api/openspot/import/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tracks: selectedTracks })
      });
      if (!res.ok) throw new Error(`Import batch failed (${res.status})`);
      const data = await res.json();

      data.imported.forEach(sid => this.importedTrackIds.add(sid));

      if (data.imported.length > 0) {
        this.showToast(`✨ Imported ${data.imported.length} tracks into RL catalog! Catalog now: ${data.catalog_size} songs.`);
      }
      if (data.failed.length > 0) {
        this.showToast(`⚠️ ${data.failed.length} tracks failed to import. Check console for details.`, 'error');
        console.warn('[BatchImport] Failed tracks:', data.failed);
      }

      // Re-render grid to update Import buttons
      this.renderOpenSpotGrid(this.openSpotCurrentResults);
      this.selectedOpenSpotIds.clear();
      this._updateOpenSpotBulkActions();
      this.loadCatalog();
    } catch (e) {
      console.error('Batch import error:', e);
      this.showToast(`❌ Batch import error: ${e.message}`, 'error');
    } finally {
      if (btn) { btn.disabled = false; this._updateOpenSpotBulkActions(); }
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
      const isSelected = this.selectedOpenSpotIds.has(track.song_id);
      const jsonStr = encodeURIComponent(JSON.stringify(track));

      return `
        <div class="openspot-track-card ${isSelected ? 'selected' : ''}" id="card-${track.song_id}">
          <div class="card-top-row">
            <label class="openspot-checkbox-wrap" title="Select for batch import">
              <input type="checkbox" class="openspot-track-checkbox" ${isSelected ? 'checked' : ''}
                onchange="window.app.toggleOpenSpotSelection('${track.song_id}')">
            </label>
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
            <button class="btn-small" onclick="window.app.addTrackToQueue(JSON.parse(decodeURIComponent('${jsonStr}')), false)" title="Add to Queue">
              <i class="fa-solid fa-plus"></i> Queue
            </button>
            <button class="btn-import-rl ${isImported ? 'imported' : ''}" id="btn-import-${track.song_id}" onclick="window.app.importOpenSpotTrack('${jsonStr}')">
              <i class="fa-solid ${isImported ? 'fa-check' : 'fa-brain'}"></i> ${isImported ? 'In RL' : 'Import RL'}
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
      // Also add remaining OpenSpot results to the queue so skipping works
      if (this.openSpotCurrentResults && this.openSpotCurrentResults.length > 1) {
        const remaining = this.openSpotCurrentResults.filter(t => t.song_id !== track.song_id);
        remaining.forEach(t => {
          if (!this.currentQueueTracks.find(q => q.song_id === t.song_id)) {
            this.currentQueueTracks.push(t);
          }
        });
        this.renderQueue();
      }
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
          btn.innerHTML = '<i class="fa-solid fa-check"></i> In RL';
        }
        // Also add to queue for immediate use
        this.addTrackToQueue(track, false);
        this.showToast(`✨ Imported "${track.title}" into RL catalog (${data.catalog_size} total). Added to queue!`);
        this.loadCatalog();
      } else {
        this.showToast(`❌ Import failed: ${data.detail || 'Unknown error'}`, 'error');
      }
    } catch (e) {
      console.error('Error importing track:', e);
    }
  }

  showToast(message, type = 'success') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = 'toast-message' + (type === 'error' ? ' toast-error' : '');
    const icon = type === 'error' ? 'fa-triangle-exclamation' : 'fa-circle-check';
    toast.innerHTML = `<i class="fa-solid ${icon}"></i> <span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      setTimeout(() => toast.remove(), 300);
    }, type === 'error' ? 7000 : 4000);
  }

  /**
   * Loads the Song Catalog tab with pagination and Load More support.
   * Browse mode: 100 per page. Search mode: all results (server returns up to 500 matches).
   */
  async loadCatalog(appendOffset = 0) {
    const searchInp = document.getElementById('catalog-search');
    const genreSel = document.getElementById('catalog-genre-filter');
    const tbody = document.getElementById('catalog-table-body');
    if (!tbody) return;

    const query = searchInp ? searchInp.value : '';
    const genre = genreSel ? genreSel.value : 'All';

    // Reset to first page on new search/genre change
    if (appendOffset === 0) {
      this._catalogOffset = 0;
    }
    const offset = appendOffset;
    const PAGE_SIZE = 100;

    try {
      let url = `/api/songs?genre=${encodeURIComponent(genre)}&search=${encodeURIComponent(query)}`;
      if (!query) {
        // Browse mode — paginate with a real page size
        url += `&offset=${offset}&limit=${PAGE_SIZE}`;
      }
      // Note: search mode returns all matches up to 500 (no offset needed)
      const res = await fetch(url);
      if (!res.ok) throw new Error(`Catalog fetch failed (${res.status})`);
      const data = await res.json();
      const songs = data.songs || data;
      const total = data.total != null ? data.total : songs.length;

      const renderRows = songs.map(s => {
        const cover = s.image_url || DEFAULT_COVER_ART;
        return `
          <tr>
            <td><img class="catalog-thumb" src="${cover}" alt="Cover"></td>
            <td>
              <strong>${s.title}</strong>
              <div style="font-size:11.5px; color:var(--text-secondary);">${s.artist_name}</div>
            </td>
            <td><span class="track-genre-badge">${s.genre}</span></td>
            <td>${(s.energy||0).toFixed(2)}</td>
            <td>${(s.danceability||0).toFixed(2)}</td>
            <td>${(s.valence||0).toFixed(2)}</td>
            <td>${Math.round(s.tempo||0)} BPM</td>
            <td>
              <div style="display:flex; gap:6px;">
                <button class="btn-small" onclick="window.app.playCatalogSong('${s.song_id}')" title="Play Now">
                  <i class="fa-solid fa-play"></i> Play
                </button>
                <button class="btn-small" onclick="window.app.addTrackToQueue(${JSON.stringify(s).replace(/"/g, '&quot;')}, false)" title="Add to Queue">
                  <i class="fa-solid fa-plus"></i> Queue
                </button>
                <button class="btn-small" onclick="window.app.openRelatedHub('${s.song_id}')" title="Find Related Songs (Vector Search)">
                  <i class="fa-solid fa-wand-magic-sparkles"></i> Related
                </button>
              </div>
            </td>
          </tr>
        `;
      }).join('');

      if (offset === 0) {
        tbody.innerHTML = renderRows;
      } else {
        tbody.innerHTML += renderRows;
      }

      // Update or remove Load More button
      const nextOffset = offset + songs.length;
      const loadMoreId = 'catalog-load-more-btn';
      let existingBtn = document.getElementById(loadMoreId);
      if (existingBtn) existingBtn.remove();

      if (!query && nextOffset < total) {
        const remaining = total - nextOffset;
        const btn = document.createElement('button');
        btn.id = loadMoreId;
        btn.className = 'btn-small';
        btn.style.cssText = 'margin:12px auto; display:block; min-width:180px;';
        btn.innerHTML = `<i class="fa-solid fa-chevron-down"></i> Load More (${remaining} remaining)`;
        btn.addEventListener('click', () => this.loadCatalog(nextOffset));
        tbody.parentElement.parentElement.appendChild(btn);
      }

      // Update header count if element exists
      const countEl = document.getElementById('catalog-count-badge');
      if (countEl) countEl.innerText = `${total} tracks`;
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
      // Use stream/resolve directly — avoids the fragile limit=5 text-search that would miss
      // most songs when searching by song_id.
      const resolveRes = await fetch(`/api/stream/resolve?song_id=${encodeURIComponent(songId)}`);
      if (!resolveRes.ok) throw new Error(`stream/resolve failed: ${resolveRes.status}`);
      const resolveData = await resolveRes.json();

      // Fetch full metadata from catalog via exact ID match
      const metaRes = await fetch(`/api/songs?search=${encodeURIComponent(songId)}`);
      const metaData = await metaRes.json();
      const songs = metaData.songs || metaData;
      let track = songs.find(s => s.song_id === songId) || null;

      // Merge resolved audio/image URLs into track metadata
      if (track) {
        if (resolveData.audio_url) track.audio_url = resolveData.audio_url;
        if (resolveData.image_url) track.image_url = resolveData.image_url;
      } else if (resolveData && resolveData.song_id) {
        track = resolveData;
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

    // Reset scroll to top of view
    const mainContent = document.querySelector('.main-content');
    if (mainContent) mainContent.scrollTop = 0;

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

  // =========================================================================
  // Spotify Aesthetic Modules & Shelves Implementation
  // =========================================================================

  renderSpotifyHomeShelves(recommendations, qDist = []) {
    const madeForYouShelf = document.getElementById('shelf-made-for-you');
    const discoverShelf = document.getElementById('shelf-discover-weekly');
    const trendingShelf = document.getElementById('shelf-trending-hits');
    const greetingTitle = document.getElementById('home-greeting-title');
    const profileBadge = document.getElementById('home-profile-name');

    // Dynamic greeting based on time of day
    const hour = new Date().getHours();
    let greet = 'Good morning';
    if (hour >= 12 && hour < 17) greet = 'Good afternoon';
    else if (hour >= 17 || hour < 5) greet = 'Good evening';

    if (greetingTitle) greetingTitle.innerText = greet;
    if (profileBadge && this.userProfile) {
      const topG = this.userProfile.genre_affinities ? Object.keys(this.userProfile.genre_affinities)[0] : 'Lo-Fi';
      profileBadge.innerText = `${topG} Listener`;
    }

    if (!recommendations || recommendations.length === 0) return;

    // 1. Shelf 1: Made For You (RL Policy Slate)
    if (madeForYouShelf) {
      madeForYouShelf.innerHTML = recommendations.slice(0, 8).map((track, idx) => {
        const cover = track.image_url || DEFAULT_COVER_ART;
        const qVal = qDist[idx] ? `Q: +${Number(qDist[idx].q_score).toFixed(2)}` : 'RL Pick';
        return `
          <div class="spotify-card" onclick="window.app.playSongDirectly('${track.song_id}')">
            <div class="sp-card-thumb-box">
              <img class="sp-card-img" src="${cover}" alt="Cover">
              <button class="sp-card-floating-play" title="Play ${this.escapeHtml(track.title)}">
                <i class="fa-solid fa-play"></i>
              </button>
            </div>
            <div class="sp-card-title" title="${this.escapeHtml(track.title)}">${this.escapeHtml(track.title)}</div>
            <div class="sp-card-desc">${this.escapeHtml(track.artist_name)} • ${this.escapeHtml(track.genre)}</div>
            <span class="sp-card-q-badge"><i class="fa-solid fa-brain"></i> ${qVal}</span>
          </div>
        `;
      }).join('');
    }

    // 2. Shelf 2: Discover Weekly (AI Exploration)
    if (discoverShelf) {
      // Reverse or filter higher exploration tracks
      const discTracks = [...recommendations].reverse().slice(0, 8);
      discoverShelf.innerHTML = discTracks.map((track) => {
        const cover = track.image_url || DEFAULT_COVER_ART;
        return `
          <div class="spotify-card" onclick="window.app.playSongDirectly('${track.song_id}')">
            <div class="sp-card-thumb-box">
              <img class="sp-card-img" src="${cover}" alt="Cover">
              <button class="sp-card-floating-play" title="Play ${this.escapeHtml(track.title)}">
                <i class="fa-solid fa-play"></i>
              </button>
            </div>
            <div class="sp-card-title" title="${this.escapeHtml(track.title)}">${this.escapeHtml(track.title)}</div>
            <div class="sp-card-desc">${this.escapeHtml(track.artist_name)} • High Novelty</div>
            <span class="sp-card-q-badge" style="background:rgba(56,189,248,0.15); color:#38bdf8;">
              <i class="fa-solid fa-wand-magic-sparkles"></i> AI Discovery
            </span>
          </div>
        `;
      }).join('');
    }

    // 3. Shelf 3: Trending Hits (from OpenSpot if available, or catalog top popularity)
    if (trendingShelf) {
      const trendTracks = this.openSpotCurrentResults.length > 0
        ? this.openSpotCurrentResults.slice(0, 8)
        : recommendations.slice(2, 10);

      trendingShelf.innerHTML = trendTracks.map((track) => {
        const cover = track.image_url || DEFAULT_COVER_ART;
        return `
          <div class="spotify-card" onclick="window.app.previewOpenSpotTrack(encodeURIComponent(JSON.stringify(${JSON.stringify(track).replace(/"/g, '&quot;')})))">
            <div class="sp-card-thumb-box">
              <img class="sp-card-img" src="${cover}" alt="Cover">
              <button class="sp-card-floating-play" title="Stream 320kbps">
                <i class="fa-solid fa-play"></i>
              </button>
            </div>
            <div class="sp-card-title" title="${this.escapeHtml(track.title)}">${this.escapeHtml(track.title)}</div>
            <div class="sp-card-desc">${this.escapeHtml(track.artist_name || 'Popular Artist')} • OpenSpot 320k</div>
            <span class="sp-card-q-badge" style="background:rgba(29,185,84,0.15); color:var(--accent-green-bright);">
              <i class="fa-solid fa-bolt"></i> Trending
            </span>
          </div>
        `;
      }).join('');
    }
  }

  renderSpotifyLibrary() {
    const listContainer = document.getElementById('library-tracks-list');
    const likedCountEl = document.getElementById('lib-liked-count');
    const sidebarCountEl = document.getElementById('sidebar-liked-count');
    const userLabelEl = document.getElementById('lib-user-label');

    if (userLabelEl && this.userProfile) {
      userLabelEl.innerText = this.userProfile.genre_affinities ? Object.keys(this.userProfile.genre_affinities)[0] : 'Lo-Fi Coder';
    }

    const likedEvents = this.sessionTrajectoryHistory.filter(h => h.action === 'liked');
    const likedTracks = likedEvents.map(h => h.track);

    if (likedCountEl) likedCountEl.innerText = `${likedTracks.length} song${likedTracks.length === 1 ? '' : 's'}`;
    if (sidebarCountEl) sidebarCountEl.innerText = `${likedTracks.length} saved songs`;

    if (!listContainer) return;

    if (likedTracks.length === 0) {
      listContainer.innerHTML = `
        <div style="padding: 32px 16px; text-align: center; color: var(--text-secondary);">
          <i class="fa-regular fa-heart" style="font-size: 28px; color: var(--text-muted); margin-bottom: 10px;"></i>
          <p style="font-size: 13px; margin: 0 0 10px;">Songs you like will appear here and train your RL policy.</p>
          <button class="btn-small" onclick="window.app.switchTab('player-tab')">
            <i class="fa-solid fa-play"></i> Start Listening & Liking
          </button>
        </div>
      `;
      return;
    }

    listContainer.innerHTML = likedTracks.map((track, idx) => {
      const cover = track.image_url || DEFAULT_COVER_ART;
      const dur = track.duration_formatted || '3:30';
      return `
        <div class="lib-track-row" onclick="window.app.playSongDirectly('${track.song_id}')">
          <span style="color: var(--text-muted); font-size: 12px;">${idx + 1}</span>
          <div class="lib-col-title">
            <img class="lib-thumb" src="${cover}" alt="Art">
            <div>
              <div style="color: #fff; font-weight: 700;">${this.escapeHtml(track.title)}</div>
              <div style="font-size: 11px; color: var(--text-muted);">${this.escapeHtml(track.artist_name)}</div>
            </div>
          </div>
          <div>${this.escapeHtml(track.artist_name)}</div>
          <div><span class="track-genre-badge" style="font-size: 10px;">${this.escapeHtml(track.genre)}</span></div>
          <div style="font-family: var(--font-mono); font-size: 11px;">${dur}</div>
          <div>
            <button class="btn-icon-small" title="Liked (+1.0 RL Reward)" style="color: var(--accent-green-bright);" onclick="event.stopPropagation();">
              <i class="fa-solid fa-heart"></i>
            </button>
            <button class="btn-icon-small" title="Play Now" onclick="event.stopPropagation(); window.app.playSongDirectly('${track.song_id}');">
              <i class="fa-solid fa-play"></i>
            </button>
          </div>
        </div>
      `;
    }).join('');
  }

  playDailyMix(mixNum) {
    if (mixNum === 1) {
      this.currentModelType = 'wolpertinger';
      this.switchTab('player-tab');
      this.fetchNextRecommendation();
      this.showToast('☕ Starting Daily Mix 1: Chill Study Beats');
    } else {
      this.switchTab('openspot-tab');
      this.loadOpenSpotCharts('Pop');
      this.showToast('⚡ Starting Release Radar: Top Trending OpenSpot Hits');
    }
  }

  updateSpotifyPlayerBar() {
    if (!this.currentTrack) return;
    const t = this.currentTrack;
    const cover = t.image_url || DEFAULT_COVER_ART;

    const coverEl = document.getElementById('sp-player-cover');
    const titleEl = document.getElementById('sp-player-title');
    const artistEl = document.getElementById('sp-player-artist');
    const playBtn = document.getElementById('sp-btn-play');
    const likeBtn = document.getElementById('sp-btn-like');
    const durEl = document.getElementById('sp-time-dur');
    const curEl = document.getElementById('sp-time-cur');
    const scrubber = document.getElementById('sp-audio-scrubber');

    if (coverEl) coverEl.src = cover;
    if (titleEl) titleEl.innerText = t.title || 'No Track';
    if (artistEl) artistEl.innerText = `${t.artist_name || 'Artist'} • ${t.genre || 'Music'}`;

    const isPlaying = window.audioSynth.isPlaying;
    if (playBtn) {
      playBtn.innerHTML = isPlaying ? '<i class="fa-solid fa-pause"></i>' : '<i class="fa-solid fa-play"></i>';
    }

    const isLiked = this.historyLikes.length > 0 && this.historyLikes[this.historyLikes.length - 1] === 1;
    if (likeBtn) {
      likeBtn.innerHTML = isLiked ? '<i class="fa-solid fa-heart" style="color:var(--accent-green-bright);"></i>' : '<i class="fa-regular fa-heart"></i>';
    }

    const cur = window.audioSynth.getCurrentTime();
    const dur = window.audioSynth.getDuration();
    if (durEl && dur > 0) {
      const min = Math.floor(dur / 60);
      const sec = Math.floor(dur % 60).toString().padStart(2, '0');
      durEl.innerText = `${min}:${sec}`;
    }
    if (curEl) {
      const min = Math.floor(cur / 60);
      const sec = Math.floor(cur % 60).toString().padStart(2, '0');
      curEl.innerText = `${min}:${sec}`;
    }
    if (scrubber && dur > 0) {
      scrubber.value = (cur / dur) * 100;
    }
  }

  // =========================================================================
  // Custom Playlist Management & Modals
  // =========================================================================

  openCreatePlaylistModal() {
    const modal = document.getElementById('create-playlist-modal');
    const nameInput = document.getElementById('input-playlist-name');
    const descInput = document.getElementById('input-playlist-desc');
    if (nameInput) nameInput.value = '';
    if (descInput) descInput.value = '';
    if (modal) modal.style.display = 'flex';
    if (nameInput) setTimeout(() => nameInput.focus(), 100);
  }

  closeCreatePlaylistModal() {
    const modal = document.getElementById('create-playlist-modal');
    if (modal) modal.style.display = 'none';
  }

  submitCreatePlaylist() {
    const nameInput = document.getElementById('input-playlist-name');
    const descInput = document.getElementById('input-playlist-desc');
    const name = nameInput ? nameInput.value.trim() : '';
    const desc = descInput ? descInput.value.trim() : '';

    if (!name) {
      this.showToast('⚠️ Please enter a playlist name.');
      return;
    }

    const newPl = {
      id: `pl_${Date.now()}`,
      name: name,
      desc: desc || 'Custom created playlist',
      tracks: []
    };

    this.customPlaylists.push(newPl);
    localStorage.setItem('soundspace_custom_playlists', JSON.stringify(this.customPlaylists));
    this.closeCreatePlaylistModal();
    this.renderCustomPlaylists();
    this.showToast(`✨ Created playlist "${name}"!`);
  }

  openAddToPlaylistModal(track) {
    this.pendingTrackForPlaylist = track || this.currentTrack;
    if (!this.pendingTrackForPlaylist) return;

    const modal = document.getElementById('add-to-playlist-modal');
    const previewThumb = document.getElementById('modal-preview-thumb');
    const previewTitle = document.getElementById('modal-preview-title');
    const previewArtist = document.getElementById('modal-preview-artist');
    const plList = document.getElementById('modal-playlists-list');

    const t = this.pendingTrackForPlaylist;
    if (previewThumb) previewThumb.src = t.image_url || DEFAULT_COVER_ART;
    if (previewTitle) previewTitle.innerText = t.title;
    if (previewArtist) previewArtist.innerText = t.artist_name || 'Unknown Artist';

    if (plList) {
      if (this.customPlaylists.length === 0) {
        plList.innerHTML = '<p style="color:var(--text-muted); font-size:12px; padding:10px 0;">No custom playlists yet. Create one below!</p>';
      } else {
        plList.innerHTML = this.customPlaylists.map(pl => {
          const isIncluded = pl.tracks.some(tr => tr.song_id === t.song_id);
          return `
            <div class="modal-pl-item" onclick="window.app.toggleTrackInPlaylist('${pl.id}')">
              <div class="modal-pl-meta">
                <div class="modal-pl-icon"><i class="fa-solid fa-music"></i></div>
                <div>
                  <div style="font-weight:700; color:#fff; font-size:13px;">${this.escapeHtml(pl.name)}</div>
                  <div style="font-size:11px; color:var(--text-muted);">${pl.tracks.length} songs</div>
                </div>
              </div>
              <div>
                ${isIncluded
                  ? '<span style="color:var(--accent-green-bright); font-size:12px; font-weight:700;"><i class="fa-solid fa-check"></i> Added</span>'
                  : '<button class="btn-icon-small" style="font-size:11px;"><i class="fa-solid fa-plus"></i></button>'
                }
              </div>
            </div>
          `;
        }).join('');
      }
    }

    if (modal) modal.style.display = 'flex';
  }

  closeAddToPlaylistModal() {
    const modal = document.getElementById('add-to-playlist-modal');
    if (modal) modal.style.display = 'none';
  }

  toggleTrackInPlaylist(playlistId) {
    const pl = this.customPlaylists.find(p => p.id === playlistId);
    const track = this.pendingTrackForPlaylist;
    if (!pl || !track) return;

    const existingIdx = pl.tracks.findIndex(t => t.song_id === track.song_id);
    if (existingIdx >= 0) {
      pl.tracks.splice(existingIdx, 1);
      this.showToast(`Removed "${track.title}" from "${pl.name}"`);
    } else {
      pl.tracks.push(track);
      this.showToast(`Added "${track.title}" to "${pl.name}"`);
    }

    localStorage.setItem('soundspace_custom_playlists', JSON.stringify(this.customPlaylists));
    this.openAddToPlaylistModal(track);
    this.renderCustomPlaylists();
  }

  renderCustomPlaylists() {
    const sidebarPlContainer = document.querySelector('.quick-playlist-items');
    if (sidebarPlContainer) {
      const customHtml = this.customPlaylists.map(pl => `
        <div class="quick-pl-item" data-playlist="${pl.id}" onclick="window.app.playCustomPlaylist('${pl.id}')">
          <div class="pl-icon-gradient" style="background: linear-gradient(135deg, #11998e, #38ef7d);">
            <i class="fa-solid fa-music"></i>
          </div>
          <div class="pl-meta">
            <span class="pl-name">${this.escapeHtml(pl.name)}</span>
            <span class="pl-sub">${pl.tracks.length} songs • Custom</span>
          </div>
        </div>
      `).join('');

      const defaultItems = `
        <div class="quick-pl-item" data-playlist="liked" onclick="window.app.switchTab('library-tab')">
          <div class="pl-icon-gradient"><i class="fa-solid fa-heart"></i></div>
          <div class="pl-meta">
            <span class="pl-name">Liked Songs</span>
            <span class="pl-sub" id="sidebar-liked-count">${this.sessionTrajectoryHistory.filter(h => h.action === 'liked').length} saved songs</span>
          </div>
        </div>
        <div class="quick-pl-item" data-playlist="discover" onclick="window.app.switchTab('home-tab')">
          <div class="pl-icon-gradient discover"><i class="fa-solid fa-wand-magic-sparkles"></i></div>
          <div class="pl-meta">
            <span class="pl-name">Discover Weekly</span>
            <span class="pl-sub">AI RL Curated</span>
          </div>
        </div>
        <div class="quick-pl-item" data-playlist="daily1" onclick="window.app.playDailyMix(1)">
          <div class="pl-icon-gradient daily"><i class="fa-solid fa-bolt"></i></div>
          <div class="pl-meta">
            <span class="pl-name">Daily Mix 1</span>
            <span class="pl-sub">Lo-Fi & Chillhop</span>
          </div>
        </div>
        <div class="quick-pl-item" data-playlist="daily2" onclick="window.app.playDailyMix(2)">
          <div class="pl-icon-gradient radar"><i class="fa-solid fa-tower-broadcast"></i></div>
          <div class="pl-meta">
            <span class="pl-name">Release Radar</span>
            <span class="pl-sub">New OpenSpot Hits</span>
          </div>
        </div>
      `;

      sidebarPlContainer.innerHTML = defaultItems + customHtml;
    }
  }

  playCustomPlaylist(playlistId) {
    const pl = this.customPlaylists.find(p => p.id === playlistId);
    if (!pl) return;

    if (pl.tracks.length === 0) {
      this.showToast(`"${pl.name}" is empty. Click + on any track to add songs!`);
      return;
    }

    const [first, ...rest] = pl.tracks;
    this.currentQueueTracks = [...rest];
    this.renderQueue();
    this.playSongDirectly(first.song_id);
    this.showToast(`▶ Playing "${pl.name}" (${pl.tracks.length} songs)!`);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.app = new App();
});
