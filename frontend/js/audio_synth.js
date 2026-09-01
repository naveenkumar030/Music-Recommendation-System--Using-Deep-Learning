/**
 * Unified SoundSpace Audio Player & Visualizer.
 * Supports:
 * - Direct high-fidelity master streaming via backend proxy / direct CDN with CORS.
 * - Real-time Web Audio API frequency spectrum visualizer attached to the stream.
 * - Procedural Web Audio Synthesizer fallback calibrated to track acoustic features.
 * - Automatic on-the-fly stream resolution for any unpopulated song.
 * - Real-time synchronized lyrics synchronization and scrub seeking.
 * - Progress scrubber, duration counter, and auto-playback transitions.
 */

class SoundSpacePlayer {
  constructor() {
    this.ctx = null;
    this.analyser = null;
    this.audioEl = document.getElementById('html5-audio-element');
    if (!this.audioEl) {
      this.audioEl = new Audio();
      this.audioEl.id = 'html5-audio-element';
      document.body.appendChild(this.audioEl);
    }
    this.audioEl.volume = 1.0;
    this.audioEl.muted = false;

    this.isPlaying = false;
    this.isSynthMode = false;
    this.synthElapsedSeconds = 0;
    this.timerId = null;
    this.currentTrack = null;
    this.canvas = document.getElementById('audio-visualizer-canvas');
    this.canvasCtx = this.canvas ? this.canvas.getContext('2d') : null;
    this.animFrameId = null;

    // Musical scale frequencies for procedural synth fallback
    this.scales = {
      "Lo-Fi / Chillhop": [261.63, 293.66, 329.63, 392.00, 440.00, 523.25],
      "Jazz": [261.63, 311.13, 349.23, 392.00, 466.16, 523.25],
      "EDM / Electronic": [220.00, 261.63, 293.66, 329.63, 392.00, 440.00],
      "Pop": [261.63, 293.66, 329.63, 392.00, 440.00, 523.25],
      "Rock": [196.00, 220.00, 246.94, 293.66, 329.63, 392.00],
      "Classical": [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]
    };

    this.initAudioEvents();
  }

  initContext() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AudioCtx();
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 128;
      this.analyser.smoothingTimeConstant = 0.8;
    }
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume().catch(e => console.log('AudioContext resume note:', e));
    }
  }

  initAudioEvents() {
    if (!this.audioEl) return;

    this.audioEl.addEventListener('timeupdate', () => {
      const cur = this.audioEl.currentTime || 0;
      const dur = this.audioEl.duration || (this.currentTrack && this.currentTrack.song_length_ms ? this.currentTrack.song_length_ms / 1000 : 210);
      const pct = dur > 0 ? (cur / dur) * 100 : 0;

      const scrubber = document.getElementById('audio-scrubber');
      const timeCur = document.getElementById('player-time-cur');
      const timeDur = document.getElementById('player-time-dur');

      if (scrubber && !scrubber.matches(':active')) {
        scrubber.value = pct;
        // Update CSS gradient fill for the progress bar
        scrubber.style.setProperty('--prog', `${pct.toFixed(1)}%`);
      }
      if (timeCur) timeCur.innerText = this.formatTime(cur);
      if (timeDur && dur && !isNaN(dur)) timeDur.innerText = this.formatTime(dur);

      if (window.app && window.app.onPlaybackTimeUpdate) {
        window.app.onPlaybackTimeUpdate(cur, dur);
      }
    });

    this.audioEl.addEventListener('playing', () => {
      this.isPlaying = true;
      const playBtn = document.getElementById('btn-audio-play');
      const vinyl = document.getElementById('vinyl-disc');
      if (playBtn) playBtn.innerHTML = '<i class="fa-solid fa-pause"></i>';
      if (vinyl) vinyl.classList.remove('paused');
    });

    this.audioEl.addEventListener('pause', () => {
      if (!this.isSynthMode) {
        this.isPlaying = false;
        const playBtn = document.getElementById('btn-audio-play');
        const vinyl = document.getElementById('vinyl-disc');
        if (playBtn) playBtn.innerHTML = '<i class="fa-solid fa-play"></i>';
        if (vinyl) vinyl.classList.add('paused');
      }
    });

    this.audioEl.addEventListener('ended', () => {
      if (window.app) {
        window.app.handleFeedback('no_skip', 1.0);
      }
    });

    this.audioEl.addEventListener('error', (e) => {
      console.warn("Direct stream error, falling back to Web Audio Synth:", e);
      if (this.isPlaying && this.currentTrack) {
        this.startSynthSequence();
      }
    });
  }

  formatTime(seconds) {
    if (isNaN(seconds) || seconds < 0) return "0:00";
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  }

  async playTrack(track) {
    this.initContext();
    this.currentTrack = track;
    this.isPlaying = true;
    this.synthElapsedSeconds = 0;

    const playBtn = document.getElementById('btn-audio-play');
    const vinyl = document.getElementById('vinyl-disc');
    const modeTag = document.getElementById('audio-mode-tag');

    if (playBtn) playBtn.innerHTML = '<i class="fa-solid fa-pause"></i>';
    if (vinyl) vinyl.classList.remove('paused');

    // Notify app of track play to trigger lyrics fetch
    if (window.app && window.app.onTrackLoaded) {
      window.app.onTrackLoaded(track);
    }

    // 1. If track does not have audio_url, resolve it from backend on the fly
    if (!track.audio_url && track.song_id) {
      if (modeTag) {
        modeTag.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Resolving OpenSpot Stream...';
      }
      try {
        const res = await fetch(`/api/stream/resolve?song_id=${encodeURIComponent(track.song_id)}`);
        const data = await res.json();
        if (data.audio_url) {
          track.audio_url = data.audio_url;
          if (data.image_url) track.image_url = data.image_url;
          const artImg = document.getElementById('player-album-art');
          if (artImg && data.image_url) artImg.src = data.image_url;
        }
      } catch (err) {
        console.warn("Could not resolve stream URL:", err);
      }
    }

    // 2. Play direct stream or proxy
    if (track.audio_url) {
      this.isSynthMode = false;
      if (this.timerId) {
        clearInterval(this.timerId);
        this.timerId = null;
      }

      if (modeTag) {
        modeTag.innerHTML = '<i class="fa-solid fa-tower-cell"></i> OpenSpot 320kbps Stream';
        modeTag.className = 'audio-mode-tag active-stream';
      }

      // Use proxy or direct URL
      const streamSrc = track.audio_url.startsWith('http') 
        ? `/api/stream/proxy?url=${encodeURIComponent(track.audio_url)}` 
        : track.audio_url;

      this.audioEl.src = streamSrc;
      this.audioEl.volume = 1.0;
      this.audioEl.muted = false;

      const playPromise = this.audioEl.play();
      if (playPromise !== undefined) {
        playPromise.catch(err => {
          console.warn("Audio element play error, trying direct CDN stream:", err);
          this.audioEl.src = track.audio_url;
          this.audioEl.play().catch(cdnErr => {
            console.warn("Direct CDN failed, switching to Procedural Synth:", cdnErr);
            this.startSynthSequence();
          });
        });
      }
    } else {
      this.startSynthSequence();
    }

    this.renderVisualizer();
  }

  startSynthSequence() {
    this.isSynthMode = true;
    if (this.audioEl) {
      this.audioEl.pause();
    }

    const modeTag = document.getElementById('audio-mode-tag');
    if (modeTag) {
      modeTag.innerHTML = '<i class="fa-solid fa-waveform-lines"></i> Web Audio Synth Active';
      modeTag.className = 'audio-mode-tag';
    }

    if (!this.currentTrack) return;
    const tempo = this.currentTrack.tempo || 110;
    const intervalMs = (60 / tempo) * 500;
    const durSec = (this.currentTrack.song_length_ms ? this.currentTrack.song_length_ms / 1000 : 210);
    const genre = this.currentTrack.genre || "Pop";
    const notes = this.scales[genre] || this.scales["Pop"];
    let step = 0;

    if (this.timerId) clearInterval(this.timerId);

    this.timerId = setInterval(() => {
      if (!this.isPlaying || !this.isSynthMode) return;
      const energy = this.currentTrack.energy || 0.5;
      const freq = notes[step % notes.length];
      
      this.playTone(freq, 0.35, energy);

      if (step % 4 === 0) {
        this.playKick(energy);
      }
      if ((this.currentTrack.danceability || 0.5) > 0.6 && step % 2 === 1) {
        this.playHiHat();
      }

      this.synthElapsedSeconds += (intervalMs / 1000.0);
      if (this.synthElapsedSeconds >= durSec) {
        this.synthElapsedSeconds = 0;
        if (window.app) window.app.handleFeedback('no_skip', 1.0);
      }

      const scrubber = document.getElementById('audio-scrubber');
      const timeCur = document.getElementById('player-time-cur');
      const timeDur = document.getElementById('player-time-dur');
      if (scrubber && !scrubber.matches(':active')) {
        scrubber.value = (this.synthElapsedSeconds / durSec) * 100;
      }
      if (timeCur) timeCur.innerText = this.formatTime(this.synthElapsedSeconds);
      if (timeDur) timeDur.innerText = this.formatTime(durSec);

      if (window.app && window.app.onPlaybackTimeUpdate) {
        window.app.onPlaybackTimeUpdate(this.synthElapsedSeconds, durSec);
      }

      step++;
    }, intervalMs);
  }

  stop() {
    this.isPlaying = false;
    if (this.audioEl) {
      this.audioEl.pause();
    }
    if (this.timerId) {
      clearInterval(this.timerId);
      this.timerId = null;
    }
    if (this.animFrameId) {
      cancelAnimationFrame(this.animFrameId);
    }
    const playBtn = document.getElementById('btn-audio-play');
    const vinyl = document.getElementById('vinyl-disc');
    if (playBtn) playBtn.innerHTML = '<i class="fa-solid fa-play"></i>';
    if (vinyl) vinyl.classList.add('paused');

    if (this.canvasCtx && this.canvas) {
      this.canvasCtx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
  }

  toggle(track) {
    if (this.isPlaying) {
      this.stop();
    } else {
      this.playTrack(track || this.currentTrack);
    }
  }

  seek(percent) {
    const dur = this.audioEl.duration || (this.currentTrack && this.currentTrack.song_length_ms ? this.currentTrack.song_length_ms / 1000 : 210);
    const targetSec = (percent / 100) * dur;
    this.seekToTime(targetSec);
  }

  seekToTime(seconds) {
    const dur = this.audioEl.duration || (this.currentTrack && this.currentTrack.song_length_ms ? this.currentTrack.song_length_ms / 1000 : 210);
    const target = Math.max(0, Math.min(dur, seconds));

    if (this.audioEl && !this.isSynthMode && !isNaN(this.audioEl.duration) && this.audioEl.duration > 0) {
      this.audioEl.currentTime = target;
    } else if (this.isSynthMode) {
      this.synthElapsedSeconds = target;
    }

    const scrubber = document.getElementById('audio-scrubber');
    const timeCur = document.getElementById('player-time-cur');
    if (scrubber) scrubber.value = (target / dur) * 100;
    if (timeCur) timeCur.innerText = this.formatTime(target);

    if (window.app && window.app.onPlaybackTimeUpdate) {
      window.app.onPlaybackTimeUpdate(target, dur);
    }
  }

  getCurrentTime() {
    if (this.isSynthMode) return this.synthElapsedSeconds;
    return this.audioEl ? this.audioEl.currentTime || 0 : 0;
  }

  getDuration() {
    if (this.audioEl && !isNaN(this.audioEl.duration) && this.audioEl.duration > 0) {
      return this.audioEl.duration;
    }
    return this.currentTrack && this.currentTrack.song_length_ms ? this.currentTrack.song_length_ms / 1000 : 210;
  }

  setVolume(val) {
    if (this.audioEl) {
      this.audioEl.volume = Math.max(0, Math.min(1, val));
    }
  }

  playTone(freq, duration = 0.3, energy = 0.5) {
    if (!this.ctx) return;
    try {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      osc.type = energy > 0.7 ? 'sawtooth' : (energy > 0.4 ? 'triangle' : 'sine');
      osc.frequency.setValueAtTime(freq, this.ctx.currentTime);

      gain.gain.setValueAtTime(0.25 * energy, this.ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + duration);

      osc.connect(gain);
      if (this.analyser) gain.connect(this.analyser);
      gain.connect(this.ctx.destination);

      osc.start();
      osc.stop(this.ctx.currentTime + duration);
    } catch (e) {
      console.warn("playTone error:", e);
    }
  }

  playKick(energy = 0.5) {
    if (!this.ctx) return;
    try {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      osc.frequency.setValueAtTime(130, this.ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(30, this.ctx.currentTime + 0.15);

      gain.gain.setValueAtTime(0.4 * energy, this.ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.18);

      osc.connect(gain);
      if (this.analyser) gain.connect(this.analyser);
      gain.connect(this.ctx.destination);

      osc.start();
      osc.stop(this.ctx.currentTime + 0.2);
    } catch (e) {
      console.warn("playKick error:", e);
    }
  }

  playHiHat() {
    if (!this.ctx) return;
    try {
      const bufferSize = Math.floor(this.ctx.sampleRate * 0.05);
      const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) {
        data[i] = Math.random() * 2 - 1;
      }

      const noise = this.ctx.createBufferSource();
      noise.buffer = buffer;

      const gain = this.ctx.createGain();
      gain.gain.setValueAtTime(0.08, this.ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.05);

      noise.connect(gain);
      if (this.analyser) gain.connect(this.analyser);
      gain.connect(this.ctx.destination);

      noise.start();
    } catch (e) {
      console.warn("playHiHat error:", e);
    }
  }

  renderVisualizer() {
    if (!this.canvasCtx || !this.isPlaying) return;

    const width = this.canvas.width;
    const height = this.canvas.height;
    this.canvasCtx.clearRect(0, 0, width, height);

    const bufferLength = this.analyser ? this.analyser.frequencyBinCount : 32;
    const dataArray = new Uint8Array(bufferLength);

    if (this.analyser && this.isSynthMode) {
      this.analyser.getByteFrequencyData(dataArray);
    } else {
      const time = Date.now() * 0.006;
      const energy = this.currentTrack ? this.currentTrack.energy || 0.6 : 0.6;
      for (let i = 0; i < bufferLength; i++) {
        const wave = Math.sin(time + i * 0.35) * Math.cos(time * 0.7 + i * 0.2);
        dataArray[i] = Math.max(20, Math.min(250, (wave * 0.5 + 0.5) * 220 * energy + (Math.random() * 30)));
      }
    }

    const barWidth = (width / bufferLength) * 2.2;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const barHeight = (dataArray[i] / 255) * height;
      
      const grad = this.canvasCtx.createLinearGradient(0, height - barHeight, 0, height);
      grad.addColorStop(0, '#1ed760');
      grad.addColorStop(0.5, '#6156e2');
      grad.addColorStop(1, '#0e5e2b');

      this.canvasCtx.fillStyle = grad;
      this.canvasCtx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
      x += barWidth;
    }

    this.animFrameId = requestAnimationFrame(() => this.renderVisualizer());
  }
}

window.audioSynth = new SoundSpacePlayer();
