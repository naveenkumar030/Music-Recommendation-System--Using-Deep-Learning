/**
 * RL Brain Visualizer: Q-value charts, Feature Radar, and 2D Latent Space Trajectory.
 */

class RLVisualizer {
  constructor() {
    this.qChart = null;
    this.radarChart = null;
    this.trajectoryCanvas = document.getElementById('trajectory-canvas');
    this.trajCtx = this.trajectoryCanvas ? this.trajectoryCanvas.getContext('2d') : null;
    this.trajectoryPoints = [];

    this.initCharts();
  }

  initCharts() {
    // 1. Q-Values Bar Chart
    const qCtx = document.getElementById('q-values-chart');
    if (qCtx) {
      this.qChart = new Chart(qCtx, {
        type: 'bar',
        data: {
          labels: ['Cand 1', 'Cand 2', 'Cand 3', 'Cand 4', 'Cand 5', 'Cand 6'],
          datasets: [{
            label: 'Predicted Q(s, a)',
            data: [2.8, 2.4, 2.1, 1.8, 1.2, 0.9],
            backgroundColor: 'rgba(168, 85, 247, 0.7)',
            borderColor: '#a855f7',
            borderWidth: 1,
            borderRadius: 6
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false }
          },
          scales: {
            x: {
              ticks: { color: '#8b949e', font: { size: 11 } },
              grid: { color: 'rgba(255, 255, 255, 0.05)' }
            },
            y: {
              ticks: { color: '#8b949e' },
              grid: { color: 'rgba(255, 255, 255, 0.05)' }
            }
          }
        }
      });
    }

    // 2. Feature Radar Chart
    const rCtx = document.getElementById('radar-chart');
    if (rCtx) {
      this.radarChart = new Chart(rCtx, {
        type: 'radar',
        data: {
          labels: ['Energy', 'Danceability', 'Valence', 'Acousticness', 'Popularity'],
          datasets: [
            {
              label: 'User Persona Pref',
              data: [0.4, 0.5, 0.5, 0.7, 0.6],
              backgroundColor: 'rgba(56, 189, 248, 0.25)',
              borderColor: '#38bdf8',
              borderWidth: 2,
              pointBackgroundColor: '#38bdf8'
            },
            {
              label: 'Recommended Song',
              data: [0.35, 0.55, 0.45, 0.75, 0.55],
              backgroundColor: 'rgba(29, 185, 84, 0.25)',
              borderColor: '#1ed760',
              borderWidth: 2,
              pointBackgroundColor: '#1ed760'
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              labels: { color: '#f0f6fc', font: { size: 11 } }
            }
          },
          scales: {
            r: {
              ticks: { display: false },
              grid: { color: 'rgba(255, 255, 255, 0.08)' },
              angleLines: { color: 'rgba(255, 255, 255, 0.08)' },
              pointLabels: { color: '#8b949e', font: { size: 11, weight: '600' } },
              min: 0,
              max: 1
            }
          }
        }
      });
    }

    this.initTrajectoryCanvas();
  }

  updateQChart(qDist) {
    if (!this.qChart || !qDist || qDist.length === 0) return;
    const labels = qDist.map(d => d.title.length > 14 ? d.title.substring(0, 12) + '..' : d.title);
    const values = qDist.map(d => d.q_score);

    this.qChart.data.labels = labels;
    this.qChart.data.datasets[0].data = values;
    this.qChart.update();
  }

  updateRadar(userProfile, songMeta) {
    if (!this.radarChart) return;
    
    const userVals = [
      userProfile.pref_energy || 0.5,
      userProfile.pref_danceability || 0.5,
      userProfile.pref_valence || 0.5,
      0.6,
      0.5
    ];

    const songVals = [
      songMeta.energy || 0.5,
      songMeta.danceability || 0.5,
      songMeta.valence || 0.5,
      songMeta.acousticness || 0.5,
      (songMeta.popularity || 50) / 100.0
    ];

    this.radarChart.data.datasets[0].data = userVals;
    this.radarChart.data.datasets[1].data = songVals;
    this.radarChart.update();
  }

  initTrajectoryCanvas() {
    if (!this.trajCtx || !this.trajectoryCanvas) return;
    this.resetTrajectory();
  }

  resetTrajectory() {
    this.trajectoryPoints = [];
    this.addTrajectoryPoint(0.5, 0.5, 'Start');
  }

  addTrajectoryPoint(xNorm, yNorm, label = '') {
    this.trajectoryPoints.push({ x: xNorm, y: yNorm, label });
    this.drawTrajectory();
  }

  drawTrajectory() {
    if (!this.trajCtx || !this.trajectoryCanvas) return;
    const ctx = this.trajCtx;
    const w = this.trajectoryCanvas.width;
    const h = this.trajectoryCanvas.height;

    ctx.clearRect(0, 0, w, h);

    // Draw background grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let x = 40; x < w; x += 50) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 30; y < h; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    if (this.trajectoryPoints.length < 1) return;

    // Draw connection lines
    ctx.strokeStyle = '#1ed760';
    ctx.lineWidth = 3;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();

    for (let i = 0; i < this.trajectoryPoints.length; i++) {
      const pt = this.trajectoryPoints[i];
      const px = 60 + pt.x * (w - 120);
      const py = 40 + pt.y * (h - 80);

      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw nodes
    for (let i = 0; i < this.trajectoryPoints.length; i++) {
      const pt = this.trajectoryPoints[i];
      const px = 60 + pt.x * (w - 120);
      const py = 40 + pt.y * (h - 80);

      ctx.beginPath();
      ctx.arc(px, py, i === this.trajectoryPoints.length - 1 ? 8 : 5, 0, Math.PI * 2);
      ctx.fillStyle = i === this.trajectoryPoints.length - 1 ? '#38bdf8' : (i === 0 ? '#f59e0b' : '#1db954');
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();

      if (pt.label) {
        ctx.fillStyle = '#f0f6fc';
        ctx.font = '10px JetBrains Mono';
        ctx.fillText(pt.label, px + 10, py + 4);
      }
    }
  }
}

window.rlVisualizer = new RLVisualizer();
