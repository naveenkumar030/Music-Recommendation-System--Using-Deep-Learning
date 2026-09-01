/**
 * A/B Simulation Lab Runner and Multi-Model Comparison Charts.
 */

class BenchmarkView {
  constructor() {
    this.simChart = null;
    this.initChart();
  }

  initChart() {
    const ctx = document.getElementById('sim-comparison-chart');
    if (!ctx) return;

    this.simChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: Array.from({ length: 15 }, (_, i) => `Sess ${i + 1}`),
        datasets: [
          {
            label: 'Wolpertinger RL',
            data: [4.2, 5.8, 7.1, 8.4, 9.2, 11.0, 10.5, 12.4, 13.8, 14.1, 15.0, 14.8, 16.2, 15.9, 17.1],
            borderColor: '#1ed760',
            backgroundColor: 'rgba(29, 185, 84, 0.1)',
            fill: true,
            tension: 0.35,
            borderWidth: 3
          },
          {
            label: 'Two-Tower Baseline',
            data: [6.5, 6.2, 5.9, 6.4, 5.8, 6.1, 6.0, 5.9, 6.3, 5.7, 6.2, 5.9, 6.1, 5.8, 6.0],
            borderColor: '#38bdf8',
            borderDash: [5, 5],
            fill: false,
            tension: 0.2,
            borderWidth: 2
          },
          {
            label: 'Random Exploration',
            data: [0.5, -0.8, 1.2, -1.0, 0.2, -0.5, 0.8, -0.2, 0.4, -0.9, 0.1, -0.4, 0.6, -0.7, 0.2],
            borderColor: '#a855f7',
            borderDash: [2, 2],
            fill: false,
            tension: 0.2,
            borderWidth: 1.5
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: '#f0f6fc', font: { size: 12 } }
          }
        },
        scales: {
          x: {
            ticks: { color: '#8b949e' },
            grid: { color: 'rgba(255, 255, 255, 0.05)' }
          },
          y: {
            ticks: { color: '#8b949e' },
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            title: { display: true, text: 'Cumulative Episode Return', color: '#8b949e' }
          }
        }
      }
    });
  }

  async runSimulation(numSessions = 25) {
    const btn = document.getElementById('btn-run-simulation');
    if (btn) btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running Simulation...';

    try {
      const res = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          num_sessions: numSessions,
          models_to_compare: ['wolpertinger', 'baseline', 'random']
        })
      });

      const data = await res.json();
      const comp = data.comparison;

      // Update metric cards
      document.getElementById('sim-rl-return').innerText = `+${comp.wolpertinger.mean_episode_return.toFixed(2)}`;
      document.getElementById('sim-rl-skip').innerText = `${comp.wolpertinger.skip_rate_pct.toFixed(1)}%`;
      document.getElementById('sim-rl-like').innerText = `${comp.wolpertinger.like_rate_pct.toFixed(1)}%`;

      document.getElementById('sim-base-return').innerText = `+${comp.baseline.mean_episode_return.toFixed(2)}`;
      document.getElementById('sim-base-skip').innerText = `${comp.baseline.skip_rate_pct.toFixed(1)}%`;
      document.getElementById('sim-base-like').innerText = `${comp.baseline.like_rate_pct.toFixed(1)}%`;

      document.getElementById('sim-rand-return').innerText = `${comp.random.mean_episode_return.toFixed(2)}`;
      document.getElementById('sim-rand-skip').innerText = `${comp.random.skip_rate_pct.toFixed(1)}%`;
      document.getElementById('sim-rand-like').innerText = `${comp.random.like_rate_pct.toFixed(1)}%`;

      // Update chart
      if (this.simChart) {
        this.simChart.data.datasets[0].data = comp.wolpertinger.trajectory_points;
        this.simChart.data.datasets[1].data = comp.baseline.trajectory_points;
        this.simChart.data.datasets[2].data = comp.random.trajectory_points;
        this.simChart.update();
      }
    } catch (e) {
      console.error('Simulation error:', e);
    } finally {
      if (btn) btn.innerHTML = '<i class="fa-solid fa-play"></i> Run Simulation Benchmark';
    }
  }
}

window.benchmarkView = new BenchmarkView();
