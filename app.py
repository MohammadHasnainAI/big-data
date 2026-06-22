<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Intelligence AI — Sentiment Analyzer</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #F5F5F7;
    color: #1D1D1F;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* ── NAV ── */
  nav {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid #E5E5EA;
    padding: 0 40px;
    display: flex;
    align-items: center;
    gap: 0;
    position: sticky;
    top: 0;
    z-index: 100;
    height: 52px;
  }

  .nav-logo {
    font-size: 17px;
    font-weight: 700;
    color: #1D1D1F;
    margin-right: 40px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .nav-links { display: flex; gap: 0; }

  .nav-link {
    padding: 0 18px;
    height: 52px;
    display: flex;
    align-items: center;
    font-size: 14px;
    font-weight: 500;
    color: #86868B;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 0.2s ease;
    text-decoration: none;
  }

  .nav-link:hover { color: #1D1D1F; }
  .nav-link.active { color: #0071E3; border-bottom-color: #0071E3; }

  .nav-badge {
    margin-left: auto;
    font-size: 12px;
    color: #86868B;
    background: #F5F5F7;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid #E5E5EA;
  }

  /* ── PAGES ── */
  .page { display: none; }
  .page.active { display: block; }

  /* ── HOME PAGE ── */
  .home-hero {
    text-align: center;
    padding: 80px 24px 60px;
    max-width: 720px;
    margin: 0 auto;
  }

  .home-hero h1 {
    font-size: 56px;
    font-weight: 700;
    color: #1D1D1F;
    letter-spacing: -1px;
    margin-bottom: 16px;
    line-height: 1.1;
  }

  .home-hero p {
    font-size: 19px;
    color: #86868B;
    margin-bottom: 36px;
    line-height: 1.5;
  }

  .hero-img {
    width: 100%;
    max-height: 400px;
    object-fit: cover;
    border-radius: 20px;
    margin-bottom: 36px;
  }

  .launch-btn {
    background: #0071E3;
    color: white;
    border: none;
    border-radius: 980px;
    padding: 14px 36px;
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  .launch-btn:hover { background: #0077ED; transform: scale(1.02); }

  /* ── TOOL PAGE ── */
  .tool-page {
    max-width: 780px;
    margin: 0 auto;
    padding: 48px 24px;
    width: 100%;
  }

  .tool-page h1 {
    font-size: 36px;
    font-weight: 700;
    color: #1D1D1F;
    margin-bottom: 8px;
  }

  .tool-page .subtitle {
    font-size: 16px;
    color: #86868B;
    margin-bottom: 32px;
  }

  /* Input box */
  .input-card {
    background: white;
    border: 1px solid #E5E5EA;
    border-radius: 20px;
    padding: 18px 20px 14px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    transition: border 0.2s, box-shadow 0.2s;
    margin-bottom: 12px;
  }

  .input-card:focus-within {
    border-color: #0071E3;
    box-shadow: 0 0 0 3px rgba(0,113,227,0.1);
  }

  textarea {
    width: 100%;
    background: transparent;
    border: none;
    outline: none;
    color: #1D1D1F;
    font-size: 15px;
    line-height: 1.6;
    resize: none;
    min-height: 72px;
    max-height: 180px;
    font-family: inherit;
  }

  textarea::placeholder { color: #C7C7CC; }

  .input-actions {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #F0F0F5;
  }

  .left-actions { display: flex; align-items: center; gap: 10px; }

  .mic-btn {
    width: 38px; height: 38px;
    border-radius: 50%;
    border: 1.5px solid #E5E5EA;
    background: #F5F5F7;
    color: #1D1D1F;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    font-size: 17px;
    transition: all 0.2s ease;
  }

  .mic-btn:hover { background: #E5E5EA; border-color: #0071E3; }

  .mic-btn.recording {
    background: #FF3B30;
    border-color: #FF3B30;
    animation: pulse 1.2s infinite;
  }

  @keyframes pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(255,59,48,0.3); }
    50% { box-shadow: 0 0 0 8px rgba(255,59,48,0); }
  }

  .mic-label { font-size: 13px; color: #86868B; }
  .mic-label.active { color: #FF3B30; font-weight: 500; }

  .analyze-btn {
    background: #0071E3;
    color: white;
    border: none;
    border-radius: 980px;
    padding: 9px 22px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  .analyze-btn:hover { background: #0077ED; }
  .analyze-btn:disabled { background: #C7C7CC; cursor: not-allowed; }

  /* Transcription */
  .transcription-box {
    display: none;
    background: white;
    border: 1px solid #E5E5EA;
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 16px;
    font-size: 14px;
    color: #86868B;
    align-items: flex-start;
    gap: 10px;
  }
  .transcription-box.show { display: flex; }
  .transcription-box strong { color: #1D1D1F; }

  /* Success badge */
  .success-badge {
    display: none;
    background: #E8F8EE;
    color: #34C759;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 12px;
    width: fit-content;
  }
  .success-badge.show { display: block; }

  /* Divider */
  .or-divider {
    display: flex;
    align-items: center;
    gap: 12px;
    color: #86868B;
    font-size: 13px;
    margin: 16px 0;
  }
  .or-divider::before,.or-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #E5E5EA;
  }

  /* Loading */
  .loading {
    display: none;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    margin-top: 28px;
    color: #86868B;
    font-size: 14px;
  }
  .loading.show { display: flex; }
  .spinner {
    width: 28px; height: 28px;
    border: 2.5px solid #E5E5EA;
    border-top-color: #0071E3;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Results */
  .results {
    display: none;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-top: 28px;
  }
  .results.show { display: grid; }

  .result-card {
    background: white;
    border: 1px solid #E5E5EA;
    border-radius: 18px;
    padding: 24px;
    position: relative;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    transition: all 0.2s ease;
  }
  .result-card:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
  .result-card.winner { border: 2px solid #0071E3; box-shadow: 0 4px 20px rgba(0,113,227,0.12); }

  .winner-tag {
    position: absolute;
    top: -12px; right: 16px;
    background: #0071E3;
    color: white;
    font-size: 10px;
    font-weight: 700;
    padding: 3px 14px;
    border-radius: 20px;
    letter-spacing: 0.5px;
  }

  .engine-name {
    font-size: 11px;
    color: #86868B;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 10px;
  }

  .sentiment-val {
    font-size: 30px;
    font-weight: 700;
    margin-bottom: 6px;
  }
  .sentiment-val.positive { color: #34C759; }
  .sentiment-val.negative { color: #FF3B30; }
  .conf-text { font-size: 13px; color: #86868B; }

  /* ── PROJECT DETAILS PAGE ── */
  .details-page {
    max-width: 900px;
    margin: 0 auto;
    padding: 48px 24px;
  }

  .details-page h1 {
    font-size: 36px;
    font-weight: 700;
    color: #1D1D1F;
    margin-bottom: 36px;
  }

  .section-title {
    font-size: 20px;
    font-weight: 700;
    color: #1D1D1F;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid #E5E5EA;
  }

  .info-card {
    background: white;
    border: 1px solid #E5E5EA;
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
  }

  .pipeline-steps { list-style: none; counter-reset: steps; }
  .pipeline-steps li {
    counter-increment: steps;
    display: flex;
    gap: 16px;
    margin-bottom: 16px;
    align-items: flex-start;
    font-size: 15px;
    color: #1D1D1F;
    line-height: 1.6;
  }
  .pipeline-steps li::before {
    content: counter(steps);
    background: #0071E3;
    color: white;
    width: 26px; height: 26px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 2px;
  }

  .engine-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }

  .engine-card {
    background: white;
    border: 1px solid #E5E5EA;
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
  }

  .engine-card h3 { font-size: 16px; font-weight: 700; color: #1D1D1F; margin-bottom: 16px; }

  .tag {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 12px;
  }
  .tag-blue { background: #E8F0FE; color: #0071E3; }
  .tag-purple { background: #F0E8FE; color: #5856D6; }

  .detail-list { list-style: none; }
  .detail-list li {
    font-size: 14px;
    color: #1D1D1F;
    padding: 4px 0;
    display: flex;
    gap: 8px;
    line-height: 1.5;
  }
  .detail-list li::before { content: '•'; color: #86868B; flex-shrink: 0; }

  .detail-label { font-weight: 600; color: #86868B; font-size: 12px; text-transform: uppercase; margin-bottom: 6px; margin-top: 12px; }

  /* Stats grid */
  .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px; }

  .stat-card {
    background: white;
    border: 1px solid #E5E5EA;
    border-radius: 18px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
  }

  .stat-value { font-size: 32px; font-weight: 700; color: #0071E3; margin-bottom: 4px; }
  .stat-label { font-size: 13px; color: #86868B; font-weight: 500; }

  /* Bar chart */
  .bar-chart { background: white; border: 1px solid #E5E5EA; border-radius: 18px; padding: 28px; box-shadow: 0 2px 12px rgba(0,0,0,0.04); }
  .bar-row { display: flex; align-items: center; gap: 14px; margin-bottom: 16px; }
  .bar-label { font-size: 13px; font-weight: 600; color: #86868B; width: 70px; text-align: right; }
  .bar-track { flex: 1; background: #F5F5F7; border-radius: 6px; height: 28px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 6px; display: flex; align-items: center; padding-left: 12px; font-size: 12px; font-weight: 600; color: white; transition: width 1s ease; }
  .bar-pos { background: #34C759; }
  .bar-neg { background: #FF3B30; }
  .bar-count { font-size: 13px; color: #86868B; width: 60px; }

  /* FOOTER */
  footer {
    text-align: center;
    padding: 24px;
    color: #86868B;
    font-size: 13px;
    border-top: 1px solid #E5E5EA;
    background: white;
    margin-top: auto;
  }

  /* MOBILE */
  @media (max-width: 640px) {
    .home-hero h1 { font-size: 36px; }
    .results.show { grid-template-columns: 1fr; }
    .engine-grid { grid-template-columns: 1fr; }
    .stats-grid { grid-template-columns: 1fr; }
    nav { padding: 0 16px; }
    .nav-badge { display: none; }
  }
</style>
</head>
<body>

<!-- NAV -->
<nav>
  <div class="nav-logo">🧠 Intelligence AI</div>
  <div class="nav-links">
    <a class="nav-link active" onclick="showPage('home')">Home</a>
    <a class="nav-link" onclick="showPage('tool')">Intelligence Tool</a>
    <a class="nav-link" onclick="showPage('details')">Project Details</a>
  </div>
  <div class="nav-badge">Mohammad Hasnain · BS AI</div>
</nav>

<!-- ── HOME PAGE ── -->
<div class="page active" id="page-home">
  <div class="home-hero">
    <h1>Intelligence</h1>
    <p>Pro-level sentiment analysis powered by dual AI engines trained on 40,000 real reviews.</p>
    <img class="hero-img" src="https://images.unsplash.com/photo-1551434678-e076c223a692?q=80&w=2850&auto=format&fit=crop" alt="AI">
    <br>
    <button class="launch-btn" onclick="showPage('tool')">Launch Intelligence Tool</button>
  </div>
</div>

<!-- ── TOOL PAGE ── -->
<div class="page" id="page-tool">
  <div class="tool-page">
    <h1>Feedback Analyzer</h1>
    <p class="subtitle">Type your review or speak it — we'll analyze the sentiment instantly.</p>

    <!-- Success badge -->
    <div class="success-badge" id="successBadge">✓ Voice converted successfully</div>

    <!-- Transcription box -->
    <div class="transcription-box" id="transcriptionBox">
      🎙️ &nbsp;<div><div style="font-size:11px;font-weight:600;color:#86868B;text-transform:uppercase;margin-bottom:4px;">What you said</div><strong id="transcriptionText"></strong></div>
    </div>

    <!-- Input box -->
    <div class="input-card">
      <textarea
        id="reviewText"
        placeholder="Write your review here, or tap the mic to speak..."
        rows="3"
        oninput="autoResize(this)"
      ></textarea>
      <div class="input-actions">
        <div class="left-actions">
          <button class="mic-btn" id="micBtn" onclick="toggleRecording()" title="Click to speak">🎙️</button>
          <span class="mic-label" id="micLabel">Click mic to speak</span>
        </div>
        <button class="analyze-btn" id="analyzeBtn" onclick="analyzeSentiment()">Analyze →</button>
      </div>
    </div>

    <!-- Loading -->
    <div class="loading" id="loading">
      <div class="spinner"></div>
      Analyzing sentiment...
    </div>

    <!-- Results -->
    <div class="results" id="results">
      <div class="result-card" id="nbCard">
        <div class="engine-name">⚡ Fast Engine — Naive Bayes</div>
        <div class="sentiment-val" id="nbSentiment"></div>
        <div class="conf-text" id="nbConf"></div>
      </div>
      <div class="result-card" id="dlCard">
        <div class="engine-name">🧠 Deep Engine — Attention LSTM</div>
        <div class="sentiment-val" id="dlSentiment"></div>
        <div class="conf-text" id="dlConf"></div>
      </div>
    </div>
  </div>
</div>

<!-- ── PROJECT DETAILS PAGE ── -->
<div class="page" id="page-details">
  <div class="details-page">
    <h1>Architecture</h1>

    <!-- Pipeline -->
    <div class="info-card">
      <div class="section-title">🛠️ System Architecture — The Big Data Pipeline</div>
      <p style="color:#86868B;font-size:14px;margin-bottom:20px;">This system was built to handle the <strong>Volume</strong> and <strong>Variety</strong> of the Yelp Open Dataset.</p>
      <ol class="pipeline-steps">
        <li><strong>Data Ingestion (Chunking):</strong> The raw file was 8.6 GB (JSON). Used Python Generators to stream data line-by-line without loading everything into memory.</li>
        <li><strong>ETL & Preprocessing:</strong> Parsed JSON to CSV, removed neutral 3-star reviews to create a clear positive/negative split.</li>
        <li><strong>Balancing:</strong> Applied Undersampling to achieve a perfect 50/50 split — 20,000 positive and 20,000 negative reviews.</li>
        <li><strong>Vectorization:</strong> Used TF-IDF for the Fast Engine and Tokenization with Embedding and Padding for the Deep Engine.</li>
      </ol>
    </div>

    <!-- Dual Engine -->
    <div class="section-title" style="margin-bottom:16px;">🧠 Dual-Engine Intelligence</div>
    <div class="engine-grid">
      <div class="engine-card">
        <h3>⚡ Fast Engine</h3>
        <span class="tag tag-blue">Naive Bayes · Statistical ML</span>
        <div class="detail-label">Strengths</div>
        <ul class="detail-list">
          <li>Extremely fast predictions</li>
          <li>Works well on large datasets</li>
          <li>Easy to train and explain</li>
        </ul>
        <div class="detail-label">Weaknesses</div>
        <ul class="detail-list">
          <li>Uses Bag-of-Words approach</li>
          <li>Ignores word order and grammar</li>
          <li>Struggles with sarcasm and contrast words</li>
        </ul>
      </div>
      <div class="engine-card">
        <h3>🧠 Deep Engine</h3>
        <span class="tag tag-purple">Attention LSTM · Deep Learning</span>
        <div class="detail-label">Strengths</div>
        <ul class="detail-list">
          <li>Learns contextual relationships between words using LSTM memory</li>
          <li>Attention layer focuses on important phrases</li>
          <li>Performs better on complex reviews</li>
        </ul>
        <div class="detail-label">Weaknesses</div>
        <ul class="detail-list">
          <li>Slower than Naive Bayes</li>
          <li>Requires more data and computing power</li>
          <li>Can struggle with unusual sentence structures</li>
        </ul>
      </div>
    </div>

    <!-- Stats -->
    <div class="section-title" style="margin-bottom:16px;">📊 Dataset Statistics</div>
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-value">40,000</div>
        <div class="stat-label">Total Records</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">20,000</div>
        <div class="stat-label">Positive Samples</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">20,000</div>
        <div class="stat-label">Negative Samples</div>
      </div>
    </div>

    <!-- Bar chart -->
    <div class="bar-chart">
      <div class="bar-row">
        <div class="bar-label">Positive</div>
        <div class="bar-track">
          <div class="bar-fill bar-pos" style="width:100%">20,000</div>
        </div>
        <div class="bar-count">50%</div>
      </div>
      <div class="bar-row">
        <div class="bar-label">Negative</div>
        <div class="bar-track">
          <div class="bar-fill bar-neg" style="width:100%">20,000</div>
        </div>
        <div class="bar-count">50%</div>
      </div>
    </div>

  </div>
</div>

<!-- FOOTER -->
<footer>
  Built by Mohammad Hasnain · BS Artificial Intelligence · Trained on 40,000 Yelp Reviews
</footer>

<script>
  // ── BACKEND URL — paste your Render URL here after deploying ──
  const BACKEND_URL = "YOUR_RENDER_BACKEND_URL_HERE";

  // ── PAGE NAVIGATION ──
  function showPage(name) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    document.getElementById('page-' + name).classList.add('active');
    document.querySelectorAll('.nav-link')[['home','tool','details'].indexOf(name)].classList.add('active');
    window.scrollTo(0, 0);
  }

  // ── AUTO RESIZE TEXTAREA ──
  function autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 180) + 'px';
  }

  // ── RECORDING ──
  let mediaRecorder = null;
  let audioChunks = [];
  let isRecording = false;

  async function toggleRecording() {
    if (!isRecording) await startRecording();
    else stopRecording();
  }

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];
      mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
      mediaRecorder.onstop = async () => {
        const blob = new Blob(audioChunks, { type: 'audio/wav' });
        await transcribeAudio(blob);
        stream.getTracks().forEach(t => t.stop());
      };
      mediaRecorder.start();
      isRecording = true;
      document.getElementById('micBtn').classList.add('recording');
      document.getElementById('micBtn').innerHTML = '⏹️';
      document.getElementById('micLabel').textContent = 'Recording... tap to stop';
      document.getElementById('micLabel').classList.add('active');
    } catch {
      alert('Microphone access denied. Please allow microphone permission.');
    }
  }

  function stopRecording() {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
      isRecording = false;
      document.getElementById('micBtn').classList.remove('recording');
      document.getElementById('micBtn').innerHTML = '🎙️';
      document.getElementById('micLabel').textContent = 'Converting voice...';
      document.getElementById('micLabel').classList.remove('active');
    }
  }

  async function transcribeAudio(blob) {
    const form = new FormData();
    form.append('audio', blob, 'audio.wav');
    try {
      const res = await fetch(`${BACKEND_URL}/transcribe`, { method: 'POST', body: form });
      const data = await res.json();
      if (data.text) {
        document.getElementById('reviewText').value = data.text;
        document.getElementById('transcriptionText').textContent = data.text;
        document.getElementById('transcriptionBox').classList.add('show');
        document.getElementById('successBadge').classList.add('show');
        document.getElementById('micLabel').textContent = 'Click mic to speak';
        autoResize(document.getElementById('reviewText'));
      }
    } catch {
      document.getElementById('micLabel').textContent = 'Click mic to speak';
      alert('Voice transcription failed. Please try again.');
    }
  }

  // ── ANALYZE ──
  async function analyzeSentiment() {
    const text = document.getElementById('reviewText').value.trim();
    if (!text) { alert('Please write a review or record your voice first.'); return; }

    document.getElementById('results').classList.remove('show');
    document.getElementById('loading').classList.add('show');
    document.getElementById('analyzeBtn').disabled = true;

    try {
      const res = await fetch(`${BACKEND_URL}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      const data = await res.json();
      showResults(data);
    } catch {
      alert('Analysis failed. Please try again.');
    } finally {
      document.getElementById('loading').classList.remove('show');
      document.getElementById('analyzeBtn').disabled = false;
    }
  }

  function showResults(data) {
    const nb = data.naive_bayes;
    const dl = data.deep_model;

    const nbS = document.getElementById('nbSentiment');
    nbS.textContent = nb.sentiment.toUpperCase();
    nbS.className = `sentiment-val ${nb.sentiment}`;
    document.getElementById('nbConf').textContent = `Confidence: ${nb.confidence}%`;

    const nbCard = document.getElementById('nbCard');
    nbCard.className = nb.winner ? 'result-card winner' : 'result-card';
    let nbTag = nbCard.querySelector('.winner-tag');
    if (nb.winner) { if (!nbTag) { nbTag = document.createElement('div'); nbTag.className='winner-tag'; nbTag.textContent='MOST TRUSTED'; nbCard.appendChild(nbTag); } }
    else if (nbTag) nbTag.remove();

    const dlS = document.getElementById('dlSentiment');
    dlS.textContent = dl.sentiment.toUpperCase();
    dlS.className = `sentiment-val ${dl.sentiment}`;
    document.getElementById('dlConf').textContent = `Confidence: ${dl.confidence}%`;

    const dlCard = document.getElementById('dlCard');
    dlCard.className = dl.winner ? 'result-card winner' : 'result-card';
    let dlTag = dlCard.querySelector('.winner-tag');
    if (dl.winner) { if (!dlTag) { dlTag = document.createElement('div'); dlTag.className='winner-tag'; dlTag.textContent='MOST TRUSTED'; dlCard.appendChild(dlTag); } }
    else if (dlTag) dlTag.remove();

    document.getElementById('results').classList.add('show');
  }

  document.addEventListener('keydown', e => { if (e.key === 'Enter' && e.ctrlKey) analyzeSentiment(); });
</script>

</body>
</html>
