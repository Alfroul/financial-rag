"""Custom CSS for Financial RAG Gradio UI — dark financial-blue theme."""

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg-deep: #0A0E27;
    --bg-card: #0D1421;
    --bg-input: #111832;
    --border: #1E293B;
    --accent-blue: #1a73e8;
    --accent-blue-hover: #1565c0;
    --accent-gold: #C9A84C;
    --accent-gold-dim: rgba(201,168,76,0.2);
    --text-primary: #E8E6E3;
    --text-muted: #9CA3AF;
    --text-dim: #4B5563;
    --success: #22C55E;
    --warning: #EAB308;
    --error: #EF4444;
}

.gradio-container {
    font-family: 'Noto Sans SC', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
    max-width: 1400px !important;
}

/* ===== Sidebar Column ===== */
.sidebar-col {
    background: linear-gradient(180deg, #0D1421 0%, #091018 100%);
    border-right: 2px solid var(--accent-gold);
    padding: 1rem 0.8rem !important;
    min-width: 280px;
}

.sidebar-title {
    font-size: 0.85rem;
    color: var(--accent-gold);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 0.1rem;
}

.sidebar-subtitle {
    font-size: 0.6rem;
    color: var(--text-dim);
    letter-spacing: 1px;
    text-transform: uppercase;
}

.section-header {
    font-size: 0.72rem;
    color: var(--accent-gold);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    border-bottom: 1px solid var(--accent-gold-dim);
    padding-bottom: 0.25rem;
    margin-top: 1rem;
    margin-bottom: 0.4rem;
}

/* ===== Chat Messages ===== */
.message.user {
    background-color: #111832 !important;
    border-left: 3px solid var(--accent-gold) !important;
    border-radius: 4px !important;
}

.message.bot {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}

/* ===== Source Cards ===== */
.source-card {
    border-left: 3px solid var(--accent-blue);
    padding: 8px 12px;
    margin: 8px 0;
    background: var(--bg-input);
    border-radius: 4px;
}

.source-title {
    color: var(--accent-blue);
    font-weight: 600;
    font-size: 0.85rem;
}

.source-score {
    color: var(--text-muted);
    font-size: 0.75rem;
}

.source-content {
    color: var(--text-primary);
    font-size: 0.8rem;
    margin-top: 4px;
}

/* ===== Agent Steps ===== */
.agent-step {
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 8px 12px;
    margin: 6px 0;
}

.step-label {
    color: var(--accent-gold);
    font-weight: 600;
    font-size: 0.85rem;
}

.step-content {
    color: var(--text-primary);
    font-size: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ===== Agent Timeline ===== */
.agent-timeline {
    position: relative;
    padding-left: 24px;
}

.timeline-step {
    position: relative;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 0;
}

.timeline-dot {
    position: absolute;
    left: -24px;
    top: 8px;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--accent-gold);
    border: 2px solid var(--bg-deep);
    z-index: 1;
}

.timeline-connector {
    position: relative;
    left: -19px;
    width: 2px;
    height: 20px;
    background: var(--border);
}

.timeline-content {
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 8px 12px;
    width: 100%;
}

/* ===== Correction Report ===== */
.correction-pass {
    color: var(--success);
    font-weight: 600;
}

.correction-flag {
    color: var(--warning);
    font-weight: 600;
}

/* ===== Metrics Cards ===== */
.metric-card {
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 16px;
    text-align: center;
}

.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    color: var(--accent-gold);
    font-weight: 600;
}

.metric-label {
    font-size: 0.72rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 2px;
}

/* ===== Branding Footer ===== */
.brand-footer {
    margin-top: 1.5rem;
    padding: 0.5rem 0;
    border-top: 1px solid var(--accent-gold-dim);
    text-align: center;
}

.brand-name {
    font-size: 0.7rem;
    color: var(--accent-gold);
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 600;
}

.brand-version {
    font-size: 0.6rem;
    color: var(--text-dim);
    letter-spacing: 1px;
    margin-top: 0.15rem;
}

/* ===== Status Indicator ===== */
.status-connected {
    color: var(--success);
    font-size: 0.78rem;
    letter-spacing: 0.5px;
}

.status-waiting {
    color: var(--warning);
    font-size: 0.78rem;
    letter-spacing: 0.5px;
}

/* ===== Scrollbar ===== */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2A3A5C; }
"""
