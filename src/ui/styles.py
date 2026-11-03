"""
JurisGraph Design System & Custom CSS
Implements the clinical dark-mode legal AI workspace specification:
- Background: #08090C to #0B0D12
- Surfaces/Cards: #11141B to #161922
- Borders: 1px solid #262C38
- Primary Text: #E7E9EE, Secondary Text: #9AA1AE
- Typography: Inter & JetBrains Mono
- Restrained transitions (150-250ms)
- No emojis, accessible contrast, solid professional colors.
"""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Base Application Reset & Typography */
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #08090C !important;
    color: #E7E9EE !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #0D1017 !important;
    border-right: 1px solid #1E232F !important;
    padding-top: 1rem !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #9AA1AE !important;
}

/* Wordmark & Brand */
.jg-wordmark-container {
    padding: 0 4px 16px 4px;
    border-bottom: 1px solid #1E232F;
    margin-bottom: 16px;
}

.jg-wordmark-title {
    font-size: 20px;
    font-weight: 700;
    color: #E7E9EE;
    letter-spacing: -0.02em;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.jg-wordmark-badge {
    font-size: 9px;
    font-weight: 600;
    background: rgba(76, 111, 255, 0.15);
    color: #7C98FF;
    border: 1px solid rgba(76, 111, 255, 0.3);
    padding: 2px 6px;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.jg-wordmark-desc {
    font-size: 11px;
    color: #8E97A8;
    margin-top: 4px;
    margin-bottom: 0;
    line-height: 1.35;
}

/* Navigation Buttons */
.stButton button.jg-nav-btn {
    width: 100% !important;
    text-align: left !important;
    background-color: transparent !important;
    color: #9AA1AE !important;
    border: 1px solid transparent !important;
    border-radius: 6px !important;
    padding: 8px 12px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all 180ms ease-in-out !important;
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
}

.stButton button.jg-nav-btn:hover {
    background-color: #141822 !important;
    color: #E7E9EE !important;
    border-color: #262C38 !important;
}

.stButton button.jg-nav-btn-active {
    width: 100% !important;
    text-align: left !important;
    background-color: #161B26 !important;
    color: #E7E9EE !important;
    border: 1px solid #262C38 !important;
    border-left: 3px solid #4C6FFF !important;
    border-radius: 6px !important;
    padding: 8px 12px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

/* Page Header */
.jg-page-header {
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid #1E232F;
}

.jg-page-title {
    font-size: 24px;
    font-weight: 700;
    color: #E7E9EE;
    letter-spacing: -0.02em;
    margin: 0;
}

.jg-page-subtitle {
    font-size: 13px;
    color: #8E97A8;
    margin-top: 4px;
    margin-bottom: 0;
}

/* Clinical Dark Surface Cards */
.jg-card {
    background-color: #11141B;
    border: 1px solid #262C38;
    border-radius: 8px;
    padding: 18px 20px;
    margin-bottom: 16px;
    transition: border-color 200ms ease;
}

.jg-card:hover {
    border-color: #323A4A;
}

.jg-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.jg-card-title {
    font-size: 14px;
    font-weight: 600;
    color: #E7E9EE;
    letter-spacing: -0.01em;
    margin: 0;
}

.jg-card-caption {
    font-size: 12px;
    color: #8E97A8;
    margin: 0;
}

/* KPI / Metric Box */
.jg-kpi-card {
    background-color: #11141B;
    border: 1px solid #262C38;
    border-radius: 8px;
    padding: 14px 16px;
    text-align: left;
    height: 100%;
}

.jg-kpi-label {
    font-size: 11px;
    font-weight: 500;
    color: #8E97A8;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 6px;
}

.jg-kpi-value {
    font-size: 24px;
    font-weight: 700;
    color: #E7E9EE;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.02em;
    line-height: 1.1;
}

.jg-kpi-subtext {
    font-size: 11px;
    color: #63687A;
    margin-top: 4px;
}

/* Document Status Banner */
.jg-status-banner {
    background-color: #11141B;
    border: 1px solid #262C38;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 13px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.jg-status-banner.loaded {
    border-left: 3px solid #3DD68C;
}

.jg-status-banner.empty {
    border-left: 3px solid #63687A;
    color: #8E97A8;
}

/* Rhetorical Role Tags */
.jg-tag {
    display: inline-flex;
    align-items: center;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

.jg-tag-facts { background-color: rgba(56, 189, 248, 0.12); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.28); }
.jg-tag-issue { background-color: rgba(167, 139, 250, 0.12); color: #A78BFA; border: 1px solid rgba(167, 139, 250, 0.28); }
.jg-tag-petitioner { background-color: rgba(96, 165, 250, 0.12); color: #60A5FA; border: 1px solid rgba(96, 165, 250, 0.28); }
.jg-tag-respondent { background-color: rgba(251, 146, 60, 0.12); color: #FB923C; border: 1px solid rgba(251, 146, 60, 0.28); }
.jg-tag-reasoning { background-color: rgba(192, 132, 252, 0.12); color: #C084FC; border: 1px solid rgba(192, 132, 252, 0.28); }
.jg-tag-decision { background-color: rgba(61, 214, 140, 0.12); color: #3DD68C; border: 1px solid rgba(61, 214, 140, 0.28); }
.jg-tag-none { background-color: rgba(99, 104, 122, 0.12); color: #9AA1AE; border: 1px solid rgba(99, 104, 122, 0.28); }

/* Role Distribution Pills */
.jg-role-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 10px;
}

.jg-role-pill {
    background-color: #0E1118;
    border: 1px solid #262C38;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 12px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
}

.jg-role-pill-name {
    color: #8E97A8;
    font-weight: 500;
}

.jg-role-pill-count {
    color: #E7E9EE;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}

/* Verification Badges */
.jg-badge-verified {
    background-color: rgba(61, 214, 140, 0.10);
    color: #3DD68C;
    border: 1px solid #3DD68C;
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'JetBrains Mono', monospace;
}

.jg-badge-unverified {
    background-color: rgba(240, 101, 75, 0.10);
    color: #F0654B;
    border: 1px solid #F0654B;
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'JetBrains Mono', monospace;
}

.jg-badge-offline {
    background-color: rgba(99, 104, 122, 0.12);
    color: #9AA1AE;
    border: 1px solid #4B5263;
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'JetBrains Mono', monospace;
}

/* Grading Badges */
.jg-grade-correct {
    background-color: rgba(61, 214, 140, 0.12);
    color: #3DD68C;
    border: 1px solid rgba(61, 214, 140, 0.3);
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

.jg-grade-ambiguous {
    background-color: rgba(245, 184, 65, 0.12);
    color: #F5B841;
    border: 1px solid rgba(245, 184, 65, 0.3);
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

.jg-grade-incorrect {
    background-color: rgba(99, 104, 122, 0.15);
    color: #9AA1AE;
    border: 1px solid rgba(99, 104, 122, 0.3);
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

/* Monospace Utilities */
.jg-mono {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #8E97A8;
}

.jg-mono-score {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    color: #E7E9EE;
}

/* Source Passage Item */
.jg-source-item {
    background-color: #0E1118;
    border: 1px solid #262C38;
    border-radius: 6px;
    padding: 12px 14px;
    margin-top: 8px;
    font-size: 13px;
    line-height: 1.5;
}

/* Confidence Gauge Container */
.jg-gauge-container {
    background-color: #11141B;
    border: 1px solid #262C38;
    border-radius: 8px;
    padding: 14px 16px;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.jg-gauge-label {
    font-size: 11px;
    font-weight: 600;
    color: #8E97A8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}

.jg-gauge-caption {
    font-size: 11px;
    color: #63687A;
    margin-top: 4px;
    line-height: 1.3;
}

/* Pipeline Architecture Nodes */
.jg-pipe-stage-card {
    background-color: #11141B;
    border: 1px solid #262C38;
    border-radius: 8px;
    padding: 14px 16px;
    height: 100%;
    position: relative;
}

.jg-pipe-stage-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background-color: #1E2536;
    color: #7C98FF;
    font-size: 11px;
    font-weight: 700;
    margin-bottom: 8px;
}

.jg-pipe-stage-title {
    font-size: 13px;
    font-weight: 600;
    color: #E7E9EE;
    margin-bottom: 8px;
}

.jg-pipe-stage-item {
    font-size: 11px;
    color: #8E97A8;
    margin-bottom: 3px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.jg-pipe-stage-item::before {
    content: "•";
    color: #4C6FFF;
}

/* Pipeline Flow Diagrams */
.jg-flow-box {
    background-color: #0E1118;
    border: 1px solid #262C38;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 12px;
    text-align: center;
    font-weight: 500;
}

.jg-flow-box.active {
    border-color: #4C6FFF;
    background-color: #141A29;
    color: #E7E9EE;
}

.jg-flow-arrow {
    display: flex;
    align-items: center;
    justify-content: center;
    color: #4C6FFF;
    font-weight: bold;
    font-size: 14px;
}

/* Native Button & Input Overrides */
button[kind="primary"] {
    background-color: #4C6FFF !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    transition: background-color 150ms ease !important;
}

button[kind="primary"]:hover {
    background-color: #3B5BDB !important;
}

button[kind="secondary"] {
    background-color: #161B26 !important;
    color: #E7E9EE !important;
    border: 1px solid #262C38 !important;
    border-radius: 6px !important;
    transition: all 150ms ease !important;
}

button[kind="secondary"]:hover {
    background-color: #1F2533 !important;
    border-color: #3E4656 !important;
}

/* Text Input & Selectbox */
div[data-baseweb="input"] {
    background-color: #11141B !important;
    border-color: #262C38 !important;
    border-radius: 6px !important;
    color: #E7E9EE !important;
}

div[data-baseweb="select"] > div {
    background-color: #11141B !important;
    border-color: #262C38 !important;
    border-radius: 6px !important;
    color: #E7E9EE !important;
}

/* File Uploader */
div[data-testid="stFileUploader"] {
    background-color: #11141B !important;
    border: 1px dashed #262C38 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    transition: border-color 200ms ease !important;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #4C6FFF !important;
}

/* Expanders */
div[data-testid="stExpander"] {
    background-color: #11141B !important;
    border: 1px solid #262C38 !important;
    border-radius: 6px !important;
    margin-top: 10px !important;
}

/* Tables */
table, .jg-table {
    border-collapse: collapse;
    width: 100%;
    font-size: 12px;
}

th, .jg-table th {
    border-bottom: 1px solid #262C38;
    color: #8E97A8;
    padding: 8px 10px;
    font-weight: 600;
    text-align: left;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

td, .jg-table td {
    border-bottom: 1px solid #1E232F;
    color: #E7E9EE;
    padding: 8px 10px;
}

tr:hover td, .jg-table tr:hover td {
    background-color: #141822;
}

/* Code & Pre overrides for dark theme */
pre, code {
    background-color: #0E1118 !important;
    color: #E7E9EE !important;
    border-color: #262C38 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

code {
    padding: 2px 5px !important;
    border-radius: 4px !important;
    font-size: 11px !important;
}
</style>
"""
