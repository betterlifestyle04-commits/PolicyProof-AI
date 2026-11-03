#!/usr/bin/env bash
set -euo pipefail
exec python -m streamlit run research_dashboard.py \
  --server.address 0.0.0.0 \
  --server.port "${PORT:-10000}" \
  --server.headless true \
  --browser.gatherUsageStats false
