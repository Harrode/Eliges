#!/bin/bash
set -e

# ── Start Elasticsearch via Docker ─────────────────────────────────────
echo "[1/4] Starting Elasticsearch..."
if docker ps --format '{{.Names}}' | grep -q es-demo; then
    echo "  ES container already running."
else
    docker run -d --name es-demo \
        -p 9200:9200 \
        -e "discovery.type=single-node" \
        -e "xpack.security.enabled=false" \
        -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
        docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    echo "  Waiting for ES to start..."
    sleep 15
fi

# Wait for ES to be ready
for i in $(seq 1 30); do
    if curl -s http://localhost:9200 > /dev/null 2>&1; then
        echo "  ES is ready!"
        break
    fi
    echo "  Waiting... ($i/30)"
    sleep 2
done

# ── Install Python dependencies ────────────────────────────────────────
echo "[2/4] Installing dependencies..."
pip3 install -q -r requirements.txt 2>/dev/null || \
pip install -q -r requirements.txt 2>/dev/null

# ── Initialize ES with mock data ───────────────────────────────────────
echo "[3/4] Initializing Elasticsearch with mock patient data..."
python3 scripts/init_es.py

# ── Start Backend ──────────────────────────────────────────────────────
echo "[4/4] Starting backend server on http://localhost:8000"
echo ""
echo "=========================================="
echo "  Clinical Eligibility Screening System"
echo "  URL: http://localhost:8000"
echo "  ES:  http://localhost:9200"
echo "=========================================="
echo ""
python3 backend.py
