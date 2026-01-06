#!/bin/bash
# Quick progress checker for bandit workflow
# Usage: ./check_progress.sh <RUN_ID>
# Example: ./check_progress.sh bandit_20260104_131923

RUN_ID="${1:-bandit_20260104_131923}"

echo "=== Bandit Workflow Progress Check ==="
echo "Run ID: $RUN_ID"
echo ""

# Step 1: Corpus
echo "Step 1 - Corpus Building:"
if [ -f "runs/${RUN_ID}/cache/representation_corpus.json" ]; then
    SIZE=$(python3 -c "import json; print(len(json.load(open('runs/${RUN_ID}/cache/representation_corpus.json'))))" 2>/dev/null)
    echo "  ✅ Corpus built: $SIZE examples"
else
    echo "  ⏳ Corpus not built yet"
fi

# Step 2: Metrics
echo ""
echo "Step 2 - Metrics Pre-computation:"
METRIC_COUNT=$(find runs/${RUN_ID}/cache/metrics -name "*.json" 2>/dev/null | wc -l)
if [ "$METRIC_COUNT" -gt 0 ]; then
    echo "  ✅ Metrics computed: $METRIC_COUNT files"
else
    echo "  ⏳ Metrics not computed yet"
fi

# Step 3: Bandit State
echo ""
echo "Step 3 - Bandit Evaluations:"
if [ -f "runs/${RUN_ID}/bandit_state.json" ]; then
    EVALS=$(python3 -c "import json; print(len(json.load(open('runs/${RUN_ID}/bandit_state.json')).get('trajectory', [])))" 2>/dev/null)
    if [ "$EVALS" -gt 0 ]; then
        echo "  ✅ Evaluations completed: $EVALS"
        # Show latest evaluation
        python3 -c "import json; t=json.load(open('runs/${RUN_ID}/bandit_state.json')).get('trajectory', []); print(f\"  Latest: {t[-1].get('arm')} (reward: {t[-1].get('reward', 0):.4f})\")" 2>/dev/null || true
    else
        echo "  ⏳ Evaluations started but none completed yet (check logs for errors)"
    fi
else
    echo "  ⏳ Evaluations not started yet"
fi

# Also check progress CSV for more details
if [ -f "runs/${RUN_ID}/bandit_progress.csv" ]; then
    CSV_ROWS=$(tail -n +2 "runs/${RUN_ID}/bandit_progress.csv" 2>/dev/null | wc -l)
    if [ "$CSV_ROWS" -gt 0 ]; then
        echo "  📊 Progress CSV: $CSV_ROWS rows"
    fi
fi

# Step 4: Results
echo ""
echo "Step 4 - Final Results:"
if [ -f "runs/${RUN_ID}/bandit_results.json" ]; then
    echo "  ✅ Workflow complete!"
    python3 -c "import json; r=json.load(open('runs/${RUN_ID}/bandit_results.json')); print(f\"  Best arm: {r.get('best_arm')}\"); print(f\"  Budget used: {r.get('budget_used')}\")" 2>/dev/null
else
    echo "  ⏳ Workflow still running"
fi

# Latest log entries
echo ""
echo "Latest log entries:"
tail -5 "runs/${RUN_ID}/logs/bandit_runner.log" 2>/dev/null || echo "  Log file not found"

