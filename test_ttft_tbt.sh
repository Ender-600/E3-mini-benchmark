#!/bin/bash

# TTFT/TBT Feature Quick Test Script
# Validates that the new latency metrics are working correctly

echo "=========================================="
echo "  TTFT/TBT Latency Metrics Test"
echo "=========================================="
echo ""

# Check Python environment
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python not found"
    exit 1
fi

echo "✅ Python environment check passed"
echo ""

# Create test configuration file
TEST_CONFIG="/tmp/test_ttft_tbt.yaml"
cat > $TEST_CONFIG << 'EOF'
name: "ttft_tbt_test"
arch: "decoder"
context_lengths: [128, 256]  # Test two context lengths
num_tokens: 20               # Generate fewer tokens for faster testing
num_runs: 2                  # Reduce number of runs
warmup_runs: 1
EOF

echo "📝 Created test configuration: $TEST_CONFIG"
echo ""

# Test 1: Quick inference test (GPT-2 small model)
echo "=========================================="
echo "Test 1: GPT-2 Inference (TTFT/TBT Measurement)"
echo "=========================================="
echo ""

if [ -f "configs/model/gpt2-medium.yaml" ]; then
    echo "Running inference test..."
    python -m src.e3bench.eval.bench_infer \
        --model_cfg configs/model/gpt2-medium.yaml \
        --bench_cfg $TEST_CONFIG \
        --output_dir results 2>&1 | tee /tmp/ttft_test_output.log
    
    # Check if output contains TTFT/TBT metrics
    if grep -q "TTFT:" /tmp/ttft_test_output.log && grep -q "TBT:" /tmp/ttft_test_output.log; then
        echo ""
        echo "✅ Test passed: Found TTFT and TBT metrics"
    else
        echo ""
        echo "⚠️  Warning: TTFT/TBT metrics not found in output"
    fi
else
    echo "⚠️  Skipped: gpt2-medium config file not found"
fi

echo ""
echo "=========================================="
echo "Test 2: Check JSON Output"
echo "=========================================="
echo ""

# Find the latest result file
LATEST_RESULT=$(find results -name "inference*.json" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" ")

if [ -n "$LATEST_RESULT" ] && [ -f "$LATEST_RESULT" ]; then
    echo "📄 Latest result file: $LATEST_RESULT"
    echo ""
    
    # Check if new metrics are present
    echo "Checking metric fields..."
    
    HAS_TTFT=$(jq 'if .metrics then .metrics | to_entries | map(select(.value | type == "object" and has("ttft_ms"))) | length > 0 else false end' "$LATEST_RESULT")
    HAS_TBT=$(jq 'if .metrics then .metrics | to_entries | map(select(.value | type == "object" and has("tbt_ms"))) | length > 0 else false end' "$LATEST_RESULT")
    HAS_E2E=$(jq 'if .metrics then .metrics | to_entries | map(select(.value | type == "object" and has("e2e_latency_ms"))) | length > 0 else false end' "$LATEST_RESULT")
    
    if [ "$HAS_TTFT" = "true" ]; then
        echo "✅ ttft_ms: Present"
        # Display actual value
        TTFT_VALUE=$(jq -r '.metrics | to_entries | map(select(.value | type == "object" and has("ttft_ms"))) | .[0].value.ttft_ms' "$LATEST_RESULT")
        echo "   Value: ${TTFT_VALUE} ms"
    else
        echo "❌ ttft_ms: Missing"
    fi
    
    if [ "$HAS_TBT" = "true" ]; then
        echo "✅ tbt_ms: Present"
        TBT_VALUE=$(jq -r '.metrics | to_entries | map(select(.value | type == "object" and has("tbt_ms"))) | .[0].value.tbt_ms' "$LATEST_RESULT")
        echo "   Value: ${TBT_VALUE} ms"
    else
        echo "❌ tbt_ms: Missing"
    fi
    
    if [ "$HAS_E2E" = "true" ]; then
        echo "✅ e2e_latency_ms: Present"
        E2E_VALUE=$(jq -r '.metrics | to_entries | map(select(.value | type == "object" and has("e2e_latency_ms"))) | .[0].value.e2e_latency_ms' "$LATEST_RESULT")
        echo "   Value: ${E2E_VALUE} ms"
    else
        echo "❌ e2e_latency_ms: Missing"
    fi
    
    echo ""
    
    # Validate formula E2E ≈ TTFT + (m-1) × TBT
    if [ "$HAS_TTFT" = "true" ] && [ "$HAS_TBT" = "true" ] && [ "$HAS_E2E" = "true" ]; then
        echo "Validating formula: E2E ≈ TTFT + (m-1) × TBT"
        NUM_TOKENS=20
        EXPECTED_E2E=$(python3 -c "print(${TTFT_VALUE} + ${TBT_VALUE} * (${NUM_TOKENS} - 1))")
        ERROR=$(python3 -c "print(abs(${E2E_VALUE} - ${EXPECTED_E2E}) / ${E2E_VALUE} * 100)")
        
        echo "  TTFT: ${TTFT_VALUE} ms"
        echo "  TBT: ${TBT_VALUE} ms"
        echo "  Expected E2E: ${EXPECTED_E2E} ms"
        echo "  Actual E2E: ${E2E_VALUE} ms"
        echo "  Error: ${ERROR}%"
        
        if (( $(echo "$ERROR < 10" | bc -l) )); then
            echo "✅ Formula validation passed (error < 10%)"
        else
            echo "⚠️  Formula validation failed (error >= 10%)"
        fi
    fi
else
    echo "⚠️  Result file not found, skipping JSON validation"
fi

echo ""
echo "=========================================="
echo "Test 3: Aggregation and Visualization"
echo "=========================================="
echo ""

# Test aggregation functionality
if [ -f "latest/inference/gpt2.json" ] || [ -n "$LATEST_RESULT" ]; then
    echo "Running result aggregation..."
    python -m src.e3bench.report.aggregate --results_dir results --out_dir /tmp/test_tables
    
    if [ -f "/tmp/test_tables/inference_results.csv" ]; then
        echo "✅ Aggregation successful: /tmp/test_tables/inference_results.csv"
        
        # Check columns in CSV
        HEADERS=$(head -1 /tmp/test_tables/inference_results.csv)
        echo ""
        echo "CSV column check:"
        
        for col in "ttft_ms" "tbt_ms" "e2e_latency_ms"; do
            if echo "$HEADERS" | grep -q "$col"; then
                echo "  ✅ $col"
            else
                echo "  ❌ $col"
            fi
        done
        
        # Test plotting functionality
        echo ""
        echo "Generating visualization plots..."
        python -m src.e3bench.report.plots --tables /tmp/test_tables --out /tmp/test_figs
        
        if [ -f "/tmp/test_figs/ttft_tbt_curves.png" ]; then
            echo "✅ TTFT/TBT curves plot: /tmp/test_figs/ttft_tbt_curves.png"
        else
            echo "⚠️  TTFT/TBT curves plot not generated (may need scaling data)"
        fi
        
        if [ -f "/tmp/test_figs/latency_composition.png" ]; then
            echo "✅ Latency composition plot: /tmp/test_figs/latency_composition.png"
        else
            echo "⚠️  Latency composition plot not generated (may need scaling data)"
        fi
    else
        echo "❌ Aggregation failed"
    fi
else
    echo "⚠️  Skipped: No inference results available"
fi

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""

# Cleanup
rm -f $TEST_CONFIG

echo "📝 Log file: /tmp/ttft_test_output.log"
echo "📊 Test tables: /tmp/test_tables/"
echo "📈 Test plots: /tmp/test_figs/"
echo ""

echo "Test complete! Please review the output above to confirm functionality."
echo ""
echo "To run full context scaling tests, execute:"
echo "  ./scripts/bench_scaling.sh"
echo ""
echo "View detailed documentation:"
echo "  docs/ttft_tbt_metrics.md"
echo ""
