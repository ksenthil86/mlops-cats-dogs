#!/bin/bash
# ============================================
# Smoke Test Script
# Tests health and prediction endpoints after deployment.
# Fails with exit code 1 if any test fails.
# ============================================

set -e

# Configuration
SERVICE_URL="${SERVICE_URL:-http://localhost:80}"
MAX_RETRIES=10
RETRY_DELAY=5

echo "========================================="
echo "  Smoke Tests - Cats vs Dogs Classifier"
echo "========================================="
echo "Service URL: $SERVICE_URL"
echo ""

# ----- Helper Functions -----
wait_for_service() {
    echo "[INFO] Waiting for service to be ready..."
    for i in $(seq 1 $MAX_RETRIES); do
        if curl -s -o /dev/null -w "%{http_code}" "$SERVICE_URL/health" | grep -q "200"; then
            echo "[OK] Service is ready!"
            return 0
        fi
        echo "[WAIT] Attempt $i/$MAX_RETRIES - Service not ready, retrying in ${RETRY_DELAY}s..."
        sleep $RETRY_DELAY
    done
    echo "[FAIL] Service did not become ready after $MAX_RETRIES attempts"
    return 1
}

# ----- Test 1: Health Check -----
test_health() {
    echo ""
    echo "[TEST] Health Check Endpoint..."
    RESPONSE=$(curl -s -w "\n%{http_code}" "$SERVICE_URL/health")
    HTTP_CODE=$(echo "$RESPONSE" | tail -1)
    BODY=$(echo "$RESPONSE" | head -1)

    if [ "$HTTP_CODE" -eq 200 ]; then
        echo "[PASS] Health check returned 200"
        echo "  Response: $BODY"
    else
        echo "[FAIL] Health check returned $HTTP_CODE"
        echo "  Response: $BODY"
        return 1
    fi
}

# ----- Test 2: Prediction Endpoint -----
test_prediction() {
    echo ""
    echo "[TEST] Prediction Endpoint..."

    # Create a simple test image (red 224x224 square)
    python3 -c "
from PIL import Image
import numpy as np
img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
img.save('/tmp/test_image.jpg')
print('Test image created.')
"

    RESPONSE=$(curl -s -w "\n%{http_code}" \
        -X POST "$SERVICE_URL/predict" \
        -F "file=@/tmp/test_image.jpg")
    HTTP_CODE=$(echo "$RESPONSE" | tail -1)
    BODY=$(echo "$RESPONSE" | head -1)

    if [ "$HTTP_CODE" -eq 200 ]; then
        echo "[PASS] Prediction returned 200"
        echo "  Response: $BODY"

        # Verify response has expected fields
        if echo "$BODY" | python3 -c "import sys, json; d=json.load(sys.stdin); assert 'label' in d and 'probability' in d"; then
            echo "[PASS] Response has required fields (label, probability)"
        else
            echo "[FAIL] Response missing required fields"
            return 1
        fi
    else
        echo "[FAIL] Prediction returned $HTTP_CODE"
        echo "  Response: $BODY"
        return 1
    fi

    # Cleanup
    rm -f /tmp/test_image.jpg
}

# ----- Test 3: Metrics Endpoint -----
test_metrics() {
    echo ""
    echo "[TEST] Metrics Endpoint..."
    RESPONSE=$(curl -s -w "\n%{http_code}" "$SERVICE_URL/metrics")
    HTTP_CODE=$(echo "$RESPONSE" | tail -1)

    if [ "$HTTP_CODE" -eq 200 ]; then
        echo "[PASS] Metrics endpoint returned 200"
    else
        echo "[FAIL] Metrics endpoint returned $HTTP_CODE"
        return 1
    fi
}

# ----- Run All Tests -----
echo ""
echo "Starting smoke tests..."
echo ""

wait_for_service
test_health
test_prediction
test_metrics

echo ""
echo "========================================="
echo "  ALL SMOKE TESTS PASSED!"
echo "========================================="
exit 0
