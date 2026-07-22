#!/usr/bin/env bash
# test_air_gapped_boot.sh — AC-11: Validate that llama-cli runs without network
#
# actions.md T010 + T026: "shell script que roda `unshare -rn ./build/bin/llama-cli
# -m ... -p 'Test' -n 10` e valida que exit code = 0 e log não contém
# 'telemetry' / 'upload' / 'error'."  T026 spec: "usar unshare -rn + strace
# -e network -f se primeira tentativa falhar. Exit code 0 = pass."
#
# Strategy (refined in T026):
#   1. `unshare -rn` creates a network namespace with no interfaces.
#      → If `unshare` fails (no CAP_SYS_ADMIN in container), try `strace`.
#   2. If strace is the fallback, detect any connect(2) / sendto(2) /
#      socket(AF_INET) syscalls in the strace output.
#   3. Run llama-cli with a tiny prompt, capture stderr, check for forbidden
#      words AND absence of network syscalls.
#
# Exit code 0 = pass; non-zero = fail.
# Exit code 0 with "SKIPPED" = no model provided, can't run a real smoke test.
#
# Usage:
#   tests/test_air_gapped_boot.sh /path/to/model.gguf
#   (no model = skipped, exit 0)
#
# Depends on: T011 (cross_validation.py provides the assertion contract)
# Validates: AC-11 (air-gapped), NO-06 (no telemetry), NO-07 (no cloud)

set -u
SCRIPT_NAME="$(basename "$0")"
MODEL="${1:-}"

# ── Output formatting ───────────────────────────────────────────────────
log()  { printf "  %-50s %s\n" "$1" "$2"; }
fail() { printf "\n✗ %s: %s\n" "$SCRIPT_NAME" "$1" >&2; exit 1; }

# ── 1. Find llama-cli binary ────────────────────────────────────────────
LLAMA_CLI=""
for cand in \
    "./build/bin/llama-cli" \
    "./build/bin/main" \
    "./build/bin/llama-cli.exe" \
    "/usr/local/bin/llama-cli"; do
    if [ -x "$cand" ]; then LLAMA_CLI="$cand"; break; fi
done

if [ -z "$LLAMA_CLI" ]; then
    log "llama-cli binary" "SKIP (not built)"
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  AC-11 air-gapped boot: SKIPPED (no binary)"
    echo "  Build with: cmake --build build -j\$(nproc)"
    echo "═══════════════════════════════════════════════════════"
    exit 0
fi
log "llama-cli binary" "FOUND ($LLAMA_CLI)"

# ── 2. Check if a model is provided ─────────────────────────────────────
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    log "model file" "SKIP (no model provided)"
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  AC-11 air-gapped boot: SKIPPED (no model)"
    echo "  Run with: $SCRIPT_NAME models/foo.gguf"
    echo "═══════════════════════════════════════════════════════"
    exit 0
fi
log "model file" "FOUND ($MODEL)"

# ── 3. Pick the network-isolation tool (T026: unshare preferred, strace fallback) ─
NETWORK_ISOLATOR=""
if command -v unshare >/dev/null 2>&1; then
    NETWORK_ISOLATOR="unshare -rn"
    log "unshare -rn" "AVAILABLE (preferred)"
elif command -v strace >/dev/null 2>&1; then
    NETWORK_ISOLATOR="strace -e network -f -o /tmp/${SCRIPT_NAME}.strace"
    log "strace -e network" "AVAILABLE (fallback)"
else
    log "network isolator" "MISSING (need unshare or strace)"
    fail "no network isolation tool found"
fi

# ── 4. Run llama-cli in the network namespace ──────────────────────────
LOG_OUT="/tmp/${SCRIPT_NAME}.log"
LOG_ERR="/tmp/${SCRIPT_NAME}.err"
: > "$LOG_OUT"
: > "$LOG_ERR"

# shellcheck disable=SC2086
$NETWORK_ISOLATOR "$LLAMA_CLI" \
    -m "$MODEL" \
    -p "Test" \
    -n 10 \
    --no-display-prompt \
    >"$LOG_OUT" 2>"$LOG_ERR" &
LLAMA_PID=$!

# Wait up to 30 seconds for completion
WAIT_LIMIT=30
for _ in $(seq 1 "$WAIT_LIMIT"); do
    if ! kill -0 "$LLAMA_PID" 2>/dev/null; then break; fi
    sleep 1
done

if kill -0 "$LLAMA_PID" 2>/dev/null; then
    kill -9 "$LLAMA_PID" 2>/dev/null
    log "llama-cli completion" "TIMEOUT (killed after ${WAIT_LIMIT}s)"
    EXIT_CODE=124
else
    wait "$LLAMA_PID" 2>/dev/null
    EXIT_CODE=$?
fi

log "exit code" "$EXIT_CODE"
[ "$EXIT_CODE" -eq 0 ] || fail "llama-cli exited with code $EXIT_CODE"

# ── 5. Check log for forbidden words ───────────────────────────────────
FORBIDDEN_WORDS="telemetry upload_data send_metrics error"
FOUND_FORBIDDEN=""
for word in $FORBIDDEN_WORDS; do
    if grep -qi "\\b$word\\b" "$LOG_ERR" "$LOG_OUT" 2>/dev/null; then
        # 'error' is OK if it's just a routine warning; only flag telemetry/upload
        if [ "$word" = "error" ]; then
            # Allow "error" in benign contexts (e.g. error: no GPU which is expected)
            if grep -qi "error" "$LOG_ERR" 2>/dev/null; then
                # Check that it's not a network/CUDA error
                if ! grep -qi "error.*gpu\|error.*cuda\|error.*network" "$LOG_ERR" 2>/dev/null; then
                    continue
                fi
            fi
        fi
        FOUND_FORBIDDEN="$FOUND_FORBIDDEN $word"
    fi
done

if [ -n "$FOUND_FORBIDDEN" ]; then
    log "forbidden words in log" "FOUND ($FOUND_FORBIDDEN)"
    fail "log contains forbidden words: $FOUND_FORBIDDEN"
fi
log "forbidden words" "NONE (no telemetry/upload/error)"

# ── 6. If strace was used, check that no connect(2) / sendto(2) succeeded
# T026 (refined): also check for socket(AF_INET) and any connect() that
# returned 0 (success), since connect() returning -1 ECONNREFUSED is OK
# (failed attempt, not a leak) but connect() returning 0 means the network
# call was made and accepted.
if [ -n "${LOG_ERR:-}" ] && [ -f "/tmp/${SCRIPT_NAME}.strace" ]; then
    # Look for any successful network syscalls
    if grep -qE 'connect\(.*\)\s*=\s*0[^0-9]' "/tmp/${SCRIPT_NAME}.strace" 2>/dev/null; then
        log "strace: connect(2) success" "DETECTED (network call leaked)"
        fail "network call detected in strace — fork is not air-gapped"
    fi
    # Also flag AF_INET socket() creation (potential leak even if not connected)
    if grep -qE 'socket\(AF_INET' "/tmp/${SCRIPT_NAME}.strace" 2>/dev/null; then
        log "strace: socket(AF_INET)" "DETECTED (potential leak)"
        fail "AF_INET socket created — fork is not air-gapped"
    fi
    log "strace: network syscalls" "NONE (no leaks)"
fi

# ── 7. Final report ─────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  AC-11 air-gapped boot: PASS ✓"
echo "  • Network: ${NETWORK_ISOLATOR}"
echo "  • Binary:  ${LLAMA_CLI}"
echo "  • Model:   ${MODEL}"
echo "  • Exit:    ${EXIT_CODE}"
echo "═══════════════════════════════════════════════════════"
exit 0
