#!/usr/bin/env bash
#
# apply-dispatch-patches.sh
#
# Aplica os patches de dispatch do BitNet CPU-Universal sobre o
# 3rdparty/llama.cpp após `git submodule update --init`.
#
# Contexto:
#   O submodule 3rdparty/llama.cpp aponta para o fork upstream
#   (https://github.com/Eddie-Wang1120/llama.cpp.git, base commit 1f86f05).
#   Os commits de feature foram consolidados em dois patches cumulativos:
#
#   04-ACDC-rect-FFN.patch   — L3 ACDC + L5 HRR + L4 K_i8 cache + FaseIII rect
#                              (inclui build_falcon ACDC rect gate)
#   05-ACDC-rect-LLaMA.patch — adds ACDC rect gate to build_llama
#                              (needed for Falcon3-3B/10B which report arch=llama)
#
#   Patches 01-03 existem como referência histórica mas não são mais usados.
#
# Uso:
#   ./scripts/apply-dispatch-patches.sh           # aplica
#   ./scripts/apply-dispatch-patches.sh --check   # só verifica
#   ./scripts/apply-dispatch-patches.sh --reverse # reverte
#
# Pré-requisitos:
#   - 3rdparty/llama.cpp/ existe e está checked-out na base 1f86f05
#   - patches/llama.cpp/04-ACDC-rect-FFN.patch existe
#   - patches/llama.cpp/05-ACDC-rect-LLaMA.patch existe
#
# Saída:
#   - Aplica patches 04 + 05 em sequência
#   - Idempotente: detecta se já aplicados e sai 0
#   - Falha com mensagem clara se patch não aplicar (sai 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SUBMODULE="$REPO_ROOT/3rdparty/llama.cpp"
PATCHES_DIR="$REPO_ROOT/patches/llama.cpp"

PATCH_04="$PATCHES_DIR/04-ACDC-rect-FFN.patch"
PATCH_05="$PATCHES_DIR/05-ACDC-rect-LLaMA.patch"

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

err()  { echo -e "${RED}[ERROR]${NC} $*" >&2; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

# Pré-condições
if [ ! -d "$SUBMODULE" ]; then
    err "submodule não encontrado: $SUBMODULE"
    err "rode 'git submodule update --init --recursive' antes"
    exit 1
fi
if [ ! -f "$PATCH_04" ]; then
    err "patch não encontrado: $PATCH_04"
    exit 1
fi
if [ ! -f "$PATCH_05" ]; then
    err "patch não encontrado: $PATCH_05"
    exit 1
fi

MODE="apply"
if [ "${1:-}" = "--check" ]; then MODE="check"; fi
if [ "${1:-}" = "--reverse" ]; then MODE="reverse"; fi

cd "$SUBMODULE"

CURRENT_HEAD=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo "submodule HEAD: $CURRENT_HEAD"

# Sentinela patch 04: llm_build_ffn_acdc_rect (unique to patch 04)
is_04_applied() {
    grep -qF 'llm_build_ffn_acdc_rect' src/llama.cpp
}

# Sentinela patch 05: bitnet_acdc_ffn_rect_llama (unique to patch 05)
is_05_applied() {
    grep -qF 'bitnet_acdc_ffn_rect_llama' src/llama.cpp
}

case "$MODE" in
    check)
        all_ok=true
        if is_04_applied; then
            ok "patch 04 aplicado (L3+L5+L4cache+FaseIII)"
        else
            warn "patch 04 NÃO aplicado"
            all_ok=false
        fi
        if is_05_applied; then
            ok "patch 05 aplicado (ACDC rect LLaMA)"
        else
            warn "patch 05 NÃO aplicado"
            all_ok=false
        fi
        $all_ok && exit 0 || exit 1
        ;;
    reverse)
        if is_05_applied; then
            git apply --reverse "$PATCH_05"
            ok "patch 05 revertido"
        else
            ok "patch 05 já estava ausente (nada a reverter)"
        fi
        if is_04_applied; then
            git apply --reverse "$PATCH_04"
            ok "patch 04 revertido"
        else
            ok "patch 04 já estava ausente (nada a reverter)"
        fi
        exit 0
        ;;
    apply)
        if is_04_applied; then
            ok "patch 04 já aplicado (idempotente)"
        else
            echo "aplicando patch 04 (L3 ACDC + L5 HRR + L4 K_i8 cache + Fase III ACDC rect)..."
            if ! git apply "$PATCH_04"; then
                err "patch 04 falhou — base incompatível com $CURRENT_HEAD (esperado: 1f86f05)"
                exit 1
            fi
            ok "patch 04 aplicado"
        fi
        if is_05_applied; then
            ok "patch 05 já aplicado (idempotente)"
        else
            echo "aplicando patch 05 (ACDC rect gate para build_llama)..."
            if ! git apply "$PATCH_05"; then
                err "patch 05 falhou — requer patch 04 aplicado primeiro"
                exit 1
            fi
            ok "patch 05 aplicado"
        fi
        ok "dispatch patches prontos (04 + 05)"
        exit 0
        ;;
esac
