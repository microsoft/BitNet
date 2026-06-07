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
#   Os commits de feature (L3 ACDC, L5 HRR, L4 K_i8 cache, Fase II/III ACDC
#   rect) foram desenvolvidos localmente e consolidados em um único patch
#   cumulativo (04-ACDC-rect-FFN.patch) que aplica tudo de uma vez.
#
#   Patches 01-03 existem como referência histórica mas não são mais usados
#   pelo script principal (04 os substitui).
#
# Uso:
#   ./scripts/apply-dispatch-patches.sh           # aplica
#   ./scripts/apply-dispatch-patches.sh --check   # só verifica
#   ./scripts/apply-dispatch-patches.sh --reverse # reverte
#
# Pré-requisitos:
#   - 3rdparty/llama.cpp/ existe e está checked-out na base 1f86f05
#   - patches/llama.cpp/04-ACDC-rect-FFN.patch existe
#
# Saída:
#   - Aplica patch cumulativo 04 (inclui L3+L5+L4cache+FaseIII)
#   - Idempotente: detecta se já aplicado e sai 0
#   - Falha com mensagem clara se patch não aplicar (sai 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SUBMODULE="$REPO_ROOT/3rdparty/llama.cpp"
PATCHES_DIR="$REPO_ROOT/patches/llama.cpp"

COMBINED_PATCH="$PATCHES_DIR/04-ACDC-rect-FFN.patch"

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
if [ ! -f "$COMBINED_PATCH" ]; then
    err "patch não encontrado: $COMBINED_PATCH"
    exit 1
fi

MODE="apply"
if [ "${1:-}" = "--check" ]; then MODE="check"; fi
if [ "${1:-}" = "--reverse" ]; then MODE="reverse"; fi

cd "$SUBMODULE"

CURRENT_HEAD=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo "submodule HEAD: $CURRENT_HEAD"

# Sentinela: o patch 04 adiciona llm_build_ffn_acdc_rect — string única
is_applied() {
    grep -qF 'llm_build_ffn_acdc_rect' src/llama.cpp
}

case "$MODE" in
    check)
        if is_applied; then
            ok "patch cumulativo 04 aplicado (L3+L5+L4cache+FaseIII)"
            exit 0
        else
            warn "patch 04 NÃO aplicado"
            exit 1
        fi
        ;;
    reverse)
        if is_applied; then
            git apply --reverse "$COMBINED_PATCH"
            ok "patch 04 revertido"
        else
            ok "patch 04 já estava ausente (nada a reverter)"
        fi
        exit 0
        ;;
    apply)
        if is_applied; then
            ok "patch 04 já aplicado (idempotente)"
        else
            echo "aplicando patch cumulativo 04 (L3 ACDC + L5 HRR + L4 K_i8 cache + Fase III ACDC rect)..."
            if ! git apply "$COMBINED_PATCH"; then
                err "patch 04 falhou — base incompatível com $CURRENT_HEAD (esperado: 1f86f05)"
                exit 1
            fi
            ok "patch 04 aplicado"
        fi
        ok "dispatch patches prontos"
        exit 0
        ;;
esac
