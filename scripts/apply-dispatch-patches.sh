#!/usr/bin/env bash
#
# apply-dispatch-patches.sh
#
# Aplica os patches de dispatch do BitNet CPU-Universal sobre o
# 3rdparty/llama.cpp após `git submodule update --init`.
#
# Contexto:
#   O submodule 3rdparty/llama.cpp aponta para um fork upstream
#   (https://github.com/Eddie-Wang1120/llama.cpp.git) cuja branch
#   merge-dev foi reescrita (force-push) após nossos patches de
#   L3 ACDC (commit 707f316) e L5 HRR cleanup (commit 3dfc2df)
#   terem sido mergeados. Como resultado, esses commits são
#   inacessíveis para clones novos, quebrando o build.
#
#   Para tornar o build reproduzível, esta abordagem vendoriza os
#   dois patches em patches/llama.cpp/ e os aplica após o submodule
#   init. Os patches são idempotentes (verificam se já estão aplicados
#   via `git apply --reverse --check`).
#
# Uso:
#   ./scripts/apply-dispatch-patches.sh           # aplica
#   ./scripts/apply-dispatch-patches.sh --check   # só verifica
#   ./scripts/apply-dispatch-patches.sh --reverse # reverte
#
# Pré-requisitos:
#   - 3rdparty/llama.cpp/ existe e está checked-out
#   - patches/llama.cpp/*.patch existem
#
# Saída:
#   - Aplica patches em ordem (L3 primeiro, L5 depois — L5 depende
#     do guard #if adicionado pelo L3)
#   - Idempotente: detecta se já aplicado e sai 0
#   - Falha com mensagem clara se patch não aplicar (sai 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SUBMODULE="$REPO_ROOT/3rdparty/llama.cpp"
PATCHES_DIR="$REPO_ROOT/patches/llama.cpp"

L3_PATCH="$PATCHES_DIR/01-L3-ACDC-FFN-dispatch.patch"
L5_PATCH="$PATCHES_DIR/02-L5-HRR-cleanup-dispatch.patch"

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
if [ ! -f "$L3_PATCH" ] || [ ! -f "$L5_PATCH" ]; then
    err "patches não encontrados em $PATCHES_DIR"
    exit 1
fi

MODE="apply"
if [ "${1:-}" = "--check" ]; then MODE="check"; fi
if [ "${1:-}" = "--reverse" ]; then MODE="reverse"; fi

cd "$SUBMODULE"

# Verifica estado atual
CURRENT_HEAD=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo "submodule HEAD: $CURRENT_HEAD"

is_applied() {
    # Detecção de "já aplicado" via sentinela: procura a string
    # característica que o patch adiciona. Se presente, patch já aplicado.
    # Argumento: tag identificadora (L3 ou L5).
    case "$1" in
        L3)
            # L3 adiciona "#  include \"ggml-bitnet-dispatch.h\""
            grep -qF '#  include "ggml-bitnet-dispatch.h"' src/llama.cpp
            ;;
        L5)
            # L5 muda o guard para incluir BITNET_L5_HRR
            grep -qF 'BITNET_L4_TROPICAL) || defined(BITNET_L3_ACDC) || defined(BITNET_L5_HRR)' src/llama.cpp
            ;;
        *)
            return 1
            ;;
    esac
}

case "$MODE" in
    check)
        if is_applied L3 && is_applied L5; then
            ok "ambos patches aplicados (L3 + L5)"
            exit 0
        else
            warn "patches não totalmente aplicados"
            is_applied L3 && ok "L3 aplicado" || warn "L3 NÃO aplicado"
            is_applied L5 && ok "L5 aplicado" || warn "L5 NÃO aplicado"
            exit 1
        fi
        ;;
    reverse)
        if is_applied L5; then
            git apply --reverse "$L5_PATCH"
            ok "L5 revertido"
        fi
        if is_applied L3; then
            git apply --reverse "$L3_PATCH"
            ok "L3 revertido"
        fi
        ok "patches revertidos"
        exit 0
        ;;
    apply)
        # L3 primeiro (L5 depende do guard que L3 adiciona)
        if is_applied L3; then
            ok "L3 já aplicado (idempotente)"
        else
            echo "aplicando L3 ACDC FFN dispatch..."
            if ! git apply "$L3_PATCH"; then
                err "L3 patch falhou — contexto incompatível com $CURRENT_HEAD"
                exit 1
            fi
            ok "L3 aplicado"
        fi
        if is_applied L5; then
            ok "L5 já aplicado (idempotente)"
        else
            echo "aplicando L5 HRR cleanup dispatch..."
            if ! git apply "$L5_PATCH"; then
                err "L5 patch falhou — verifique que L3 foi aplicado primeiro"
                exit 1
            fi
            ok "L5 aplicado"
        fi
        ok "dispatch patches prontos"
        exit 0
        ;;
esac
