#!/usr/bin/env bash
#
# apply-dispatch-patches.sh
#
# Aplica o patch de dispatch do BitNet CPU-Universal sobre o
# 3rdparty/llama.cpp após `git submodule update --init`.
#
# Contexto:
#   O submodule 3rdparty/llama.cpp aponta para o fork upstream
#   (https://github.com/Eddie-Wang1120/llama.cpp.git, base commit 1f86f05,
#    src/llama.cpp blob 666fcc4).
#
#   Um único patch cumulativo é usado:
#
#   05-ACDC-rect-LLaMA.patch  — patch combinado:
#     • Dispatch includes (L3 ACDC + L5 HRR + L4 K_i8 cache)
#     • llm_build_ffn_acdc_rect  (model-agnostic rectangular ACDC FFN)
#     • llm_build_ffn_acdc_bitnet (BitNet-2B hardcoded dims, legacy)
#     • llm_build_kqv tropical + HRR attention gates
#     • build_falcon ACDC rect gate  (Falcon3-3B/10B: n_ff/n_embd = 3-7.5×)
#     • build_llama  ACDC rect gate  (LLaMA-arch: Falcon3 reports arch=llama)
#
#   04-ACDC-rect-FFN.patch existem como referência histórica (subset do 05).
#   Patches 01-03 existem como referência histórica mas não são usados no CI.
#
#   NOTA TÉCNICA (por que não 04+05 em sequência):
#     Ambos foram criados da mesma base (blob 666fcc4).  Aplicados em sequência,
#     o patch 05 falha no hunk @@ -28 porque o 04 já adicionou as linhas de
#     include que o 05 também tenta adicionar.  O 05 é superset do 04 e deve
#     ser aplicado sozinho a partir da base limpa.
#
# Uso:
#   ./scripts/apply-dispatch-patches.sh           # aplica
#   ./scripts/apply-dispatch-patches.sh --check   # só verifica
#   ./scripts/apply-dispatch-patches.sh --reverse # reverte
#
# Pré-requisitos:
#   - 3rdparty/llama.cpp/ existe e está checked-out na base 1f86f05
#   - patches/llama.cpp/05-ACDC-rect-LLaMA.patch existe
#
# Saída:
#   - Aplica patch 05 (combinado)
#   - Idempotente: detecta se já aplicado e sai 0
#   - Falha com mensagem clara se patch não aplicar (sai 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SUBMODULE="$REPO_ROOT/3rdparty/llama.cpp"
PATCHES_DIR="$REPO_ROOT/patches/llama.cpp"

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

# Sentinela — llm_build_ffn_acdc_rect: adicionado pelo patch combinado (05)
is_applied() {
    grep -qF 'llm_build_ffn_acdc_rect' src/llama.cpp && \
    grep -qF 'bitnet_acdc_ffn_rect_llama' src/llama.cpp
}

case "$MODE" in
    check)
        if is_applied; then
            ok "patch combinado aplicado (L3+L5+L4cache+FaseIII rect+LLaMA gate)"
            exit 0
        else
            warn "patch combinado NÃO aplicado"
            exit 1
        fi
        ;;
    reverse)
        if is_applied; then
            git apply --reverse "$PATCH_05"
            ok "patch 05 revertido"
        else
            ok "patch já estava ausente (nada a reverter)"
        fi
        exit 0
        ;;
    apply)
        if is_applied; then
            ok "patch combinado já aplicado (idempotente)"
        else
            echo "aplicando patch combinado (L3 ACDC + L5 HRR + L4 K_i8 cache + FaseIII rect + LLaMA gate)..."
            if ! git apply "$PATCH_05"; then
                err "patch 05 falhou — base incompatível com $CURRENT_HEAD (esperado blob 666fcc4)"
                err "rode 'git checkout src/llama.cpp' no submodule antes de tentar novamente"
                exit 1
            fi
            ok "patch combinado aplicado"
        fi
        ok "dispatch patch pronto"
        exit 0
        ;;
esac
