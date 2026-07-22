# ADR-007: Adicionar weights_only=True ao torch.load (segurança)

**Status:** Aceito  
**Data:** 2026-03-09 (commit `eb60fc3`, PR #421)  
**Confiança:** 🟢 CONFIRMADO

---

## Contexto

`torch.load()` sem `weights_only=True` usa o módulo `pickle` do Python para deserialização, que permite execução arbitrária de código. Um arquivo `.pt` malicioso poderia executar código no sistema do usuário no momento do carregamento.

A pipeline GPU (`gpu/generate.py`, `gpu/convert_checkpoint.py`) carregava checkpoints sem esta proteção desde a introdução do branch GPU (maio 2025 a março 2026 — ~10 meses de exposição).

```
# Antes (vulnerável):
torch.load(fp16_ckpt_path, map_location="cpu")

# Depois (seguro):
torch.load(fp16_ckpt_path, map_location="cpu", weights_only=True)
```

Os scripts em `utils/` já usavam `weights_only=True` corretamente (servindo como referência para o fix).

## Motivação

CVE/CWE-502 (Deserialization of Untrusted Data). O fix foi identificado e proposto por colaborador externo via PR.

## Consequências

**Positivas:**
- Elimina vetor de ataque de execução de código via checkpoint malicioso
- Alinha pipeline GPU com práticas já seguidas em utils/

**Negativas:**
- `weights_only=True` falha com checkpoints que contêm objetos Python além de tensores
- Se algum checkpoint existente contiver objetos Python customizados, o carregamento falhará após o fix
- Não foi possível retroativamente revogar exposição dos usuários que carregaram checkpoints entre mai/2025 e mar/2026
