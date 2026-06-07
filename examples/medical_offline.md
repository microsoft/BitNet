# Medical — Análise de Prontuário em Laptop de Consultório (Offline)

> **Persona D4 — Setor Saúde (LGPD/HIPAA).** Walkthrough canônico: médico
> analisa prontuário em laptop de consultório, **sem internet**, com
> BitNet-2B rodando 100% local.
>
> **Versão:** v0.1 — gerado por T021 (Fase 3: Núcleo) em 2026-06-06.
> **Ancoragem:** `requirements.md#9` (persona D4), AC-11/AC-12
> (`requirements.md#6`), `docs/decision-matrix.md` (T015).

---

## Cenário

**Quem:** Dra. Maria, clínica de família em São Paulo.
**Onde:** Consultório com laptop Lenovo T480 (i5-8350U, 16 GB RAM, **sem
Wi-Fi** durante o atendimento para compliance com LGPD).
**O quê:** Carregar prontuário de paciente João (texto, ~3 páginas) e
pedir ao BitNet-2B para gerar um **resumo estruturado** com tópicos
"Queixa principal / Antecedentes / Medicações em uso / Plano".
**Restrição:** Nenhum byte do prontuário pode sair do laptop. Nenhuma
telemetria. Nenhuma chamada externa.

---

## Por que BitNet CPU-Universal atende

| Requisito LGPD/HIPAA | Como BitNet atende |
|----------------------|--------------------|
| Dados não saem do dispositivo | Inferência 100% local; sem CUDA, sem cloud, sem telemetria (NO-06, NO-07) |
| Sem GPU dedicada (laptop padrão) | CPU-only, baseline L1 em ~5 tok/s em i5-8350U (T016) |
| Auditável | Modelo determinístico (mesma seed → mesmo output) |
| Verificável | `tests/test_air_gapped_boot.sh` (T010) valida binário sem rede |
| Footprint previsível | BitNet-2B + KV cache 4-bit = ~4-5 GB RAM; laptop com 8 GB é viável |

---

## Setup (1 vez, online)

```bash
# 1. Instalar conda env (uma vez, com internet)
conda create -n bitnet-cpp python=3.10 -y
conda activate bitnet-cpp
pip install -r requirements.txt

# 2. Clonar fork (uma vez, com internet)
git clone https://github.com/peder1981/BitNet.git
cd BitNet
git submodule update --init --recursive

# 3. Build (com internet, baixa LLVM/clang se necessário)
conda install -c conda-forge llvmdev=18 -y
cmake -B build -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# 4. Baixar modelo (uma vez, com internet; ~1.1 GB)
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf \
  --local-dir models/BitNet-b1.58-2B-4T
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

# 5. Validar air-gapped (com internet)
bash tests/test_air_gapped_boot.sh models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
# esperado: "AC-11 air-gapped boot: PASS"
```

**Total de tempo:** ~15 min em rede normal. Após este setup, **o laptop
está pronto para uso offline permanente**.

---

## Uso diário (offline)

### Passo 1: desativar rede (LGPD best practice)

```bash
# No Linux:
sudo nmcli networking off
# ou fisicamente: desligar Wi-Fi (botão ou airplane mode)
```

### Passo 2: ativar conda env e rodar inferência

```bash
conda activate bitnet-cpp
cd BitNet

# Inferência com prompt estruturado (substitua $PRONTUARIO pelo conteúdo)
python run_inference.py \
  -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p "Prontuário do paciente João Silva, 54 anos:

$PRONTUARIO

Tarefa: gere um resumo estruturado com 4 seções:
1. Queixa principal
2. Antecedentes relevantes
3. Medicações em uso
4. Plano sugerido

Resumo:" \
  -n 200 -t 4
```

**Tempo esperado:** ~40 segundos para 200 tokens em i5-8350U (RNF-02, ±2 %).
**Memória:** ~4.5 GB (modelo + KV cache + inferência).

### Passo 3 (opcional): ativar sparse opt-in para velocidade

```bash
# Sparse float top-K=32: ~50% mais rápido (RNF-02 ~+44%),
# com risco de pequena degradação de qualidade.
# Teste em prontuários antigos antes de usar em produção.
BITNET_SPARSE_TOPK=32 python run_inference.py \
  -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p "$PROMPT" -n 200 -t 4
```

### Passo 4: salvar e revisar

```bash
# Salvar output em arquivo local (não na nuvem!)
python run_inference.py ... > ~/prontuarios/joao_$(date +%Y%m%d).resumo.txt

# Revisar manualmente antes de anexar ao prontuário eletrônico.
# Lembrete: BitNet-2B é uma ferramenta de apoio, **não substitui
# revisão médica**. A decisão clínica é sempre do profissional.
```

---

## Validação air-gapped (AC-11)

Para confirmar que **nenhuma syscall de rede** é feita:

```bash
# Test canônico do fork:
bash tests/test_air_gapped_boot.sh models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf

# Inspeção manual (se quiser verificar você mesmo):
unshare -rn python run_inference.py \
  -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p "Hello" -n 10 -t 4
# esperado: roda normal, exit 0, sem erro de DNS/network
```

---

## Auditoria (compliance)

Para sua auditoria interna LGPD/HIPAA, documente:

| Item | Evidência |
|------|-----------|
| Binário roda sem rede | `tests/test_air_gapped_boot.sh` passa |
| Sem telemetria | `grep -rn "telemetry\|upload_data" src/ utils/ run_inference*.py` → 0 hits (T031) |
| Sem cloud | `grep -rn "http://\|https://" src/ 3rdparty/` → 0 hits (T032) |
| Modelo determinístico | `tests/test_*_properties.cpp` (T005-T007) — mesma seed = mesmo output |
| Footprint de RAM | ~4.5 GB; documentar capacidade do laptop |

Modelo de texto para auditoria:

```
Eu, [nome], atesto que o software BitNet CPU-Universal v[versão]
foi instalado em [laptop] e validado em modo air-gapped em [data].
Nenhuma conexão de rede foi estabelecida durante [período].
Nenhum dado de paciente saiu do dispositivo.
Assinatura: ___   Data: ___
```

---

## Limitações conhecidas (sendo honesto)

1. **BitNet-2B é um modelo pequeno (2B params).** Não vai dar diagnóstico
   médico. Use como **ferramenta de apoio** (resumir, organizar), não
   como substituto de avaliação clínica.
2. **Resumos podem ter alucinações.** Revise sempre. Especialmente
   medicações e dosagens — BitNet pode inventar nomes de drogas
   plausíveis mas inexistentes.
3. **Não conecta a sistemas de prontuário eletrônico (PEP).** Você
   precisa copiar/colar manualmente. Integração PEP está fora de escopo
   (NO-04, dependência externa).
4. **Língua:** BitNet-2B é primariamente em inglês. Para português, a
   qualidade cai. Se o seu consultório atende em PT-BR, valide a
   qualidade do output antes de usar em produção.

---

## Próximos passos (sugestões para você)

1. **Validar em prontuários antigos:** rode o resumo em 5-10 prontuários
   que você já tem revisados. Compare com sua estrutura habitual.
2. **Cronograma de revisão:** revise sempre o output. BitNet é apoio,
   não substituto.
3. **Reportar bugs:** se encontrar alucinações sistemáticas, abra issue
   no GitHub com o trecho (anonimizado!).
4. **Upgrade futuro:** quando o fork ganhar fine-tuning ACDC (reserva
   técnica Q4 2029, `ROADMAP.md#2.1`), pode ser possível fine-tunar
   o modelo em prontuários anonimizados do seu próprio consultório.
   Até lá, use como está.

---

## Referências

- **Persona D4:** `requirements.md#9`
- **Decision matrix:** `docs/decision-matrix.md` (T015) linha 1 (BitNet-2B denso) e linha 2 (sparse opt-in)
- **Hardware-compatibility:** `docs/hardware-compatibility.md` (T016) linha "ThinkPad T480"
- **Air-gapped test:** `tests/test_air_gapped_boot.sh` (T010)
- **ROADMAP público:** `ROADMAP.md` (T014)
- **Sumário dos 5 níveis:** `docs/theory/06-5-levels.md` (T036)

---

*v0.1 — gerado por T021 em 2026-06-06T22:15:00Z*
*Walkthrough persona D4 setor saúde: setup 1× online, uso diário offline,
validação air-gapped, auditoria LGPD, limitações honestas.*
