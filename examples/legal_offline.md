# Legal — Resumo de Petição Inicial em Escritório de Advocacia (Offline)

> **Persona D4 — Setor Jurídico (sigilo profissional).** Walkthrough
> canônico: advogado resume petição inicial em escritório pequeno,
> **sem internet**, com BitNet-2B rodando 100% local.
>
> **Versão:** v0.1 — gerado por T022 (Fase 3: Núcleo) em 2026-06-06.
> **Ancoragem:** `requirements.md#9` (persona D4), AC-11/AC-12
> (`requirements.md#6`), `docs/decision-matrix.md` (T015).

---

## Cenário

**Quem:** Dr. Carlos, advogado autônomo em Belo Horizonte.
**Onde:** Escritório com Dell Latitude 5490 (i5-8250U, 8 GB RAM).
**O quê:** Carregar petição inicial de um caso de direito do consumidor
(~15 páginas) e pedir ao BitNet-2B para gerar um **resumo executivo**
com 5 seções: "Partes / Fatos / Fundamentos jurídicos / Pedidos /
Valor da causa".
**Restrição:** Sigilo profissional (Estatuto da OAB, art. 25:
"é direito do advogado a inviolabilidade de seu escritório"). Nenhum
byte da petição pode sair do laptop.

---

## Por que BitNet CPU-Universal atende

| Requisito OAB / sigilo | Como BitNet atende |
|------------------------|--------------------|
| Sigilo do escritório | Inferência 100% local; sem cloud (NO-07), sem telemetria (NO-06) |
| Sem custo de cloud (escritório pequeno) | Free, open-source, sem assinatura |
| Auditável | Modelo determinístico (mesma seed → mesmo output) |
| Verificável | `tests/test_air_gapped_boot.sh` (T010) valida binário sem rede |
| Cabe em hardware legado | Latitude 5490 (i5-8250U, 8 GB) é baseline D4 (`requirements.md#9`) |

---

## Setup (1 vez, online)

```bash
# 1. Instalar conda env
conda create -n bitnet-cpp python=3.10 -y
conda activate bitnet-cpp
pip install -r requirements.txt

# 2. Clonar fork
git clone https://github.com/peder1981/BitNet.git
cd BitNet
git submodule update --init --recursive

# 3. Build (com Clang 18; ajuste para GCC se necessário)
conda install -c conda-forge llvmdev=18 -y
cmake -B build -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# 4. Baixar modelo
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf \
  --local-dir models/BitNet-b1.58-2B-4T
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

# 5. Validar air-gapped
bash tests/test_air_gapped_boot.sh models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
# esperado: "AC-11 air-gapped boot: PASS"
```

**Total de tempo:** ~15 min em rede normal. Após este setup, o laptop
está pronto para uso offline permanente.

---

## Uso diário (offline)

### Passo 1: desativar rede (sigilo best practice)

```bash
# No Linux:
sudo nmcli networking off
# ou fisicamente: desligar Wi-Fi (airplane mode)
```

### Passo 2: preparar texto da petição

```bash
# Converter PDF da petição para texto (se necessário)
# Recomendado: pdftotext (poppler-utils) — não usa rede
pdftotext -layout peticao_inicial.pdf peticao_inicial.txt

# Verificar que está OK
wc -l peticao_inicial.txt
```

### Passo 3: rodar inferência

```bash
conda activate bitnet-cpp
cd BitNet

PROMPT="$(cat <<'EOF'
Petição inicial do processo 0012345-67.2024.8.13.0024:

$(cat peticao_inicial.txt)

Tarefa: gere um resumo executivo com 5 seções:
1. Partes (polo ativo e polo passivo)
2. Fatos (síntese cronológica)
3. Fundamentos jurídicos (artigos de lei e teses)
4. Pedidos (lista enumerada)
5. Valor da causa

Resumo executivo:
EOF
)"

python run_inference.py \
  -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p "$PROMPT" \
  -n 200 -t 4
```

**Tempo esperado:** ~40-50 segundos para 200 tokens em i5-8250U.
**Memória:** ~4.5 GB (modelo + KV cache).

### Passo 4: salvar e revisar

```bash
python run_inference.py ... > ~/peticoes/0012345_resumo.txt

# **REVISÃO OBRIGATÓRIA** antes de usar.
# BitNet-2B é ferramenta de apoio, não substitui leitura técnica.
# Verificar especialmente:
#   - número do processo
#   - nomes das partes
#   - artigos de lei citados (BitNet pode inventar artigos)
#   - valor da causa
```

---

## Validação air-gapped (AC-11)

```bash
bash tests/test_air_gapped_boot.sh models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
# esperado: "AC-11 air-gapped boot: PASS"

# Inspeção manual:
unshare -rn python run_inference.py \
  -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p "Teste" -n 10 -t 4
```

---

## Auditoria (compliance OAB)

Documente para sua auditoria interna / processo ético:

| Item | Evidência |
|------|-----------|
| Binário roda sem rede | `tests/test_air_gapped_boot.sh` passa |
| Sem telemetria | `grep -rn "telemetry\|upload_data" src/ utils/ run_inference*.py` → 0 hits (T031) |
| Sem cloud | `grep -rn "http://\|https://" src/ 3rdparty/` → 0 hits (T032) |
| Modelo determinístico | `tests/test_*_properties.cpp` (T005-T007) — mesma seed = mesmo output |
| Footprint de RAM | ~4.5 GB em 8 GB disponíveis |

Modelo de texto para auditoria:

```
Eu, Dr(a). [nome], OAB [UF] [número], atesto que o software
BitNet CPU-Universal v[versão] foi instalado em [laptop] e validado
em modo air-gapped em [data]. Nenhuma conexão de rede foi estabelecida
durante [período]. Nenhum dado de cliente saiu do dispositivo.
Assinatura: ___   Data: ___   OAB: ___
```

---

## Limitações conhecidas (sendo honesto)

1. **BitNet-2B pode inventar artigos de lei.** Risco **ALTO** — a
   alucinação mais perigosa para uso jurídico. Revise **sempre** o
   output. Verifique cada artigo no diário oficial.
2. **BitNet-2B é pequeno (2B).** Para petições muito técnicas
   (tributário, previdencial complexo), a qualidade cai. Use como
   **primeira passada** de resumo, não como versão final.
3. **Língua:** primariamente inglês. Para português jurídico,
   valide a qualidade com casos antigos antes de usar em produção.
4. **Não substitui leitura técnica da petição.** O resumo serve
   para você **decidir se vale a pena ler a petição inteira**, não
   para usá-lo direto na peça.
5. **Sem integração com PJe (processo judicial eletrônico).** Você
   precisa copiar/colar manualmente. Integração PJe está fora de
   escopo (NO-04, dependência externa).

---

## Quando **NÃO** usar BitNet-2B

- Petições com **dados sensíveis de crianças/adolescentes** (Estatuto
  da Criança) — risco de LGPD é alto; use servidor dedicado ou
  redação manual.
- Casos com **segredo de justiça** — mesmo com air-gapped, o laptop
  pode ser apreendido. Use máquina isolada ou workstation dedicada.
- Casos com **valor estratégico muito alto** — não confie em
  resumo automático; leia integralmente.

---

## Próximos passos (sugestões)

1. **Validar em petições antigas:** rode o resumo em 5-10 petições
   que você já tem revisadas. Compare com sua estrutura habitual.
2. **Criar template de revisão:** tenha um checklist próprio do
   escritório (partes, artigos, pedidos, valor da causa) para
   revisar cada resumo.
3. **Treinar estagiários:** use o BitNet-2B para ensinar estagiários
   a **identificar seções** de uma petição. Eles revisam o output.
4. **Upgrade futuro:** quando o fork ganhar fine-tuning ACDC
   (reserva técnica Q4 2029, `ROADMAP.md#2.1`), pode ser possível
   fine-tunar em petições anonimizadas do seu próprio escritório.

---

## Referências

- **Persona D4:** `requirements.md#9`
- **Decision matrix:** `docs/decision-matrix.md` (T015) linha 1 (BitNet-2B denso) e linha 2 (sparse opt-in)
- **Hardware-compatibility:** `docs/hardware-compatibility.md` (T016) linha "Dell Latitude 5490"
- **Air-gapped test:** `tests/test_air_gapped_boot.sh` (T010)
- **ROADMAP público:** `ROADMAP.md` (T014)
- **Sumário dos 5 níveis:** `docs/theory/06-5-levels.md` (T036)

---

*v0.1 — gerado por T022 em 2026-06-06T22:30:00Z*
*Walkthrough persona D4 setor jurídico: setup 1× online, uso diário
offline, validação air-gapped, auditoria OAB, limitações honestas
(inventar artigos é o risco mais alto).*
