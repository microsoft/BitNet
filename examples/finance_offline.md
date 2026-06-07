# Finance — Categorização de Despesas em Workstation Bancária Restrita (Offline)

> **Persona D4 — Setor Financeiro (compliance BCB/GLBA).** Walkthrough
> canônico: analista financeiro categoriza despesas em workstation
> bancária **sem internet**, com BitNet-2B rodando 100% local.
>
> **Versão:** v0.1 — gerado por T023 (Fase 3: Núcleo) em 2026-06-06.
> **Ancoragem:** `requirements.md#9` (persona D4), AC-11/AC-12
> (`requirements.md#6`), `docs/decision-matrix.md` (T015).

---

## Cenário

**Quem:** Ana, analista financeiro em banco de médio porte.
**Onde:** Workstation bancária restrita (i5-8350U, 16 GB RAM,
**sem acesso à internet** por política de segurança — firewall
bloqueia tudo exceto lista branca de domínios internos).
**O quê:** Carregar extrato CSV mensal (~500 transações) e pedir
ao BitNet-2B para **categorizar** cada transação em uma das 12
categorias (Alimentação, Transporte, Moradia, Saúde, Educação,
Lazer, Vestuário, Serviços, Impostos, Investimentos, Receitas,
Outros) e **identificar padrões suspeitos** (gastos recorrentes
anômalos, duplicidades, valores fora do padrão).
**Restrição:** Compliance BCB (Resolução 4.658) e GLBA — dados
financeiros não podem ser processados em serviços externos.

---

## Por que BitNet CPU-Universal atende

| Requisito compliance | Como BitNet atende |
|---------------------|--------------------|
| Dados não saem do dispositivo | Inferência 100% local; sem cloud (NO-07), sem telemetria (NO-06) |
| Sem custo de cloud privada | Free, open-source, sem assinatura |
| Auditável | Modelo determinístico (mesma seed → mesmo output); logs locais |
| Verificável | `tests/test_air_gapped_boot.sh` (T010) valida binário sem rede |
| Cabe em workstation padrão | i5-8350U, 16 GB é baseline D4 (`requirements.md#9`) |
| Footprint de RAM previsível | BitNet-2B + KV cache = ~4-5 GB; 16 GB disponível |

---

## Setup (1 vez, online — em máquina de desenvolvimento)

```bash
# 1. Instalar conda env (em máquina online)
conda create -n bitnet-cpp python=3.10 -y
conda activate bitnet-cpp
pip install -r requirements.txt

# 2. Clonar fork
git clone https://github.com/peder1981/BitNet.git
cd BitNet
git submodule update --init --recursive

# 3. Build
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

# 6. (Opcional) empacotar para transferência offline
tar czf bitnet-offline.tar.gz BitNet/ models/
# Mover via USB / share interno para a workstation restrita
```

---

## Uso diário (offline, na workstation restrita)

### Passo 1: confirmar que workstation está sem rede

```bash
# Tentar ping/saída HTTP — esperado: falha
ping -c 1 google.com  # esperado: 100% packet loss
curl https://google.com  # esperado: falha de DNS ou timeout
```

### Passo 2: preparar extrato CSV

```bash
# Exemplo: extrato_jan2024.csv com colunas: data, descrição, valor
head -3 extrato_jan2024.csv
# 2024-01-02,IFOOD *RESTAURANTE X,-45.90
# 2024-01-03,UBER *VIAGEM Y,-23.50
# 2024-01-05,SALARIO EMPRESA Z,8500.00
```

### Passo 3: categorizar em lote

```bash
conda activate bitnet-cpp
cd BitNet

# Dividir extrato em chunks de ~30 transações (contexto L1 ~ 4K tokens)
split -l 30 extrato_jan2024.csv chunk_

for chunk in chunk_*; do
  python run_inference.py \
    -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    -p "Categorize cada transação do extrato abaixo em uma das 12
categorias: Alimentação, Transporte, Moradia, Saúde, Educação, Lazer,
Vestuário, Serviços, Impostos, Investimentos, Receitas, Outros.

Extrato:
$(cat $chunk)

Formato de saída: data | descrição | valor | categoria
Para cada transação, marque (suspeita:sim/não) se o valor está fora
do padrão histórico ou se há duplicidade.

Output:
" \
    -n 200 -t 4 > "${chunk}.categorizado"
done

# Concatenar
cat chunk_*.categorizado > extrato_jan2024_categorizado.txt
```

**Tempo esperado:** ~40-60 segundos por chunk (30 transações) em
i5-8350U. Para 500 transações: ~15-20 min total.

### Passo 4: revisar e gerar relatório

```bash
# Agregar categorias (script Python local, sem rede)
python3 <<'EOF'
import re
from collections import Counter

with open("extrato_jan2024_categorizado.txt") as f:
    text = f.read()

# Parsear linhas "data | desc | valor | categoria"
categorias = Counter()
suspeitas = []
for line in text.split("\n"):
    m = re.match(r"(\S+)\s*\|\s*(.+?)\s*\|\s*(-?[\d.]+)\s*\|\s*(\w+)", line)
    if m:
        data, desc, valor, cat = m.groups()
        categorias[cat] += 1
        if "sim" in line.lower() and "suspeita" in line.lower():
            suspeitas.append((data, desc, valor, cat))

print("=== Resumo por categoria ===")
for cat, count in categorias.most_common():
    print(f"  {cat}: {count}")

print(f"\n=== Suspeitas ({len(suspeitas)}) ===")
for s in suspeitas:
    print(f"  {s}")
EOF
```

---

## Validação air-gapped (AC-11)

```bash
bash tests/test_air_gapped_boot.sh models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf

# Inspeção manual:
unshare -rn python run_inference.py \
  -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p "Teste" -n 10 -t 4
```

---

## Auditoria (compliance BCB/GLBA)

Documente para sua auditoria interna:

| Item | Evidência |
|------|-----------|
| Binário roda sem rede | `tests/test_air_gapped_boot.sh` passa |
| Sem telemetria | `grep -rn "telemetry\|upload_data" src/ utils/ run_inference*.py` → 0 hits (T031) |
| Sem cloud | `grep -rn "http://\|https://" src/ 3rdparty/` → 0 hits (T032) |
| Modelo determinístico | `tests/test_*_properties.cpp` (T005-T007) — mesma seed = mesmo output |
| Footprint de RAM | ~4.5 GB em 16 GB disponíveis |
| Logs locais (não na nuvem) | Output em `~/extratos/`, com timestamp |
| Workstation sem rede | `ping -c 1 google.com` → 100 % packet loss |

Modelo de texto para auditoria:

```
Eu, [nome], matrícula [nº], atesto que o software BitNet
CPU-Universal v[versão] foi instalado em [workstation] e validado
em modo air-gapped em [data]. Nenhuma conexão de rede foi estabelecida
durante o processamento do extrato [período]. Nenhum dado financeiro
saiu do dispositivo. O output foi revisado por [analista sênior] em [data].
Assinatura: ___   Data: ___   Matrícula: ___
```

---

## Limitações conhecidas (sendo honesto)

1. **BitNet-2B pode inventar categorias.** Revise **sempre** o output.
   Categoria errada em compliance é um risco regulatório.
2. **Detecção de "suspeita" é heurística, não auditoria forense.**
   BitNet pode marcar transações legítimas como suspeitas (falso
   positivo) ou deixar passar fraude real (falso negativo). Use como
   **triagem inicial**, não como detecção final.
3. **BitNet-2B é pequeno (2B).** Para padrões muito sutis
   (lavagem de dinheiro estruturada, smurfing), use software
  专门izado (ex: ACL, SAS, OFAC screening).
4. **Língua:** primariamente inglês. Para descrições em português,
   valide a qualidade com extratos antigos antes de usar em produção.
5. **Sem integração com ERP/sistema bancário.** Você precisa
   copiar/colar manualmente. Integração SAP/Oracle/etc. está fora
   de escopo (NO-04).

---

## Quando **NÃO** usar BitNet-2B

- **Detecção de fraude crítica** (lavagem, financiamento ao
  terrorismo) — use software专门izado com regras atualizadas.
- **Compliance OFAC / sanções internacionais** — use listas
  atualizadas diariamente (BitNet não tem dados de sanções).
- **Auditoria final** — BitNet é triagem; auditoria humana é
  obrigatória.

---

## Próximos passos (sugestões)

1. **Validar em extratos antigos:** rode em 3-5 meses de extrato
   que você já categorizou manualmente. Compare.
2. **Criar catálogo de descrições ambíguas:** tenha um dicionário
   interno de "IFOOD = Alimentação", "UBER = Transporte", etc.
   Use como ground truth para revisar o output.
3. **Definir threshold de suspeita:** o que conta como "suspeita"
   para o seu contexto? Valor > R$ 1000? Recorrência > 3x/mês?
4. **Upgrade futuro:** quando o fork ganhar fine-tuning ACDC
   (reserva técnica Q4 2029, `ROADMAP.md#2.1`), pode ser possível
   fine-tunar em extratos categorizados manualmente do seu
   próprio histórico (anonimizando PII).

---

## Referências

- **Persona D4:** `requirements.md#9`
- **Decision matrix:** `docs/decision-matrix.md` (T015) linha 1 (BitNet-2B denso) e linha 2 (sparse opt-in)
- **Hardware-compatibility:** `docs/hardware-compatibility.md` (T016) linha "ThinkPad T480"
- **Air-gapped test:** `tests/test_air_gapped_boot.sh` (T010)
- **ROADMAP público:** `ROADMAP.md` (T014)
- **Sumário dos 5 níveis:** `docs/theory/06-5-levels.md` (T036)

---

*v0.1 — gerado por T023 em 2026-06-06T22:45:00Z*
*Walkthrough persona D4 setor financeiro: setup 1× online, uso diário
offline em workstation restrita, categorização em lote, auditoria
BCB/GLBA, limitações honestas (heurística ≠ auditoria forense).*
