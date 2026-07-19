# Pre-registration — C1-vs-C2-vs-C7 pilot (the GATE)

> **Note (public release).** This is the pre-registration frozen before the first
> record was collected (2026-07-08). It is reproduced here as evidence of method.
> For this public package, absolute local paths and private credential references
> were redacted; **no hypothesis, parameter, decision rule, number, or date was
> changed.** The document is in the author's working language (Portuguese); the
> paper itself is in English. Internal project codenames (e.g. the study's own
> id, the downstream "Cúpula") are kept as written to preserve the frozen record.

> Gate barato e decisivo ANTES de construir qualquer pipeline. Pré-registrado: fixar tudo aqui ANTES de coletar. Data: 2026-07-08.

## Pergunta decisiva

Sob budget casado, geração **condicionada-por-domínio** (specs reais aterrados) produz mais **diversidade semântica objetiva** (clusters/ideia) do que (a) self-consistency por temperatura e (b) style-personas sem substrato? Ou o **structural coupling** (agentes de um só modelo-base) colapsa a diversidade, como prevê a literatura (`2604.18005`, `2604.02460`, `2502.08788`)?

**Por que proxy, não a Cúpula completa:** a M150 do 004 (extração atômica -> clustering -> abelhas -> veto) NÃO existe; `Split/` é o v1. E a camada de abelhas comprovadamente COLAPSA clusters (anti-métrica). Então o piloto testa só o **estágio de geração** — o único lugar onde a diversidade pode nascer. Se ela não nasce aqui, nenhuma máquina downstream a cria.

## Braços (todos: mesmo modelo M, mesma temperatura T, mesmo N ideias, mesmas 2 queries)

| Braço | Condicionamento | O que isola |
|---|---|---|
| **C1** self-consistency | prompt criativo genérico, sem persona/domínio | baseline de escala pura (o que 2402.05120 enaltece) |
| **C2** style-persona | rótulo de estilo genérico (ex.: "o contrário", "o visionário", "o minimalista"), SEM substrato de conhecimento | o confound "domínio = persona-prompt" — **o contraste que o veredito diz ser O teste** |
| **C7-proxy** domínio | spec real completo (domínio, modo, persona) dos 11 domínios de `arcabouco/`, texto integral no contexto | o tratamento: substrato epistêmico aterrado |

`C7-proxy` cicla pelas combinações (**10 domínios usáveis** × 2 modos × 5 personas = 100; o harness descobre domínios dinamicamente — dom-16 só tem babelfish, não qualifica) amostrando N distintas. `C2` usa N rótulos de estilo sem conhecimento. `C1` repete o prompt genérico N vezes (só temperatura decorrelaciona).

## Casamento de budget (o ponto que quebra o 004)

`C7-proxy` paga input-tokens do spec que `C1` não paga. **Não dá pra casar propostas E tokens ao mesmo tempo.** Reportar AMBAS as visões:
- **Propostas-casadas:** mesmo N (=90) por braço. DV = clusters/ideia.
- **Tokens-casados (a DECISIVA):** DV = clusters por 1k input-tokens; OU dar a C1/C2 amostras extras até igualar os input-tokens totais de C7-proxy e recontar. Logar input+output tokens por chamada, sempre.

## Emenda 2026-07-08 — modelo M: Cerebras Qwen -> Grok-4.20 (via Hermes)

> [!IMPORTANT]
> **Emenda pré-coleta** (nenhum record real de geração existia — o smoke de geração nunca cruzou a parede de auth; só o smoke de scoring rodou em mock). Registrada ANTES do primeiro record, portanto legítima e não é p-hacking.
>
> - **O quê:** M trocado de Cerebras `qwen-3-235b` para OpenRouter `x-ai/grok-4.20` com **reasoning DESLIGADO** (non-reasoning, rápido).
> - **Por quê:** (1) a key Cerebras pré-registrada dá 401 "Wrong API Key" e a key OpenRouter do Split também está morta (401 "User not found") — M original inalcançável; (2) a decisão-travada já fixava o run no Grok-4.20 registrado no estado do projeto — a emenda realinha o piloto a esse lock; (3) grok-4.20 non-reasoning é mais rápido/barato e evita CoT contaminar a medida de diversidade.
> - **Validade interna preservada:** os TRÊS braços (C1/C2/C7-proxy) seguem com o MESMO M, temperatura, max_tokens, N e texto de query; só o condicionamento de system-prompt difere. Generalidade externa passa a ser Grok-específica (era Qwen-específica).
> - **Acesso:** M é alcançado por OpenRouter com a chave OpenRouter lida do ambiente. Verificada 2026-07-08 (retornou `x-ai/grok-4.20-20260309`, reasoning vazio).
> - **Inalterado:** seed, temperatura, N, braços, queries, prompts, scoring, gate_rule. Custo cai para ~$0.30 no run cheio.

## Geração (fixar — emendado 2026-07-08)

- Modelo M: **OpenRouter `x-ai/grok-4.20`, reasoning DESLIGADO** (`reasoning:{"enabled":false}`) — wired em the project router (mapeamento `grok-4.20` + passthrough de `reasoning`). Registrar a versão exata retornada pela API (o harness loga `model_version_returned` = `data["model"]`, ex.: `x-ai/grok-4.20-20260309`). *(Original pré-emenda: Cerebras `qwen-3-235b`.)*
- Temperatura T: **0.9** (fixa em todos os braços). Seed determinístico onde o provider permitir; senão, fixar N e ordem.
- N = **90 ideias/braço**; 3 ideias por chamada -> 30 chamadas/braço/query. Total = 3 braços × 2 queries × 30 = **180 chamadas**.
- Queries: `AUT "brick"` (comparável a LLM Discussion / 2405.06373) + 1 design funcional (ex.: "compartimento para 500ml de água mineral", do próprio 004). Fixar o texto exato.
- Regras: geração INDEPENDENTE (sem histórico compartilhado, sem debate, sem memória). Persistir tudo em JSONL com SHA-256 de prompt+output.

## Scoring (fixar — roda local, sem LLM)

1. Extrair ideias atômicas (1 proposição por linha da resposta); dedup exato.
2. Embeddar com modelo geral fixo e congelado (**domain-blind**: `sentence-transformers/all-MiniLM-L6-v2` ou `all-mpnet-base-v2` — registrar qual; opcional: remover nomes-próprios/jargão de domínio antes de embeddar, pra não medir a semeadura).
3. Clusterizar com **DOIS** algoritmos pré-registrados: HDBSCAN (`min_cluster_size=3`) **e** agglomerative (distância cosseno, threshold=0.35). Resultado válido = **ordenação invariante** entre os dois.
4. **DV primária:** clusters distintos / nº ideias válidas (clusters/ideia), por braço, nas duas visões de budget.
5. **Secundárias:** contagem crua de clusters; distância cosseno média par-a-par (índice de diversidade tipo Vendi); fluency (nº ideias válidas).
6. **rho (checagem de descorrelação):** por braço, correlação de cobertura-de-cluster entre identidades-fonte (agentes-domínio cobrem clusters DIFERENTES?). C7-proxy com rho baixo = descorrelação funcionando; C1 = baseline (distribuição única).
7. IC por **bootstrap** (10k reamostragens) sobre a diferença entre braços.

## Predições pré-registradas (falsificáveis)

- **H_piloto:** clusters/ideia: `C7-proxy > C2 > C1`, e C7-proxy mantém vantagem na visão **tokens-casados**.
- **Null:** sem diferença ordenada; OU `C7-proxy ≤ C1` sob tokens-casados (structural coupling domina).

## Regra de decisão (o GATE — pré-registrada)

- **C7-proxy NÃO supera C1 (tokens-casados, além do IC bootstrap)** -> structural coupling confirmado -> **amputação validada empiricamente. NÃO construir a Cúpula. Parar e reportar o null (é resultado válido).**
- **C7-proxy > C1 (tokens-casados) E C7-proxy > C2** -> domínio bate estilo E escala -> mecânica tem pernas -> prosseguir para a **ablação estreita** (curva k, âncora humana, pré-registro OSF).
- **C7-proxy > C1 mas C7-proxy ≈ C2** -> é persona/estilo, não domínio -> a tese domain-specific CAI (resultado válido; publicar como negativo).

## Determinismo / artefatos

- SEED=42; T e N fixos; versão do modelo logada; SHA-256 de todos inputs/outputs; JSONL incremental (resiliente a crash).
- Harness: `code/` neste pacote (`run_pilot_c1c7.py` geração; `score_pilot.py` scoring; `pilot_config.json`).
- Saída durável: `code/ + data/` (JSONL de ideias + `pilot_scores.json` + `pilot_report.md`).

## Custo estimado

180 chamadas OpenRouter `x-ai/grok-4.20` non-reasoning ($1.25/$2.50 por Mtok) ≈ **$0.30** (extrapolado do smoke: 6 chamadas = $0.0090). Embedding + clustering local (grátis). Wall-clock: ~minutos.

## Estado

- [x] **Harness construído** (2026-07-08) em the pilot harness: `pilot_config.json`, `pilot_common.py`, `run_pilot_c1c7.py`, `score_pilot.py`. Compila.
- [x] **Smoke de scoring PASSOU** (mock, sem custo): embedding real MiniLM baixou; produziu tabela C1/C2/C7-proxy + rho + gate. (O gate "AMPUTATE" no mock é RUÍDO — ideias mock arbitrárias; só prova que o pipeline computa a decisão, não é resultado.)
- [x] **BLOCKER RESOLVIDO** (emenda 2026-07-08): Cerebras key morta (401) + OpenRouter-Split morta (401) -> M emendado para OpenRouter `x-ai/grok-4.20` non-reasoning via the OpenRouter key from the environment. Ver "Emenda 2026-07-08".
- [x] **Smoke de GERAÇÃO PASSOU** (2026-07-08, 6 chamadas reais, $0.0090): `key source=env`; 6/6 OK via `x-ai/grok-4.20`; non-reasoning confirmado (output 100-113 tok, sem CoT); versão datada logada (`x-ai/grok-4.20-20260309`); C7-proxy carregou spec real (in≈2200-2500 tok vs C1/C2≈320 — confirma a assimetria de budget que o gate testa); `record_sha256` presente. Saída: `results/pilot_gen_smoke.jsonl`.
- [x] **Autor autorizou o run completo** (2026-07-08).
- [x] **Run completo ✅** (180/180 records, 30 por braço×query, 180 SHA-256 únicos, versão `x-ai/grok-4.20-20260309`, custo ≈ $0.28). Resiliência a stall provada (1 `ReadTimeout` do Grok → resume + retry bounded adicionado ao harness → completou).
- [x] **Scoring ✅** e **decisão do gate registrada** (abaixo).

## Veredito do gate (2026-07-08) — AMPUTATE (amputação validada empiricamente)

Artefatos: `code/ + data/{pilot_gen_full.jsonl, pilot_scores.json, pilot_report.md}`.

| Braço | Ideias | Input tok | clusters/ideia (A) | **clusters/1k-tok (A) — decisiva** | rho |
|---|---|---|---|---|---|
| C1 self-consistency | 180 | 18.810 | 0.550 | **5.263** | n/a |
| C2 style-persona | 180 | 19.076 | 0.806 | **7.601** | -0.012 |
| C7-proxy domínio | 180 | 127.470 | 0.894 | **1.263** | -0.014 |

Bootstrap 95% CI da diferença (agglomerative):
- **C7 vs C1, tokens-casados: -2.894, CI [-3.286, -2.502]** — inteiramente < 0 (C7 perde **decisivamente**; o `pilot_report.md` descreve o CI como "crosses 0", o que está **errado** — não cruza).
- C7 vs C2, tokens-casados: -4.246, CI [-4.667, -3.826] — C7 perde.
- C7 vs C1, propostas-casadas: +0.194, CI [0.133, 0.256] — C7 vence (mas essa NÃO é a visão decisiva).
- C7 vs C2, propostas-casadas: +0.045, CI [-0.017, 0.111] — **cruza 0 = C7 ≈ C2** (padrão `PERSONA_NOT_DOMAIN`).

**Regra pré-registrada aplicada:** visão decisiva = tokens-casados. C7-proxy NÃO supera C1 (além do IC) → **AMPUTATE**. **NÃO construir a Cúpula. Reportar o null (resultado válido).**

**Robustez / honestidade (obrigatório):**
- `ordering_invariant=False` — HDBSCAN e agglomerative divergem MUITO na contagem bruta (ex.: C2 = 2 vs 145). O pré-registro manda tratar como **métrica frágil, reportada como tal, NÃO ajustada** (proibido tunar o threshold pós-hoc).
- **MAS o veredito sobrevive à fragilidade:** na visão tokens-casados, C7-proxy é **ÚLTIMO sob AMBOS** os algoritmos (H: C1>C2>C7; A: C2>C1>C7). Motivo aritmético: C7-proxy precisaria de ~6,8× os clusters de C1 (gap de tokens 127k vs 19k) e tem no máximo 1,6× (A) ou menos (H). Nenhuma escolha de clustering flipa isso.
- **Mecanismo refinado (corrige a narrativa do escrutínio):** rho de C7-proxy (-0,014) ≈ rho de C2 (-0,012) ≈ 0 → a descorrelação do domínio **funciona** (não é "coupling que colapsa diversidade"). O domínio falha porque a descorrelação que ele compra **não é token-eficiente** vs escalar um prompt genérico barato. Nuance provisória (rho monta sobre o mesmo clustering frágil).
- Escopo: proxy do estágio de **geração** apenas (a Cúpula M150 não existe). Suficiente para o gate: se a diversidade não nasce token-eficiente na geração, nenhuma máquina downstream a cria — e as abelhas já colapsam clusters (escrutínio).

## Watch-item metodológico (do smoke)

Com MiniLM e N minúsculo (mock), o threshold agglomerative 0.35 pré-registrado deixou cada ideia no próprio cluster e o `ordering_invariant` (HDBSCAN vs Agglomerative) deu **False**. Em N=90/braço real espera-se mais quase-duplicatas a fundir. **Fidelidade pré-registro:** NÃO tunar o threshold pós-hoc — se `ordering_invariant=False` persistir em N real, isso é resultado informativo (métrica frágil), reportado como tal, não ajustado.
