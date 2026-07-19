# Robustness re-scores (F5.5, LOCK-safe) — 2026-07-17

> Re-análise do dado CONGELADO (`data/pilot_gen_full.jsonl`, 180 records) com o pipeline
> pré-registrado (`code/score_pilot.py::score`), trocando UM knob por vez. NENHUMA API/geração nova —
> só re-embed + re-cluster. Driver: `scripts/rescore_robustness.py`. Outputs: `data/robustness/rescore_*.json`.
>
> **Validação:** a variante `baseline` (config verbatim) reproduz `pilot_scores.json` **exatamente**
> (todos os clusters agg+hdb idênticos; todos os contrastes bootstrap batem) → harness fiel.

## Variantes
- **baseline** — all-MiniLM-L6-v2 (encoder EN), config original.
- **multilingual** [C4] — paraphrase-multilingual-MiniLM-L12-v2 (PT-aware) sobre as ideias PT-BR.
- **minstop** [C5] — stopword mínimo: tira só scaffolding (dominio/modo/persona/style) + rótulos
  persona/estilo; **mantém** vocabulário de domínio (arquitetura/acustica/ventilacao/urbanismo…).
- **tfidf** (incidental) — embedding lexical puro (1ª rodada caiu nele por timeout de download).

## Tabela comparativa (agglomerative salvo indicado)

| Contraste | baseline (EN) | multilingual (PT) | minstop (EN) | tfidf | Robusto? |
|---|---|---|---|---|---|
| clusters agg C1/C2/C7 | 99/145/161 | 66/109/141 | 99/146/161 | 180/180/180 | — |
| clusters hdb C1/C2/C7 | 11/2/9 | 7/2/2 | 11/2/9 | 9/2/2 | — |
| **C7 último tokens-casados (agg)** | **sim** | **sim** | **sim** | **sim** | ✅ |
| **C7 último tokens-casados (hdb)** | **sim** | **sim** | **sim** | **sim** | ✅ |
| C7 vs C1 tokens agg | −2,894 [−3,29,−2,50] | −1,858 [−2,17,−1,54] | −2,894 | −5,163 | ✅ perde (CI<0) |
| C7 vs C1 tokens hdb | −0,511 [<0] | −0,353 [<0] | −0,510 [<0] | — | ✅ perde |
| C7 vs C1 **proposals** agg | +0,194 [>0] | +0,260 [>0] | +0,194 [>0] | −0,0 [≈0] | ✅ C7 vence (EN/PT) |
| C7 vs C2 **proposals** agg | +0,045 (**≈0**) | **+0,104 [>0]** | +0,044 (≈0) | −0,0 | ✗ **encoder-dep** |
| C7 vs C2 proposals hdb | +0,038 [>0] | 0,000 (≈0) | +0,039 [>0] | — | ✗ frágil |
| cos-dist C1/C2/C7 | 0,606/0,593/**0,681** | 0,593/**0,665**/0,660 | 0,606/0,594/0,681 | — | ✗ C7-maior encoder-dep |
| ρ C2 / C7 | −0,012 / −0,014 | +0,004 / −0,012 | −0,011 / −0,014 | — | ~0 (estável) |
| ordering_invariant | False | False | False | True | — |

## Conclusões honestas

### 1. A ESPINHA sobrevive a tudo (mais robusta que antes) ✅
"Grounding nunca é a fronteira token-eficiente; perde de C1 per-token" vale sob **2 algoritmos × 3 encoders
semânticos + TF-IDF**. É garantido por aritmética: tokens são congelados, clusters ≤ 180, então
C7 ≤ 180/127,47 = 1,41 < C1 (mín observado 3,51). Re-embed não pode salvar C7.

### 2. C4 (encoder) NÃO é só cosmético — muda um SECUNDÁRIO ⚠️ (achado material)
O baseline usa encoder **inglês** em texto **português**. Sob o encoder PT-apropriado:
- **C7 BATE C2 per-proposta** (+0,104, CI>0) — no baseline empatavam (+0,045, cruza 0).
  → A leitura **"grounding = persona, não domínio"** é um **artefato do encoder EN**. Sob o encoder certo,
    grounding compra SIM mais diversidade per-proposta que a persona barata.
- **C7 deixa de ter a maior cos-dist** (C2 0,665 > C7 0,660) — o "C7 lidera no sinal contínuo" também é
  encoder-dependente.
- C7 vence C1 per-proposta em AMBOS encoders (+0,194 EN, +0,260 PT) — grounding **ajuda**, robustamente.

### 3. C5 (stopword) — REFUTADO como confound ✅
Manter o vocabulário de domínio deixou C7 **idêntico** (161 clusters agg; C2 145→146). O stripping
domain-blind **não** enviesou contra C7. Objeção do reviewer respondida.

## Implicação de framing (autoral — requer operador)
O resultado "não-tautológico" que o reviewer hostil queria elevar (C7≈C2 per-proposta / "persona não
domínio") **evaporou** sob o encoder correto. O que resta, robusto: **grounding ajuda per-proposta mas
nunca paga por token**. Título "Grounding Doesn't Pay" fica MAIS certo (grounding não é inútil — só não
paga). A sensibilidade-ao-encoder vira finding de robustez + concessão honesta.
