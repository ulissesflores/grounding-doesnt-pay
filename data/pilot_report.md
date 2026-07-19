# Pilot Scores (C1 / C2 / C7-proxy)

> Pre-registration: `data/PREREGISTRATION.md`. Scored 2026-07-09T20:55:21.722997+00:00.

## TL;DR

| | |
|--|--|
| Embedding backend | `sentence-transformers` |
| Clustering | HDBSCAN(min_cluster_size=3) + Agglomerative(cosine, thr=0.35) |
| Ordering invariant across algos | False |
| Bootstrap resamples | 10,000 |
| Gate decision | **AMPUTATE** |

## Primary metrics

| Arm | Ideas | Input tok | Clusters (H/A) | clusters/idea (H/A) | clusters/1k-tok (H/A) | mean cos-dist | rho |
|---|---|---|---|---|---|---|---|
| C1 | 180 | 18,810 | 11/99 | 0.061/0.550 | 0.585/5.263 | 0.606 | n/a |
| C2 | 180 | 19,076 | 2/145 | 0.011/0.806 | 0.105/7.601 | 0.593 | -0.012 |
| C7-proxy | 180 | 127,470 | 9/161 | 0.050/0.894 | 0.071/1.263 | 0.681 | -0.014 |

H = HDBSCAN, A = Agglomerative. rho = mean pairwise cluster-coverage correlation between source identities (low = decorrelation working; n/a for C1's single distribution).

## Bootstrap contrasts (Agglomerative, 95% CI of the difference)

| Contrast | View | mean diff | CI low | CI high | C7 wins? |
|---|---|---|---|---|---|
| C7-proxy_vs_C1 | proposals_matched | 0.194 | 0.133 | 0.256 | yes |
| C7-proxy_vs_C1 | tokens_matched | -2.894 | -3.286 | -2.502 | no |
| C7-proxy_vs_C2 | proposals_matched | 0.045 | -0.017 | 0.111 | no |
| C7-proxy_vs_C2 | tokens_matched | -4.246 | -4.667 | -3.826 | no |

## Gate decision (pre-registered rule)

> **AMPUTATE** — C7-proxy does NOT beat C1 on tokens-matched (bootstrap CI of the difference lies entirely below 0 -> C7-proxy is strictly WORSE than C1). Structural coupling confirmed -> amputation validated: DO NOT build the Cupula; report the null.

- Decision view: `tokens_matched` | algorithm: `agglomerative`
- Ordering invariant across algorithms: `False`
- Validity requires ordering invariance across HDBSCAN and Agglomerative; if False, treat the reading as provisional.

Pre-registered branches:

- **AMPUTATE** — C7-proxy NAO supera C1 (tokens-casados, alem do IC bootstrap) -> structural coupling confirmado -> NAO construir a Cupula; reportar o null.
- **PROCEED** — C7-proxy > C1 (tokens-casados) E C7-proxy > C2 -> dominio bate estilo E escala -> prosseguir para ablacao estreita.
- **PERSONA_NOT_DOMAIN** — C7-proxy > C1 mas C7-proxy ~= C2 -> e persona/estilo, nao dominio -> tese domain-specific cai (publicar como negativo).
