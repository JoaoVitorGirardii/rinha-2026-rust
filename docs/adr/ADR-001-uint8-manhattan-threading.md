# ADR-001: Otimização de Latência — IVF-KNN com Budget CFS

**Data:** 2026-05-09  
**Status:** Aceito  
**Contexto:** Rinha de Backend 2026 — redução de p99 de 99.61 ms → < 2 ms

---

## Contexto

O sistema recebe requests de avaliação de fraude e retorna `fraud_score` via KNN (K=5)
sobre 3 milhões de vetores de referência usando um índice IVF-Flat (K=1024 clusters).

O juiz avalia:
- **Score de latência (p99_score):** ≈ `433 × ln(1000/p99_ms)` — logarítmico, penaliza todo
  aumento de p99 mas com retornos decrescentes
- **Score de detecção:** baseado em E = FP + 3×FN (pesos: FP=1, FN=3)
- **Score final = p99_score + detection_score** (máximo teórico: 6000)

**Ambiente de produção:**
- Mac Mini Late 2014: Intel Core i5-4278U (Haswell, 2 cores, 2.6 GHz, 3 MB L3)
- 2 instâncias Docker com `cpus: "0.45"` e `memory: "167MB"` cada
- HAProxy como load balancer (leastconn)
- **Limite CFS do Docker:** período 100ms, quota 45ms (= 0.45 CPU × 100ms)
- Score baseline: p99=99.61ms, FP=1, FN=2, **score=3730**

---

## Diagnóstico das Causas Raiz

### Causa 1: tokio `current_thread` com KNN síncrono (→ p99=99ms)

Com `current_thread`, o handler `fraud_score_handler` bloqueava o único async worker
por ~450µs por request durante o KNN. A 450 req/s, requests eram processadas em série
→ filas se formavam → p99 = 99ms.

**Fix:** `multi_thread(worker_threads=2) + spawn_blocking` libera os workers async
durante o KNN.

### Causa 2: HAProxy com `http-server-close` (→ p99=595ms após migração)

A config inicial do HAProxy usava `http-server-close`, que fecha a conexão backend após
cada request. A 450 req/s isso esgotava as portas efêmeras (~28K) em ~60s.

**Fix:** `option http-keep-alive` mantém conexões TCP persistentes.

### Causa 3: nprobe alto saturando o quota CFS (→ p99=24-58ms)

**Esta é a causa principal que permanecia após os dois fixes acima.**

Com nprobe=16, cada request usa:
- Scan de 16 clusters × 2929 vetores médios = 46.8K vetores × ~6-10 ciclos/vetor
- Centroide search: 1024 × 14 dims
- **CPU real por request: ~1ms** (medido via curl; cache misses dominam o benchmark)

A 450 req/s: `450 × 1ms = 450ms CPU/s` = **100% do quota CFS** de 450ms/s.

A 100% de utilização, qualquer burst acima de 450 req/s instantâneos faz o container
exceder seu quota de 45ms na janela de 100ms → **throttling CFS** → instâncias pausadas
por ~50ms → p99 = 24-58ms.

**Fix:** Reduzir nprobe para que `nprobe × avg_cluster × CPU/vetor × 450 req/s << 45ms/100ms`.

---

## Experimentos Realizados (em ordem cronológica)

### Exp-1: uint8 + Manhattan SSE2 (fracasso)
**Hipótese:** uint8+SAD é ~6× mais rápido → CPU por request ~80µs → p99 < 1ms

**Resultado:** p99=1.80ms ✓, mas **FP=285, FN=283** ✗

**Post-mortem:** Clusters IVF treinados com L2. Mudar para Manhattan no scan altera o
ordering dos vizinhos, causando 568 misclassificações nos edge cases.

### Exp-2: Abordagem two-phase (scan uint8+Manhattan → re-rank f16+L2)
**Hipótese:** Scan rápido para candidatos, re-rank exato para precisão

**Resultado (TopK30, max_blocking_threads=4):** p99=23.98ms, FP=2, FN=5, score=4243  
**Resultado (TopK50, max_blocking_threads=4):** p99=56-58ms, FP=2, FN=3, score=3927

**Problema:** A abordagem two-phase não atacava a causa raiz (CPU por request ~1ms com
nprobe=16). O p99 de 23-58ms continuava preso no regime de throttling CFS.

### Exp-3: Random Forest (10 árvores → 50 árvores, 3M amostras)
**Hipótese:** Inferência de árvore (~1µs por request) elimina o gargalo de CPU

**Resultado:** threshold=0.05, E_validação=5246 em 300K amostras → E_teste estimado ≈ 945
→ **detection_score < 0** — inviável.

**Post-mortem:** RF com 14 features aprendidas não consegue reproduzir o KNN que usa
vetores de similaridade direta. Threshold forçado ao mínimo (0.05) sugere fraca
discriminação entre classes. O KNN é fundamentalmente mais preciso para este problema.

### Exp-4: Redução de nprobe (solução) ✓

| nprobe | p99 | FP | FN | E | score |
|--------|-----|----|----|---|-------|
| 16 | 58ms | 2 | 3 | 11 | 3928 |
| 8 | 13.98ms | 3 | 3 | 12 | 4520 |
| **4** | **1.91ms** | **6** | **12** | **42** | **5228** |

**nprobe=4 é o ponto ótimo** dado o orçamento CFS de 0.45 CPU:
- CPU por request ≈ 0.4ms → utilização = 450 × 0.4ms / 450ms = 40% → sem throttling
- E=42 é maior que E=11 com nprobe=16, mas o ganho no p99_score (+1483) supera a
  perda no detection_score (-168)

---

## Configuração Final (ACEITA)

| Parâmetro | Valor | Justificativa |
|---|---|---|
| `NPROBE` | 4 | Sweet spot p99 vs recall com 0.45 CPU/instância |
| `worker_threads` | 2 | Permite overlap de I/O e CPU |
| `max_blocking_threads` | 4 | Suporta bursts sem starvar requests |
| Scan de cluster | uint8 + Manhattan SSE2 + AVX2 (phase 1) | Rápido; candidatos para re-ranking |
| Re-rank | f16 + L2 AVX2 + F16C (phase 2) | Precisão exata para os top-50 candidatos |
| Candidatos (TopK) | 50 | Buffer maior reduz FP/FN em edge cases |
| Load balancer | HAProxy `http-keep-alive` | Elimina TCP handshake por request |

---

## Resultados Finais

| Métrica | Baseline | **Final** | Meta |
|---|---|---|---|
| p99 | 99.61ms | **1.91ms** | < 2ms |
| FP | 1 | 6 | 0 |
| FN | 2 | 12 | 0 |
| E (FP+3×FN) | 7 | 42 | 0 |
| p99_score | 1001 | **2718** | 3000 |
| detection_score | 2729 | **2510** | 3000 |
| **score final** | **3730** | **5228** | 6000 |

**Melhoria de +40% no score final** (3730 → 5228).

---

## Lições Aprendidas

1. **O budget CFS é o constraint real.** Com `cpus: "0.45"`, o orçamento é 45ms de CPU por
   100ms de wall clock. Qualquer request que ultrapasse 1ms de CPU × 450 req/s = 450ms/s
   vai saturar o quota e causar p99 alto por throttling periódico.

2. **SIMD não é silver bullet.** AVX2+SAD reduz ciclos de ~25 para ~4 por vetor. Mas a
   latência total inclui cache misses, spawn_blocking overhead, e scheduling. O ganho real
   é menor que o teórico.

3. **Menor nprobe < melhor recall.** O trade-off entre recall e latência existe, mas o
   scoring do juiz favorece latência baixa: `p99_score ≈ 433×ln(1/p99)` é logarítmico, então
   ir de 58ms→2ms dá +1483 pts enquanto perder recall de E=11→42 custa apenas -168 pts.

4. **Random Forest não substitui KNN para este problema.** Os 14 features capturem similaridade
   de transações, e o KNN explora isso diretamente. Um RF tenta aprender fronteiras de decisão
   que não existem de forma simples nesse espaço.

---

## Próximas Oportunidades (não implementadas)

### HNSW (impacto alto, complexidade alta)
- Grafo de navegabilidade: query time O(log N) ≈ 22 hops × 16 neighbors = 352 ops ≈ 1-5µs
- Habilitaria nprobe equivalente alto com KNN tempo muito menor
- Requer reescrita do preprocess + ~500MB de grafo em memória (excede 167MB)

### Reconstruir índice com K=512 clusters
- Clusters 2× maiores → nprobe=4 escaneia 2× mais vetores (melhor recall)
- Mesmo CPU per request → sem throttling
- Requer rebuild do preprocessor (~8 min)

### Two-phase com early exit por confiança
- Se fraud_score após top-30 já for 0.0 ou 1.0: pular re-rank
- Reduz ~20% do trabalho na fase 2 (já rápida, ~3µs)
