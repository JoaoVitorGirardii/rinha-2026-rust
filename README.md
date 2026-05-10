# Rinha de Backend 2026 — Rust

Solução em Rust para a [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026):
detecção de fraude em transações via busca KNN sobre 3 milhões de vetores de referência.

**Score obtido: 5228 / 6000**  
`p99 = 1.91ms | FP = 6 | FN = 12`

---

## Arquitetura

```
Client
  │
  ▼
HAProxy :9999
  │  roundrobin, http-keep-alive
  ├──► api1 :3000 (0.45 CPU, 167 MB)
  └──► api2 :3000 (0.45 CPU, 167 MB)
```

Cada instância carrega o índice IVF-Flat em memória (~94 MB) e responde requisições
de avaliação de fraude.

---

## Como funciona

### 1. Representação dos dados

Cada transação é convertida em um **vetor de 14 dimensões** por `vectorize.rs`:

| dim | feature                            | normalização                      |
| --- | ---------------------------------- | --------------------------------- |
| 0   | `amount`                           | `/ 10000.0`, clip [0,1]           |
| 1   | `installments`                     | `/ 12.0`, clip [0,1]              |
| 2   | `amount / customer.avg_amount`     | `/ 10.0`, clip [0,1]              |
| 3   | hora do dia (UTC)                  | `/ 23.0`                          |
| 4   | dia da semana (Mon=0)              | `/ 6.0`                           |
| 5   | minutos desde última transação     | `/ 1440.0` ou -1 se ausente       |
| 6   | km da última transação até a atual | `/ 1000.0` ou -1 se ausente       |
| 7   | `terminal.km_from_home`            | `/ 1000.0`                        |
| 8   | `customer.tx_count_24h`            | `/ 20.0`, clip [0,1]              |
| 9   | `terminal.is_online`               | 0 ou 1                            |
| 10  | `terminal.card_present`            | 0 ou 1                            |
| 11  | comerciante desconhecido           | 0 (conhecido) ou 1 (desconhecido) |
| 12  | risco do MCC                       | valor pré-definido por categoria  |
| 13  | `merchant.avg_amount`              | `/ 10000.0`, clip [0,1]           |

Campos ausentes (`last_transaction`) usam sentinela `-1.0`.

---

### 2. Índice IVF-Flat

Construído em tempo de build por `crates/preprocess`:

- **K-means++**: inicialização dos K=1024 centroides com amostragem ponderada por distância
- **25 iterações Lloyd**: convergência dos centroides
- **Organização por cluster**: vetores reorganizados na memória para acesso sequencial
- **Validação de recall**: 1000 amostras aleatórias verificam recall ≥ 99% antes de escrever o índice

Formato binário (`ivf_index.bin`, ~94 MB):

```
[header 64B] [centroids: K×14×f32] [offsets: K×u32] [sizes: K×u32]
             [labels: N×u8] [vectors: N×16×f16]
```

Os vetores são padded de 14 para 16 dimensões para alinhamento SIMD.

---

### 3. Busca KNN — Two-phase

#### Phase 1: scan rápido uint8 + Manhattan SSE2/AVX2

Ao carregar o índice, os vetores f16 são quantizados para uint8:

```
q_u8 = round((v + 1.0) × 127.5)  →  [-1,1] → [0,255]
```

A query também é quantizada. O scan usa `_mm_sad_epu8` (SSE2) — soma de diferenças
absolutas de 16 bytes em **1 instrução**:

```
SAD = Σ|query_u8[i] - ref_u8[i]|  para i in 0..16
```

Com AVX2 disponível, 2 vetores são processados por instrução `_mm256_sad_epu8`
(32 bytes = 2× vetores de 16 bytes por ciclo).

Resulta em um heap **TopK50** com os 50 candidatos de menor distância Manhattan.

#### Phase 2: re-ranking exato f16 + L2 AVX2+F16C

Os 50 candidatos são re-rankeados com a distância euclidiana exata sobre os vetores f16
usando instruções AVX2 + F16C:

```rust
_mm256_cvtph_ps  →  converte 8 f16 para f32 em 1 instrução
_mm256_sub_ps    →  subtrai
_mm256_mul_ps    →  multiplica
```

Um heap **TopK5** mantém os 5 vizinhos mais próximos.

#### fraud_score

```
fraud_score = contagem de vizinhos com label "fraude" / 5
```

`approved = fraud_score < 0.6` (≥ 3 de 5 vizinhos fraudulentos → bloqueia).

---

### 4. Seleção de clusters (centroide search)

Antes do scan, os `nprobe` clusters mais próximos da query são selecionados por distância
L2 sobre os centroides f32 (1024 × 14 dims, ~22µs).

**`NPROBE=4`** é o valor configurado: com 0.45 CPU por instância, é o sweet spot entre
recall e latência.

---

## Por que NPROBE=4 é o sweet spot

O Docker limita cada instância a **0.45 CPU** (quota CFS: 45ms de CPU por 100ms de
wall-clock). Com nprobe=16, cada request usa ~1ms de CPU real (scan de clusters + cache
misses de L3). A 450 req/s de pico:

```
450 req/s × 1ms/req = 450ms/s = 100% do quota CFS → throttling → p99 = 58ms
```

Com nprobe=4 (~0.4ms de CPU por request):

```
450 req/s × 0.4ms/req = 180ms/s = 40% do quota → sem throttling → p99 = 1.91ms
```

O trade-off: nprobe=4 tem recall um pouco menor (FP=6, FN=12) que nprobe=16 (FP=2,
FN=3), mas o scoring da competição favorece latência baixa (fórmula logarítmica).

---

## Modelo de threading

```rust
tokio::runtime::Builder::new_multi_thread()
    .worker_threads(2)        // parse JSON + I/O de rede em paralelo
    .max_blocking_threads(4)  // até 4 KNNs simultâneos no blocking pool
    .build()?
```

O KNN é executado em `spawn_blocking` para não bloquear o executor async:

```rust
let fraud_score = tokio::task::spawn_blocking(move || {
    state.index.search(&query, state.nprobe)
}).await.unwrap();
```

---

## Build

O `Dockerfile` usa build multi-stage:

1. **builder**: compila `api` e `preprocess` com `target-cpu=haswell` (AVX2+F16C+FMA)
2. **preprocessor**: executa `preprocess` sobre os 3M vetores de referência
   - k-means++ → 25 iterações → validação de recall → grava `ivf_index.bin`
   - Esta etapa é **cacheada** pelo Docker; só re-executa se `resources/` mudar
3. **runtime**: imagem mínima com `api` + `ivf_index.bin` (~94MB)

```bash
docker compose up -d
curl -s http://localhost:9999/ready   # → 200 OK
```

---

## Resultados por configuração

| nprobe | p99        | FP    | FN     | E      | score    |
| ------ | ---------- | ----- | ------ | ------ | -------- |
| 16     | 58ms       | 2     | 3      | 11     | 3928     |
| 8      | 14ms       | 3     | 3      | 12     | 4520     |
| **4**  | **1.91ms** | **6** | **12** | **42** | **5228** |

---

## Abordagens tentadas e descartadas

**uint8 + Manhattan puro (sem re-ranking):** p99=1.8ms mas FP=285, FN=283. O índice IVF
foi treinado com distância L2; mudar para Manhattan no scan altera o ordering dos vizinhos
nas fronteiras de decisão dos 797 edge cases.

**Random Forest (50 árvores, 3M amostras):** threshold=0.05 (limiar muito baixo, modelo
sem discriminação), E estimado ≈ 945 no conjunto de teste. O KNN é fundamentalmente mais
preciso para problemas de similaridade.

**HNSW:** O(log N) ≈ 5µs por query vs ~400µs do IVF. Requer ~500MB de memória para
3M vetores — excede o limite de 167MB por instância.

---

## Estrutura do repositório

```
crates/
  api/          API HTTP (axum) + busca KNN
  preprocess/   Geração do índice IVF-Flat (k-means + export binário)
docs/adr/       Decisões de arquitetura
haproxy/        Configuração HAProxy
resources/      references.json.gz (dados de referência)
```
