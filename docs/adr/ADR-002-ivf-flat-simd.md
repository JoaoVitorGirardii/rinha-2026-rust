# ADR-002: Algoritmo de Busca Vetorial — IVF-Flat com SIMD AVX2+F16C

**Data**: 2026-05-09  
**Status**: Aceito

## Contexto

O desafio exige buscar os 5 vizinhos mais próximos em 3 milhões de vetores 14D para cada request. A máquina de avaliação é um Mac Mini Late 2014 (Intel Core i5 Haswell, ~2.6 GHz). Cada instância da API recebe 0.45 vCPU.

### Análise de alternativas

**Brute force escalar**: 3M × 14 floats = 42M operações por query  
→ ~4 ms por query com 0.5 CPU → p99 ~4 ms → score_p99 = +1398 pontos  

**Brute force SIMD (AVX2)**: 3M × (168 MB de dados a ler)  
→ Limitado por bandwidth de DRAM: 168 MB / 12.8 GB/s ≈ 13 ms → muito lento  

**HNSW**: Excelente latência (~0.1 ms), mas overhead de grafo = 192 MB extra → memória total > 350 MB  

**IVF-Flat (K=1024, nprobe=16)**:
- Scan: 16 × (3M/1024) = ~47K vetores por query
- Dados lidos: 47K × 32 bytes (f16 padded) = 1.5 MB
- Bandwidth bound: 1.5 MB / 12.8 GB/s ≈ 120 µs → sub-millisecond ✓
- Recall@5 esperado com 14 dimensões: >97%  

## Decisão

**IVF-Flat** com:
- **K = 1024** clusters (treinados com k-means++ + 25 iterações Lloyd)
- **nprobe = 16** (padrão, configurável via env var `NPROBE`)
- **Distância euclidiana ao quadrado** (sem sqrt — monotônica para comparação)
- **TopK5 com array fixo** ao invés de `BinaryHeap` (k=5 é pequeno demais para heap)
- **Hot path SIMD**: `dist_sq_f16_avx2` com `#[target_feature(enable = "avx2,f16c")]`
  - Carrega 8 f16 → `_mm_loadu_si128` → `_mm256_cvtph_ps` (F16C conversion)
  - Dois batches para 16 dims (14 reais + 2 padding)
  - Horizontal sum com `_mm_movehl_ps` + `_mm_movehdup_ps`
- **Fallback escalar** com runtime detection via `is_x86_feature_detected!`

## Consequências

- Build-time da imagem Docker: ~2-4 min para k-means (aceito — custo único)
- Recall@5 ~97-99% significa <3% dos requests podem ter resultado diferente do brute-force
- failure_rate esperada < 3%, muito abaixo do cutoff de 15%
- p99 estimado: ~0.4-0.8 ms → score_p99 = 3000 (máximo)
- nprobe configurável permite ajuste sem rebuild da imagem
