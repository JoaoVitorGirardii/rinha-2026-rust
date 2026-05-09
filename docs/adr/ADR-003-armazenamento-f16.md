# ADR-003: Armazenamento de Vetores em f16 (Half-Precision Float)

**Data**: 2026-05-09  
**Status**: Aceito

## Contexto

3 milhões de vetores 14D precisam residir em memória para acesso rápido. O budget total de memória é 350 MB para todos os serviços, com 2 instâncias de API independentes (cada uma carrega o índice separadamente).

### Análise de tamanhos

| Formato | Bytes/vetor | Total (3M vetores) | 2 instâncias |
|---|---|---|---|
| f64 (double) | 14 × 8 = 112 bytes | 336 MB | **672 MB** — inviável |
| f32 (float) | 14 × 4 = 56 bytes | 168 MB | **336 MB** — beira o limite |
| f32 padded 16 | 16 × 4 = 64 bytes | 192 MB | **384 MB** — excede |
| f16 (half) | 14 × 2 = 28 bytes | 84 MB | **168 MB** ✓ |
| f16 padded 16 | 16 × 2 = 32 bytes | 96 MB | **192 MB** ✓ |

## Decisão

Armazenar vetores como **f16 (IEEE 754 half-precision) com padding para 16 valores** (STORAGE_DIMS=16):

- Dimensões 0–13: valores reais convertidos de f32 para f16 no build
- Dimensões 14–15: `f16(0.0)` — padding para alinhamento SIMD
- Total por vetor: **32 bytes = exatamente 1/2 cache line** (64 bytes)
- Computação: conversão f16→f32 via instrução F16C (`_mm256_cvtph_ps`) antes de aritmética

### Erro de arredondamento f16

- Precisão f16: ~3 casas decimais significativas (mantissa 10 bits)
- Erro máximo absoluto no intervalo [-1, 1]: < 0.001
- Impacto em distância euclidiana quadrada (14 dims): < 0.014
- Impacto na decisão fraud_score < 0.6: insignificante em casos não-borderline
- Sentinelas -1 (dims 5/6 quando `last_transaction` é null): **representáveis exatamente em f16**

## Consequências

- Footprint de memória por instância: **~100 MB** (96 MB vetores + 3 MB labels + overhead)
- Margem no budget: 350 MB − 2 × 100 MB − 16 MB nginx = **134 MB de folga**
- Bandwidth de memória reduzida à metade vs f32 → hot path mais rápido
- Build do índice gera arquivo de ~94 MB (vs ~170 MB em f32)
- Cálculo dos centroides do IVF mantido em f32 (são apenas 1024 × 14 = 57 KB)
