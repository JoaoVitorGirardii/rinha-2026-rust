# ADR-004: Pipeline de Build com Pré-processamento no Docker

**Data**: 2026-05-09  
**Status**: Aceito

## Contexto

Os dados de referência (`references.json.gz`) são estáticos durante o teste. Construir o índice IVF-Flat (k-means sobre 3M vetores) a cada inicialização do container seria proibitivo em tempo. O desafio permite pré-processar os dados no build da imagem Docker.

## Decisão

**Dockerfile multi-stage** com 3 estágios:

### Stage 1: `builder` (rust:1.82-slim)
- Compila ambos os binários: `api` e `preprocess`
- Usa RUSTFLAGS para target-cpu=haswell e features AVX2+F16C+FMA
- Cache de dependências via layer intermediário (Cargo.toml + stubs)

### Stage 2: `preprocessor` (debian:bookworm-slim)
- Executa `preprocess references.json.gz /app/ivf_index.bin 1024 25 16`
- K-means usa rayon para paralelismo total no build (sem restrição de CPU)
- Tempo estimado: 3-5 min com todos os cores
- Gera `/app/ivf_index.bin` (~94 MB)
- Valida recall antes de finalizar (sample de 1000 vetores)

### Stage 3: `runtime` (debian:bookworm-slim)
- Copia apenas `api` binary + `ivf_index.bin`
- Imagem final: ~100 MB
- Zero dependências de build no runtime

### Benefícios

- Startup da API: < 1 segundo (apenas leitura de ~94 MB do disco)
- O estágio de preprocessamento é cacheado pelo Docker se `references.json.gz` não mudar
- Rebuild da API (sem mudar o índice): usa o layer do stage 2 do cache → rápido
- A separação de estágios permite ajustar parâmetros K/nprobe via ARGs Docker sem rerrodar k-means se só mudar o nprobe

## Consequências

- Build da imagem é lento na primeira vez (~5-10 min incluindo k-means)
- Imagem é "gorda" durante o build mas leve no runtime (~100 MB)
- `references.json.gz` deve estar no contexto de build (pasta `resources/`)
- O índice binário está baked na imagem — não precisa de volume externo
