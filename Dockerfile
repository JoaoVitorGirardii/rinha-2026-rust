# =============================================================================
# Stage 1: Build da toolchain Rust — preprocessamento e API
# =============================================================================
FROM rust:1.86-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends pkg-config && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Cache de dependências antes de copiar o código-fonte
# Cargo.lock é opcional — se não existir, cargo resolve as versões
COPY Cargo.toml ./
COPY crates/api/Cargo.toml ./crates/api/Cargo.toml
COPY crates/preprocess/Cargo.toml ./crates/preprocess/Cargo.toml
COPY .cargo/ ./.cargo/

# Cria stubs para cachear compilação de dependências externas
RUN mkdir -p crates/api/src crates/preprocess/src && \
    echo 'fn main(){}' > crates/api/src/main.rs && \
    echo 'fn main(){}' > crates/preprocess/src/main.rs && \
    cargo build --release 2>/dev/null || true

# Copia o código real e reconstrói
COPY crates/ ./crates/
# Toca os arquivos para forçar recompilação
RUN touch crates/api/src/main.rs crates/preprocess/src/main.rs

# Flags de compilação para Haswell (AVX2+F16C+FMA)
ENV RUSTFLAGS="-C target-cpu=haswell -C target-feature=+avx2,+f16c,+fma -C opt-level=3"

RUN cargo build --release -p api -p preprocess

# =============================================================================
# Stage 2: Geração do índice IVF-Flat (k-means sobre 3M vetores)
# Sem restrições de CPU — usa todos os cores disponíveis via rayon
# =============================================================================
FROM debian:bookworm-slim AS preprocessor

RUN apt-get update && apt-get install -y --no-install-recommends libgcc-s1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/preprocess /usr/local/bin/preprocess
COPY resources/references.json.gz /resources/references.json.gz

# K=1024 clusters, 25 iterações, nprobe padrão=16
RUN mkdir -p /app && preprocess \
    /resources/references.json.gz \
    /app/ivf_index.bin \
    1024 \
    25 \
    16

# =============================================================================
# Stage 3: Imagem de runtime — apenas o binário e o índice
# =============================================================================
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgcc-s1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/api /app/api
COPY --from=preprocessor /app/ivf_index.bin /app/ivf_index.bin

ENV INDEX_PATH=/app/ivf_index.bin
ENV PORT=3000
ENV NPROBE=16

EXPOSE 3000

CMD ["/app/api"]
