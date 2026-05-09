# ADR-001: Stack Técnico

**Data**: 2026-05-09  
**Status**: Aceito

## Contexto

O desafio exige performance extrema (p99 < 1ms) num ambiente severamente restrito: 1 CPU total + 350 MB para todos os serviços. A lógica de negócio é CPU-bound (busca em 3 milhões de vetores). Qualquer overhead de runtime, GC ou abstração excessiva degradará diretamente o score.

## Decisão

| Componente | Escolha | Alternativas Descartadas |
|---|---|---|
| Linguagem | **Rust** | Go (GC), Java (GC + JVM heap), C++ (sem async ergonomico) |
| Async runtime | **tokio `current_thread`** | tokio multi-thread (overhead desnecessário com 0.5 CPU) |
| HTTP framework | **axum 0.7** | hyper puro (mais boilerplate sem ganho mensurável), actix-web (overhead maior) |
| JSON | **serde_json** | simd-json (ganho marginal para payloads de 400 bytes, complexidade maior) |
| Datetime | **crate `time`** | chrono (mais features do que necessário), parse manual (propenso a bugs) |
| Load balancer | **nginx** | haproxy (similar), envoy (muito pesado), caddy (mais pesado) |
| Compilação | **target-cpu=haswell + AVX2 + F16C + FMA** | Generic x86_64 (sem SIMD) |

## Consequências

- Rust elimina GC pauses e permite controle explícito de layout de memória.
- `current_thread` evita wake-ups de thread desnecessários com 0.5 vCPU por instância.
- axum sobre hyper 1.0 tem overhead de roteamento < 1 µs.
- Compilar para haswell permite instruções AVX2 e F16C, críticas para o hot path de distância vetorial.
- nginx com `keepalive 128` upstream mantém pool de conexões persistentes para os backends, eliminando overhead de TCP handshake por request.
