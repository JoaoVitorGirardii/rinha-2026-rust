# ADR-005: Budget de Memória e Arquitetura de Serviços

**Data**: 2026-05-09  
**Status**: Aceito

## Contexto

O desafio impõe um limite total de **350 MB de RAM** para todos os serviços combinados, com a obrigação de ter no mínimo 1 load balancer e 2 instâncias de API. Cada container tem seu próprio limite declarado no `docker-compose.yml`.

## Decisão

### Alocação de recursos

| Serviço | CPUs | Memória | Justificativa |
|---|---|---|---|
| `api1` | 0.45 | 167 MB | 96 MB vetores f16 + labels + heap Rust |
| `api2` | 0.45 | 167 MB | idem |
| `nginx` | 0.10 | 16 MB | worker_processes=1, buffers mínimos |
| **Total** | **1.00** | **350 MB** | exato |

### Breakdown de memória por instância de API

| Componente | Tamanho |
|---|---|
| Vetores f16 padded (3M × 32 bytes) | 96.0 MB |
| Labels (3M × 1 byte) | 3.0 MB |
| Centroides (1024 × 14 × 4 bytes) | 0.1 MB |
| Offsets e tamanhos (1024 × 8 bytes) | 0.1 MB |
| Heap Rust (tokio, axum, buffers HTTP) | ~5 MB |
| Stack de threads + código | ~3 MB |
| **Total** | **~107 MB** ✓ |

Margem por instância: 167 − 107 = **60 MB de folga**.

### Por que não compartilhar memória entre instâncias?

Duas opções de compartilhamento foram consideradas:
1. **mmap + Docker volume**: complexidade no docker-compose (volume init container)
2. **Mesma imagem → overlay2 page cache sharing**: funciona no host, mas cgroups v2 contam o page cache por container

Conclusão: o armazenamento em f16 torna desnecessário o compartilhamento — 107 MB/instância está bem dentro dos 167 MB.

## Consequências

- Sem necessidade de volumes externos ou shared memory
- Cold start: ~1 segundo (leitura de 94 MB para cada instância)
- O nginx com `keepalive 128` mantém pool de conexões para os backends, eliminando overhead por request
- Se o índice crescer (mais vetores ou dimensões maiores), reconsiderar f16 vs mmap sharing
