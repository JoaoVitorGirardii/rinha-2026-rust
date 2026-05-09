use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
};

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use half::f16;
use rayon::prelude::*;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Estruturas de entrada
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ReferenceEntry {
    vector: Vec<f32>,
    label: String,
}

// ---------------------------------------------------------------------------
// Constantes
// ---------------------------------------------------------------------------

const DIMS: usize = 14;
const STORAGE_DIMS: usize = 16; // padded para SIMD (2 × AVX2 de 8 floats)

// ---------------------------------------------------------------------------
// Carregamento do references.json.gz
// ---------------------------------------------------------------------------

fn load_references(path: &str) -> Result<(Vec<[f32; DIMS]>, Vec<u8>)> {
    eprintln!("Carregando {path}...");
    let file = File::open(path).with_context(|| format!("abrindo {path}"))?;
    let gz = GzDecoder::new(BufReader::with_capacity(1 << 22, file));

    // references.json.gz é um array JSON — carregamos tudo de uma vez.
    // No stage de build não há restrição de memória (300-400 MB durante parse é OK).
    eprintln!("  Descomprimindo e parseando JSON...");
    let entries: Vec<ReferenceEntry> =
        serde_json::from_reader(gz).context("parse de references.json.gz")?;

    let count = entries.len();
    let mut vectors: Vec<[f32; DIMS]> = Vec::with_capacity(count);
    let mut labels: Vec<u8> = Vec::with_capacity(count);

    for (i, e) in entries.into_iter().enumerate() {
        if e.vector.len() != DIMS {
            anyhow::bail!("vetor {i} com {} dims (esperado {})", e.vector.len(), DIMS);
        }
        let mut arr = [0.0f32; DIMS];
        arr.copy_from_slice(&e.vector);
        vectors.push(arr);
        labels.push(if e.label == "fraud" { 1 } else { 0 });
    }

    let frauds = labels.iter().filter(|&&l| l == 1).count();
    eprintln!("Total: {count} vetores ({frauds} fraudes, {}%)", frauds * 100 / count);
    Ok((vectors, labels))
}

// ---------------------------------------------------------------------------
// K-Means++ initialization
// ---------------------------------------------------------------------------

fn kmeans_plus_plus(vectors: &[[f32; DIMS]], k: usize, seed: u64) -> Vec<[f32; DIMS]> {
    use rand::{Rng, SeedableRng};
    use rand::rngs::SmallRng;

    let n = vectors.len();
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut centroids: Vec<[f32; DIMS]> = Vec::with_capacity(k);

    // Primeiro centroide aleatório
    let first = rng.gen_range(0..n);
    centroids.push(vectors[first]);

    // Demais centroides com probabilidade ∝ dist²
    let mut min_dists: Vec<f32> = vec![f32::INFINITY; n];

    for c_idx in 1..k {
        // Atualiza distâncias mínimas para o novo centroide adicionado
        let prev_c = &centroids[c_idx - 1];
        min_dists.par_iter_mut().zip(vectors.par_iter()).for_each(|(d, v)| {
            let dist = dist_sq_14d(v, prev_c);
            if dist < *d {
                *d = dist;
            }
        });

        // Amostragem proporcional a dist²
        let total: f64 = min_dists.iter().map(|&d| d as f64).sum();
        let threshold = rng.gen::<f64>() * total;
        let mut cumsum = 0.0f64;
        let mut chosen = n - 1;
        for (i, &d) in min_dists.iter().enumerate() {
            cumsum += d as f64;
            if cumsum >= threshold {
                chosen = i;
                break;
            }
        }
        centroids.push(vectors[chosen]);

        if (c_idx + 1) % 100 == 0 || c_idx + 1 == k {
            eprintln!("  k-means++ init: {}/{k} centroides", c_idx + 1);
        }
    }

    centroids
}

// ---------------------------------------------------------------------------
// K-Means Lloyd iterations (paralelas com rayon)
// ---------------------------------------------------------------------------

fn run_kmeans(
    vectors: &[[f32; DIMS]],
    k: usize,
    iterations: usize,
) -> (Vec<[f32; DIMS]>, Vec<u32>) {
    eprintln!("Inicializando {k} centroides com k-means++...");
    let mut centroids = kmeans_plus_plus(vectors, k, 42);

    let n = vectors.len();
    let mut assignments = vec![0u32; n];

    for iter in 0..iterations {
        // Assign: cada vetor ao centroide mais próximo (paralelo)
        let new_assignments: Vec<u32> = vectors
            .par_iter()
            .map(|v| nearest_centroid(v, &centroids) as u32)
            .collect();

        // Conta mudanças
        let changes: usize = assignments
            .iter()
            .zip(new_assignments.iter())
            .filter(|(a, b)| a != b)
            .count();

        assignments = new_assignments;

        // Update: recalcula centroides como média dos pontos atribuídos
        let mut sums = vec![[0.0f64; DIMS]; k];
        let mut counts = vec![0u64; k];

        for (v, &c) in vectors.iter().zip(assignments.iter()) {
            let ci = c as usize;
            counts[ci] += 1;
            for d in 0..DIMS {
                sums[ci][d] += v[d] as f64;
            }
        }

        let mut max_delta = 0.0f32;
        for i in 0..k {
            if counts[i] == 0 {
                continue; // centroide vazio, mantém o anterior
            }
            let mut new_c = [0.0f32; DIMS];
            for d in 0..DIMS {
                new_c[d] = (sums[i][d] / counts[i] as f64) as f32;
            }
            let delta = dist_sq_14d(&new_c, &centroids[i]).sqrt();
            if delta > max_delta {
                max_delta = delta;
            }
            centroids[i] = new_c;
        }

        eprintln!(
            "Iteração {}/{iterations}: {changes} mudanças, delta_max={max_delta:.6}",
            iter + 1
        );

        if max_delta < 1e-5 {
            eprintln!("Convergiu na iteração {}", iter + 1);
            break;
        }
    }

    (centroids, assignments)
}

// ---------------------------------------------------------------------------
// Escrita do índice binário
// ---------------------------------------------------------------------------

fn write_index(
    output_path: &str,
    k: usize,
    n: usize,
    nprobe_default: usize,
    centroids: &[[f32; DIMS]],
    sorted_vectors: &[[f32; DIMS]],
    sorted_labels: &[u8],
    cluster_offsets: &[u32],
    cluster_sizes: &[u32],
) -> Result<()> {
    eprintln!("Escrevendo índice em {output_path}...");
    let file = File::create(output_path)
        .with_context(|| format!("criando arquivo {output_path}"))?;
    let mut w = BufWriter::with_capacity(1 << 22, file);

    // Header: 64 bytes
    w.write_all(b"IVFFLAT1")?;
    w.write_all(&1u32.to_le_bytes())?;                          // version
    w.write_all(&(k as u32).to_le_bytes())?;                    // K
    w.write_all(&(n as u32).to_le_bytes())?;                    // N
    w.write_all(&(DIMS as u32).to_le_bytes())?;                  // dims reais
    w.write_all(&(STORAGE_DIMS as u32).to_le_bytes())?;          // storage dims (padded)
    w.write_all(&(nprobe_default as u32).to_le_bytes())?;        // nprobe default
    // Reserved: 28 bytes de zeros para preencher até 64 bytes
    // (já temos 8+4+4+4+4+4+4 = 32 bytes, precisamos de 32 de reserved)
    w.write_all(&[0u8; 32])?;

    // Centroids: k * DIMS * 4 bytes
    for c in centroids {
        for &val in c.iter() {
            w.write_all(&val.to_le_bytes())?;
        }
    }

    // ClusterOffsets: k * 4 bytes
    for &off in cluster_offsets {
        w.write_all(&off.to_le_bytes())?;
    }

    // ClusterSizes: k * 4 bytes
    for &sz in cluster_sizes {
        w.write_all(&sz.to_le_bytes())?;
    }

    // Labels: n bytes
    w.write_all(sorted_labels)?;

    // Vectors: n * STORAGE_DIMS * 2 bytes (f16 padded)
    for vec_f32 in sorted_vectors {
        for i in 0..STORAGE_DIMS {
            let val = if i < DIMS { vec_f32[i] } else { 0.0f32 };
            let h = f16::from_f32(val);
            w.write_all(&h.to_bits().to_le_bytes())?;
        }
    }

    w.flush()?;
    eprintln!("Índice escrito com sucesso.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Funções auxiliares de distância
// ---------------------------------------------------------------------------

#[inline(always)]
fn dist_sq_14d(a: &[f32; DIMS], b: &[f32; DIMS]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..DIMS {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

#[inline(always)]
fn nearest_centroid(v: &[f32; DIMS], centroids: &[[f32; DIMS]]) -> usize {
    let mut best_i = 0;
    let mut best_d = f32::INFINITY;
    for (i, c) in centroids.iter().enumerate() {
        let d = dist_sq_14d(v, c);
        if d < best_d {
            best_d = d;
            best_i = i;
        }
    }
    best_i
}

// ---------------------------------------------------------------------------
// Validação simples do índice gerado (brute-force em amostra pequena)
// ---------------------------------------------------------------------------

fn validate_recall(
    vectors: &[[f32; DIMS]],
    labels: &[u8],
    centroids: &[[f32; DIMS]],
    sorted_vectors: &[[f32; DIMS]],
    sorted_labels: &[u8],
    cluster_offsets: &[u32],
    cluster_sizes: &[u32],
    n_samples: usize,
    nprobe: usize,
) {
    use rand::{seq::SliceRandom, SeedableRng};
    use rand::rngs::SmallRng;

    let k = centroids.len();
    let mut rng = SmallRng::seed_from_u64(123);
    let indices: Vec<usize> = (0..vectors.len()).collect();
    let sample: Vec<usize> = indices
        .choose_multiple(&mut rng, n_samples)
        .cloned()
        .collect();

    let mut matches = 0usize;
    for &qi in &sample {
        let query = &vectors[qi];

        // Brute-force: 5 vizinhos mais próximos
        let mut dists: Vec<(f32, usize)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (dist_sq_14d(query, v), i))
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let bf_fraud: f32 = dists[0..5].iter().filter(|(_, i)| labels[*i] == 1).count() as f32 / 5.0;
        let bf_approved = bf_fraud < 0.6;

        // IVF search
        let mut centroid_dists: Vec<(f32, usize)> = centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (dist_sq_14d(query, c), i))
            .collect();
        centroid_dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // TopK5 simples para validação
        let mut topk_dists = [f32::INFINITY; 5];
        let mut topk_labels = [0u8; 5];
        let mut max_d = f32::INFINITY;
        let mut max_i = 0usize;

        for &(_, ci) in &centroid_dists[..nprobe] {
            let start = cluster_offsets[ci] as usize;
            let size = cluster_sizes[ci] as usize;
            for j in 0..size {
                let rv = &sorted_vectors[start + j];
                let mut d = 0.0f32;
                for dim in 0..DIMS {
                    let diff = query[dim] - rv[dim];
                    d += diff * diff;
                }
                if d < max_d {
                    topk_dists[max_i] = d;
                    topk_labels[max_i] = sorted_labels[start + j];
                    max_d = f32::NEG_INFINITY;
                    for ii in 0..5 {
                        if topk_dists[ii] > max_d {
                            max_d = topk_dists[ii];
                            max_i = ii;
                        }
                    }
                }
            }
        }

        let ivf_fraud: f32 = topk_labels.iter().filter(|&&l| l == 1).count() as f32 / 5.0;
        let ivf_approved = ivf_fraud < 0.6;

        if bf_approved == ivf_approved {
            matches += 1;
        }
    }

    let recall_pct = matches as f64 / n_samples as f64 * 100.0;
    eprintln!(
        "Validação recall (nprobe={nprobe}): {matches}/{n_samples} ({recall_pct:.1}%)"
    );
    if recall_pct < 95.0 {
        eprintln!("AVISO: recall < 95%, considere aumentar nprobe ou K");
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Uso: preprocess <input.json.gz> <output.bin> [K=1024] [iterations=25] [nprobe=16]");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];
    let k: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let iterations: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(25);
    let nprobe_default: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(16);

    eprintln!("Parâmetros: K={k}, iterations={iterations}, nprobe_default={nprobe_default}");

    let (vectors, labels) = load_references(input_path)?;
    let n = vectors.len();

    let (centroids, assignments) = run_kmeans(&vectors, k, iterations);

    // Ordena vetores e labels por cluster
    eprintln!("Reorganizando vetores por cluster...");
    let mut cluster_indices: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &ci) in assignments.iter().enumerate() {
        cluster_indices[ci as usize].push(i);
    }

    let mut cluster_offsets = vec![0u32; k];
    let mut cluster_sizes = vec![0u32; k];
    let mut sorted_vectors: Vec<[f32; DIMS]> = Vec::with_capacity(n);
    let mut sorted_labels: Vec<u8> = Vec::with_capacity(n);

    let mut offset = 0u32;
    for ci in 0..k {
        cluster_offsets[ci] = offset;
        cluster_sizes[ci] = cluster_indices[ci].len() as u32;
        for &idx in &cluster_indices[ci] {
            sorted_vectors.push(vectors[idx]);
            sorted_labels.push(labels[idx]);
        }
        offset += cluster_sizes[ci];
    }

    // Valida recall numa amostra de 1000 vetores antes de escrever
    eprintln!("Validando recall...");
    validate_recall(
        &vectors,
        &labels,
        &centroids,
        &sorted_vectors,
        &sorted_labels,
        &cluster_offsets,
        &cluster_sizes,
        1000,
        nprobe_default,
    );

    write_index(
        output_path,
        k,
        n,
        nprobe_default,
        &centroids,
        &sorted_vectors,
        &sorted_labels,
        &cluster_offsets,
        &cluster_sizes,
    )?;

    // Mostra estatísticas de clusters
    let min_size = cluster_sizes.iter().min().copied().unwrap_or(0);
    let max_size = cluster_sizes.iter().max().copied().unwrap_or(0);
    let avg_size = n as f64 / k as f64;
    eprintln!("Clusters: min={min_size}, max={max_size}, avg={avg_size:.0}");

    Ok(())
}
