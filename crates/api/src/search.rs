use std::fs;
use anyhow::{Context, Result, bail};
use half::f16;

const MAGIC: &[u8; 8] = b"IVFFLAT1";

pub struct IvfIndex {
    pub k: usize,
    pub n: usize,
    pub nprobe_default: usize,
    centroids: Vec<[f32; 14]>,
    cluster_offsets: Vec<u32>,
    cluster_sizes: Vec<u32>,
    labels: Vec<u8>,
    // Vetores armazenados como f16 em u16, padded para 16 valores por vetor
    vectors: Vec<u16>,
}

impl IvfIndex {
    pub fn load(path: &str) -> Result<Self> {
        let data = fs::read(path).with_context(|| format!("lendo índice: {path}"))?;

        if data.len() < 64 {
            bail!("arquivo de índice muito pequeno");
        }
        if &data[0..8] != MAGIC {
            bail!("magic inválido no índice");
        }

        let version = u32::from_le_bytes(data[8..12].try_into().unwrap());
        if version != 1 {
            bail!("versão de índice não suportada: {version}");
        }

        let k = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
        let n = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
        let _dims = u32::from_le_bytes(data[20..24].try_into().unwrap()) as usize;
        let storage_dims = u32::from_le_bytes(data[24..28].try_into().unwrap()) as usize;
        let nprobe_default = u32::from_le_bytes(data[28..32].try_into().unwrap()) as usize;

        let mut offset = 64usize;

        // Centroids: k * 14 * f32
        let centroid_bytes = k * 14 * 4;
        let centroid_slice = &data[offset..offset + centroid_bytes];
        let mut centroids: Vec<[f32; 14]> = Vec::with_capacity(k);
        for i in 0..k {
            let mut c = [0.0f32; 14];
            for j in 0..14 {
                let b = &centroid_slice[(i * 14 + j) * 4..(i * 14 + j) * 4 + 4];
                c[j] = f32::from_le_bytes(b.try_into().unwrap());
            }
            centroids.push(c);
        }
        offset += centroid_bytes;

        // ClusterOffsets: k * u32
        let offsets_bytes = k * 4;
        let mut cluster_offsets: Vec<u32> = Vec::with_capacity(k);
        for i in 0..k {
            let b = &data[offset + i * 4..offset + i * 4 + 4];
            cluster_offsets.push(u32::from_le_bytes(b.try_into().unwrap()));
        }
        offset += offsets_bytes;

        // ClusterSizes: k * u32
        let sizes_bytes = k * 4;
        let mut cluster_sizes: Vec<u32> = Vec::with_capacity(k);
        for i in 0..k {
            let b = &data[offset + i * 4..offset + i * 4 + 4];
            cluster_sizes.push(u32::from_le_bytes(b.try_into().unwrap()));
        }
        offset += sizes_bytes;

        // Labels: n bytes
        let labels = data[offset..offset + n].to_vec();
        offset += n;

        // Vectors: n * storage_dims * 2 bytes (f16 as u16)
        let vec_bytes = n * storage_dims * 2;
        let vec_slice = &data[offset..offset + vec_bytes];
        // Interpreta como u16 little-endian
        let mut vectors: Vec<u16> = Vec::with_capacity(n * storage_dims);
        for i in 0..n * storage_dims {
            let b = &vec_slice[i * 2..i * 2 + 2];
            vectors.push(u16::from_le_bytes(b.try_into().unwrap()));
        }

        Ok(IvfIndex {
            k,
            n,
            nprobe_default,
            centroids,
            cluster_offsets,
            cluster_sizes,
            labels,
            vectors,
        })
    }

    /// Retorna o fraud_score (0.0..=1.0) para o vetor query dado.
    /// query deve ser [f32; 16] com dims 14 e 15 = 0 (padding).
    #[inline]
    pub fn search(&self, query: &[f32; 16], nprobe: usize) -> f32 {
        // Passo 1: encontrar os nprobe centroides mais próximos
        // Stack-allocated para evitar heap alloc: K=1024 entries × 8 bytes = 8 KB
        let mut centroid_dists = [(0.0f32, 0u32); 1024];
        let k = self.k.min(1024);

        for i in 0..k {
            let d = dist_sq_14d_f32(query, &self.centroids[i]);
            centroid_dists[i] = (d, i as u32);
        }

        // Particiona para encontrar os nprobe menores sem sort completo
        let nprobe = nprobe.min(k);
        centroid_dists[..k].select_nth_unstable_by(nprobe - 1, |a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Passo 2: escanear os nprobe clusters com SIMD
        let mut topk = TopK5::new();

        for idx in 0..nprobe {
            let ci = centroid_dists[idx].1 as usize;
            let start = self.cluster_offsets[ci] as usize;
            let size = self.cluster_sizes[ci] as usize;

            for j in 0..size {
                let vec_start = (start + j) * 16;
                // SAFETY: vec_start + 16 é sempre <= vectors.len() pois o índice foi
                // construído com storage_dims=16 para cada um dos n vetores.
                let ref_f16 = unsafe {
                    self.vectors.as_ptr().add(vec_start) as *const [u16; 16]
                };
                let dist = dist_sq_f16(query, unsafe { &*ref_f16 });
                topk.push(dist, self.labels[start + j]);
            }
        }

        topk.fraud_score()
    }
}

// ---------------------------------------------------------------------------
// TopK5: mantém os 5 menores sem heap — mais rápido para k=5
// ---------------------------------------------------------------------------

struct TopK5 {
    dists: [f32; 5],
    labels: [u8; 5],
    max_dist: f32,
    max_idx: usize,
}

impl TopK5 {
    #[inline(always)]
    fn new() -> Self {
        Self {
            dists: [f32::INFINITY; 5],
            labels: [0u8; 5],
            max_dist: f32::INFINITY,
            max_idx: 0,
        }
    }

    #[inline(always)]
    fn push(&mut self, dist: f32, label: u8) {
        if dist < self.max_dist {
            let idx = self.max_idx;
            self.dists[idx] = dist;
            self.labels[idx] = label;
            // Atualiza o máximo com scan linear em 5 elementos
            self.max_dist = f32::NEG_INFINITY;
            for i in 0..5 {
                if self.dists[i] > self.max_dist {
                    self.max_dist = self.dists[i];
                    self.max_idx = i;
                }
            }
        }
    }

    #[inline(always)]
    fn fraud_score(&self) -> f32 {
        let frauds = self.labels.iter().filter(|&&l| l == 1).count();
        frauds as f32 / 5.0
    }
}

// ---------------------------------------------------------------------------
// Distância euclidiana ao quadrado — centraloides (f32 × 14)
// ---------------------------------------------------------------------------

#[inline(always)]
fn dist_sq_14d_f32(a: &[f32; 16], b: &[f32; 14]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..14 {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

// ---------------------------------------------------------------------------
// Distância euclidiana ao quadrado — vetores f16 (hot path)
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn dist_sq_f16(query: &[f32; 16], ref_f16: &[u16; 16]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            return unsafe { dist_sq_f16_avx2(query, ref_f16) };
        }
    }
    dist_sq_f16_scalar(query, ref_f16)
}

/// Fallback escalar: converte f16 → f32 e computa distância euclidiana quadrada.
#[inline]
fn dist_sq_f16_scalar(query: &[f32; 16], ref_f16: &[u16; 16]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..16 {
        let r = f16::from_bits(ref_f16[i]).to_f32();
        let d = query[i] - r;
        sum += d * d;
    }
    sum
}

/// Hot path SIMD: AVX2 + F16C.
/// Carrega 16 f16 em 2 batches de 8, converte para f32, computa dist² horizontal.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn dist_sq_f16_avx2(query: &[f32; 16], ref_f16: &[u16; 16]) -> f32 {
    use std::arch::x86_64::*;

    // Carrega query como dois registradores f32×8
    let q0 = _mm256_loadu_ps(query.as_ptr());           // query[0..8]
    let q1 = _mm256_loadu_ps(query.as_ptr().add(8));    // query[8..16]

    // Carrega ref como dois __m128i (8 f16 × 2 bytes = 16 bytes cada)
    let h0 = _mm_loadu_si128(ref_f16.as_ptr() as *const __m128i);        // ref_f16[0..8]
    let h1 = _mm_loadu_si128((ref_f16.as_ptr() as *const __m128i).add(1)); // ref_f16[8..16]

    // Converte f16 → f32 via F16C
    let r0 = _mm256_cvtph_ps(h0);
    let r1 = _mm256_cvtph_ps(h1);

    // Diferença
    let d0 = _mm256_sub_ps(q0, r0);
    let d1 = _mm256_sub_ps(q1, r1);

    // Quadrado
    let s0 = _mm256_mul_ps(d0, d0);
    let s1 = _mm256_mul_ps(d1, d1);

    // Soma dos dois registradores
    let sum8 = _mm256_add_ps(s0, s1);

    // Redução horizontal: 8 floats → 1 float
    hsum256(sum8)
}

/// Redução horizontal de __m256 (8 floats) para um único f32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;

    // Divide em duas metades de 128-bit e soma
    let low  = _mm256_castps256_ps128(v);
    let high = _mm256_extractf128_ps(v, 1);
    let sum2 = _mm_add_ps(low, high);

    // Reduz 4 floats para 2
    let shuf = _mm_movehdup_ps(sum2);
    let sum4 = _mm_add_ps(sum2, shuf);

    // Reduz 2 floats para 1
    let hi64 = _mm_movehl_ps(sum4, sum4);
    let sum8 = _mm_add_ss(sum4, hi64);

    _mm_cvtss_f32(sum8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topk5_basic() {
        let mut topk = TopK5::new();
        topk.push(0.5, 1);
        topk.push(0.1, 0);
        topk.push(0.9, 1);
        topk.push(0.3, 0);
        topk.push(0.7, 1);
        // fraud_score = 3/5 = 0.6
        assert!((topk.fraud_score() - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_topk5_replaces_worst() {
        let mut topk = TopK5::new();
        // Insere 5 valores com labels
        topk.push(1.0, 1);
        topk.push(0.9, 1);
        topk.push(0.8, 1);
        topk.push(0.7, 1);
        topk.push(0.6, 1);
        // Insere valor melhor → deve substituir o pior (1.0)
        topk.push(0.1, 0);
        // Agora labels devem ter 4 fraudes e 1 legit
        assert!((topk.fraud_score() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_dist_scalar_vs_simd() {
        let query = [0.5f32, 0.1, -1.0, 0.8, 0.3, -1.0, -1.0, 0.05, 0.15, 0.0, 1.0, 0.0, 0.5, 0.006, 0.0, 0.0];
        // Cria vetor ref em f16
        let mut ref_f16 = [0u16; 16];
        for i in 0..16 {
            ref_f16[i] = half::f16::from_f32(query[i] * 0.9).to_bits();
        }
        let scalar = dist_sq_f16_scalar(&query, &ref_f16);

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            let simd = unsafe { dist_sq_f16_avx2(&query, &ref_f16) };
            assert!(
                (scalar - simd).abs() < 1e-3,
                "scalar={scalar} simd={simd} devem ser iguais dentro da tolerância f16"
            );
        }
    }
}
