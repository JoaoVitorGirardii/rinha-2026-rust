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
    // f16 como u16 — usado no re-ranking exato (hot path fase 2)
    vectors: Vec<u16>,
    // uint8 quantizado — usado no scan rápido (hot path fase 1)
    vectors_u8: Vec<u8>,
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
        let mut vectors: Vec<u16> = Vec::with_capacity(n * storage_dims);
        for i in 0..n * storage_dims {
            let b = &vec_slice[i * 2..i * 2 + 2];
            vectors.push(u16::from_le_bytes(b.try_into().unwrap()));
        }

        // Deriva uint8 dos vetores f16 para o scan rápido.
        // Fórmula: ((f32 + 1.0) * 127.5) → [0..255]
        // Faixa [-1,1] → [0,255]; -1.0→0, 0.0→127, 1.0→255
        let vectors_u8: Vec<u8> = vectors
            .iter()
            .map(|&bits| {
                let v = f16::from_bits(bits).to_f32();
                ((v + 1.0) * 127.5).clamp(0.0, 255.0).round() as u8
            })
            .collect();

        Ok(IvfIndex {
            k,
            n,
            nprobe_default,
            centroids,
            cluster_offsets,
            cluster_sizes,
            labels,
            vectors,
            vectors_u8,
        })
    }

    pub fn warmup(&self, nprobe: usize, topk: usize) {
        // Resolve page faults tocando 1 byte por página (4KB) de ambos os buffers
        let _t1: u64 = self.vectors_u8.chunks(4096).fold(0u64, |a, c| a + c[0] as u64);
        let _t2: u64 = self.vectors.chunks(4096).fold(0u64, |a, c| a + c[0] as u64);

        // Treina branch predictor e caches SIMD com ~32 searches distribuídos pelos centroides
        let step = (self.k / 32).max(1);
        for centroid in self.centroids.iter().step_by(step) {
            let mut q = [0.0f32; 16];
            q[..14].copy_from_slice(centroid);
            let _ = self.search(&q, nprobe, topk);
        }
    }

    /// Retorna o fraud_score (0.0..=1.0) para o vetor query dado.
    /// query deve ser [f32; 16] com dims 14 e 15 = 0 (padding).
    #[inline]
    pub fn search(&self, query: &[f32; 16], nprobe: usize, topk: usize) -> f32 {
        match topk {
            40 => self.search_inner::<40>(query, nprobe),
            60 => self.search_inner::<60>(query, nprobe),
            _  => self.search_inner::<50>(query, nprobe),
        }
    }

    #[inline]
    fn search_inner<const N: usize>(&self, query: &[f32; 16], nprobe: usize) -> f32 {
        // Passo 1: encontrar os nprobe centroides mais próximos (f32+L2)
        let mut centroid_dists = [(0.0f32, 0u32); 1024];
        let k = self.k.min(1024);

        for i in 0..k {
            let d = dist_sq_14d_f32(query, &self.centroids[i]);
            centroid_dists[i] = (d, i as u32);
        }

        let nprobe = nprobe.min(k);
        centroid_dists[..k].select_nth_unstable_by(nprobe - 1, |a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Passo 2: quantiza query para uint8 para o scan rápido
        let query_u8: [u8; 16] = std::array::from_fn(|i| {
            ((query[i] + 1.0) * 127.5).clamp(0.0, 255.0).round() as u8
        });

        // Passo 3: scan rápido uint8+Manhattan → mantém top-N candidatos
        let mut cands = TopKN::<N>::new();

        #[cfg(target_arch = "x86_64")]
        let use_avx2 = is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch = "x86_64"))]
        let use_avx2 = false;

        for idx in 0..nprobe {
            let ci = centroid_dists[idx].1 as usize;
            let start = self.cluster_offsets[ci] as usize;
            let size = self.cluster_sizes[ci] as usize;
            let base_ptr = unsafe { self.vectors_u8.as_ptr().add(start * 16) };

            if use_avx2 {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    use std::arch::x86_64::*;
                    // Repete query nos dois halves de 128 bits do registrador YMM
                    let q128 = _mm_loadu_si128(query_u8.as_ptr() as *const __m128i);
                    let query_x2 = _mm256_set_m128i(q128, q128);

                    let pairs = size / 2;
                    for j in 0..pairs {
                        let ptr = base_ptr.add(j * 32);
                        let (d0, d1) = manhattan_u8x2_avx2(query_x2, ptr);
                        cands.push(d0, (start + j * 2) as u32);
                        cands.push(d1, (start + j * 2 + 1) as u32);
                    }
                    // Vetor ímpar restante
                    if size % 2 == 1 {
                        let global = start + size - 1;
                        let ref_u8 = base_ptr.add((size - 1) * 16) as *const [u8; 16];
                        let dist = manhattan_u8(&query_u8, &*ref_u8);
                        cands.push(dist, global as u32);
                    }
                }
            } else {
                for j in 0..size {
                    let global = start + j;
                    let ref_u8 = unsafe { base_ptr.add(j * 16) as *const [u8; 16] };
                    let dist = manhattan_u8(&query_u8, unsafe { &*ref_u8 });
                    cands.push(dist, global as u32);
                }
            }
        }

        // Passo 4: re-ranking exato com f16+L2 sobre os top-N candidatos
        let mut topk = TopK5::new();
        let count = cands.count;

        for i in 0..count {
            let global = cands.indices[i] as usize;
            let vec_start = global * 16;
            let ref_f16 = unsafe {
                self.vectors.as_ptr().add(vec_start) as *const [u16; 16]
            };
            let dist = dist_sq_f16(query, unsafe { &*ref_f16 });
            topk.push(dist, self.labels[global]);
        }

        topk.fraud_score()
    }
}

// ---------------------------------------------------------------------------
// TopKN<N>: mantém os N menores índices por distância uint8+Manhattan
// Usado para selecionar candidatos para re-ranking exato.
// ---------------------------------------------------------------------------

struct TopKN<const N: usize> {
    dists: [u32; N],
    indices: [u32; N],
    max_dist: u32,
    max_idx: usize,
    count: usize,
}

impl<const N: usize> TopKN<N> {
    #[inline(always)]
    fn new() -> Self {
        Self {
            dists: [u32::MAX; N],
            indices: [0u32; N],
            max_dist: u32::MAX,
            max_idx: 0,
            count: 0,
        }
    }

    #[inline(always)]
    fn push(&mut self, dist: u32, global_idx: u32) {
        if self.count < N {
            let i = self.count;
            self.dists[i] = dist;
            self.indices[i] = global_idx;
            self.count += 1;
            if dist > self.max_dist || i == 0 {
                self.max_dist = dist;
                self.max_idx = i;
            }
            if self.count == N {
                self.max_dist = u32::MIN;
                for j in 0..N {
                    if self.dists[j] > self.max_dist {
                        self.max_dist = self.dists[j];
                        self.max_idx = j;
                    }
                }
            }
        } else if dist < self.max_dist {
            let idx = self.max_idx;
            self.dists[idx] = dist;
            self.indices[idx] = global_idx;
            self.max_dist = u32::MIN;
            for j in 0..N {
                if self.dists[j] > self.max_dist {
                    self.max_dist = self.dists[j];
                    self.max_idx = j;
                }
            }
        }
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
// Distância Manhattan uint8 — hot path fase 1
// SSE2: _mm_sad_epu8 → 16 bytes por instrução
// AVX2: _mm256_sad_epu8 → 32 bytes (2 vetores) por instrução — 2× throughput
// ---------------------------------------------------------------------------

#[inline(always)]
fn manhattan_u8(a: &[u8; 16], b: &[u8; 16]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        return unsafe { manhattan_u8_sse2(a, b) };
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        manhattan_u8_scalar(a, b)
    }
}

// Scan de 2 vetores consecutivos de 16 bytes com AVX2.
// Retorna (dist_v0, dist_v1) em uma única instrução _mm256_sad_epu8.
// Layout de entrada: data_ptr aponta para [v0[0..16], v1[0..16]] = 32 bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn manhattan_u8x2_avx2(
    query_x2: std::arch::x86_64::__m256i,
    data_ptr: *const u8,
) -> (u32, u32) {
    use std::arch::x86_64::*;
    // Carrega 32 bytes = 2 vetores consecutivos
    let data = _mm256_loadu_si256(data_ptr as *const __m256i);
    // SAD sobre 32 bytes: 4 acumuladores de 64 bits
    // [SAD(q[0..7],v0[0..7]), SAD(q[8..15],v0[8..15]),
    //  SAD(q[0..7],v1[0..7]), SAD(q[8..15],v1[8..15])]
    let sad = _mm256_sad_epu8(query_x2, data);

    // Extrai soma para v0 (lower 128 bits)
    let lo128 = _mm256_castsi256_si128(sad);
    let d0_a = _mm_cvtsi128_si64(lo128) as u32;
    let shifted = _mm_srli_si128(lo128, 8);
    let d0_b = _mm_cvtsi128_si64(shifted) as u32;
    let dist0 = d0_a + d0_b;

    // Extrai soma para v1 (upper 128 bits)
    let hi128 = _mm256_extracti128_si256(sad, 1);
    let d1_a = _mm_cvtsi128_si64(hi128) as u32;
    let shifted2 = _mm_srli_si128(hi128, 8);
    let d1_b = _mm_cvtsi128_si64(shifted2) as u32;
    let dist1 = d1_a + d1_b;

    (dist0, dist1)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn manhattan_u8_sse2(a: &[u8; 16], b: &[u8; 16]) -> u32 {
    use std::arch::x86_64::*;
    let va = _mm_loadu_si128(a.as_ptr() as *const __m128i);
    let vb = _mm_loadu_si128(b.as_ptr() as *const __m128i);
    let sad = _mm_sad_epu8(va, vb);
    let lo = _mm_cvtsi128_si64(sad) as u32;
    let hi_reg = _mm_srli_si128(sad, 8);
    let hi = _mm_cvtsi128_si64(hi_reg) as u32;
    lo + hi
}

#[allow(dead_code)]
fn manhattan_u8_scalar(a: &[u8; 16], b: &[u8; 16]) -> u32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x.abs_diff(y) as u32).sum()
}

// ---------------------------------------------------------------------------
// Distância euclidiana ao quadrado — centroides (f32 × 14)
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
// Distância euclidiana ao quadrado — vetores f16 — hot path fase 2
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn dist_sq_f16(query: &[f32; 16], ref_f16: &[u16; 16]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
        return unsafe { dist_sq_f16_avx2(query, ref_f16) };
    }
    dist_sq_f16_scalar(query, ref_f16)
}

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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn dist_sq_f16_avx2(query: &[f32; 16], ref_f16: &[u16; 16]) -> f32 {
    use std::arch::x86_64::*;

    let q0 = _mm256_loadu_ps(query.as_ptr());
    let q1 = _mm256_loadu_ps(query.as_ptr().add(8));

    let h0 = _mm_loadu_si128(ref_f16.as_ptr() as *const __m128i);
    let h1 = _mm_loadu_si128((ref_f16.as_ptr() as *const __m128i).add(1));

    let r0 = _mm256_cvtph_ps(h0);
    let r1 = _mm256_cvtph_ps(h1);

    let d0 = _mm256_sub_ps(q0, r0);
    let d1 = _mm256_sub_ps(q1, r1);

    let s0 = _mm256_mul_ps(d0, d0);
    let s1 = _mm256_mul_ps(d1, d1);

    let sum8 = _mm256_add_ps(s0, s1);

    hsum256(sum8)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;

    let low  = _mm256_castps256_ps128(v);
    let high = _mm256_extractf128_ps(v, 1);
    let sum2 = _mm_add_ps(low, high);

    let shuf = _mm_movehdup_ps(sum2);
    let sum4 = _mm_add_ps(sum2, shuf);

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
        assert!((topk.fraud_score() - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_topk5_replaces_worst() {
        let mut topk = TopK5::new();
        topk.push(1.0, 1);
        topk.push(0.9, 1);
        topk.push(0.8, 1);
        topk.push(0.7, 1);
        topk.push(0.6, 1);
        topk.push(0.1, 0);
        assert!((topk.fraud_score() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_manhattan_u8() {
        let a = [0u8; 16];
        let b = [1u8; 16];
        assert_eq!(manhattan_u8(&a, &b), 16);
        let c = [255u8; 16];
        assert_eq!(manhattan_u8(&a, &c), 255 * 16);
    }

    #[test]
    fn test_topk50_fills_and_replaces() {
        let mut tk = TopK50::new();
        for i in 0..55u32 {
            tk.push(i * 10, i);
        }
        // Deve ter apenas os 50 menores (0..50 com dists 0..490)
        assert_eq!(tk.count, 50);
        let max = tk.dists[..50].iter().copied().max().unwrap();
        assert_eq!(max, 490); // dist 49 * 10 = 490
    }
}
