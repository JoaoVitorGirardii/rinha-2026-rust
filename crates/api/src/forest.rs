use std::fs;
use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Deserialize)]
struct RawModel {
    n_trees: usize,
    threshold: f32,
    trees: Vec<RawTree>,
}

#[derive(Deserialize)]
struct RawTree {
    n_nodes: usize,
    features: Vec<i16>,
    thresholds: Vec<f32>,
    lefts: Vec<i32>,
    rights: Vec<i32>,
    probs: Vec<f32>,
}

/// Random Forest com layout de arrays planos (melhor localidade de cache).
pub struct Forest {
    n_trees: usize,
    threshold: f32,
    // Arrays planos — todos os nós de todas as árvores concatenados.
    feat:    Vec<i16>,   // feature index; -2 em folhas
    thresh:  Vec<f32>,   // limiar de divisão
    left:    Vec<i32>,   // índice global do filho esquerdo; -1 em folhas
    right:   Vec<i32>,   // índice global do filho direito
    probs:   Vec<f32>,   // probabilidade de fraude (só nas folhas; -1 nos nós internos)
    offsets: Vec<usize>, // offsets[t] = posição de início da árvore t
}

impl Forest {
    pub fn load(path: &str) -> Result<Self> {
        let data = fs::read_to_string(path)
            .with_context(|| format!("lendo modelo: {path}"))?;
        let raw: RawModel = sonic_rs::from_str(&data)
            .context("parseando model-tree.json")?;

        let total_nodes: usize = raw.trees.iter().map(|t| t.n_nodes).sum();
        let n_trees = raw.n_trees.min(raw.trees.len());

        let mut feat    = Vec::with_capacity(total_nodes);
        let mut thresh  = Vec::with_capacity(total_nodes);
        let mut left    = Vec::with_capacity(total_nodes);
        let mut right   = Vec::with_capacity(total_nodes);
        let mut probs   = Vec::with_capacity(total_nodes);
        let mut offsets = Vec::with_capacity(n_trees + 1);

        let mut pos: i32 = 0;
        for tree in raw.trees.iter().take(n_trees) {
            offsets.push(pos as usize);
            for i in 0..tree.n_nodes {
                feat.push(tree.features[i]);
                thresh.push(tree.thresholds[i]);
                // Converte índices locais → índices globais absolutos
                left.push(if tree.lefts[i] == -1 { -1 } else { pos + tree.lefts[i] });
                right.push(if tree.rights[i] == -1 { -1 } else { pos + tree.rights[i] });
                probs.push(tree.probs[i]);
            }
            pos += tree.n_nodes as i32;
        }
        offsets.push(pos as usize);

        let node_counts: Vec<usize> = raw.trees.iter().take(n_trees).map(|t| t.n_nodes).collect();
        let avg_nodes = total_nodes as f64 / n_trees as f64;
        eprintln!(
            "Modelo carregado: {} árvores, {} nós totais (avg={:.0}/árvore), threshold={:.4}",
            n_trees, total_nodes, avg_nodes, raw.threshold
        );
        let _ = node_counts;

        Ok(Forest {
            n_trees,
            threshold: raw.threshold,
            feat,
            thresh,
            left,
            right,
            probs,
            offsets,
        })
    }

    /// Retorna (approved, fraud_score) para um vetor de features.
    /// features: dims 0–13 (14 features); dims 14–15 são padding e ignorados.
    #[inline]
    pub fn predict(&self, features: &[f32; 16]) -> f32 {
        let mut acc = 0.0f32;

        for t in 0..self.n_trees {
            let mut idx = self.offsets[t];
            // Percorre até a folha
            loop {
                let l = self.left[idx];
                if l == -1 {
                    acc += self.probs[idx];
                    break;
                }
                let f = self.feat[idx] as usize;
                idx = if features[f] <= self.thresh[idx] {
                    l as usize
                } else {
                    self.right[idx] as usize
                };
            }
        }

        acc / self.n_trees as f32
    }

    #[inline]
    pub fn fraud_score(&self, features: &[f32; 16]) -> f32 {
        self.predict(features)
    }

    #[inline]
    pub fn is_fraud(&self, features: &[f32; 16]) -> bool {
        self.predict(features) >= self.threshold
    }

    #[inline]
    pub fn threshold(&self) -> f32 {
        self.threshold
    }
}
