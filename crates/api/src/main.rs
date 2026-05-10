mod search;
mod vectorize;

use std::sync::Arc;

use axum::{
    Router,
    body::Bytes,
    extract::State,
    http::{StatusCode, header},
    response::IntoResponse,
    routing::{get, post},
};
use search::IvfIndex;

struct AppState {
    index: IvfIndex,
    nprobe: usize,
    topk: usize,
}

type SharedState = Arc<AppState>;

fn main() -> anyhow::Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .max_blocking_threads(4)
        .enable_all()
        .build()?
        .block_on(run())
}

async fn run() -> anyhow::Result<()> {
    pin_to_first_core();

    let index_path = std::env::var("INDEX_PATH")
        .unwrap_or_else(|_| "/app/ivf_index.bin".into());

    eprintln!("Carregando índice de {index_path}...");
    let index = IvfIndex::load(&index_path)?;
    eprintln!("Índice carregado: {} vetores, {} clusters", index.n, index.k);

    let nprobe = std::env::var("NPROBE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7usize);

    let topk = std::env::var("TOPK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60usize);

    index.warmup(nprobe, topk);
    eprintln!("Warmup concluído — topk={topk}");

    let state: SharedState = Arc::new(AppState { index, nprobe, topk });

    let app = Router::new()
        .route("/fraud-score", post(fraud_score_handler))
        .route("/ready", get(ready_handler))
        .with_state(state);

    let port = std::env::var("PORT").unwrap_or_else(|_| "3000".into());
    let addr = format!("0.0.0.0:{port}");
    eprintln!("Escutando em {addr} com nprobe={nprobe}");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

fn pin_to_first_core() {
    #[cfg(target_os = "linux")]
    {
        if let Some(ids) = core_affinity::get_core_ids() {
            if let Some(id) = ids.into_iter().next() {
                let _ = core_affinity::set_for_current(id);
            }
        }
    }
}

async fn ready_handler() -> impl IntoResponse {
    StatusCode::OK
}

// K=5 vizinhos → fraud_score ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} — 6 valores fixos
static RESPONSES: [&[u8]; 6] = [
    b"{\"approved\":true,\"fraud_score\":0.0}",
    b"{\"approved\":true,\"fraud_score\":0.2}",
    b"{\"approved\":true,\"fraud_score\":0.4}",
    b"{\"approved\":false,\"fraud_score\":0.6}",
    b"{\"approved\":false,\"fraud_score\":0.8}",
    b"{\"approved\":false,\"fraud_score\":1.0}",
];

async fn fraud_score_handler(
    State(state): State<SharedState>,
    body: Bytes,
) -> impl IntoResponse {
    let query = vectorize::vectorize_raw(&body);

    let fraud_score = tokio::task::spawn_blocking(move || {
        state.index.search(&query, state.nprobe, state.topk)
    })
    .await
    .unwrap();

    let idx = (fraud_score * 5.0).round() as usize;
    ([(header::CONTENT_TYPE, "application/json")], RESPONSES[idx])
}
