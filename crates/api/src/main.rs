mod models;
mod search;
mod vectorize;

use std::sync::Arc;

use axum::{
    Router,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json,
};
use models::{FraudRequest, FraudResponse};
use search::IvfIndex;

struct AppState {
    index: IvfIndex,
    nprobe: usize,
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
        .unwrap_or(index.nprobe_default);

    let dummy = [0.5f32; 16];
    let _ = index.search(&dummy, nprobe);

    let state: SharedState = Arc::new(AppState { index, nprobe });

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

async fn fraud_score_handler(
    State(state): State<SharedState>,
    Json(req): Json<FraudRequest>,
) -> impl IntoResponse {
    let query = vectorize::vectorize(&req);

    let fraud_score = tokio::task::spawn_blocking(move || {
        state.index.search(&query, state.nprobe)
    })
    .await
    .unwrap();

    Json(FraudResponse {
        approved: fraud_score < 0.6,
        fraud_score,
    })
}
