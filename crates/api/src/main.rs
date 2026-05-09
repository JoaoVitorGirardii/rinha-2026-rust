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

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let index_path = std::env::var("INDEX_PATH")
        .unwrap_or_else(|_| "/app/ivf_index.bin".into());

    eprintln!("Carregando índice de {index_path}...");
    let index = IvfIndex::load(&index_path)?;
    eprintln!("Índice carregado: {} vetores, {} clusters", index.n, index.k);

    // nprobe lido uma única vez no startup — sem overhead por request
    let nprobe = std::env::var("NPROBE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(index.nprobe_default);

    // Warm-up: popula caches e valida path de código antes de aceitar requests
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

async fn ready_handler() -> impl IntoResponse {
    StatusCode::OK
}

async fn fraud_score_handler(
    State(state): State<SharedState>,
    Json(req): Json<FraudRequest>,
) -> impl IntoResponse {
    let query = vectorize::vectorize(&req);
    let fraud_score = state.index.search(&query, state.nprobe);

    Json(FraudResponse {
        approved: fraud_score < 0.6,
        fraud_score,
    })
}
