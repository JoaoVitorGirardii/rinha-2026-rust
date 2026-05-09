use crate::models::FraudRequest;

const MAX_AMOUNT: f32 = 10000.0;
const MAX_INSTALLMENTS: f32 = 12.0;
const AMOUNT_VS_AVG_RATIO: f32 = 10.0;
const MAX_MINUTES: f32 = 1440.0;
const MAX_KM: f32 = 1000.0;
const MAX_TX_COUNT_24H: f32 = 20.0;
const MAX_MERCHANT_AVG: f32 = 10000.0;

#[inline(always)]
fn mcc_risk(mcc: &str) -> f32 {
    match mcc {
        "5411" => 0.15,
        "5812" => 0.30,
        "5912" => 0.20,
        "5944" => 0.45,
        "7801" => 0.80,
        "7802" => 0.75,
        "7995" => 0.85,
        "4511" => 0.35,
        "5311" => 0.25,
        "5999" => 0.50,
        _ => 0.50,
    }
}

// ---------------------------------------------------------------------------
// Parse manual de timestamps ISO 8601 UTC — "YYYY-MM-DDTHH:MM:SSZ"
// Posições fixas, sem dependências externas.
// ---------------------------------------------------------------------------

#[inline(always)]
fn d(b: u8) -> u32 {
    (b - b'0') as u32
}

/// Retorna (hour_of_day, day_of_week_mon0, unix_seconds).
/// day_of_week: Mon=0 … Sun=6, conforme especificação do desafio.
#[inline]
fn parse_ts(ts: &str) -> (u32, u32, i64) {
    // "2026-03-11T18:45:53Z"
    //  0123456789012345678901
    let b = ts.as_bytes();
    let year  = d(b[0]) * 1000 + d(b[1]) * 100 + d(b[2]) * 10 + d(b[3]);
    let month = d(b[5]) * 10 + d(b[6]);
    let day   = d(b[8]) * 10 + d(b[9]);
    let hour  = d(b[11]) * 10 + d(b[12]);
    let min   = d(b[14]) * 10 + d(b[15]);
    let sec   = d(b[17]) * 10 + d(b[18]);

    let dow = day_of_week_mon0(year, month, day);
    let unix = to_unix(year, month, day, hour, min, sec);
    (hour, dow, unix)
}

/// Algoritmo de Tomohiko Sakamoto: retorna 0=Dom..6=Sáb, depois convertemos para Mon=0.
#[inline]
fn day_of_week_mon0(year: u32, month: u32, day: u32) -> u32 {
    const T: [u32; 12] = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let y = if month < 3 { year - 1 } else { year };
    let dow_sun = (y + y / 4 - y / 100 + y / 400 + T[(month - 1) as usize] + day) % 7;
    // 0=Dom → Mon=1→0, Ter=2→1, …, Dom=0→6
    (dow_sun + 6) % 7
}

/// Unix timestamp (segundos desde 1970-01-01T00:00:00Z), calendário gregoriano.
#[inline]
fn to_unix(year: u32, month: u32, day: u32, hour: u32, min: u32, sec: u32) -> i64 {
    // Algoritmo proleptic Gregorian → days since Unix epoch
    let y = year as i64;
    let m = month as i64;
    let d = day as i64;
    let (y, m) = if m <= 2 { (y - 1, m + 9) } else { (y, m - 3) };
    let era = y.div_euclid(400);
    let yoe = y - era * 400; // 0..399
    let doy = (153 * m + 2) / 5 + d - 1; // 0..365
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;

    days * 86400 + hour as i64 * 3600 + min as i64 * 60 + sec as i64
}

// ---------------------------------------------------------------------------
// Vectorização principal
// ---------------------------------------------------------------------------

/// Transforma um request em vetor [f32; 16]: 14 dimensões + 2 zeros de padding SIMD.
pub fn vectorize(req: &FraudRequest) -> [f32; 16] {
    let mut v = [0.0f32; 16];

    let amount = req.transaction.amount as f32;
    let avg_amount = req.customer.avg_amount as f32;

    // dim 0: amount normalizado
    v[0] = (amount / MAX_AMOUNT).clamp(0.0, 1.0);

    // dim 1: installments
    v[1] = (req.transaction.installments as f32 / MAX_INSTALLMENTS).clamp(0.0, 1.0);

    // dim 2: razão amount vs média do cliente
    v[2] = if avg_amount > 0.0 {
        (amount / (avg_amount * AMOUNT_VS_AVG_RATIO)).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let (hour, dow, cur_unix) = parse_ts(&req.transaction.requested_at);

    // dim 3: hora do dia (UTC)
    v[3] = hour as f32 / 23.0;

    // dim 4: dia da semana (Mon=0, Sun=6)
    v[4] = dow as f32 / 6.0;

    // dims 5 e 6: minutos desde última tx e km — ou -1 se ausente
    if let Some(last) = &req.last_transaction {
        let (_, _, last_unix) = parse_ts(&last.timestamp);
        let delta_min = ((cur_unix - last_unix).max(0) / 60) as f32;
        v[5] = (delta_min / MAX_MINUTES).clamp(0.0, 1.0);
        v[6] = (last.km_from_current as f32 / MAX_KM).clamp(0.0, 1.0);
    } else {
        v[5] = -1.0;
        v[6] = -1.0;
    }

    // dim 7: km do lar
    v[7] = (req.terminal.km_from_home as f32 / MAX_KM).clamp(0.0, 1.0);

    // dim 8: transações nas últimas 24h
    v[8] = (req.customer.tx_count_24h as f32 / MAX_TX_COUNT_24H).clamp(0.0, 1.0);

    // dim 9: terminal online
    v[9] = if req.terminal.is_online { 1.0 } else { 0.0 };

    // dim 10: cartão presente
    v[10] = if req.terminal.card_present { 1.0 } else { 0.0 };

    // dim 11: comerciante desconhecido (1 = NÃO está na lista — lógica invertida)
    let merchant_id: &str = &req.merchant.id;
    v[11] = if req
        .customer
        .known_merchants
        .iter()
        .any(|m| m.as_ref() == merchant_id)
    {
        0.0
    } else {
        1.0
    };

    // dim 12: risco do MCC
    v[12] = mcc_risk(&req.merchant.mcc);

    // dim 13: avg_amount do comerciante
    v[13] = (req.merchant.avg_amount as f32 / MAX_MERCHANT_AVG).clamp(0.0, 1.0);

    // dims 14–15: padding = 0.0
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::*;

    fn make_legit() -> FraudRequest {
        FraudRequest {
            transaction: Transaction {
                amount: 41.12,
                installments: 2,
                requested_at: "2026-03-11T18:45:53Z".into(),
            },
            customer: Customer {
                avg_amount: 82.24,
                tx_count_24h: 3,
                known_merchants: vec!["MERC-003".into(), "MERC-016".into()],
            },
            merchant: Merchant {
                id: "MERC-016".into(),
                mcc: "5411".into(),
                avg_amount: 60.25,
            },
            terminal: Terminal {
                is_online: false,
                card_present: true,
                km_from_home: 29.23,
            },
            last_transaction: None,
        }
    }

    #[test]
    fn test_vectorize_legit() {
        let req = make_legit();
        let v = vectorize(&req);
        let eps = 1e-3;

        // Valores esperados do DETECTION_RULES.md
        assert!((v[0] - 0.0041).abs() < eps, "dim0 amount: {}", v[0]);
        assert!((v[1] - 0.1667).abs() < eps, "dim1 installments: {}", v[1]);
        assert!((v[2] - 0.05).abs() < eps, "dim2 amount_vs_avg: {}", v[2]);
        assert!((v[3] - 0.7826).abs() < eps, "dim3 hour 18/23: {}", v[3]);
        // 2026-03-11 = quarta-feira = Mon=0 basis → wed=2 → 2/6=0.3333
        assert!((v[4] - 0.3333).abs() < eps, "dim4 dow: {}", v[4]);
        assert_eq!(v[5], -1.0, "dim5 sem last_tx");
        assert_eq!(v[6], -1.0, "dim6 sem last_tx");
        assert!((v[7] - 0.02923).abs() < eps, "dim7 km_home: {}", v[7]);
        assert!((v[8] - 0.15).abs() < eps, "dim8 tx_count: {}", v[8]);
        assert_eq!(v[9], 0.0, "dim9 is_online");
        assert_eq!(v[10], 1.0, "dim10 card_present");
        assert_eq!(v[11], 0.0, "dim11 known merchant");
        assert!((v[12] - 0.15).abs() < eps, "dim12 mcc 5411: {}", v[12]);
        assert!((v[13] - 0.006025).abs() < eps, "dim13 merchant_avg: {}", v[13]);
        assert_eq!(v[14], 0.0);
        assert_eq!(v[15], 0.0);
    }

    #[test]
    fn test_unknown_merchant() {
        let mut req = make_legit();
        req.merchant.id = "MERC-999".into();
        assert_eq!(vectorize(&req)[11], 1.0);
    }

    #[test]
    fn test_minutes_since_last_tx() {
        let mut req = make_legit();
        req.last_transaction = Some(LastTransaction {
            timestamp: "2026-03-11T12:45:53Z".into(), // 6h antes = 360 min
            km_from_current: 100.0,
        });
        let v = vectorize(&req);
        assert!((v[5] - 0.25).abs() < 1e-3, "dim5 360/1440: {}", v[5]);
        assert!((v[6] - 0.10).abs() < 1e-3, "dim6 100/1000: {}", v[6]);
    }

    #[test]
    fn test_mcc_default() {
        let mut req = make_legit();
        req.merchant.mcc = "9999".into();
        assert_eq!(vectorize(&req)[12], 0.5);
    }

    #[test]
    fn test_unix_epoch() {
        assert_eq!(to_unix(1970, 1, 1, 0, 0, 0), 0);
        assert_eq!(to_unix(1970, 1, 1, 0, 1, 0), 60);
        assert_eq!(to_unix(2026, 1, 1, 0, 0, 0), 1767225600);
    }

    #[test]
    fn test_day_of_week() {
        // 2026-03-11 = quarta-feira = 2 (Mon=0)
        assert_eq!(day_of_week_mon0(2026, 3, 11), 2);
        // 2026-01-01 = quinta-feira = 3
        assert_eq!(day_of_week_mon0(2026, 1, 1), 3);
        // 2026-03-15 = domingo = 6
        assert_eq!(day_of_week_mon0(2026, 3, 15), 6);
    }
}
