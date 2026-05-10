const MAX_AMOUNT: f32 = 10000.0;
const MAX_INSTALLMENTS: f32 = 12.0;
const AMOUNT_VS_AVG_RATIO: f32 = 10.0;
const MAX_MINUTES: f32 = 1440.0;
const MAX_KM: f32 = 1000.0;
const MAX_TX_COUNT_24H: f32 = 20.0;
const MAX_MERCHANT_AVG: f32 = 10000.0;

// ---------------------------------------------------------------------------
// Scanner zero-alloc
// Percorre o buffer JSON para frente, campo a campo, na ordem do schema.
// ---------------------------------------------------------------------------

#[inline(always)]
fn skip_ws(d: &[u8], p: &mut usize) {
    while *p < d.len() && matches!(d[*p], b' ' | b'\t' | b'\n' | b'\r') {
        *p += 1;
    }
}

/// Avança pos até encontrar `"key":` e posiciona logo após o `:`.
#[inline]
fn seek(d: &[u8], pos: &mut usize, key: &[u8]) -> bool {
    let klen = key.len();
    while *pos + klen + 2 <= d.len() {
        if d[*pos] == b'"'
            && d[*pos + 1..*pos + 1 + klen] == *key
            && d[*pos + 1 + klen] == b'"'
        {
            *pos += klen + 2;
            skip_ws(d, pos);
            if *pos < d.len() && d[*pos] == b':' {
                *pos += 1;
            }
            skip_ws(d, pos);
            return true;
        }
        *pos += 1;
    }
    false
}

/// Parseia um número f64 JSON (inteiro, decimal ou notação científica).
#[inline]
fn parse_f64(d: &[u8], pos: &mut usize) -> f64 {
    let start = *pos;
    if *pos < d.len() && d[*pos] == b'-' {
        *pos += 1;
    }
    while *pos < d.len() {
        match d[*pos] {
            b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-' => *pos += 1,
            _ => break,
        }
    }
    // SAFETY: JSON só tem ASCII em números
    unsafe { std::str::from_utf8_unchecked(&d[start..*pos]) }
        .parse::<f64>()
        .unwrap_or(0.0)
}

/// Parseia um inteiro u32 não-negativo.
#[inline]
fn parse_u32(d: &[u8], pos: &mut usize) -> u32 {
    let mut n = 0u32;
    while *pos < d.len() && d[*pos].is_ascii_digit() {
        n = n * 10 + (d[*pos] - b'0') as u32;
        *pos += 1;
    }
    n
}

/// Parseia `true` (4 bytes) ou `false` (5 bytes).
#[inline]
fn parse_bool(d: &[u8], pos: &mut usize) -> bool {
    if *pos + 4 <= d.len() && &d[*pos..*pos + 4] == b"true" {
        *pos += 4;
        true
    } else {
        *pos += 5;
        false
    }
}

/// Retorna o slice do conteúdo de uma string JSON (sem as aspas). Avança pos após o `"` final.
#[inline]
fn str_slice<'a>(d: &'a [u8], pos: &mut usize) -> &'a [u8] {
    if *pos < d.len() && d[*pos] == b'"' {
        *pos += 1;
    }
    let start = *pos;
    while *pos < d.len() && d[*pos] != b'"' {
        *pos += 1;
    }
    let s = &d[start..*pos];
    if *pos < d.len() {
        *pos += 1; // skip closing "
    }
    s
}

/// Retorna o conteúdo interno do array JSON `[...]` (sem os colchetes). Avança pos após `]`.
#[inline]
fn array_slice<'a>(d: &'a [u8], pos: &mut usize) -> &'a [u8] {
    if *pos < d.len() && d[*pos] == b'[' {
        *pos += 1;
    }
    let start = *pos;
    let mut depth = 1i32;
    while *pos < d.len() && depth > 0 {
        match d[*pos] {
            b'[' => depth += 1,
            b']' => depth -= 1,
            _ => {}
        }
        *pos += 1;
    }
    &d[start..*pos - 1]
}

// ---------------------------------------------------------------------------
// MCC risk (bytes)
// ---------------------------------------------------------------------------

#[inline(always)]
fn mcc_risk_bytes(mcc: &[u8]) -> f32 {
    match mcc {
        b"5411" => 0.15,
        b"5812" => 0.30,
        b"5912" => 0.20,
        b"5944" => 0.45,
        b"7801" => 0.80,
        b"7802" => 0.75,
        b"7995" => 0.85,
        b"4511" => 0.35,
        b"5311" => 0.25,
        b"5999" => 0.50,
        _ => 0.50,
    }
}

// ---------------------------------------------------------------------------
// Parse manual de timestamps ISO 8601 UTC — "YYYY-MM-DDTHH:MM:SSZ"
// ---------------------------------------------------------------------------

#[inline(always)]
fn db(b: u8) -> u32 {
    (b - b'0') as u32
}

#[inline]
fn parse_ts_bytes(b: &[u8]) -> (u32, u32, i64) {
    let year  = db(b[0]) * 1000 + db(b[1]) * 100 + db(b[2]) * 10 + db(b[3]);
    let month = db(b[5]) * 10 + db(b[6]);
    let day   = db(b[8]) * 10 + db(b[9]);
    let hour  = db(b[11]) * 10 + db(b[12]);
    let min   = db(b[14]) * 10 + db(b[15]);
    let sec   = db(b[17]) * 10 + db(b[18]);
    let dow = day_of_week_mon0(year, month, day);
    let unix = to_unix(year, month, day, hour, min, sec);
    (hour, dow, unix)
}

#[inline]
fn day_of_week_mon0(year: u32, month: u32, day: u32) -> u32 {
    const T: [u32; 12] = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let y = if month < 3 { year - 1 } else { year };
    let dow_sun = (y + y / 4 - y / 100 + y / 400 + T[(month - 1) as usize] + day) % 7;
    (dow_sun + 6) % 7
}

#[inline]
fn to_unix(year: u32, month: u32, day: u32, hour: u32, min: u32, sec: u32) -> i64 {
    let y = year as i64;
    let m = month as i64;
    let d = day as i64;
    let (y, m) = if m <= 2 { (y - 1, m + 9) } else { (y, m - 3) };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * m + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    days * 86400 + hour as i64 * 3600 + min as i64 * 60 + sec as i64
}

// ---------------------------------------------------------------------------
// Vectorização principal — lê diretamente do buffer JSON
// Assume campos na ordem do schema: transaction → customer → merchant → terminal → last_transaction
// ---------------------------------------------------------------------------

/// Transforma o body JSON cru em vetor [f32; 16]: 14 dimensões + 2 zeros de padding SIMD.
pub fn vectorize_raw(body: &[u8]) -> [f32; 16] {
    let mut v = [0.0f32; 16];
    let mut pos = 0usize;

    // dim 0: transaction.amount
    seek(body, &mut pos, b"amount");
    let amount = parse_f64(body, &mut pos) as f32;
    v[0] = (amount / MAX_AMOUNT).clamp(0.0, 1.0);

    // dim 1: transaction.installments
    seek(body, &mut pos, b"installments");
    v[1] = (parse_u32(body, &mut pos) as f32 / MAX_INSTALLMENTS).clamp(0.0, 1.0);

    // dims 3/4 + base para dims 5/6: transaction.requested_at
    seek(body, &mut pos, b"requested_at");
    let ts = str_slice(body, &mut pos);
    let (hour, dow, cur_unix) = parse_ts_bytes(ts);
    v[3] = hour as f32 / 23.0;
    v[4] = dow as f32 / 6.0;

    // dim 2: razão amount vs avg_amount do cliente
    seek(body, &mut pos, b"avg_amount");
    let avg_amount = parse_f64(body, &mut pos) as f32;
    v[2] = if avg_amount > 0.0 {
        (amount / (avg_amount * AMOUNT_VS_AVG_RATIO)).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // dim 8: tx_count_24h
    seek(body, &mut pos, b"tx_count_24h");
    v[8] = (parse_u32(body, &mut pos) as f32 / MAX_TX_COUNT_24H).clamp(0.0, 1.0);

    // dim 11: guarda slice do array known_merchants para checar depois
    seek(body, &mut pos, b"known_merchants");
    let merchants_arr = array_slice(body, &mut pos);

    // merchant.id
    seek(body, &mut pos, b"id");
    let merchant_id = str_slice(body, &mut pos);

    // dim 11: comerciante desconhecido (1.0) vs conhecido (0.0)
    let mlen = merchant_id.len();
    v[11] = if mlen > 0
        && merchants_arr.windows(mlen + 2).any(|w| {
            w[0] == b'"' && w[mlen + 1] == b'"' && &w[1..mlen + 1] == merchant_id
        })
    {
        0.0
    } else {
        1.0
    };

    // dim 12: risco do MCC
    seek(body, &mut pos, b"mcc");
    let mcc = str_slice(body, &mut pos);
    v[12] = mcc_risk_bytes(mcc);

    // dim 13: avg_amount do comerciante (segunda ocorrência de "avg_amount")
    seek(body, &mut pos, b"avg_amount");
    v[13] = (parse_f64(body, &mut pos) as f32 / MAX_MERCHANT_AVG).clamp(0.0, 1.0);

    // dim 9: terminal.is_online
    seek(body, &mut pos, b"is_online");
    v[9] = if parse_bool(body, &mut pos) { 1.0 } else { 0.0 };

    // dim 10: terminal.card_present
    seek(body, &mut pos, b"card_present");
    v[10] = if parse_bool(body, &mut pos) { 1.0 } else { 0.0 };

    // dim 7: terminal.km_from_home
    seek(body, &mut pos, b"km_from_home");
    v[7] = (parse_f64(body, &mut pos) as f32 / MAX_KM).clamp(0.0, 1.0);

    // dims 5/6: last_transaction (opcional — null ou objeto)
    if seek(body, &mut pos, b"last_transaction")
        && pos < body.len()
        && body[pos] != b'n'
    {
        seek(body, &mut pos, b"timestamp");
        let last_ts = str_slice(body, &mut pos);
        let (_, _, last_unix) = parse_ts_bytes(last_ts);
        let delta_min = ((cur_unix - last_unix).max(0) / 60) as f32;
        v[5] = (delta_min / MAX_MINUTES).clamp(0.0, 1.0);

        seek(body, &mut pos, b"km_from_current");
        v[6] = (parse_f64(body, &mut pos) as f32 / MAX_KM).clamp(0.0, 1.0);
    } else {
        v[5] = -1.0;
        v[6] = -1.0;
    }

    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legit_json() -> &'static [u8] {
        br#"{"transaction":{"amount":41.12,"installments":2,"requested_at":"2026-03-11T18:45:53Z"},"customer":{"avg_amount":82.24,"tx_count_24h":3,"known_merchants":["MERC-003","MERC-016"]},"merchant":{"id":"MERC-016","mcc":"5411","avg_amount":60.25},"terminal":{"is_online":false,"card_present":true,"km_from_home":29.23},"last_transaction":null}"#
    }

    #[test]
    fn test_vectorize_legit() {
        let v = vectorize_raw(legit_json());
        let eps = 1e-3;

        assert!((v[0]  - 0.0041  ).abs() < eps, "dim0 amount: {}",       v[0]);
        assert!((v[1]  - 0.1667  ).abs() < eps, "dim1 installments: {}",  v[1]);
        assert!((v[2]  - 0.05    ).abs() < eps, "dim2 amount_vs_avg: {}", v[2]);
        assert!((v[3]  - 0.7826  ).abs() < eps, "dim3 hour 18/23: {}",    v[3]);
        assert!((v[4]  - 0.3333  ).abs() < eps, "dim4 dow: {}",           v[4]);
        assert_eq!(v[5], -1.0,                   "dim5 sem last_tx");
        assert_eq!(v[6], -1.0,                   "dim6 sem last_tx");
        assert!((v[7]  - 0.02923 ).abs() < eps, "dim7 km_home: {}",       v[7]);
        assert!((v[8]  - 0.15    ).abs() < eps, "dim8 tx_count: {}",      v[8]);
        assert_eq!(v[9],  0.0,                   "dim9 is_online");
        assert_eq!(v[10], 1.0,                   "dim10 card_present");
        assert_eq!(v[11], 0.0,                   "dim11 known merchant");
        assert!((v[12] - 0.15    ).abs() < eps, "dim12 mcc 5411: {}",     v[12]);
        assert!((v[13] - 0.006025).abs() < eps, "dim13 merchant_avg: {}",  v[13]);
        assert_eq!(v[14], 0.0);
        assert_eq!(v[15], 0.0);
    }

    #[test]
    fn test_unknown_merchant() {
        let json = br#"{"transaction":{"amount":41.12,"installments":2,"requested_at":"2026-03-11T18:45:53Z"},"customer":{"avg_amount":82.24,"tx_count_24h":3,"known_merchants":["MERC-003","MERC-016"]},"merchant":{"id":"MERC-999","mcc":"5411","avg_amount":60.25},"terminal":{"is_online":false,"card_present":true,"km_from_home":29.23},"last_transaction":null}"#;
        assert_eq!(vectorize_raw(json)[11], 1.0);
    }

    #[test]
    fn test_minutes_since_last_tx() {
        let json = br#"{"transaction":{"amount":41.12,"installments":2,"requested_at":"2026-03-11T18:45:53Z"},"customer":{"avg_amount":82.24,"tx_count_24h":3,"known_merchants":[]},"merchant":{"id":"MERC-016","mcc":"5411","avg_amount":60.25},"terminal":{"is_online":false,"card_present":true,"km_from_home":29.23},"last_transaction":{"timestamp":"2026-03-11T12:45:53Z","km_from_current":100.0}}"#;
        let v = vectorize_raw(json);
        assert!((v[5] - 0.25).abs() < 1e-3, "dim5 360/1440: {}", v[5]);
        assert!((v[6] - 0.10).abs() < 1e-3, "dim6 100/1000: {}", v[6]);
    }

    #[test]
    fn test_mcc_default() {
        let json = br#"{"transaction":{"amount":41.12,"installments":2,"requested_at":"2026-03-11T18:45:53Z"},"customer":{"avg_amount":82.24,"tx_count_24h":3,"known_merchants":[]},"merchant":{"id":"MERC-016","mcc":"9999","avg_amount":60.25},"terminal":{"is_online":false,"card_present":true,"km_from_home":29.23},"last_transaction":null}"#;
        assert_eq!(vectorize_raw(json)[12], 0.5);
    }

    #[test]
    fn test_unix_epoch() {
        assert_eq!(to_unix(1970, 1, 1, 0, 0, 0), 0);
        assert_eq!(to_unix(1970, 1, 1, 0, 1, 0), 60);
        assert_eq!(to_unix(2026, 1, 1, 0, 0, 0), 1767225600);
    }

    #[test]
    fn test_day_of_week() {
        assert_eq!(day_of_week_mon0(2026, 3, 11), 2); // quarta = 2
        assert_eq!(day_of_week_mon0(2026, 1, 1),  3); // quinta = 3
        assert_eq!(day_of_week_mon0(2026, 3, 15), 6); // domingo = 6
    }
}
