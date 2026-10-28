//! E-step kernels for joli.joint.JointEM over one library in flat layout:
//! equivalence class e occupies positions ec_ptr[e]..ec_ptr[e+1] of the
//! arrays tx (transcript index) and c (compatibility weight c_et); counts[e]
//! is its read count and mu (n_tx) the library's molar mixture.
//!
//! denom_e = sum_{p in e} mu[tx_p] c_p. ECs with denom_e <= f64::MIN_POSITIVE
//! are skipped (zero likelihood contribution), as in the numpy reference.
//!
//! Threads take contiguous blocks of ECs of about equal numbers of positions.

use std::slice;

const TINY: f64 = f64::MIN_POSITIVE;

/// Minimum positions per thread; smaller libraries run on fewer threads.
const MIN_POS_PER_THREAD: usize = 100_000;

/// EC block boundaries giving each of up to n_threads about the same number of positions.
fn blocks(ec_ptr: &[i64], n_threads: usize) -> Vec<usize> {
    let n_ec = ec_ptr.len() - 1;
    let total = ec_ptr[n_ec] as f64;
    let n_threads = n_threads.min((ec_ptr[n_ec] as usize / MIN_POS_PER_THREAD).max(1));
    let mut b = vec![0usize];
    for k in 1..n_threads {
        let target = (total * k as f64 / n_threads as f64) as i64;
        let e = ec_ptr.partition_point(|&x| x < target).min(n_ec);
        if e > *b.last().unwrap() {
            b.push(e);
        }
    }
    b.push(n_ec);
    b
}

#[inline]
fn denom(e: usize, ec_ptr: &[i64], tx: &[i64], c: &[f64], mu: &[f64]) -> f64 {
    let (lo, hi) = (ec_ptr[e] as usize, ec_ptr[e + 1] as usize);
    let mut d = 0.0;
    for p in lo..hi {
        d += mu[tx[p] as usize] * c[p];
    }
    d
}

/// g[t] = sum_e counts[e] c_et / denom_e over transcripts t (written to g_out, length n_tx);
/// returns sum_e counts[e] log denom_e.
#[no_mangle]
pub unsafe extern "C" fn joli_estep(
    n_ec: usize,
    ec_ptr: *const i64,
    tx: *const i64,
    c: *const f64,
    counts: *const f64,
    mu: *const f64,
    n_tx: usize,
    g_out: *mut f64,
    n_threads: usize,
) -> f64 {
    let ec_ptr = slice::from_raw_parts(ec_ptr, n_ec + 1);
    let n_pos = ec_ptr[n_ec] as usize;
    let tx = slice::from_raw_parts(tx, n_pos);
    let c = slice::from_raw_parts(c, n_pos);
    let counts = slice::from_raw_parts(counts, n_ec);
    let mu = slice::from_raw_parts(mu, n_tx);
    let g_out = slice::from_raw_parts_mut(g_out, n_tx);
    let b = blocks(ec_ptr, n_threads.max(1));

    let work = |e0: usize, e1: usize, g: &mut [f64]| -> f64 {
        let mut ll = 0.0;
        for e in e0..e1 {
            let d = denom(e, ec_ptr, tx, c, mu);
            if d > TINY {
                let n = counts[e];
                ll += n * d.ln();
                let r = n / d;
                for p in ec_ptr[e] as usize..ec_ptr[e + 1] as usize {
                    g[tx[p] as usize] += r * c[p];
                }
            }
        }
        ll
    };

    g_out.iter_mut().for_each(|x| *x = 0.0);
    if b.len() <= 2 {
        return work(b[0], b[1], g_out);
    }
    let (lls, gs): (Vec<f64>, Vec<Vec<f64>>) = std::thread::scope(|s| {
        let handles: Vec<_> = b
            .windows(2)
            .map(|w| {
                let (e0, e1) = (w[0], w[1]);
                s.spawn(move || {
                    let mut g = vec![0.0; n_tx];
                    let ll = work(e0, e1, &mut g);
                    (ll, g)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).unzip()
    });
    for g in &gs {
        for (o, v) in g_out.iter_mut().zip(g) {
            *o += v;
        }
    }
    lls.iter().sum()
}

/// z_p = counts[e] mu[tx_p] c_p / denom_e for every position p (z_out, length n_pos).
#[no_mangle]
pub unsafe extern "C" fn joli_position_counts(
    n_ec: usize,
    ec_ptr: *const i64,
    tx: *const i64,
    c: *const f64,
    counts: *const f64,
    mu: *const f64,
    n_tx: usize,
    z_out: *mut f64,
    n_threads: usize,
) {
    let ec_ptr = slice::from_raw_parts(ec_ptr, n_ec + 1);
    let n_pos = ec_ptr[n_ec] as usize;
    let tx = slice::from_raw_parts(tx, n_pos);
    let c = slice::from_raw_parts(c, n_pos);
    let counts = slice::from_raw_parts(counts, n_ec);
    let mu = slice::from_raw_parts(mu, n_tx);
    let z = slice::from_raw_parts_mut(z_out, n_pos);
    let b = blocks(ec_ptr, n_threads.max(1));

    let fill = |e0: usize, e1: usize, zpart: &mut [f64]| {
        let off = ec_ptr[e0] as usize;
        for e in e0..e1 {
            let d = denom(e, ec_ptr, tx, c, mu);
            let (lo, hi) = (ec_ptr[e] as usize, ec_ptr[e + 1] as usize);
            if d > TINY {
                let r = counts[e] / d;
                for p in lo..hi {
                    zpart[p - off] = r * mu[tx[p] as usize] * c[p];
                }
            } else {
                for p in lo..hi {
                    zpart[p - off] = 0.0;
                }
            }
        }
    };
    // split z at block boundaries so each thread owns its positions
    let mut rest: &mut [f64] = z;
    let mut parts = Vec::new();
    let mut consumed = 0usize;
    for w in b.windows(2) {
        let len = ec_ptr[w[1]] as usize - consumed;
        let (head, tail) = rest.split_at_mut(len);
        parts.push((w[0], w[1], head));
        rest = tail;
        consumed += len;
    }
    if parts.len() == 1 {
        let (e0, e1, zpart) = parts.pop().unwrap();
        fill(e0, e1, zpart);
        return;
    }
    std::thread::scope(|s| {
        for (e0, e1, zpart) in parts {
            s.spawn(move || fill(e0, e1, zpart));
        }
    });
}

/// Multi-threaded E-step without per-thread accumulators. Pass 1 (over EC
/// blocks) writes r_e = counts[e] / denom_e (0 if denom_e is not positive)
/// to ratio_buf (n_ec) and sums counts[e] log denom_e. Pass 2 (over
/// transcript blocks) gathers g[t] = sum_{p of t} r_{ec(p)} c_p through the
/// transcript-sorted index: positions perm[tx_ptr[t]..tx_ptr[t+1]] belong to
/// transcript t and pos_ec_sorted[i] = EC of position perm[i].
#[no_mangle]
pub unsafe extern "C" fn joli_estep_gather(
    n_ec: usize,
    ec_ptr: *const i64,
    tx: *const i64,
    c: *const f64,
    counts: *const f64,
    mu: *const f64,
    n_tx: usize,
    tx_ptr: *const i64,
    perm: *const i64,
    pos_ec_sorted: *const i64,
    ratio_buf: *mut f64,
    g_out: *mut f64,
    n_threads: usize,
) -> f64 {
    let ec_ptr = slice::from_raw_parts(ec_ptr, n_ec + 1);
    let n_pos = ec_ptr[n_ec] as usize;
    let tx = slice::from_raw_parts(tx, n_pos);
    let c = slice::from_raw_parts(c, n_pos);
    let counts = slice::from_raw_parts(counts, n_ec);
    let mu = slice::from_raw_parts(mu, n_tx);
    let tx_ptr = slice::from_raw_parts(tx_ptr, n_tx + 1);
    let perm = slice::from_raw_parts(perm, n_pos);
    let pos_ec_sorted = slice::from_raw_parts(pos_ec_sorted, n_pos);
    let ratio = slice::from_raw_parts_mut(ratio_buf, n_ec);
    let g = slice::from_raw_parts_mut(g_out, n_tx);

    // pass 1: ratios per EC
    let b = blocks(ec_ptr, n_threads.max(1));
    let mut parts = Vec::new();
    let mut rest: &mut [f64] = ratio;
    for w in b.windows(2) {
        let (head, tail) = rest.split_at_mut(w[1] - w[0]);
        parts.push((w[0], head));
        rest = tail;
    }
    let ll: f64 = std::thread::scope(|s| {
        let hs: Vec<_> = parts
            .into_iter()
            .map(|(e0, rpart)| {
                s.spawn(move || {
                    let mut ll = 0.0;
                    for (i, r) in rpart.iter_mut().enumerate() {
                        let e = e0 + i;
                        let d = denom(e, ec_ptr, tx, c, mu);
                        if d > TINY {
                            ll += counts[e] * d.ln();
                            *r = counts[e] / d;
                        } else {
                            *r = 0.0;
                        }
                    }
                    ll
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).sum()
    });

    // pass 2: gather per transcript
    let ratio: &[f64] = ratio;
    let bt = blocks(tx_ptr, n_threads.max(1));
    let mut gparts = Vec::new();
    let mut grest: &mut [f64] = g;
    for w in bt.windows(2) {
        let (head, tail) = grest.split_at_mut(w[1] - w[0]);
        gparts.push((w[0], head));
        grest = tail;
    }
    std::thread::scope(|s| {
        for (t0, gpart) in gparts {
            s.spawn(move || {
                for (i, gt) in gpart.iter_mut().enumerate() {
                    let t = t0 + i;
                    let mut acc = 0.0;
                    for k in tx_ptr[t] as usize..tx_ptr[t + 1] as usize {
                        acc += ratio[pos_ec_sorted[k] as usize] * c[perm[k] as usize];
                    }
                    *gt = acc;
                }
            });
        }
    });
    ll
}
