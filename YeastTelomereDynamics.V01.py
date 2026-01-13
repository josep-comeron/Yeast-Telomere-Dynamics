# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
YeastTelomereDynamics — Numba
Version: V01
Requires numba>=0.60 and numpy>=1.26.
"""

from numba import njit
import numpy as np, math

VERSION = "V01"
PROG = f"YeastTelomereDynamics.{VERSION}.py"
UINT16_MAX = 65535
N_ENDS = 32

@njit(cache=True)
def _rng_step(state):
    s0 = state[0]
    s1 = state[1]
    result = (s0 + s1) & np.uint64(0xFFFFFFFFFFFFFFFF)
    s1 ^= s0
    state[0] = ((s0 << np.uint64(55)) | (s0 >> np.uint64(9))) ^ s1 ^ (s1 << np.uint64(14))
    state[1] = (s1 << np.uint64(36)) | (s1 >> np.uint64(28))
    return result

@njit(cache=True)
def rng_int(state, high:int)->int:
    if high <= 0: return 0
    return int(_rng_step(state) % np.uint64(high))

@njit(cache=True)
def rng_uniform01(state)->float:
    return rng_int(state, 1<<24) / float(1<<24)

@njit(cache=True)
def poisson_knuth(lam: float, state) -> int:
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        u = rng_uniform01(state)
        p *= u
    return k - 1

def make_state_from_py_rng(py_seed: int)->np.ndarray:
    import numpy as _np
    rng = _np.random.default_rng(py_seed)
    s = rng.integers(0, (1<<63)-1, size=2, dtype=_np.int64).astype(_np.uint64)
    return s

@njit(cache=True)
def apply_deletions(parent_row_tel, child_row_tel, del_rate_user, state):
    # user-visible del_rate is per telomere; kernel applies to half the ends on average
    lam = 2.0 * del_rate_user
    for i in range(N_ENDS):
        if rng_int(state, 2) == 1:
            k = poisson_knuth(lam, state)
            v = int(parent_row_tel[i]) - k
            if v < 0: v = 0
            parent_row_tel[i] = np.uint16(v)
        if rng_int(state, 2) == 1:
            k = poisson_knuth(lam, state)
            v = int(child_row_tel[i]) - k
            if v < 0: v = 0
            child_row_tel[i] = np.uint16(v)

@njit(cache=True)
def pd_step_pre_symm_both_numba(tel, y, Ls, del_rate_user, state):
    N = tel.shape[0]
    ncols = tel.shape[1]
    next_tel = np.empty((2*N, ncols), dtype=np.uint16)
    next_y   = np.empty((2*N, ncols), dtype=np.uint16)
    for i in range(N):
        for j in range(ncols):
            next_tel[i, j] = tel[i, j]
            next_y[i, j]   = y[i, j]
    pool = N
    while pool < 2*N:
        p = rng_int(state, pool)
        ip = int(p)
        # mutate parent symmetrically
        for j in range(ncols):
            k = poisson_knuth(del_rate_user, state)
            sign = 1 if rng_int(state, 2)==1 else -1
            v = int(next_tel[ip, j]) + sign*k
            if v < Ls: v = Ls
            if v > UINT16_MAX: v = UINT16_MAX
            next_tel[ip, j] = np.uint16(v)
        # copy to child
        for j in range(ncols):
            next_tel[pool, j] = next_tel[ip, j]
            next_y  [pool, j] = next_y  [ip, j]
        # mutate child symmetrically
        for j in range(ncols):
            k = poisson_knuth(del_rate_user, state)
            sign = 1 if rng_int(state, 2)==1 else -1
            v = int(next_tel[pool, j]) + sign*k
            if v < Ls: v = Ls
            if v > UINT16_MAX: v = UINT16_MAX
            next_tel[pool, j] = np.uint16(v)
        pool += 1
    out_tel = np.empty((pool, ncols), dtype=np.uint16)
    out_y   = np.empty((pool, ncols), dtype=np.uint16)
    for i in range(pool):
        for j in range(ncols):
            out_tel[i, j] = next_tel[i, j]
            out_y  [i, j] = next_y  [i, j]
    return pool, out_tel, out_y

@njit(cache=True)
def pick_uniform_eligible(row_tel, row_y, Ls, eff_model, r, require_y_ge1, state)->int:
    n = row_tel.shape[0]
    cnt = 0
    for i in range(n):
        if i == r: continue
        if eff_model == 1 and row_tel[i] < Ls: continue
        if require_y_ge1 == 1 and row_y[i] == 0: continue
        cnt += 1
    if cnt == 0: return -1
    step = rng_int(state, cnt)
    acc = 0
    for i in range(n):
        if i == r: continue
        if eff_model == 1 and row_tel[i] < Ls: continue
        if require_y_ge1 == 1 and row_y[i] == 0: continue
        if acc == step: return i
        acc += 1
    return -1

@njit(cache=True)
def pick_weighted_donor(row_tel, Ls, eff_model, r, state)->int:
    n = row_tel.shape[0]
    tot = 0
    for i in range(n):
        if i == r: continue
        v = int(row_tel[i])
        if eff_model == 1 and v < Ls: v = 0
        if v > 0: tot += v
    if tot <= 0:
        z = np.zeros(n, dtype=np.uint16)
        return pick_uniform_eligible(row_tel, z, Ls, eff_model, r, 0, state)
    rint = rng_int(state, tot)
    acc = 0
    for i in range(n):
        if i == r: continue
        v = int(row_tel[i])
        if eff_model == 1 and v < Ls: v = 0
        if v <= 0: continue
        acc += v
        if rint < acc: return i
    return -1

@njit(cache=True)
def pick_max_donor(row_tel, Ls, eff_model, r)->int:
    n = row_tel.shape[0]
    best = -1
    bestv = -1
    for i in range(n):
        if i == r: continue
        v = int(row_tel[i])
        if eff_model == 1 and v < Ls: continue
        if v > bestv:
            bestv = v; best = i
    return best

@njit(cache=True)
def pick_y_donor_weighted_by_count(row_tel, row_y, Ls, eff_model, r, state):
    n = row_tel.shape[0]
    tot = 0
    for i in range(n):
        if i == r: continue
        if eff_model == 1 and row_tel[i] < Ls: continue
        yi = int(row_y[i])
        if yi > 0: tot += yi
    if tot <= 0: return -1, -1
    rint = rng_int(state, tot)
    acc = 0
    pick = -1
    for i in range(n):
        if i == r: continue
        if eff_model == 1 and row_tel[i] < Ls: continue
        yi = int(row_y[i])
        if yi <= 0: continue
        acc += yi
        if rint < acc:
            pick = i; break
    if pick < 0: return -1, -1
    yi = int(row_y[pick])
    dj1 = rng_int(state, yi) + 1
    return pick, dj1

@njit(cache=True)
def recombine_cell(row_tel, row_y, Ls, rec_model, rec_y_weighted, donor_mode,
                   rec_tel_mode, prob_circle_cell, circle_len, prob_ts,
                   min_len_circle_gen, prob_each_circle, dynamic_circles, p_circle_row,
                   max_Ys, state):
    if rec_model == 0:
        return
    n = row_tel.shape[0]
    eff_model = 1 if rec_model == 1 else 0  # 1: donors must be >= Ls; 0: any
    if eff_model == 1:
        any_ge = False
        for i in range(n):
            if row_tel[i] >= Ls:
                any_ge = True; break
        if not any_ge:
            return

    for r in range(n):
        if row_tel[r] >= Ls:
            continue
        ycnt = int(row_y[r])
        total = 3 + ycnt
        k = rng_int(state, total)
        if k == 0:
            # Telomere recombination; first consider circles
            if prob_circle_cell > 0.0 and rng_uniform01(state) < prob_circle_cell:
                new_tel = int(row_tel[r]) + int(circle_len)
                if new_tel > UINT16_MAX: new_tel = UINT16_MAX
                row_tel[r] = np.uint16(new_tel)
                if new_tel >= min_len_circle_gen and dynamic_circles == 1:
                    p = float(p_circle_row[0]) + float(prob_each_circle)
                    if p > 1.0: p = 1.0
                    p_circle_row[0] = p
                continue
            # otherwise choose donor per donor_mode
            if donor_mode == 2:
                d = pick_max_donor(row_tel, Ls, eff_model, r)
            elif donor_mode == 1:
                d = pick_weighted_donor(row_tel, Ls, eff_model, r, state)
            else:
                z = np.zeros(n, dtype=np.uint16)
                d = pick_uniform_eligible(row_tel, z, Ls, eff_model, r, 0, state)
            if d < 0: continue
            TelR_old = int(row_tel[r]); TelD = int(row_tel[d])
            if rec_tel_mode == 0:      # copy
                rR = 0; rD = 0
            elif rec_tel_mode == 2:    # end
                rR = TelR_old; rD = rng_int(state, TelD+1)
            else:                      # rnd
                rR = rng_int(state, TelR_old+1); rD = rng_int(state, TelD+1)
            new_tel = rR + (TelD - rD)
            if new_tel < 0: new_tel = 0
            if new_tel > UINT16_MAX: new_tel = UINT16_MAX
            row_tel[r] = np.uint16(new_tel)
            if new_tel >= min_len_circle_gen and dynamic_circles == 1:
                p = float(p_circle_row[0]) + float(prob_each_circle)
                if p > 1.0: p = 1.0
                p_circle_row[0] = p
            # template switching
            if prob_ts > 0.0:
                last_d = d
                for _ in range(5):
                    if rng_uniform01(state) >= prob_ts: break
                    cnt = 0
                    for i in range(n):
                        if i == r or i == last_d: continue
                        if eff_model == 1 and row_tel[i] < Ls: continue
                        cnt += 1
                    if cnt == 0: break
                    step = rng_int(state, cnt)
                    acc = 0; nd = -1
                    for i in range(n):
                        if i == r or i == last_d: continue
                        if eff_model == 1 and row_tel[i] < Ls: continue
                        if acc == step: nd = i; break
                        acc += 1
                    if nd < 0: break
                    TelR_old = int(row_tel[r]); TelD = int(row_tel[nd])
                    rR = TelR_old; rD = rng_int(state, TelD+1)   # TS always 'end' on receptor
                    new_tel = rR + (TelD - rD)
                    if new_tel < 0: new_tel = 0
                    if new_tel > UINT16_MAX: new_tel = UINT16_MAX
                    row_tel[r] = np.uint16(new_tel)
                    if new_tel >= min_len_circle_gen and dynamic_circles == 1:
                        p = float(p_circle_row[0]) + float(prob_each_circle)
                        if p > 1.0: p = 1.0
                        p_circle_row[0] = p
                    last_d = nd
        elif k == 1:
            # X: copy tel length only
            d = pick_uniform_eligible(row_tel, row_y, Ls, eff_model, r, 0, state)
            if d >= 0:
                row_tel[r] = row_tel[d]
        elif k == 2:
            # C: copy tel length and Y-count
            d = pick_uniform_eligible(row_tel, row_y, Ls, eff_model, r, 0, state)
            if d >= 0:
                row_tel[r] = row_tel[d]
                yy = int(row_y[d])
                if yy < 0: yy = 0
                if yy > max_Ys: yy = max_Ys
                row_y[r]   = np.uint16(yy)
        else:
            # Y recombination
            if ycnt <= 0: 
                continue
            rec_i = k - 2
            if rec_y_weighted == 1:
                d, dj1 = pick_y_donor_weighted_by_count(row_tel, row_y, Ls, eff_model, r, state)
                if d >= 0:
                    donor_y = int(row_y[d])
                    new_y = rec_i + (donor_y - int(dj1))
                    if new_y < 0: new_y = 0
                    if new_y > max_Ys: new_y = max_Ys
                    row_y[r] = np.uint16(new_y)
                    row_tel[r] = row_tel[d]
            else:
                d = pick_uniform_eligible(row_tel, row_y, Ls, eff_model, r, 1, state)
                if d >= 0 and row_y[d] >= 1:
                    donor_y = int(row_y[d])
                    dj0 = rng_int(state, donor_y); dj1 = dj0 + 1
                    new_y = rec_i + (donor_y - int(dj1))
                    if new_y < 0: new_y = 0
                    if new_y > max_Ys: new_y = max_Ys
                    row_y[r] = np.uint16(new_y)
                    row_tel[r] = row_tel[d]



@njit(cache=True)
def pd_step_numba(tel, y, p_circle, Ls, del_rate_user,
                  rec_model, rec_y_weighted, donor_mode,
                  rec_tel_mode, prob_circle_global, circle_len,
                  prob_ts, p_sen_death, min_len_circle_gen, prob_each_circle,
                  dynamic_circles, max_Ys, state):
    N = tel.shape[0]
    ncols = tel.shape[1]

    if N == 0:
        return N, tel, y, p_circle

    # Preallocate next generation buffers
    next_tel = np.empty((2*N, ncols), dtype=np.uint16)
    next_y   = np.empty((2*N, ncols), dtype=np.uint16)
    next_pc  = np.empty((2*N,), dtype=np.float32)

    # Copy current pool
    for i in range(N):
        for j in range(ncols):
            next_tel[i, j] = tel[i, j]
            next_y  [i, j] = y[i, j]
        next_pc[i] = p_circle[i]

    pool = int(N)

    # ===== FAST PATH: NO RECOMBINATION ======================================
    if rec_model == 0:
        # Build initial list of dividers in [0 .. pool-1]
        div_idx = np.empty(pool, dtype=np.int32)
        div_count = 0
        for i in range(pool):
            ok = True
            for j in range(ncols):
                if next_tel[i, j] < Ls:
                    ok = False
                    break
            if ok:
                div_idx[div_count] = i
                div_count += 1

        if div_count == 0:
            # No dividers -> PD ends immediately
            out_tel = np.empty((pool, ncols), dtype=np.uint16)
            out_y   = np.empty((pool, ncols), dtype=np.uint16)
            out_pc  = np.empty((pool,), dtype=np.float32)
            for i in range(pool):
                for j in range(ncols):
                    out_tel[i, j] = next_tel[i, j]
                    out_y  [i, j] = next_y  [i, j]
                out_pc[i] = next_pc[i]
            return pool, out_tel, out_y, out_pc

        while pool < 2*N:
            if div_count == 0:
                break  # cannot create more children in this PD

            # Pick a divider parent uniformly
            k  = int(rng_int(state, div_count))
            ip = int(div_idx[k])

            # Child starts as a copy of parent
            for j in range(ncols):
                next_tel[pool, j] = next_tel[ip, j]
                next_y  [pool, j] = next_y  [ip, j]
            # Split circles between parent and child
            next_pc[pool] = next_pc[ip] * 0.5
            next_pc[ip]   *= 0.5

            # Apply deletions to both parent (in-place) and child
            apply_deletions(next_tel[ip], next_tel[pool], del_rate_user, state)

            # Update the divider list for the parent
            parent_div = True
            for j in range(ncols):
                if next_tel[ip, j] < Ls:
                    parent_div = False
                    break
            if not parent_div:
                last = div_count - 1
                if k != last:
                    div_idx[k] = div_idx[last]
                div_count -= 1

            # If child is a divider, append to list
            child_div = True
            for j in range(ncols):
                if next_tel[pool, j] < Ls:
                    child_div = False
                    break
            if child_div:
                if div_count < div_idx.shape[0]:
                    div_idx[div_count] = int(pool)
                    div_count += 1

            pool += 1

        # Finalize
        out_tel = np.empty((pool, ncols), dtype=np.uint16)
        out_y   = np.empty((pool, ncols), dtype=np.uint16)
        out_pc  = np.empty((pool,), dtype=np.float32)
        for i in range(pool):
            for j in range(ncols):
                out_tel[i, j] = next_tel[i, j]
                out_y  [i, j] = next_y  [i, j]
            out_pc[i] = next_pc[i]
        return pool, out_tel, out_y, out_pc

    # ===== GENERAL PATH: RECOMBINATION ALLOWED ===============================
    while pool < 2*N:
        if pool == 0:
            break

        ip = int(rng_int(state, pool))
        ip_i = int(ip)

        # Is parent senescent?
        can_divide = True
        for j in range(ncols):
            if next_tel[ip_i, j] < Ls:
                can_divide = False
                break

        if not can_divide:
            # Optional death of senescent parent
            if p_sen_death > 0.0 and rng_uniform01(state) < p_sen_death:
                last = pool - 1
                for j in range(ncols):
                    tmp  = next_tel[ip_i, j]; next_tel[ip_i, j] = next_tel[last, j]; next_tel[last, j] = tmp
                    tmp2 = next_y  [ip_i, j]; next_y  [ip_i, j] = next_y  [last, j]; next_y  [last, j] = tmp2
                tmp3 = next_pc[ip_i]; next_pc[ip_i] = next_pc[last]; next_pc[last] = tmp3
                pool -= 1
                continue

            # Try to rescue by recombination
            pc = next_pc[ip_i] if dynamic_circles == 1 else prob_circle_global
            recombine_cell(next_tel[ip_i], next_y[ip_i], Ls, rec_model, rec_y_weighted, donor_mode,
                           rec_tel_mode, pc, circle_len, prob_ts,
                           min_len_circle_gen, prob_each_circle,
                           dynamic_circles, next_pc[ip_i:ip_i+1], max_Ys, state)

            # Re-check if it can now divide
            can_divide = True
            for j in range(ncols):
                if next_tel[ip_i, j] < Ls:
                    can_divide = False
                    break
            if not can_divide:
                continue  # still senescent; try another parent

        # Spawn child from dividing parent
        for j in range(ncols):
            next_tel[pool, j] = next_tel[ip_i, j]
            next_y  [pool, j] = next_y  [ip_i, j]
        # split circles
        next_pc[pool] = next_pc[ip_i] * 0.5
        next_pc[ip_i] *= 0.5
        # deletions to both parent and child
        apply_deletions(next_tel[ip_i], next_tel[pool], del_rate_user, state)

        pool += 1

    # Finalize
    out_tel = np.empty((pool, ncols), dtype=np.uint16)
    out_y   = np.empty((pool, ncols), dtype=np.uint16)
    out_pc  = np.empty((pool,), dtype=np.float32)
    for i in range(pool):
        for j in range(ncols):
            out_tel[i, j] = next_tel[i, j]
            out_y  [i, j] = next_y  [i, j]
        out_pc[i] = next_pc[i]
    return pool, out_tel, out_y, out_pc


import sys, time, argparse, numpy as np

def q5set(a: np.ndarray):
    if a.size == 0: return (np.nan,)*5
    p = np.percentile(a, [5,10,50,90,95])
    return (float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]))

def fmt(x):
    if x is None: return "NA"
    try:
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)): return "NA"
    except Exception:
        pass
    return f"{x}"

def pre_evolve_numba(init_tel, init_y, pd_pre, del_rate_user, Ls,
                     hard_threshold, hard_keep_fraction, py_rng):
    tel = init_tel[np.newaxis,:].copy()
    y   = init_y[np.newaxis,:].copy()
    for _ in range(pd_pre):
        if tel.shape[0] > hard_threshold:
            keep = max(1, int(tel.shape[0] * hard_keep_fraction))
            idx = py_rng.choice(tel.shape[0], size=keep, replace=False)
            tel = tel[idx].copy(); y = y[idx].copy()
        state = make_state_from_py_rng(int(py_rng.integers(0, 2**31-1)))
        _, tel, y = pd_step_pre_symm_both_numba(tel, y, Ls, del_rate_user, state)
        if tel.shape[0] == 0: break
    return tel, y

def compute_metrics_py(tel: np.ndarray, y: np.ndarray, Ls: int):
    N = int(tel.shape[0])
    if N == 0:
        return dict(N_cells=0, frac_senescent=1.0,
                    tel_mean=np.nan, tel_p5=np.nan, tel_p10=np.nan, tel_p50=np.nan, tel_p90=np.nan, tel_p95=np.nan,
                    y_mean=np.nan, y_p5=np.nan, y_p10=np.nan, y_p50=np.nan, y_p90=np.nan, y_p95=np.nan)
    sen_mask = (tel < Ls).any(axis=1)
    frac_sen = float(np.mean(sen_mask))
    flat_tel = tel.reshape(-1).astype(np.float64)
    tel_mean = float(np.mean(flat_tel))
    tel_p5, tel_p10, tel_p50, tel_p90, tel_p95 = q5set(flat_tel)
    y_sum = y.sum(axis=1).astype(np.float64)
    y_mean = float(np.mean(y_sum))
    y_p5, y_p10, y_p50, y_p90, y_p95 = q5set(y_sum)
    return dict(N_cells=N, frac_senescent=frac_sen,
                tel_mean=tel_mean, tel_p5=tel_p5, tel_p10=tel_p10, tel_p50=tel_p50, tel_p90=tel_p90, tel_p95=tel_p95,
                y_mean=y_mean, y_p5=y_p5, y_p10=y_p10, y_p50=y_p50, y_p90=y_p90, y_p95=y_p95)

def write_replicate_blocks(matrix, out_prefix, num_replicates):
    metrics = ["N_cells","frac_senescent",
               "tel_mean","tel_p5","tel_p10","tel_p50","tel_p90","tel_p95",
               "y_mean","y_p5","y_p10","y_p50","y_p90","y_p95"]
    max_pd = max(r['PD'] for rep in matrix for r in rep if r['PD'] is not None)
    path = f"{out_prefix}.replicates.tsv"
    with open(path, "w") as f:
        f.write(f"# Program: {PROG}\n")
        f.write(f"# Version: {VERSION}\n")
        f.write(f"# Command: {' '.join(map(str, sys.argv))}\n")
        f.write("# Format: stacked by metric; rows=PD 0..max; cols=Rep1..RepK\n\n")
        for metric in metrics:
            f.write(f"## {metric}\n")
            cols = ["PD"] + [f"Rep{ri}" for ri in range(1, num_replicates+1)]
            f.write("\t".join(cols) + "\n")
            for pd in range(0, max_pd+1):
                row_vals = [str(pd)]
                for ri, rep_rows in enumerate(matrix, start=1):
                    match = None
                    for r in rep_rows:
                        if r["PD"] == pd:
                            match = r; break
                    row_vals.append("NA" if match is None else fmt(match[metric]))
                f.write("\t".join(row_vals) + "\n")
            f.write("\n")

def simulate_once(pd_max, del_rate_user, Ls, rec_model, rec_y_weighted, donor_mode, rec_tel_mode,
                  prob_circle, circle_len, prob_ts, p_sen_death, max_freq_senesc,
                  min_len_circle_gen, prob_each_circle, dynamic_circles,
                  hard_threshold, hard_keep_fraction, subsample_pds, subsample_size,
                  init_n_cells, tel_pre, y_pre, out_prefix, max_Ys, base_seed, rep_idx):
    rng = np.random.default_rng(base_seed + rep_idx*7867)
    init_n = min(init_n_cells, tel_pre.shape[0])
    idx_seed = rng.choice(tel_pre.shape[0], size=init_n, replace=False)
    tel = np.array(tel_pre[idx_seed], copy=True)
    y   = np.array(y_pre[idx_seed],   copy=True)
    if dynamic_circles:
        p_circle = np.zeros((tel.shape[0],), dtype=np.float32)
    else:
        p_circle = np.full((tel.shape[0],), float(prob_circle), dtype=np.float32)
    rows = []
    m = compute_metrics_py(tel, y, Ls); m["PD"]=0
    if m["frac_senescent"] >= max_freq_senesc:
        m["frac_senescent"] = 1.0; rows.append(m); return rows
    rows.append(m)
    for pd in range(1, pd_max+1):
        if tel.shape[0] > hard_threshold:
            keep = max(1, int(tel.shape[0] * hard_keep_fraction))
            idx = rng.choice(tel.shape[0], size=keep, replace=False)
            tel = tel[idx].copy(); y = y[idx].copy(); p_circle = p_circle[idx].copy()
        state = make_state_from_py_rng(int(rng.integers(0, 2**31-1)))
        pool, tel, y, p_circle = pd_step_numba(tel, y, p_circle, Ls, del_rate_user,
                                               rec_model, 1 if rec_y_weighted else 0, donor_mode, rec_tel_mode,
                                               prob_circle, circle_len, prob_ts, p_sen_death,
                                               min_len_circle_gen, prob_each_circle,
                                               1 if dynamic_circles else 0, max_Ys, state)
        if pd in subsample_pds and pool > subsample_size:
            idx = rng.choice(pool, size=subsample_size, replace=False)
            tel = tel[idx].copy(); y = y[idx].copy(); p_circle = p_circle[idx].copy(); pool = subsample_size
        m = compute_metrics_py(tel, y, Ls); m["PD"]=pd
        if m["frac_senescent"] >= max_freq_senesc:
            m["frac_senescent"] = 1.0; rows.append(m); break
        rows.append(m)
        if m["frac_senescent"] >= 1.0: break
    return rows

def run_replicates_serial(num_replicates, seed, auto_seed,
                          pd_pre, init_len, init_len_file, init_Ys, init_Y_file,
                          init_n_cells, pd_max, del_rate_user, Ls,
                          rec_model, rec_y_weighted, donor_mode, rec_tel_mode,
                          prob_circle, circle_len, prob_ts, p_sen_death, max_freq_senesc,
                          min_len_circle_gen, prob_each_circle, dynamic_circles,
                          hard_threshold, hard_keep_fraction,
                          subsample_pds, subsample_size, out_prefix, max_Ys):
    t0 = time.time()
    # RNG
    if seed is not None: base_rng = np.random.default_rng(seed)
    elif auto_seed:      base_rng = np.random.default_rng(np.random.SeedSequence().entropy)
    else:                base_rng = np.random.default_rng()
    # init tel
    if init_len is not None:
        rng_tmp = np.random.default_rng(base_rng.integers(0, 2**31-1))
        tel0 = rng_tmp.poisson(init_len, size=N_ENDS).astype(np.int64)
        tel0 = np.clip(tel0, Ls, UINT16_MAX).astype(np.uint16)
    elif init_len_file:
        vals = [int(x.strip()) for x in open(init_len_file) if x.strip()]
        if len(vals) != N_ENDS:
            raise ValueError(f"--init-len-file must have {N_ENDS} lines")
        tel0 = np.array([max(Ls, min(int(v), UINT16_MAX)) for v in vals], dtype=np.uint16)
    else:
        tel0 = np.full(N_ENDS, max(Ls, 225), dtype=np.uint16)
    # init Ys
    if init_Y_file:
        vals = [int(x.strip()) for x in open(init_Y_file) if x.strip()]
        if len(vals) != N_ENDS:
            raise ValueError("--init-Y-file must have 32 lines")
        y0 = np.array([min(max(int(v),0), UINT16_MAX) for v in vals], dtype=np.uint16)
    else:
        if init_Ys is not None:
            rng_tmp = np.random.default_rng(base_rng.integers(0, 2**31-1))
            probs = np.full(N_ENDS, 1.0 / N_ENDS, dtype=float)
            counts = rng_tmp.multinomial(int(init_Ys), probs).astype(np.int64)
            counts = np.clip(counts, 0, UINT16_MAX)
            y0 = counts.astype(np.uint16)
        else:
            y0 = np.zeros(N_ENDS, dtype=np.uint16)
    # pre-evo
    rng_pre = np.random.default_rng(base_rng.integers(0, 2**31-1))
    tel_pre, y_pre = pre_evolve_numba(tel0, y0, pd_pre, del_rate_user, Ls,
                                      hard_threshold, hard_keep_fraction, rng_pre)
    if tel_pre.shape[0] == 0:
        raise RuntimeError("Pre-evolution collapsed to zero cells; adjust init_len/del_rate/Ls")

    matrix = []
    base_seed = int(base_rng.integers(0, 2**31-1))
    for rep in range(1, num_replicates+1):
        rows = simulate_once(pd_max, del_rate_user, Ls, rec_model, bool(rec_y_weighted),
                             donor_mode, rec_tel_mode, prob_circle, circle_len, prob_ts, p_sen_death,
                             max_freq_senesc, min_len_circle_gen, prob_each_circle, bool(dynamic_circles),
                             hard_threshold, hard_keep_fraction, subsample_pds, subsample_size,
                             init_n_cells, tel_pre, y_pre, out_prefix, max_Ys, base_seed, rep)
        matrix.append(rows)

    # outputs
    write_replicate_blocks(matrix, out_prefix, num_replicates)
    # summary
    max_pd = max(r['PD'] for rep in matrix for r in rep if r['PD'] is not None)
    metrics = ["N_cells","frac_senescent",
               "tel_mean","tel_p5","tel_p10","tel_p50","tel_p90","tel_p95",
               "y_mean","y_p5","y_p10","y_p50","y_p90","y_p95"]
    summary = []
    for pd in range(0, max_pd+1):
        agg = {"PD": pd}
        for m in metrics:
            vals = []
            for rep_rows in matrix:
                for r in rep_rows:
                    if r["PD"] == pd:
                        vals.append(r[m]); break
            arr = np.array(vals, dtype=float)
            agg[m] = float(np.nanmean(arr)) if arr.size else np.nan
        summary.append(agg)
    wall = time.time() - t0
    with open(f"{out_prefix}.summary.tsv", "w") as f:
        f.write(f"# Program: {PROG}\n# Version: {VERSION}\n# Command: {' '.join(map(str, sys.argv))}\n")
        f.write("# WallTimeSeconds: %.6f\n" % wall)
        f.write("PD\tN_cells\tfrac_senescent\ttel_mean\ttel_p5\ttel_p10\ttel_p50\ttel_p90\ttel_p95\ty_mean\ty_p5\ty_p10\ty_p50\ty_p90\ty_p95\n")
        for r in summary:
            f.write(f"{r['PD']}\t{r['N_cells']}\t{r['frac_senescent']}"
                    f"\t{r['tel_mean']}\t{r['tel_p5']}\t{r['tel_p10']}\t{r['tel_p50']}\t{r['tel_p90']}\t{r['tel_p95']}"
                    f"\t{r['y_mean']}\t{r['y_p5']}\t{r['y_p10']}\t{r['y_p50']}\t{r['y_p90']}\t{r['y_p95']}\n")
    return np.array(summary, dtype=object)

def parse_args(argv=None):
    p = argparse.ArgumentParser(prog=PROG, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--pd-max", type=int, default=50)
    p.add_argument("--pd-pre", type=int, default=10)
    p.add_argument("--num-replicates", type=int, default=10)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--auto-seed", action="store_true")
    p.add_argument("--init-len", type=int, default=225)
    p.add_argument("--init-len-file", type=str, default=None)
    p.add_argument("--init-Ys", type=int, default=40)
    p.add_argument("--init-Y-file", type=str, default=None)
    p.add_argument("--init-n-cells", type=int, default=10)
    p.add_argument("--del-rate", type=float, default=6.0)
    p.add_argument("--Ls", type=int, default=60)
    p.add_argument("--rec-model", type=int, choices=[0,1], default=1)
    p.add_argument("--rec-y-weighted", action="store_true")
    p.add_argument("--donor-mode", type=int, choices=[0,1,2], default=2)
    p.add_argument("--rec-tel-mode", type=str, choices=["copy","rnd","end"], default="copy")
    p.add_argument("--prob-circle", type=float, default=0.0)
    p.add_argument("--circle-len", type=int, default=2000)
    p.add_argument("--prob-ts", type=float, default=0.0)
    p.add_argument("--dynamic-circles", action="store_true")
    p.add_argument("--min-len-circle-generation", type=int, default=120)
    p.add_argument("--prob-each-circle", type=float, default=0.001)
    p.add_argument("--p-sen-death", type=float, default=0.1)
    p.add_argument("--max-freq-senesc", type=float, default=0.9999)
    p.add_argument("--hard-threshold", type=int, default=512000)
    p.add_argument("--hard-keep-fraction", type=float, default=0.1)
    p.add_argument("--subsample-pds", type=int, nargs="*", default=[20,30,40])
    p.add_argument("--subsample-size", type=int, default=10000)
    p.add_argument("--max-Ys", type=int, default=50)
    p.add_argument("--out-prefix", type=str, default="TelomereDynamics")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    if (args.seed is None) and (not args.auto_seed):
        args.auto_seed = True
    args.max_freq_senesc = min(1.0, max(0.0, args.max_freq_senesc))
    rec_tel_mode = {"copy":0, "rnd":1, "end":2}[args.rec_tel_mode]
    _ = run_replicates_serial(num_replicates=args.num_replicates,
                              seed=args.seed, auto_seed=args.auto_seed,
                              pd_pre=args.pd_pre, init_len=args.init_len, init_len_file=args.init_len_file,
                              init_Ys=args.init_Ys, init_Y_file=args.init_Y_file,
                              init_n_cells=args.init_n_cells,
                              pd_max=args.pd_max, del_rate_user=args.del_rate, Ls=args.Ls,
                              rec_model=args.rec_model, rec_y_weighted=bool(args.rec_y_weighted),
                              donor_mode=args.donor_mode, rec_tel_mode=rec_tel_mode,
                              prob_circle=args.prob_circle, circle_len=args.circle_len,
                              prob_ts=args.prob_ts, p_sen_death=args.p_sen_death, max_freq_senesc=args.max_freq_senesc,
                              min_len_circle_gen=args.min_len_circle_generation,
                              prob_each_circle=args.prob_each_circle, dynamic_circles=bool(args.dynamic_circles),
                              hard_threshold=args.hard_threshold, hard_keep_fraction=args.hard_keep_fraction,
                              subsample_pds=args.subsample_pds, subsample_size=args.subsample_size,
                              out_prefix=args.out_prefix, max_Ys=args.max_Ys)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
