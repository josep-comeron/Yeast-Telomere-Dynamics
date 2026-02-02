# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from numba import njit
import numpy as np, math
import sys, time, argparse, numpy as np

"""
YeastTelomereDynamics Numba
Version: V01
Requires numba>=0.60 and numpy>=1.26.
"""

VERSION = "V01"
PROG = "TelomereDynamics.V01.py"
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
    # del_rate is the observed (average) per telomere per division; kernel applies to half the ends 2*del_rate
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
    # Pre-evolution step (shared among replicates)
    # The initial cell generates a large pop. of cells with variance in telomere lengths around initial values, within and between cells (Y's do not change)
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
        # copy to child
        for j in range(ncols):
            next_tel[pool, j] = next_tel[ip, j]
            next_y  [pool, j] = next_y  [ip, j]
        # mutate parent and child symmetrically (50%-50% erosion or extension) to add variance (always >=Ls)
        lam = 2.0 * del_rate_user
        for j in range(ncols):
            if rng_int(state, 2) == 1:
                k = poisson_knuth(lam, state)
                sign = 1 if rng_int(state, 2)==1 else -1
                v = int(next_tel[ip, j]) + sign*k
                if v < Ls: v = Ls
                next_tel[ip, j] = np.uint16(v)
            if rng_int(state, 2) == 1:
                k = poisson_knuth(lam, state)
                sign = 1 if rng_int(state, 2)==1 else -1
                v = int(next_tel[pool, j]) + sign*k
                if v < Ls: v = Ls
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
def recombine_cell(row_tel, row_y, Ls, rec_model, rec_y_weighted, donor_mode,
                   rec_tel_mode, prob_circle, circle_len, prob_ts,
                   min_len_circle_gen, prob_each_circle,
                   dynamic_circles, p_circle_view, max_Ys, state):

    # If rec_model==1 (recombination allowed):
    # Recombination can occur in senescent cells.
    # All chromosome ends with telomere <Ls (receptors) can attempt recombination with elegible donor chromosome ends (different than receptor). 
    # Elegible donor chromosome ends must have telomere length >= Ls. If a cell has no ends >= Ls, it cannot recombine.
    # TS (template switching) can occur after a receptor has recombined with a donor (telomere, Y' or X), with multiple possible 'jumps' or 'hops' to a different elegible telomere donor.
 
    if rec_model == 0:
        return

    ncols = row_tel.shape[0]

    # If there is no possible donor in this cell, skip recombination entirely
    has_ge = False
    for j in range(ncols):
        if row_tel[j] >= Ls:
            has_ge = True
            break
    if not has_ge:
        return

    def add_circle_if_eligible(new_tel_val):
        if dynamic_circles == 1 and new_tel_val >= min_len_circle_gen:
            pc = p_circle_view[0] + prob_each_circle
            if pc > 1.0:
                pc = 1.0
            p_circle_view[0] = pc

    def count_eligible_hr(exclude_idx, require_y_ge1):
        c = 0
        for j in range(ncols):
            if j == exclude_idx:
                continue
            if row_tel[j] < Ls:
                continue
            if require_y_ge1 == 1 and row_y[j] < 1:
                continue
            c += 1
        return c

    def choose_donor_hr_X_rndY(exclude_idx,require_y_ge1):
        # uniform among eligible (>=Ls; no receptor; any for X rec, >0 Ys for Y' rec) 
        c = count_eligible_hr(exclude_idx, require_y_ge1)
        if c <= 0:
            return -1
        k = int(rng_int(state, c))
        acc = 0
        for j in range(ncols):
            if j == exclude_idx:
                continue
            if row_tel[j] < Ls:
                continue
            if require_y_ge1 == 1 and row_y[j] < 1:
                continue
            if acc == k:
                return j
            acc += 1
        return -1

    def choose_donor_hr_wY(exclude_idx,require_y_ge1):
        # donor weighted based on number of Ys among elegible (>=Ls; no receptor, >0 Ys for Y' rec)
        c = count_eligible_hr(exclude_idx, require_y_ge1)
        if c <= 0:
            return -1
        tot = 0
        for j in range(ncols):
            if j == exclude_idx:
                continue
            if row_tel[j] < Ls:
                continue
            if require_y_ge1 == 1 and row_y[j] < 1:
                continue
            yi = int(row_y[j])
            if yi > 0: tot += yi
        if tot <= 0: return -1
        rint = rng_int(state, tot)
        acc = 0
        for j in range(ncols):
            if j == exclude_idx:
                continue
            if row_tel[j] < Ls:
                continue
            if require_y_ge1 == 1 and row_y[j] < 1:
                continue
            yi = int(row_y[j])
            if yi > 0: acc += yi
            if rint < acc:
                return j
        return -1

    def choose_donor_hr_tel(exclude_idx):
        # Eligible HR donors: tel>=Ls, Y'>=1 optional if rec is at Y', and not the receptor.
        if donor_mode == 2:
            best = -1
            bestv = -1
            for j in range(ncols):
                if j == exclude_idx:
                    continue
                if row_tel[j] < Ls:
                    continue
                v = int(row_tel[j])
                if v > bestv:
                    bestv = v
                    best = j
            return best

        if donor_mode == 1:
            S = 0
            for j in range(ncols):
                if j == exclude_idx:
                    continue
                if row_tel[j] < Ls:
                    continue
                S += int(row_tel[j])
            if S <= 0:
                return -1
            r = int(rng_int(state, S))
            acc = 0
            for j in range(ncols):
                if j == exclude_idx:
                    continue
                if row_tel[j] < Ls:
                    continue
                acc += int(row_tel[j])
                if r < acc:
                    return j
            return -1

        # donor_mode == 0: uniform among eligible
        c = count_eligible_hr(exclude_idx, 0)
        if c <= 0:
            return -1
        k = int(rng_int(state, c))
        acc = 0
        for j in range(ncols):
            if j == exclude_idx:
                continue
            if row_tel[j] < Ls:
                continue
            if acc == k:
                return j
            acc += 1
        return -1

    def count_eligible_ts(exclude_idx, last_donor):
        c = 0
        for j in range(ncols):
            if j == exclude_idx:
                continue
            if j == last_donor:
                continue
            if row_tel[j] < Ls:
                continue
            c += 1
        return c

    def choose_donor_ts(exclude_idx, last_donor):
        # Eligible telomere TS donors: tel>=Ls, not receptor, not last used donor 
        if donor_mode == 2:
            best = -1
            bestv = -1
            for j in range(ncols):
                if j == exclude_idx or j == last_donor:
                    continue
                if row_tel[j] < Ls:
                    continue
                v = int(row_tel[j])
                if v > bestv:
                    bestv = v
                    best = j
            return best

        if donor_mode == 1:
            S = 0
            for j in range(ncols):
                if j == exclude_idx or j == last_donor:
                    continue
                if row_tel[j] < Ls:
                    continue
                S += int(row_tel[j])
            if S <= 0:
                return -1
            r = int(rng_int(state, S))
            acc = 0
            for j in range(ncols):
                if j == exclude_idx or j == last_donor:
                    continue
                if row_tel[j] < Ls:
                    continue
                acc += int(row_tel[j])
                if r < acc:
                    return j
            return -1

        c = count_eligible_ts(exclude_idx, last_donor)
        if c <= 0:
            return -1
        k = int(rng_int(state, c))
        acc = 0
        for j in range(ncols):
            if j == exclude_idx or j == last_donor:
                continue
            if row_tel[j] < Ls:
                continue
            if acc == k:
                return j
            acc += 1
        return -1

    def recombine_ts(idx, last_donor):
      # TS hops (up to 5)
        hops = 0
        while hops < 5 and rng_uniform01(state) < prob_ts:
            d2 = choose_donor_ts(idx, last_donor)
            if d2 < 0:
                break
            last_donor = d2
            TelD2 = int(row_tel[d2])
            rR2 = int(row_tel[idx])
            rD2 = int(rng_int(state, TelD2 + 1))
            new_tel = rR2 + (TelD2 - rD2)
            if new_tel < 0:
                new_tel = 0
            if new_tel > 65535:
                new_tel = 65535
            row_tel[idx] = np.uint16(new_tel)
            hops += 1
        return

    def recombine_one_end(idx):
        # Circle recombination first
        if prob_circle > 0.0 and rng_uniform01(state) < prob_circle:
            new_tel = int(row_tel[idx]) + int(circle_len)
            if new_tel > 65535:
                new_tel = 65535
            row_tel[idx] = np.uint16(new_tel)
            return

        # Chose what element of eroded chromosome end recombines (tel, one of the Y's or X)
        ycnt = int(row_y[idx])
        total = 2 + ycnt  # Tel + Ys + X
        pick = int(rng_int(state, total))

        if pick == 0:
            # HR telomere recombination
            d = choose_donor_hr_tel(idx)
            if d < 0:
                return
            last_donor = d

            TelR = int(row_tel[idx])
            TelD = int(row_tel[d])

            if rec_tel_mode == 2:
                rR = TelR
                rD = int(rng_int(state, TelD + 1))
            elif rec_tel_mode == 1:
                rR = int(rng_int(state, TelR + 1))
                rD = int(rng_int(state, TelD + 1))
            else:
                rR = 0
                rD = 0

            new_tel = rR + (TelD - rD)
            if new_tel < 0:
                new_tel = 0
            if new_tel > 65535:
                new_tel = 65535
            row_tel[idx] = np.uint16(new_tel)

            # TS hops
            recombine_ts(idx,last_donor)

            # Add circles to this cell if dynamic-circles 
            add_circle_if_eligible(int(row_tel[idx]))
            return

        if pick <= ycnt:
            # Y recombination: HR donor must have at least one Y'
            if rec_y_weighted == 1:
                d = choose_donor_hr_wY(idx, 1)
            else:
                d = choose_donor_hr_X_rndY(idx, 1)
            if d < 0:
                return
            last_donor = d

            y_donor_cnt = int(row_y[d])
            if y_donor_cnt <= 0:
                return
            yk = int(rng_int(state, y_donor_cnt))
            y_rec_idx = pick - 1
            new_y = (y_rec_idx + 1) + (y_donor_cnt - yk)
            if new_y < 0:
                new_y = 0
            if new_y > max_Ys:
                new_y = max_Ys
            row_y[idx] = np.uint16(new_y)
            row_tel[idx] = np.uint16(row_tel[d])

            # TS hops
            recombine_ts(idx,last_donor)

            # Add circles to this cell if dynamic-circles 
            add_circle_if_eligible(int(row_tel[idx]))
            return

        # X recombination
        d = choose_donor_hr_X_rndY(idx, 0)
        if d < 0:
            return
        last_donor = d
        row_tel[idx] = np.uint16(row_tel[d])
        yy = int(row_y[d])
        if yy > max_Ys:
            yy = max_Ys
        row_y[idx] = np.uint16(yy)

        # TS hops
        recombine_ts(idx,last_donor)

        # Add circles to this cell if dynamic-circles 
        add_circle_if_eligible(int(row_tel[idx]))
        return

    for i in range(ncols):
        if row_tel[i] < Ls:
            recombine_one_end(i)

@njit(cache=True)
def pd_step(tel, y, p_circle, Ls, del_rate_user,
                  rec_model, rec_y_weighted, donor_mode,
                  rec_tel_mode, prob_circle_global, circle_len,
                  prob_ts, p_sen_death, min_len_circle_gen, prob_each_circle,
                  dynamic_circles, max_Ys, state):
    N = tel.shape[0]
    ncols = tel.shape[1]

    # Any non-senescent (dividers) in the pop.? Stop if False 
    has_divider = False
    for i in range(N):
        ok = True
        for j in range(ncols):
            if tel[i, j] < Ls:
                ok = False
                break
        if ok:
            has_divider = True
            break
    if (N == 0) or (not has_divider):
        return N, tel, y, p_circle

    next_tel = np.empty((2*N, ncols), dtype=np.uint16)
    next_y   = np.empty((2*N, ncols), dtype=np.uint16)
    next_pc  = np.empty((2*N,), dtype=np.float32)
    for i in range(N):
        for j in range(ncols):
            next_tel[i, j] = tel[i, j]
            next_y[i, j]   = y[i, j]
        next_pc[i] = p_circle[i]

    pool = int(N)
    while pool < 2*N:
        if pool == 0:
            break
        p = rng_int(state, pool)
        ip = int(p)

        alive = True
        for j in range(ncols):
            if next_tel[ip, j] < Ls:
                alive = False
                break

        if not alive:
            if p_sen_death > 0.0 and rng_uniform01(state) < p_sen_death:
                last = pool - 1
                for j in range(ncols):
                    tmp  = next_tel[ip, j]; next_tel[ip, j] = next_tel[last, j]; next_tel[last, j] = tmp
                    tmp2 = next_y  [ip, j]; next_y  [ip, j] = next_y  [last, j]; next_y  [last, j] = tmp2
                tmp3 = next_pc[ip]; next_pc[ip] = next_pc[last]; next_pc[last] = tmp3
                pool -= 1
                continue

            if rec_model != 0:
                pc = next_pc[ip] if dynamic_circles == 1 else prob_circle_global
                recombine_cell(next_tel[ip], next_y[ip], Ls, rec_model, rec_y_weighted, donor_mode,
                               rec_tel_mode, pc, circle_len, prob_ts,
                               min_len_circle_gen, prob_each_circle,
                               dynamic_circles, next_pc[ip:ip+1], max_Ys, state)
                can_divide = True
                for j in range(ncols):
                    if next_tel[ip, j] < Ls:
                        can_divide = False
                        break
                if not can_divide:
                    continue
            else:
                # anti-stall: scan current pool for any divider; if none, bail
                has_divider_now = False
                for ii in range(pool):
                    ok_now = True
                    for jj in range(ncols):
                        if next_tel[ii, jj] < Ls:
                            ok_now = False
                            break
                    if ok_now:
                        has_divider_now = True
                        break
                if not has_divider_now:
                    break
                continue

        # generate child
        for j in range(ncols):
            next_tel[pool, j] = next_tel[ip, j]
            next_y  [pool, j] = next_y  [ip, j]
        # circle number split at division
        next_pc[pool] = next_pc[ip] * 0.5
        next_pc[ip]   *= 0.5
        # deletions to both parent and child
        apply_deletions(next_tel[ip], next_tel[pool], del_rate_user, state)

        pool += 1

    out_tel = np.empty((pool, ncols), dtype=np.uint16)
    out_y   = np.empty((pool, ncols), dtype=np.uint16)
    out_pc  = np.empty((pool,), dtype=np.float32)
    for i in range(pool):
        for j in range(ncols):
            out_tel[i, j] = next_tel[i, j]
            out_y  [i, j] = next_y  [i, j]
        out_pc[i] = next_pc[i]
    return pool, out_tel, out_y, out_pc

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

def compute_metrics_py(tel: np.ndarray, y: np.ndarray, p_circle: np.ndarray, Ls: int, prob_each_circle: float, dynamic_circles: bool):
    N = int(tel.shape[0])
    if N == 0:
        return dict(N_cells=0, frac_senescent=1.0,
                    tel_mean=np.nan, tel_p5=np.nan, tel_p10=np.nan, tel_p50=np.nan, tel_p90=np.nan, tel_p95=np.nan,
                    y_mean=np.nan, y_p5=np.nan, y_p10=np.nan, y_p50=np.nan, y_p90=np.nan, y_p95=np.nan,
                    avg_circles_per_cell=np.nan)
    sen_mask = (tel < Ls).any(axis=1)
    frac_sen = float(np.mean(sen_mask))
    flat_tel = tel.reshape(-1).astype(np.float64)
    tel_mean = float(np.mean(flat_tel))
    tel_p5, tel_p10, tel_p50, tel_p90, tel_p95 = q5set(flat_tel)
    y_sum = y.sum(axis=1).astype(np.float64)
    y_mean = float(np.mean(y_sum))
    y_p5, y_p10, y_p50, y_p90, y_p95 = q5set(y_sum)
    if dynamic_circles and prob_each_circle > 0.0:
        avg_circles = float(np.mean(p_circle[:N].astype(np.float64))) / float(prob_each_circle)
    else:
        avg_circles = np.nan
    return dict(N_cells=N, frac_senescent=frac_sen,
                tel_mean=tel_mean, tel_p5=tel_p5, tel_p10=tel_p10, tel_p50=tel_p50, tel_p90=tel_p90, tel_p95=tel_p95,
                y_mean=y_mean, y_p5=y_p5, y_p10=y_p10, y_p50=y_p50, y_p90=y_p90, y_p95=y_p95,
                avg_circles_per_cell=avg_circles)

def write_replicate_blocks(matrix, out_prefix, num_replicates):
    metrics = ["N_cells","frac_senescent",
               "tel_mean","tel_p5","tel_p10","tel_p50","tel_p90","tel_p95",
               "y_mean","y_p5","y_p10","y_p50","y_p90","y_p95","avg_circles_per_cell"]
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
                    row_vals.append("NA" if match is None else fmt(match.get(metric, float("nan"))))
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
    m = compute_metrics_py(tel, y, p_circle, Ls, prob_each_circle, dynamic_circles); m["PD"]=0
    if m["frac_senescent"] >= max_freq_senesc:
        m["frac_senescent"] = 1.0; rows.append(m); return rows
    rows.append(m)
    for pd in range(1, pd_max+1):
        if tel.shape[0] > hard_threshold:
            keep = max(1, int(tel.shape[0] * hard_keep_fraction))
            idx = rng.choice(tel.shape[0], size=keep, replace=False)
            tel = tel[idx].copy(); y = y[idx].copy(); p_circle = p_circle[idx].copy()
        state = make_state_from_py_rng(int(rng.integers(0, 2**31-1)))
        pool, tel, y, p_circle = pd_step(tel, y, p_circle, Ls, del_rate_user,
                                               rec_model, 1 if rec_y_weighted else 0, donor_mode, rec_tel_mode,
                                               prob_circle, circle_len, prob_ts, p_sen_death,
                                               min_len_circle_gen, prob_each_circle,
                                               1 if dynamic_circles else 0, max_Ys, state)
        if pd in subsample_pds and pool > subsample_size:
            idx = rng.choice(pool, size=subsample_size, replace=False)
            tel = tel[idx].copy(); y = y[idx].copy(); p_circle = p_circle[idx].copy(); pool = subsample_size
        m = compute_metrics_py(tel, y, p_circle, Ls, prob_each_circle, dynamic_circles); m["PD"]=pd
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
    # base RNG
    if seed is not None: base_rng = np.random.default_rng(seed)
    elif auto_seed:      base_rng = np.random.default_rng(np.random.SeedSequence().entropy)
    else:                base_rng = np.random.default_rng()

    # Generation of initial tel lengths
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

    # Generation of initial Ys number and distribution
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
    
    # pre-evolution step (once, shared for all replicates)
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
               "y_mean","y_p5","y_p10","y_p50","y_p90","y_p95","avg_circles_per_cell"]
    summary = []
    for pd in range(0, max_pd+1):
        agg = {"PD": pd}
        for m in metrics:
            vals = []
            for rep_rows in matrix:
                for r in rep_rows:
                    if r["PD"] == pd:
                        vals.append(r.get(m, float("nan"))); break
            arr = np.array(vals, dtype=float)
            if arr.size == 0 or not np.isfinite(arr).any():
                agg[m] = np.nan
            else:
                agg[m] = float(np.nanmean(arr))
        summary.append(agg)
    wall = time.time() - t0
    with open(f"{out_prefix}.summary.tsv", "w") as f:
        f.write(f"# Program: {PROG}\n# Version: {VERSION}\n# Command: {' '.join(map(str, sys.argv))}\n")
        f.write("# WallTimeSeconds: %.6f\n" % wall)
        f.write("PD	N_cells	frac_senescent	tel_mean	tel_p5	tel_p10	tel_p50	tel_p90	tel_p95	y_mean	y_p5	y_p10	y_p50	y_p90	y_p95	avg_circles_per_cell\n")
        for r in summary:
            f.write(f"{r['PD']}\t{r['N_cells']}\t{r['frac_senescent']}"
                    f"\t{r['tel_mean']}\t{r['tel_p5']}\t{r['tel_p10']}\t{r['tel_p50']}\t{r['tel_p90']}\t{r['tel_p95']}"
                    f"\t{r['y_mean']}\t{r['y_p5']}\t{r['y_p10']}\t{r['y_p50']}\t{r['y_p90']}\t{r['y_p95']}\t{r['avg_circles_per_cell']}\n")
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
    p.add_argument("--init-Ys", type=int, default=None)
    p.add_argument("--init-Y-file", type=str, default=None)
    p.add_argument("--init-n-cells", type=int, default=10)
    p.add_argument("--del-rate", type=float, default=6.0)
    p.add_argument("--Ls", type=int, default=60)
    p.add_argument("--rec-model", type=int, choices=[0,1], default=1)
    p.add_argument("--rec-y-weighted", action="store_true")
    p.add_argument("--donor-mode", type=int, choices=[0,1,2], default=1)
    p.add_argument("--rec-tel-mode", type=str, choices=["copy","rnd","end"], default="copy")
    p.add_argument("--prob-circle", type=float, default=0.0)
    p.add_argument("--circle-len", type=int, default=2000)
    p.add_argument("--prob-ts", type=float, default=0.0)
    p.add_argument("--dynamic-circles", action="store_true")
    p.add_argument("--min-len-circle-generation", type=int, default=120)
    p.add_argument("--prob-each-circle", type=float, default=0.001)
    p.add_argument("--p-sen-death", type=float, default=0.1)
    p.add_argument("--max-freq-senesc", type=float, default=0.999)
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
