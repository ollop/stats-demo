from __future__ import annotations

import csv
import io
import re
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from scipy import stats

app = FastAPI(title="Inferential Stats Demo")
templates = Jinja2Templates(directory="templates")

DesignType = Literal["between", "within"]
CorrStructure = Literal["cs", "ar1"]

# Stores the last dataset produced by /api/simulate (in-memory, per server process).
LAST_DATASET: dict | None = None


class ConditionSpec(BaseModel):
    label: str = Field(min_length=1)
    mean: float
    sd: float = Field(gt=0)


class SimulateRequest(BaseModel):
    design_type: DesignType = "between"
    n_factors: int = Field(ge=1, le=2)
    levels: List[int] = Field(min_length=1, max_length=2)

    # Between: participants per condition
    # Within: number of subjects
    sample_size: int = Field(ge=1, le=20000)

    # Participant score is the mean across these iid trials
    trials_per_participant: int = Field(ge=1, le=500, default=1)

    seed: Optional[int] = None
    pdf_points: int = Field(ge=50, le=2000, default=401)
    conditions: List[ConditionSpec] = Field(min_length=1)

    # Within-subjects only
    corr_structure: CorrStructure = "cs"
    corr: float = Field(default=0.5)  # CS: r, AR1: rho
    preview_subjects: int = Field(ge=0, le=200, default=60)


class MonteCarloRequest(BaseModel):
    design_type: DesignType = "between"
    n_factors: int = Field(ge=1, le=2)
    levels: List[int] = Field(min_length=1, max_length=2)

    sample_size: int = Field(ge=2, le=20000)
    trials_per_participant: int = Field(ge=1, le=500, default=1)

    seed: Optional[int] = None
    conditions: List[ConditionSpec] = Field(min_length=1)

    n_sims: int = Field(ge=10, le=1000000, default=1000)
    alpha: float = Field(gt=0.0, lt=1.0, default=0.05)

    # Within-subjects only
    corr_structure: CorrStructure = "cs"
    corr: float = Field(default=0.5)


def normal_pdf(x: np.ndarray, mean: float, sd: float) -> np.ndarray:
    coef = 1.0 / (sd * np.sqrt(2.0 * np.pi))
    z = (x - mean) / sd
    return coef * np.exp(-0.5 * z * z)


def parse_factor_levels(label: str) -> Tuple[Optional[int], Optional[int]]:
    ma = re.search(r"\bA(\d+)\b", label)
    mb = re.search(r"\bB(\d+)\b", label)
    a = int(ma.group(1)) - 1 if ma else None
    b = int(mb.group(1)) - 1 if mb else None
    return a, b


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    half = (z * np.sqrt((phat * (1 - phat) / n) + (z * z) / (4 * n * n))) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return float(lo), float(hi)


def validate_design(n_factors: int, levels: List[int], conditions: List[ConditionSpec]) -> Optional[str]:
    if n_factors == 1 and len(levels) != 1:
        return "For 1 factor, provide exactly one levels value."
    if n_factors == 2 and len(levels) != 2:
        return "For 2 factors, provide exactly two levels values."
    if any(L < 1 or L > 50 for L in levels):
        return "Levels must be between 1 and 50."
    expected = int(np.prod(levels))
    if len(conditions) != expected:
        return f"Expected {expected} conditions but got {len(conditions)}."
    return None


def order_conditions(payload_conditions: List[ConditionSpec], n_factors: int, levels: List[int]) -> List[ConditionSpec]:
    if n_factors == 1:
        ordered: List[ConditionSpec] = [None] * levels[0]  # type: ignore
        for c in payload_conditions:
            ai, _ = parse_factor_levels(c.label)
            if ai is None or ai < 0 or ai >= levels[0]:
                raise ValueError(f'Could not parse label "{c.label}". Use labels like "A1".')
            ordered[ai] = c
        if any(x is None for x in ordered):
            raise ValueError("Some factor levels were missing.")
        return ordered  # type: ignore

    a, b = levels
    ordered2: List[ConditionSpec] = [None] * (a * b)  # type: ignore
    for c in payload_conditions:
        ai, bi = parse_factor_levels(c.label)
        if ai is None or bi is None:
            raise ValueError(f'Could not parse label "{c.label}". Use labels like "A1 B2".')
        if not (0 <= ai < a and 0 <= bi < b):
            raise ValueError(f'Label "{c.label}" has indices outside the specified levels.')
        idx = ai * b + bi
        ordered2[idx] = c
    if any(x is None for x in ordered2):
        raise ValueError("Some cells were missing.")
    return ordered2  # type: ignore


def effective_sds(trial_sds: np.ndarray, trials: int) -> np.ndarray:
    # Mean of T iid trials has SD = sd / sqrt(T)
    return trial_sds / np.sqrt(float(trials))


def build_corr_matrix(k: int, structure: CorrStructure, corr: float) -> np.ndarray:
    if k < 1:
        raise ValueError("k must be at least 1.")
    if structure == "cs":
        r = float(corr)
        if not (-0.99 < r < 0.99):
            raise ValueError("For CS correlation, r must be between -0.99 and 0.99.")
        R = np.full((k, k), r, dtype=float)
        np.fill_diagonal(R, 1.0)
        return R

    rho = float(corr)
    if not (-0.99 < rho < 0.99):
        raise ValueError("For AR(1), rho must be between -0.99 and 0.99.")
    idx = np.arange(k)
    D = np.abs(idx[:, None] - idx[None, :])
    return rho ** D


def build_cov_from_sds(sds: np.ndarray, structure: CorrStructure, corr: float) -> np.ndarray:
    R = build_corr_matrix(sds.size, structure, corr)
    D = np.diag(sds)
    return D @ R @ D


def cholesky_or_error(cov: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as e:
        raise ValueError("Covariance matrix not positive definite. Reduce correlation or adjust SDs.") from e


# ---------- analysis ----------
def analyze_independent_t(g1: np.ndarray, g2: np.ndarray) -> dict:
    n1, n2 = g1.size, g2.size
    m1, m2 = g1.mean(), g2.mean()
    s1 = g1.std(ddof=1) if n1 > 1 else 0.0
    s2 = g2.std(ddof=1) if n2 > 1 else 0.0

    df = n1 + n2 - 2
    sp2 = (((n1 - 1) * s1**2) + ((n2 - 1) * s2**2)) / df if df > 0 else np.nan
    sp = np.sqrt(sp2)

    t = (m1 - m2) / (sp * np.sqrt(1.0 / n1 + 1.0 / n2)) if sp > 0 else np.nan
    p = 2.0 * stats.t.sf(np.abs(t), df) if np.isfinite(t) else np.nan
    d = (m1 - m2) / sp if sp > 0 else np.nan

    return {
        "kind": "ttest",
        "test_name": "Independent samples t test (pooled variance)",
        "t": float(t),
        "df": int(df),
        "p": float(p),
        "cohens_d": float(d),
    }


def analyze_one_way_between(groups: List[np.ndarray]) -> dict:
    k = len(groups)
    ns = np.array([g.size for g in groups], dtype=float)
    means = np.array([g.mean() for g in groups], dtype=float)
    grand = np.sum(ns * means) / np.sum(ns)

    ss_between = np.sum(ns * (means - grand) ** 2)
    ss_within = np.sum([np.sum((g - g.mean()) ** 2) for g in groups])
    ss_total = ss_between + ss_within

    df_between = k - 1
    df_within = int(np.sum(ns) - k)

    ms_between = ss_between / df_between if df_between > 0 else np.nan
    ms_within = ss_within / df_within if df_within > 0 else np.nan

    f = ms_between / ms_within
    p = stats.f.sf(f, df_between, df_within)
    eta2 = ss_between / ss_total if ss_total > 0 else np.nan

    return {
        "kind": "anova1",
        "test_name": "One way ANOVA",
        "df_between": int(df_between),
        "df_within": int(df_within),
        "F": float(f),
        "p": float(p),
        "eta2": float(eta2),
    }


def analyze_paired_t(y: np.ndarray) -> dict:
    d = y[:, 1] - y[:, 0]
    n = d.size
    df = n - 1
    md = d.mean()
    sd = d.std(ddof=1) if n > 1 else np.nan
    t = md / (sd / np.sqrt(n)) if sd > 0 else np.nan
    p = 2.0 * stats.t.sf(np.abs(t), df) if np.isfinite(t) else np.nan
    dz = md / sd if sd > 0 else np.nan
    return {
        "kind": "paired_ttest",
        "test_name": "Paired samples t test",
        "t": float(t),
        "df": int(df),
        "p": float(p),
        "cohens_dz": float(dz),
    }


def analyze_rm_anova1(y: np.ndarray) -> dict:
    n, k = y.shape
    if n < 2 or k < 3:
        return {"kind": "none", "test_name": "RM ANOVA requires at least 2 subjects and at least 3 levels."}

    gm = y.mean()
    cond_means = y.mean(axis=0)
    subj_means = y.mean(axis=1)

    ss_cond = n * np.sum((cond_means - gm) ** 2)
    ss_subj = k * np.sum((subj_means - gm) ** 2)
    ss_total = np.sum((y - gm) ** 2)
    ss_error = ss_total - ss_cond - ss_subj

    df_cond = k - 1
    df_error = (n - 1) * (k - 1)

    ms_cond = ss_cond / df_cond
    ms_error = ss_error / df_error if df_error > 0 else np.nan

    f = ms_cond / ms_error
    p = stats.f.sf(f, df_cond, df_error)
    partial_eta2 = ss_cond / (ss_cond + ss_error) if (ss_cond + ss_error) > 0 else np.nan

    return {
        "kind": "rm_anova1",
        "test_name": "One way repeated measures ANOVA (sphericity assumed)",
        "df_cond": int(df_cond),
        "df_error": int(df_error),
        "F": float(f),
        "p": float(p),
        "partial_eta2": float(partial_eta2),
    }


def analyze_rm_anova2(y: np.ndarray, a: int, b: int) -> dict:
    n = y.shape[0]
    if n < 2 or a < 2 or b < 2:
        return {"kind": "none", "test_name": "Two way RM ANOVA requires at least 2 subjects and at least 2 levels per factor."}

    grand = y.mean()
    Y_s__ = y.mean(axis=(1, 2))
    Y__a_ = y.mean(axis=(0, 2))
    Y___b = y.mean(axis=(0, 1))
    Y__ab = y.mean(axis=0)
    Y_s_a = y.mean(axis=2)
    Y_s__b = y.mean(axis=1)

    ss_a = b * n * np.sum((Y__a_ - grand) ** 2)
    ss_b = a * n * np.sum((Y___b - grand) ** 2)
    ss_ab = n * np.sum((Y__ab - Y__a_[:, None] - Y___b[None, :] + grand) ** 2)

    ss_sa = b * np.sum((Y_s_a - Y_s__[:, None] - Y__a_[None, :] + grand) ** 2)
    ss_sb = a * np.sum((Y_s__b - Y_s__[:, None] - Y___b[None, :] + grand) ** 2)

    res = (
        y
        - Y_s_a[:, :, None]
        - Y_s__b[:, None, :]
        - Y__ab[None, :, :]
        + Y_s__[:, None, None]
        + Y__a_[None, :, None]
        + Y___b[None, None, :]
        - grand
    )
    ss_sab = np.sum(res**2)

    df_a = a - 1
    df_b = b - 1
    df_ab = (a - 1) * (b - 1)
    df_sa = (n - 1) * (a - 1)
    df_sb = (n - 1) * (b - 1)
    df_sab = (n - 1) * (a - 1) * (b - 1)

    f_a = (ss_a / df_a) / (ss_sa / df_sa)
    f_b = (ss_b / df_b) / (ss_sb / df_sb)
    f_ab = (ss_ab / df_ab) / (ss_sab / df_sab)

    p_a = stats.f.sf(f_a, df_a, df_sa)
    p_b = stats.f.sf(f_b, df_b, df_sb)
    p_ab = stats.f.sf(f_ab, df_ab, df_sab)

    pet2_a = ss_a / (ss_a + ss_sa) if (ss_a + ss_sa) > 0 else np.nan
    pet2_b = ss_b / (ss_b + ss_sb) if (ss_b + ss_sb) > 0 else np.nan
    pet2_ab = ss_ab / (ss_ab + ss_sab) if (ss_ab + ss_sab) > 0 else np.nan

    return {
        "kind": "rm_anova2",
        "test_name": "Two way repeated measures ANOVA (sphericity assumed)",
        "table": [
            {"effect": "Factor A", "df1": int(df_a), "df2": int(df_sa), "F": float(f_a), "p": float(p_a), "partial_eta2": float(pet2_a)},
            {"effect": "Factor B", "df1": int(df_b), "df2": int(df_sb), "F": float(f_b), "p": float(p_b), "partial_eta2": float(pet2_b)},
            {"effect": "A × B", "df1": int(df_ab), "df2": int(df_sab), "F": float(f_ab), "p": float(p_ab), "partial_eta2": float(pet2_ab)},
        ],
    }


# ---------- download helpers ----------
def _levels_for_label(label: str) -> Tuple[str, str]:
    ai, bi = parse_factor_levels(label)
    A = f"A{ai+1}" if ai is not None else ""
    B = f"B{bi+1}" if bi is not None else ""
    return A, B


def _sanitize_col(label: str) -> str:
    # Column names with spaces are usually fine, but underscores are safer in many tools.
    return label.replace(" ", "_")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/simulate")
def simulate(payload: SimulateRequest):
    err = validate_design(payload.n_factors, payload.levels, payload.conditions)
    if err:
        return {"error": err}

    try:
        ordered = order_conditions(payload.conditions, payload.n_factors, payload.levels)
    except ValueError as e:
        return {"error": str(e)}

    labels = [c.label for c in ordered]
    means = np.array([c.mean for c in ordered], dtype=float)
    trial_sds = np.array([c.sd for c in ordered], dtype=float)

    trials = int(payload.trials_per_participant)
    sds = effective_sds(trial_sds, trials)  # participant mean SD

    # Plot “true distribution” of participant mean scores
    lo = float(np.min(means - 4.0 * sds))
    hi = float(np.max(means + 4.0 * sds))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(means.min() - 1.0), float(means.max() + 1.0)
    x = np.linspace(lo, hi, payload.pdf_points)
    curves: Dict[str, List[float]] = {lab: normal_pdf(x, m, sd).tolist() for lab, m, sd in zip(labels, means, sds)}

    rng = np.random.default_rng(payload.seed)

    samples: Dict[str, List[float]] = {}
    subject_preview: Optional[List[List[float]]] = None
    analysis: dict = {"kind": "none", "test_name": "No test computed."}

    if payload.design_type == "between":
        n = payload.sample_size

        if payload.n_factors == 1:
            group_arrays: List[np.ndarray] = []
            for m, sd, lab in zip(means, sds, labels):
                g = rng.normal(loc=m, scale=sd, size=n)
                samples[lab] = g.tolist()
                group_arrays.append(g)

            if n >= 2:
                if payload.levels[0] == 2:
                    analysis = analyze_independent_t(group_arrays[0], group_arrays[1])
                elif payload.levels[0] >= 3:
                    analysis = analyze_one_way_between(group_arrays)

        else:
            a, b = payload.levels
            y = np.zeros((a, b, n), dtype=float)
            for c, sd_eff in zip(ordered, sds):
                ai, bi = parse_factor_levels(c.label)
                assert ai is not None and bi is not None
                cell = rng.normal(loc=c.mean, scale=sd_eff, size=n)
                samples[c.label] = cell.tolist()
                y[ai, bi, :] = cell

            grand = y.mean()
            mean_a = y.mean(axis=(1, 2))
            mean_b = y.mean(axis=(0, 2))
            mean_ab = y.mean(axis=2)

            ss_a = b * n * np.sum((mean_a - grand) ** 2)
            ss_b = a * n * np.sum((mean_b - grand) ** 2)
            ss_ab = n * np.sum((mean_ab - mean_a[:, None] - mean_b[None, :] + grand) ** 2)
            ss_e = np.sum((y - mean_ab[:, :, None]) ** 2)

            df_a = a - 1
            df_b = b - 1
            df_ab = (a - 1) * (b - 1)
            df_e = a * b * (n - 1)

            ms_e = ss_e / df_e if df_e > 0 else np.nan
            f_a = (ss_a / df_a) / ms_e
            f_b = (ss_b / df_b) / ms_e
            f_ab = (ss_ab / df_ab) / ms_e

            p_a = stats.f.sf(f_a, df_a, df_e)
            p_b = stats.f.sf(f_b, df_b, df_e)
            p_ab = stats.f.sf(f_ab, df_ab, df_e)

            analysis = {
                "kind": "anova2",
                "test_name": "Two way ANOVA (between subjects, balanced)",
                "table": [
                    {"effect": "Factor A", "df": int(df_a), "F": float(f_a), "p": float(p_a)},
                    {"effect": "Factor B", "df": int(df_b), "F": float(f_b), "p": float(p_b)},
                    {"effect": "A × B", "df": int(df_ab), "F": float(f_ab), "p": float(p_ab)},
                ],
            }

    else:
        n_subj = payload.sample_size
        k = len(labels)

        try:
            cov = build_cov_from_sds(sds, payload.corr_structure, payload.corr)
            L = cholesky_or_error(cov)
        except ValueError as e:
            return {"error": str(e)}

        z = rng.standard_normal((n_subj, k))
        y_flat = z @ L.T + means[None, :]  # subject mean scores per condition

        for j, lab in enumerate(labels):
            samples[lab] = y_flat[:, j].tolist()

        p = int(min(payload.preview_subjects, n_subj))
        if p > 0:
            subject_preview = y_flat[:p, :].tolist()

        if n_subj >= 2:
            if payload.n_factors == 1:
                if payload.levels[0] == 2:
                    analysis = analyze_paired_t(y_flat[:, :2])
                elif payload.levels[0] >= 3:
                    analysis = analyze_rm_anova1(y_flat)
            else:
                a, b = payload.levels
                y = y_flat.reshape(n_subj, a, b)
                analysis = analyze_rm_anova2(y, a, b)

    # Cache dataset used for boxplot and inferential test
    global LAST_DATASET
    LAST_DATASET = {
        "design_type": payload.design_type,
        "n_factors": payload.n_factors,
        "levels": payload.levels,
        "trials_per_participant": trials,
        "corr_structure": payload.corr_structure,
        "corr": float(payload.corr),
        "condition_order": labels,
        "samples": samples,  # participant mean scores, exactly what plots/tests use
    }

    return {
        "x": x.tolist(),
        "curves": curves,
        "samples": samples,
        "analysis": analysis,
        "design_type": payload.design_type,
        "condition_order": labels,
        "subject_preview": subject_preview,
        "trials_per_participant": trials,
    }


@app.post("/api/montecarlo")
def montecarlo(payload: MonteCarloRequest):
    err = validate_design(payload.n_factors, payload.levels, payload.conditions)
    if err:
        return {"error": err}

    try:
        ordered = order_conditions(payload.conditions, payload.n_factors, payload.levels)
    except ValueError as e:
        return {"error": str(e)}

    rng = np.random.default_rng(payload.seed)

    means = np.array([c.mean for c in ordered], dtype=float)
    trial_sds = np.array([c.sd for c in ordered], dtype=float)
    trials = int(payload.trials_per_participant)
    sds = effective_sds(trial_sds, trials)

    n = payload.sample_size
    s = payload.n_sims
    alpha = float(payload.alpha)
    chunk = 250
    idx = 0

    if payload.design_type == "between":
        if payload.n_factors == 1:
            a = payload.levels[0]

            if a == 2:
                sig = 0
                while idx < s:
                    m = min(chunk, s - idx)
                    g1 = rng.normal(loc=means[0], scale=sds[0], size=(m, n))
                    g2 = rng.normal(loc=means[1], scale=sds[1], size=(m, n))

                    m1 = g1.mean(axis=1)
                    m2 = g2.mean(axis=1)
                    v1 = g1.var(axis=1, ddof=1)
                    v2 = g2.var(axis=1, ddof=1)

                    df = 2 * n - 2
                    sp2 = (((n - 1) * v1) + ((n - 1) * v2)) / df
                    sp = np.sqrt(sp2)
                    t = (m1 - m2) / (sp * np.sqrt(2.0 / n))
                    pvals = 2.0 * stats.t.sf(np.abs(t), df)
                    pvals = np.nan_to_num(pvals, nan=1.0)

                    sig += int((pvals < alpha).sum())
                    idx += m

                power = sig / s
                lo, hi = wilson_ci(sig, s)
                return {
                    "kind": "ttest",
                    "test_name": "Monte Carlo power for independent samples t test",
                    "alpha": alpha,
                    "n_sims": s,
                    "power": float(power),
                    "ci95": {"lo": lo, "hi": hi},
                    "trials_per_participant": trials,
                }

            if a >= 3:
                sig = 0
                df_between = a - 1
                df_within = a * (n - 1)
                while idx < s:
                    m = min(chunk, s - idx)
                    y = rng.normal(loc=means[None, :, None], scale=sds[None, :, None], size=(m, a, n))

                    means_sim = y.mean(axis=2)
                    grand = means_sim.mean(axis=1, keepdims=True)
                    ss_between = n * np.sum((means_sim - grand) ** 2, axis=1)
                    ss_within = np.sum((y - means_sim[:, :, None]) ** 2, axis=(1, 2))

                    f = (ss_between / df_between) / (ss_within / df_within)
                    pvals = stats.f.sf(f, df_between, df_within)
                    pvals = np.nan_to_num(pvals, nan=1.0)

                    sig += int((pvals < alpha).sum())
                    idx += m

                power = sig / s
                lo, hi = wilson_ci(sig, s)
                return {
                    "kind": "anova1",
                    "test_name": "Monte Carlo power for one way ANOVA",
                    "alpha": alpha,
                    "n_sims": s,
                    "power": float(power),
                    "ci95": {"lo": lo, "hi": hi},
                    "df": {"between": int(df_between), "within": int(df_within)},
                    "trials_per_participant": trials,
                }

            return {"error": "No test available for this design."}

        # Between: 2-factor (balanced). Power for A, B, AxB and any.
        a, b = payload.levels
        if a < 2 or b < 2:
            return {"error": "Two way ANOVA power requires at least 2 levels per factor."}

        means_ab = means.reshape(a, b)
        sds_ab = sds.reshape(a, b)

        sig_a = sig_b = sig_ab = sig_any = 0

        while idx < s:
            m = min(chunk, s - idx)
            y = rng.normal(
                loc=means_ab[None, :, :, None],
                scale=sds_ab[None, :, :, None],
                size=(m, a, b, n),
            )

            grand = y.mean(axis=(1, 2, 3))
            mean_a = y.mean(axis=(2, 3))
            mean_b = y.mean(axis=(1, 3))
            mean_ab = y.mean(axis=3)

            ss_a = b * n * np.sum((mean_a - grand[:, None]) ** 2, axis=1)
            ss_b = a * n * np.sum((mean_b - grand[:, None]) ** 2, axis=1)
            ss_int = n * np.sum(
                (mean_ab - mean_a[:, :, None] - mean_b[:, None, :] + grand[:, None, None]) ** 2,
                axis=(1, 2),
            )
            ss_e = np.sum((y - mean_ab[:, :, :, None]) ** 2, axis=(1, 2, 3))

            df_a = a - 1
            df_b = b - 1
            df_int = (a - 1) * (b - 1)
            df_e = a * b * (n - 1)

            ms_e = ss_e / df_e
            f_a = (ss_a / df_a) / ms_e
            f_b = (ss_b / df_b) / ms_e
            f_int = (ss_int / df_int) / ms_e

            p_a = stats.f.sf(f_a, df_a, df_e)
            p_b = stats.f.sf(f_b, df_b, df_e)
            p_int = stats.f.sf(f_int, df_int, df_e)

            sa = p_a < alpha
            sb = p_b < alpha
            sint = p_int < alpha
            sany = sa | sb | sint

            sig_a += int(sa.sum())
            sig_b += int(sb.sum())
            sig_ab += int(sint.sum())
            sig_any += int(sany.sum())
            idx += m

        def pack(k_sig: int) -> dict:
            power = k_sig / s
            lo, hi = wilson_ci(k_sig, s)
            return {"power": float(power), "ci95": {"lo": lo, "hi": hi}, "significant": int(k_sig)}

        return {
            "kind": "anova2",
            "test_name": "Monte Carlo power for two way ANOVA (between subjects, balanced)",
            "alpha": alpha,
            "n_sims": s,
            "effects": {"A": pack(sig_a), "B": pack(sig_b), "AxB": pack(sig_ab), "any_effect": pack(sig_any)},
            "trials_per_participant": trials,
        }

    # Within-subjects Monte Carlo
    k = len(ordered)
    try:
        cov = build_cov_from_sds(sds, payload.corr_structure, payload.corr)
        L = cholesky_or_error(cov)
    except ValueError as e:
        return {"error": str(e)}

    if payload.n_factors == 1:
        a = payload.levels[0]

        if a == 2:
            sig = 0
            while idx < s:
                m = min(chunk, s - idx)
                z = rng.standard_normal((m, n, k))
                y = z @ L.T + means[None, None, :]  # (m,n,2)

                d = y[:, :, 1] - y[:, :, 0]
                md = d.mean(axis=1)
                sd_d = d.std(axis=1, ddof=1)
                t = md / (sd_d / np.sqrt(n))
                pvals = 2.0 * stats.t.sf(np.abs(t), n - 1)
                pvals = np.nan_to_num(pvals, nan=1.0)

                sig += int((pvals < alpha).sum())
                idx += m

            power = sig / s
            lo, hi = wilson_ci(sig, s)
            return {
                "kind": "paired_ttest",
                "test_name": "Monte Carlo power for paired samples t test",
                "alpha": alpha,
                "n_sims": s,
                "power": float(power),
                "ci95": {"lo": lo, "hi": hi},
                "corr_structure": payload.corr_structure,
                "corr": float(payload.corr),
                "trials_per_participant": trials,
            }

        if a >= 3:
            sig = 0
            df_cond = a - 1
            df_err = (n - 1) * (a - 1)

            while idx < s:
                m = min(chunk, s - idx)
                z = rng.standard_normal((m, n, k))
                y = z @ L.T + means[None, None, :]  # (m,n,a)

                gm = y.mean(axis=(1, 2))
                cond_means = y.mean(axis=1)
                subj_means = y.mean(axis=2)

                ss_cond = n * np.sum((cond_means - gm[:, None]) ** 2, axis=1)
                ss_subj = a * np.sum((subj_means - gm[:, None]) ** 2, axis=1)
                ss_total = np.sum((y - gm[:, None, None]) ** 2, axis=(1, 2))
                ss_err = ss_total - ss_cond - ss_subj

                f = (ss_cond / df_cond) / (ss_err / df_err)
                pvals = stats.f.sf(f, df_cond, df_err)
                pvals = np.nan_to_num(pvals, nan=1.0)

                sig += int((pvals < alpha).sum())
                idx += m

            power = sig / s
            lo, hi = wilson_ci(sig, s)
            return {
                "kind": "rm_anova1",
                "test_name": "Monte Carlo power for one way repeated measures ANOVA (sphericity assumed)",
                "alpha": alpha,
                "n_sims": s,
                "power": float(power),
                "ci95": {"lo": lo, "hi": hi},
                "df": {"cond": int(df_cond), "error": int(df_err)},
                "corr_structure": payload.corr_structure,
                "corr": float(payload.corr),
                "trials_per_participant": trials,
            }

        return {"error": "No test available for this design."}

    # Two-factor within: power for A, B, AxB and any
    a, b = payload.levels
    if a < 2 or b < 2:
        return {"error": "Two way repeated measures power requires at least 2 levels per factor."}

    df_a = a - 1
    df_b = b - 1
    df_ab = (a - 1) * (b - 1)
    df_sa = (n - 1) * (a - 1)
    df_sb = (n - 1) * (b - 1)
    df_sab = (n - 1) * (a - 1) * (b - 1)

    sig_a = sig_b = sig_ab = sig_any = 0

    while idx < s:
        m = min(chunk, s - idx)
        z = rng.standard_normal((m, n, k))
        y_flat = z @ L.T + means[None, None, :]
        y = y_flat.reshape(m, n, a, b)

        grand = y.mean(axis=(1, 2, 3))
        Y_s__ = y.mean(axis=(2, 3))
        Y__a_ = y.mean(axis=(1, 3))
        Y___b = y.mean(axis=(1, 2))
        Y__ab = y.mean(axis=1)
        Y_s_a = y.mean(axis=3)
        Y_s__b = y.mean(axis=2)

        ss_a = b * n * np.sum((Y__a_ - grand[:, None]) ** 2, axis=1)
        ss_b = a * n * np.sum((Y___b - grand[:, None]) ** 2, axis=1)
        ss_ab = n * np.sum((Y__ab - Y__a_[:, :, None] - Y___b[:, None, :] + grand[:, None, None]) ** 2, axis=(1, 2))

        ss_sa = b * np.sum((Y_s_a - Y_s__[:, :, None] - Y__a_[:, None, :] + grand[:, None, None]) ** 2, axis=(1, 2))
        ss_sb = a * np.sum((Y_s__b - Y_s__[:, :, None] - Y___b[:, None, :] + grand[:, None, None]) ** 2, axis=(1, 2))

        res = (
            y
            - Y_s_a[:, :, :, None]
            - Y_s__b[:, :, None, :]
            - Y__ab[:, None, :, :]
            + Y_s__[:, :, None, None]
            + Y__a_[:, None, :, None]
            + Y___b[:, None, None, :]
            - grand[:, None, None, None]
        )
        ss_sab = np.sum(res**2, axis=(1, 2, 3))

        f_a = (ss_a / df_a) / (ss_sa / df_sa)
        f_b = (ss_b / df_b) / (ss_sb / df_sb)
        f_int = (ss_ab / df_ab) / (ss_sab / df_sab)

        p_a = stats.f.sf(f_a, df_a, df_sa)
        p_b = stats.f.sf(f_b, df_b, df_sb)
        p_int = stats.f.sf(f_int, df_ab, df_sab)

        sa = p_a < alpha
        sb = p_b < alpha
        sint = p_int < alpha
        sany = sa | sb | sint

        sig_a += int(sa.sum())
        sig_b += int(sb.sum())
        sig_ab += int(sint.sum())
        sig_any += int(sany.sum())
        idx += m

    def pack(k_sig: int) -> dict:
        power = k_sig / s
        lo, hi = wilson_ci(k_sig, s)
        return {"power": float(power), "ci95": {"lo": lo, "hi": hi}, "significant": int(k_sig)}

    return {
        "kind": "rm_anova2",
        "test_name": "Monte Carlo power for two way repeated measures ANOVA (sphericity assumed)",
        "alpha": alpha,
        "n_sims": s,
        "effects": {"A": pack(sig_a), "B": pack(sig_b), "AxB": pack(sig_ab), "any_effect": pack(sig_any)},
        "corr_structure": payload.corr_structure,
        "corr": float(payload.corr),
        "trials_per_participant": trials,
    }


@app.get("/api/download.csv")
def download_csv():
    if LAST_DATASET is None:
        return {"error": "No dataset available yet. Click 'Draw sample and run test' first."}

    design_type: str = LAST_DATASET["design_type"]
    n_factors: int = LAST_DATASET["n_factors"]
    trials: int = LAST_DATASET["trials_per_participant"]
    samples: Dict[str, List[float]] = LAST_DATASET["samples"]
    order: List[str] = LAST_DATASET.get("condition_order", list(samples.keys()))

    out = io.StringIO()
    writer = None

    if design_type == "between":
        # Long format: one row per participant
        # Columns: subject, score, A, (B), condition
        fields = ["subject", "score", "A"]
        if n_factors == 2:
            fields.append("B")
        fields.append("condition")

        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()

        for lab in order:
            ys = samples.get(lab, [])
            A, B = _levels_for_label(lab)
            for i, y in enumerate(ys, start=1):
                row = {"subject": f"{lab}_S{i}", "score": y, "A": A, "condition": lab}
                if n_factors == 2:
                    row["B"] = B
                writer.writerow(row)

        filename = f"simulated_data_between_{n_factors}factor_trials{trials}.csv"

    else:
        # Wide format: one row per subject, one column per condition.
        # This is very convenient for JASP paired t tests and RM ANOVA.
        # Columns: subject, A1, A2, ... (or A1_B1, ...)
        if not order:
            return {"error": "No data available to download."}

        n_subj = len(samples[order[0]])
        cond_cols = [_sanitize_col(lab) for lab in order]
        fields = ["subject"] + cond_cols

        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()

        for sidx in range(n_subj):
            row = {"subject": f"S{sidx+1}"}
            for lab, col in zip(order, cond_cols):
                row[col] = samples[lab][sidx]
            writer.writerow(row)

        filename = f"simulated_data_within_{n_factors}factor_trials{trials}.csv"

    csv_bytes = out.getvalue().encode("utf-8")
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}

    return StreamingResponse(io.BytesIO(csv_bytes), media_type="text/csv; charset=utf-8", headers=headers)
