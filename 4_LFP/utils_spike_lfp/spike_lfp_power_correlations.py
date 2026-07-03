#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spike_lfp_power_correlations.py
===============================

Corrélations time-resolved entre firing rate single-unit et puissance/enveloppe
Hilbert LFP, à partir de la base commune spike-LFP.

Principe
--------
Entrées principales :
    - CommonSessionBundle produit par spike_lfp_common_preprocess ;
    - spikes unitaires en référentiel micro, sous forme dict {unit_id: spike_times_s}
      ou objet convertible ;
    - exports Hilbert déjà chargés dans bundle.hilbert.

Les epochs Hilbert attendues ont la forme :
    band_epochs[band] = array (n_trials, n_channels, n_times)

L'axe temps Hilbert est concaténé :
    - temps < 0 : fenêtre pré-stim extraite avant la stimulation ;
    - temps >= 0 : fenêtre post-stim extraite après fin de stimulation + epsilon.

Pour chaque largeur de bin demandée, typiquement 100 ou 500 ms :
    1. on construit des bins temporels pré et post ;
    2. on calcule le firing rate du neurone dans le bin micro correspondant ;
    3. on normalise ce firing rate par la baseline pré-stim du même trial ;
    4. on moyenne la puissance/enveloppe Hilbert du même band/channel/trial/bin ;                     [LFP brut ?? quid du normalisé ? z-score ? log-ratio ?]
    5. on corrèle FR normalisé et power LFP across trials.

Sorties
-------
    - table longue optionnelle : unit × trial × channel × band × bin ;
    - table résumé : corrélations Pearson/Spearman par unit × channel × band × time bin.

Remarques
---------
- Les spikes restent en référentiel micro ; le LFP/Hilbert reste en référentiel macro.
  Les deux sont associés par l'index de trial et le temps relatif pré/post.
- La largeur de bin 100/500 ms correspond ici à un lissage rectangulaire :
  FR = nombre de spikes dans le bin / durée du bin.                                          [quid du deadfile ?]
- Pour une analyse plus strictement "post-stim", laisser windows_to_correlate=('post',).
- Le LFP Hilbert étant déjà normalisé dans ton pipeline, la colonne par défaut
  utilisée pour la corrélation est lfp_power_bin_mean. Des colonnes LFP normalisées
  par le pré-trial sont quand même produites.

Auteure : Aube Darves-Bornoz
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from spike_lfp_common_preprocess import (
        CommonSessionBundle,
        as_spike_times_array,
        get_spikes_in_interval,
        parse_bipolar_shaft,
        channel_locality_for_trial,
        safe_name,
    )

# =============================================================================
# CONFIGURATION
# =============================================================================

FRNormMethod = Literal["raw", "subtract", "percent", "logratio", "zscore"]
LFPValueColumn = Literal[
    "lfp_power_bin_mean",
    "lfp_power_pre_subtract",
    "lfp_power_pre_percent",
    "lfp_power_pre_logratio",
    "lfp_power_pre_zscore",
]
CorrelationMethod = Literal["spearman", "pearson"]
WindowName = Literal["pre", "post"]
GroupingName = Literal["all", "group_label", "locality", "group_label_x_locality"]
SmoothingMethod = Literal["boxcar_bin", "gaussian_bins", "none"]


@dataclass
class FRPowerCorrelationConfig:
    """Configuration des corrélations spike–power."""

    # Largeurs de bins / lissage rectangulaire.
    bin_width_ms_options: Tuple[float, ...] = (100.0, 500.0)

    # Par défaut, on teste les bins post-stim uniquement ; les bins pré restent
    # calculés pour normaliser le FR et éventuellement contrôler les corrélations pré.
    windows_to_correlate: Tuple[WindowName, ...] = ("post",)

    # Normalisation du firing rate par baseline pré-stim trial-wise.
    # logratio avec epsilon_fr_hz est robuste pour les unités à faible FR.
    fr_norm_method: FRNormMethod = "logratio"
    epsilon_fr_hz: float = 0.1

    # LFP utilisé dans la corrélation. Par défaut : valeur Hilbert exportée/binée.
    # Les colonnes LFP normalisées par baseline pré-trial sont aussi calculées.
    lfp_value_col: LFPValueColumn = "lfp_power_bin_mean"
    epsilon_lfp: float = 1e-12

    # Méthodes de corrélation et seuils minimaux.
    correlation_methods: Tuple[CorrelationMethod, ...] = ("spearman", "pearson")
    min_trials_for_corr: int = 6
    min_finite_pairs_for_corr: int = 6

    # Groupements statistiques : produit par défaut les corrélations globales,
    # par condition cognitive, par localité, et condition×localité.
    correlation_groupings: Tuple[GroupingName, ...] = (
        "all",
        "group_label",
        "locality",
        "group_label_x_locality",
    )

    # Sélection des canaux LFP selon localité par rapport au shaft stimulé.
    # Options : ('local',), ('distant',), ('local','distant'), ('all',).
    localities_to_include: Tuple[str, ...] = ("local", "distant")

    # Bandes Hilbert à tester. Si None, toutes les bandes chargées dans bundle.hilbert.
    bands_to_test: Optional[Tuple[str, ...]] = ("theta", "alpha", "beta", "low_gamma", "high_gamma")

    # Optionnel : sélectionner certains canaux / unités.
    channel_names: Optional[Tuple[str, ...]] = None
    unit_ids: Optional[Tuple[Any, ...]] = None

    # Lissage optionnel en plus du binning. Par défaut, la largeur du bin est
    # le lissage rectangulaire. gaussian_bins lisse sur les bins adjacents.
    smoothing_method: SmoothingMethod = "boxcar_bin"
    gaussian_sigma_bins: float = 1.0

    # Sauvegardes.
    save_trial_bin_tables: bool = True
    save_fr_table: bool = True
    save_lfp_table: bool = False
    compression: Optional[str] = None  # ex. "gzip" pour csv.gz si souhaité

    # Divers.
    verbose: bool = True


# =============================================================================
# HELPERS GÉNÉRAUX
# =============================================================================


def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg, flush=True)



def ensure_dir(path: str | Path) -> Path:
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()



def _jsonify(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj



def fdr_bh(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg global sur un vecteur de p-values."""
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if finite.sum() == 0:
        return q
    p_use = p[finite]
    m = len(p_use)
    order = np.argsort(p_use)
    ranked = p_use[order]
    q_sorted = ranked * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q_use = np.empty_like(p_use)
    q_use[order] = q_sorted
    q[finite] = q_use
    return q



def _csv_path(out_dir: Path, basename: str, compression: Optional[str]) -> Path:
    if compression == "gzip":
        return out_dir / f"{basename}.csv.gz"
    return out_dir / f"{basename}.csv"



def _to_csv(df: pd.DataFrame, out_dir: Path, basename: str, compression: Optional[str]) -> Path:
    fp = _csv_path(out_dir, basename, compression)
    df.to_csv(fp, index=False, compression=compression)
    return fp


# =============================================================================
# BINS TEMPORELS RELATIFS
# =============================================================================


def _make_edges_exact(t0: float, t1: float, bin_width_s: float, include_partial: bool = False) -> np.ndarray:
    if bin_width_s <= 0:
        raise ValueError("bin_width_s doit être > 0")
    n_full = int(np.floor((t1 - t0) / bin_width_s + 1e-12))
    edges = t0 + np.arange(n_full + 1) * bin_width_s
    if include_partial and edges[-1] < t1 - 1e-12:
        edges = np.r_[edges, t1]
    return edges



def make_relative_bin_table(pre_length: float,
                            post_length: float,
                            bin_width_s: float,
                            include_partial_bins: bool = False) -> pd.DataFrame:
    """
    Construit les bins relatifs pré et post.

    Pré :  [-pre_length, 0)
    Post : [0, post_length)

    Les bins pré et post sont gardés séparés pour ne jamais lisser/corréler
    à travers l'intervalle stimulation + epsilon.
    """
    rows: List[Dict[str, Any]] = []

    pre_edges = _make_edges_exact(-float(pre_length), 0.0, bin_width_s, include_partial_bins)
    post_edges = _make_edges_exact(0.0, float(post_length), bin_width_s, include_partial_bins)

    idx = 0
    for window, edges in [("pre", pre_edges), ("post", post_edges)]:
        for i in range(len(edges) - 1):
            b0 = float(edges[i])
            b1 = float(edges[i + 1])
            if b1 <= b0:
                continue
            rows.append({
                "time_bin_id": idx,
                "window": window,
                "bin_index_in_window": i,
                "bin_width_ms": float(bin_width_s * 1000.0),
                "bin_start_rel": b0,
                "bin_end_rel": b1,
                "bin_center_rel": (b0 + b1) / 2.0,
                "bin_duration_s": b1 - b0,
            })
            idx += 1

    return pd.DataFrame(rows)



def hilbert_indices_for_bin(times: np.ndarray, b0: float, b1: float) -> np.ndarray:
    """Indices de l'axe Hilbert dans [b0, b1)."""
    times = np.asarray(times, dtype=float)
    return np.where((times >= b0) & (times < b1))[0]



def relative_bin_to_absolute_micro(row: pd.Series, window: str, b0: float, b1: float) -> Tuple[float, float]:
    """
    Convertit un bin relatif en intervalle absolu micro.

    Important : dans le pipeline Hilbert, les temps pré sont concaténés de
    -pre_length à 0 mais correspondent à [pre_start, pre_end]. Les temps post
    correspondent à [post_start, post_end].
    """
    if window == "pre":
        pre_start = float(row["pre_start_micro"])
        pre_length = float(row["pre_end_micro"] - row["pre_start_micro"])
        t0 = pre_start + (float(b0) + pre_length)
        t1 = pre_start + (float(b1) + pre_length)
        return t0, t1
    if window == "post":
        post_start = float(row["post_start_micro"])
        t0 = post_start + float(b0)
        t1 = post_start + float(b1)
        return t0, t1
    raise ValueError(f"window invalide: {window}")


# =============================================================================
# SPIKES -> FIRING RATE BINNÉ ET NORMALISÉ
# =============================================================================


def gaussian_smooth_1d_nanaware(x: np.ndarray, sigma_bins: float) -> np.ndarray:
    """Lissage gaussien 1D simple, sans scipy.ndimage, NaN-aware."""
    x = np.asarray(x, dtype=float)
    if sigma_bins <= 0 or len(x) == 0:
        return x
    radius = max(1, int(math.ceil(4 * sigma_bins)))
    kx = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (kx / sigma_bins) ** 2)
    kernel /= kernel.sum()

    valid = np.isfinite(x).astype(float)
    x0 = np.where(np.isfinite(x), x, 0.0)
    num = np.convolve(x0, kernel, mode="same")
    den = np.convolve(valid, kernel, mode="same")
    out = np.full_like(x, np.nan, dtype=float)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out



def smooth_rates_within_windows(fr_df: pd.DataFrame,
                                method: SmoothingMethod,
                                sigma_bins: float) -> pd.DataFrame:
    """Lisse fr_hz séparément dans la fenêtre pré et post, par unit×trial×bin_width."""
    out = fr_df.copy()
    out["fr_hz_smooth"] = out["fr_hz"].astype(float)
    if method in {"none", "boxcar_bin"}:
        return out
    if method != "gaussian_bins":
        raise ValueError(f"smoothing_method inconnu: {method}")

    group_cols = ["unit_id", "trial_idx", "bin_width_ms", "window"]
    for _, idx in out.groupby(group_cols, sort=False).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        idx_arr = idx_arr[np.argsort(out.loc[idx_arr, "bin_index_in_window"].to_numpy())]
        vals = out.loc[idx_arr, "fr_hz"].to_numpy(float)
        out.loc[idx_arr, "fr_hz_smooth"] = gaussian_smooth_1d_nanaware(vals, sigma_bins=sigma_bins)
    return out



def normalize_fr_by_trial_pre(fr_df: pd.DataFrame,
                              method: FRNormMethod,
                              epsilon_fr_hz: float = 0.1) -> pd.DataFrame:
    """Ajoute les colonnes de baseline et fr_norm, par unit × trial × bin_width."""
    out = fr_df.copy()
    out["fr_pre_mean_hz"] = np.nan
    out["fr_pre_sd_hz"] = np.nan
    out["fr_norm"] = np.nan
    out["fr_norm_method"] = method

    group_cols = ["unit_id", "trial_idx", "bin_width_ms"]
    for _, idx in out.groupby(group_cols, sort=False).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        pre_idx = idx_arr[out.loc[idx_arr, "window"].to_numpy() == "pre"]
        pre_vals = out.loc[pre_idx, "fr_hz_smooth"].to_numpy(float)
        pre_vals = pre_vals[np.isfinite(pre_vals)]
        if len(pre_vals) == 0:
            continue
        mu = float(np.mean(pre_vals))
        sd = float(np.std(pre_vals, ddof=1)) if len(pre_vals) > 1 else np.nan
        vals = out.loc[idx_arr, "fr_hz_smooth"].to_numpy(float)

        if method == "raw":
            norm = vals
        elif method == "subtract":
            norm = vals - mu
        elif method == "percent":
            norm = 100.0 * (vals - mu) / (mu + epsilon_fr_hz)
        elif method == "logratio":
            norm = np.log((vals + epsilon_fr_hz) / (mu + epsilon_fr_hz))
        elif method == "zscore":
            if not np.isfinite(sd) or sd <= 0:
                norm = np.full_like(vals, np.nan, dtype=float)
            else:
                norm = (vals - mu) / sd
        else:
            raise ValueError(f"fr_norm_method inconnu: {method}")

        out.loc[idx_arr, "fr_pre_mean_hz"] = mu
        out.loc[idx_arr, "fr_pre_sd_hz"] = sd
        out.loc[idx_arr, "fr_norm"] = norm

    return out



def build_fr_bin_table_for_units(bundle: CommonSessionBundle,
                                 units: Dict[Any, Any],
                                 bin_tables: Dict[float, pd.DataFrame],
                                 cfg: FRPowerCorrelationConfig,
                                 dead_intervals_by_unit: Optional[Dict[Any, Any]] = None,
                                 unit_metadata: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calcule FR biné pour toutes les unités, tous les trials, toutes les largeurs de bin.

    Parameters
    ----------
    units : dict
        {unit_id: spike_times_s ou objet convertible par as_spike_times_array}
    dead_intervals_by_unit : dict optionnel
        {unit_id: intervals n×2 en secondes}.
    """
    trials = bundle.trials.reset_index(drop=True)
    selected_units = list(units.keys())
    if cfg.unit_ids is not None:
        wanted = set(cfg.unit_ids)
        selected_units = [u for u in selected_units if u in wanted]

    rows: List[Dict[str, Any]] = []
    dead_intervals_by_unit = dead_intervals_by_unit or {}

    for ui, unit_id in enumerate(selected_units):
        spk = as_spike_times_array(units[unit_id])
        dead = dead_intervals_by_unit.get(unit_id, None)
        if cfg.verbose and (ui + 1) % 20 == 0:
            log(f"[INFO] FR unités: {ui + 1}/{len(selected_units)}", cfg.verbose)

        for trial_idx, trial_row in trials.iterrows():
            for bin_width_ms, bin_df in bin_tables.items():
                for _, b in bin_df.iterrows():
                    t0, t1 = relative_bin_to_absolute_micro(
                        trial_row,
                        window=str(b["window"]),
                        b0=float(b["bin_start_rel"]),
                        b1=float(b["bin_end_rel"]),
                    )
                    spk_bin = get_spikes_in_interval(spk, t0, t1, dead_intervals=dead)
                    dur = max(float(t1 - t0), np.finfo(float).eps)
                    n_spikes = int(len(spk_bin))
                    rows.append({
                        "session": bundle.session_name,
                        "unit_id": unit_id,
                        "trial_idx": int(trial_idx),
                        "stim_index": int(trial_row["stim_index"]),
                        "group_label": trial_row.get("group_label", np.nan),
                        "cog_labels": trial_row.get("cog_labels", np.nan),
                        "label_stim": trial_row.get("label_stim", np.nan),
                        "stim_shaft": trial_row.get("stim_shaft", np.nan),
                        "stim_bipolar_label": trial_row.get("stim_bipolar_label", np.nan),
                        "bin_width_ms": float(bin_width_ms),
                        "time_bin_id": int(b["time_bin_id"]),
                        "window": str(b["window"]),
                        "bin_index_in_window": int(b["bin_index_in_window"]),
                        "bin_start_rel": float(b["bin_start_rel"]),
                        "bin_end_rel": float(b["bin_end_rel"]),
                        "bin_center_rel": float(b["bin_center_rel"]),
                        "bin_duration_s": float(b["bin_duration_s"]),
                        "bin_abs_start_micro": float(t0),
                        "bin_abs_end_micro": float(t1),
                        "n_spikes": n_spikes,
                        "fr_hz": n_spikes / dur,
                    })

    fr = pd.DataFrame(rows)
    if fr.empty:
        raise RuntimeError("Aucun firing rate calculé. Vérifie units, trials et binning.")

    fr = smooth_rates_within_windows(
        fr,
        method=cfg.smoothing_method,
        sigma_bins=cfg.gaussian_sigma_bins,
    )
    fr = normalize_fr_by_trial_pre(
        fr,
        method=cfg.fr_norm_method,
        epsilon_fr_hz=cfg.epsilon_fr_hz,
    )

    if unit_metadata is not None and not unit_metadata.empty:
        unit_metadata = unit_metadata.copy()
        if "unit_id" not in unit_metadata.columns:
            raise ValueError("unit_metadata doit contenir une colonne 'unit_id'")
        fr = fr.merge(unit_metadata, on="unit_id", how="left", validate="many_to_one")

    return fr


# =============================================================================
# HILBERT POWER -> VALEUR BINNÉE
# =============================================================================


def _get_bp_names_from_bundle(bundle: CommonSessionBundle) -> List[str]:
    if bundle.hilbert is None:
        raise ValueError("bundle.hilbert est None. Recharge le bundle avec load_hilbert=True.")
    meta = bundle.hilbert.get("metadata", {})
    bp_names = meta.get("bipolar_names", None)
    if not bp_names:
        raise ValueError("bipolar_names absent de bundle.hilbert['metadata']")
    return list(bp_names)



def _select_channels(bp_names: Sequence[str], cfg: FRPowerCorrelationConfig) -> List[Tuple[int, str]]:
    if cfg.channel_names is None:
        return [(i, ch) for i, ch in enumerate(bp_names)]
    wanted = set(cfg.channel_names)
    return [(i, ch) for i, ch in enumerate(bp_names) if ch in wanted]



def normalize_lfp_by_trial_pre(power_df: pd.DataFrame,
                               epsilon_lfp: float = 1e-12) -> pd.DataFrame:
    """Ajoute des colonnes LFP normalisées par baseline pré-trial/ch/band/bin_width."""
    out = power_df.copy()
    for col in [
        "lfp_power_pre_mean",
        "lfp_power_pre_sd",
        "lfp_power_pre_subtract",
        "lfp_power_pre_percent",
        "lfp_power_pre_logratio",
        "lfp_power_pre_zscore",
    ]:
        out[col] = np.nan

    group_cols = ["band", "channel_idx", "trial_idx", "bin_width_ms"]
    for _, idx in out.groupby(group_cols, sort=False).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        pre_idx = idx_arr[out.loc[idx_arr, "window"].to_numpy() == "pre"]
        pre_vals = out.loc[pre_idx, "lfp_power_bin_mean"].to_numpy(float)
        pre_vals = pre_vals[np.isfinite(pre_vals)]
        if len(pre_vals) == 0:
            continue
        mu = float(np.mean(pre_vals))
        sd = float(np.std(pre_vals, ddof=1)) if len(pre_vals) > 1 else np.nan
        vals = out.loc[idx_arr, "lfp_power_bin_mean"].to_numpy(float)

        out.loc[idx_arr, "lfp_power_pre_mean"] = mu
        out.loc[idx_arr, "lfp_power_pre_sd"] = sd
        out.loc[idx_arr, "lfp_power_pre_subtract"] = vals - mu
        out.loc[idx_arr, "lfp_power_pre_percent"] = 100.0 * (vals - mu) / (mu + epsilon_lfp)
        out.loc[idx_arr, "lfp_power_pre_logratio"] = np.log((vals + epsilon_lfp) / (mu + epsilon_lfp))
        if np.isfinite(sd) and sd > 0:
            out.loc[idx_arr, "lfp_power_pre_zscore"] = (vals - mu) / sd

    return out



def build_lfp_power_bin_table_for_band(bundle: CommonSessionBundle,
                                       band: str,
                                       bin_tables: Dict[float, pd.DataFrame],
                                       cfg: FRPowerCorrelationConfig) -> pd.DataFrame:
    """Binne les epochs Hilbert d'une bande en trial × channel × bin."""
    if bundle.hilbert is None:
        raise ValueError("bundle.hilbert est None. Recharge le bundle avec load_hilbert=True.")
    epochs_by_band = bundle.hilbert.get("epochs_by_band", {})
    if band not in epochs_by_band:
        raise FileNotFoundError(f"Bande Hilbert absente dans bundle: {band}")

    arr = np.asarray(epochs_by_band[band])  # peut être memmap
    if arr.ndim != 3:
        raise ValueError(f"{band}: shape attendue (n_trials,n_channels,n_times), reçue {arr.shape}")

    times = np.asarray(bundle.hilbert["times"], dtype=float)
    trials = bundle.trials.reset_index(drop=True)
    bp_names = _get_bp_names_from_bundle(bundle)
    selected_channels = _select_channels(bp_names, cfg)

    # Sécurités d'alignement.
    n_trials = min(len(trials), arr.shape[0])
    if len(trials) != arr.shape[0]:
        log(
            f"[WARN] {bundle.session_name} {band}: n_trials common={len(trials)} vs Hilbert={arr.shape[0]}; "
            f"utilisation des {n_trials} premiers trials.",
            cfg.verbose,
        )

    locality_allowed = set(cfg.localities_to_include)
    include_all_localities = "all" in locality_allowed
    rows: List[Dict[str, Any]] = []

    # Pré-calcul indices temporels par bin.
    idx_by_width_bin: Dict[Tuple[float, int], np.ndarray] = {}
    for bin_width_ms, bin_df in bin_tables.items():
        for _, b in bin_df.iterrows():
            idx_by_width_bin[(float(bin_width_ms), int(b["time_bin_id"]))] = hilbert_indices_for_bin(
                times,
                float(b["bin_start_rel"]),
                float(b["bin_end_rel"]),
            )

    for trial_idx in range(n_trials):
        trial_row = trials.iloc[trial_idx]
        for ch_idx, ch_name in selected_channels:
            if ch_idx >= arr.shape[1]:
                continue
            locality = channel_locality_for_trial(ch_name, trial_row)
            if not include_all_localities and locality not in locality_allowed:
                continue
            channel_shaft = parse_bipolar_shaft(ch_name)

            for bin_width_ms, bin_df in bin_tables.items():
                for _, b in bin_df.iterrows():
                    tid = int(b["time_bin_id"])
                    t_idx = idx_by_width_bin[(float(bin_width_ms), tid)]
                    if len(t_idx) == 0:
                        val = np.nan
                        n_samples = 0
                    else:
                        vals = np.asarray(arr[trial_idx, ch_idx, t_idx], dtype=float)
                        val = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
                        n_samples = int(len(t_idx))
                    rows.append({
                        "session": bundle.session_name,
                        "band": band,
                        "trial_idx": int(trial_idx),
                        "stim_index": int(trial_row["stim_index"]),
                        "group_label": trial_row.get("group_label", np.nan),
                        "cog_labels": trial_row.get("cog_labels", np.nan),
                        "label_stim": trial_row.get("label_stim", np.nan),
                        "stim_shaft": trial_row.get("stim_shaft", np.nan),
                        "stim_bipolar_label": trial_row.get("stim_bipolar_label", np.nan),
                        "channel_idx": int(ch_idx),
                        "channel_name": ch_name,
                        "channel_shaft": channel_shaft,
                        "locality": locality,
                        "bin_width_ms": float(bin_width_ms),
                        "time_bin_id": tid,
                        "window": str(b["window"]),
                        "bin_index_in_window": int(b["bin_index_in_window"]),
                        "bin_start_rel": float(b["bin_start_rel"]),
                        "bin_end_rel": float(b["bin_end_rel"]),
                        "bin_center_rel": float(b["bin_center_rel"]),
                        "bin_duration_s": float(b["bin_duration_s"]),
                        "n_hilbert_samples_in_bin": n_samples,
                        "lfp_power_bin_mean": val,
                    })

    power = pd.DataFrame(rows)
    if power.empty:
        raise RuntimeError(f"Aucune valeur LFP binée pour band={band}")

    power = normalize_lfp_by_trial_pre(power, epsilon_lfp=cfg.epsilon_lfp)
    return power


# =============================================================================
# FUSION FR × LFP ET CORRÉLATIONS
# =============================================================================


def build_fr_power_trial_bin_table(fr_df: pd.DataFrame,
                                   lfp_df: pd.DataFrame,
                                   cfg: FRPowerCorrelationConfig) -> pd.DataFrame:
    """Fusionne FR et LFP sur trial × bin_width × time_bin."""
    key_cols = [
        "session",
        "trial_idx",
        "stim_index",
        "bin_width_ms",
        "time_bin_id",
        "window",
        "bin_index_in_window",
        "bin_start_rel",
        "bin_end_rel",
        "bin_center_rel",
        "bin_duration_s",
    ]

    keep_lfp_cols = key_cols + [
        "band",
        "channel_idx",
        "channel_name",
        "channel_shaft",
        "locality",
        "n_hilbert_samples_in_bin",
        "lfp_power_bin_mean",
        "lfp_power_pre_mean",
        "lfp_power_pre_sd",
        "lfp_power_pre_subtract",
        "lfp_power_pre_percent",
        "lfp_power_pre_logratio",
        "lfp_power_pre_zscore",
    ]
    keep_lfp_cols = [c for c in keep_lfp_cols if c in lfp_df.columns]

    # Colonnes communes group_label/cog_labels existent dans FR et LFP ; on garde celles du FR.
    keep_fr_cols = [c for c in fr_df.columns if c not in {"channel_idx", "channel_name", "channel_shaft", "locality", "band"}]

    merged = pd.merge(
        fr_df[keep_fr_cols],
        lfp_df[keep_lfp_cols],
        on=key_cols,
        how="inner",
        validate="many_to_many",
    )

    if cfg.windows_to_correlate is not None:
        merged = merged.loc[merged["window"].isin(cfg.windows_to_correlate)].reset_index(drop=True)

    return merged



def _corr_one(x: np.ndarray, y: np.ndarray, method: CorrelationMethod) -> Tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    n = int(len(x))
    if n < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan, n
    if method == "spearman":
        res = spearmanr(x, y)
        return float(res.statistic), float(res.pvalue), n
    if method == "pearson":
        res = pearsonr(x, y)
        return float(res.statistic), float(res.pvalue), n
    raise ValueError(f"Méthode inconnue: {method}")



def _add_condition_columns_for_grouping(df: pd.DataFrame, grouping: GroupingName) -> pd.DataFrame:
    out = df.copy()
    if grouping == "all":
        out["corr_grouping"] = "all"
        out["corr_condition"] = "all"
    elif grouping == "group_label":
        out["corr_grouping"] = "group_label"
        out["corr_condition"] = out["group_label"].astype(str)
    elif grouping == "locality":
        out["corr_grouping"] = "locality"
        out["corr_condition"] = out["locality"].astype(str)
    elif grouping == "group_label_x_locality":
        out["corr_grouping"] = "group_label_x_locality"
        out["corr_condition"] = out["group_label"].astype(str) + "::" + out["locality"].astype(str)
    else:
        raise ValueError(f"grouping inconnu: {grouping}")
    return out



def compute_correlations_from_trial_bin_table(trial_bin_df: pd.DataFrame,
                                              cfg: FRPowerCorrelationConfig) -> pd.DataFrame:
    """Calcule les corrélations across trials pour chaque unit×channel×band×time_bin."""
    if trial_bin_df.empty:
        return pd.DataFrame()
    if cfg.lfp_value_col not in trial_bin_df.columns:
        raise ValueError(f"Colonne LFP absente: {cfg.lfp_value_col}")

    all_rows: List[Dict[str, Any]] = []

    base_group_cols = [
        "session",
        "unit_id",
        "band",
        "channel_idx",
        "channel_name",
        "channel_shaft",
        "bin_width_ms",
        "window",
        "time_bin_id",
        "bin_index_in_window",
        "bin_start_rel",
        "bin_end_rel",
        "bin_center_rel",
    ]

    # Ajoute quelques colonnes unitaires si présentes.
    optional_unit_cols = [
        c for c in ["tetrode", "lobe_tt", "loca_tt", "lobe_tt_noLat", "loca_tt_noLat"]
        if c in trial_bin_df.columns
    ]

    for grouping in cfg.correlation_groupings:
        df_g = _add_condition_columns_for_grouping(trial_bin_df, grouping)
        group_cols = base_group_cols + optional_unit_cols + ["corr_grouping", "corr_condition"]

        for keys, sub in df_g.groupby(group_cols, dropna=False, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            info = dict(zip(group_cols, keys))
            n_trials_total = int(sub["trial_idx"].nunique())
            if n_trials_total < cfg.min_trials_for_corr:
                continue

            x = sub["fr_norm"].to_numpy(float)
            y = sub[cfg.lfp_value_col].to_numpy(float)

            for method in cfg.correlation_methods:
                r, p, n_pairs = _corr_one(x, y, method=method)
                if n_pairs < cfg.min_finite_pairs_for_corr:
                    r, p = np.nan, np.nan
                all_rows.append({
                    **info,
                    "method": method,
                    "fr_value_col": "fr_norm",
                    "fr_norm_method": cfg.fr_norm_method,
                    "lfp_value_col": cfg.lfp_value_col,
                    "n_trials_total": n_trials_total,
                    "n_finite_pairs": n_pairs,
                    "rho_or_r": r,
                    "p_value": p,
                    "fr_mean": float(np.nanmean(x)) if np.isfinite(x).any() else np.nan,
                    "fr_sd": float(np.nanstd(x, ddof=1)) if np.isfinite(x).sum() > 1 else np.nan,
                    "lfp_mean": float(np.nanmean(y)) if np.isfinite(y).any() else np.nan,
                    "lfp_sd": float(np.nanstd(y, ddof=1)) if np.isfinite(y).sum() > 1 else np.nan,
                })

    corr = pd.DataFrame(all_rows)
    if corr.empty:
        return corr

    # FDR par méthode et grouping, sur toute la session.
    corr["q_value_fdr_bh"] = np.nan
    for _, idx in corr.groupby(["method", "corr_grouping"], dropna=False).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        corr.loc[idx_arr, "q_value_fdr_bh"] = fdr_bh(corr.loc[idx_arr, "p_value"].to_numpy(float))

    return corr


# =============================================================================
# ORCHESTRATEUR SESSION
# =============================================================================


def _infer_bands_to_test(bundle: CommonSessionBundle, cfg: FRPowerCorrelationConfig) -> List[str]:
    if bundle.hilbert is None:
        raise ValueError("bundle.hilbert est None. Recharge le bundle avec load_hilbert=True.")
    available = list(bundle.hilbert.get("epochs_by_band", {}).keys())
    if cfg.bands_to_test is None:
        return available
    wanted = list(cfg.bands_to_test)
    missing = [b for b in wanted if b not in available]
    if missing:
        log(f"[WARN] bandes demandées absentes des exports Hilbert: {missing}", cfg.verbose)
    return [b for b in wanted if b in available]



def _write_config(out_dir: Path, cfg: FRPowerCorrelationConfig, bundle: CommonSessionBundle) -> Path:
    fp = out_dir / f"{bundle.session_name}_fr_power_corr_config.json"
    payload = {
        "session": bundle.session_name,
        "patient": bundle.patient,
        "session_num": bundle.session_num,
        "config": asdict(cfg),
    }
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(_jsonify(payload), f, ensure_ascii=False, indent=2)
    return fp



def run_session_fr_power_correlations(bundle: CommonSessionBundle,
                                      units: Dict[Any, Any],
                                      out_dir: str | Path,
                                      cfg: Optional[FRPowerCorrelationConfig] = None,
                                      dead_intervals_by_unit: Optional[Dict[Any, Any]] = None,
                                      unit_metadata: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Lance toute l'analyse spike–power pour une session.

    Parameters
    ----------
    bundle : CommonSessionBundle
        Produit par prepare_common_session ou load_common_session_bundle(..., load_hilbert=True).
    units : dict
        {unit_id: spike_times_s}. Les temps doivent être en référentiel micro.
    out_dir : str | Path
        Dossier racine de sortie. Le module créera out_dir/SESSION/.
    cfg : FRPowerCorrelationConfig
        Configuration.
    dead_intervals_by_unit : dict optionnel
        {unit_id: intervals n×2 en secondes}, pour exclure les artefacts micro.
    unit_metadata : pd.DataFrame optionnel
        Table avec colonne 'unit_id' et infos à propager : tetrode, lobe_tt, etc.

    Returns
    -------
    dict avec chemins des fichiers sauvegardés et DataFrames principaux.
    """
    cfg = cfg or FRPowerCorrelationConfig()
    if bundle.hilbert is None:
        raise ValueError("bundle.hilbert est None. Recharge avec load_common_session_bundle(..., load_hilbert=True).")

    session_out = ensure_dir(Path(out_dir).expanduser() / bundle.session_name)
    _write_config(session_out, cfg, bundle)

    bands = _infer_bands_to_test(bundle, cfg)
    if len(bands) == 0:
        raise RuntimeError("Aucune bande Hilbert exploitable.")

    bin_tables = {
        float(ms): make_relative_bin_table(
            pre_length=float(bundle.trials["pre_end_micro"].iloc[0] - bundle.trials["pre_start_micro"].iloc[0]),
            post_length=float(bundle.trials["post_end_micro"].iloc[0] - bundle.trials["post_start_micro"].iloc[0]),
            bin_width_s=float(ms) / 1000.0,
            include_partial_bins=False,
        )
        for ms in cfg.bin_width_ms_options
    }

    log(f"[INFO] {bundle.session_name}: calcul FR biné pour {len(units)} unités", cfg.verbose)
    fr_df = build_fr_bin_table_for_units(
        bundle=bundle,
        units=units,
        bin_tables=bin_tables,
        cfg=cfg,
        dead_intervals_by_unit=dead_intervals_by_unit,
        unit_metadata=unit_metadata,
    )

    saved_files: Dict[str, Any] = {}
    if cfg.save_fr_table:
        saved_files["fr_table"] = str(_to_csv(
            fr_df,
            session_out,
            f"{bundle.session_name}_fr_bins",
            cfg.compression,
        ))

    corr_tables: List[pd.DataFrame] = []
    trial_bin_files: List[str] = []
    lfp_files: List[str] = []

    for band in bands:
        log(f"[INFO] {bundle.session_name}: binning LFP band={band}", cfg.verbose)
        lfp_df = build_lfp_power_bin_table_for_band(
            bundle=bundle,
            band=band,
            bin_tables=bin_tables,
            cfg=cfg,
        )

        if cfg.save_lfp_table:
            fp_lfp = _to_csv(
                lfp_df,
                session_out,
                f"{bundle.session_name}_lfp_power_bins_{safe_name(band)}",
                cfg.compression,
            )
            lfp_files.append(str(fp_lfp))

        log(f"[INFO] {bundle.session_name}: fusion FR×LFP band={band}", cfg.verbose)
        trial_bin = build_fr_power_trial_bin_table(fr_df=fr_df, lfp_df=lfp_df, cfg=cfg)
        if trial_bin.empty:
            log(f"[WARN] {bundle.session_name} {band}: table FR×LFP vide", cfg.verbose)
            continue

        if cfg.save_trial_bin_tables:
            fp_trial = _to_csv(
                trial_bin,
                session_out,
                f"{bundle.session_name}_fr_power_trial_bins_{safe_name(band)}",
                cfg.compression,
            )
            trial_bin_files.append(str(fp_trial))

        log(f"[INFO] {bundle.session_name}: corrélations band={band}", cfg.verbose)
        corr = compute_correlations_from_trial_bin_table(trial_bin, cfg=cfg)
        if not corr.empty:
            corr_tables.append(corr)

        # Libère explicitement les grosses tables par bande.
        del lfp_df, trial_bin

    if len(corr_tables) == 0:
        corr_all = pd.DataFrame()
    else:
        corr_all = pd.concat(corr_tables, axis=0, ignore_index=True)

    fp_corr = _to_csv(
        corr_all,
        session_out,
        f"{bundle.session_name}_fr_power_correlations",
        cfg.compression,
    )
    saved_files["correlations"] = str(fp_corr)
    saved_files["trial_bin_tables"] = trial_bin_files
    saved_files["lfp_tables"] = lfp_files

    summary = {
        "session": bundle.session_name,
        "n_units_input": int(len(units)),
        "n_units_in_fr_table": int(fr_df["unit_id"].nunique()) if not fr_df.empty else 0,
        "bands_tested": bands,
        "bin_width_ms_options": list(cfg.bin_width_ms_options),
        "n_correlation_rows": int(len(corr_all)),
        "saved_files": saved_files,
    }
    with open(session_out / f"{bundle.session_name}_fr_power_corr_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(_jsonify(summary), f, ensure_ascii=False, indent=2)

    log(f"[OK] {bundle.session_name}: corrélations sauvegardées -> {fp_corr}", cfg.verbose)

    return {
        "session_out": session_out,
        "fr_table": fr_df,
        "correlations": corr_all,
        "saved_files": saved_files,
        "summary": summary,
    }


# =============================================================================
# UTILITAIRES POUR PRÉPARER units / unit_metadata
# =============================================================================


def make_units_dict_from_spikes_object(spikes: Any,
                                       unit_ids: Optional[Sequence[Any]] = None) -> Dict[Any, np.ndarray]:
    """
    Convertit un objet de spikes en dict {unit_id: spike_times_s}.

    Compatible avec :
    - dict déjà formé ;
    - TsGroup/pynapple-like indexable : spikes[unit_id].index.values ;
    - DataFrame avec colonnes unit_id/spike_time.
    """
    if isinstance(spikes, dict):
        ids = list(spikes.keys()) if unit_ids is None else list(unit_ids)
        return {uid: as_spike_times_array(spikes[uid]) for uid in ids if uid in spikes}

    if isinstance(spikes, pd.DataFrame):
        if not {"unit_id", "spike_time"}.issubset(spikes.columns):
            raise ValueError("DataFrame spikes doit contenir unit_id et spike_time")
        ids = spikes["unit_id"].unique().tolist() if unit_ids is None else list(unit_ids)
        return {
            uid: np.sort(spikes.loc[spikes["unit_id"] == uid, "spike_time"].to_numpy(float))
            for uid in ids
        }

    # Objet TsGroup-like.
    if unit_ids is None:
        if hasattr(spikes, "keys"):
            unit_ids = list(spikes.keys())
        elif hasattr(spikes, "index"):
            unit_ids = list(spikes.index)
        else:
            raise ValueError("Impossible d'inférer unit_ids. Passe unit_ids explicitement.")

    out = {}
    for uid in unit_ids:
        try:
            out[uid] = as_spike_times_array(spikes[uid])
        except Exception as exc:
            raise ValueError(f"Impossible d'extraire les spikes de l'unité {uid}: {exc}") from exc
    return out


__all__ = [
    "FRPowerCorrelationConfig",
    "make_relative_bin_table",
    "relative_bin_to_absolute_micro",
    "build_fr_bin_table_for_units",
    "build_lfp_power_bin_table_for_band",
    "build_fr_power_trial_bin_table",
    "compute_correlations_from_trial_bin_table",
    "run_session_fr_power_correlations",
    "make_units_dict_from_spikes_object",
]
