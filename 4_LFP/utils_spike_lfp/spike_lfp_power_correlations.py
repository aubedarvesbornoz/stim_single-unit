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
    4. on moyenne la puissance/enveloppe Hilbert du même band/channel/trial/bin ;
    5. on corrèle FR normalisé et power LFP across trials.

Sorties
-------
    - table longue optionnelle : unit × trial × channel × band × bin ;
    - table résumé : corrélations Pearson/Spearman par unit × channel × band × time bin.

Remarques méthodologiques
-------------------------
- Les spikes restent en référentiel micro ; le LFP/Hilbert reste en référentiel macro.
  Les deux sont associés par l'index de trial et le temps relatif pré/post.
- La largeur de bin 100/500 ms correspond ici à un lissage rectangulaire :
  FR = nombre de spikes dans le bin / durée du bin.
- Pour une analyse plus strictement "post-stim", laisser windows_to_correlate=('post',).
- Le LFP Hilbert étant déjà normalisé dans ton pipeline, la colonne par défaut
  utilisée pour la corrélation est lfp_power_bin_mean. Des colonnes LFP normalisées
  par le pré-trial sont quand même produites.

Auteure du projet : Aube Darves-Bornoz
Module généré pour les analyses spike–power SEIC.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr, spearmanr

from utils_spike_lfp.spike_lfp_common_preprocess_session import (
    CommonSessionBundle,
    as_spike_times_array,
    get_spikes_in_interval,
    parse_bipolar_shaft,
    channel_locality_for_trial,
    safe_name,
    get_nwb,
    load_common_session_bundle,
    build_dead_intervals_by_unit,
    build_unit_metadata_from_spikes,
    intervals_overlap_duration,
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
GroupingName = Literal["all", "group_label", "locality", "group_label_x_locality", "cog_subcategory", "cog_subcategory_x_locality"]
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
    lfp_value_col: LFPValueColumn = "lfp_power_pre_logratio"
    epsilon_lfp: float = 1e-12

    # Méthodes de corrélation et seuils minimaux.
    correlation_methods: Tuple[CorrelationMethod, ...] = ("spearman",) # teste une relation monotone, pas forcément linéaire, contrairement au Pearson
    min_trials_for_corr: int = 6
    min_finite_pairs_for_corr: int = 6

    # Groupements statistiques : produit par défaut les corrélations globales,
    # par condition cognitive, par localité, et condition×localité.
    correlation_groupings: Tuple[GroupingName, ...] = (
        "all",
        "group_label",
        "locality",
        "group_label_x_locality",
        "cog_subcategory",
        "cog_subcategory_x_locality",
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
    # Alternative plus interprétable : sigma en millisecondes. Si renseigné,
    # sigma_bins = gaussian_sigma_ms / bin_width_ms pour chaque largeur de bin.
    gaussian_sigma_ms: Optional[float] = None

    # Sauvegardes.
    save_trial_bin_tables: bool = True
    save_fr_table: bool = True
    save_lfp_table: bool = False
    compression: Optional[str] = None  # ex. "gzip" pour csv.gz si souhaité

    # Export d'un sous-ensemble significatif plus léger.
    save_significant_correlations: bool = True
    significance_p_col: str = "p_value"      # "p_value" ou "q_value_fdr_bh"
    significance_alpha: float = 0.05

    # Figures de répartition des rho/r significatifs.
    make_significant_histograms: bool = True
    histogram_method: str = "spearman"       # souvent plus lisible que spearman+pearson mélangés
    histogram_significance_col: str = "p_value"
    histogram_alpha: float = 0.05
    histogram_bins: int = 30

    # Deadfiles micro : si un bin chevauche une dead period de la tétrade du neurone,
    # on l'invalide pour éviter de transformer un artefact supprimé en FR=0.
    exclude_bins_overlapping_dead: bool = True
    min_valid_bin_fraction: float = 1.0
    use_deadfiles: bool = True

    # Reuse/overwrite des sorties session.
    overwrite: bool = True
    reuse_existing_if_available: bool = False

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


def existing_session_corr_outputs(session_out: Path,
                                  session_name: str,
                                  cfg: FRPowerCorrelationConfig) -> Dict[str, Path]:
    """Chemins attendus pour les sorties principales d'une session."""
    return {
        "correlations": _csv_path(session_out, f"{session_name}_fr_power_correlations", cfg.compression),
        "significant": _csv_path(session_out, f"{session_name}_fr_power_correlation_signif", cfg.compression),
        "config": session_out / f"{session_name}_fr_power_corr_config.json",
        "summary": session_out / f"{session_name}_fr_power_corr_run_summary.json",
    }


def load_existing_session_fr_power_result(session_out: Path,
                                          session_name: str,
                                          cfg: FRPowerCorrelationConfig) -> Dict[str, Any]:
    """Recharge une table session déjà calculée, sans recalculer spikes/LFP."""
    fps = existing_session_corr_outputs(session_out, session_name, cfg)
    if not fps["correlations"].exists():
        raise FileNotFoundError(fps["correlations"])
    corr = pd.read_csv(fps["correlations"], compression="infer")
    if fps["significant"].exists():
        sig = pd.read_csv(fps["significant"], compression="infer")
    elif cfg.significance_p_col in corr.columns:
        sig = get_significant_correlations(corr, p_col=cfg.significance_p_col, alpha=cfg.significance_alpha)
    else:
        sig = pd.DataFrame()
    summary = {
        "session": session_name,
        "reused_existing": True,
        "n_correlation_rows": int(len(corr)),
        "n_significant_rows": int(len(sig)),
        "saved_files": {k: str(v) if v.exists() else None for k, v in fps.items()},
    }
    return {
        "session_out": session_out,
        "fr_table": pd.DataFrame(),
        "correlations": corr,
        "saved_files": summary["saved_files"],
        "summary": summary,
    }


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
                                sigma_bins: float,
                                sigma_ms: Optional[float] = None) -> pd.DataFrame:
    """Lisse fr_hz séparément dans la fenêtre pré et post, par unit×trial×bin_width.

    Notes
    -----
    - ``boxcar_bin`` : pas de lissage additionnel ; le bin lui-même est le lissage
      rectangulaire. C'est le choix recommandé pour commencer.
    - ``gaussian_bins`` : lissage sur bins adjacents. Si ``sigma_ms`` est fourni,
      il est converti pour chaque largeur de bin : ``sigma_bins = sigma_ms / bin_width_ms``.
      Sinon, la valeur historique ``gaussian_sigma_bins`` est utilisée.
    """
    out = fr_df.copy()
    out["fr_hz_smooth"] = out["fr_hz"].astype(float)
    out["fr_smoothing_method"] = method
    out["fr_smoothing_sigma_bins"] = np.nan
    out["fr_smoothing_sigma_ms"] = sigma_ms if sigma_ms is not None else np.nan

    if method in {"none", "boxcar_bin"}:
        return out
    if method != "gaussian_bins":
        raise ValueError(f"smoothing_method inconnu: {method}")

    group_cols = ["unit_id", "trial_idx", "bin_width_ms", "window"]
    for _, idx in out.groupby(group_cols, sort=False).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        idx_arr = idx_arr[np.argsort(out.loc[idx_arr, "bin_index_in_window"].to_numpy())]
        vals = out.loc[idx_arr, "fr_hz"].to_numpy(float)
        bin_width_ms = float(out.loc[idx_arr[0], "bin_width_ms"]) if len(idx_arr) > 0 else np.nan
        sigma_use = float(sigma_ms) / bin_width_ms if sigma_ms is not None and bin_width_ms > 0 else float(sigma_bins)
        out.loc[idx_arr, "fr_smoothing_sigma_bins"] = sigma_use
        out.loc[idx_arr, "fr_hz_smooth"] = gaussian_smooth_1d_nanaware(vals, sigma_bins=sigma_use)
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
                    dead_overlap_s = intervals_overlap_duration(t0, t1, dead) if dead is not None else 0.0
                    dur = max(float(t1 - t0), np.finfo(float).eps)
                    valid_fraction = max(0.0, (dur - dead_overlap_s) / dur)
                    bin_valid = True
                    if cfg.exclude_bins_overlapping_dead and dead_overlap_s > 0:
                        bin_valid = False
                    if valid_fraction < float(cfg.min_valid_bin_fraction):
                        bin_valid = False

                    if bin_valid:
                        spk_bin = get_spikes_in_interval(spk, t0, t1, dead_intervals=dead)
                        n_spikes = int(len(spk_bin))
                        fr_hz = n_spikes / dur
                    else:
                        n_spikes = np.nan
                        fr_hz = np.nan

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
                        "dead_overlap_s": float(dead_overlap_s),
                        "valid_bin_fraction": float(valid_fraction),
                        "bin_valid_after_deadfile": bool(bin_valid),
                        "n_spikes": n_spikes,
                        "fr_hz": fr_hz,
                    })

    fr = pd.DataFrame(rows)
    if fr.empty:
        raise RuntimeError("Aucun firing rate calculé. Vérifie units, trials et binning.")

    fr = smooth_rates_within_windows(
        fr,
        method=cfg.smoothing_method,
        sigma_bins=cfg.gaussian_sigma_bins,
        sigma_ms=cfg.gaussian_sigma_ms,
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



def _parse_cog_labels_cell(value: Any) -> List[str]:
    """Parse une cellule cog_labels robuste, compatible liste Python sérialisée ou chaîne."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip() and str(x).strip().lower() != "nan"]
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    s = str(value).strip()
    if not s or s.lower() == "nan" or s == "[]":
        return []
    # Essai de parsing Python littéral.
    try:
        import ast
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple, set)):
            return [str(x).strip() for x in obj if str(x).strip()]
        return [str(obj).strip()] if str(obj).strip() else []
    except Exception:
        pass
    parts = re.split(r"[;,/]", s)
    return [p.strip().strip("'\"") for p in parts if p.strip()]


def _expand_cog_subcategory_rows(df: pd.DataFrame, with_locality: bool) -> pd.DataFrame:
    """Duplique les lignes cog+ pour chaque sous-label cognitif."""
    if "cog_labels" not in df.columns or "group_label" not in df.columns:
        return df.iloc[0:0].copy()
    rows = []
    for _, row in df.iterrows():
        if str(row.get("group_label", "")).strip() != "cog+":
            continue
        labels = _parse_cog_labels_cell(row.get("cog_labels"))
        for lab in labels:
            if not lab:
                continue
            r = row.copy()
            r["corr_grouping"] = "cog_subcategory_x_locality" if with_locality else "cog_subcategory"
            if with_locality:
                r["corr_condition"] = f"cog::{lab}::{row.get('locality', 'unknown')}"
            else:
                r["corr_condition"] = f"cog::{lab}"
            r["cog_subcategory"] = lab
            rows.append(r)
    if len(rows) == 0:
        out = df.iloc[0:0].copy()
        out["corr_grouping"] = []
        out["corr_condition"] = []
        out["cog_subcategory"] = []
        return out
    return pd.DataFrame(rows).reset_index(drop=True)


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
    elif grouping == "cog_subcategory":
        out = _expand_cog_subcategory_rows(out, with_locality=False)
    elif grouping == "cog_subcategory_x_locality":
        out = _expand_cog_subcategory_rows(out, with_locality=True)
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
        Dossier racine de sortie. Le module créera out_dir/SESSION/. {/per_session/ est en fait deja renseigné dans out_dir}
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
    
    session_out = ensure_dir(Path(out_dir).expanduser() / "per_session" / bundle.session_name)
    existing = existing_session_corr_outputs(session_out, bundle.session_name, cfg)
    can_reuse = (
        cfg.reuse_existing_if_available
        and not cfg.overwrite
        and existing["correlations"].exists()
    )
    if can_reuse:
        log(f"[REUSE] {bundle.session_name}: corrélations existantes -> {existing['correlations']}", cfg.verbose)
        return load_existing_session_fr_power_result(session_out, bundle.session_name, cfg)

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

    if cfg.save_significant_correlations and not corr_all.empty and cfg.significance_p_col in corr_all.columns:
        sig = get_significant_correlations(corr_all, p_col=cfg.significance_p_col, alpha=cfg.significance_alpha)
        fp_sig = _to_csv(
            sig,
            session_out,
            f"{bundle.session_name}_fr_power_correlation_signif",
            cfg.compression,
        )
        saved_files["significant_correlations"] = str(fp_sig)
    else:
        sig = pd.DataFrame()

    if cfg.make_significant_histograms and not corr_all.empty:
        fig_dir = ensure_dir(session_out / "figures_significant_histograms")
        fig_files = plot_significant_histograms_suite(
            corr_df=corr_all,
            out_dir=fig_dir,
            prefix=bundle.session_name,
            cfg=cfg,
        )
        saved_files["significant_histograms"] = [str(x) for x in fig_files]

    saved_files["trial_bin_tables"] = trial_bin_files
    saved_files["lfp_tables"] = lfp_files

    summary = {
        "session": bundle.session_name,
        "n_units_input": int(len(units)),
        "n_units_in_fr_table": int(fr_df["unit_id"].nunique()) if not fr_df.empty else 0,
        "bands_tested": bands,
        "bin_width_ms_options": list(cfg.bin_width_ms_options),
        "n_correlation_rows": int(len(corr_all)),
        "n_significant_rows": int(len(sig)) if 'sig' in locals() else 0,
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



def prepare_unit_metadata_and_dead_intervals(patient: str,
                                             session: str | int,
                                             root_micro: str | Path,
                                             spikes: Any,
                                             use_deadfiles: bool = True) -> Tuple[pd.DataFrame, Optional[Dict[Any, Any]]]:
    """
    Helper pour un run session unique.

    Retourne :
        unit_metadata, dead_intervals_by_unit

    dead_intervals_by_unit vaut None si use_deadfiles=False ou si les fichiers/mapping
    ne sont pas disponibles.
    """
    unit_metadata = build_unit_metadata_from_spikes(patient, str(session), root_micro, spikes)
    dead_intervals_by_unit = None
    if use_deadfiles:
        dead_intervals_by_unit = build_dead_intervals_by_unit(patient, str(session), root_micro, spikes)
    return unit_metadata, dead_intervals_by_unit


# =============================================================================
# SIGNIFICATIVITÉ ET FIGURES DE DISTRIBUTION DES CORRÉLATIONS
# =============================================================================


def get_significant_correlations(corr_df: pd.DataFrame,
                                 p_col: str = "p_value",
                                 alpha: float = 0.05,
                                 method: Optional[str] = None) -> pd.DataFrame:
    """Retourne les lignes significatives selon p_col < alpha, optionnellement pour une méthode."""
    if corr_df is None or corr_df.empty:
        return pd.DataFrame()
    if p_col not in corr_df.columns:
        raise ValueError(f"Colonne de significativité absente: {p_col}")
    df = corr_df.copy()
    if method is not None and "method" in df.columns:
        df = df.loc[df["method"].astype(str) == str(method)].copy()
    df[p_col] = pd.to_numeric(df[p_col], errors="coerce")
    df = df.loc[np.isfinite(df[p_col]) & (df[p_col] < alpha)].copy()
    return df.reset_index(drop=True)


def _condition_locality_from_corr_condition(value: Any) -> Tuple[Optional[str], Optional[str]]:
    """Parse corr_condition de type 'cog+::local' ou 'cog::aphasie::distant'."""
    s = str(value)
    parts = s.split("::")
    if len(parts) >= 2 and parts[-1] in {"local", "distant", "unknown"}:
        return "::".join(parts[:-1]), parts[-1]
    return s, None


def _prepare_hist_df(corr_df: pd.DataFrame,
                     p_col: str,
                     alpha: float,
                     method: Optional[str],
                     corr_grouping: str,
                     corr_condition_prefix: Optional[str] = None,
                     corr_condition_exact: Optional[str] = None) -> pd.DataFrame:
    df = get_significant_correlations(corr_df, p_col=p_col, alpha=alpha, method=method)
    if df.empty:
        return df
    df = df.loc[df["corr_grouping"].astype(str) == corr_grouping].copy()
    if corr_condition_exact is not None:
        df = df.loc[df["corr_condition"].astype(str) == corr_condition_exact].copy()
    if corr_condition_prefix is not None:
        df = df.loc[df["corr_condition"].astype(str).str.startswith(corr_condition_prefix)].copy()
    if df.empty:
        return df
    parsed = df["corr_condition"].apply(_condition_locality_from_corr_condition)
    df["hist_condition"] = parsed.apply(lambda x: x[0])
    df["hist_locality"] = parsed.apply(lambda x: x[1])
    if "hist_locality" not in df.columns or df["hist_locality"].isna().all():
        if "corr_condition" in df.columns and corr_grouping == "locality":
            df["hist_locality"] = df["corr_condition"].astype(str)
        elif "locality" in df.columns:
            df["hist_locality"] = df["locality"].astype(str)
    return df.reset_index(drop=True)


def plot_rho_hist_grid(df: pd.DataFrame,
                       out_file: str | Path,
                       title: str,
                       bands: Sequence[str],
                       localities: Sequence[str] = ("local", "distant"),
                       bins: int = 30,
                       rho_col: str = "rho_or_r") -> Optional[Path]:
    """
    Figure de distribution des rho/r significatifs.

    Colonnes = bandes de fréquence ; lignes = localités.
    """
    if df is None or df.empty:
        return None
    if rho_col not in df.columns:
        raise ValueError(f"Colonne absente: {rho_col}")

    df = df.copy()
    df[rho_col] = pd.to_numeric(df[rho_col], errors="coerce")
    df = df.loc[np.isfinite(df[rho_col])].copy()
    if df.empty:
        return None

    bands = [b for b in bands if b in set(df["band"].astype(str))]
    if len(bands) == 0:
        return None
    localities = list(localities)

    fig, axes = plt.subplots(
        nrows=len(localities),
        ncols=len(bands),
        figsize=(3.4 * len(bands), 2.8 * len(localities)),
        squeeze=False,
        sharex=True,
        sharey=False,
    )

    hist_range = (-1.0, 1.0)
    for r, loc in enumerate(localities):
        for c, band in enumerate(bands):
            ax = axes[r, c]
            sub = df.loc[(df["band"].astype(str) == str(band)) & (df["hist_locality"].astype(str) == str(loc))]
            vals = sub[rho_col].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                ax.hist(vals, bins=bins, range=hist_range)
                ax.axvline(0, linestyle=":", linewidth=1.0)
                ax.set_title(f"{band}\nN={len(vals)}", fontsize=9)
            else:
                ax.text(0.5, 0.5, "N=0", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{band}\nN=0", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{loc}\ncount")
            if r == len(localities) - 1:
                ax.set_xlabel("rho / r")
            ax.set_xlim(hist_range)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_file

def plot_effect_overlay_rho_hist_lines(
    corr_df: pd.DataFrame,
    out_file: str | Path,
    cfg: FRPowerCorrelationConfig,
    bands: Optional[Sequence[str]] = None,
    localities: Sequence[str] = ("local", "distant"),
    effects: Sequence[str] = ("cog+", "controle", "negatif"),
    effect_colors: Optional[Dict[str, str]] = None,
    bins: Optional[int] = None,
    smooth_sigma_bins: float = 1.2,
    density: bool = True,
    rho_col: str = "rho_or_r",
) -> Optional[Path]:
    """
    Trace une figure de distributions rho/r significatifs avec courbes superposées.

    Colonnes = bandes.
    Lignes = localité.
    Courbes = effets principaux : cog+, controle, negatif.

    Utilise les lignes corr_grouping == 'group_label_x_locality',
    donc corr_condition doit être du type :
        cog+::local
        controle::distant
        negatif::local
    """
    if corr_df is None or corr_df.empty:
        return None

    p_col = cfg.histogram_significance_col
    alpha = cfg.histogram_alpha
    method = cfg.histogram_method
    bins = bins or cfg.histogram_bins

    if effect_colors is None:
        effect_colors = {
            "cog+": "red",
            "controle": "green",
            "negatif": "blue",
        }

    if p_col not in corr_df.columns:
        raise ValueError(f"Colonne de significativité absente: {p_col}")

    df = get_significant_correlations(
        corr_df,
        p_col=p_col,
        alpha=alpha,
        method=method,
    )

    if df.empty:
        return None

    df = df.loc[df["corr_grouping"].astype(str) == "group_label_x_locality"].copy()
    if df.empty:
        return None

    parsed = df["corr_condition"].apply(_condition_locality_from_corr_condition)
    df["hist_effect"] = parsed.apply(lambda x: x[0])
    df["hist_locality"] = parsed.apply(lambda x: x[1])

    df = df.loc[
        df["hist_effect"].isin(effects)
        & df["hist_locality"].isin(localities)
    ].copy()

    if df.empty:
        return None

    df[rho_col] = pd.to_numeric(df[rho_col], errors="coerce")
    df = df.loc[np.isfinite(df[rho_col])].copy()
    if df.empty:
        return None

    if bands is None:
        if cfg.bands_to_test is not None:
            bands = list(cfg.bands_to_test)
        else:
            bands = sorted(df["band"].dropna().astype(str).unique())

    bands = [b for b in bands if b in set(df["band"].astype(str))]
    if len(bands) == 0:
        return None

    localities = list(localities)
    effects = [e for e in effects if e in set(df["hist_effect"].astype(str))]
    if len(effects) == 0:
        return None

    hist_range = (-1.0, 1.0)
    edges = np.linspace(hist_range[0], hist_range[1], bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0

    fig, axes = plt.subplots(
        nrows=len(localities),
        ncols=len(bands),
        figsize=(3.8 * len(bands), 2.9 * len(localities)),
        squeeze=False,
        sharex=True,
        sharey=False,
    )

    for r, loc in enumerate(localities):
        for c, band in enumerate(bands):
            ax = axes[r, c]

            plotted_any = False
            for effect in effects:
                sub = df.loc[
                    (df["band"].astype(str) == str(band))
                    & (df["hist_locality"].astype(str) == str(loc))
                    & (df["hist_effect"].astype(str) == str(effect))
                ]

                vals = sub[rho_col].to_numpy(float)
                vals = vals[np.isfinite(vals)]

                if len(vals) == 0:
                    continue

                y, _ = np.histogram(vals, bins=edges, density=density)

                if smooth_sigma_bins is not None and smooth_sigma_bins > 0:
                    y = gaussian_filter1d(y.astype(float), sigma=float(smooth_sigma_bins), mode="nearest")

                label = f"{effect} (N={len(vals)})"
                ax.plot(
                    centers,
                    y,
                    linewidth=2.0,
                    color=effect_colors.get(effect, None),
                    label=label,
                )
                plotted_any = True

            ax.axvline(0, linestyle=":", linewidth=1.0, color="black")
            ax.set_xlim(hist_range)

            if plotted_any:
                ax.set_title(str(band), fontsize=9)
                ax.legend(fontsize=7, frameon=False)
            else:
                ax.text(0.5, 0.5, "N=0", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(str(band), fontsize=9)

            if c == 0:
                ax.set_ylabel(f"{loc}\n{'density' if density else 'count'}")
            if r == len(localities) - 1:
                ax.set_xlabel("rho / r")

    fig.suptitle(
        f"{Path(out_file).stem} | signif {p_col}<{alpha} | {method}",
        y=1.02,
    )
    fig.tight_layout()

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_file


def _available_general_effects(corr_df: pd.DataFrame) -> List[str]:
    """Effets principaux à tracer depuis group_label_x_locality."""
    if corr_df.empty or "corr_condition" not in corr_df.columns:
        return []
    sub = corr_df.loc[corr_df["corr_grouping"].astype(str) == "group_label_x_locality"].copy()
    labels = []
    for cond in sub["corr_condition"].dropna().astype(str).unique():
        effect, loc = _condition_locality_from_corr_condition(cond)
        if loc in {"local", "distant"} and effect not in labels:
            labels.append(effect)
    preferred = [x for x in ["cog+", "negatif", "controle"] if x in labels]
    rest = sorted([x for x in labels if x not in preferred])
    return preferred + rest


def _available_cog_subtypes(corr_df: pd.DataFrame) -> List[str]:
    """Sous-types cog::label disponibles depuis cog_subcategory_x_locality."""
    if corr_df.empty or "corr_condition" not in corr_df.columns:
        return []
    sub = corr_df.loc[corr_df["corr_grouping"].astype(str) == "cog_subcategory_x_locality"].copy()
    labels = []f
    for cond in sub["corr_condition"].dropna().astype(str).unique():
        effect, loc = _condition_locality_from_corr_condition(cond)
        if loc in {"local", "distant"} and str(effect).startswith("cog::") and effect not in labels:
            labels.append(effect)
    return sorted(labels)


def plot_significant_histograms_suite(corr_df: pd.DataFrame,
                                      out_dir: str | Path,
                                      prefix: str,
                                      cfg: FRPowerCorrelationConfig,
                                      bands: Optional[Sequence[str]] = None) -> List[Path]:
    """
    Génère les figures demandées sur les lignes significatives seulement :
      1. toutes stims confondues : bande × localité ;
      2. par effet principal : cog+, negatif, controle si disponible ;
      3. par sous-type cog+ : cog::aphasie, cog::déjà-vu, etc.
    """
    out_dir = ensure_dir(out_dir)
    if bands is None:
        if cfg.bands_to_test is not None:
            bands = list(cfg.bands_to_test)
        elif corr_df is not None and not corr_df.empty and "band" in corr_df.columns:
            bands = sorted(corr_df["band"].dropna().astype(str).unique())
        else:
            bands = ["theta", "alpha", "beta", "low_gamma", "high_gamma"]

    p_col = cfg.histogram_significance_col
    alpha = cfg.histogram_alpha
    method = cfg.histogram_method
    bins = cfg.histogram_bins
    files: List[Path] = []

    # 1) Toutes stimulations confondues, séparé par localité.
    df_all = _prepare_hist_df(
        corr_df,
        p_col=p_col,
        alpha=alpha,
        method=method,
        corr_grouping="locality",
    )
    fp = plot_rho_hist_grid(
        df_all,
        out_file=out_dir / f"{safe_name(prefix)}_rho_hist_SIGNIF_all_stims_by_band_x_locality.png",
        title=f"{prefix} | signif {p_col}<{alpha} | {method} | all stims",
        bands=bands,
        localities=("local", "distant"),
        bins=bins,
    )
    if fp is not None:
        files.append(fp)

    # 2) Effets principaux : cog+, negatif, contrôle si présent.
    for effect in _available_general_effects(corr_df):
        if effect == 'unknown' : # parce qu'on s'en fiche, all_stims suffit deja
            continue
        df_eff = _prepare_hist_df(
            corr_df,
            p_col=p_col,
            alpha=alpha,
            method=method,
            corr_grouping="group_label_x_locality",
            corr_condition_prefix=f"{effect}::",
        )
        fp = plot_rho_hist_grid(
            df_eff,
            out_file=out_dir / f"{safe_name(prefix)}_rho_hist_SIGNIF_effect_{safe_name(effect)}_by_band_x_locality.png",
            title=f"{prefix} | signif {p_col}<{alpha} | {method} | effect={effect}",
            bands=bands,
            localities=("local", "distant"),
            bins=bins,
        )
        if fp is not None:
            files.append(fp)

    # 3) Sous-types cog+.
    for subtype in _available_cog_subtypes(corr_df):
        df_sub = _prepare_hist_df(
            corr_df,
            p_col=p_col,
            alpha=alpha,
            method=method,
            corr_grouping="cog_subcategory_x_locality",
            corr_condition_prefix=f"{subtype}::",
        )
        subtype_label = subtype.replace("cog::", "")
        fp = plot_rho_hist_grid(
            df_sub,
            out_file=out_dir / f"{safe_name(prefix)}_rho_hist_SIGNIF_cog_subtype_{safe_name(subtype_label)}_by_band_x_locality.png",
            title=f"{prefix} | signif {p_col}<{alpha} | {method} | {subtype}",
            bands=bands,
            localities=("local", "distant"),
            bins=bins,
        )
        if fp is not None:
            files.append(fp)

    # 4) Effets principaux superposés : cog+ / controle / negatif.
    fp = plot_effect_overlay_rho_hist_lines(
        corr_df=corr_df,
        out_file=out_dir / f"{safe_name(prefix)}_rho_hist_SIGNIF_effect_overlay_by_band_x_locality.png",
        cfg=cfg,
        bands=bands,
        localities=("local", "distant"),
        effects=("cog+", "controle", "negatif"),
        effect_colors={
            "cog+": "red",
            "controle": "green",
            "negatif": "blue",
        },
        smooth_sigma_bins=1.2,
        density=True,
    )
    if fp is not None:
        files.append(fp)

    return files


# =============================================================================
# ORCHESTRATEUR MULTI-SESSIONS / POOLING DES CORRÉLATIONS
# =============================================================================


@dataclass
class PooledFRPowerCorrelationConfig:
    """Configuration de l'orchestrateur multi-sessions."""

    common_root: str
    root_micro: str
    output_root: str

    hilbert_bands: Tuple[str, ...] = ("theta", "alpha", "beta", "low_gamma", "high_gamma")
    session_cfg: FRPowerCorrelationConfig = field(default_factory=FRPowerCorrelationConfig)

    # Vérifie qu'un .nwb existe avant d'essayer la session.
    require_existing_nwb: bool = True

    # overwrite_session_outputs=False : réutilise les corrélations per-session déjà présentes.
    overwrite_session_outputs: bool = True
    # Ancien alias gardé pour compatibilité.
    skip_existing_session_outputs: bool = False

    # FDR recalculé après concaténation des lignes de corrélation de toutes les sessions.
    recompute_pooled_fdr: bool = True

    # Plots pooled sur les lignes significatives seulement.
    make_pooled_histograms: bool = True

    verbose: bool = True


def parse_session_name_for_patient_session(session_name: str) -> Tuple[str, str]:
    """Parse 'P119_FM71_stim4' -> ('P119_FM71', '4')."""
    m = re.match(r"^(?P<patient>.+)_stim(?P<session>\d+)$", str(session_name))
    if not m:
        raise ValueError(f"Nom de session non reconnu: {session_name}")
    return m.group("patient"), m.group("session")


def find_common_session_dirs(common_root: str | Path,
                             hilbert_root: Optional[str | Path] = None) -> List[Path]:
    """Liste les dossiers SESSION contenant SESSION_common_trials.csv, optionnellement présents dans hilbert_root."""
    root = Path(common_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    out = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        trials = d / f"{d.name}_common_trials.csv"
        meta = d / f"{d.name}_common_metadata.json"
        if not trials.exists() or not meta.exists():
            continue
        if hilbert_root is not None:
            hdir = Path(hilbert_root).expanduser().resolve() / d.name
            if not hdir.exists():
                continue
        out.append(d)
    return out


def find_nwb_file(root_micro: str | Path,
                  patient: str,
                  session: str | int) -> Optional[Path]:
    """Recherche PATIENT_stimN.nwb dans les structures micro habituelles."""
    root = Path(root_micro).expanduser()
    session = str(session)
    session_name = f"{patient}_stim{session}"
    candidates = [
        root / "Spike-sorting" / "Data_folders" / patient / session_name / f"{session_name}.nwb",
        root / "Data_folders" / patient / session_name / f"{session_name}.nwb",
        root / patient / session_name / f"{session_name}.nwb",
        root / session_name / f"{session_name}.nwb",
    ]
    for fp in candidates:
        if fp.exists():
            return fp.resolve()
    return None


def _root_with_trailing_sep(root: str | Path) -> str:
    """get_nwb concatène souvent root + 'Spike-sorting/...'; on sécurise le '/' final."""
    return str(root).rstrip("/") + "/"


def _load_common_bundle_for_session(session_dir: Path,
                                    hilbert_bands: Sequence[str]) -> CommonSessionBundle:
    """Recharge un common bundle avec les exports Hilbert demandés."""
    return load_common_session_bundle(
        session_dir,
        load_hilbert=True,
        hilbert_bands=hilbert_bands,
    )


def pool_saved_session_correlations(session_output_dirs: Sequence[str | Path],
                                    out_dir: str | Path,
                                    cfg: Optional[FRPowerCorrelationConfig] = None,
                                    recompute_fdr: bool = True,
                                    prefix: str = "ALL_SESSIONS") -> Dict[str, Any]:
    """Concatène les tables de corrélations sauvegardées par session et génère un pooled summary."""
    cfg = cfg or FRPowerCorrelationConfig()
    out_dir = ensure_dir(out_dir)
    frames = []
    used_files = []
    missing = []

    for d in session_output_dirs:
        d = Path(d).expanduser().resolve()
        session = d.name
        fp = _csv_path(d, f"{session}_fr_power_correlations", cfg.compression)
        if not fp.exists():
            # fallback sans compression si cfg différent de celui utilisé au run session.
            fp_alt = d / f"{session}_fr_power_correlations.csv"
            fp = fp_alt if fp_alt.exists() else fp
        if not fp.exists():
            missing.append(str(fp))
            continue
        df = pd.read_csv(fp, compression="infer")
        if not df.empty:
            frames.append(df)
            used_files.append(str(fp))

    pooled = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()

    if recompute_fdr and not pooled.empty and "p_value" in pooled.columns:
        pooled["q_value_fdr_bh_pooled"] = np.nan
        group_cols = [c for c in ["method", "corr_grouping"] if c in pooled.columns]
        if group_cols:
            for _, idx in pooled.groupby(group_cols, dropna=False).groups.items():
                idx_arr = np.asarray(list(idx), dtype=int)
                pooled.loc[idx_arr, "q_value_fdr_bh_pooled"] = fdr_bh(pooled.loc[idx_arr, "p_value"].to_numpy(float))
        else:
            pooled["q_value_fdr_bh_pooled"] = fdr_bh(pooled["p_value"].to_numpy(float))

    fp_pooled = _to_csv(pooled, out_dir, f"{prefix}_fr_power_correlations_pooled", cfg.compression)

    sig = get_significant_correlations(
        pooled,
        p_col=cfg.significance_p_col,
        alpha=cfg.significance_alpha,
    ) if not pooled.empty and cfg.significance_p_col in pooled.columns else pd.DataFrame()
    fp_sig = _to_csv(sig, out_dir, f"{prefix}_fr_power_correlations_pooled_SIGNIF", cfg.compression)

    fig_files = []
    if cfg.make_significant_histograms and not pooled.empty:
        fig_dir = ensure_dir(out_dir / "figures_significant_histograms")
        fig_files = plot_significant_histograms_suite(
            corr_df=pooled,
            out_dir=fig_dir,
            prefix=prefix,
            cfg=cfg,
        )

    summary = {
        "n_session_files_used": len(used_files),
        "used_files": used_files,
        "missing_files": missing,
        "n_rows_pooled": int(len(pooled)),
        "n_rows_significant": int(len(sig)),
        "pooled_file": str(fp_pooled),
        "pooled_significant_file": str(fp_sig),
        "figure_files": [str(x) for x in fig_files],
    }
    with open(out_dir / f"{prefix}_fr_power_pool_summary.json", "w", encoding="utf-8") as f:
        json.dump(_jsonify(summary), f, ensure_ascii=False, indent=2)

    return {"pooled": pooled, "significant": sig, "summary": summary, "out_dir": out_dir}


def run_all_available_sessions_fr_power_correlations(pool_cfg: PooledFRPowerCorrelationConfig,
                                                     unit_metadata_func: Optional[Any] = None,
                                                     dead_intervals_func: Optional[Any] = None) -> Dict[str, Any]:
    """
    Lance spike–power sur toutes les sessions common disponibles, si le NWB existe.

    Parameters
    ----------
    pool_cfg : PooledFRPowerCorrelationConfig
        Contient common_root, root_micro, output_root et session_cfg.
    unit_metadata_func : callable optionnel
        unit_metadata_func(patient, session, spikes) -> DataFrame avec colonne unit_id.
    dead_intervals_func : callable optionnel
        dead_intervals_func(patient, session, spikes) -> dict {unit_id: intervals n×2}.
    """
    cfg = pool_cfg.session_cfg
    output_root = ensure_dir(pool_cfg.output_root)
    session_out_root = ensure_dir(output_root / "per_session")
    pooled_out = ensure_dir(output_root / "pooled_across_sessions")

    # Si possible, on filtre sur le hilbert_root lu depuis metadata, mais common_root suffit.
    common_dirs = find_common_session_dirs(pool_cfg.common_root)
    if len(common_dirs) == 0:
        raise RuntimeError(f"Aucun common bundle trouvé dans {pool_cfg.common_root}")

    session_output_dirs: List[Path] = []
    rows_summary: List[Dict[str, Any]] = []
    errors: List[Tuple[str, str]] = []
    skipped: List[Tuple[str, str]] = []

    for session_dir in common_dirs:
        session_name = session_dir.name
        try:
            patient, session = parse_session_name_for_patient_session(session_name)
        except Exception as exc:
            skipped.append((session_name, f"parse_session_name_failed: {exc}"))
            continue

        out_session_dir = session_out_root / session_name
        corr_fp = _csv_path(out_session_dir, f"{session_name}_fr_power_correlations", cfg.compression)
        reuse_existing = (pool_cfg.skip_existing_session_outputs or not pool_cfg.overwrite_session_outputs) and corr_fp.exists()
        if reuse_existing:
            session_output_dirs.append(out_session_dir)
            rows_summary.append({
                "session": session_name,
                "patient": patient,
                "session_num": session,
                "status": "reused_existing",
                "nwb_file": None,
                "out_dir": str(out_session_dir),
            })
            continue

        nwb_fp = find_nwb_file(pool_cfg.root_micro, patient, session)
        if pool_cfg.require_existing_nwb and nwb_fp is None:
            skipped.append((session_name, "NWB absent"))
            rows_summary.append({"session": session_name, "patient": patient, "session_num": session, "status": "skipped_nwb_absent"})
            continue

        try:
            log(f"\n=== FR-power multi-session | {session_name} ===", pool_cfg.verbose)
            bundle = _load_common_bundle_for_session(session_dir, pool_cfg.hilbert_bands)
            spikes = get_nwb(patient, str(session), _root_with_trailing_sep(pool_cfg.root_micro))
            units = make_units_dict_from_spikes_object(spikes)

            if unit_metadata_func is not None:
                unit_metadata = unit_metadata_func(patient, str(session), spikes)
            elif cfg.use_deadfiles:
                unit_metadata = build_unit_metadata_from_spikes(patient, str(session), pool_cfg.root_micro, spikes)
            else:
                unit_metadata = None

            if dead_intervals_func is not None:
                dead_intervals_by_unit = dead_intervals_func(patient, str(session), spikes)
            elif cfg.use_deadfiles:
                dead_intervals_by_unit = build_dead_intervals_by_unit(patient, str(session), pool_cfg.root_micro, spikes)
            else:
                dead_intervals_by_unit = None

            cfg_session = replace(
                cfg,
                overwrite=pool_cfg.overwrite_session_outputs,
                reuse_existing_if_available=(pool_cfg.skip_existing_session_outputs or not pool_cfg.overwrite_session_outputs),
            )
            out = run_session_fr_power_correlations(
                bundle=bundle,
                units=units,
                out_dir=session_out_root,
                cfg=cfg_session,
                dead_intervals_by_unit=dead_intervals_by_unit,
                unit_metadata=unit_metadata,
            )
            session_output_dirs.append(Path(out["session_out"]))
            rows_summary.append({
                "session": session_name,
                "patient": patient,
                "session_num": session,
                "status": "ok",
                "nwb_file": str(nwb_fp) if nwb_fp is not None else None,
                "out_dir": str(out["session_out"]),
                "n_units": int(out["summary"].get("n_units_input", 0)),
                "n_correlation_rows": int(out["summary"].get("n_correlation_rows", 0)),
                "n_significant_rows": int(out["summary"].get("n_significant_rows", 0)),
            })
        except Exception as exc:
            errors.append((session_name, repr(exc)))
            log(f"[ERROR] {session_name}: {exc}", pool_cfg.verbose)

    pool_result = pool_saved_session_correlations(
        session_output_dirs=session_output_dirs,
        out_dir=pooled_out,
        cfg=cfg,
        recompute_fdr=pool_cfg.recompute_pooled_fdr,
        prefix="ALL_SESSIONS",
    ) if len(session_output_dirs) > 0 else {"summary": {}, "pooled": pd.DataFrame(), "significant": pd.DataFrame()}

    run_summary = {
        "config": _jsonify(asdict(pool_cfg)),
        "n_common_sessions_found": len(common_dirs),
        "n_sessions_run_or_reused": len(session_output_dirs),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "n_errors": len(errors),
        "errors": errors,
        "pooled_summary": pool_result.get("summary", {}),
    }
    with open(pooled_out / "run_all_fr_power_correlations_summary.json", "w", encoding="utf-8") as f:
        json.dump(_jsonify(run_summary), f, ensure_ascii=False, indent=2)

    pd.DataFrame(rows_summary).to_csv(pooled_out / "run_all_fr_power_correlations_sessions.csv", index=False)

    return {
        "session_summary": pd.DataFrame(rows_summary),
        "pooled": pool_result.get("pooled", pd.DataFrame()),
        "pooled_significant": pool_result.get("significant", pd.DataFrame()),
        "run_summary": run_summary,
        "output_root": output_root,
    }


# =============================================================================
# VARIANTES INDÉPENDANTES FR × LFP
# =============================================================================


def metric_variant_subdir(fr_norm_method: str, lfp_value_col: str) -> str:
    """Nom de sous-dossier stable pour une variante d'analyse."""
    return f"fr_{safe_name(fr_norm_method)}__lfp_{safe_name(lfp_value_col.replace('lfp_power_', ''))}"


def run_session_fr_power_correlation_variants(bundle: CommonSessionBundle,
                                              units: Dict[Any, Any],
                                              out_dir: str | Path,
                                              base_cfg: Optional[FRPowerCorrelationConfig] = None,
                                              fr_norm_methods: Tuple[FRNormMethod, ...] = ("logratio",),
                                              lfp_value_cols: Tuple[LFPValueColumn, ...] = ("lfp_power_pre_logratio",),
                                              correlation_methods: Optional[Tuple[CorrelationMethod, ...]] = None,
                                              dead_intervals_by_unit: Optional[Dict[Any, Any]] = None,
                                              unit_metadata: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Lance plusieurs variantes indépendantes pour une session.

    Hiérarchie créée :
        out_dir / fr_<méthode_FR>__lfp_<métrique_LFP> / SESSION / ...

    Cela évite de mélanger des résultats issus de normalisations différentes.
    """
    base_cfg = base_cfg or FRPowerCorrelationConfig()
    root = ensure_dir(out_dir)
    rows = []
    results: Dict[str, Any] = {}

    for fr_method in fr_norm_methods:
        for lfp_col in lfp_value_cols:
            cfg = replace(base_cfg, fr_norm_method=fr_method, lfp_value_col=lfp_col)
            if correlation_methods is not None:
                cfg = replace(cfg, correlation_methods=correlation_methods)
            variant = metric_variant_subdir(fr_method, lfp_col)
            variant_root = ensure_dir(root / variant)
            try:
                out = run_session_fr_power_correlations(
                    bundle=bundle,
                    units=units,
                    out_dir=variant_root,
                    cfg=cfg,
                    dead_intervals_by_unit=dead_intervals_by_unit,
                    unit_metadata=unit_metadata,
                )
                results[variant] = out
                rows.append({
                    "variant": variant,
                    "fr_norm_method": fr_method,
                    "lfp_value_col": lfp_col,
                    "status": "ok",
                    "session_out": str(out.get("session_out")),
                    "n_correlation_rows": int(out.get("summary", {}).get("n_correlation_rows", 0)),
                    "n_significant_rows": int(out.get("summary", {}).get("n_significant_rows", 0)),
                })
            except Exception as exc:
                rows.append({
                    "variant": variant,
                    "fr_norm_method": fr_method,
                    "lfp_value_col": lfp_col,
                    "status": "error",
                    "error": repr(exc),
                })

    summary_df = pd.DataFrame(rows)
    summary_fp = root / f"{bundle.session_name}_fr_power_variant_summary.csv"
    summary_df.to_csv(summary_fp, index=False)
    return {"results": results, "summary": summary_df, "summary_file": summary_fp, "out_dir": root}


@dataclass
class PooledFRPowerCorrelationVariantsConfig:
    """Configuration multi-sessions × variantes métriques."""

    common_root: str
    root_micro: str
    output_root: str
    hilbert_bands: Tuple[str, ...] = ("theta", "alpha", "beta", "low_gamma", "high_gamma")
    base_session_cfg: FRPowerCorrelationConfig = field(default_factory=FRPowerCorrelationConfig)
    fr_norm_methods: Tuple[FRNormMethod, ...] = ("logratio",)
    lfp_value_cols: Tuple[LFPValueColumn, ...] = ("lfp_power_pre_logratio",)
    correlation_methods: Optional[Tuple[CorrelationMethod, ...]] = None
    require_existing_nwb: bool = True
    overwrite_session_outputs: bool = True
    skip_existing_session_outputs: bool = False
    recompute_pooled_fdr: bool = True
    verbose: bool = True


def run_all_available_sessions_fr_power_correlation_variants(pool_cfg: PooledFRPowerCorrelationVariantsConfig,
                                                             unit_metadata_func: Optional[Any] = None,
                                                             dead_intervals_func: Optional[Any] = None) -> Dict[str, Any]:
    """Lance le pipeline multi-sessions pour plusieurs variantes indépendantes.

    Hiérarchie :
        output_root/
            fr_logratio__lfp_pre_logratio/
                per_session/...
                pooled_across_sessions/...
            fr_zscore__lfp_pre_zscore/
                ...
    """
    output_root = ensure_dir(pool_cfg.output_root)
    rows = []
    results: Dict[str, Any] = {}

    for fr_method in pool_cfg.fr_norm_methods:
        for lfp_col in pool_cfg.lfp_value_cols:
            cfg = replace(pool_cfg.base_session_cfg, fr_norm_method=fr_method, lfp_value_col=lfp_col)
            if pool_cfg.correlation_methods is not None:
                cfg = replace(cfg, correlation_methods=pool_cfg.correlation_methods)
            variant = metric_variant_subdir(fr_method, lfp_col)
            variant_root = ensure_dir(output_root / variant)
            sub_pool_cfg = PooledFRPowerCorrelationConfig(
                common_root=pool_cfg.common_root,
                root_micro=pool_cfg.root_micro,
                output_root=str(variant_root),
                hilbert_bands=pool_cfg.hilbert_bands,
                session_cfg=cfg,
                require_existing_nwb=pool_cfg.require_existing_nwb,
                overwrite_session_outputs=pool_cfg.overwrite_session_outputs,
                skip_existing_session_outputs=pool_cfg.skip_existing_session_outputs,
                recompute_pooled_fdr=pool_cfg.recompute_pooled_fdr,
                make_pooled_histograms=cfg.make_significant_histograms,
                verbose=pool_cfg.verbose,
            )
            try:
                out = run_all_available_sessions_fr_power_correlations(
                    pool_cfg=sub_pool_cfg,
                    unit_metadata_func=unit_metadata_func,
                    dead_intervals_func=dead_intervals_func,
                )
                results[variant] = out
                rows.append({
                    "variant": variant,
                    "fr_norm_method": fr_method,
                    "lfp_value_col": lfp_col,
                    "status": "ok",
                    "output_root": str(variant_root),
                    "n_session_rows": int(len(out.get("session_summary", []))),
                    "n_pooled_rows": int(len(out.get("pooled", []))),
                    "n_pooled_significant_rows": int(len(out.get("pooled_significant", []))),
                })
            except Exception as exc:
                rows.append({
                    "variant": variant,
                    "fr_norm_method": fr_method,
                    "lfp_value_col": lfp_col,
                    "status": "error",
                    "error": repr(exc),
                    "output_root": str(variant_root),
                })

    summary_df = pd.DataFrame(rows)
    summary_fp = output_root / "run_all_fr_power_variants_summary.csv"
    summary_df.to_csv(summary_fp, index=False)
    with open(output_root / "run_all_fr_power_variants_summary.json", "w", encoding="utf-8") as f:
        json.dump(_jsonify({"rows": rows}), f, ensure_ascii=False, indent=2)
    return {"results": results, "summary": summary_df, "summary_file": summary_fp, "output_root": output_root}


__all__ = [
    "FRPowerCorrelationConfig",
    "make_relative_bin_table",
    "relative_bin_to_absolute_micro",
    "build_fr_bin_table_for_units",
    "build_lfp_power_bin_table_for_band",
    "build_fr_power_trial_bin_table",
    "compute_correlations_from_trial_bin_table",
    "existing_session_corr_outputs",
    "load_existing_session_fr_power_result",
    "run_session_fr_power_correlations",
    "make_units_dict_from_spikes_object",
    "prepare_unit_metadata_and_dead_intervals",
    "PooledFRPowerCorrelationConfig",
    "find_common_session_dirs",
    "find_nwb_file",
    "run_all_available_sessions_fr_power_correlations",
    "pool_saved_session_correlations",
    "get_significant_correlations",
    "plot_significant_histograms_suite",
    "metric_variant_subdir",
    "run_session_fr_power_correlation_variants",
    "PooledFRPowerCorrelationVariantsConfig",
    "run_all_available_sessions_fr_power_correlation_variants",
]
