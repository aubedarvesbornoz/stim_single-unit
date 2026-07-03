#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lfp_hilbert_stats.py
====================

Statistiques sur les enveloppes Hilbert déjà exportées.

Philosophie
-----------
Ce module reprend la logique statistique développée pour les résultats Morlet,
mais l'applique à des séries temporelles Hilbert déjà agrégées par bandes.

L'unité d'observation reste un couple (session, trial, channel). Les observations
peuvent être regroupées :
- par condition principale (`cog+`, `controle`, `negatif`) ;
- par sous-catégories cognitives (`cog::souvenir`, etc.) ;
- par localité (`local` / `distant`) ;
- au niveau d'une session ou pooled across sessions.

Comme le signal Hilbert est déjà réduit à une courbe temporelle par bande, les tests
portent ici sur l'axe temps uniquement.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon
# import mne
from mne.stats import permutation_cluster_1samp_test

from lfp_preprocess_utils import (
    log,
    ensure_dir,
    safe_name,
    # parse_list_cell,
    parse_bipolar_shaft,
)

from lfp_morlet_stats import (
    fdr_bh,
    build_main_condition_index,
    build_cog_subcategory_index,
    get_channel_locality_for_trial
)

from lfp_hilbert_utils import (
    # HilbertConfig,
    # load_hilbert_session_exports,
    load_hilbert_band_epochs,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class HilbertStatsConfig:
    """Configuration des statistiques Hilbert."""

    input_root: str = "/home/aube/Documents/article_neuronal_stimic/effets_cog/theta_gamma_cog/results_hilbert"
    output_root: str = "/home/aube/Documents/article_neuronal_stimic/effets_cog/theta_gamma_cog/results_hilbert_stats"

    bands_to_test: Tuple[str, ...] = ("theta", "alpha", "beta", "low_gamma", "high_gamma")

    run_wilcoxon_fdr: bool = True
    run_cluster_perm: bool = True
    alpha_fdr: float = 0.05

    cluster_alpha: float = 0.05
    n_permutations: int = 2000
    cluster_threshold_p: float = 0.05
    tail: int = 0
    seed: int = 13

    min_trials_per_condition: int = 5
    make_main_groups: bool = True
    make_cog_subgroups: bool = True
    keep_main_groups_in_subgroup_mode: bool = True
    localities_to_test: Tuple[str, ...] = ("local", "distant")

    group_across_sessions: bool = True
    also_run_per_session: bool = False
    pooled_output_subdir: str = "pooled_across_sessions"

    save_figures: bool = True
    figure_dpi: int = 150
    verbose: bool = True


# ============================================================================
# CHARGEMENT ET EMPILAGE
# ============================================================================

def load_hilbert_session_metadata(session_dir: Path, session: str) -> dict:
    fp = session_dir / f"{session}_metadata.json"
    if not fp.exists():
        raise FileNotFoundError(fp)
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)



def load_hilbert_trials_table(session_dir: Path, session: str) -> pd.DataFrame:
    fp = session_dir / f"{session}_trial_table.csv"
    if not fp.exists():
        raise FileNotFoundError(fp)
    return pd.read_csv(fp)



def load_hilbert_times(session_dir: Path, session: str) -> np.ndarray:
    fp = session_dir / f"{session}_times.npy"
    if not fp.exists():
        raise FileNotFoundError(fp)
    return np.load(fp)



def stack_hilbert_band_condition_locality(session: str,
                                          session_dir: Path,
                                          trials_df: pd.DataFrame,
                                          bp_names: Sequence[str],
                                          trial_indices: np.ndarray,
                                          locality: str,
                                          band_name: str,
                                          condition_name: Optional[str] = None
                                          ) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Empile les courbes Hilbert d'une bande pour une condition × localité au sein d'une session.

    Retour
    ------
    X_stack : shape = (n_observations, n_times)
    obs_df  : description des observations empilées
    """
    if locality not in {"local", "distant"}:
        raise ValueError(f"locality doit être 'local' ou 'distant', reçu: {locality}")

    arr = load_hilbert_band_epochs(session_dir=session_dir, session=session, band_name=band_name)
    # arr shape = (n_trials, n_channels, n_times)
    if arr.ndim != 3:
        raise ValueError(f"{session} {band_name}: shape inattendue {arr.shape}")

    trial_indices = np.asarray(np.unique(trial_indices), dtype=int)
    stacks: List[np.ndarray] = []
    rows: List[dict] = []

    for ch_idx, ch_name in enumerate(bp_names):
        if ch_idx >= arr.shape[1]:
            continue
        channel_shaft = parse_bipolar_shaft(ch_name)

        selected_rows = []
        for trial_idx in trial_indices:
            if trial_idx >= len(trials_df):
                continue
            if trial_idx >= arr.shape[0]:
                continue

            stim_shaft = trials_df.loc[trial_idx, "stim_shaft"]
            obs_locality = get_channel_locality_for_trial(ch_name, stim_shaft)
            if obs_locality == "unknown" or obs_locality != locality:
                continue

            selected_rows.append(trial_idx)
            rows.append({
                "session": session,
                "condition": condition_name,
                "trial_idx": int(trial_idx),
                "channel_name": ch_name,
                "channel_shaft": channel_shaft,
                "stim_shaft": stim_shaft,
                "label_stim": trials_df.loc[trial_idx, "label_stim"],
                "group_label": trials_df.loc[trial_idx, "group_label"],
                "cog_labels": trials_df.loc[trial_idx, "cog_labels"] if "cog_labels" in trials_df.columns else None,
                "locality": locality,
                "band_name": band_name,
            })

        if len(selected_rows) > 0:
            stacks.append(arr[selected_rows, ch_idx, :])

    if len(stacks) == 0:
        raise ValueError(f"Aucune observation pour {session} | {condition_name} | {locality} | {band_name}")

    X_stack = np.concatenate(stacks, axis=0).astype(np.float32)
    obs_df = pd.DataFrame(rows)

    if len(obs_df) != X_stack.shape[0]:
        raise ValueError(f"Incohérence X_stack/obs_df: {X_stack.shape[0]} vs {len(obs_df)}")

    return X_stack, obs_df



def validate_same_hilbert_time_grid_across_sessions(session_dirs: Sequence[Path]) -> np.ndarray:
    """Vérifie que toutes les sessions ont le même axe temps Hilbert."""
    ref_times = None
    for session_dir in session_dirs:
        session = session_dir.name
        times = load_hilbert_times(session_dir, session)
        if ref_times is None:
            ref_times = times
            continue
        if not np.array_equal(times, ref_times):
            raise ValueError(f"{session}: axe temps Hilbert différent entre sessions")
    return ref_times



def stack_hilbert_band_condition_locality_across_sessions(session_dirs: Sequence[Path],
                                                          condition_name: str,
                                                          locality: str,
                                                          band_name: str,
                                                          cfg: HilbertStatsConfig,
                                                          subgroup_mode: bool = False
                                                          ) -> Tuple[np.ndarray, pd.DataFrame]:
    """Empile les observations Hilbert across sessions pour condition × localité × bande."""
    all_stacks = []
    all_rows = []

    for session_dir in session_dirs:
        session = session_dir.name
        trials_df = load_hilbert_trials_table(session_dir, session)
        meta = load_hilbert_session_metadata(session_dir, session)
        bp_names = meta.get("bipolar_names", [])
        if len(bp_names) == 0:
            continue
        if "stim_shaft" not in trials_df.columns:
            raise ValueError(f"{session}: colonne 'stim_shaft' absente, il faut régénérer les exports Hilbert")

        if subgroup_mode:
            cond_index = build_cog_subcategory_index(trials_df, cfg.min_trials_per_condition, cfg.keep_main_groups_in_subgroup_mode)
        else:
            cond_index = build_main_condition_index(trials_df,cfg.min_trials_per_condition)

        if condition_name not in cond_index:
            continue

        trial_indices = cond_index[condition_name]

        try:
            X_session, obs_session = stack_hilbert_band_condition_locality(
                session=session,
                session_dir=session_dir,
                trials_df=trials_df,
                bp_names=bp_names,
                trial_indices=trial_indices,
                locality=locality,
                band_name=band_name,
                condition_name=condition_name,
            )
        except Exception:
            continue

        all_stacks.append(X_session)
        all_rows.append(obs_session)

    if len(all_stacks) == 0:
        raise ValueError(f"Aucune observation pooled pour {condition_name} | {locality} | {band_name}")

    X_all = np.concatenate(all_stacks, axis=0).astype(np.float32)
    obs_df = pd.concat(all_rows, axis=0, ignore_index=True)
    return X_all, obs_df


# ============================================================================
# STATS TEMPORELLES
# ============================================================================

def wilcoxon_time_against_baseline(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Wilcoxon signed-rank à chaque temps contre 0."""
    if X.ndim != 2:
        raise ValueError(f"X doit être 2D, reçu {X.shape}")
    n_obs, n_times = X.shape
    stat = np.full(n_times, np.nan, dtype=float)
    pvals = np.full(n_times, np.nan, dtype=float)

    for ti in range(n_times):
        x = X[:, ti]
        x = x[np.isfinite(x)]
        if len(x) < 1:
            continue
        if np.allclose(x, 0.0):
            stat[ti] = 0.0
            pvals[ti] = 1.0
            continue
        try:
            res = wilcoxon(x, y=None, zero_method="wilcox", alternative="two-sided", mode="auto", correction=False)
            stat[ti] = float(res.statistic)
            pvals[ti] = float(res.pvalue)
        except ValueError:
            stat[ti] = np.nan
            pvals[ti] = 1.0

    return stat, pvals


def cluster_1samp_time_against_baseline(X: np.ndarray,
                                        cluster_alpha: float,
                                        n_permutations: int,
                                        tail: int,
                                        seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cluster permutation 1D sur l'axe temps.

    X shape = (n_observations, n_times)
    """
    if X.ndim != 2:
        raise ValueError(f"X doit être 2D, reçu {X.shape}")

    n_obs, n_times = X.shape
    if n_obs < 2:
        raise ValueError("Cluster permutation requiert au moins 2 observations")

    if tail == 0:
        thr = t.ppf(1.0 - cluster_alpha / 2.0, df=n_obs - 1)
    elif tail == 1:
        thr = t.ppf(1.0 - cluster_alpha, df=n_obs - 1)
    elif tail == -1:
        thr = -t.ppf(1.0 - cluster_alpha, df=n_obs - 1)
    else:
        raise ValueError(f"tail invalide: {tail}")

    T_obs, clusters, cluster_pvals, _ = permutation_cluster_1samp_test(
        X=X.astype(np.float64),
        threshold=thr,
        n_permutations=n_permutations,
        tail=tail,
        out_type="mask",
        seed=seed,
        verbose=False,
    )

    sig_mask = np.zeros(n_times, dtype=bool)

    for cluster, pval in zip(clusters, cluster_pvals):
        if pval > 0.05:
            continue

        # Cas 1 : MNE renvoie directement un masque booléen
        if isinstance(cluster, np.ndarray) and cluster.dtype == bool:
            cl_mask = cluster.astype(bool)

        # Cas 2 : MNE renvoie un tuple contenant le masque
        elif isinstance(cluster, tuple):
            if len(cluster) == 1:
                item = cluster[0]

                # tuple d'indices
                if isinstance(item, np.ndarray) and item.dtype != bool:
                    cl_mask = np.zeros(n_times, dtype=bool)
                    cl_mask[item] = True

                # tuple contenant un masque
                elif isinstance(item, np.ndarray) and item.dtype == bool:
                    cl_mask = item.astype(bool)

                # slice éventuel
                elif isinstance(item, slice):
                    cl_mask = np.zeros(n_times, dtype=bool)
                    cl_mask[item] = True

                else:
                    raise TypeError(f"Format de cluster non géré: {type(item)}")

            else:
                raise TypeError(f"Tuple cluster inattendu: len={len(cluster)}")

        # Cas 3 : MNE renvoie un slice
        elif isinstance(cluster, slice):
            cl_mask = np.zeros(n_times, dtype=bool)
            cl_mask[cluster] = True

        else:
            raise TypeError(f"Format de cluster non géré: {type(cluster)}")

        if cl_mask.shape[0] != n_times:
            raise ValueError(f"cl_mask shape inattendue: {cl_mask.shape}, attendu ({n_times},)")

        sig_mask = sig_mask | cl_mask

    return np.asarray(T_obs), sig_mask, np.asarray(cluster_pvals)

# ============================================================================
# FIGURES
# ============================================================================

def plot_hilbert_timecourse_with_significance(mean_trace: np.ndarray,
                                              sig_mask: np.ndarray,
                                              times: np.ndarray,
                                              out_file: Path,
                                              title: str,
                                              dpi: int = 150) -> None:
    """Trace une courbe moyenne Hilbert avec surimpression des temps significatifs."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(times, mean_trace, color="black", linewidth=1.8)
    ax.axhline(0, color="k", linestyle=":", linewidth=1.0)

    if sig_mask is not None and np.any(sig_mask):
        y_sig = mean_trace.copy()
        y_sig[~sig_mask] = np.nan
        ax.plot(times, y_sig, color="red", linewidth=2.6)

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hilbert amplitude")
    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# SAUVEGARDE CONFIG
# ============================================================================

def save_hilbert_stats_config(out_dir: Path, cfg: HilbertStatsConfig) -> None:
    with open(out_dir / "hilbert_stats_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)


# ============================================================================
# ORCHESTRATEURS STATS
# ============================================================================

def run_stats_for_one_hilbert_condition(out_condition_dir: Path,
                                        condition_name: str,
                                        locality: str,
                                        band_name: str,
                                        X: np.ndarray,
                                        obs_df: pd.DataFrame,
                                        times: np.ndarray,
                                        cfg: HilbertStatsConfig,
                                        summary_rows: List[dict],
                                        scope: str,
                                        session_name: Optional[str] = None) -> None:
    """Exécute les stats pour une condition × localité × bande sur des courbes Hilbert."""
    if X.ndim != 2:
        raise ValueError(f"X doit être 2D, reçu {X.shape}")

    out_dir = out_condition_dir / locality / band_name
    ensure_dir(out_dir)

    mean_trace = np.mean(X, axis=0)
    median_trace = np.median(X, axis=0)
    np.save(out_dir / "mean_trace.npy", mean_trace.astype(np.float32))
    np.save(out_dir / "median_trace.npy", median_trace.astype(np.float32))
    obs_df.to_csv(out_dir / "observations_table.csv", index=False)

    summary_base = {
        "scope": scope,
        "session": session_name,
        "condition": condition_name,
        "locality": locality,
        "band_name": band_name,
        "n_observations": int(X.shape[0]),
        "n_times": int(X.shape[1]),
        "n_unique_trials": int(obs_df[["session", "trial_idx"]].drop_duplicates().shape[0]) if "session" in obs_df.columns else int(obs_df["trial_idx"].nunique()),
        "n_unique_channels": int(obs_df[["session", "channel_name"]].drop_duplicates().shape[0]) if "session" in obs_df.columns else int(obs_df["channel_name"].nunique()),
        "n_unique_sessions": int(obs_df["session"].nunique()) if "session" in obs_df.columns else 1,
    }

    if cfg.run_wilcoxon_fdr:
        stat, pvals = wilcoxon_time_against_baseline(X)
        sig_mask_fdr, pvals_fdr = fdr_bh(pvals, alpha=cfg.alpha_fdr)
        np.save(out_dir / "stat_wilcoxon.npy", stat.astype(np.float32))
        np.save(out_dir / "pvals_wilcoxon.npy", pvals.astype(np.float32))
        np.save(out_dir / "pvals_wilcoxon_fdr.npy", pvals_fdr.astype(np.float32))
        np.save(out_dir / "sig_mask_wilcoxon_fdr.npy", sig_mask_fdr.astype(np.uint8))

        if cfg.save_figures:
            plot_hilbert_timecourse_with_significance(
                mean_trace=mean_trace,
                sig_mask=sig_mask_fdr,
                times=times,
                out_file=out_dir / "figure_wilcoxon_fdr.png",
                title=f"{scope} | {condition_name} | {locality} | {band_name} | Wilcoxon+FDR",
                dpi=cfg.figure_dpi,
            )

        summary_rows.append({
            **summary_base,
            "method": "wilcoxon_fdr",
            "n_sig_timepoints": int(sig_mask_fdr.sum()),
            "frac_sig_timepoints": float(sig_mask_fdr.mean()),
        })

    if cfg.run_cluster_perm and X.shape[0] >= 2:
        T_obs, sig_mask_cluster, cluster_pvals = cluster_1samp_time_against_baseline(
            X=X,
            cluster_alpha=cfg.cluster_threshold_p,
            n_permutations=cfg.n_permutations,
            tail=cfg.tail,
            seed=cfg.seed,
        )
        np.save(out_dir / "T_obs_cluster.npy", T_obs.astype(np.float32))
        np.save(out_dir / "sig_mask_cluster.npy", sig_mask_cluster.astype(np.uint8))
        np.save(out_dir / "cluster_pvals.npy", cluster_pvals.astype(np.float32))

        if cfg.save_figures:
            plot_hilbert_timecourse_with_significance(
                mean_trace=mean_trace,
                sig_mask=sig_mask_cluster,
                times=times,
                out_file=out_dir / "figure_cluster.png",
                title=f"{scope} | {condition_name} | {locality} | {band_name} | Cluster perm",
                dpi=cfg.figure_dpi,
            )

        summary_rows.append({
            **summary_base,
            "method": "cluster_perm",
            "n_sig_timepoints": int(sig_mask_cluster.sum()),
            "frac_sig_timepoints": float(sig_mask_cluster.mean()),
            "n_clusters_total": int(len(cluster_pvals)),
            "n_clusters_sig": int(np.sum(cluster_pvals <= cfg.cluster_alpha)),
        })



def run_hilbert_session_condition_stats(session_dir: Path, cfg: HilbertStatsConfig) -> Path:
    """Stats Hilbert session par session."""
    session = session_dir.name
    log(f"\n=== Hilbert stats session {session} ===", cfg.verbose)

    out_session_root = Path(cfg.output_root) / session
    ensure_dir(out_session_root)
    save_hilbert_stats_config(out_session_root, cfg)

    trials_df = load_hilbert_trials_table(session_dir, session)
    meta = load_hilbert_session_metadata(session_dir, session)
    times = load_hilbert_times(session_dir, session)
    bp_names = meta.get("bipolar_names", [])
    if len(bp_names) == 0:
        raise ValueError(f"{session}: pas de bipolar_names dans metadata")

    summary_rows: List[dict] = []

    def _run_group_block(cond_index: Dict[str, np.ndarray], root_out: Path, scope_label: str):
        ensure_dir(root_out)
        for condition_name, trial_indices in cond_index.items():
            cond_out = root_out / safe_name(condition_name)
            ensure_dir(cond_out)
            for locality in cfg.localities_to_test:
                for band_name in cfg.bands_to_test:
                    try:
                        X, obs_df = stack_hilbert_band_condition_locality(
                            session=session,
                            session_dir=session_dir,
                            trials_df=trials_df,
                            bp_names=bp_names,
                            trial_indices=trial_indices,
                            locality=locality,
                            band_name=band_name,
                            condition_name=condition_name,
                        )
                        run_stats_for_one_hilbert_condition(
                            out_condition_dir=cond_out,
                            condition_name=condition_name,
                            locality=locality,
                            band_name=band_name,
                            X=X,
                            obs_df=obs_df,
                            times=times,
                            cfg=cfg,
                            summary_rows=summary_rows,
                            scope=scope_label,
                            session_name=session,
                        )
                    except Exception as exc:
                        log(f"[WARN] {session} {condition_name} {locality} {band_name}: {exc}", cfg.verbose)

    if cfg.make_main_groups:
        cond_index_main = build_main_condition_index(trials_df, cfg.min_trials_per_condition)
        _run_group_block(cond_index_main, out_session_root / "condition_main", scope_label="per_session")

    if cfg.make_cog_subgroups:
        cond_index_sub = build_cog_subcategory_index(trials_df, cfg.min_trials_per_condition, cfg.keep_main_groups_in_subgroup_mode)
        _run_group_block(cond_index_sub, out_session_root / "condition_subcategories", scope_label="per_session")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_session_root / "summary_hilbert_stats.csv", index=False)
    return out_session_root



def run_pooled_hilbert_condition_stats(cfg: HilbertStatsConfig) -> Path:
    """Stats Hilbert pooled across sessions."""
    input_root = Path(cfg.input_root)
    out_root = Path(cfg.output_root) / cfg.pooled_output_subdir
    ensure_dir(out_root)
    save_hilbert_stats_config(out_root, cfg)

    session_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])
    if len(session_dirs) == 0:
        raise RuntimeError(f"Aucun sous-dossier session trouvé dans {input_root}")

    ref_times = validate_same_hilbert_time_grid_across_sessions(session_dirs)
    summary_rows: List[dict] = []

    def _collect_condition_names(subgroup_mode: bool) -> List[str]:
        names = set()
        for session_dir in session_dirs:
            session = session_dir.name
            trials_df = load_hilbert_trials_table(session_dir, session)
            if subgroup_mode:
                cond_index = build_cog_subcategory_index(trials_df, cfg.min_trials_per_condition, cfg.keep_main_groups_in_subgroup_mode)
            else:
                cond_index = build_main_condition_index(trials_df, cfg.min_trials_per_condition)
            names.update(cond_index.keys())
        return sorted(names)

    def _run_group_block(condition_names: Sequence[str], root_out: Path, subgroup_mode: bool):
        ensure_dir(root_out)
        for condition_name in condition_names:
            cond_out = root_out / safe_name(condition_name)
            ensure_dir(cond_out)
            for locality in cfg.localities_to_test:
                for band_name in cfg.bands_to_test:
                    try:
                        X, obs_df = stack_hilbert_band_condition_locality_across_sessions(
                            session_dirs=session_dirs,
                            condition_name=condition_name,
                            locality=locality,
                            band_name=band_name,
                            cfg=cfg,
                            subgroup_mode=subgroup_mode,
                        )
                        run_stats_for_one_hilbert_condition(
                            out_condition_dir=cond_out,
                            condition_name=condition_name,
                            locality=locality,
                            band_name=band_name,
                            X=X,
                            obs_df=obs_df,
                            times=ref_times,
                            cfg=cfg,
                            summary_rows=summary_rows,
                            scope="across_sessions",
                            session_name=None,
                        )
                    except Exception as exc:
                        log(f"[WARN] pooled {condition_name} {locality} {band_name}: {exc}", cfg.verbose)

    if cfg.make_main_groups:
        _run_group_block(_collect_condition_names(False), out_root / "condition_main", subgroup_mode=False)
    if cfg.make_cog_subgroups:
        _run_group_block(_collect_condition_names(True), out_root / "condition_subcategories", subgroup_mode=True)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_root / "summary_hilbert_stats.csv", index=False)
    return out_root



def run_all_hilbert_stats(cfg: HilbertStatsConfig) -> Dict[str, Any]:
    """Point d'entrée principal des stats Hilbert."""
    output_root = Path(cfg.output_root)
    ensure_dir(output_root)
    errors: List[Tuple[str, str]] = []

    if cfg.group_across_sessions:
        try:
            run_pooled_hilbert_condition_stats(cfg)
        except Exception as exc:
            errors.append(("pooled_across_sessions", repr(exc)))
            log(f"[ERROR] pooled Hilbert: {exc}", cfg.verbose)

    if cfg.also_run_per_session:
        input_root = Path(cfg.input_root)
        session_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])
        for session_dir in session_dirs:
            try:
                run_hilbert_session_condition_stats(session_dir, cfg)
            except Exception as exc:
                errors.append((session_dir.name, repr(exc)))
                log(f"[ERROR] Hilbert {session_dir.name}: {exc}", cfg.verbose)

    summary = {"config": asdict(cfg), "n_errors": len(errors), "errors": errors}
    with open(output_root / "run_summary_hilbert_stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


__all__ = [
    "HilbertStatsConfig",
    "load_hilbert_session_metadata",
    "load_hilbert_trials_table",
    "load_hilbert_times",
    "stack_hilbert_band_condition_locality",
    "validate_same_hilbert_time_grid_across_sessions",
    "stack_hilbert_band_condition_locality_across_sessions",
    "wilcoxon_time_against_baseline",
    "cluster_1samp_time_against_baseline",
    "plot_hilbert_timecourse_with_significance",
    "save_hilbert_stats_config",
    "run_stats_for_one_hilbert_condition",
    "run_hilbert_session_condition_stats",
    "run_pooled_hilbert_condition_stats",
    "run_all_hilbert_stats",
]
