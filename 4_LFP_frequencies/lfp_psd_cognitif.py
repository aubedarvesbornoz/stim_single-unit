#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PSD grand-average pour stimulations intracérébrales cognitives vs contrôles.

Objectif
--------
Construire des PSD simples à partir d'epochs SEEG bipolaires, groupées en :
    - pre-EBS        : fenêtres pré-stimulation de toutes les EBS incluses
    - post-EBS_cog   : fenêtres post-stimulation des EBS avec effet cognitif
    - post-EBS_ctrl  : fenêtres post-stimulation des EBS contrôle

Le module réutilise les fonctions utilitaires de `lfp_preprocess.py` et fournit
une API pratique pour notebook.

Exemple notebook
----------------
from pathlib import Path
import lfp_psd_cognitif as psd

cfg = psd.PSDConfig(
    root_dir=Path('/home/aube/Documents/article_neuronal_stimic/effets_cog/theta_gamma_cog'),
    output_dir=Path('/home/aube/Documents/article_neuronal_stimic/effets_cog/theta_gamma_cog/PSD_cognitif'),
    pre_length=2.0,
    post_length=2.0,
    epsilon=0.2,
    fmin=1.0,
    fmax=150.0,
)

all_psd, grand = psd.run_psd_pipeline(cfg)
psd.plot_grand_average_psd(grand, cfg)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

# Adapte l'import si ton fichier s'appelle autrement ou est dans un sous-dossier.
from lfp_preprocess import (
    log,
    ensure_dir,
    safe_name,
    list_trc_sessions,
    load_bad_channels_table,
    get_bad_channels_for_session,
    load_trc_as_mne_raw,
    apply_filters,
    build_adjacent_bipolar_pairs,
    make_bipolar_data,
    recover_precise_macro_stim_events,
    find_cog_file,
    read_cog_file,
    merge_event_tables,
    add_windows_to_trials,
    keep_trials_fitting_signal,
    extract_pre_post_epochs,
)


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class PSDConfig:
    root_dir: Path
    output_dir: Path

    # Fenêtres temporelles autour de la stimulation.
    pre_length: float = 2.0
    post_length: float = 2.0
    epsilon: float = 0.2

    # PSD Welch.
    fmin: float = 1.0
    fmax: float = 150.0
    nperseg_s: float = 0.5
    noverlap_frac: float = 0.5
    window: str = "hann"
    average: str = "mean"  # scipy.signal.welch: 'mean' ou 'median' selon version scipy

    # Prétraitement.
    do_notch: bool = False
    notch_freqs: Tuple[float, ...] = (50.0, 100.0, 150.0)
    notch_q: float = 30.0
    do_highpass: bool = True
    highpass_hz: float = 0.1

    # Groupes inclus.
    include_cog: bool = True
    include_controle: bool = True
    include_negatif: bool = True

    # Sorties.
    save_tables: bool = True
    save_figures: bool = True
    verbose: bool = True


# =============================================================================
# PSD CORE
# =============================================================================

def _welch_params(sfreq: float, n_times: int, cfg: PSDConfig) -> Tuple[int, int]:
    """Convertit les paramètres Welch en échantillons, bornés par la longueur d'epoch."""
    nperseg = int(round(cfg.nperseg_s * sfreq))
    nperseg = max(8, min(nperseg, n_times))
    noverlap = int(round(cfg.noverlap_frac * nperseg))
    noverlap = max(0, min(noverlap, nperseg - 1))
    return nperseg, noverlap


def compute_epochs_psd(
    epochs: np.ndarray,
    sfreq: float,
    cfg: PSDConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule une PSD Welch pour un tenseur d'epochs.

    Paramètres
    ----------
    epochs : np.ndarray
        Forme attendue : (n_trials, n_channels, n_times).
    sfreq : float
        Fréquence d'échantillonnage.
    cfg : PSDConfig
        Configuration PSD.

    Retour
    ------
    freqs : np.ndarray
        Fréquences conservées, en Hz.
    psd : np.ndarray
        PSD de forme (n_trials, n_channels, n_freqs), en V²/Hz.
    """
    if epochs.ndim != 3:
        raise ValueError(f"epochs doit avoir la forme (trials, channels, times), reçu {epochs.shape}")
    if epochs.shape[0] == 0:
        raise ValueError("Aucune epoch à analyser")

    nperseg, noverlap = _welch_params(sfreq, epochs.shape[-1], cfg)

    try:
        freqs, pxx = signal.welch(
            epochs,
            fs=sfreq,
            window=cfg.window,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            return_onesided=True,
            scaling="density",
            axis=-1,
            average=cfg.average,
        )
    except TypeError:
        # Compatibilité avec scipy plus ancien sans argument average.
        freqs, pxx = signal.welch(
            epochs,
            fs=sfreq,
            window=cfg.window,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            return_onesided=True,
            scaling="density",
            axis=-1,
        )

    keep = (freqs >= cfg.fmin) & (freqs <= cfg.fmax)
    return freqs[keep], pxx[..., keep]


def psd_to_long_table(
    psd_arr: np.ndarray,
    freqs: np.ndarray,
    session: str,
    bp_names: Sequence[str],
    trials_df: pd.DataFrame,
    condition: str,
    epoch_kind: str,
) -> pd.DataFrame:
    """
    Convertit un tenseur PSD en table longue.

    Sortie : une ligne par session × condition × trial × canal × fréquence.
    """
    rows = []
    n_trials, n_channels, n_freqs = psd_arr.shape

    if len(bp_names) != n_channels:
        raise ValueError("Nombre de noms bipolaires incompatible avec l'axe canal PSD")
    if len(trials_df) != n_trials:
        raise ValueError("Nombre de lignes trials_df incompatible avec l'axe trial PSD")

    for ti in range(n_trials):
        trial = trials_df.iloc[ti]
        for ci, ch in enumerate(bp_names):
            rows.append(pd.DataFrame({
                "session": session,
                "condition": condition,
                "epoch_kind": epoch_kind,
                "trial_local_index": ti,
                "stim_index": int(trial.get("stim_index", ti)),
                "label_stim": str(trial.get("label_stim", "")),
                "group_label": str(trial.get("group_label", "")),
                "lobe": str(trial.get("lobe", "")),
                "bp_channel": ch,
                "freq_hz": freqs,
                "psd_v2_hz": psd_arr[ti, ci, :],
            }))

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# =============================================================================
# SESSION-LEVEL PIPELINE
# =============================================================================

def _select_trials_for_psd(trials: pd.DataFrame, cfg: PSDConfig) -> pd.DataFrame:
    """Garde les groupes utiles pour l'analyse PSD cognitive."""
    groups = []
    if cfg.include_cog:
        groups.append("cog+")
    if cfg.include_controle:
        groups.append("controle")
    if cfg.include_negatif:
        groups.append("negatif")

    out = trials.loc[trials["group_label"].isin(groups)].copy()
    return out.reset_index(drop=True)


def compute_session_psd(session: str, cfg: PSDConfig) -> pd.DataFrame:
    """
    Calcule les PSD pré/post d'une session TRC si inexistant, sinon récupère sur disque.

    Retour
    ------
    pd.DataFrame
        Table longue avec les conditions : pre-EBS, post-EBS_cog, post-EBS_ctrl.
    """
    out_path = Path(cfg.output_dir) / "session_tables" / f"{safe_name(session)}_psd_long__pre={float(cfg.pre_length)}s_post={float(cfg.post_length)}s.parquet"

    if os.path.exists(out_path):
        out = pd.read_parquet(out_path)
    
    else : 
        root_dir = Path(cfg.root_dir)
        trc_path = root_dir / f"{session}.TRC"

        log(f"\n[SESSION] {session}", cfg.verbose)

        # Chargements événements.
        cog_df = read_cog_file(find_cog_file(root_dir, session))
        trc_corr_df = recover_precise_macro_stim_events(session, root_dir)
        trials = merge_event_tables(session, cog_df, trc_corr_df)
        trials = _select_trials_for_psd(trials, cfg)
        if len(trials) == 0:
            log(f"[WARN] {session}: aucune stimulation cog/contrôle utilisable", cfg.verbose)
            return pd.DataFrame()

        # Chargement signal et montage bipolaire.
        raw = load_trc_as_mne_raw(trc_path, verbose=cfg.verbose)
        sfreq = float(raw.info["sfreq"])
        mono_ch_names = list(raw.ch_names)
        mono_data = raw.get_data()

        bad_df = load_bad_channels_table(root_dir)
        bad_channels = get_bad_channels_for_session(bad_df, session)
        bipolar_pairs = build_adjacent_bipolar_pairs(mono_ch_names, bad_channels)

        if len(bipolar_pairs) == 0:
            log(f"[WARN] {session}: aucun canal bipolaire adjacent valide", cfg.verbose)
            return pd.DataFrame()

        filt = apply_filters(
            mono_data,
            sfreq=sfreq,
            do_notch=cfg.do_notch,
            notch_freqs=cfg.notch_freqs,
            notch_q=cfg.notch_q,
            do_highpass=cfg.do_highpass,
            highpass_hz=cfg.highpass_hz,
        )
        data_bp, bp_names = make_bipolar_data(filt, mono_ch_names, bipolar_pairs)

        # Epoching pré/post.
        signal_duration_s = data_bp.shape[-1] / sfreq
        trials = add_windows_to_trials(trials, cfg.pre_length, cfg.post_length, cfg.epsilon)
        trials = keep_trials_fitting_signal(trials, signal_duration_s, verbose=cfg.verbose)

        if len(trials) == 0:
            log(f"[WARN] {session}: aucune stimulation avec fenêtres complètes", cfg.verbose)
            return pd.DataFrame()

        pre_epochs, post_epochs, _, _ = extract_pre_post_epochs(
            data_bp=data_bp,
            sfreq=sfreq,
            stims_df=trials,
            pre_length=cfg.pre_length,
            post_length=cfg.post_length,
        )

        # Sécurité : extract_pre_post_epochs peut ignorer des essais si tailles inattendues.
        n_kept = min(len(trials), pre_epochs.shape[0], post_epochs.shape[0])
        trials = trials.iloc[:n_kept].reset_index(drop=True)
        pre_epochs = pre_epochs[:n_kept]
        post_epochs = post_epochs[:n_kept]

        if n_kept == 0:
            log(f"[WARN] {session}: aucune epoch extraite", cfg.verbose)
            return pd.DataFrame()

        # PSD pré : toutes les baselines incluses.
        freqs, pre_psd = compute_epochs_psd(pre_epochs, sfreq, cfg)
        tables = [psd_to_long_table(
            pre_psd, freqs, session, bp_names, trials,
            condition="pre-EBS", epoch_kind="pre",
        )]

        # PSD post cog.
        cog_mask = trials["group_label"].eq("cog+").to_numpy()
        if cog_mask.any():
            _, post_cog_psd = compute_epochs_psd(post_epochs[cog_mask], sfreq, cfg)
            tables.append(psd_to_long_table(
                post_cog_psd, freqs, session, bp_names, trials.loc[cog_mask].reset_index(drop=True),
                condition="post-EBS_cog", epoch_kind="post",
            ))

        # PSD post contrôle.
        ctrl_mask = trials["group_label"].eq("controle").to_numpy()
        if ctrl_mask.any():
            _, post_ctrl_psd = compute_epochs_psd(post_epochs[ctrl_mask], sfreq, cfg)
            tables.append(psd_to_long_table(
                post_ctrl_psd, freqs, session, bp_names, trials.loc[ctrl_mask].reset_index(drop=True),
                condition="post-EBS_ctrl", epoch_kind="post",
            ))

        # PSD post négatif.
        neg_mask = trials["group_label"].eq("negatif").to_numpy()
        if neg_mask.any():
            _, post_neg_psd = compute_epochs_psd(post_epochs[neg_mask], sfreq, cfg)
            tables.append(psd_to_long_table(
                post_neg_psd, freqs, session, bp_names, trials.loc[neg_mask].reset_index(drop=True),
                condition="post-EBS_neg", epoch_kind="post",
            ))

        out = pd.concat([t for t in tables if len(t) > 0], ignore_index=True)
        if cfg.save_tables:
            ensure_dir(Path(cfg.output_dir) / "session_tables")
            out.to_parquet(out_path, index=False)

    return out


# =============================================================================
# GRAND AVERAGE
# =============================================================================

def compute_grand_average(
    psd_long: pd.DataFrame,
    average_first: str = "trial_channel",
) -> pd.DataFrame:
    """
    Calcule un grand average PSD par condition.

    Paramètres
    ----------
    psd_long : pd.DataFrame
        Table longue issue de compute_session_psd ou run_psd_pipeline.
    average_first : str
        - 'trial_channel' : moyenne directe de toutes les lignes condition × fréquence.
        - 'session'       : moyenne d'abord par session, puis grand average des sessions.
        - 'session_channel': moyenne d'abord par session × canal, puis grand average.

    Retour
    ------
    pd.DataFrame
        Colonnes : condition, freq_hz, mean_psd_v2_hz, sem_psd_v2_hz, n_units.
    """
    if len(psd_long) == 0:
        return pd.DataFrame()

    if average_first == "trial_channel":
        unit_cols = ["condition", "freq_hz"]
        tmp = psd_long.copy()
    elif average_first == "session":
        tmp = (
            psd_long
            .groupby(["session", "condition", "freq_hz"], as_index=False)["psd_v2_hz"]
            .mean()
        )
        unit_cols = ["condition", "freq_hz"]
    elif average_first == "session_channel":
        tmp = (
            psd_long
            .groupby(["session", "bp_channel", "condition", "freq_hz"], as_index=False)["psd_v2_hz"]
            .mean()
        )
        unit_cols = ["condition", "freq_hz"]
    else:
        raise ValueError("average_first doit être 'trial_channel', 'session' ou 'session_channel'")
    grand = (
        tmp.groupby(unit_cols)["psd_v2_hz"]
        .agg(mean_psd_v2_hz="mean", sd_psd_v2_hz="std", n_units="count")
        .reset_index()
    )
    grand["sem_psd_v2_hz"] = grand["sd_psd_v2_hz"] / np.sqrt(grand["n_units"].clip(lower=1))
    return grand


# =============================================================================
# PLOTS
# =============================================================================

def plot_grand_average_psd(
    grand: pd.DataFrame,
    cfg: PSDConfig,
    yscale: str = "log",
    show_sem: bool = True,
    title: str = "Grand-average PSD",
) -> plt.Figure:
    """Trace les PSD grand-averaged par condition."""
    fig, ax = plt.subplots(figsize=(8, 12))

    styles = {
        "pre-EBS": {"label": "pre-EBS", "color": "gray"},
        "post-EBS_cog": {"label": "post-EBS cog", "color": "green"},
        "post-EBS_ctrl": {"label": "post-EBS contrôle", "color": "blue"},
        "post-EBS_neg": {"label": "post-EBS négatif", "color": "red"},
    }

    for cond, style in styles.items():
        sub = grand.loc[grand["condition"].eq(cond)].sort_values("freq_hz")
        if len(sub) == 0:
            continue
        x = sub["freq_hz"].to_numpy()
        y = sub["mean_psd_v2_hz"].to_numpy()

        n = int(sub["n_units"].iloc[0]) # n = nombre de trials par condition
        ax.plot(x, y, label=f"{style['label']} (n={n})", color=style["color"]) 
        if show_sem and "sem_psd_v2_hz" in sub:
            sem = sub["sem_psd_v2_hz"].to_numpy()
            ax.fill_between(x, y - sem, y + sem, alpha=0.2, color=style["color"])

    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("PSD (V²/Hz)")
    ax.set_title(title)
    ax.set_xlim(cfg.fmin, cfg.fmax)
    ax.set_yscale(yscale)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if cfg.save_figures:
        ensure_dir(Path(cfg.output_dir) / "figures")
        fig.savefig(Path(cfg.output_dir) / "figures" / f"grand_average_psd__pre={cfg.pre_length}s_post={cfg.post_length}s.png", dpi=300)

    return fig


# =============================================================================
# FULL PIPELINE
# =============================================================================

def run_psd_pipeline(
    cfg: PSDConfig,
    sessions: Optional[Sequence[str]] = None,
    average_first: str = "trial_channel",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lance l'analyse PSD sur plusieurs sessions.

    Retour
    ------
    all_psd : pd.DataFrame
        Table longue complète.
    grand : pd.DataFrame
        Grand average par condition × fréquence.
    """

    cfg.root_dir = Path(cfg.root_dir)
    cfg.output_dir = Path(cfg.output_dir)
    ensure_dir(cfg.output_dir)

    if sessions is None:
        sessions = list_trc_sessions(cfg.root_dir)

    all_tables = []
    for session in sessions:
        try:
            tab = compute_session_psd(session, cfg)
            if len(tab) > 0:
                all_tables.append(tab)
        except Exception as exc:
            log(f"[ERROR] {session}: {type(exc).__name__}: {exc}", cfg.verbose)

    if not all_tables:
        return pd.DataFrame(), pd.DataFrame()

    all_psd = pd.concat(all_tables, ignore_index=True)
    grand = compute_grand_average(all_psd, average_first=average_first)

    if cfg.save_tables:
        all_psd.to_parquet(cfg.output_dir / f"all_sessions_psd_long__pre={float(cfg.pre_length)}s_post={float(cfg.post_length)}s.parquet", index=False)
        grand.to_csv(cfg.output_dir / f"grand_average_psd__pre={float(cfg.pre_length)}s_post={float(cfg.post_length)}s.csv", index=False)
    return all_psd, grand


__all__ = [
    "PSDConfig",
    "compute_epochs_psd",
    "psd_to_long_table",
    "compute_session_psd",
    "compute_grand_average",
    "plot_grand_average_psd",
    "run_psd_pipeline",
]
