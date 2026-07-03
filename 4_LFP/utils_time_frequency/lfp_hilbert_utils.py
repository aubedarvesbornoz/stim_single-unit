
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lfp_hilbert_utils.py
====================

Pipeline Hilbert pour compléter l'exploration Morlet.

Principe méthodologique
-----------------------
Ce module implémente une seconde étape d'analyse inspirée de Vidal et al. (2010),
adaptée au contexte de stimulations intracérébrales :

1. on utilise d'abord l'analyse Morlet exploratoire pour repérer, pour chaque type
   d'effet et/ou pour chaque région, les bandes fréquentielles d'intérêt/à cibler ;
2. on revient ensuite au signal temporel continu et on calcule une enveloppe
   d'amplitude par transformée de Hilbert dans des sous-bandes définies a priori ;
3. on normalise ces enveloppes au niveau de la session entière (ou d'un segment
   de référence choisi) pour obtenir une mesure plus robuste et plus simple à
   interpréter que la carte TF complète ;
4. on époche ensuite ces enveloppes autour des stimulations, puis on regroupe les
   observations par condition cognitive et par localité (local/distant).

Points d'adaptation à ce projet
-------------------------------
- les fenêtres temporelles sont définies autour de la fin de la stimulation selon
  la logique déjà utilisée pour Morlet (`post_start = stim_end + epsilon`) ;
- les groupes sont définis par `group_label`, `cog_labels`, et la localité du canal
  relativement au shaft stimulé (`stim_shaft`) ;
- le pooling across sessions est pris en charge côté stats.

Ce module contient :
- la configuration Hilbert ;
- la définition des banques de sous-bandes ;
- le calcul des enveloppes Hilbert sur signal continu ;
- la normalisation session-wise ;
- l'epoching des enveloppes ;
- l'agrégation par bande principale (theta, alpha, beta, low_gamma, high_gamma, etc.) ;
- les orchestrateurs de sauvegarde, de manière parallèle au pipeline Morlet.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

# import mne
import numpy as np
import pandas as pd
from scipy import signal
import os

from lfp_preprocess_utils import (
    log,
    ensure_dir,
    safe_name,
    normalize_channel_name,
    list_trc_sessions,
    read_stim_events,
    recover_precise_macro_stim_events,
    find_cog_file,
    # find_duration_file,
    read_cog_file,
    merge_event_tables,
    load_bad_channels_table,
    get_bad_channels_for_session,
    load_trc_as_mne_raw,
    apply_filters,
    build_adjacent_bipolar_pairs,
    make_bipolar_data,
    add_windows_to_trials,
    keep_trials_fitting_signal,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class HilbertConfig:
    """Configuration du pipeline Hilbert."""

    # Dossiers
    root_dir: str = "/home/aube/Documents/article_neuronal_stimic/effets_cog/theta_gamma_cog"
    output_dir: str = "/home/aube/Documents/article_neuronal_stimic/effets_cog/theta_gamma_cog/results_hilbert"

    # Fenêtres temporelles, cohérentes avec Morlet
    pre_length: float = 3.0
    post_length: float = 3.0
    epsilon: float = 0.1

    # Prétraitement continu avant Hilbert
    do_notch: bool = True
    notch_freqs: Tuple[float, ...] = (50.0, 100.0, 150.0)
    notch_q: float = 30.0
    do_highpass: bool = True
    highpass_hz: float = 0.1

    # Définition des sous-bandes Hilbert
    # gamma : sous-bandes non recouvrantes de 10 Hz ; alpha-beta : 4 Hz ; theta : 2 Hz ; delta optionnel : 1 Hz
    gamma_low_hz: float = 30.0
    gamma_high_hz: float = 150.0
    gamma_step_hz: float = 10.0

    beta_low_hz: float = 12.0
    beta_high_hz: float = 30.0
    beta_step_hz: float = 4.0

    alpha_low_hz: float = 8.0
    alpha_high_hz: float = 12.0
    alpha_step_hz: float = 2.0

    theta_low_hz: float = 4.0
    theta_high_hz: float = 8.0
    theta_step_hz: float = 2.0

    include_delta: bool = False
    delta_low_hz: float = 1.0
    delta_high_hz: float = 4.0
    delta_step_hz: float = 1.0

    # Agrégation des sous-bandes vers des bandes principales
    main_bands: Tuple[str, ...] = ("delta", "theta", "alpha", "beta", "low_gamma", "high_gamma")
    low_gamma_split: Tuple[float, float] = (30.0, 80.0)
    high_gamma_split: Tuple[float, float] = (80.0, 150.0)
    alpha_range: Tuple[float, float] = (8.0, 12.0)
    beta_range: Tuple[float, float] = (12.0, 30.0)
    theta_range: Tuple[float, float] = (4.0, 8.0)
    delta_range: Tuple[float, float] = (1.0, 4.0)

    # Normalisation de l'enveloppe
    normalization_mode: str = "pre_first_stim_mean"  # modif par rapport à Vidal 2010 qui prenait session entière
    eps: float = 1e-12

    # Décimation finale pour augmenter la puissance statistique, comme dans Vidal 2010
    target_dt_ms: float = 16.0  # ~ 16 ms / sample si possible

    # Sauvegardes
    save_subband_epochs: bool = False
    save_main_band_epochs: bool = True
    verbose: bool = True


# ============================================================================
# BANDES ET AFFECTATIONS
# ============================================================================

def build_non_overlapping_bands(f_low: float, f_high: float, step_hz: float) -> List[Tuple[float, float]]:
    """Construit des sous-bandes contiguës et non recouvrantes."""
    bands: List[Tuple[float, float]] = []
    start = float(f_low)
    while start < f_high:
        stop = min(start + step_hz, f_high)
        if stop > start:
            bands.append((float(start), float(stop)))
        start = stop
    return bands # par ex : (4.0, 8.0, 2.0) -> [(4.0, 6.0), (6.0, 8.0)]



def build_hilbert_subbands(cfg: HilbertConfig) -> Dict[str, List[Tuple[float, float]]]:
    """
    Construit les banques de sous-bandes par famille fréquentielle, sous forme de dict avec bande e key, et liste de sous-bandes en tuples.
    Par ex : {'theta': [('theta', 4.0, 6.0), ('theta', 6.0, 8.0)],
              'alpha': [('alpha', 8.0, 10.0), ('alpha', 10.0, 12.0)], ... }
    """
    out = {
        "theta": build_non_overlapping_bands(cfg.theta_low_hz, cfg.theta_high_hz, cfg.theta_step_hz), # (4.0, 8.0, 2.0) -> [(4.0, 6.0), (6.0, 8.0)]
        "alpha": build_non_overlapping_bands(cfg.alpha_low_hz, cfg.alpha_high_hz, cfg.alpha_step_hz),
        "beta": build_non_overlapping_bands(cfg.beta_low_hz, cfg.beta_high_hz, cfg.beta_step_hz),
        "gamma": build_non_overlapping_bands(cfg.gamma_low_hz, cfg.gamma_high_hz, cfg.gamma_step_hz),
    }
    if cfg.include_delta:
        out["delta"] = build_non_overlapping_bands(cfg.delta_low_hz, cfg.delta_high_hz, cfg.delta_step_hz)
    return out #



def flatten_subbands(subbands_by_family: Dict[str, List[Tuple[float, float]]]) -> List[Tuple[str, float, float]]:
    """
    Aplatissement sous forme [(family, f_low, f_high), ...].
        Par ex : {'theta': [('theta', 4.0, 6.0), ('theta', 6.0, 8.0)],
                'alpha': [('alpha', 8.0, 10.0), ('alpha', 10.0, 12.0)], ... } 
        devient : 
        [('theta', 4.0, 6.0), ('theta', 6.0, 8.0), ('alpha', 8.0, 10.0), ('alpha', 10.0, 12.0), ...]
    """
    flat: List[Tuple[str, float, float]] = []
    for family, bands in subbands_by_family.items():
        for f_low, f_high in bands:
            flat.append((family, float(f_low), float(f_high)))
    return flat



def format_subband_name(family: str, f_low: float, f_high: float) -> str:
    """Nom stable de sous-bande pour sauvegarde."""
    def _fmt(x: float) -> str:
        return str(int(x)) if float(x).is_integer() else f"{x:g}"
    return f"{family}_{_fmt(f_low)}_{_fmt(f_high)}Hz"



def assign_subband_to_main_band(f_low: float, f_high: float, cfg: HilbertConfig) -> Optional[str]:
    """Assigne une sous-bande à une bande principale selon sa fréquence centrale."""
    fc = (f_low + f_high) / 2.0
    if cfg.include_delta and cfg.delta_range[0] <= fc < cfg.delta_range[1]:
        return "delta"
    if cfg.theta_range[0] <= fc < cfg.theta_range[1]:
        return "theta"
    if cfg.alpha_range[0] <= fc < cfg.alpha_range[1]:
        return "alpha"
    if cfg.beta_range[0] <= fc <= cfg.beta_range[1]:
        return "beta"
    if cfg.low_gamma_split[0] <= fc < cfg.low_gamma_split[1]:
        return "low_gamma"
    if cfg.high_gamma_split[0] <= fc <= cfg.high_gamma_split[1]:
        return "high_gamma"
    return None



def build_main_band_to_subbands(cfg: HilbertConfig) -> Dict[str, List[Tuple[str, float, float]]]:
    """Construit l'affectation des sous-bandes vers les bandes principales."""
    flat = flatten_subbands(build_hilbert_subbands(cfg))
    out: Dict[str, List[Tuple[str, float, float]]] = {band: [] for band in cfg.main_bands}
    for family, f_low, f_high in flat: # pour chaque tuple de la liste des sous-bandes (avec chq élément composé d'un triplet)
        main_band = assign_subband_to_main_band(f_low, f_high, cfg) # permet de réassigner high_gamma a gamma par ex
        if main_band is not None and main_band in out:
            out[main_band].append((family, f_low, f_high))
    return {k: v for k, v in out.items() if len(v) > 0}


# ============================================================================
# NORMALISATION / HILBERT CONTINU
# ============================================================================

def compute_target_decim_factor(sfreq: float, target_dt_ms: float) -> int:
    """Calcule un facteur de décimation entier approchant le pas temporel voulu."""
    if target_dt_ms <= 0:
        return 1
    target_dt_s = target_dt_ms / 1000.0
    decim = int(round(target_dt_s * sfreq))
    return max(decim, 1)



def bandpass_and_hilbert(data: np.ndarray,
                         sfreq: float,
                         f_low: float,
                         f_high: float) -> np.ndarray:
    """Filtre continu dans une sous-bande puis extrait l'enveloppe par Hilbert."""
    if f_high >= sfreq / 2.0:
        raise ValueError(f"f_high={f_high} >= Nyquist={sfreq/2.0}")
    sos = signal.butter(N=4, Wn=[f_low, f_high], btype="bandpass", fs=sfreq, output="sos")
    x_filt = signal.sosfiltfilt(sos, data, axis=-1) # filtrage aller-retour -> phase nulle, représentation en second-order sections
    analytic = signal.hilbert(x_filt, axis=-1) 
    # hilbert() renvoie le signal analytique complexe : 
    # x_a(t) = x_sig_filtered(t) + i * x^(t), avec x^(t) la transformée de Hilbert. 
    # L'enveloppe instantanée correspond alors à |x^(t)|, d'où :
    envelope = np.abs(analytic) # extrait l'amplitude instantanée (enveloppe) du signal analytique complexe
    return envelope.astype(np.float32)



def normalize_envelope(envelope: np.ndarray,
                       mode: str = "whole_session_mean_percent",
                       eps: float = 1e-12,
                       reference_segment: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Normalise l'enveloppe canal par canal, depuis session entière ou référence 
    prise sur la période 0 -> première stim.

    Paramètres
    ----------
    envelope : np.ndarray
        shape = (n_channels, n_samples)
    mode : str
        - 'whole_session_mean'
        - 'whole_session_zscore'
        - 'reference_mean'
        - 'reference_zscore'
    reference_segment : np.ndarray | None
        Segment de référence shape = (n_channels, n_ref_samples), requis pour les modes 'reference_*'
    """
    if mode == "whole_session_mean":
        ref = np.mean(envelope, axis=-1, keepdims=True)
        return ((envelope / (ref + eps)) * 100.0).astype(np.float32)

    if mode == "whole_session_zscore":
        mu = np.mean(envelope, axis=-1, keepdims=True)
        sd = np.std(envelope, axis=-1, keepdims=True, ddof=0)
        return ((envelope - mu) / (sd + eps)).astype(np.float32)

    if mode == "reference_mean":
        if reference_segment is None:
            raise ValueError("reference_segment requis pour mode='reference_mean'")
        ref = np.mean(reference_segment, axis=-1, keepdims=True)
        return ((envelope / (ref + eps)) * 100.0).astype(np.float32)

    if mode == "reference_zscore":
        if reference_segment is None:
            raise ValueError("reference_segment requis pour mode='reference_zscore'")
        mu = np.mean(reference_segment, axis=-1, keepdims=True)
        sd = np.std(reference_segment, axis=-1, keepdims=True, ddof=0)
        return ((envelope - mu) / (sd + eps)).astype(np.float32)

    raise ValueError(f"normalization_mode inconnu: {mode}")


def build_pre_first_stim_reference_segment(data: np.ndarray,
                                           sfreq: float,
                                           first_stim_t_start: float) -> np.ndarray:
    """
    Segment de référence allant du début de la session à la première stimulation.
    """
    stop_idx = int(round(first_stim_t_start * sfreq))
    stop_idx = max(stop_idx, 1)
    return data[:, :stop_idx]


def compute_hilbert_subband_envelopes(data_bp: np.ndarray,
                                      sfreq: float,
                                      stims_df: pd.DataFrame,
                                      cfg: HilbertConfig) -> Dict[str, np.ndarray]:
    """
    Calcule les enveloppes Hilbert normalisées pour toutes les sous-bandes, 
    à partir de la baseline (0 -> première stim).
    """
    out: Dict[str, np.ndarray] = {}

    first_stim_t = float(stims_df["t_start"].min())
    reference_raw = build_pre_first_stim_reference_segment(data_bp, sfreq, first_stim_t)

    for family, f_low, f_high in flatten_subbands(build_hilbert_subbands(cfg)):
        name = format_subband_name(family, f_low, f_high)

        env = bandpass_and_hilbert(data_bp, sfreq, f_low, f_high)

        if cfg.normalization_mode == "pre_first_stim_mean":
            env_ref = bandpass_and_hilbert(reference_raw, sfreq, f_low, f_high)
            env_norm = normalize_envelope(
                env,
                mode="reference_mean",
                reference_segment=env_ref,
                eps=cfg.eps,
            )

        elif cfg.normalization_mode == "pre_first_stim_zscore":
            env_ref = bandpass_and_hilbert(reference_raw, sfreq, f_low, f_high)
            env_norm = normalize_envelope(
                env,
                mode="reference_zscore",
                reference_segment=env_ref,
                eps=cfg.eps,
            )

        else:
            env_norm = normalize_envelope(
                env,
                mode=cfg.normalization_mode,
                eps=cfg.eps,
                reference_segment=None,
            )

        out[name] = env_norm.astype(np.float32)

    return out


# ============================================================================
# EPOCHING DES ENVELOPPES HILBERT
# ============================================================================

def extract_epochs_from_continuous_feature(feature_data: np.ndarray,
                                           sfreq: float,
                                           stims_df: pd.DataFrame,
                                           pre_length: float,
                                           post_length: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Époque une caractéristique continue déjà calculée (enveloppe Hilbert, par exemple).

    On concatène explicitement :
    - la fenêtre pré  [pre_start, pre_end]
    - la fenêtre post [post_start, post_end]

    en excluant l'intervalle de stimulation et les marges epsilon entre les deux.

    Retour
    ------
    epochs : shape = (n_trials, n_channels, n_pre + n_post)
    times  : shape = (n_pre + n_post,)
    keep_indices : indices des essais effectivement extraits
    """
    n_pre = int(round(pre_length * sfreq))
    n_post = int(round(post_length * sfreq))

    epochs = []
    keep_idx = []

    for i, row in stims_df.iterrows():
        pre_start_idx = int(round(row["pre_start"] * sfreq))
        pre_end_idx = pre_start_idx + n_pre

        post_start_idx = int(round(row["post_start"] * sfreq))
        post_end_idx = post_start_idx + n_post

        pre_ep = feature_data[:, pre_start_idx:pre_end_idx]
        post_ep = feature_data[:, post_start_idx:post_end_idx]

        if pre_ep.shape[1] != n_pre or post_ep.shape[1] != n_post:
            continue

        epoch = np.concatenate([pre_ep, post_ep], axis=-1)  # shape = (n_channels, n_pre+n_post)
        epochs.append(epoch)
        keep_idx.append(int(i))

    if len(epochs) == 0:
        return np.empty((0, feature_data.shape[0], n_pre + n_post), dtype=np.float32), \
               np.r_[np.arange(n_pre) / sfreq - pre_length, np.arange(n_post) / sfreq].astype(np.float32), \
               np.asarray([], dtype=int)

    epochs_arr = np.asarray(epochs, dtype=np.float32)
    times = np.r_[np.arange(n_pre) / sfreq - pre_length, np.arange(n_post) / sfreq].astype(np.float32)

    return epochs_arr, times, np.asarray(keep_idx, dtype=int)


def decimate_times(times: np.ndarray, decim: int) -> np.ndarray:
    """
    Décime uniquement l'axe temporel associé aux epochs Hilbert.
    """
    if decim <= 1:
        return times.astype(np.float32)
    return times[::decim].astype(np.float32)


def decimate_feature_epochs(epochs: np.ndarray, decim: int) -> np.ndarray:
    """
    Décime uniquement les époques Hilbert.
    epochs shape = (n_trials, n_channels, n_times)
    """
    if decim <= 1:
        return epochs.astype(np.float32)
    return epochs[..., ::decim].astype(np.float32)


def aggregate_subbands_to_main_bands(subband_epochs: Dict[str, np.ndarray],
                                     cfg: HilbertConfig) -> Dict[str, np.ndarray]:
    """Moyenne les sous-bandes appartenant à une même bande principale."""
    assign = build_main_band_to_subbands(cfg)
    out: Dict[str, np.ndarray] = {}

    for main_band, defs in assign.items():
        arrs: List[np.ndarray] = []
        for family, f_low, f_high in defs:
            sub_name = format_subband_name(family, f_low, f_high)
            if sub_name in subband_epochs:
                arrs.append(subband_epochs[sub_name])
        if len(arrs) == 0:
            continue
        out[main_band] = np.mean(np.stack(arrs, axis=0), axis=0).astype(np.float32)

    return out


# ============================================================================
# SAUVEGARDE / CHARGEMENT
# ============================================================================

def save_hilbert_session_metadata(session_out: Path,
                                  session: str,
                                  cfg: HilbertConfig,
                                  stims_df: pd.DataFrame,
                                  raw_ch_names: Sequence[str],
                                  bad_channels: Sequence[str],
                                  bp_names: Sequence[str]) -> None:
    """Sauvegarde la trial table et les métadonnées de session Hilbert."""
    ensure_dir(session_out)
    stims_df.to_csv(session_out / f"{session}_trial_table.csv", index=False)
    meta = {
        "session": session,
        "config": asdict(cfg),
        "n_trials": int(len(stims_df)),
        "n_raw_channels": int(len(raw_ch_names)),
        "n_bad_channels": int(len(bad_channels)),
        "n_bipolar_channels": int(len(bp_names)),
        "raw_ch_names": list(raw_ch_names),
        "bad_channels": list(bad_channels),
        "bipolar_names": list(bp_names),
        "main_band_to_subbands": {
            k: [{"family": fam, "f_low": f_low, "f_high": f_high} for fam, f_low, f_high in v]
            for k, v in build_main_band_to_subbands(cfg).items()
        },
    }
    with open(session_out / f"{session}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)



def save_hilbert_epochs(session_out: Path,
                       session: str,
                       band_name: str,
                       epochs: np.ndarray) -> None:
    """Sauvegarde un tenseur d'époques Hilbert."""
    np.save(session_out / f"{session}_hilbert_{safe_name(band_name)}.npy", epochs.astype(np.float32))



def load_hilbert_session_exports(session_dir: Path, session: Optional[str] = None) -> Dict[str, Any]:
    """Recharge les exports de base d'une session Hilbert."""
    session_name = session or session_dir.name
    meta_file = session_dir / f"{session_name}_metadata.json"
    trials_file = session_dir / f"{session_name}_trial_table.csv"
    times_file = session_dir / f"{session_name}_times.npy"

    if not meta_file.exists():
        raise FileNotFoundError(meta_file)
    if not trials_file.exists():
        raise FileNotFoundError(trials_file)
    if not times_file.exists():
        raise FileNotFoundError(times_file)

    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "session": session_name,
        "metadata": metadata,
        "stims_df": pd.read_csv(trials_file),
        "times": np.load(times_file),
    }



def load_hilbert_band_epochs(session_dir: Path,
                             session: str,
                             band_name: str) -> np.ndarray:
    """Recharge les époques Hilbert d'une bande principale."""
    fp = session_dir / f"{session}_hilbert_{safe_name(band_name)}.npy"
    if not fp.exists():
        raise FileNotFoundError(fp)
    arr = np.load(fp)
    if arr.ndim != 3:
        raise ValueError(f"{fp.name}: shape inattendue {arr.shape}")
    return np.asarray(arr, dtype=np.float32)


# ============================================================================
# ORCHESTRATEURS
# ============================================================================

def prepare_hilbert_session_data(session: str,
                                 root_dir: Path,
                                 bad_df: pd.DataFrame,
                                 cfg: HilbertConfig) -> Dict[str, Any]:
    """Prépare les signaux bipolaires continus et la trial table pour une session."""
    verbose = cfg.verbose
    log(f"\n=== Préparation Hilbert session {session} ===", verbose)

    trc_path = root_dir / f"{session}.TRC"
    if not trc_path.exists():
        raise FileNotFoundError(f"TRC introuvable pour {session}: {trc_path}")

    cog_df = read_cog_file(find_cog_file(root_dir, session)) # annotations cognitives et lobe, ordre des stims considéré comme fiable
    trc_corr_path = root_dir / f"{session}_stim_events_TRC_corrected.txt"
    if os.path.exists(trc_corr_path):    
        trc_corr_df = read_stim_events(trc_corr_path)  # temps réels de début et durée des stims
    else: # n'existe pas encore, a creer
        trc_corr_df = recover_precise_macro_stim_events(session, root_dir)

    stims_df = merge_event_tables(session, cog_df, trc_corr_df)
    stims_df = add_windows_to_trials(stims_df, pre_length=cfg.pre_length, post_length=cfg.post_length, epsilon=cfg.epsilon)

    raw = load_trc_as_mne_raw(trc_path, verbose=verbose)
    sfreq = float(raw.info["sfreq"])
    raw_ch_names = [normalize_channel_name(ch) for ch in raw.ch_names]
    raw_data = raw.get_data()
    signal_duration_s = raw_data.shape[1] / sfreq

    stims_df = keep_trials_fitting_signal(stims_df, signal_duration_s, verbose=verbose)
    if len(stims_df) == 0:
        raise RuntimeError(f"{session}: aucun essai exploitable après contrôle des fenêtres")

    bad_channels = get_bad_channels_for_session(bad_df, session)
    bipolar_pairs = build_adjacent_bipolar_pairs(raw_ch_names, bad_channels)
    if len(bipolar_pairs) == 0:
        raise RuntimeError(f"{session}: aucune paire bipolaire construite")

    data_filt = apply_filters(
        raw_data,
        sfreq=sfreq,
        do_notch=cfg.do_notch,
        notch_freqs=cfg.notch_freqs,
        notch_q=cfg.notch_q,
        do_highpass=cfg.do_highpass,
        highpass_hz=cfg.highpass_hz,
    )
    data_bp, bp_names = make_bipolar_data(data_filt, raw_ch_names, bipolar_pairs)
    if data_bp.size == 0 or len(bp_names) == 0:
        raise RuntimeError(f"{session}: pas de données bipolaires exploitables")

    return {
        "session": session,
        "sfreq": sfreq,
        "stims_df": stims_df,
        "raw_ch_names": raw_ch_names,
        "bad_channels": bad_channels,
        "bp_names": bp_names,
        "data_bp": data_bp,
    }



def run_session_hilbert(session_data: Dict[str, Any],
                        out_dir: Path,
                        cfg: HilbertConfig) -> Path:
    """Calcule et sauvegarde les enveloppes Hilbert époquées pour une session."""
    verbose = cfg.verbose
    session = session_data["session"]
    sfreq = float(session_data["sfreq"])
    stims_df = session_data["stims_df"].copy()
    raw_ch_names = session_data["raw_ch_names"]
    bad_channels = session_data["bad_channels"]
    bp_names = session_data["bp_names"]
    data_bp = session_data["data_bp"]

    log(f"\n=== Hilbert session {session} ===", verbose)

    subband_cont = compute_hilbert_subband_envelopes(data_bp=data_bp, sfreq=sfreq, stims_df=stims_df, cfg=cfg)
    subband_epochs: Dict[str, np.ndarray] = {}
    keep_idx_ref = None
    times_ref = None

    for sub_name, env_cont in subband_cont.items():
        epochs, times, keep_idx = extract_epochs_from_continuous_feature(
            feature_data=env_cont,
            sfreq=sfreq,
            stims_df=stims_df,
            pre_length=cfg.pre_length,
            post_length=cfg.post_length,
        )
        if keep_idx_ref is None:
            keep_idx_ref = keep_idx
            times_ref = times
        else:
            # on s'aligne sur la même sélection d'essais pour toutes les sous-bandes
            if not np.array_equal(keep_idx_ref, keep_idx):
                raise ValueError(f"{session}: keep_idx différents entre sous-bandes Hilbert")
        subband_epochs[sub_name] = epochs

    if keep_idx_ref is None or times_ref is None or len(keep_idx_ref) == 0:
        raise RuntimeError(f"{session}: aucun epoch Hilbert extrait")

    stims_df = stims_df.iloc[keep_idx_ref].reset_index(drop=True)

    # décimation optionnelle vers un pas proche de 16 ms
    decim = compute_target_decim_factor(sfreq=sfreq, target_dt_ms=cfg.target_dt_ms)
    log(f"[INFO] {session}: facteur de décimation Hilbert = {decim}", verbose)

    times_ref = decimate_times(times_ref, decim)

    for sub_name in list(subband_epochs.keys()):
        subband_epochs[sub_name] = decimate_feature_epochs(subband_epochs[sub_name], decim)
    
    main_band_epochs = aggregate_subbands_to_main_bands(subband_epochs=subband_epochs, cfg=cfg)

    session_out = out_dir / session
    ensure_dir(session_out)
    np.save(session_out / f"{session}_times.npy", times_ref.astype(np.float32))
    save_hilbert_session_metadata(session_out, session, cfg, stims_df, raw_ch_names, bad_channels, bp_names)

    if cfg.save_subband_epochs:
        for sub_name, arr in subband_epochs.items():
            save_hilbert_epochs(session_out, session, f"subband_{sub_name}", arr)

    if cfg.save_main_band_epochs:
        for band_name, arr in main_band_epochs.items():
            save_hilbert_epochs(session_out, session, band_name, arr)

    log(f"[OK] {session}: résultats Hilbert sauvegardés dans {session_out}", verbose)
    return session_out



def run_all_sessions_hilbert(cfg: HilbertConfig) -> Dict[str, Any]:
    """Lance le pipeline Hilbert sur toutes les sessions disponibles."""
    root_dir = Path(cfg.root_dir)
    out_dir = Path(cfg.output_dir)
    ensure_dir(out_dir)

    bad_df = load_bad_channels_table(root_dir)
    sessions = list_trc_sessions(root_dir)
    if len(sessions) == 0:
        raise RuntimeError(f"Aucun fichier TRC trouvé dans {root_dir}")

    errors: List[Tuple[str, str]] = []
    log(f"{len(sessions)} sessions TRC trouvées pour Hilbert", cfg.verbose)

    for session in sessions:
        try:
            session_data = prepare_hilbert_session_data(session=session, root_dir=root_dir, bad_df=bad_df, cfg=cfg)
            run_session_hilbert(session_data=session_data, out_dir=out_dir, cfg=cfg)
        except Exception as exc:
            errors.append((session, repr(exc)))
            log(f"[ERROR] Hilbert {session}: {exc}", cfg.verbose)

    summary = {"config": asdict(cfg), "n_sessions": len(sessions), "n_errors": len(errors), "errors": errors}
    with open(out_dir / "run_summary_hilbert.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


__all__ = [
    "HilbertConfig",
    "build_non_overlapping_bands",
    "build_hilbert_subbands",
    "flatten_subbands",
    "format_subband_name",
    "assign_subband_to_main_band",
    "build_main_band_to_subbands",
    "compute_target_decim_factor",
    "bandpass_and_hilbert",
    "normalize_envelope",
    "build_pre_first_stim_reference_segment",
    "compute_hilbert_subband_envelopes",
    "extract_epochs_from_continuous_feature",
    "decimate_times",
    "decimate_feature_epochs", 
    "aggregate_subbands_to_main_bands",
    "save_hilbert_session_metadata",
    "save_hilbert_epochs",
    "load_hilbert_session_exports",
    "load_hilbert_band_epochs",
    "prepare_hilbert_session_data",
    "run_session_hilbert",
    "run_all_sessions_hilbert",
]
