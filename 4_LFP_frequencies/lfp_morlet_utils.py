#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lfp_utils.py
============

Module utilitaire pour l'analyse exploratoire et inférentielle des dynamiques
LFP post-stimulation.

Ce module regroupe, dans un seul fichier réutilisable depuis un notebook :
- la lecture et la fusion des tables d'événements,
- la gestion des canaux invalides et du montage bipolaire adjacent,
- la lecture de TRC Micromed,
- le prétraitement du signal (filtres, montage) et l'extraction des epochs pré/post,
- le calcul temps-fréquence par ondelettes de Morlet,
- la normalisation à la baseline selon plusieurs métriques,
- la sauvegarde des variables intermédiaires par canal (en .npy),
- la génération de figures essai par essai (temps-fréquences brut/normalisé par canal et stim),
- les regroupements d'essais par condition cognitive et topographique,
- les statistiques par condition sur cartes TF déjà exportées,
- les orchestrateurs session par session ou sur l'ensemble des sessions.

Critères de regroupement des essais en conditions :
- regroupement des stims selon catégories cog (avec/sans sous-types)
- regroupement des canaux d'une session selon distance à la stim : local (= meme électrode que stim) VS distant (= électrode differente)

Le module est conçu pour être piloté depuis un ou plusieurs notebooks, avec des
appels séparés pour :
1) la préparation et le calcul Morlet,
2) la génération de figures exploratoires,
3) les statistiques par condition sur les cartes déjà exportées.

Auteur : Aube Darves-Bornoz
"""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from lfp_preprocess import (
    log,
    ensure_dir,
    safe_name,
    make_freqs,
    reduce_baseline_stat,
    normalize_channel_name,
    build_adjacent_bipolar_pairs,
    find_cog_file,
    find_duration_file,
    read_cog_file,
    read_duration_file,
    merge_event_tables,
    get_bad_channels_for_session,
    load_bad_channels_table,
    list_trc_sessions,
    load_trc_as_mne_raw,
    apply_filters,
    make_bipolar_data,
    add_windows_to_trials,
    keep_trials_fitting_signal,
    extract_pre_post_epochs,
    build_global_baseline_segment,
    group_bipolar_channels_by_shaft
)

# ============================================================================
# CONFIGURATION OBJECT
# ============================================================================

@dataclass
class MorletConfig:
    """
    Configuration du pipeline exploratoire par Morlet.

    Cette configuration rassemble à la fois :
    - les chemins d'entrée/sortie,
    - les paramètres d'extraction des fenêtres/epochs,
    - les paramètres de filtrage,
    - les paramètres temps-fréquence,
    - les options de baseline,
    - les options de sauvegarde et de visualisation.

    Notes méthodologiques
    ---------------------
    - `pre_length` et `post_length` définissent la durée des fenêtres utilisées
      pour baseline et réponse post-stimulation.
    - `epsilon` définit une marge de sécurité avant le début et après la fin de
      la stimulation afin d'éviter de capturer l'artefact immédiat.
    - `baseline_mode='trial_pre'` :mode recommandé pour comparer chaque
      réponse post-stim à sa propre baseline pré-stim, mais également possible
      de choisir 'global_pre_first_stim' soit toute la période de 0 à la première stim.
    """

    # Dossiers principaux
    root_dir: str = "/home/aube/Documents/article_neuronal_stimic/effets_cog/theta_gamma_cog" # où trouver les TRC et les fichiers d'events ; et où exporter les variables/figures
    output_dir: str = root_dir + "/results_morlet_exploratoire"

    # Epoching autour de la stim
    pre_length: float = 3.0     # fenêtre pré-stim = [t_start - pre_length - epsilon ; t_start - epsilon]
    post_length: float = 3.0    # fenêtre post-stim = [t_end + epsilon ; t_end + post_length + epsilon]
    epsilon: float = 0.01        # (en sec) marge avant début stim et après fin stim, pour éviter artéfact de filtrage sur la saturation electrique de la stim
    allow_global_baseline: bool = True     # # option pour autoriser la baseline globale [0, première stim] (n'engage à rien, choisi après)

    # Filtrage du signal avant bipolarisation
    do_notch: bool = True
    notch_freqs: Tuple[float, ...] = (50.0, 100.0, 150.0)
    notch_q: float = 30.0 # facteur Q du notch : determine finesse de la coupure, ici haute finesse
    do_highpass: bool = True
    highpass_hz: float = 0.1

    # Paramètres Morlet
    fmin: float = 4.0           # descendre à 2 est ok seulement si duration post est plus longue que 3, car sinon l'ondelette est très longue par rapport a l'epoch 
    fmax: float = 150.0
    n_freqs: int = 20           #80 #(plante avec 149)
    freq_scale: str = "linear"  # "linear" | "log" | "semilog"
    n_cycles: float = 7.0       # nombre de repetitions de cycles pour chaque frequence => l'info correspondant à 4Hz en chq point est alors basée sur 1.75 s ; l'info correspondant à 80Hz en chq point est basée sur 87.5ms
    decim: int = 16             # décimation après TFR si besoin : facteur de réduction/division du power pour simplification computationnelle
    n_jobs: int = 1

    # Baseline et métriques
    baseline_mode: str = "trial_pre"  # "trial_pre" | "global_pre_first_stim"
    baseline_stat: str = "median"     # "median" | "mean". Ce qu'on utilise pour calculer P_pre(f) pour chaque essai
    metrics_to_compute: Tuple[str, ...] = ("logratio", "percent", "zscore", "subtract")
    eps: float = 1e-12

    # Sauvegardes complémentaires
    save_raw_epochs: bool = False
    save_filtered_signal_preview: bool = False

    # Figures exploratoires essai par essai
    make_figures: bool = True
    figure_dpi: int = 150
    max_cols_per_figure: int = 3
    cmap_raw: str = "viridis"         # colormap pour cartes TF des puissances brutes (par essai)
    cmap_metric_div: str = "RdBu_r"   # colormap pour cartes TF des puissances normalisées (par essai)
    raw_display_mode: str = "log10"   # "raw" | "log10"
    z_threshold: float = 3.0          # seuillage par | Z-scored P_post(t,f) | > 3 pour identifier des zones TF potentiellement modulées (grossier, appliqué essai par essai)
    lineplot_time_stat: str = "mean"  # "mean" | "median"
    save_png: bool = True

    # Divers
    verbose: bool = True


# ============================================================================
# MORLET AND BASELINE NORMALIZATION
# ============================================================================

def compute_morlet_power(
    epochs: np.ndarray,
    sfreq: float,
    freqs: np.ndarray,
    n_cycles: float | np.ndarray,
    decim: int = 1,
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Calcule la puissance temps-fréquence Morlet à partir d'époques.

    Paramètres
    ----------
    epochs : np.ndarray
        Array de forme (n_epochs, n_channels, n_times).
    sfreq : float
        Fréquence d'échantillonnage en Hz.
    freqs : np.ndarray
        Fréquences analysées.
    n_cycles : float | np.ndarray
        Nombre de cycles par fréquence.
    decim : int
        Facteur de décimation sur l'axe temps après TFR.
    n_jobs : int
        Nombre de jobs parallèles pour MNE.

    Retour
    ------
    np.ndarray
        Array de forme (n_epochs, n_channels, n_freqs, n_times_decim).
    """
    power = mne.time_frequency.tfr_array_morlet(
        data=epochs,
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=n_cycles,
        output="power",
        decim=decim,
        n_jobs=n_jobs,
        zero_mean=True,
    )
    return np.asarray(power, dtype=np.float32)



def compute_baseline_reference(
    power_pre: Optional[np.ndarray] = None,
    power_global: Optional[np.ndarray] = None,
    mode: str = "trial_pre",
    stat: str = "median",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Calcule la référence de baseline utilisée pour normaliser la puissance post-stimulation.

    Paramètres
    ----------
    power_pre : Optional[np.ndarray]
        Puissance pré-stimulation, typiquement de forme
        (n_trials, n_channels, n_freqs, n_pre_times) si mode='trial_pre'.
    power_global : Optional[np.ndarray]
        Puissance de baseline globale, typiquement de forme
        (n_channels, n_freqs, n_base_times) si mode='global_pre_first_stim'.
    mode : str
        'trial_pre' ou 'global_pre_first_stim'.
    stat : str
        'median' ou 'mean'.

    Retour
    ------
    Tuple[np.ndarray, Optional[np.ndarray]]
        - baseline_ref : array de référence (sans axe temps)
        - power_for_z  : baseline temporelle à réutiliser pour le z-score, si pertinent

    Notes
    -----
    - en mode `trial_pre`, la baseline est propre à chaque essai ; c'est le mode
      le plus naturel pour comparer un post à son pré immédiat.
    - en mode `global_pre_first_stim`, la baseline est commune à tous les essais
      d'une session, calculée sur le segment avant la première stimulation.
    """
    if mode == "trial_pre":
        if power_pre is None:
            raise ValueError("power_pre est requis si mode='trial_pre'")
        baseline_ref = reduce_baseline_stat(power_pre, stat=stat)
        return baseline_ref.astype(np.float32), power_pre

    if mode == "global_pre_first_stim":
        if power_global is None:
            raise ValueError("power_global est requis si mode='global_pre_first_stim'")
        baseline_ref = reduce_baseline_stat(power_global, stat=stat)
        return baseline_ref.astype(np.float32), power_global

    raise ValueError(f"baseline_mode inconnu: {mode}")



def compute_metric(
    power_post: np.ndarray,
    baseline_ref: np.ndarray,
    metric: str,
    power_pre_for_z: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Calcule une métrique de modulation post-stimulation relativement à une baseline.

    Paramètres
    ----------
    power_post : np.ndarray
        Puissance post-stimulation, de forme (..., n_post_times).
    baseline_ref : np.ndarray
        Référence de baseline sans axe temps, de forme (...) ou broadcastable.
    metric : str
        'logratio', 'percent', 'subtract' ou 'zscore'.
    power_pre_for_z : Optional[np.ndarray]
        Baseline temporelle complète, nécessaire uniquement pour le z-score.
    eps : float
        Petite constante pour éviter les divisions par zéro.

    Retour
    ------
    np.ndarray
        Array de même forme que `power_post`.
    """
    b = baseline_ref[..., None] if baseline_ref.ndim == power_post.ndim - 1 else baseline_ref

    if metric == "logratio":
        return np.log((power_post + eps) / (b + eps)).astype(np.float32)

    if metric == "percent":
        return (((power_post - b) / (b + eps)) * 100.0).astype(np.float32)

    if metric == "subtract":
        return (power_post - b).astype(np.float32)

    if metric == "zscore":
        if power_pre_for_z is None:
            raise ValueError("power_pre_for_z est requis pour metric='zscore'")
        mu = np.mean(power_pre_for_z, axis=-1, keepdims=True)
        sd = np.std(power_pre_for_z, axis=-1, ddof=0, keepdims=True)
        return ((power_post - mu) / (sd + eps)).astype(np.float32)

    raise ValueError(f"metric inconnue: {metric}")



def compute_metrics_for_channel(
    power_pre_ch: np.ndarray,
    power_post_ch: np.ndarray,
    metrics: Sequence[str],
    baseline_mode: str,
    baseline_stat: str,
    eps: float,
    power_global_base_ch: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Calcule toutes les métriques demandées pour un canal donné.

    Paramètres
    ----------
    power_pre_ch : np.ndarray
        Puissance pré de forme (n_trials, 1, n_freqs, n_pre_times).
    power_post_ch : np.ndarray
        Puissance post de forme (n_trials, 1, n_freqs, n_post_times).
    metrics : Sequence[str]
        Liste des métriques à calculer.
    baseline_mode : str
        'trial_pre' ou 'global_pre_first_stim'.
    baseline_stat : str
        'median' ou 'mean'.
    eps : float
        Constante numérique de sécurité.
    power_global_base_ch : Optional[np.ndarray]
        Puissance de baseline globale pour le canal, de forme (1, n_freqs, n_base_times),
        requise si baseline_mode='global_pre_first_stim'.

    Retour
    ------
    Tuple[Dict[str, np.ndarray], np.ndarray]
        - dictionnaire metric -> array (n_trials, 1, n_freqs, n_post_times)
        - baseline_values de forme (n_trials, 1, n_freqs) en mode trial_pre,
          ou (1, n_freqs) en mode global
    """
    if baseline_mode == "trial_pre":
        baseline_ref, power_for_z = compute_baseline_reference(
            power_pre=power_pre_ch,
            mode=baseline_mode,
            stat=baseline_stat,
        )
        metrics_dict = {
            metric: compute_metric(
                power_post=power_post_ch,
                baseline_ref=baseline_ref,
                metric=metric,
                power_pre_for_z=power_for_z if metric == "zscore" else None,
                eps=eps,
            )
            for metric in metrics
        }
        return metrics_dict, baseline_ref.astype(np.float32)

    if baseline_mode == "global_pre_first_stim":
        if power_global_base_ch is None:
            raise ValueError("power_global_base_ch est requis si baseline_mode='global_pre_first_stim'")
        baseline_ref, power_for_z = compute_baseline_reference(
            power_global=power_global_base_ch,
            mode=baseline_mode,
            stat=baseline_stat,
        )
        metrics_dict = {
            metric: compute_metric(
                power_post=power_post_ch,
                baseline_ref=baseline_ref[None, ...],  # ajout axe essai pour broadcasting sur n_trials
                metric=metric,
                power_pre_for_z=power_for_z[None, ...] if metric == "zscore" else None,
                eps=eps,
            )
            for metric in metrics
        }
        return metrics_dict, baseline_ref.astype(np.float32)

    raise ValueError(f"baseline_mode inconnu: {baseline_mode}")


# ============================================================================
# SESSION EXPORT HELPERS
# ============================================================================

def channel_file(session_dir: Path, session: str, kind: str, ch_name: str, ext: str = ".npy") -> Path:
    """
    Construit le chemin standardisé d'un fichier par canal.

    Paramètres
    ----------
    session_dir : Path
        Dossier de la session.
    session : str
        Nom de session.
    kind : str
        Type de contenu, par exemple 'power_pre', 'power_post', 'logratio'.
    ch_name : str
        Nom du canal bipolaire.
    ext : str
        Extension du fichier, '.npy' par défaut.

    Retour
    ------
    Path
        Chemin complet vers le fichier.
    """
    return session_dir / f"{session}_{kind}_{safe_name(ch_name)}{ext}"



def save_json(obj: Dict[str, Any], filepath: Path) -> None:
    """
    Sauvegarde un dictionnaire Python au format JSON UTF-8 indenté.

    Paramètres
    ----------
    obj : Dict[str, Any]
        Objet à sérialiser.
    filepath : Path
        Fichier de destination.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)



def save_session_metadata(
    session_out: Path,
    session: str,
    config: MorletConfig,
    stims_df: pd.DataFrame,
    raw_ch_names: Sequence[str],
    bad_channels: Sequence[str],
    bp_names: Sequence[str],
) -> None:
    """
    Sauvegarde la table d'essais et les métadonnées descriptives d'une session Morlet.

    Paramètres
    ----------
    session_out : Path
        Dossier de sortie de la session.
    session : str
        Nom de session.
    config : MorletConfig
        Configuration utilisée.
    stims_df : pd.DataFrame
        Table finale d'essais.
    raw_ch_names : Sequence[str]
        Canaux monopolaires lus dans le TRC.
    bad_channels : Sequence[str]
        Canaux invalides exclus.
    bp_names : Sequence[str]
        Canaux bipolaires effectivement construits.
    """
    ensure_dir(session_out)
    stims_df.to_csv(session_out / f"{session}_stims_table.csv", index=False)

    meta = {
        "session": session,
        "config": asdict(config),
        "n_trials": int(len(stims_df)),
        "n_raw_channels": int(len(raw_ch_names)),
        "n_bad_channels": int(len(bad_channels)),
        "n_bipolar_channels": int(len(bp_names)),
        "raw_ch_names": list(raw_ch_names),
        "bad_channels": list(bad_channels),
        "bipolar_names": list(bp_names),
    }
    save_json(meta, session_out / f"{session}_metadata.json")



def save_channel_morlet_outputs(
    session_out: Path,
    session: str,
    ch_name: str,
    power_pre_ch: np.ndarray,
    power_post_ch: np.ndarray,
    baseline_values_ch: np.ndarray,
    metrics_dict_ch: Dict[str, np.ndarray],
) -> None:
    """
    Sauvegarde les sorties Morlet d'un canal donné.

    Paramètres
    ----------
    session_out : Path
        Dossier de sortie de la session.
    session : str
        Nom de session.
    ch_name : str
        Nom du canal bipolaire.
    power_pre_ch : np.ndarray
        Puissance pré de forme (n_trials, n_freqs, n_times_pre).
    power_post_ch : np.ndarray
        Puissance post de forme (n_trials, n_freqs, n_times_post).
    baseline_values_ch : np.ndarray
        Valeurs de baseline réduites, soit (n_trials, n_freqs) soit (n_freqs,).
    metrics_dict_ch : Dict[str, np.ndarray]
        Dictionnaire des métriques exportables, chaque array ayant forme
        (n_trials, n_freqs, n_times_post).
    """
    np.save(channel_file(session_out, session, "power_pre", ch_name), power_pre_ch)
    np.save(channel_file(session_out, session, "power_post", ch_name), power_post_ch)
    np.save(channel_file(session_out, session, "baseline_values", ch_name), baseline_values_ch)

    for metric, arr in metrics_dict_ch.items():
        np.save(channel_file(session_out, session, metric, ch_name), arr)


# ============================================================================
# MORLET WORKFLOW
# ============================================================================

def prepare_session_data(
    session: str,
    root_dir: Path,
    bad_df: pd.DataFrame,
    cfg: MorletConfig,
) -> Dict[str, Any]:
    """
    Prépare toutes les données nécessaires au calcul Morlet d'une session.

    Paramètres
    ----------
    session : str
        Nom de session.
    root_dir : Path
        Dossier racine contenant TRC et tables d'événements.
    bad_df : pd.DataFrame
        Table globale des canaux invalides.
    cfg : MorletConfig
        Configuration Morlet.

    Retour
    ------
    Dict[str, Any]
        Dictionnaire structuré contenant :
        {
            'session', 'sfreq', 'stims_df', 'raw_ch_names', 'bad_channels',
            'bp_names', 'bipolar_pairs', 'data_bp', 'pre_epochs', 'post_epochs',
            'pre_times', 'post_times'
        }

    Notes
    -----
    Cette fonction centralise toute la préparation commune aux calculs Morlet ; elle
    peut être appelée seule dans un notebook pour inspecter une session avant tout
    calcul lourd.
    """
    verbose = cfg.verbose
    log(f"\n=== Préparation session {session} ===", verbose)

    trc_path = root_dir / f"{session}.TRC"
    if not trc_path.exists():
        raise FileNotFoundError(f"TRC introuvable pour {session}: {trc_path}")

    cog_file = find_cog_file(root_dir, session)
    dur_file = find_duration_file(root_dir, session)

    cog_df = read_cog_file(cog_file)  # annotations cognitives et lobe, ordre des stims considéré comme fiable
    dur_df = read_duration_file(dur_file)  # temps réels de début et durée des stims
    stims_df = merge_event_tables(session, cog_df, dur_df)
    stims_df = add_windows_to_trials(
        stims_df,
        pre_length=cfg.pre_length,
        post_length=cfg.post_length,
        epsilon=cfg.epsilon,
    )

    raw = load_trc_as_mne_raw(trc_path, verbose=verbose)
    sfreq = float(raw.info["sfreq"])
    raw_ch_names = [normalize_channel_name(ch) for ch in raw.ch_names]
    raw_data = raw.get_data()  # forme : (n_channels, n_samples)
    signal_duration_s = raw_data.shape[1] / sfreq

    stims_df = keep_trials_fitting_signal(stims_df, signal_duration_s, verbose=verbose)
    if len(stims_df) == 0:
        raise RuntimeError(f"{session}: aucun essai exploitable après contrôle des fenêtres")

    bad_channels = get_bad_channels_for_session(bad_df, session)
    bipolar_pairs = build_adjacent_bipolar_pairs(raw_ch_names, bad_channels)  # liste (nom_bp, ch1, ch2)
    if len(bipolar_pairs) == 0:
        raise RuntimeError(f"{session}: aucune paire bipolaire construite")

    filtered_data = apply_filters(
        raw_data,
        sfreq=sfreq,
        do_notch=cfg.do_notch,
        notch_freqs=cfg.notch_freqs,
        notch_q=cfg.notch_q,
        do_highpass=cfg.do_highpass,
        highpass_hz=cfg.highpass_hz,
    )

    data_bp, bp_names = make_bipolar_data(filtered_data, raw_ch_names, bipolar_pairs)
    if data_bp.size == 0 or len(bp_names) == 0:
        raise RuntimeError(f"{session}: pas de données bipolaires exploitables")

    pre_epochs, post_epochs, pre_times, post_times = extract_pre_post_epochs(
        data_bp=data_bp,
        sfreq=sfreq,
        stims_df=stims_df,
        pre_length=cfg.pre_length,
        post_length=cfg.post_length,
    )

    if pre_epochs.shape[0] == 0 or post_epochs.shape[0] == 0:
        raise RuntimeError(f"{session}: aucun epoch extrait")

    n_ok = min(pre_epochs.shape[0], post_epochs.shape[0], len(stims_df))  # réalignement prudent si un essai a sauté à l'extraction
    pre_epochs = pre_epochs[:n_ok]
    post_epochs = post_epochs[:n_ok]
    stims_df = stims_df.iloc[:n_ok].reset_index(drop=True)

    return {
        "session": session,
        "sfreq": sfreq,
        "stims_df": stims_df,
        "raw_ch_names": raw_ch_names,
        "bad_channels": bad_channels,
        "bipolar_pairs": bipolar_pairs,
        "bp_names": bp_names,
        "data_bp": data_bp,
        "pre_epochs": pre_epochs,
        "post_epochs": post_epochs,
        "pre_times": pre_times,
        "post_times": post_times,
    }



def run_session_morlet(
    session_data: Dict[str, Any],
    out_dir: Path,
    cfg: MorletConfig,
) -> Path:
    """
    Exécute le calcul Morlet et la normalisation baseline pour une session préparée.

    Paramètres
    ----------
    session_data : Dict[str, Any]
        Sortie de `prepare_session_data`.
    out_dir : Path
        Dossier racine de sortie des exports Morlet.
    cfg : MorletConfig
        Configuration du pipeline.

    Retour
    ------
    Path
        Dossier de sortie de la session.

    Notes
    -----
    Le calcul est fait canal par canal pour limiter la charge mémoire ; c'est plus
    lent qu'un calcul massif sur tous les canaux, mais plus robuste pour des sessions
    longues et des configurations riches en fréquences/essais.
    """
    verbose = cfg.verbose
    session = session_data["session"]
    log(f"\n=== Morlet session {session} ===", verbose)

    sfreq = float(session_data["sfreq"])
    stims_df = session_data["stims_df"].copy()
    raw_ch_names = session_data["raw_ch_names"]
    bad_channels = session_data["bad_channels"]
    bp_names = session_data["bp_names"]
    data_bp = session_data["data_bp"]
    pre_epochs = session_data["pre_epochs"]
    post_epochs = session_data["post_epochs"]
    pre_times = session_data["pre_times"]
    post_times = session_data["post_times"]

    freqs = make_freqs(cfg.fmin, cfg.fmax, cfg.n_freqs, cfg.freq_scale)
    pre_times_decim = pre_times[::cfg.decim]
    post_times_decim = post_times[::cfg.decim]

    session_out = out_dir / session
    ensure_dir(session_out)

    np.save(session_out / f"{session}_freqs.npy", freqs)
    np.save(session_out / f"{session}_pre_times.npy", pre_times_decim)
    np.save(session_out / f"{session}_post_times.npy", post_times_decim)
    save_session_metadata(
        session_out=session_out,
        session=session,
        config=cfg,
        stims_df=stims_df,
        raw_ch_names=raw_ch_names,
        bad_channels=bad_channels,
        bp_names=bp_names,
    )

    base_seg = None
    if cfg.baseline_mode == "global_pre_first_stim":
        if not cfg.allow_global_baseline:
            raise ValueError("baseline_mode='global_pre_first_stim' mais allow_global_baseline=False")
        first_stim_t = float(stims_df["t_start"].min())
        base_seg, _ = build_global_baseline_segment(data_bp, sfreq, first_stim_t)

    for ch_idx, ch_name in enumerate(bp_names):
        log(f"[INFO] {session}: Morlet canal {ch_idx + 1}/{len(bp_names)} -> {ch_name}", verbose)

        pre_ep_ch = pre_epochs[:, ch_idx:ch_idx + 1, :]   # forme : (n_trials, 1, n_pre_samples)
        post_ep_ch = post_epochs[:, ch_idx:ch_idx + 1, :] # forme : (n_trials, 1, n_post_samples)
        log(f"[INFO] {session} {ch_name}: pre_ep_ch shape = {pre_ep_ch.shape}", verbose)
        log(f"[INFO] {session} {ch_name}: post_ep_ch shape = {post_ep_ch.shape}", verbose)

        power_pre_ch = compute_morlet_power(
            epochs=pre_ep_ch,
            sfreq=sfreq,
            freqs=freqs,
            n_cycles=cfg.n_cycles,
            decim=cfg.decim,
            n_jobs=cfg.n_jobs,
        )
        power_post_ch = compute_morlet_power(
            epochs=post_ep_ch,
            sfreq=sfreq,
            freqs=freqs,
            n_cycles=cfg.n_cycles,
            decim=cfg.decim,
            n_jobs=cfg.n_jobs,
        )
        log(f"[INFO] {session} {ch_name}: power_pre shape = {power_pre_ch.shape}", verbose)
        log(f"[INFO] {session} {ch_name}: power_post shape = {power_post_ch.shape}", verbose)

        power_global_base_ch = None
        if cfg.baseline_mode == "global_pre_first_stim":
            base_seg_ch = base_seg[ch_idx:ch_idx + 1, :]  # un seul canal bipolaire à la fois
            power_global_base_ch = compute_morlet_power(
                epochs=base_seg_ch[None, :, :],
                sfreq=sfreq,
                freqs=freqs,
                n_cycles=cfg.n_cycles,
                decim=cfg.decim,
                n_jobs=cfg.n_jobs,
            )[0]
            log(f"[INFO] {session} {ch_name}: power_global_base shape = {power_global_base_ch.shape}", verbose)

        metrics_dict_ch, baseline_values_ch = compute_metrics_for_channel(
            power_pre_ch=power_pre_ch,
            power_post_ch=power_post_ch,
            metrics=cfg.metrics_to_compute,
            baseline_mode=cfg.baseline_mode,
            baseline_stat=cfg.baseline_stat,
            eps=cfg.eps,
            power_global_base_ch=power_global_base_ch,
        )

        power_pre_ch_sq = power_pre_ch[:, 0, :, :]  # squeeze axe canal singleton -> (n_trials, n_freqs, n_times)
        power_post_ch_sq = power_post_ch[:, 0, :, :]
        metrics_dict_ch_sq = {metric: arr[:, 0, :, :] for metric, arr in metrics_dict_ch.items()}

        if baseline_values_ch.ndim == 3:
            baseline_values_ch_sq = baseline_values_ch[:, 0, :]  # mode trial_pre -> (n_trials, n_freqs)
        elif baseline_values_ch.ndim == 2:
            baseline_values_ch_sq = baseline_values_ch            # mode global -> (n_freqs,)
        else:
            raise ValueError(f"baseline_values_ch shape inattendue: {baseline_values_ch.shape}")

        save_channel_morlet_outputs(
            session_out=session_out,
            session=session,
            ch_name=ch_name,
            power_pre_ch=power_pre_ch_sq,
            power_post_ch=power_post_ch_sq,
            baseline_values_ch=baseline_values_ch_sq,
            metrics_dict_ch=metrics_dict_ch_sq,
        )
        log(f"[INFO] {session} {ch_name}: sauvegarde canal OK", verbose)

        del pre_ep_ch, post_ep_ch, power_pre_ch, power_post_ch, metrics_dict_ch  # libère les objets transitoires les plus volumineux

    if cfg.make_figures:
        generate_session_figures(session_out=session_out, cfg=cfg)

    log(f"[OK] {session}: résultats Morlet sauvegardés dans {session_out}", verbose)
    return session_out



def run_all_sessions_morlet(cfg: MorletConfig) -> Dict[str, Any]:
    """
    Exécute la préparation et le calcul Morlet sur toutes les sessions TRC trouvées.

    Paramètres
    ----------
    cfg : MorletConfig
        Configuration Morlet.

    Retour
    ------
    Dict[str, Any]
        Résumé d'exécution contenant le nombre de sessions et les erreurs éventuelles.
    """
    root_dir = Path(cfg.root_dir)
    out_dir = Path(cfg.output_dir)
    ensure_dir(out_dir)

    bad_df = load_bad_channels_table(root_dir)
    sessions = list_trc_sessions(root_dir)
    if len(sessions) == 0:
        raise RuntimeError(f"Aucun fichier TRC trouvé dans {root_dir}")

    log(f"{len(sessions)} sessions TRC trouvées", cfg.verbose)
    errors: List[Tuple[str, str]] = []

    for session in sessions:
        try:
            session_data = prepare_session_data(session=session, root_dir=root_dir, bad_df=bad_df, cfg=cfg)
            run_session_morlet(session_data=session_data, out_dir=out_dir, cfg=cfg)
        except Exception as exc:
            errors.append((session, repr(exc)))
            log(f"[ERROR] {session}: {exc}", cfg.verbose)

    summary = {
        "n_sessions": len(sessions),
        "n_errors": len(errors),
        "errors": errors,
        "config": asdict(cfg),
    }
    save_json(summary, out_dir / "run_summary.json")
    return summary



def load_session_exports(session_dir: Path, session: Optional[str] = None) -> Dict[str, Any]:
    """
    Recharge les exports de base d'une session déjà traitée par le pipeline Morlet.
    Utile pour refaire les figures sans tout recalculer, par ex.

    Paramètres
    ----------
    session_dir : Path
        Dossier de sortie propre à une session, p.ex. results_morlet_exploratoire/SESSION_X.
    session : Optional[str]
        Nom de session. Si None, le nom du dossier est utilisé.

    Retour
    ------
    Dict[str, Any]
        Dictionnaire contenant :
        {
            'session': str,
            'metadata': dict,
            'stims_df': pd.DataFrame,
            'freqs': np.ndarray,
            'pre_times': Optional[np.ndarray],
            'post_times': np.ndarray,
        }
    """
    session_name = session or session_dir.name
    meta_file = session_dir / f"{session_name}_metadata.json"
    trials_file = session_dir / f"{session_name}_stims_table.csv"
    freqs_file = session_dir / f"{session_name}_freqs.npy"
    pre_times_file = session_dir / f"{session_name}_pre_times.npy"
    post_times_file = session_dir / f"{session_name}_post_times.npy"

    if not meta_file.exists():
        raise FileNotFoundError(meta_file)
    if not trials_file.exists():
        raise FileNotFoundError(trials_file)
    if not freqs_file.exists():
        raise FileNotFoundError(freqs_file)
    if not post_times_file.exists():
        raise FileNotFoundError(post_times_file)

    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "session": session_name,
        "metadata": metadata,
        "stims_df": pd.read_csv(trials_file),
        "freqs": np.load(freqs_file),
        "pre_times": np.load(pre_times_file) if pre_times_file.exists() else None,
        "post_times": np.load(post_times_file),
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def compute_session_raw_limits_from_files(
    session_out: Path,
    session: str,
    bp_names: Sequence[str],
    raw_display_mode: str = "log10",
    q_low: float = 5.0,
    q_high: float = 95.0,
) -> Tuple[float, float]:
    """
    Calcule une échelle commune pour les figures de raw power d'une session.

    Paramètres
    ----------
    session_out : Path
        Dossier de sortie de la session Morlet.
    session : str
        Nom de session.
    bp_names : Sequence[str]
        Liste des canaux bipolaires à inspecter.
    raw_display_mode : str
        'raw' ou 'log10'.
    q_low : float
        Quantile bas.
    q_high : float
        Quantile haut.

    Retour
    ------
    Tuple[float, float]
        Limites globales (vmin, vmax).
    """
    vals: List[np.ndarray] = []
    for ch_name in bp_names:
        fp = channel_file(session_out, session, "power_post", ch_name)
        if not fp.exists():
            continue
        arr = np.load(fp, mmap_mode="r")
        if raw_display_mode == "log10":
            arr = np.log10(arr + 1e-12)
        vals.append(np.percentile(arr, [q_low, q_high]))

    if not vals:
        return 0.0, 1.0

    vals_arr = np.asarray(vals)
    return float(np.min(vals_arr[:, 0])), float(np.max(vals_arr[:, 1]))



def compute_session_metric_limits_from_files(
    session_out: Path,
    session: str,
    bp_names: Sequence[str],
    metric: str,
    q_abs: float = 99.0,
) -> Tuple[float, float]:
    """
    Calcule une échelle divergente commune pour une métrique donnée au sein d'une session.

    Paramètres
    ----------
    session_out : Path
        Dossier de sortie de la session Morlet.
    session : str
        Nom de session.
    bp_names : Sequence[str]
        Liste des canaux bipolaires à inspecter.
    metric : str
        Nom de la métrique, ex. 'logratio' ou 'zscore'.
    q_abs : float
        Quantile utilisé sur la valeur absolue.

    Retour
    ------
    Tuple[float, float]
        Limites (vmin, vmax) symétriques autour de zéro.
    """
    vals: List[float] = []
    for ch_name in bp_names:
        fp = channel_file(session_out, session, metric, ch_name)
        if not fp.exists():
            continue
        arr = np.load(fp, mmap_mode="r")
        vals.append(float(np.percentile(np.abs(arr), q_abs)))

    if not vals:
        return -1.0, 1.0

    vmax = float(np.max(vals))
    return -vmax, vmax



def plot_tf_panel(
    ax: plt.Axes,
    data_tf: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    add_z_mask: bool = False,
    z_threshold: float = 3.0,
):
    """
    Affiche une carte TF simple sur un axe Matplotlib.

    Paramètres
    ----------
    ax : plt.Axes
        Axe de destination.
    data_tf : np.ndarray
        Carte de forme (n_freqs, n_times).
    times : np.ndarray
        Axe temps.
    freqs : np.ndarray
        Axe fréquences.
    title : str
        Titre du panneau.
    cmap : str
        Colormap Matplotlib.
    vmin : float
        Borne basse de couleur.
    vmax : float
        Borne haute de couleur.
    add_z_mask : bool
        Si True, ajoute le contour des bins |z| > `z_threshold`.
    z_threshold : float
        Seuil utilisé pour ce contour descriptif.

    Retour
    ------
    matplotlib.image.AxesImage
        Objet image retourné par imshow.
    """
    im = ax.imshow(
        data_tf,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    if add_z_mask:
        mask = np.abs(data_tf) > z_threshold
        if mask.any():
            ax.contour(times, freqs, mask.astype(float), levels=[0.5], colors="k", linewidths=0.8)
            txt = f"|Z| > {z_threshold}"
            ax.text(
                0.98,
                0.98,
                txt,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.8),
            )
    return im



def make_tf_figure_for_shaft(
    trial_idx: int,
    shaft: str,
    ch_entries: List[Tuple[int, str]],
    data_by_channel: Dict[str, np.ndarray],
    times: np.ndarray,
    freqs: np.ndarray,
    out_file: Path,
    cmap: str,
    vmin: float,
    vmax: float,
    add_z_mask: bool = False,
    z_threshold: float = 3.0,
    dpi: int = 150,
    max_cols: int = 3,
    suptitle: Optional[str] = None,
    cmap_label: str = "Power",
) -> None:
    """
    Génère une figure multi-panneaux des cartes TF d'un même plot pour un essai donné.

    Paramètres
    ----------
    trial_idx : int
        Index de l'essai à représenter.
    shaft : str
        Nom du plot.
    ch_entries : List[Tuple[int, str]]
        Liste des canaux du plot sous forme (index_global, nom_bp).
    data_by_channel : Dict[str, np.ndarray]
        Dictionnaire nom_bp -> array (n_trials, n_freqs, n_times).
    times : np.ndarray
        Axe temps.
    freqs : np.ndarray
        Axe fréquences.
    out_file : Path
        Fichier PNG de sortie.
    cmap, vmin, vmax : paramètres d'affichage
        Paramètres classiques de colormap.
    add_z_mask : bool
        Ajoute le contour |Z| > seuil pour les cartes zscore.
    z_threshold : float
        Seuil z utilisé pour ce contour.
    dpi : int
        Résolution figure.
    max_cols : int
        Nombre maximal de colonnes de sous-figures.
    suptitle : Optional[str]
        Titre global.
    cmap_label : str
        Label de la colorbar.
    """
    n = len(ch_entries)
    ncols = min(max_cols, n)
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    axes = axes.ravel()

    last_im = None
    for ax, (_, ch_name) in zip(axes, ch_entries):
        if ch_name not in data_by_channel:
            ax.axis("off")
            continue
        arr = data_by_channel[ch_name][trial_idx]  # carte TF d'un essai pour un canal donné
        last_im = plot_tf_panel(
            ax=ax,
            data_tf=arr,
            times=times,
            freqs=freqs,
            title=ch_name,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            add_z_mask=add_z_mask,
            z_threshold=z_threshold,
        )

    for ax in axes[len(ch_entries):]:
        ax.axis("off")

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=12)

    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.08, top=0.90, wspace=0.30, hspace=0.35)
    if last_im is not None:
        cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.set_label(cmap_label)

    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)



def reduce_time_for_lineplot(arr: np.ndarray, stat: str = "mean") -> np.ndarray:
    """
    Réduit une carte TF sur l'axe temps pour produire un profil fréquentiel.

    Paramètres
    ----------
    arr : np.ndarray
        Carte de forme (n_freqs, n_times).
    stat : str
        'mean' ou 'median'.

    Retour
    ------
    np.ndarray
        Profil de forme (n_freqs,).
    """
    if stat == "mean":
        return np.mean(arr, axis=-1)
    if stat == "median":
        return np.median(arr, axis=-1)
    raise ValueError(f"lineplot_time_stat inconnu: {stat}")



def make_lineplot_figure_for_trial(
    trial_idx: int,
    shaft_groups: Dict[str, List[Tuple[int, str]]],
    data_by_channel: Dict[str, np.ndarray],
    freqs: np.ndarray,
    out_file: Path,
    stat: str = "mean",
    dpi: int = 150,
    suptitle: Optional[str] = None,
) -> None:
    """
    Génère, pour un essai donné, un ensemble de profils fréquentiels par plot.

    Paramètres
    ----------
    trial_idx : int
        Index de l'essai représenté.
    shaft_groups : Dict[str, List[Tuple[int, str]]]
        Groupement des canaux bipolaires par plot.
    data_by_channel : Dict[str, np.ndarray]
        Dictionnaire nom_bp -> array (n_trials, n_freqs, n_times).
    freqs : np.ndarray
        Axe fréquentiel.
    out_file : Path
        Fichier PNG de sortie.
    stat : str
        'mean' ou 'median' sur l'axe temps.
    dpi : int
        Résolution figure.
    suptitle : Optional[str]
        Titre global.
    """
    shafts = sorted(shaft_groups.keys())
    if len(shafts) == 0:
        return

    n = len(shafts)
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax, shaft in zip(axes, shafts):
        entries = shaft_groups[shaft]
        colors = plt.cm.jet(np.linspace(0, 1, len(entries)))  # une couleur locale par canal du plot
        plotted_any = False
        for local_idx, (_, ch_name) in enumerate(entries):
            if ch_name not in data_by_channel:
                continue
            arr = data_by_channel[ch_name]
            if trial_idx >= arr.shape[0]:
                continue
            y = reduce_time_for_lineplot(arr[trial_idx], stat=stat)
            ax.plot(freqs, y, label=ch_name, linewidth=1.2, color=colors[local_idx])
            plotted_any = True
        ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
        ax.set_title(shaft)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Value")
        if plotted_any:
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=9)

    for ax in axes[len(shafts):]:
        ax.axis("off")

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=12)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.90, wspace=0.30, hspace=0.35)
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)



def plot_tf_with_mask(
    data_map: np.ndarray,
    sig_mask: Optional[np.ndarray],
    freqs: np.ndarray,
    times: np.ndarray,
    title: str,
    out_file: Path,
    cmap: str = "RdBu_r",
    dpi: int = 150,
    contour: bool = True,
) -> None:
    """
    Trace une carte TF moyenne avec surimpression d'un masque significatif.

    Paramètres
    ----------
    data_map : np.ndarray
        Carte TF, typiquement moyenne de condition, forme (n_freqs, n_times).
    sig_mask : Optional[np.ndarray]
        Masque booléen de même forme, ou None.
    freqs : np.ndarray
        Axe fréquentiel.
    times : np.ndarray
        Axe temps.
    title : str
        Titre figure.
    out_file : Path
        Fichier PNG de sortie.
    cmap : str
        Colormap Matplotlib.
    dpi : int
        Résolution figure.
    contour : bool
        Si True, affiche le contour du masque significatif.
    """
    vmin, vmax = compute_diverging_limits(data_map)
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    im = ax.imshow(
        data_map,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    if sig_mask is not None and np.any(sig_mask) and contour:
        ax.contour(times, freqs, sig_mask.astype(float), levels=[0.5], colors="k", linewidths=0.9)

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Effect")

    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)



def generate_session_figures(session_out: Path, cfg: MorletConfig) -> None:
    """
    Génère les figures exploratoires essai par essai à partir des fichiers déjà exportés.

    Paramètres
    ----------
    session_out : Path
        Dossier de sortie d'une session Morlet.
    cfg : MorletConfig
        Configuration d'affichage.

    Notes
    -----
    Les figures produites sont :
    - cartes TF de raw power,
    - cartes TF pour chaque métrique,
    - lineplots fréquentiels par essai et par plot.

    L'organisation est de type :
    session_out/
        SESSION_figures/
            stim_00_LABEL/
                tf_raw/
                tf_logratio/
                ...
    """
    exports = load_session_exports(session_out)
    session = exports["session"]
    stims_df = exports["stims_df"]
    freqs = exports["freqs"]
    post_times = exports["post_times"]
    bp_names = exports["metadata"].get("bipolar_names", [])
    if not bp_names:
        log(f"[WARN] {session}: pas de bipolar_names dans metadata, figures annulées", cfg.verbose)
        return

    shaft_groups = group_bipolar_channels_by_shaft(bp_names)
    raw_by_channel: Dict[str, np.ndarray] = {}
    metric_data: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in cfg.metrics_to_compute}
    available_bp_names: List[str] = []

    for ch_name in bp_names:
        fp_raw = channel_file(session_out, session, "power_post", ch_name)
        if fp_raw.exists():
            arr = np.load(fp_raw, mmap_mode="r")
            if cfg.raw_display_mode == "log10":
                arr = np.log10(arr + 1e-12)
            raw_by_channel[ch_name] = arr
            available_bp_names.append(ch_name)

        for metric in cfg.metrics_to_compute:
            fp_m = channel_file(session_out, session, metric, ch_name)
            if fp_m.exists():
                metric_data[metric][ch_name] = np.load(fp_m, mmap_mode="r")

    available_bp_names = sorted(set(available_bp_names))
    if len(available_bp_names) == 0:
        log(f"[WARN] {session}: aucun fichier canal disponible pour les figures", cfg.verbose)
        return

    filtered_shaft_groups: Dict[str, List[Tuple[int, str]]] = {}
    for shaft, ch_entries in shaft_groups.items():
        kept = [(idx, ch_name) for idx, ch_name in ch_entries if ch_name in available_bp_names]
        if kept:
            filtered_shaft_groups[shaft] = kept
    if not filtered_shaft_groups:
        log(f"[WARN] {session}: aucun groupe d'électrodes avec données disponibles", cfg.verbose)
        return

    raw_vmin, raw_vmax = compute_session_raw_limits_from_files(
        session_out=session_out,
        session=session,
        bp_names=bp_names,
        raw_display_mode=cfg.raw_display_mode,
    )
    metric_limits = {
        metric: compute_session_metric_limits_from_files(
            session_out=session_out,
            session=session,
            bp_names=bp_names,
            metric=metric,
        )
        for metric in cfg.metrics_to_compute
    }

    n_trials = len(stims_df)
    for trial_idx in range(n_trials):
        stim_label = str(stims_df.loc[trial_idx, "label_stim"])
        if len(stim_label) > 8:
            stim_label = stim_label[:-8]  # retire éventuellement un suffixe du type '1025µsec' peu informatif en titre
        stim_group = str(stims_df.loc[trial_idx, "group_label"])
        stim_dir = session_out / f"{session}_figures" / f"stim_{trial_idx:02d}_{safe_name(stim_label)}"
        ensure_dir(stim_dir)

        raw_dir = stim_dir / "tf_raw"
        ensure_dir(raw_dir)
        for shaft, ch_entries in filtered_shaft_groups.items():
            out_file = raw_dir / f"{session}_{trial_idx:02d}_{shaft}_rawTF.png"
            make_tf_figure_for_shaft(
                trial_idx=trial_idx,
                shaft=shaft,
                ch_entries=ch_entries,
                data_by_channel=raw_by_channel,
                times=post_times,
                freqs=freqs,
                out_file=out_file,
                cmap=cfg.cmap_raw,
                vmin=raw_vmin,
                vmax=raw_vmax,
                add_z_mask=False,
                dpi=cfg.figure_dpi,
                max_cols=cfg.max_cols_per_figure,
                suptitle=f"{session}_{trial_idx} | {stim_label} | group: {stim_group} | raw power",
            )

        for metric in cfg.metrics_to_compute:
            metric_dir = stim_dir / f"tf_{metric}"
            ensure_dir(metric_dir)
            vmin, vmax = metric_limits[metric]
            add_mask = metric == "zscore"
            data_metric = metric_data[metric]

            filtered_metric_groups: Dict[str, List[Tuple[int, str]]] = {}
            for shaft, ch_entries in filtered_shaft_groups.items():
                kept = [(idx, ch_name) for idx, ch_name in ch_entries if ch_name in data_metric]
                if kept:
                    filtered_metric_groups[shaft] = kept

            for shaft, ch_entries in filtered_metric_groups.items():
                out_file = metric_dir / f"{session}_{trial_idx:02d}_{shaft}_{metric}TF.png"
                make_tf_figure_for_shaft(
                    trial_idx=trial_idx,
                    shaft=shaft,
                    ch_entries=ch_entries,
                    data_by_channel=data_metric,
                    times=post_times,
                    freqs=freqs,
                    out_file=out_file,
                    cmap=cfg.cmap_metric_div,
                    vmin=vmin,
                    vmax=vmax,
                    add_z_mask=add_mask,
                    z_threshold=cfg.z_threshold,
                    dpi=cfg.figure_dpi,
                    max_cols=cfg.max_cols_per_figure,
                    suptitle=f"{session}_{trial_idx} | {stim_label} | group: {stim_group} | metric: {metric}",
                    cmap_label=metric,
                )

            if filtered_metric_groups:
                out_file = stim_dir / f"{session}_{trial_idx:02d}_{metric}_lineplot.png"
                make_lineplot_figure_for_trial(
                    trial_idx=trial_idx,
                    shaft_groups=filtered_metric_groups,
                    data_by_channel=data_metric,
                    freqs=freqs,
                    out_file=out_file,
                    stat=cfg.lineplot_time_stat,
                    dpi=cfg.figure_dpi,
                    suptitle=f"{session}_{trial_idx} | {stim_label} | group: {stim_group} | metric: {metric} per frequency",
                )

    log(f"[OK] {session}: figures exploratoires générées", cfg.verbose)


def generate_session_figures_from_exports(session_dir: Path, cfg: MorletConfig) -> None:
    """
    Alias explicite pour générer les figures exploratoires à partir des exports déjà présents.

    Paramètres
    ----------
    session_dir : Path
        Dossier session dans `results_morlet_exploratoire`.
    cfg : MorletConfig
        Configuration d'affichage.
    """
    generate_session_figures(session_out=session_dir, cfg=cfg)


# ============================================================================
# MODULE EXPORTS (API notebook)
# ============================================================================

# la liste sert dans ce contexte : from lfp_utils import *
# dans ce cas, toutes les fonctions citées dans la liste __all__ sont importées

# Elle sert a contrôler ce qui est exposé comme API publique du module,
# documenter les fonctions considérées comme “fonctions utilisables depuis l’extérieur”,
# et éviter qu’un import * fasse remonter des helpers internes non voulus.

# Par contre, __all__ n'intervient pas ici : import lfp_utils

__all__ = [
    # Config
    "MorletConfig",
    
    # Morlet / baseline
    "compute_morlet_power",
    "compute_baseline_reference",
    "compute_metric",
    "compute_metrics_for_channel",

    # Sauvegarde
    "channel_file",
    "save_json",
    "save_session_metadata",
    "save_channel_morlet_outputs",

    # Orchestrateurs Morlet
    "prepare_session_data",
    "run_session_morlet",
    "run_all_sessions_morlet",

    # # Entrée / chargement
    "load_session_exports",

    # Visualisation TF 
    "compute_session_raw_limits_from_files",
    "compute_session_metric_limits_from_files",
    "plot_tf_panel",
    "make_tf_figure_for_shaft",
    "reduce_time_for_lineplot",
    "make_lineplot_figure_for_trial",
    "plot_tf_with_mask",

    # Visualisation en appel global
    "generate_session_figures",
    "generate_session_figures_from_exports",

]
