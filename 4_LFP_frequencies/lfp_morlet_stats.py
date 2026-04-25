#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lfp_morlet_stats.py
===================

Module utilitaire pour l'analyse exploratoire et inférentielle des dynamiques
LFP post-stimulation.

Ce module regroupe, dans un seul fichier réutilisable depuis un notebook :- les regroupements d'essais par condition cognitive et topographique,
- les statistiques par condition sur cartes TF déjà exportées,
- les orchestrateurs session par session ou sur l'ensemble des sessions.

Critères de regroupement des essais en conditions :
- regroupement des stims selon catégories cog (avec/sans sous-types cognitifs)
- regroupement des canaux d'une session selon distance à la stim : local (= meme électrode que stim) VS distant (= électrode différente)

Le module est conçu pour être piloté depuis un ou plusieurs notebooks, avec des
appels séparés pour :
1) la génération de figures exploratoires,
2) les statistiques par condition sur les cartes déjà exportées.

Auteur : Aube Darves-Bornoz
"""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne.stats import combine_adjacency, permutation_cluster_1samp_test
from scipy.stats import t, wilcoxon

from lfp_preprocess import (
    log,
    ensure_dir,
    safe_name,
    parse_list_cell,
    reduce_baseline_stat,
    parse_bipolar_shaft,
    normalize_channel_name,
)

from lfp_morlet_utils import (
    channel_file,
    save_json,
    load_session_exports,
    plot_tf_with_mask,
    compute_metric,
)


# ============================================================================
# CONFIGURATION OBJECT
# ============================================================================

@dataclass
class StatsConfig:
    """
    Configuration du pipeline de statistiques sur TF par condition.

    Le pipeline stats part des exports déjà calculés par `run_session_morlet` et
    n'a pas besoin de relire les TRC. Il teste, pour chaque bin TF, si une
    métrique normalisée à la baseline diffère de 0 à travers les essais d'une
    condition.

    Deux méthodes employées : 
    -------------------------
        a) Wilcoxon-FDR : pour chaque bin t-f : on en crée une série, à travers les essais
    de la condition, puis on compare la série à 0 par wilcoxon (correction FDR), et on identifie les bins t-f significatifs
        b) Cluster-based permutation tests 2D (t-f) sur chaque condition : les cartes
    sont considérées comme des observations répétées d’un même champ t-f, et les 
    clusters sont définis comme des ensembles contigus de bins dépassant un seuil initial. 
    La significativité des clusters est ensuite évaluée par permutations de signe à un échantillon contre zéro. 
    Permet de tenir compte de la dépendance spatiale naturelle des cartes t-f (car deux points t-f adjacents ne sont pas 
    indépendants) : plus robuste que tests binaires indépendants. Zones significatives = clusters dont probabilité corrigée est < 0.05

    Notes générales
    ---------------
    - `metric='logratio'` est le choix le plus robuste pour une
      inférence de groupe contre la baseline.
    - le cluster permutation 1-échantillon teste H0 : moyenne(X)=0 à travers les
      essais d'une condition, avec contiguïté temps × fréquence.
    - `min_trials_per_condition` évite d'interpréter des groupes trop petits.
    """

    # Dossiers d'entrée/sortie
    root_dir: str = "/home/aube/Documents/article_neuronal_stimic/effets_cog/theta_gamma_cog"
    input_root: str = root_dir + "/results_morlet_exploratoire" # où trouver les variables TF par essai
    output_root: str = root_dir + "/results_stats_conditions"

    # Métrique testée contre 0
    metric: str = "logratio"

    # Méthodes statistiques
    run_wilcoxon_fdr: bool = True
    run_cluster_perm: bool = True

    # Wilcoxon + FDR
    alpha_fdr: float = 0.05

    # Cluster permutation
    cluster_alpha: float = 0.05          # seuil corrigé pour retenir un cluster significatif
    n_permutations: int = 2000
    cluster_threshold_p: float = 0.05    # seuil initial de formation de cluster via t
    tail: int = 0                        # 0 = bilatéral, 1 = positif, -1 = négatif
    seed: int = 13

    # Conditions / groupes d'essais
    min_trials_per_condition: int = 5    # minimum de trials (canal+stim) de cog + topographie identiques pour créer une condition
    make_main_groups: bool = True        # comparaison générale de cog+ / controle / négatif
    make_cog_subgroups: bool = True      # comparaison des sous-types de cog+ / controle / négatif
    keep_main_groups_in_subgroup_mode: bool = True
    localities_to_test: Tuple[str, ...] = ("local", "distant")
    group_across_sessions: bool = True # Niveau d'agrégation statistique : sur toutes les sessions dispo
    also_run_per_session: bool = False # Si True, produit aussi les stats session par session
    pooled_output_subdir: str = "pooled_across_sessions"
    
    # Fallback si la métrique n'a pas été exportée, mais power_pre / power_post oui => calcul de log-ratio ici
    fallback_compute_metric_from_raw: bool = True
    baseline_stat_fallback: str = "median" # "mean" | "median". Ce qu'on utilise pour calculer P_pre(f) pour chaque essai
    eps: float = 1e-12

    # Figures de cartes TF 
    save_figures: bool = True
    figure_dpi: int = 150
    cmap: str = "RdBu_r"               # colormap pour les cartes de significativité TF par condition
    add_cluster_contour: bool = True   # on encadre les zones TF significatives

    # Représentation graphique par bandes de fréquences
    make_band_lineplots: bool = True
    freq_bands: Dict[str, Tuple[float, float]] = None # Bornes des bandes en Hz
    include_delta_if_available: bool = True # Si True, inclut delta seulement si des fréquences <= 4 Hz existent réellement

    verbose: bool = True

    def __post_init__(self):
        if self.freq_bands is None:
            self.freq_bands = {
                "delta": (1.0, 4.0),
                "theta": (4.0, 8.0),
                "alpha": (8.0, 12.0),
                "beta": (12.0, 30.0),
                "low_gamma": (30.0, 80.0),
                "high_gamma": (80.0, 150.0),
            }


# ============================================================================
# STATS FUNCTIONS
# ============================================================================

def fdr_bh(pvals: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Correction de Benjamini-Hochberg pour comparaisons multiples.

    Paramètres
    ----------
    pvals : np.ndarray
        Carte ou vecteur de p-values.
    alpha : float
        Seuil FDR cible.

    Retour
    ------
    Tuple[np.ndarray, np.ndarray]
        - reject_mask : masque booléen des tests retenus après FDR
        - pvals_adj   : p-values ajustées BH
    """
    p = np.asarray(pvals, dtype=float)
    shp = p.shape
    p_flat = p.ravel()

    finite = np.isfinite(p_flat)
    out_rej = np.zeros_like(p_flat, dtype=bool)
    out_adj = np.full_like(p_flat, np.nan, dtype=float)
    if finite.sum() == 0:
        return out_rej.reshape(shp), out_adj.reshape(shp)

    p_use = p_flat[finite]
    m = len(p_use)
    order = np.argsort(p_use)
    p_sorted = p_use[order]
    ranks = np.arange(1, m + 1)

    thresh = alpha * ranks / m
    below = p_sorted <= thresh
    if np.any(below):
        kmax = np.max(np.where(below)[0])
        cutoff = p_sorted[kmax]
        rej_use = p_use <= cutoff
    else:
        rej_use = np.zeros_like(p_use, dtype=bool)

    p_adj_sorted = p_sorted * m / ranks
    p_adj_sorted = np.minimum.accumulate(p_adj_sorted[::-1])[::-1]
    p_adj_sorted = np.clip(p_adj_sorted, 0, 1)

    p_adj_use = np.empty_like(p_use)
    p_adj_use[order] = p_adj_sorted
    out_rej[finite] = rej_use
    out_adj[finite] = p_adj_use

    return out_rej.reshape(shp), out_adj.reshape(shp)



def compute_metric_from_raw(
    power_pre: np.ndarray,
    power_post: np.ndarray,
    metric: str,
    baseline_stat: str = "median",
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Recalcule une métrique à partir de `power_pre` et `power_post` déjà exportés.

    Paramètres
    ----------
    power_pre : np.ndarray
        Array de forme (n_trials, n_freqs, n_pre_times).
    power_post : np.ndarray
        Array de forme (n_trials, n_freqs, n_post_times).
    metric : str
        'logratio', 'percent', 'subtract' ou 'zscore'.
    baseline_stat : str
        Statistique de baseline ('median' ou 'mean').
    eps : float
        Constante numérique de sécurité.

    Retour
    ------
    np.ndarray
        Carte métrique de forme (n_trials, n_freqs, n_post_times).
    """
    baseline = reduce_baseline_stat(power_pre, baseline_stat)  # une valeur de baseline par essai et par fréquence
    return compute_metric(
        power_post=power_post,
        baseline_ref=baseline,
        metric=metric,
        power_pre_for_z=power_pre if metric == "zscore" else None,
        eps=eps,
    )


def load_session_metadata(session_dir: Path, session: str) -> dict:
    fp = session_dir / f"{session}_metadata.json"
    if not fp.exists():
        raise FileNotFoundError(fp)
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def load_trials_table(session_dir: Path, session: str) -> pd.DataFrame:
    fp = session_dir / f"{session}_stims_table.csv"
    if not fp.exists():
        raise FileNotFoundError(fp)
    return pd.read_csv(fp)


def load_freqs_times(session_dir: Path, session: str) -> Tuple[np.ndarray, np.ndarray]:
    freqs = np.load(session_dir / f"{session}_freqs.npy")
    post_times = np.load(session_dir / f"{session}_post_times.npy")
    return freqs, post_times


def load_metric_or_compute(
    session_dir: Path,
    session: str,
    ch_name: str,
    metric: str,
    fallback_compute: bool = True,
    baseline_stat_fallback: str = "median",
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Recharge une métrique déjà exportée, ou la recalcule à partir de power_pre/power_post.

    Paramètres
    ----------
    session_dir : Path
        Dossier de la session Morlet déjà exportée.
    session : str
        Nom de session.
    ch_name : str
        Nom du canal bipolaire.
    metric : str
        Métrique à récupérer.
    fallback_compute : bool
        Si True, autorise le recalcul si le fichier métrique manque.
    baseline_stat_fallback : str
        Statistique de baseline à utiliser en fallback.
    eps : float
        Constante numérique de sécurité.

    Retour
    ------
    np.ndarray
        Array de forme (n_trials, n_freqs, n_post_times).
    """
    fp_metric = channel_file(session_dir, session, metric, ch_name)
    if fp_metric.exists():
        arr = np.load(fp_metric)
        if arr.ndim != 3:
            raise ValueError(f"{fp_metric.name}: shape inattendue {arr.shape}")
        return np.asarray(arr, dtype=np.float32)

    if not fallback_compute:
        raise FileNotFoundError(f"Fichier métrique introuvable : {fp_metric}")

    fp_pre = channel_file(session_dir, session, "power_pre", ch_name)
    fp_post = channel_file(session_dir, session, "power_post", ch_name)
    if not fp_pre.exists() or not fp_post.exists():
        raise FileNotFoundError(
            f"Métrique {metric} absente et fallback impossible pour {ch_name} (power_pre/power_post manquants)"
        )

    power_pre = np.load(fp_pre)
    power_post = np.load(fp_post)
    if power_pre.ndim != 3 or power_post.ndim != 3:
        raise ValueError(f"Shapes inattendues pour {ch_name}: pre={power_pre.shape}, post={power_post.shape}")

    return compute_metric_from_raw(
        power_pre=power_pre,
        power_post=power_post,
        metric=metric,
        baseline_stat=baseline_stat_fallback,
        eps=eps,
    )



def wilcoxon_map_against_zero(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Test de Wilcoxon signed-rank bin par bin contre 0.

    Paramètres
    ----------
    X : np.ndarray
        Array de forme (n_trials, n_freqs, n_times).

    Retour
    ------
    Tuple[np.ndarray, np.ndarray]
        - stat_map : carte des statistiques de Wilcoxon, forme (n_freqs, n_times)
        - p_map    : carte des p-values, forme (n_freqs, n_times)

    Notes
    -----
    Cette approche est simple et non paramétrique, mais multiplie fortement le
    nombre de tests ; elle doit donc être suivie d'une correction multiple.
    """
    if X.ndim != 3:
        raise ValueError(f"X doit être 3D, reçu {X.shape}")

    _, n_freqs, n_times = X.shape
    stat_map = np.full((n_freqs, n_times), np.nan, dtype=float)
    p_map = np.full((n_freqs, n_times), np.nan, dtype=float)

    for fi in range(n_freqs):
        for ti in range(n_times):
            x = X[:, fi, ti]
            x = x[np.isfinite(x)]  # ignore les valeurs non finies éventuelles
            if len(x) < 1:
                continue
            if np.allclose(x, 0.0):
                stat_map[fi, ti] = 0.0
                p_map[fi, ti] = 1.0
                continue
            try:
                res = wilcoxon(
                    x,
                    y=None,
                    zero_method="wilcox",
                    alternative="two-sided",
                    mode="auto",
                    correction=False,
                )
                stat_map[fi, ti] = float(res.statistic)
                p_map[fi, ti] = float(res.pvalue)
            except ValueError:
                stat_map[fi, ti] = np.nan
                p_map[fi, ti] = 1.0

    return stat_map, p_map



def cluster_1samp_map_against_zero(
    X: np.ndarray,
    cluster_alpha: float,
    n_permutations: int,
    tail: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Test de permutation par clusters 1-échantillon sur cartes TF.

    Paramètres
    ----------
    X : np.ndarray
        Array de forme (n_trials, n_freqs, n_times).
    cluster_alpha : float
        Seuil utilisé pour former les clusters via la statistique t.
    n_permutations : int
        Nombre de permutations.
    tail : int
        0 = bilatéral, 1 = positif, -1 = négatif.
    seed : int
        Graine aléatoire.

    Retour
    ------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - T_obs         : carte de statistiques observées
        - sig_mask      : masque booléen des bins appartenant à un cluster significatif
        - cluster_pvals : p-values des clusters détectés

    Notes méthodologiques
    ---------------------
    Ce test est particulièrement adapté aux cartes TF, car il tient compte de la
    contiguïté temps × fréquence et réduit l'inflation du nombre de comparaisons.
    """
    if X.ndim != 3:
        raise ValueError(f"X doit être 3D, reçu {X.shape}")

    n_trials, n_freqs, n_times = X.shape
    if n_trials < 2:
        raise ValueError("Cluster permutation requiert au moins 2 essais")

    if tail == 0:
        thr = t.ppf(1.0 - cluster_alpha / 2.0, df=n_trials - 1)
    elif tail == 1:
        thr = t.ppf(1.0 - cluster_alpha, df=n_trials - 1)
    elif tail == -1:
        thr = -t.ppf(1.0 - cluster_alpha, df=n_trials - 1)
    else:
        raise ValueError(f"tail invalide: {tail}")

    adjacency = combine_adjacency(n_freqs, n_times)  # matrice d'adjacence 2D temps × fréquence
    T_obs, clusters, cluster_pvals, _ = permutation_cluster_1samp_test(
        X=X.astype(np.float64),
        threshold=thr,
        n_permutations=n_permutations,
        tail=tail,
        adjacency=adjacency,
        out_type="mask",
        seed=seed,
        verbose=False,
    )

    sig_mask = np.zeros((n_freqs, n_times), dtype=bool)
    for cl_mask, pval in zip(clusters, cluster_pvals):
        if pval <= 0.05:
            sig_mask |= cl_mask

    return np.asarray(T_obs), sig_mask, np.asarray(cluster_pvals)



# ============================================================================
# CONDITION GROUPING
# ============================================================================

def build_main_condition_index(stims_df: pd.DataFrame, min_trials: int) -> Dict[str, np.ndarray]:
    """
    Construit les groupes d'essais principaux à partir de `group_label`.

    Paramètres
    ----------
    stims_df : pd.DataFrame
        Table d'essais exportée par le pipeline Morlet.
    min_trials : int
        Nombre minimal d'essais requis pour conserver une condition.

    Retour
    ------
    Dict[str, np.ndarray]
        Dictionnaire condition -> indices d'essais.
    """
    if "group_label" not in stims_df.columns:
        raise ValueError("La colonne 'group_label' est absente de session_stims_table.csv")

    out: Dict[str, List[int]] = {}
    for i, row in stims_df.iterrows():
        g = str(row["group_label"]).strip()
        if g in {"cog+", "controle", "negatif"}:
            out.setdefault(g, []).append(i)

    return {k: np.asarray(v, dtype=int) for k, v in out.items() if len(v) >= min_trials}



def build_cog_subcategory_index(
    stims_df: pd.DataFrame,
    min_trials: int,
    keep_main_groups: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Construit les groupes d'essais par sous-catégorie cognitive.

    Paramètres
    ----------
    stims_df : pd.DataFrame
        Table d'essais exportée par le pipeline Morlet.
    min_trials : int
        Nombre minimal d'essais requis pour conserver une condition.
    keep_main_groups : bool
        Si True, conserve aussi 'cog+', 'controle' et 'negatif' dans ce mode.

    Retour
    ------
    Dict[str, np.ndarray]
        Dictionnaire condition -> indices d'essais.

    Notes méthodologiques
    ---------------------
    Un essai `cog+` peut appartenir à plusieurs sous-groupes si plusieurs labels
    cognitifs sont annotés ; l'appartenance est alors dupliquée.
    """
    if "group_label" not in stims_df.columns:
        raise ValueError("La colonne 'group_label' est absente de session_stims_table.csv")
    if "cog_labels" not in stims_df.columns:
        raise ValueError("La colonne 'cog_labels' est absente de session_stims_table.csv")

    out: Dict[str, List[int]] = {}
    for i, row in stims_df.iterrows():
        group_label = str(row["group_label"]).strip()

        if keep_main_groups and group_label in {"controle", "negatif"}:
            out.setdefault(group_label, []).append(i)

        if group_label == "cog+":
            labels = parse_list_cell(row["cog_labels"])
            if keep_main_groups:
                out.setdefault("cog+", []).append(i)
            for lab in labels:
                if lab:
                    out.setdefault(f"cog::{lab}", []).append(i)

    return {k: np.asarray(v, dtype=int) for k, v in out.items() if len(v) >= min_trials}


def get_channel_locality_for_trial(channel_name: str, stim_shaft) -> str:
    """
    Compare le shaft du canal enregistré au shaft stimulé pour un essai donné.

    Retour :
      - 'local'   : même shaft
      - 'distant' : shaft différent
      - 'unknown' : impossible à déterminer
    """
    ch_shaft = parse_bipolar_shaft(channel_name)

    if pd.isna(stim_shaft):
        return "unknown"

    stim_shaft_norm = normalize_channel_name(str(stim_shaft))
    if not stim_shaft_norm:
        return "unknown"

    return "local" if ch_shaft == stim_shaft_norm else "distant"


def stack_condition_locality_maps(session: str,
                                  session_dir: Path,
                                  stims_df: pd.DataFrame,
                                  bp_names: Sequence[str],
                                  trial_indices: np.ndarray,
                                  locality: str,
                                  metric: str,
                                  fallback_compute: bool = True,
                                  baseline_stat_fallback: str = "median",
                                  eps: float = 1e-12,
                                  condition_name: Optional[str] = None
                                  ) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Empile toutes les cartes TF appartenant à une même condition d'essais et à une
    même localité ('local' ou 'distant').

    Chaque observation finale correspond à un couple (trial, channel).

    Paramètres
    ----------
    session : str
        Nom de session.
    session_dir : Path
        Dossier de la session dans les exports Morlet.
    stims_df : pd.DataFrame
        Table des essais exportée, contenant au minimum :
        - label_stim
        - group_label
        - stim_shaft
    bp_names : Sequence[str]
        Liste des canaux bipolaires disponibles dans la session.
    trial_indices : np.ndarray
        Indices des essais appartenant à la condition d'intérêt.
    locality : str
        'local' ou 'distant'.
    metric : str
        Métrique à charger ('logratio', 'percent', 'subtract', 'zscore', ...).
    fallback_compute : bool
        Si True, recalcule la métrique à partir de power_pre/power_post si le fichier
        métrique n'existe pas.
    baseline_stat_fallback : str
        'median' ou 'mean', utilisé si fallback_compute=True.
    eps : float
        Petite constante numérique.
    condition_name : Optional[str]
        Nom de la condition, uniquement pour enrichir obs_df.

    Retour
    ------
    X_stack : np.ndarray
        Array de shape (n_observations, n_freqs, n_times)
    obs_df : pd.DataFrame
        Table décrivant chaque observation empilée.
    """
    if locality not in {"local", "distant"}:
        raise ValueError(f"locality doit être 'local' ou 'distant', reçu: {locality}")

    required_cols = {"label_stim", "group_label", "stim_shaft"}
    missing = required_cols - set(stims_df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans stims_df pour stack_condition_locality_maps: {missing}")

    stacks: List[np.ndarray] = []
    rows: List[dict] = []

    # sécurise le type et enlève d'éventuels doublons
    trial_indices = np.asarray(trial_indices, dtype=int)
    trial_indices = np.unique(trial_indices)

    for ch_name in bp_names:
        arr = load_metric_or_compute(
            session_dir=session_dir,
            session=session,
            ch_name=ch_name,
            metric=metric,
            fallback_compute=fallback_compute,
            baseline_stat_fallback=baseline_stat_fallback,
            eps=eps,
        )  # shape attendu = (n_trials, n_freqs, n_times)

        if arr.ndim != 3:
            raise ValueError(f"{session} {ch_name}: shape inattendue pour {metric}: {arr.shape}")

        selected: List[int] = []
        channel_shaft = parse_bipolar_shaft(ch_name)

        for trial_idx in trial_indices:
            # sécurité si la table stims_df et les arrays ne sont pas parfaitement alignés
            if trial_idx >= len(stims_df):
                continue
            if trial_idx >= arr.shape[0]:
                continue

            stim_shaft = stims_df.loc[trial_idx, "stim_shaft"]
            obs_locality = get_channel_locality_for_trial(ch_name, stim_shaft)

            if obs_locality == "unknown": # on ignore explicitement les cas non classés
                continue

            if obs_locality != locality:
                continue

            selected.append(trial_idx)
            rows.append({
                "session": session,
                "condition": condition_name,
                "trial_idx": int(trial_idx),
                "channel_name": ch_name,
                "channel_shaft": channel_shaft,
                "stim_shaft": stim_shaft,
                "label_stim": stims_df.loc[trial_idx, "label_stim"],
                "group_label": stims_df.loc[trial_idx, "group_label"],
                "cog_labels": stims_df.loc[trial_idx, "cog_labels"] if "cog_labels" in stims_df.columns else None,
                "stim_bipolar_label": stims_df.loc[trial_idx, "stim_bipolar_label"] if "stim_bipolar_label" in stims_df.columns else None,
                "stim_contact_pair": stims_df.loc[trial_idx, "stim_contact_pair"] if "stim_contact_pair" in stims_df.columns else None,
                "locality": locality,
                "metric": metric,
            })

        if len(selected) > 0:
            stacks.append(arr[selected])

    if len(stacks) == 0:
        raise ValueError(f"Aucune observation pour locality={locality}")

    X_stack = np.concatenate(stacks, axis=0)

    if X_stack.ndim != 3:
        raise ValueError(f"X_stack shape inattendue après concaténation: {X_stack.shape}")

    obs_df = pd.DataFrame(rows)

    if len(obs_df) != X_stack.shape[0]:
        raise ValueError(
            f"Incohérence entre nombre d'observations et table descriptive: "
            f"X_stack.shape[0]={X_stack.shape[0]}, len(obs_df)={len(obs_df)}"
        )

    return X_stack, obs_df


def stack_condition_locality_maps_across_sessions(session_dirs: Sequence[Path],
                                                  condition_name: str,
                                                  locality: str,
                                                  cfg: StatsConfig,
                                                  subgroup_mode: bool = False
                                                  ) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Empile les observations appartenant à une combinaison
    condition × localité à travers toutes les sessions disponibles.

    Retour
    ------
    X_all  : shape = (n_observations_total, n_freqs, n_times)
    obs_df : table décrivant chaque observation empilée
    """
    all_stacks = []
    all_rows = []

    expected_freqs = None
    expected_times = None

    for session_dir in session_dirs:
        session = session_dir.name

        exports = load_session_exports(session_dir)
        stims_df = exports["stims_df"]
        freqs = exports["freqs"]
        post_times = exports["post_times"]
        meta = exports["metadata"]

        bp_names = meta.get("bipolar_names", [])
        if len(bp_names) == 0:
            continue

        if "stim_shaft" not in stims_df.columns:
            raise ValueError(
                f"{session}: 'stim_shaft' absent de session_stims_table.csv. "
                "Il faut régénérer les exports Morlet."
            )

        # Vérifie la compatibilité fréquentielle/temps entre sessions
        if expected_freqs is None:
            expected_freqs = freqs
            expected_times = post_times
        else:
            if not np.array_equal(freqs, expected_freqs):
                raise ValueError(
                    f"{session}: axe freqs différent entre sessions. "
                    "Le pooling across sessions impose les mêmes fréquences."
                )
            if not np.array_equal(post_times, expected_times):
                raise ValueError(
                    f"{session}: axe post_times différent entre sessions. "
                    "Le pooling across sessions impose le même axe temps."
                )

        # Construit les indices d'essais de cette condition dans cette session
        if subgroup_mode:
            cond_index = build_cog_subcategory_index(
                stims_df=stims_df,
                min_trials=cfg.min_trials_per_condition,
                keep_main_groups=cfg.keep_main_groups_in_subgroup_mode,
            )
        else:
            cond_index = build_main_condition_index(
                stims_df=stims_df,
                min_trials=cfg.min_trials_per_condition,
            )

        if condition_name not in cond_index:
            continue

        trial_indices = cond_index[condition_name]

        try:
            X_session, obs_df_session = stack_condition_locality_maps(
                session=session,
                session_dir=session_dir,
                stims_df=stims_df,
                bp_names=bp_names,
                trial_indices=trial_indices,
                locality=locality,
                metric=cfg.metric,
                fallback_compute=cfg.fallback_compute_metric_from_raw,
                baseline_stat_fallback=cfg.baseline_stat_fallback,
                eps=cfg.eps,
                condition_name=condition_name,
            )
        except Exception:
            continue

        all_stacks.append(X_session)
        all_rows.append(obs_df_session)

    if len(all_stacks) == 0:
        raise ValueError(
            f"Aucune observation across sessions pour condition={condition_name}, locality={locality}"
        )

    X_all = np.concatenate(all_stacks, axis=0)
    obs_df = pd.concat(all_rows, axis=0, ignore_index=True)

    if len(obs_df) != X_all.shape[0]:
        raise ValueError(
            f"Incohérence pooling across sessions: X_all.shape[0]={X_all.shape[0]}, len(obs_df)={len(obs_df)}"
        )

    return X_all, obs_df



# ============================================================================
# STATS RUN
# ============================================================================

def run_stats_for_one_condition(session: str,
                                out_condition_dir: Path,
                                condition_name: str,
                                locality: str,
                                X: np.ndarray,
                                obs_df: pd.DataFrame,
                                freqs: np.ndarray,
                                times: np.ndarray,
                                cfg: StatsConfig,
                                summary_rows: List[dict]) -> None:
    """
    Exécute les statistiques pour une combinaison :
      condition cognitive × localité (local / distant. Observations de la forme:

      X shape = (n_observations, n_freqs, n_times)

    où chaque observation correspond à un couple (trial, channel).

    Paramètres
    ----------
    session : str
        Nom de session.
    out_condition_dir : Path
        Dossier de sortie de la condition courante.
    condition_name : str
        Nom de la condition (ex. 'cog+', 'controle', 'negatif', 'cog::souvenir').
    locality : str
        'local' ou 'distant'.
    X : np.ndarray
        Tenseur empilé des observations, shape = (n_observations, n_freqs, n_times).
    obs_df : pd.DataFrame
        Table décrivant chaque observation incluse dans X.
        Une ligne = une observation = un couple (trial, channel).
    freqs : np.ndarray
        Axe fréquentiel.
    times : np.ndarray
        Axe temporel post-stimulation.
    cfg : StatsConfig
        Configuration statistique.
    summary_rows : List[dict]
        Liste enrichie au fil des résultats, pour créer ensuite summary_stats.csv.
    """
    if X.ndim != 3:
        raise ValueError(f"X doit être 3D, reçu {X.shape}")

    n_obs = X.shape[0]
    if n_obs == 0:
        raise ValueError(f"Aucune observation pour condition='{condition_name}', locality='{locality}'")

    # cartes descriptives de groupe
    mean_map = np.mean(X, axis=0)      # shape = (n_freqs, n_times)
    median_map = np.median(X, axis=0)  # shape = (n_freqs, n_times)

    # dossier spécifique à la localité
    locality_out = out_condition_dir / locality
    ensure_dir(locality_out)

    # sauvegardes descriptives
    np.save(locality_out / "mean_map.npy", mean_map.astype(np.float32))
    np.save(locality_out / "median_map.npy", median_map.astype(np.float32))
    obs_df.to_csv(locality_out / "observations_table.csv", index=False)

    # infos de résumé communes aux méthodes stats
    summary_base = {
        "session": session,
        "condition": condition_name,
        "locality": locality,
        "metric": cfg.metric,
        "n_observations": int(n_obs),
        "n_unique_trials": int(obs_df["trial_idx"].nunique()) if "trial_idx" in obs_df.columns else np.nan,
        "n_unique_channels": int(obs_df["channel_name"].nunique()) if "channel_name" in obs_df.columns else np.nan,
        "n_freqs": int(X.shape[1]),
        "n_times": int(X.shape[2]),
    }

    # -----------------------------------------------------------------
    # 1) Wilcoxon + FDR
    # -----------------------------------------------------------------
    if cfg.run_wilcoxon_fdr:
        stat_map, p_map = wilcoxon_map_against_zero(X)
        sig_mask_fdr, p_map_fdr = fdr_bh(p_map, alpha=cfg.alpha_fdr)

        np.save(locality_out / "stat_wilcoxon.npy", stat_map.astype(np.float32))
        np.save(locality_out / "pvals_wilcoxon.npy", p_map.astype(np.float32))
        np.save(locality_out / "pvals_wilcoxon_fdr.npy", p_map_fdr.astype(np.float32))
        np.save(locality_out / "sig_mask_wilcoxon_fdr.npy", sig_mask_fdr.astype(np.uint8))

        if cfg.save_figures:
            plot_tf_with_mask(
                data_map=mean_map,
                sig_mask=sig_mask_fdr,
                freqs=freqs,
                times=times,
                title=f"{session} | {condition_name} | {locality} | {cfg.metric} | Wilcoxon+FDR",
                out_file=locality_out / "figure_wilcoxon_fdr.png",
                cmap=cfg.cmap,
                dpi=cfg.figure_dpi,
                contour=True,
            )

        summary_rows.append({
            **summary_base,
            "method": "wilcoxon_fdr",
            "n_sig_bins": int(sig_mask_fdr.sum()),
            "frac_sig_bins": float(sig_mask_fdr.mean()),
        })

    # -----------------------------------------------------------------
    # 2) Cluster-based permutation
    # -----------------------------------------------------------------
    if cfg.run_cluster_perm:
        if n_obs >= 2:
            T_obs, sig_mask_cluster, cluster_pvals = cluster_1samp_map_against_zero(
                X=X,
                cluster_alpha=cfg.cluster_threshold_p,
                n_permutations=cfg.n_permutations,
                tail=cfg.tail,
                seed=cfg.seed,
            )

            np.save(locality_out / "T_obs_cluster.npy", T_obs.astype(np.float32))
            np.save(locality_out / "sig_mask_cluster.npy", sig_mask_cluster.astype(np.uint8))
            np.save(locality_out / "cluster_pvals.npy", cluster_pvals.astype(np.float32))

            if cfg.save_figures:
                plot_tf_with_mask(
                    data_map=mean_map,
                    sig_mask=sig_mask_cluster,
                    freqs=freqs,
                    times=times,
                    title=f"{session} | {condition_name} | {locality} | {cfg.metric} | Cluster perm",
                    out_file=locality_out / "figure_cluster.png",
                    cmap=cfg.cmap,
                    dpi=cfg.figure_dpi,
                    contour=cfg.add_cluster_contour,
                )

            summary_rows.append({
                **summary_base,
                "method": "cluster_perm",
                "n_sig_bins": int(sig_mask_cluster.sum()),
                "frac_sig_bins": float(sig_mask_cluster.mean()),
                "n_clusters_total": int(len(cluster_pvals)),
                "n_clusters_sig": int(np.sum(cluster_pvals <= cfg.cluster_alpha)),
            })


def run_stats_for_one_condition_across_sessions(out_condition_dir: Path,
                                               condition_name: str,
                                               locality: str,
                                               X: np.ndarray,
                                               obs_df: pd.DataFrame,
                                               freqs: np.ndarray,
                                               times: np.ndarray,
                                               cfg: StatsConfig,
                                               summary_rows: List[dict]) -> None:
    """
    Exécute les stats pour une condition × localité en agrégeant toutes les sessions.
    """
    if X.ndim != 3:
        raise ValueError(f"X doit être 3D, reçu {X.shape}")

    n_obs = X.shape[0]
    if n_obs == 0:
        raise ValueError(f"Aucune observation pour condition={condition_name}, locality={locality}")

    mean_map = np.mean(X, axis=0)
    median_map = np.median(X, axis=0)

    locality_out = out_condition_dir / locality
    ensure_dir(locality_out)

    np.save(locality_out / "mean_map.npy", mean_map.astype(np.float32))
    np.save(locality_out / "median_map.npy", median_map.astype(np.float32))
    obs_df.to_csv(locality_out / "observations_table.csv", index=False)

    summary_base = {
        "scope": "across_sessions",
        "condition": condition_name,
        "locality": locality,
        "metric": cfg.metric,
        "n_observations": int(n_obs),
        "n_unique_trials": int(obs_df[["session", "trial_idx"]].drop_duplicates().shape[0]),
        "n_unique_channels": int(obs_df[["session", "channel_name"]].drop_duplicates().shape[0]),
        "n_unique_sessions": int(obs_df["session"].nunique()),
        "n_freqs": int(X.shape[1]),
        "n_times": int(X.shape[2]),
    }

    if cfg.run_wilcoxon_fdr:
        stat_map, p_map = wilcoxon_map_against_zero(X)
        sig_mask_fdr, p_map_fdr = fdr_bh(p_map, alpha=cfg.alpha_fdr)

        np.save(locality_out / "stat_wilcoxon.npy", stat_map.astype(np.float32))
        np.save(locality_out / "pvals_wilcoxon.npy", p_map.astype(np.float32))
        np.save(locality_out / "pvals_wilcoxon_fdr.npy", p_map_fdr.astype(np.float32))
        np.save(locality_out / "sig_mask_wilcoxon_fdr.npy", sig_mask_fdr.astype(np.uint8))

        if cfg.save_figures:
            plot_tf_with_mask(
                data_map=mean_map,
                sig_mask=sig_mask_fdr,
                freqs=freqs,
                times=times,
                title=f"ALL_SESSIONS | {condition_name} | {locality} | {cfg.metric} | Wilcoxon+FDR",
                out_file=locality_out / "figure_wilcoxon_fdr.png",
                cmap=cfg.cmap,
                dpi=cfg.figure_dpi,
                contour=True,
            )

        summary_rows.append({
            **summary_base,
            "method": "wilcoxon_fdr",
            "n_sig_bins": int(sig_mask_fdr.sum()),
            "frac_sig_bins": float(sig_mask_fdr.mean()),
        })

    if cfg.run_cluster_perm and n_obs >= 2:
        T_obs, sig_mask_cluster, cluster_pvals = cluster_1samp_map_against_zero(
            X=X,
            cluster_alpha=cfg.cluster_threshold_p,
            n_permutations=cfg.n_permutations,
            tail=cfg.tail,
            seed=cfg.seed,
        )

        np.save(locality_out / "T_obs_cluster.npy", T_obs.astype(np.float32))
        np.save(locality_out / "sig_mask_cluster.npy", sig_mask_cluster.astype(np.uint8))
        np.save(locality_out / "cluster_pvals.npy", cluster_pvals.astype(np.float32))

        if cfg.save_figures:
            plot_tf_with_mask(
                data_map=mean_map,
                sig_mask=sig_mask_cluster,
                freqs=freqs,
                times=times,
                title=f"ALL_SESSIONS | {condition_name} | {locality} | {cfg.metric} | Cluster perm",
                out_file=locality_out / "figure_cluster.png",
                cmap=cfg.cmap,
                dpi=cfg.figure_dpi,
                contour=cfg.add_cluster_contour,
            )

        summary_rows.append({
            **summary_base,
            "method": "cluster_perm",
            "n_sig_bins": int(sig_mask_cluster.sum()),
            "frac_sig_bins": float(sig_mask_cluster.mean()),
            "n_clusters_total": int(len(cluster_pvals)),
            "n_clusters_sig": int(np.sum(cluster_pvals <= cfg.cluster_alpha)),
        })


def validate_same_tf_grid_across_sessions(session_dirs: Sequence[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vérifie que toutes les sessions qui seront comparées ont exactement les mêmes axes fréquence et temps.
    """
    ref_freqs = None
    ref_times = None

    for session_dir in session_dirs:
        exports = load_session_exports(session_dir)
        freqs = exports["freqs"]
        times = exports["post_times"]

        if ref_freqs is None:
            ref_freqs = freqs
            ref_times = times
            continue

        if not np.array_equal(freqs, ref_freqs):
            raise ValueError(f"{session_dir.name}: axe freqs différent")
        if not np.array_equal(times, ref_times):
            raise ValueError(f"{session_dir.name}: axe post_times différent")

    return ref_freqs, ref_times


def run_pooled_condition_stats(cfg: StatsConfig) -> Path:
    """
    Exécute les statistiques pooled across sessions.

    Pour chaque condition et localité, on agrège toutes les observations de
    toutes les sessions disponibles avant de calculer les stats.
    """
    input_root = Path(cfg.input_root)
    out_root = Path(cfg.output_root) / cfg.pooled_output_subdir
    ensure_dir(out_root)
    save_stats_config(out_root, cfg)

    session_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])
    if len(session_dirs) == 0:
        raise RuntimeError(f"Aucun sous-dossier session trouvé dans {input_root}")

    # on récupère les axes TF en s'assurant qu'ils sont tous identiques a travers les sessions
    ref_freqs, ref_post_times = validate_same_tf_grid_across_sessions(session_dirs)

    summary_rows: List[dict] = []

    # ---------------------------------------------------------------------
    # 1) conditions principales
    # ---------------------------------------------------------------------
    if cfg.make_main_groups:
        all_main_conditions = set()

        for session_dir in session_dirs:
            exports = load_session_exports(session_dir)
            cond_index = build_main_condition_index(
                stims_df=exports["stims_df"],
                min_trials=cfg.min_trials_per_condition,
            )
            all_main_conditions.update(cond_index.keys())

        root_main = out_root / "condition_main" / cfg.metric
        ensure_dir(root_main)

        for condition_name in sorted(all_main_conditions):
            cond_out = root_main / safe_name(condition_name)
            ensure_dir(cond_out)

            for locality in cfg.localities_to_test:
                try:
                    X_all, obs_df = stack_condition_locality_maps_across_sessions(
                        session_dirs=session_dirs,
                        condition_name=condition_name,
                        locality=locality,
                        cfg=cfg,
                        subgroup_mode=False,
                    )

                    run_stats_for_one_condition_across_sessions(
                        out_condition_dir=cond_out,
                        condition_name=condition_name,
                        locality=locality,
                        X=X_all,
                        obs_df=obs_df,
                        freqs=ref_freqs,
                        times=ref_post_times,
                        cfg=cfg,
                        summary_rows=summary_rows,
                    )

                except Exception as exc:
                    log(f"[WARN] pooled {condition_name} {locality}: {exc}", cfg.verbose)

    # ---------------------------------------------------------------------
    # 2) sous-catégories cognitives
    # ---------------------------------------------------------------------
    if cfg.make_cog_subgroups:
        all_sub_conditions = set()

        for session_dir in session_dirs:
            exports = load_session_exports(session_dir)
            cond_index = build_cog_subcategory_index(
                stims_df=exports["stims_df"],
                min_trials=cfg.min_trials_per_condition,
                keep_main_groups=cfg.keep_main_groups_in_subgroup_mode,
            )
            all_sub_conditions.update(cond_index.keys())

        root_sub = out_root / "condition_subcategories" / cfg.metric
        ensure_dir(root_sub)

        for condition_name in sorted(all_sub_conditions):
            cond_out = root_sub / safe_name(condition_name)
            ensure_dir(cond_out)

            for locality in cfg.localities_to_test:
                try:
                    X_all, obs_df = stack_condition_locality_maps_across_sessions(
                        session_dirs=session_dirs,
                        condition_name=condition_name,
                        locality=locality,
                        cfg=cfg,
                        subgroup_mode=True,
                    )

                    run_stats_for_one_condition_across_sessions(
                        out_condition_dir=cond_out,
                        condition_name=condition_name,
                        locality=locality,
                        X=X_all,
                        obs_df=obs_df,
                        freqs=ref_freqs,
                        times=ref_post_times,
                        cfg=cfg,
                        summary_rows=summary_rows,
                    )

                except Exception as exc:
                    log(f"[WARN] pooled {condition_name} {locality}: {exc}", cfg.verbose)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_root / "summary_stats_pooled.csv", index=False)
        log(f"[OK] résumé pooled sauvé -> {out_root / 'summary_stats_pooled.csv'}", cfg.verbose)
    else:
        log("[WARN] aucun résultat pooled produit", cfg.verbose)

    if cfg.make_band_lineplots:
        generate_pooled_band_lineplots(cfg)

    return out_root


def save_stats_config(out_dir: Path, cfg: StatsConfig) -> None:
    """
    Sauvegarde la configuration statistique d'une session.

    Paramètres
    ----------
    out_dir : Path
        Dossier de sortie de la session stats.
    cfg : StatsConfig
        Configuration à sérialiser.
    """
    save_json(asdict(cfg), out_dir / "stats_config.json")


def run_session_condition_stats(session_dir: Path, cfg: StatsConfig) -> Path:
    """
    Exécute les statistiques par condition pour une session déjà exportée
    par le pipeline Morlet.

    Nouvelle logique :
    ------------------
    On ne teste plus chaque canal séparément.

    Pour chaque condition (cog+, controle, negatif, ou sous-catégorie cog),
    on construit deux regroupements :
      - local   : toutes les observations (trial, channel) dont le shaft du canal
                  est le même que le shaft stimulé
      - distant : toutes les observations (trial, channel) dont le shaft du canal
                  est différent du shaft stimulé

    Les statistiques sont ensuite réalisées sur ces tenseurs empilés.
    """
    session = session_dir.name
    log(f"\n=== Session {session} ===")

    out_session_root = Path(cfg.output_root) / session
    ensure_dir(out_session_root)
    save_stats_config(out_session_root, cfg)


    stims_df = load_trials_table(session_dir, session)
    meta = load_session_metadata(session_dir, session)
    freqs, post_times = load_freqs_times(session_dir, session)

    bp_names = meta.get("bipolar_names", [])
    if not bp_names:
        raise ValueError(f"{session}: 'bipolar_names' absent ou vide dans metadata")

    # indispensable pour local/distant
    if "stim_shaft" not in stims_df.columns:
        raise ValueError(
            f"{session}: la colonne 'stim_shaft' est absente de session_stims_table.csv. "
            "Il faut régénérer les exports Morlet avec la version mise à jour de lfp_utils.py."
        )

    summary_rows: List[dict] = []

    # ---------------------------------------------------------------------
    # 1) GROUPES PRINCIPAUX
    # ---------------------------------------------------------------------
    if cfg.make_main_groups:
        cond_index_main = build_main_condition_index(
            stims_df=stims_df,
            min_trials=cfg.min_trials_per_condition,
        )

        root_main = out_session_root / "condition_main" / cfg.metric
        ensure_dir(root_main)

        for condition_name, trial_indices in cond_index_main.items():
            log(f"[INFO] {session}: condition main '{condition_name}' | n_trials={len(trial_indices)}")

            cond_out = root_main / safe_name(condition_name)
            ensure_dir(cond_out)

            for locality in cfg.localities_to_test:
                try:
                    X_stack, obs_df = stack_condition_locality_maps(
                        session=session,
                        session_dir=session_dir,
                        stims_df=stims_df,
                        bp_names=bp_names,
                        trial_indices=trial_indices,
                        locality=locality,
                        metric=cfg.metric,
                        fallback_compute=cfg.fallback_compute_metric_from_raw,
                        baseline_stat_fallback=cfg.baseline_stat_fallback,
                        eps=cfg.eps,
                        condition_name=condition_name
                    )

                    run_stats_for_one_condition(
                        session=session,
                        out_condition_dir=cond_out,
                        condition_name=condition_name,
                        locality=locality,
                        X=X_stack,
                        obs_df=obs_df,
                        freqs=freqs,
                        times=post_times,
                        cfg=cfg,
                        summary_rows=summary_rows,
                    )

                except Exception as exc:
                    log(f"[WARN] {session} {condition_name} {locality}: {exc}")

    # ---------------------------------------------------------------------
    # 2) SOUS-CATÉGORIES COGNITIVES
    # ---------------------------------------------------------------------
    if cfg.make_cog_subgroups:
        cond_index_sub = build_cog_subcategory_index(
            stims_df=stims_df,
            min_trials=cfg.min_trials_per_condition,
            keep_main_groups=cfg.keep_main_groups_in_subgroup_mode,
        )

        root_sub = out_session_root / "condition_subcategories" / cfg.metric
        ensure_dir(root_sub)

        for condition_name, trial_indices in cond_index_sub.items():
            log(f"[INFO] {session}: condition sub '{condition_name}' | n_trials={len(trial_indices)}")

            cond_out = root_sub / safe_name(condition_name)
            ensure_dir(cond_out)

            for locality in cfg.localities_to_test:
                try:
                    X_stack, obs_df = stack_condition_locality_maps(
                        session=session,
                        session_dir=session_dir,
                        stims_df=stims_df,
                        bp_names=bp_names,
                        trial_indices=trial_indices,
                        locality=locality,
                        metric=cfg.metric,
                        fallback_compute=cfg.fallback_compute_metric_from_raw,
                        baseline_stat_fallback=cfg.baseline_stat_fallback,
                        eps=cfg.eps,
                        condition_name=condition_name
                    )

                    run_stats_for_one_condition(
                        session=session,
                        out_condition_dir=cond_out,
                        condition_name=condition_name,
                        locality=locality,
                        X=X_stack,
                        obs_df=obs_df,
                        freqs=freqs,
                        times=post_times,
                        cfg=cfg,
                        summary_rows=summary_rows,
                    )

                except Exception as exc:
                    log(f"[WARN] {session} {condition_name} {locality}: {exc}")

    # ---------------------------------------------------------------------
    # 3) RÉSUMÉ
    # ---------------------------------------------------------------------
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(out_session_root / "summary_stats.csv", index=False)
        log(f"[OK] {session}: résumé sauvé -> {out_session_root / 'summary_stats.csv'}")
    else:
        log(f"[WARN] {session}: aucun résultat statistique produit")

    return out_session_root



def run_all_sessions_stats(cfg: StatsConfig) -> Dict[str, Any]:
    """
    Exécute les statistiques par condition sur toutes les sessions Morlet disponibles :
    - des stats pooled across sessions,
    - des stats session par session,
    - ou les deux, selon la config.

    Paramètres
    ----------
    cfg : StatsConfig
        Configuration statistique.

    Retour
    ------
    Dict[str, Any]
        Résumé d'exécution global.
    """
    output_root = Path(cfg.output_root)
    ensure_dir(output_root)

    errors = []

    # -------------------------------------------------------------
    # 1) stats globales pooled across sessions
    # -------------------------------------------------------------
    if cfg.group_across_sessions:
        try:
            run_pooled_condition_stats(cfg)
        except Exception as exc:
            errors.append(("pooled_across_sessions", repr(exc)))
            log(f"[ERROR] pooled_across_sessions: {exc}", cfg.verbose)

    # -------------------------------------------------------------
    # 2) stats session par session, si demandé en plus
    # -------------------------------------------------------------
    if cfg.also_run_per_session:
        input_root = Path(cfg.input_root)
        session_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])
        if len(session_dirs) == 0:
            raise RuntimeError(f"Aucun sous-dossier session trouvé dans {input_root}")

        for session_dir in session_dirs:
            try:
                run_session_condition_stats(session_dir=session_dir, cfg=cfg)
            except Exception as exc:
                errors.append((session_dir.name, repr(exc)))
                log(f"[ERROR] {session_dir.name}: {exc}", cfg.verbose)

    summary = {
        "config": asdict(cfg),
        "n_errors": len(errors),
        "errors": errors,
    }

    with open(output_root / "run_summary_stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary

# ============================================================================
# VISUALIZATION
# ============================================================================

def get_band_frequency_indices(freqs: np.ndarray,
                               freq_bands: Dict[str, Tuple[float, float]],
                               include_delta_if_available: bool = True
                               ) -> Dict[str, np.ndarray]:
    """
    Associe à chaque bande les indices de fréquences présents dans `freqs`.
    Est utilisé dans lineplots récap par bande de fréquence.
    Retour :
      dict de type :
      {
        "theta": np.array([...]),
        "alpha": np.array([...]),
        ...
      }

    Seules les bandes ayant au moins une fréquence disponible sont conservées.
    """
    freqs = np.asarray(freqs, dtype=float)
    out = {}

    for band_name, (f_low, f_high) in freq_bands.items():
        if band_name == "delta" and not include_delta_if_available:
            continue

        idx = np.where((freqs >= f_low) & (freqs < f_high))[0]
        # cas particulier : pour la dernière bande, on peut inclure la borne supérieure
        if band_name == "high_gamma":
            idx = np.where((freqs >= f_low) & (freqs <= f_high))[0]

        if len(idx) == 0:
            continue

        # delta seulement si effectivement calculé
        if band_name == "delta" and np.nanmin(freqs) > 4.0:
            continue

        out[band_name] = idx

    return out

def reduce_tf_map_to_band_timecourse(data_map: np.ndarray,
                                     freq_indices: np.ndarray,
                                     freq_reduce: str = "mean"
                                     ) -> np.ndarray:
    """
    Réduit une carte TF (n_freqs, n_times) à une courbe temporelle pour une bande.

    data_map : np.ndarray
        Carte TF shape = (n_freqs, n_times)
    freq_indices : np.ndarray
        Indices des fréquences appartenant à la bande.
    freq_reduce : str
        'mean' ou 'median'

    Retour
    ------
    np.ndarray shape = (n_times,)
    """
    band_map = data_map[freq_indices, :]  # shape = (n_band_freqs, n_times)

    if freq_reduce == "mean":
        return np.mean(band_map, axis=0)
    elif freq_reduce == "median":
        return np.median(band_map, axis=0)
    else:
        raise ValueError(f"freq_reduce inconnu: {freq_reduce}")


def reduce_band_significance_mask(sig_mask_tf: np.ndarray,
                                  freq_indices: np.ndarray,
                                  mode: str = "any"
                                  ) -> np.ndarray:
    """
    Réduit un masque TF binaire (n_freqs, n_times) à un masque temporel pour une bande.

    Paramètres
    ----------
    sig_mask_tf : np.ndarray
        Masque significatif TF, shape = (n_freqs, n_times)
    freq_indices : np.ndarray
        Indices des fréquences de la bande
    mode : str
        'any'  : t significatif si au moins une fréquence de la bande est significative
        'all'  : t significatif si toutes les fréquences de la bande sont significatives
        'frac' : non implémenté ici, pourrait être ajouté plus tard

    Retour
    ------
    np.ndarray bool, shape = (n_times,)
    """
    band_mask = sig_mask_tf[freq_indices, :]  # (n_band_freqs, n_times)

    if mode == "any":
        return np.any(band_mask, axis=0)
    elif mode == "all":
        return np.all(band_mask, axis=0)
    else:
        raise ValueError(f"mode inconnu: {mode}")


def split_band_significance_by_direction(mean_map: np.ndarray,
                                         sig_mask_tf: np.ndarray,
                                         freq_indices: np.ndarray,
                                         sig_reduce_mode: str = "any",
                                         effect_reduce: str = "mean"
                                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    À partir de la carte moyenne TF et du masque significatif TF, détermine
    pour une bande donnée :

    - les temps significatifs avec effet positif
    - les temps significatifs avec effet négatif

    Paramètres
    ----------
    mean_map : np.ndarray
        Carte moyenne TF, shape = (n_freqs, n_times)
    sig_mask_tf : np.ndarray
        Masque significatif TF, shape = (n_freqs, n_times)
    freq_indices : np.ndarray
        Indices des fréquences de la bande
    sig_reduce_mode : str
        'any' ou 'all' pour réduire le masque TF vers le temps
    effect_reduce : str
        'mean' ou 'median' pour résumer l'effet dans la bande

    Retour
    ------
    sig_pos_t : np.ndarray bool, shape = (n_times,)
    sig_neg_t : np.ndarray bool, shape = (n_times,)
    """
    sig_t = reduce_band_significance_mask(
        sig_mask_tf=sig_mask_tf,
        freq_indices=freq_indices,
        mode=sig_reduce_mode,
    )

    y = reduce_tf_map_to_band_timecourse(
        data_map=mean_map,
        freq_indices=freq_indices,
        freq_reduce=effect_reduce,
    )

    sig_pos_t = sig_t & (y > 0)
    sig_neg_t = sig_t & (y < 0)

    return sig_pos_t, sig_neg_t


def plot_band_timecourse_with_significance(ax,
                                           times: np.ndarray,
                                           y: np.ndarray,
                                           sig_pos_t: np.ndarray,
                                           sig_neg_t: np.ndarray,
                                           color,
                                           linewidth_base: float = 1.5,
                                           linewidth_sig: float = 2.4,
                                           alpha_patch: float = 0.18,
                                           linestyle_base: str = "--") -> None:
    """
    Trace une courbe temporelle pour une bande avec :
    - courbe complète en pointillés
    - patches verticaux verts/rouges aux temps significatifs
    - surimpression en trait plein sur les segments significatifs

    Paramètres
    ----------
    ax : matplotlib.axes.Axes
        Axe de tracé.
    times : np.ndarray
        Axe temporel, shape = (n_times,)
    y : np.ndarray
        Courbe temporelle de la bande, shape = (n_times,)
    sig_pos_t : np.ndarray
        Booléen, True aux temps significatifs avec effet positif.
    sig_neg_t : np.ndarray
        Booléen, True aux temps significatifs avec effet négatif.
    color :
        Couleur de la courbe.
    linewidth_base : float
        Épaisseur du tracé pointillé.
    linewidth_sig : float
        Épaisseur du tracé plein sur segments significatifs.
    alpha_patch : float
        Transparence des patches.
    linestyle_base : str
        Style de la courbe de fond, par défaut pointillé.
    """
    times = np.asarray(times)
    y = np.asarray(y)
    sig_pos_t = np.asarray(sig_pos_t, dtype=bool)
    sig_neg_t = np.asarray(sig_neg_t, dtype=bool)

    if not (len(times) == len(y) == len(sig_pos_t) == len(sig_neg_t)):
        raise ValueError("times, y, sig_pos_t et sig_neg_t doivent avoir la même longueur")

    # -----------------------------------------------------------------
    # 1) patches verticaux semi-transparents
    # -----------------------------------------------------------------
    def add_time_patches(mask: np.ndarray, patch_color: str):
        if not np.any(mask):
            return

        mask_int = mask.astype(int)
        diff = np.diff(np.r_[0, mask_int, 0])
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1

        if len(times) > 1:
            dt = float(np.median(np.diff(times)))
        else:
            dt = 0.0

        for s, e in zip(starts, ends):
            x0 = times[s] - dt / 2.0
            x1 = times[e] + dt / 2.0
            ax.axvspan(x0, x1, color=patch_color, alpha=alpha_patch, ec=None)

    add_time_patches(sig_neg_t, patch_color="red")
    add_time_patches(sig_pos_t, patch_color="green")

    # -----------------------------------------------------------------
    # 2) courbe complète en pointillés
    # -----------------------------------------------------------------
    ax.plot(times, y, color=color, linestyle=linestyle_base, linewidth=linewidth_base)
    
    # -----------------------------------------------------------------
    # 3) surimpression en trait plein sur les temps significatifs
    # -----------------------------------------------------------------
    sig_any_t = sig_pos_t | sig_neg_t
    y_sig = y.copy()
    y_sig[~sig_any_t] = np.nan
    ax.plot(times, y_sig, color=color, linestyle="-", linewidth=linewidth_sig)

    # repère zéro
    ax.axhline(0, color="k", linestyle=":", linewidth=1.0)


def get_band_colors(band_names: Sequence[str], cmap_name: str = "viridis") -> Dict[str, Tuple[float, float, float, float]]:
    cmap = plt.get_cmap(cmap_name)
    n = len(band_names)

    if n == 1:
        vals = [0.7]
    else:
        vals = np.linspace(0.2, 0.9, n)

    return {band: cmap(v) for band, v in zip(band_names, vals)}


def format_band_label(band_name: str,
                      freq_range: Tuple[float, float]) -> str:
    """
    Formate un label de bande fréquentielle avec son intervalle.

    Exemples
    --------
    ('theta', (4.0, 8.0)) -> 'theta [4–8 Hz]'
    ('high_gamma', (70.0, 150.0)) -> 'high_gamma [70–150 Hz]'
    """
    f_low, f_high = freq_range

    def _fmt(x: float) -> str:
        return str(int(x)) if float(x).is_integer() else f"{x:g}"

    return f"{band_name} [{_fmt(f_low)}–{_fmt(f_high)} Hz]"


def plot_pooled_band_timecourses(mean_map: np.ndarray,
                                 sig_mask_tf: np.ndarray,
                                 freqs: np.ndarray,
                                 times: np.ndarray,
                                 out_file: Path,
                                 freq_bands: Dict[str, Tuple[float, float]],
                                 include_delta_if_available: bool = True,
                                 sig_reduce_mode: str = "any",
                                 freq_reduce: str = "mean",
                                 cmap_name: str = "viridis",
                                 dpi: int = 150,
                                 title: Optional[str] = None):
    """
    Trace une figure avec un subplot par bande de fréquences.

    Dans chaque subplot :
    - une courbe temporelle correspondant à la moyenne de mean_map sur la bande
    - des patches verticaux semi-transparents :
        * verts pour les temps significatifs avec effet positif
        * rouges pour les temps significatifs avec effet négatif
    """
    band_to_idx = get_band_frequency_indices(
        freqs=freqs,
        freq_bands=freq_bands,
        include_delta_if_available=include_delta_if_available,
    )

    if len(band_to_idx) == 0:
        raise ValueError("Aucune bande fréquentielle disponible")

    band_order = [b for b in ["delta", "theta", "alpha", "beta", "low_gamma", "high_gamma"] if b in band_to_idx]
    colors = get_band_colors(band_order, cmap_name=cmap_name)

    n_bands = len(band_order)
    fig, axes = plt.subplots(
        n_bands, 1,
        figsize=(10, 2.2 * n_bands),
        sharex=True,
        squeeze=False
    )
    axes = axes.ravel()

    for ax, band_name in zip(axes, band_order):
        idx = band_to_idx[band_name]

        y = reduce_tf_map_to_band_timecourse(
            data_map=mean_map,
            freq_indices=idx,
            freq_reduce=freq_reduce,
        )

        sig_pos_t, sig_neg_t = split_band_significance_by_direction(
            mean_map=mean_map,
            sig_mask_tf=sig_mask_tf,
            freq_indices=idx,
            sig_reduce_mode=sig_reduce_mode,
            effect_reduce=freq_reduce,
        )

        plot_band_timecourse_with_significance(
            ax=ax,
            times=times,
            y=y,
            sig_pos_t=sig_pos_t,
            sig_neg_t=sig_neg_t,
            color=colors[band_name]
        )

        ax.set_ylabel('Mean log-ratio')
        ax.set_title(format_band_label(band_name, freq_bands[band_name]))

    axes[-1].set_xlabel("Time (s)")

    if title is not None:
        fig.suptitle(title, y=0.995)

    fig.tight_layout()

    # définir les xticks de 0 à post_length toutes les 0.5 s
    xticks = np.arange(0, times[-1] + 1e-6, 0.5)  # on ajoute un chouia + que times[-1] (tres petit) pour etre sur d'afficher times[-1] mais pas au-dela

    for ax in axes:
        ax.set_xlim(times[0], times[-1])
        ax.set_xticks(xticks)

    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_pooled_band_lineplots(cfg: StatsConfig) -> None:
    """
    Génère les lineplots temporels par bandes de fréquences pour les résultats pooled.

    Priorité au masque cluster-based permutation.
    Si absent, fallback sur le masque Wilcoxon+FDR.
    """
    input_root = Path(cfg.input_root)
    pooled_root = Path(cfg.output_root) / cfg.pooled_output_subdir

    session_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])
    if len(session_dirs) == 0:
        raise RuntimeError(f"Aucun sous-dossier session trouvé dans {input_root}")

    ref_exports = load_session_exports(session_dirs[0])
    freqs = ref_exports["freqs"]
    times = ref_exports["post_times"]

    for grouping_name in ["condition_main", "condition_subcategories"]:
        grouping_root = pooled_root / grouping_name / cfg.metric
        if not grouping_root.exists():
            continue

        for condition_dir in sorted([p for p in grouping_root.iterdir() if p.is_dir()]):
            condition_name = condition_dir.name

            for locality in cfg.localities_to_test:
                locality_dir = condition_dir / locality
                if not locality_dir.exists():
                    continue

                mean_fp = locality_dir / "mean_map.npy"
                cluster_mask_fp = locality_dir / "sig_mask_cluster.npy"
                wilcoxon_mask_fp = locality_dir / "sig_mask_wilcoxon_fdr.npy"

                if not mean_fp.exists():
                    continue

                mean_map = np.load(mean_fp)

                if cluster_mask_fp.exists():
                    sig_mask_tf = np.load(cluster_mask_fp).astype(bool)
                    sig_method = "cluster"
                elif wilcoxon_mask_fp.exists():
                    sig_mask_tf = np.load(wilcoxon_mask_fp).astype(bool)
                    sig_method = "wilcoxon_fdr"
                else:
                    continue

                out_file = locality_dir / f"band_timecourses_{sig_method}.png"

                plot_pooled_band_timecourses(
                    mean_map=mean_map,
                    sig_mask_tf=sig_mask_tf,
                    freqs=freqs,
                    times=times,
                    out_file=out_file,
                    freq_bands=cfg.freq_bands,
                    include_delta_if_available=cfg.include_delta_if_available,
                    sig_reduce_mode="any",
                    freq_reduce="mean",
                    cmap_name="viridis",
                    dpi=cfg.figure_dpi,
                    title=f"{grouping_name} | {condition_name} | {locality} | {cfg.metric}",
                )


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
    "StatsConfig",

    # Stats functions
    "fdr_bh",
    "compute_metric_from_raw",
    "load_session_metadata",
    "load_trials_table",
    "load_freqs_times",
    "load_metric_or_compute",
    "wilcoxon_map_against_zero",
    "cluster_1samp_map_against_zero",

    # Conditions grouping
    "get_channel_locality_for_trial",
    "stack_condition_locality_maps",
    "stack_condition_locality_maps_across_sessions",

    # Stats run
    "run_stats_for_one_condition",
    "run_stats_for_one_condition_across_sessions",
    "validate_same_tf_grid_across_sessions",
    "run_pooled_condition_stats",
    "save_stats_config",
    "run_session_condition_stats",
    "run_all_sessions_stats",

    # Visualisation par bandes de fréquences
    "get_band_frequency_indices",
    "reduce_tf_map_to_band_timecourse",
    "reduce_band_significance_mask",
    "split_band_significance_by_direction",
    "plot_band_timecourse_with_significance",
    "get_band_colors",
    "format_band_label",
    "plot_pooled_band_timecourses",
    "generate_pooled_band_lineplots"
]
