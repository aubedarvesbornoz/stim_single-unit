#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lfp_preprocess.py
=================

Module utilitaire pour l'analyse exploratoire et inférentielle des dynamiques
LFP post-stimulation.

Ce module regroupe, dans un seul fichier réutilisable depuis un notebook :
- la lecture et la fusion des tables d'événements,
- la gestion des canaux invalides et du montage bipolaire adjacent,
- la lecture de TRC Micromed,
- le prétraitement du signal (filtres, montage) et l'extraction des epochs pré/post.

Auteur : Aube Darves-Bornoz
"""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import ast
import re
from pathlib import Path
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mne
import numpy as np
import pandas as pd
from scipy import signal

# ============================================================================
# GENERAL HELPERS
# ============================================================================

def log(msg: str, verbose: bool = True) -> None:
    """
    Affiche un message de log si `verbose` vaut True.
        msg : str. Message à afficher.
        verbose : bool. Active ou non l'affichage.
    """
    if verbose:
        print(msg, flush=True)


def ensure_dir(path: Path) -> None:
    """
    Crée un dossier, et ses parents, s'il n'existe pas encore. Concerne l'export des variables TF en .npy + des figures Morlet et stats
        path : Path. Dossier à créer.
    """
    path.mkdir(parents=True, exist_ok=True)


def safe_name(name: str) -> str:
    """
    Convertit un nom de canal ou de fichier en str sûre pour le filesystem.
        name : str. Chaîne potentiellement contenant des caractères spéciaux.
    Retour : str. Chaîne nettoyée, compatible avec des noms de fichiers.
    """
    return re.sub(r"[^A-Za-z0-9_\-]", "_", str(name)) # re.sub(pattern_to_replace, replacement, initial_string)


def make_freqs(fmin: float, fmax: float, n_freqs: int, scale: str = "linear") -> np.ndarray:
    """
    Construit l'axe fréquentiel utilisé pour la TFR Morlet.

    Paramètres
    ----------
    fmin : float
        Fréquence minimale en Hz.
    fmax : float
        Fréquence maximale en Hz.
    n_freqs : int
        Nombre total de fréquences échantillonnées.
    scale : str
        Répartition des fréquences, parmi 'linear', 'log' ou 'semilog'.

    Retour
    ------
    np.ndarray
        Tableau 1D de fréquences en Hz.

    Notes
    -----
    - 'linear' répartit uniformément les fréquences.
    - 'log' / 'semilog' densifient les basses fréquences, utile lorsqu'on veut
      mieux couvrir theta/alpha par rapport a gamma.
    """
    if scale == "linear":
        return np.linspace(fmin, fmax, n_freqs)
    if scale == "log":
        return np.geomspace(fmin, fmax, n_freqs)
    if scale == "semilog":
        x = np.linspace(0.0, 1.0, n_freqs)  # interpolation régulière dans l'espace log
        return fmin * (fmax / fmin) ** x
    raise ValueError(f"freq_scale inconnu: {scale}")


def parse_list_cell(value: Any) -> List[str]:
    """
    Décomposition d'une cellule contenant une liste sérialisée ou une chaîne simple. Utilisé pour les classification d'effets cog.

    Exemples acceptés
    -----------------
    - NaN
    - "['souvenir', 'emotion']"
    - "souvenir, emotion"
    - "souvenir"

    Paramètres
    ----------
    value : Any
        Valeur lue depuis une cellule pandas/CSV/Excel.

    Retour
    ------
    List[str]
        Liste de labels nettoyés.
    """
    if pd.isna(value): # si c'est un NaN, on renvoie liste vide
        return []

    s = str(value).strip()
    if not s or s.lower() == "nan": # si c'est un pseudo NaN considéré comme un str, on renvoie liste vide
        return []

    try:
        obj = ast.literal_eval(s)  # essaie d'interpréter une vraie liste Python sérialisée
        if isinstance(obj, (list, tuple, set)): # si l'objet est bien une liste
            return [str(v).strip() for v in obj if str(v).strip()] # alors on renvoie la liste de manière correctement formatée
        return [str(obj).strip()] # si ce n'est pas une liste, alors c'est un str qu'on renvoie comme une liste avec un element unique
    except Exception:
        parts = re.split(r"[;,/]", s)  # fallback pour des listes écrites à la main : on extrait les elements comme étant entre des symboles type [ ; , 
        return [p.strip().strip("'\"") for p in parts if p.strip()] # et on renvoie ces elements sous format de liste


def reduce_baseline_stat(x: np.ndarray, stat: str) -> np.ndarray:
    """
    Réduit une baseline sur l'axe temps, pour une fréquence donnée, par médiane ou moyenne. Aboutit à P_baseline(f) à partir de P_baseline(t, f).

    Paramètres
    ----------
    x : np.ndarray
        Array de forme (n_frequencies, n_times). 
    stat : str
        'median' ou 'mean'.

    Retour
    ------
    np.ndarray
        Array de forme (n_frequencies) après réduction du dernier axe.
    """
    if stat == "median":
        return np.median(x, axis=-1) 
    if stat == "mean":
        return np.mean(x, axis=-1)
    raise ValueError(f"baseline_stat inconnu: {stat}")


def compute_diverging_limits(arr: np.ndarray, q: float = 99.0) -> Tuple[float, float]:
    """
    Calcule des limites symétriques pour une échelle de couleur divergente.

    Paramètres
    ----------
    arr : np.ndarray
        Array de données, typiquement une carte TF déjà normalisée.
    q : float
        Quantile utilisé sur la valeur absolue.

    Retour
    ------
    Tuple[float, float]
        Couple (vmin, vmax) symétrique autour de zéro.
    """
    vmax = float(np.nanpercentile(np.abs(arr), q))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return -vmax, vmax


# ============================================================================
# CHANNEL AND EVENT HELPERS
# ============================================================================

def parse_bad_channels_cell(cell_value: Any) -> List[str]:
    """
    Parse le contenu d'une cellule Excel listant des canaux invalides.

    Exemples acceptés
    -----------------
    - "Bp1, C2"
    - "Bp1; C2"
    - "[Bp1, C2]"
    - NaN

    Retour
    ------
    List[str]
        Liste de contacts monopolaires à exclure.
    """
    if pd.isna(cell_value):
        return []

    s = str(cell_value).strip()
    if not s:
        return []

    s = s.strip("[](){}")  # retire d'éventuels délimiteurs décoratifs
    parts = re.split(r"[;,]", s)  # accepte séparateurs virgule ou point-virgule
    return [p.strip() for p in parts if p.strip()]



def normalize_channel_name(ch: str) -> str:
    """
    Normalise un label de canal en supprimant espaces et underscores.

    Exemples
    --------
    - 'A_1' -> 'A1'
    - 'Bp 10' -> 'Bp10'

    Paramètres
    ----------
    ch : str
        Nom de canal brut.

    Retour
    ------
    str
        Nom normalisé.
    """
    s = str(ch).strip()
    s = s.replace(" ", "")
    s = s.replace("_", "")
    return s



def parse_shaft_and_contact(ch_name: str) -> Optional[Tuple[str, int]]:
    """
    Décompose un nom de canal en identifiant de plot et numéro de contact.

    Exemples
    --------
    - 'Bp1'   -> ('Bp', 1)
    - 'Bp_1'  -> ('Bp', 1)
    - "A'10"  -> ("A'", 10)
    - 'C_12'  -> ('C', 12)

    Paramètres
    ----------
    ch_name : str
        Nom de canal monopolaire.

    Retour
    ------
    Optional[Tuple[str, int]]
        Tuple (shaft, numéro_de_contact), ou None si non parseable.
    """
    s = normalize_channel_name(ch_name)
    m = re.match(r"^([A-Za-zÀ-ÿ'_-]+?)(\d+)$", s)
    if not m:
        return None
    return m.group(1), int(m.group(2))



def build_adjacent_bipolar_pairs(
    channel_names: Sequence[str],
    bad_channels: Sequence[str],
) -> List[Tuple[str, str, str]]:
    """
    Construit les paires bipolaires adjacentes valides à partir des canaux disponibles.

    Paramètres
    ----------
    channel_names : Sequence[str]
        Liste des canaux monopolaires présents dans l'enregistrement.
    bad_channels : Sequence[str]
        Liste des canaux monopolaires à exclure.

    Retour
    ------
    List[Tuple[str, str, str]]
        Liste de tuples (nom_bp, ch_a, ch_b), par exemple
        [('A1-A2', 'A1', 'A2'), ('A2-A3', 'A2', 'A3')].

    Notes
    -----
    La règle est strictement adjacente : on accepte 1-2, 2-3, 3-4, mais jamais
    3-6 si 4 et 5 sont absents. Cela évite de construire des dérivations qui ne
    respectent pas la continuité physique du plot.
    """
    bad_set = {normalize_channel_name(ch) for ch in bad_channels}  # ensemble pour tests rapides d'appartenance
    good = [normalize_channel_name(ch) for ch in channel_names if normalize_channel_name(ch) not in bad_set]

    grouped: Dict[str, List[Tuple[int, str]]] = {}
    for ch in good:
        parsed = parse_shaft_and_contact(ch)  # ex: 'A1' -> ('A', 1)
        if parsed is None:
            continue
        shaft, num = parsed
        grouped.setdefault(shaft, []).append((num, ch))  # dict: 'A' -> [(1, 'A1'), (2, 'A2'), ...]

    pairs: List[Tuple[str, str, str]] = []
    for shaft, vals in grouped.items():
        vals = sorted(vals, key=lambda x: x[0])  # tri explicite des contacts du plot par index numérique
        num_to_ch = {num: ch for num, ch in vals}  # dict: 1 -> 'A1', 2 -> 'A2', ...
        nums = sorted(num_to_ch)

        for n in nums:
            if (n + 1) in num_to_ch:  # la bipolarisation n'est permise que si le contact suivant existe réellement
                ch1 = num_to_ch[n]
                ch2 = num_to_ch[n + 1]
                new_name = f"{shaft}{n}-{shaft}{n+1}"
                pairs.append((new_name, ch1, ch2))

    return pairs



def parse_bipolar_shaft(bp_name: str) -> str:
    """
    Extrait le nom du plot à partir d'un canal bipolaire.

    Exemples
    --------
    - 'Bp1-Bp2' -> 'Bp'
    - "A'3-A'4" -> "A'"

    Paramètres
    ----------
    bp_name : str
        Nom de canal bipolaire.

    Retour
    ------
    str
        Identifiant de plot.
    """
    left = bp_name.split("-")[0]  # le nom du plot est récupéré à gauche du tiret bipolaire
    parsed = parse_shaft_and_contact(left)
    if parsed is None:
        return left
    shaft, _ = parsed
    return shaft



def group_bipolar_channels_by_shaft(bp_names: Sequence[str]) -> Dict[str, List[Tuple[int, str]]]:
    """
    Regroupe les canaux bipolaires par plot pour la visualisation.

    Paramètres
    ----------
    bp_names : Sequence[str]
        Liste ordonnée des noms de canaux bipolaires.

    Retour
    ------
    Dict[str, List[Tuple[int, str]]]
        Dictionnaire de forme :
        {
            'Bp': [(0, 'Bp1-Bp2'), (1, 'Bp2-Bp3')],
            'A':  [(2, 'A1-A2'), (3, 'A2-A3')],
        }

    Notes
    -----
    L'index global du canal dans `bp_names` est conservé pour garder l'information
    de position dans les tenseurs où l'axe canal suit cet ordre.
    """
    out: Dict[str, List[Tuple[int, str]]] = {}
    for idx, name in enumerate(bp_names):
        shaft = parse_bipolar_shaft(name)
        out.setdefault(shaft, []).append((idx, name))
    return out



def parse_stim_label_metadata(stim_label) -> Dict[str, Optional[str]]:
    """
    Extrait, à partir du label de stimulation, les infos utiles pour la condition local/distant.

    Retourne :
      - stim_bipolar_label : ex. 'Bp1-Bp2'
      - stim_shaft         : ex. 'Bp'
      - stim_contact_pair  : ex. '1-2'
      - stim_intensity     : ex. '2 mA'
      - stim_frequency     : ex. '50 Hz'

    Hypothèse :
    le stim_label contient au moins une sous-chaîne de type 'Bp1-Bp2' et idéalement
    les paramètres de stimulation sous forme '... 2mA 50Hz ...' ou proche.
    """
    s = str(stim_label).strip()

    out = {
        "stim_bipolar_label": None,
        "stim_shaft": None,
        "stim_contact_pair": None,
        "stim_intensity": None,
        "stim_frequency": None,
    }

    if not s:
        return out

    s_compact = re.sub(r"\s+", "", s)  # enlève les espaces pour rendre le parsing plus robuste

    # intensité, ex. 2mA ou 2.5mA
    intensity = s_compact[s_compact.index('mA')-3:s_compact.index('mA')]+' '+s_compact[s_compact.index('mA'):s_compact.index('mA')+2][:-3] # toujours mm nb de caracteres
    if intensity is not None:
        out["stim_intensity"] = float(intensity)

    # fréquence, ex. 50Hz
    frequency = s_compact[s_compact.index('mA')+2:s_compact.index('Hz')-2]+' '+s_compact[s_compact.index('Hz'):s_compact.index('Hz')+2][:-3]
    if frequency is not None:
        out["stim_frequency"] = float(frequency)

    # récupère la paire stimulée, ex. Bp1-Bp2
    elec_plot = s_compact[:s_compact.index('mA')-3]
    elec_stim = normalize_channel_name(re.match(r"([A-Za-z'_]+)(\d+)-([A-Za-z'_]+)(\d+)", elec_plot).group(1))
    plots_stim = re.match(r"([A-Za-z'_]+)(\d+)-([A-Za-z'_]+)(\d+)", elec_plot).group(2) +'-'+ re.match(r"([A-Za-z'_]+)(\d+)-([A-Za-z'_]+)(\d+)", elec_plot).group(4)
    
    out["stim_shaft"] = elec_stim
    out["stim_contact_pair"] = plots_stim

    return out


def add_stim_metadata_to_stims(stims_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute à la stims_table les colonnes dérivées du label de stimulation.
    """
    meta = stims_df["label_stim"].apply(parse_stim_label_metadata)
    meta_df = pd.DataFrame(list(meta))
    return pd.concat([stims_df.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)


def classify_group_and_cog_labels(cog_value: Any) -> Tuple[str, List[str]]:
    """
    Interprète le contenu de la variable cognitive et lui assigne un groupe principal.

    Paramètres
    ----------
    cog_value : Any
        Valeur issue du fichier d'événements cognitifs, potentiellement :
        - NaN
        - 'controle'
        - 'negatif'
        - "['souvenir', 'emotion']"
        - 'souvenir, emotion'

    Retour
    ------
    Tuple[str, List[str]]
        - group_label parmi {'cog+', 'controle', 'negatif', 'unknown'}
        - liste des labels cognitifs exacts si groupe 'cog+'

    Notes méthodologiques
    ---------------------
    Les règles de priorité sont explicites :
    - si au moins un vrai label cognitif est présent, l'essai est classé 'cog+' ;
    - 'controle' et 'negatif' ne sont conservés comme groupes principaux que si
      aucun label cognitif n'est présent ;
    - les cas mixtes/ambigus sont placés en 'unknown'.
    """
    if pd.isna(cog_value):
        return "unknown", []

    s = str(cog_value).strip()
    if not s:
        return "unknown", []

    labels: List[str]
    try:
        parsed = ast.literal_eval(s)  # essaie de parser une structure littérale Python
        if isinstance(parsed, (list, tuple, set)):
            labels = [str(x).strip() for x in parsed if str(x).strip()]
        else:
            labels = [str(parsed).strip()]
    except Exception:
        parts = re.split(r"[;,/]", s)  # fallback souple pour des annotations plus libres
        labels = [p.strip().strip("'\"") for p in parts if p.strip()]

    if not labels:
        labels = [s]

    labels_low = [x.lower() for x in labels]
    has_neg = any(x == "negatif" for x in labels_low)
    has_ctrl = any(x == "controle" for x in labels_low)
    cog_labels = [lab for lab, low in zip(labels, labels_low) if low not in {"negatif", "controle"}]

    if has_neg and not has_ctrl and not cog_labels:
        return "negatif", []
    if has_ctrl and not has_neg and not cog_labels:
        return "controle", []
    if cog_labels:
        return "cog+", cog_labels

    return "unknown", []


# ============================================================================
# INPUT FILE DISCOVERY AND LOADING
# ============================================================================

def list_trc_sessions(root_dir: Path) -> List[str]:
    """
    Liste les sessions disponibles à partir des fichiers *.TRC présents dans un dossier.

    Paramètres
    ----------
    root_dir : Path
        Dossier racine contenant les fichiers TRC.

    Retour
    ------
    List[str]
        Noms de session sans suffixe '.TRC'.
    """
    sessions = [fp.stem for fp in root_dir.glob("*.TRC")]
    return sorted(set(sessions))


def find_trc_parts(root_dir: Path, session: str) -> List[Path]:
    """
    Trouve les fichiers TRC correspondant à une session logique.

    Cas simple :
        session.TRC

    Cas fragmenté :
        sessiona.TRC, sessionb.TRC, sessionc.TRC...
    """
    single = root_dir / f"{session}.TRC"
    if single.exists():
        return [single]

    parts = sorted(root_dir.glob(f"{session}*.TRC"))

    if len(parts) == 0:
        raise FileNotFoundError(f"Aucun fichier TRC trouvé pour {session}")

    return parts


def read_stim_events(path: str | Path) -> pd.DataFrame:
    """
    Lit un fichier stim_events avec colonnes :
    label_stim, t_start, duration

    Compatible séparateurs tab, virgule, point-virgule ou espaces.
    """
    path = Path(path)
    df = pd.read_csv(path, sep=None, engine="python")
    expected = ["label_stim", "t_start", "duration"]
    if list(df.columns[:3]) != expected:
        # fallback si fichier sans header, car parfois noms colonnes présents, parfois absents, 
        # selon type de fichier = 3 colonnes sans nom (events TRC bruts), ou plus de colonnes, mais labellisées.
        df = pd.read_csv(
            path,
            sep=None,
            engine="python",
            header=None,
            names=expected,
        )
        df.columns[:3] = expected
    df = df.iloc[:,:3]
    df.columns = ["label_stim", "t_start", "duration"]
    df["label_stim"] = df["label_stim"].astype(str)
    df["t_start"] = pd.to_numeric(df["t_start"])
    df["duration"] = pd.to_numeric(df["duration"])
    return df


def find_trc_event_parts(root_dir: Path, session: str) -> List[Path]:
    """
    Trouve les fichiers d'events TRC bruts correspondant à une session logique.
    (puisque parfois une session est divisée en plusieurs TRC)
    """
    single = root_dir / f"{session}_stim_events_TRC.txt"
    if single.exists():
        return [single]

    parts = sorted(root_dir.glob(f"{session}*_stim_events_TRC.txt"))

    parts = [
        fp for fp in parts
        if "_shifted" not in fp.name
        and "_re-shifted" not in fp.name
        and "_corrected" not in fp.name
    ]

    if len(parts) == 0:
        raise FileNotFoundError(f"Aucun fichier events TRC brut trouvé pour {session}")

    return parts


def read_concat_trc_event_parts(root_dir: Path, session: str) -> pd.DataFrame:
    """
    Lit les fichiers events TRC bruts partiels et les concatène
    dans l'ordre macro.
    """
    event_parts = find_trc_event_parts(root_dir, session)

    dfs = []
    for part_idx, fp in enumerate(event_parts):
        df = read_stim_events(fp)

        macro_part = fp.name.replace("_stim_events_TRC.txt", "")
        df["macro_part"] = macro_part
        df["macro_part_index"] = part_idx
        df["macro_event_index"] = np.arange(len(df), dtype=int)

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def recover_precise_macro_stim_events(
    session: str,
    root_dir: Path
) -> pd.DataFrame:
    """
    Reconstruit les bornes précises des stimulations en référentiel TRC/macro,
    à partir des corrections manuelles effectuées en référentiel micro.
    Si table déjà créée/sauvegardée, la renvoie directement.

    Parameters
    ----------
    session : str
        Nom de session.
    root_dir : Path
        Dossier racine contenant TRC et tables d'événements.
    trc_events_path:
        Fichier original en référentiel TRC/macro :
        label_stim, t_start, duration

    shifted_events_path:
        Fichier TRC translaté approximativement vers le référentiel micro :
        label_stim, t_start, duration

    reshifted_events_path:
        Fichier corrigé précisément en référentiel micro :
        label_stim, t_start, duration

    Returns
    -------
    pd.DataFrame
        DataFrame avec :
        label_stim, t_start, duration, t_end,
        correction_start, correction_end
    """

    # print('in merge event tables') # ok
    trc_corrected_path = root_dir / f"{session}_stim_events_TRC_corrected.txt"
    if os.path.exists(trc_corrected_path):
        # le fichier a déjà été créé auparavant
        return pd.read_csv(trc_corrected_path,sep='\t')
    
    else : # sinon on crée le fichier, en recalant les events TRC d'apres le recalage manuel des events micro, dans le référentiel micro
        print('in trc_corrected_path not existant')  # ok
        shifted_events_path = root_dir / f"{session}_stim_events_TRC_shifted.txt"
        reshifted_events_path = root_dir / f"{session}_stim_events_TRC_re-shifted.txt"
        print(shifted_events_path, reshifted_events_path) # ok
        
        # trc_events_path = root_dir / f"{session}_stim_events_TRC.txt"
        # trc = read_stim_events(trc_events_path) # events TRC originaux (start approximatif, et sans durée)
        trc = read_concat_trc_event_parts(root_dir, session)
        shifted = read_stim_events(shifted_events_path) # events TRC translatés approximativement vers le référentiel micro (un decalage commun par rapport a une stim)
        reshifted = read_stim_events(reshifted_events_path) # events re-corrigés précisément/manuellement en référentiel micro 
        print('event files exist?') # pas ok
        # print(trc, shifted, reshifted)
        if not (len(trc) == len(shifted) == len(reshifted)):
            raise ValueError(
                "Les trois fichiers n'ont pas le même nombre de stimulations : "
                f"TRC={len(trc)}, shifted={len(shifted)}, re-shifted={len(reshifted)}"
            )

        if not (
            trc["label_stim"].values == shifted["label_stim"].values
        ).all():
            raise ValueError(
                "Les labels ne correspondent pas entre TRC et shifted. "
                "Vérifie que l'ordre des lignes est identique."
            )

        if not (
            trc["label_stim"].values == reshifted["label_stim"].values
        ).all():
            raise ValueError(
                "Les labels ne correspondent pas entre TRC et re-shifted. "
                "Vérifie que l'ordre des lignes est identique."
            )

        trc_start = trc["t_start"].to_numpy(float)
        shifted_start = shifted["t_start"].to_numpy(float)
        reshifted_start = reshifted["t_start"].to_numpy(float)

        correction_start = reshifted_start - shifted_start
        macro_precise_start = trc_start + correction_start

        trc_corr = pd.DataFrame({
            "label_stim": trc["label_stim"].values,
            "t_start": macro_precise_start,
            "duration": reshifted["duration"].to_numpy(float), # la durée est juste la plus précise possible telle qu'identifiée a la main
            "t_end": macro_precise_start + reshifted["duration"].to_numpy(float), # a partir de debut et durée précis,  on a la fin précise
            "correction_start": correction_start, 
            "macro_part": trc["macro_part"].values, # par exemple P107_SG60_stim2b
            "macro_part_index": trc["macro_part_index"].values,
            "macro_event_index": trc["macro_event_index"].values,
        })
        # print('output file', trc_corr)
        trc_corrected_path.parent.mkdir(parents=True, exist_ok=True)
        trc_corr.to_csv(trc_corrected_path, sep='\t', index=False)

        return trc_corr


def find_cog_file(root_dir: Path, session: str) -> Path:
    """
    Recherche le fichier de catégorisation cognitive d'une session.

    Paramètres
    ----------
    root_dir : Path
        Dossier racine de la session.
    session : str
        Nom de session.

    Retour
    ------
    Path
        Chemin vers le fichier COG correspondant.
    """
    matches = list(root_dir.glob(f"{session}_stim_events_TRC_re-shifted_*COG.txt"))
    if len(matches) == 0:
        raise FileNotFoundError(f"Aucun fichier COG trouvé pour {session}")
    if len(matches) > 1:
        raise FileExistsError(f"Plusieurs fichiers COG trouvés pour {session}: {[m.name for m in matches]}")
    return matches[0]


def read_cog_file(cog_file: Path) -> pd.DataFrame:
    """
    Lit le fichier des annotations cognitives d'une session.

    Format attendu
    --------------
    stim ; t_start ; duration ; (lobe) ; cog

    Paramètres
    ----------
    cog_file : Path
        Chemin vers le fichier COG.

    Retour
    ------
    pd.DataFrame
        Table avec colonnes standardisées :
        ['label_stim', 't_start', 'duration', 'lobe', 'cog']

    Notes
    -----
    Les `t_start` contenus dans ce fichier peuvent être incorrects (référentiel micro, pas macro); 
    seuls l'ordre des labels et l'annotation cognitive sont utilisés ici.
    """
    df = pd.read_csv(cog_file, sep=";", dtype=str)
    # expected = {"stim", "t_start", "duration", "lobe", "cog"}
    # missing = expected - set(df.columns)
    # if missing:
    #     raise ValueError(f"Colonnes manquantes dans {cog_file.name}: {missing}")

    df = df.rename(columns={"stim": "label_stim"})
    return df



def merge_event_tables(session: str, cog_df: pd.DataFrame, trc_corr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fusionne la table cognitive et la table de durées d'une session.

    Paramètres
    ----------
    session : str
        Nom de session.
    cog_df : pd.DataFrame
        Table d'annotations cognitives.
    trc_corr_df : pd.DataFrame
        Table des temps exacts de début et durées.

    Retour
    ------
    pd.DataFrame
        Table fusionnée avec colonnes utiles pour la suite du pipeline.

    Notes méthodologiques
    ---------------------
    - Les temps de début fiables sont pris dans `trc_corr_df`.
    - Si l'ordre des labels diffère entre les deux tables, une tentative de merge
      strict par label est effectuée.
    - `group_label` et `cog_labels` sont dérivés ici, de sorte que la table exportée
      serve ensuite de référence unique pour les stats par condition.
    """
    if len(cog_df) != len(trc_corr_df):
        raise ValueError(
            f"{session}: nombre de stimulations différent entre COG ({len(cog_df)}) et trc_corrected ({len(trc_corr_df)})"
        )

    cog_df = cog_df.reset_index(drop=True).copy()
    trc_corr_df = trc_corr_df.reset_index(drop=True).copy()

    lbl1 = cog_df["label_stim"].astype(str).str.strip().tolist()
    lbl2 = trc_corr_df["label_stim"].astype(str).str.strip().tolist()

    if lbl1 != lbl2:
        tmp = pd.merge(
            trc_corr_df,
            cog_df[["label_stim", "lobe", "cog"]],
            on="label_stim",
            how="inner",
            validate="one_to_one",
        )
        if len(tmp) != len(trc_corr_df):
            raise ValueError(f"{session}: impossible d'aligner strictement les tables événements")
        merged = tmp.copy()
    else:
        merged = trc_corr_df.copy()
        merged["lobe"] = cog_df["lobe"].values
        merged["cog"] = cog_df["cog"].values

    groups = merged["cog"].apply(classify_group_and_cog_labels)
    merged["group_label"] = groups.apply(lambda x: x[0])
    merged["cog_labels"] = groups.apply(lambda x: x[1])
    merged["stim_index"] = np.arange(len(merged), dtype=int)  # index ordonné de stimulation dans la session

    merged["t_start"] = pd.to_numeric(merged["t_start"], errors="coerce")
    merged["duration"] = pd.to_numeric(merged["duration"], errors="coerce")
    bad_rows = merged["t_start"].isna() | merged["duration"].isna()
    if bad_rows.any():
        raise ValueError(f"{session}: t_start/duration NaN dans certaines lignes après fusion")

    merged = add_stim_metadata_to_stims(merged) # ajout infos stims

    return merged



def load_bad_channels_table(root_dir: Path) -> pd.DataFrame:
    """
    Charge la table globale des canaux invalides.

    Paramètres
    ----------
    root_dir : Path
        Dossier racine contenant le fichier Excel des canaux invalides.

    Retour
    ------
    pd.DataFrame
        Table avec au minimum les colonnes 'session' et 'bad_channels'.
    """
    xlsx = root_dir / "TRC_bad_channels.xlsx"
    if not xlsx.exists():
        raise FileNotFoundError(f"Fichier bad channels introuvable: {xlsx}")

    df = pd.read_excel(xlsx)
    if not {"session", "bad_channels"}.issubset(df.columns):
        raise ValueError("TRC_bad_channels.xlsx doit contenir les colonnes 'session' et 'bad_channels'")

    df["session"] = df["session"].astype(str).str.strip()
    return df



def get_bad_channels_for_session(bad_df: pd.DataFrame, session: str) -> List[str]:
    """
    Extrait la liste des canaux invalides pour une session donnée.

    Paramètres
    ----------
    bad_df : pd.DataFrame
        Table globale des mauvais canaux.
    session : str
        Nom de session.

    Retour
    ------
    List[str]
        Liste de noms normalisés de canaux à exclure.
    """
    sub = bad_df.loc[bad_df["session"] == session]
    if len(sub) == 0:
        return []

    vals: List[str] = []
    for cell in sub["bad_channels"].tolist():
        vals.extend(parse_bad_channels_cell(cell))
    return [normalize_channel_name(x) for x in vals]


# ============================================================================
# TRC LOADING AND SIGNAL PREPARATION
# ============================================================================

def load_trc_as_mne_raw(trc_path: Path, verbose: bool = True) -> mne.io.BaseRaw:
    """
    Lit un fichier TRC Micromed et le convertit en RawArray MNE.

    Paramètres
    ----------
    trc_path : Path
        Chemin vers le fichier TRC.
    verbose : bool
        Active les logs informatifs.

    Retour
    ------
    mne.io.BaseRaw
        Objet MNE contenant les données en Volts.

    Notes méthodologiques
    ---------------------
    - Les données Micromed sont supposées être en µV ; elles sont converties en V
      car MNE attend des unités SI pour les canaux EEG/iEEG.
    - Le type de canal est fixé à 'seeg' pour refléter l'origine intracrânienne.
    """
    from micromed_io.trc import MicromedTRC  # import local pour éviter une dépendance obligatoire au chargement du module

    mmtrc = MicromedTRC(str(trc_path))
    sfreq = float(mmtrc._header["s_freq"])
    chans = mmtrc._header["chans"]
    ch_names = [normalize_channel_name(ch["chan_name"]) for ch in chans]

    if hasattr(mmtrc, "get_data"):
        data = mmtrc.get_data()
    elif hasattr(mmtrc, "data"):
        data = mmtrc.data
    else:
        raise AttributeError("Impossible de trouver les données dans l'objet MicromedTRC (ni get_data(), ni data)")

    data = np.asarray(data, dtype=np.float64)
    log(f"[INFO] {trc_path.name}: sfreq = {sfreq}", verbose)

    if data.ndim != 2:
        raise ValueError(f"Données TRC de dimension inattendue: shape={data.shape}")

    if data.shape[0] != len(ch_names) and data.shape[1] == len(ch_names):
        data = data.T  # transpose si l'API a renvoyé (n_times, n_channels)

    if data.shape[0] != len(ch_names):
        raise ValueError(
            f"Incohérence données/canaux pour {trc_path.name}: data.shape={data.shape}, n_channels={len(ch_names)}"
        )

    units = [str(ch.get("units", "")).strip() for ch in chans]
    unique_units = sorted(set(units))
    log(f"[INFO] {trc_path.name}: unités canal détectées = {unique_units}", verbose)

    data_volts = data * 1e-6  # conversion µV -> V
    ch_types = ["seeg"] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    return mne.io.RawArray(data_volts, info, verbose=False)


def apply_filters(
    data: np.ndarray,
    sfreq: float,
    do_notch: bool = True,
    notch_freqs: Sequence[float] = (50.0, 100.0, 150.0),
    notch_q: float = 30.0,
    do_highpass: bool = True,
    highpass_hz: float = 0.1,
) -> np.ndarray:
    """
    Applique un filtrage temporel zero-phase aux signaux monopolaires.

    Paramètres
    ----------
    data : np.ndarray
        Signal de forme (n_channels, n_samples).
    sfreq : float
        Fréquence d'échantillonnage en Hz.
    do_notch : bool
        Active les notch successifs sur les fréquences secteur.
    notch_freqs : Sequence[float]
        Fréquences des notch à appliquer.
    notch_q : float
        Facteur Q des notch ; plus il est grand, plus la coupure est étroite.
    do_highpass : bool
        Active le passe-haut.
    highpass_hz : float
        Fréquence de coupure du passe-haut.

    Retour
    ------
    np.ndarray
        Signal filtré, de même forme que l'entrée.

    Notes
    -----
    Le filtrage est fait avant bipolarisation, ce qui permet d'appliquer le même
    prétraitement à tous les contacts monopolaires et d'éviter de dériver des signaux
    déjà combinés de façon hétérogène.
    """
    x = np.asarray(data, dtype=np.float64).copy()

    if do_notch:
        for f0 in notch_freqs:
            if f0 >= sfreq / 2:
                continue  # on ignore les notch au-delà de Nyquist
            b, a = signal.iirnotch(w0=f0, Q=notch_q, fs=sfreq)
            x = signal.filtfilt(b, a, x, axis=-1)  # zero-phase pour éviter un décalage temporel

    if do_highpass and highpass_hz is not None and highpass_hz > 0:
        sos = signal.butter(N=4, Wn=highpass_hz, btype="highpass", fs=sfreq, output="sos")
        x = signal.sosfiltfilt(sos, x, axis=-1)  # version SOS plus stable numériquement

    return x



def make_bipolar_data(
    data: np.ndarray,
    ch_names: Sequence[str],
    bipolar_pairs: Sequence[Tuple[str, str, str]],
) -> Tuple[np.ndarray, List[str]]:
    """
    Construit les signaux bipolaires à partir des signaux monopolaires filtrés.

    Paramètres
    ----------
    data : np.ndarray
        Données monopolaires de forme (n_channels, n_samples).
    ch_names : Sequence[str]
        Noms des canaux monopolaires dans le même ordre que `data`.
    bipolar_pairs : Sequence[Tuple[str, str, str]]
        Liste des canaux bipolaires à construire sous forme
        [(nom_bp, ch_a, ch_b), ...].

    Retour
    ------
    Tuple[np.ndarray, List[str]]
        - Array bipolaire de forme (n_bp_channels, n_samples)
        - Liste des noms des canaux bipolaires, dans le même ordre que l'array
    """
    ch_to_idx = {normalize_channel_name(ch): i for i, ch in enumerate(ch_names)}

    bp_data: List[np.ndarray] = []
    bp_names: List[str] = []
    for new_name, ch_a, ch_b in bipolar_pairs:
        key_a = normalize_channel_name(ch_a)
        key_b = normalize_channel_name(ch_b)
        if key_a not in ch_to_idx or key_b not in ch_to_idx:
            continue
        i = ch_to_idx[key_a]
        j = ch_to_idx[key_b]
        bp_data.append(data[i] - data[j])  # dérivation bipolaire adjacent
        bp_names.append(new_name)

    if len(bp_data) == 0:
        return np.empty((0, data.shape[1])), []

    return np.asarray(bp_data, dtype=np.float64), bp_names


# ============================================================================
# WINDOWING AND EPOCHING
# ============================================================================

def compute_stim_windows(row: pd.Series, pre_length: float, post_length: float, epsilon: float) -> Dict[str, float]:
    """
    Calcule les fenêtres temporelles pré- et post-stimulation pour une ligne d'essai.

    Paramètres
    ----------
    row : pd.Series
        Ligne de la table d'essais contenant au minimum `t_start` et `duration`.
    pre_length : float
        Durée de la fenêtre pré-stimulation en secondes.
    post_length : float
        Durée de la fenêtre post-stimulation en secondes.
    epsilon : float
        Marge temporelle de sécurité de part et d'autre de la stimulation.

    Retour
    ------
    Dict[str, float]
        Dictionnaire contenant :
        pre_start, pre_end, post_start, post_end, stim_start, stim_end

    Notes méthodologiques
    ---------------------
    La fenêtre post commence après `stim_end + epsilon` afin de cibler la dynamique
    post-stimulation plutôt que l'artefact électrique direct.
    """
    t_start = float(row["t_start"])
    dur = float(row["duration"])
    t_end = t_start + dur

    pre_start = t_start - pre_length - epsilon
    pre_end = t_start - epsilon
    post_start = t_end + epsilon
    post_end = post_start + post_length

    return {
        "pre_start": pre_start,
        "pre_end": pre_end,
        "post_start": post_start,
        "post_end": post_end,
        "stim_start": t_start,
        "stim_end": t_end,
    }



def add_windows_to_stims(stims_df: pd.DataFrame, pre_length: float, post_length: float, epsilon: float) -> pd.DataFrame:
    """
    Ajoute les bornes des fenêtres pré/post à la table d'essais.

    Paramètres
    ----------
    stims_df : pd.DataFrame
        Table d'essais fusionnée.
    pre_length : float
        Durée de la fenêtre pré-stimulation.
    post_length : float
        Durée de la fenêtre post-stimulation.
    epsilon : float
        Marge de sécurité autour de la stimulation.

    Retour
    ------
    pd.DataFrame
        Copie de la table d'origine enrichie des colonnes de fenêtres.
    """
    out = stims_df.copy()
    windows = out.apply(compute_stim_windows, axis=1, pre_length=pre_length, post_length=post_length, epsilon=epsilon)
    wdf = pd.DataFrame(list(windows))
    return pd.concat([out.reset_index(drop=True), wdf.reset_index(drop=True)], axis=1)



def keep_stims_fitting_signal(stims_df: pd.DataFrame, signal_duration_s: float, verbose: bool = True) -> pd.DataFrame:
    """
    Exclut les essais dont les fenêtres pré ou post débordent du signal disponible.

    Paramètres
    ----------
    stims_df : pd.DataFrame
        Table d'essais avec bornes des fenêtres.
    signal_duration_s : float
        Durée totale du signal en secondes.
    verbose : bool
        Active les logs.

    Retour
    ------
    pd.DataFrame
        Table filtrée ne contenant que les essais entièrement extrayables.
    """
    good = (stims_df["pre_start"] >= 0) & (stims_df["post_end"] <= signal_duration_s)
    dropped = int((~good).sum())
    if dropped > 0:
        log(f"[INFO] {dropped} stimulations exclues car fenêtres hors signal", verbose)
    return stims_df.loc[good].reset_index(drop=True)



def extract_pre_post_epochs(
    data_bp: np.ndarray,
    sfreq: float,
    stims_df: pd.DataFrame,
    pre_length: float,
    post_length: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extrait les tenseurs d'époques pré- et post-stimulation à partir du signal bipolaire.

    Paramètres
    ----------
    data_bp : np.ndarray
        Signal bipolaire de forme (n_channels, n_samples).
    sfreq : float
        Fréquence d'échantillonnage en Hz.
    stims_df : pd.DataFrame
        Table d'essais avec colonnes 'pre_start' et 'post_start'.
    pre_length : float
        Durée de la fenêtre pré en secondes.
    post_length : float
        Durée de la fenêtre post en secondes.

    Retour
    ------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - pre_epochs  : (n_stims, n_channels, n_pre_samples)
        - post_epochs : (n_stims, n_channels, n_post_samples)
        - pre_times   : axe temps relatif pré, en secondes
        - post_times  : axe temps relatif post, en secondes

    Notes
    -----
    Les essais dont l'extraction réelle ne donne pas le nombre exact d'échantillons
    attendu sont ignorés par sécurité.
    """
    n_pre = int(round(pre_length * sfreq))
    n_post = int(round(post_length * sfreq))

    pre_epochs: List[np.ndarray] = []
    post_epochs: List[np.ndarray] = []

    for _, row in stims_df.iterrows():
        pre_start_idx = int(round(row["pre_start"] * sfreq))
        pre_end_idx = pre_start_idx + n_pre
        post_start_idx = int(round(row["post_start"] * sfreq))
        post_end_idx = post_start_idx + n_post

        pre_ep = data_bp[:, pre_start_idx:pre_end_idx]
        post_ep = data_bp[:, post_start_idx:post_end_idx]

        if pre_ep.shape[1] != n_pre or post_ep.shape[1] != n_post:
            continue

        pre_epochs.append(pre_ep)
        post_epochs.append(post_ep)

    pre_epochs_arr = np.asarray(pre_epochs, dtype=np.float64)
    post_epochs_arr = np.asarray(post_epochs, dtype=np.float64)
    pre_times = np.arange(n_pre) / sfreq - pre_length
    post_times = np.arange(n_post) / sfreq

    return pre_epochs_arr, post_epochs_arr, pre_times, post_times



def build_global_baseline_segment(data_bp: np.ndarray, sfreq: float, first_stim_t_start: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construit un segment de baseline globale allant du début du signal à la première stimulation.

    Paramètres
    ----------
    data_bp : np.ndarray
        Signal bipolaire de forme (n_channels, n_samples).
    sfreq : float
        Fréquence d'échantillonnage en Hz.
    first_stim_t_start : float
        Temps de début de la première stimulation de la session.

    Retour
    ------
    Tuple[np.ndarray, np.ndarray]
        - segment de baseline globale, forme (n_channels, n_samples_baseline)
        - axe temps absolu correspondant, en secondes
    """
    stop_idx = int(round(first_stim_t_start * sfreq))
    stop_idx = max(stop_idx, 1)  # force une longueur minimale de 1 échantillon
    seg = data_bp[:, :stop_idx]
    times = np.arange(seg.shape[1]) / sfreq
    return seg, times

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

    # Helpers généraux
    "log",
    "ensure_dir",
    "safe_name",
    "make_freqs",
    "parse_list_cell",
    "reduce_baseline_stat",
    "compute_diverging_limits",

    # Canaux / événements
    "parse_bad_channels_cell",
    "normalize_channel_name",
    "parse_shaft_and_contact",
    "build_adjacent_bipolar_pairs",
    "parse_bipolar_shaft",
    "group_bipolar_channels_by_shaft",
    "parse_stim_label_metadata",
    "add_stim_metadata_to_stims",
    "classify_group_and_cog_labels",

    # Entrées / chargements
    "list_trc_sessions",
    "find_trc_parts",
    "read_stim_events",
    "find_trc_event_parts",
    "read_concat_trc_event_parts",
    "recover_precise_macro_stim_events",
    "find_cog_file",
    # "find_duration_file",
    "read_cog_file",
    # "read_duration_file",
    "merge_event_tables",
    "load_bad_channels_table",
    "get_bad_channels_for_session",
    # "load_session_exports",

    # Signal loading / filtering / montage
    "load_trc_as_mne_raw",
    "apply_filters",
    "make_bipolar_data",

    # Epoching
    "compute_stim_windows",
    "add_windows_to_stims",
    "keep_stims_fitting_signal",
    "extract_pre_post_epochs",
    "build_global_baseline_segment",

]
