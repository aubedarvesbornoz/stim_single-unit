#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spike_lfp_common_preprocess.py
==============================

Base commune de prétraitement pour les analyses spike–LFP après SEIC.

Objectif
--------
Ce module construit une représentation commune et robuste d'une session micro/macro :

1. lecture robuste des fichiers d'événements micro (COG) et macro (SESSION_stim_events_TRC_corrected) ;
2. parsing stable des labels de stimulation ;
3. fusion d'une table maître trial-level contenant temps start et end, en micro et macro ;
4. estimation QC optionnelle du décalage constant micro ↔ macro
(Les temps micro et macro restent stockés séparément. Le module ne transforme pas
systématiquement les temps micro en temps macro. L'offset sert uniquement au QC
et, plus tard, à échantillonner le LFP au temps des spikes si nécessaire);
5. ajout de fenêtres pré/post dans les deux référentiels ;
6. chargement des exports Hilbert existants, sans recalculer le LFP ;
7. helpers génériques pour sélectionner des spikes, des fenêtres et des canaux;
8. table trial × channel avec localité local/distant.

Ce fichier est indépendant des analyses finales : il ne calcule ni
corrélations spike-power, ni phase-locking. Il fournit les objets communs qui seront
réutilisés par ces deux pipelines.

Entrées principales
-------------------
Micro :
    micro_root / Data_folders / PATIENT / PATIENT_stimSESSION / PATIENT_stimN_stim_events_TRC_re-shifted_loca_COG.txt
ou :
    micro_root / Spike-sorting / Data_folders / PATIENT / PATIENT_stimN / ...

Macro/Hilbert :
    macro_root/
    └── PATIENT_stimN_stim_events_TRC_corrected.txt

Hilbert exports :
    hilbert_root/
    └── PATIENT_stimN/
        ├── PATIENT_stimN_metadata.json
        ├── PATIENT_stimN_trial_table.csv
        ├── PATIENT_stimN_times.npy
        └── PATIENT_stimN_hilbert_<band>.npy {theta, alpha, beta, low_gamma}

(Pour les analyses spike ultérieures, il faudra aussi :

PATIENT_stimN.nwb
    ou fichiers Neuroscope nécessaires pour le créer :
        PATIENT_stimN.xml
        PATIENT_stimN.dat
        .clu.*
        .res.*
        .fet.*...
PATIENT/mapping_anat_PATIENT.txt
PATIENT/deadCh_PATIENT.txt                         optionnel
PATIENT/PATIENT_stimN/derivatives/*deadfile*       recommandé)

Sorties
-------
    common_root/SESSION/SESSION_common_trials.csv
    common_root/SESSION/SESSION_common_metadata.json

Auteure du projet : Aube Darves-Bornoz
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION / TYPES
# =============================================================================

EventMatchStrategy = Literal["auto", "label", "order"]


@dataclass
class CommonPreprocessConfig:
    """Configuration minimale de la base commune spike-LFP."""

    # Racines des deux bases.
    micro_root: str
    macro_root: str
    hilbert_root: Optional[str] = None

    # Fenêtres communes autour des stimulations.
    pre_length: float = 3.0
    post_length: float = 3.0
    epsilon: float = 0.1

    # Fusion des événements micro/macro.
    event_match_strategy: EventMatchStrategy = "auto"
    require_same_n_events: bool = True
    max_label_mismatch_fraction_for_order_fallback: float = 1.0

    # Décalage temporel micro↔macro : QC uniquement, pas utilisé pour transformer les temps.
    # Les analyses utilisent directement t_*_micro pour les spikes et t_*_macro pour le LFP.
    compute_micro_macro_offset_qc: bool = True
    offset_outlier_mad_thresh: float = 6.0

    # Contrôles de qualité génériques.
    min_reference_duration_s: float = 1.0
    verbose: bool = True


@dataclass
class OffsetMapping:
    """Décalage constant entre référentiels : t_macro ≈ t_micro + offset_s.

    Ce n'est pas une transformation imposée aux données : c'est un contrôle qualité
    de cohérence entre les temps de stimulation micro et macro. Les fenêtres micro
    et macro restent calculées depuis leurs colonnes propres.
    """

    offset_s: float
    n_points: int
    n_used: int
    residual_median_s: float
    residual_mad_s: float
    residual_max_abs_s: float
    used_mask: List[bool]

    def micro_to_macro(self, t_micro: np.ndarray | float) -> np.ndarray | float:
        return np.asarray(t_micro) + self.offset_s

    def macro_to_micro(self, t_macro: np.ndarray | float) -> np.ndarray | float:
        return np.asarray(t_macro) - self.offset_s


@dataclass
class CommonSessionBundle:
    """Objet léger regroupant les sorties communes d'une session."""

    patient: str
    session_num: str
    session_name: str
    trials: pd.DataFrame
    offset: Optional[OffsetMapping]
    paths: Dict[str, Optional[str]]
    hilbert: Optional[Dict[str, Any]] = None


# =============================================================================
# PETITS HELPERS
# =============================================================================


def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg, flush=True)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", str(name))


def normalize_channel_name(ch: str) -> str:
    """Version locale compatible avec lfp_preprocess.normalize_channel_name."""
    return str(ch).strip().replace(" ", "").replace("_", "")


def parse_patient_session(session_name: Optional[str] = None,
                          patient: Optional[str] = None,
                          session: Optional[str | int] = None) -> Tuple[str, str, str]:
    """
    Normalise l'identité de session.

    Exemples
    --------
    parse_patient_session('P119_FM71_stim4') -> ('P119_FM71', '4', 'P119_FM71_stim4')
    parse_patient_session(patient='P119_FM71', session=4) -> idem
    """
    if session_name is not None:
        m = re.match(r"^(?P<patient>.+?)_stim(?P<session>\d+[A-Za-z]?)$", str(session_name))
        if not m:
            raise ValueError(f"session_name non reconnu: {session_name}")
        return m.group("patient"), str(m.group("session")), str(session_name)

    if patient is None or session is None:
        raise ValueError("Fournir soit session_name, soit patient + session")

    session_str = str(session).replace("stim", "")
    return str(patient), session_str, f"{patient}_stim{session_str}"


def _to_float_or_nan(x: Any) -> float:
    if x is None:
        return np.nan
    s = str(x).strip().replace(",", ".")
    if s == "" or s.lower() in {"nan", "none"}:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


# =============================================================================
# PARSING ROBUSTE DES LABELS / COG
# =============================================================================

def normalize_stim_label_for_matching(label: Any) -> str:
    """
    Normalise un label pour comparer micro et macro sans être trop sensible aux
    espaces, underscores ou casse.
    """
    s = str(label).strip()
    s = s.replace(" ", "")
    s = s.replace("_", "")
    s = s.replace(",", ".")
    return s.lower()


def parse_stim_label_robust(label: Any) -> Dict[str, Any]:
    """
    Parse un label du type :
        CU_1-CU_22.0mA1.0Hz1025µsec
        Tp3-Tp42.0mA7.0Hz1025µsec
    Retourne des champs stables pour les analyses topographiques et les stats.
    """
    s = str(label).strip()
    
    intensity = s[s.index('mA')-3:s.index('mA')]+' '+s[s.index('mA'):s.index('mA')+2] # toujours mm nb de caracteres
    freq = s[s.index('mA')+2:s.index('Hz')-2]+' '+s[s.index('Hz'):s.index('Hz')+2]  
    elec_plot = s[:s.index('mA')-3]
    elec = re.match(r"([A-Za-z'_]+)(\d+)-([A-Za-z'_]+)(\d+)", elec_plot).group(1)
    plot1 = re.match(r"([A-Za-z'_]+)(\d+)-([A-Za-z'_]+)(\d+)", elec_plot).group(2)
    plot2 = re.match(r"([A-Za-z'_]+)(\d+)-([A-Za-z'_]+)(\d+)", elec_plot).group(4)

    # out: Dict[str, Any] = {
    #     "label_stim_raw": s,
    #     "label_stim_key": normalize_stim_label_for_matching(s),
    #     "stim_bipolar_label": None,
    #     "stim_shaft": None,
    #     "stim_contact_pair": None,
    #     "stim_contact_1": np.nan,
    #     "stim_contact_2": np.nan,
    #     "stim_intensity": np.nan,
    #     "stim_frequency": np.nan,
    #     "stim_pulse_width_us": np.nan,
    #     "stim_label_parse_ok": False,
    # }
    out = {
        "label_stim_raw": s,
        "label_stim_key": normalize_stim_label_for_matching(s),
        "stim_bipolar_label": elec_plot,
        "stim_shaft": elec,
        "stim_contact_pair": plot1+'-'+plot2,
        "stim_contact_1": plot1,
        "stim_contact_2": plot2,
        "stim_intensity": intensity,
        "stim_frequency": freq,
    }
    # if not s:
    #     return out

    # shaft1 = normalize_channel_name(m.group("shaft1"))
    # shaft2 = normalize_channel_name(m.group("shaft2"))
    # c1 = int(m.group("c1"))
    # c2 = int(m.group("c2"))

    # out.update({
    #     "stim_bipolar_label": f"{shaft1}{c1}-{shaft2}{c2}",
    #     "stim_shaft": shaft1 if shaft1 == shaft2 else f"{shaft1}|{shaft2}",
    #     "stim_contact_pair": f"{c1}-{c2}",
    #     "stim_contact_1": c1,
    #     "stim_contact_2": c2,
    #     "stim_intensity": _to_float_or_nan(m.group("intensity")),
    #     "stim_frequency": _to_float_or_nan(m.group("frequency")),
    #     "stim_pulse_width_us": _to_float_or_nan(m.group("pulse_width")),
    #     "stim_label_parse_ok": True,
    # })
    return out


def parse_list_cell(value: Any) -> List[str]:
    """Décompose une cellule contenant NaN, liste Python sérialisée ou labels séparés."""
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []

    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple, set)):
            return [str(v).strip() for v in obj if str(v).strip()]
        if obj is None:
            return []
        return [str(obj).strip()] if str(obj).strip() else []
    except Exception:
        parts = re.split(r"[;,/]", s)
        return [p.strip().strip("'\"") for p in parts if p.strip()]


def classify_group_and_cog_labels(cog_value: Any) -> Tuple[str, List[str]]:
    """
    Classe une annotation cognitive en groupe principal.

    Règles :
    - vide / NaN -> unknown ;
    - ['negatif'] ou negatif -> negatif ;
    - ['controle'] ou controle -> controle ;
    - tout autre label non vide -> cog+ ;
    - label mixte avec cog réel -> cog+.
    """
    labels = parse_list_cell(cog_value)
    if not labels:
        return "unknown", []

    labels_low = [x.lower() for x in labels]
    has_neg = any(x == "negatif" for x in labels_low)
    has_ctrl = any(x == "controle" for x in labels_low)
    cog_labels = [lab for lab, low in zip(labels, labels_low) if low not in {"negatif", "controle"}]

    if cog_labels:
        return "cog+", cog_labels
    if has_neg and not has_ctrl:
        return "negatif", []
    if has_ctrl and not has_neg:
        return "controle", []
    return "unknown", []


def add_stim_and_cog_metadata(df: pd.DataFrame,
                              label_col: str = "label_stim",
                              cog_col: str = "cog") -> pd.DataFrame:
    """Ajoute parsing stim + group_label/cog_labels à une table d'événements."""
    out = df.copy().reset_index(drop=True)
    meta = pd.DataFrame([parse_stim_label_robust(x) for x in out[label_col]])
    out = pd.concat([out, meta], axis=1)

    if cog_col in out.columns:
        groups = out[cog_col].apply(classify_group_and_cog_labels)
        out["group_label"] = groups.apply(lambda x: x[0])
        out["cog_labels"] = groups.apply(lambda x: x[1])
    else:
        out["group_label"] = "unknown"
        out["cog_labels"] = [[] for _ in range(len(out))]
    return out


# =============================================================================
# LECTURE ROBUSTE DES TABLES D'ÉVÉNEMENTS
# =============================================================================


def _read_text_lines(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return [ln.rstrip("\n\r") for ln in f if ln.strip()]


def read_cog_file_robust(path: str | Path) -> pd.DataFrame:
    """
    Lit un fichier COG même si le séparateur réel varie.

    Formats acceptés :
    1. vrai CSV/TSV/semicolon avec colonnes stim;t_start;duration;lobe;cog ;
    2. lignes mixtes du type :
       CU_1-CU_22.0mA1.0Hz1025µsec,1036.82,8.97839,R Occipital;['aphasie']
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # 1) Tentative pandas souple, utile pour fichiers propres.
    for sep in [";", "\t", None, ","]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", dtype=str)
            cols_low = [str(c).strip().lower() for c in df.columns]
            if {"stim", "t_start", "duration"}.issubset(set(cols_low)) or {"label_stim", "t_start", "duration"}.issubset(set(cols_low)):
                rename = {}
                for c in df.columns:
                    cl = str(c).strip().lower()
                    if cl == "stim":
                        rename[c] = "label_stim"
                    elif cl in {"label_stim", "t_start", "duration", "lobe", "cog"}:
                        rename[c] = cl
                df = df.rename(columns=rename)
                if "label_stim" in df.columns and "t_start" in df.columns and "duration" in df.columns:
                    if "lobe" not in df.columns:
                        df["lobe"] = np.nan
                    if "cog" not in df.columns:
                        df["cog"] = np.nan
                    out = df[["label_stim", "t_start", "duration", "lobe", "cog"]].copy()
                    out["t_start"] = pd.to_numeric(out["t_start"], errors="coerce")
                    out["duration"] = pd.to_numeric(out["duration"], errors="coerce")
                    if out["t_start"].notna().any():
                        return out
        except Exception:
            pass

    # 2) Parser manuel pour lignes mixtes.
    lines = _read_text_lines(path)
    if not lines:
        raise ValueError(f"Fichier vide: {path}")

    # Ignore un header éventuel.
    data_lines = lines[1:] if re.search(r"stim.*t_start.*duration", lines[0], flags=re.I) else lines
    rows: List[Dict[str, Any]] = []

    mixed_re = re.compile(
        r"^(?P<label>.*?),(?P<t_start>-?\d+(?:[\.,]\d+)?),(?P<duration>-?\d+(?:[\.,]\d+)?),(?P<rest>.*)$"
    )
    for ln in data_lines:
        ln = ln.strip()
        if not ln:
            continue

        # Cas semicolon propre sans pandas.
        if ln.count(";") >= 3 and "," not in ln.split(";")[0]:
            parts = ln.split(";")
            label = parts[0]
            t_start = parts[1] if len(parts) > 1 else np.nan
            duration = parts[2] if len(parts) > 2 else np.nan
            lobe = parts[3] if len(parts) > 3 else np.nan
            cog = parts[4] if len(parts) > 4 else np.nan
        else:
            m = mixed_re.match(ln)
            if m is None:
                raise ValueError(f"Ligne COG non parseable dans {path.name}: {ln}")
            label = m.group("label")
            t_start = m.group("t_start")
            duration = m.group("duration")
            rest = m.group("rest")
            if ";" in rest:
                lobe, cog = rest.split(";", 1)
            else:
                lobe, cog = rest, np.nan

        rows.append({
            "label_stim": str(label).strip(),
            "t_start": _to_float_or_nan(t_start),
            "duration": _to_float_or_nan(duration),
            "lobe": str(lobe).strip() if not pd.isna(lobe) else np.nan,
            "cog": str(cog).strip() if not pd.isna(cog) and str(cog).strip() != "" else np.nan,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"Aucune ligne COG lue dans {path}")
    return out


def read_corrected_macro_events_robust(path: str | Path) -> pd.DataFrame:
    """
    Lit SESSION_stim_events_TRC_corrected.txt avec ou sans header.

    Colonnes retournées au minimum :
        label_stim, t_start, duration, t_end, correction_start,
        macro_part, macro_part_index, macro_event_index
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # Le format attendu est TSV avec header. sep=None marche aussi si espaces/tabs.
    try:
        df = pd.read_csv(path, sep="\t", engine="python")
        if "label_stim" not in df.columns:
            raise ValueError
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python", header=None)
        expected = [
            "label_stim", "t_start", "duration", "t_end", "correction_start",
            "macro_part", "macro_part_index", "macro_event_index",
        ]
        df = df.iloc[:, : min(df.shape[1], len(expected))].copy()
        df.columns = expected[: df.shape[1]]
        # Si le header a été lu comme première ligne, on l'enlève.
        if len(df) > 0 and str(df.iloc[0, 0]).strip().lower() == "label_stim":
            df = df.iloc[1:].reset_index(drop=True)

    if "label_stim" not in df.columns or "t_start" not in df.columns or "duration" not in df.columns:
        raise ValueError(f"Colonnes minimales absentes dans {path}")

    out = df.copy()
    out["label_stim"] = out["label_stim"].astype(str).str.strip()
    for col in ["t_start", "duration", "t_end", "correction_start"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "t_end" not in out.columns or out["t_end"].isna().all():
        out["t_end"] = out["t_start"] + out["duration"]
    if "correction_start" not in out.columns:
        out["correction_start"] = np.nan
    if "macro_part" not in out.columns:
        out["macro_part"] = np.nan
    if "macro_part_index" not in out.columns:
        out["macro_part_index"] = np.nan
    if "macro_event_index" not in out.columns:
        out["macro_event_index"] = np.arange(len(out), dtype=int)

    bad = out["t_start"].isna() | out["duration"].isna()
    if bad.any():
        raise ValueError(f"Lignes macro invalides dans {path}: indices {np.where(bad)[0].tolist()}")
    return out.reset_index(drop=True)


# =============================================================================
# DÉCOUVERTE DES FICHIERS MICRO/MACRO/HILBERT
# =============================================================================


def find_micro_session_dir(micro_root: str | Path,
                           patient: str,
                           session_num: str,
                           session_name: Optional[str] = None) -> Path:
    """Recherche le dossier session micro dans plusieurs structures compatibles."""
    root = Path(micro_root)
    session_name = session_name or f"{patient}_stim{session_num}"
    candidates = [
        root / "Data_folders" / patient / session_name,
        root / "Spike-sorting" / "Data_folders" / patient / session_name,
        root / patient / session_name,
        root / session_name,
    ]
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            return cand
    raise FileNotFoundError(
        "Dossier micro introuvable. Candidats testés:\n" + "\n".join(str(c) for c in candidates)
    )


def find_micro_cog_file(micro_session_dir: str | Path, session_name: str) -> Path:
    """Trouve le fichier COG dans le dossier micro de session."""
    d = Path(micro_session_dir)
    patterns = [
        f"{session_name}_stim_events_TRC_re-shifted_loca_COG.txt",
        f"{session_name}_stim_events_TRC_re-shifted_*COG.txt",
        f"{session_name}_stim_events_TRC_shifted_loca_COG.txt",
        f"{session_name}_stim_events_TRC_shifted_*COG.txt",
        "*stim_events_TRC_re-shifted_loca_COG.txt",
        "*stim_events_TRC_re-shifted_*COG.txt",
        "*stim_events_TRC_shifted_loca_COG.txt",
    ]
    matches: List[Path] = []
    for pat in patterns:
        matches.extend(sorted(d.glob(pat)))
    matches = sorted(set(matches))
    if len(matches) == 0:
        raise FileNotFoundError(f"Aucun fichier COG trouvé dans {d}")
    if len(matches) > 1:
        # On privilégie exactement le nom attendu si présent.
        exact = d / f"{session_name}_stim_events_TRC_re-shifted_loca_COG.txt"
        if exact in matches:
            return exact
        raise FileExistsError(f"Plusieurs fichiers COG possibles dans {d}: {[m.name for m in matches]}")
    return matches[0]


def find_macro_corrected_file(macro_root: str | Path, session_name: str) -> Path:
    fp = Path(macro_root) / f"{session_name}_stim_events_TRC_corrected.txt"
    if not fp.exists():
        raise FileNotFoundError(fp)
    return fp


def find_macro_cog_file(macro_root: str | Path, session_name: str) -> Path:
    root = Path(macro_root)
    matches = sorted(root.glob(f"{session_name}_stim_events_TRC_re-shifted_*COG.txt"))
    if len(matches) == 0:
        # fallback pour certains dossiers où le fichier COG n'est que côté micro.
        raise FileNotFoundError(f"Aucun fichier COG macro trouvé pour {session_name} dans {root}")
    if len(matches) > 1:
        raise FileExistsError(f"Plusieurs fichiers COG macro trouvés: {[m.name for m in matches]}")
    return matches[0]


def find_hilbert_session_dir(hilbert_root: str | Path, session_name: str) -> Path:
    d = Path(hilbert_root) / session_name
    if not d.exists():
        raise FileNotFoundError(d)
    return d


# =============================================================================
# TABLES MICRO / MACRO / MASTER TRIALS
# =============================================================================


def load_micro_trial_table(micro_root: str | Path,
                           patient: str,
                           session_num: str,
                           session_name: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Charge la table micro trial-level depuis le fichier COG du dossier micro."""
    session_name = session_name or f"{patient}_stim{session_num}"
    micro_dir = find_micro_session_dir(micro_root, patient, session_num, session_name)
    cog_file = find_micro_cog_file(micro_dir, session_name)
    df = read_cog_file_robust(cog_file)

    out = add_stim_and_cog_metadata(df, label_col="label_stim", cog_col="cog")
    out = out.rename(columns={
        "t_start": "t_start_micro",
        "duration": "duration_micro",
    })
    out["t_end_micro"] = out["t_start_micro"] + out["duration_micro"]
    out["micro_event_index"] = np.arange(len(out), dtype=int)
    out["micro_cog_file"] = str(cog_file)

    return out.reset_index(drop=True), {"micro_session_dir": str(micro_dir), "micro_cog_file": str(cog_file)}


def load_macro_trial_table(macro_root: str | Path,
                           session_name: str,
                           allow_missing_macro_cog: bool = True) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    """
    Charge la table macro trial-level depuis le corrected, puis enrichit avec le COG
    macro si disponible.
    """
    corrected_file = find_macro_corrected_file(macro_root, session_name)
    corr = read_corrected_macro_events_robust(corrected_file)

    cog_file: Optional[Path] = None
    try:
        cog_file = find_macro_cog_file(macro_root, session_name)
        cog = read_cog_file_robust(cog_file)
    except FileNotFoundError:
        if not allow_missing_macro_cog:
            raise
        cog = pd.DataFrame({
            "label_stim": corr["label_stim"].values,
            "t_start": np.nan,
            "duration": np.nan,
            "lobe": np.nan,
            "cog": np.nan,
        })

    # Fusion macro corrected + COG : les temps fiables viennent du corrected.
    if len(cog) == len(corr):
        tmp = corr.copy().reset_index(drop=True)
        tmp["lobe"] = cog["lobe"].values if "lobe" in cog.columns else np.nan
        tmp["cog"] = cog["cog"].values if "cog" in cog.columns else np.nan
    else:
        key_corr = corr["label_stim"].map(normalize_stim_label_for_matching)
        key_cog = cog["label_stim"].map(normalize_stim_label_for_matching)
        corr2 = corr.assign(_key=key_corr)
        cog2 = cog.assign(_key=key_cog)[["_key", "lobe", "cog"]]
        tmp = pd.merge(corr2, cog2, on="_key", how="left", validate="one_to_one").drop(columns="_key")

    out = add_stim_and_cog_metadata(tmp, label_col="label_stim", cog_col="cog")
    out = out.rename(columns={
        "t_start": "t_start_macro",
        "duration": "duration_macro",
        "t_end": "t_end_macro",
    })
    out["macro_corrected_file"] = str(corrected_file)
    out["macro_cog_file"] = str(cog_file) if cog_file is not None else None

    return out.reset_index(drop=True), {
        "macro_corrected_file": str(corrected_file),
        "macro_cog_file": str(cog_file) if cog_file is not None else None,
    }


def _merge_by_label(micro: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    micro2 = micro.copy().assign(_stim_key=micro["label_stim"].map(normalize_stim_label_for_matching))
    macro2 = macro.copy().assign(_stim_key=macro["label_stim"].map(normalize_stim_label_for_matching))
    merged = pd.merge(
        macro2,
        micro2,
        on="_stim_key",
        how="inner",
        suffixes=("_macro_src", "_micro_src"),
        validate="one_to_one",
    )
    if len(merged) != len(macro) or len(merged) != len(micro):
        raise ValueError(f"Fusion par label incomplète: macro={len(macro)}, micro={len(micro)}, merged={len(merged)}")
    return merged.drop(columns="_stim_key")


def _merge_by_order(micro: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    if len(micro) != len(macro):
        raise ValueError(f"Fusion par ordre impossible: micro={len(micro)}, macro={len(macro)}")
    macro2 = macro.copy().reset_index(drop=True).add_suffix("_macro_src")
    micro2 = micro.copy().reset_index(drop=True).add_suffix("_micro_src")
    return pd.concat([macro2, micro2], axis=1)


def build_common_trial_table(micro_trials: pd.DataFrame,
                             macro_trials: pd.DataFrame,
                             session_name: str,
                             strategy: EventMatchStrategy = "auto",
                             max_label_mismatch_fraction_for_order_fallback: float = 1.0) -> pd.DataFrame:
    """
    Fusionne micro et macro en table maître.

    En sortie, les colonnes importantes ont des noms uniques et stables :
    t_start_micro, t_end_micro, t_start_macro, t_end_macro, group_label, etc.
    """
    if strategy not in {"auto", "label", "order"}:
        raise ValueError(f"strategy invalide: {strategy}")

    label_keys_micro = micro_trials["label_stim"].map(normalize_stim_label_for_matching).tolist()
    label_keys_macro = macro_trials["label_stim"].map(normalize_stim_label_for_matching).tolist()
    n = min(len(label_keys_micro), len(label_keys_macro))
    mismatch_fraction = float(np.mean([label_keys_micro[i] != label_keys_macro[i] for i in range(n)])) if n else 1.0

    merge_method = strategy
    if strategy == "auto":
        if len(micro_trials) == len(macro_trials) and mismatch_fraction == 0:
            merge_method = "order"
        else:
            try:
                merged_test = _merge_by_label(micro_trials, macro_trials)
                merged = merged_test
                merge_method = "label"
            except Exception:
                if len(micro_trials) == len(macro_trials) and mismatch_fraction <= max_label_mismatch_fraction_for_order_fallback:
                    merge_method = "order"
                else:
                    raise

    if merge_method == "label" and not (strategy == "auto" and "merged" in locals()):
        merged = _merge_by_label(micro_trials, macro_trials)
    elif merge_method == "order":
        merged = _merge_by_order(micro_trials, macro_trials)
    elif not (strategy == "auto" and "merged" in locals()):
        raise RuntimeError("État de fusion inattendu")

    # Helper pour récupérer une colonne malgré les suffixes selon la méthode de merge.
    def pick(*names: str, default: Any = np.nan) -> Any:
        for name in names:
            if name in merged.columns:
                return merged[name]
        return default

    out = pd.DataFrame({
        "session": session_name,
        "stim_index": np.arange(len(merged), dtype=int),
        # "event_merge_method": merge_method,
        "event_label_mismatch_fraction_order": mismatch_fraction,

        "label_stim": pick("label_stim_macro_src", "label_stim"),
        # "label_stim_micro": pick("label_stim_micro_src"),
        # "label_stim_key_macro": pick("label_stim_key_macro_src", "label_stim_key"),
        # "label_stim_key_micro": pick("label_stim_key_micro_src"),

        "t_start_macro": pd.to_numeric(pick("t_start_macro_macro_src", "t_start_macro"), errors="coerce"),
        "duration_macro": pd.to_numeric(pick("duration_macro_macro_src", "duration_macro"), errors="coerce"),
        "t_end_macro": pd.to_numeric(pick("t_end_macro_macro_src", "t_end_macro"), errors="coerce"),
        "t_start_micro": pd.to_numeric(pick("t_start_micro_micro_src", "t_start_micro"), errors="coerce"),
        "duration_micro": pd.to_numeric(pick("duration_micro_micro_src", "duration_micro"), errors="coerce"),
        "t_end_micro": pd.to_numeric(pick("t_end_micro_micro_src", "t_end_micro"), errors="coerce"),

        "lobe": pick("lobe_macro_src", "lobe_micro_src", "lobe"),
        "cog": pick("cog_macro_src", "cog_micro_src", "cog"),
        "group_label": pick("group_label_macro_src", "group_label_micro_src", "group_label"),
        "cog_labels": pick("cog_labels_macro_src", "cog_labels_micro_src", "cog_labels"),

        "stim_bipolar_label": pick("stim_bipolar_label_macro_src", "stim_bipolar_label_micro_src", "stim_bipolar_label"),
        "stim_shaft": pick("stim_shaft_macro_src", "stim_shaft_micro_src", "stim_shaft"),
        "stim_contact_pair": pick("stim_contact_pair_macro_src", "stim_contact_pair_micro_src", "stim_contact_pair"),
        "stim_contact_1": pick("stim_contact_1_macro_src", "stim_contact_1_micro_src", "stim_contact_1"),
        "stim_contact_2": pick("stim_contact_2_macro_src", "stim_contact_2_micro_src", "stim_contact_2"),
        "stim_intensity": pick("stim_intensity_macro_src", "stim_intensity_micro_src", "stim_intensity"),
        "stim_frequency": pick("stim_frequency_macro_src", "stim_frequency_micro_src", "stim_frequency"),
        "stim_pulse_width_us": pick("stim_pulse_width_us_macro_src", "stim_pulse_width_us_micro_src", "stim_pulse_width_us"),

        "macro_part": pick("macro_part_macro_src", "macro_part"),
        "macro_part_index": pick("macro_part_index_macro_src", "macro_part_index"),
        "macro_event_index": pick("macro_event_index_macro_src", "macro_event_index"),
        "micro_event_index": pick("micro_event_index_micro_src", "micro_event_index"),
    })

    # Sérialisation stable des listes si export CSV.
    out["cog_labels"] = out["cog_labels"].apply(lambda x: x if isinstance(x, list) else parse_list_cell(x))

    # Casts numériques.
    # for col in [
    #     "t_start_macro", "duration_macro", "t_end_macro",
    #     "t_start_micro", "duration_micro", "t_end_micro",
    #     "stim_contact_1", "stim_contact_2", "stim_intensity", "stim_frequency", "stim_pulse_width_us",
    # ]:
    #     out[col] = pd.to_numeric(out[col], errors="coerce")

    # QC label par ordre.
    # out["label_match_order"] = (
    #     out["label_stim_key_macro"].fillna("").astype(str)
    #     == out["label_stim_key_micro"].fillna("").astype(str)
    # )

    return out.reset_index(drop=True)


# =============================================================================
# ALIGNEMENT TEMPOREL ET FENÊTRES
# =============================================================================


def estimate_micro_macro_offset_from_trials(trials: pd.DataFrame,
                                             residual_outlier_mad_thresh: float = 6.0) -> OffsetMapping:
    """
    Estime le décalage constant t_macro - t_micro à partir des débuts de stimulation.

    Hypothèse retenue pour ce projet : les horloges micro et macro ont la même
    échelle temporelle ; seule l'origine diffère. La médiane des différences est
    donc utilisée comme offset robuste.
    """
    x = pd.to_numeric(trials["t_start_micro"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(trials["t_start_macro"], errors="coerce").to_numpy(float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 1:
        raise ValueError("Au moins 1 point micro/macro est nécessaire pour estimer l'offset")

    delta = y - x
    offset0 = float(np.nanmedian(delta[finite]))
    resid0 = delta - offset0

    used = finite.copy()
    if finite.sum() >= 5:
        med0 = float(np.nanmedian(resid0[finite]))
        mad0 = float(np.nanmedian(np.abs(resid0[finite] - med0)))
        if mad0 == 0 or not np.isfinite(mad0):
            mad0 = float(np.nanmedian(np.abs(resid0[finite])))
        if mad0 > 0 and np.isfinite(mad0):
            used = finite & (np.abs(resid0 - med0) <= residual_outlier_mad_thresh * mad0)
            if used.sum() < 1:
                used = finite.copy()

    offset = float(np.nanmedian(delta[used]))
    resid = delta - offset
    resid_used = resid[used]
    med = float(np.nanmedian(resid_used)) if resid_used.size else np.nan
    mad = float(np.nanmedian(np.abs(resid_used - med))) if resid_used.size else np.nan
    max_abs = float(np.nanmax(np.abs(resid_used))) if resid_used.size else np.nan

    return OffsetMapping(
        offset_s=offset,
        n_points=int(finite.sum()),
        n_used=int(used.sum()),
        residual_median_s=med,
        residual_mad_s=mad,
        residual_max_abs_s=max_abs,
        used_mask=used.tolist(),
    )


def add_offset_qc_columns(trials: pd.DataFrame, offset: OffsetMapping) -> pd.DataFrame:
    """Ajoute les colonnes QC associées au décalage constant micro↔macro."""
    out = trials.copy()
    pred = offset.micro_to_macro(out["t_start_micro"].to_numpy(float))
    out["t_start_macro_predicted_from_micro_offset"] = pred
    out["micro_macro_offset_s"] = float(offset.offset_s)
    out["offset_residual_start_s"] = out["t_start_macro"].to_numpy(float) - pred
    out["offset_used_for_qc"] = offset.used_mask[: len(out)]
    return out


def add_common_trial_windows(trials: pd.DataFrame,
                             pre_length: float,
                             post_length: float,
                             epsilon: float) -> pd.DataFrame:
    """Ajoute les fenêtres pré/post dans les référentiels micro et macro."""
    out = trials.copy()

    for ref in ["micro", "macro"]:
        t0 = out[f"t_start_{ref}"].astype(float)
        t1 = out[f"t_end_{ref}"].astype(float)
        out[f"pre_start_{ref}"] = t0 - pre_length - epsilon
        out[f"pre_end_{ref}"] = t0 - epsilon
        out[f"post_start_{ref}"] = t1 + epsilon
        out[f"post_end_{ref}"] = t1 + epsilon + post_length
        # out[f"stim_start_{ref}"] = t0
        # out[f"stim_end_{ref}"] = t1

    out["pre_length"] = float(pre_length)
    out["post_length"] = float(post_length)
    out["epsilon"] = float(epsilon)
    return out


def keep_trials_with_valid_windows(trials: pd.DataFrame,
                                   micro_duration_s: Optional[float] = None,
                                   macro_duration_s: Optional[float] = None) -> pd.DataFrame:
    """Exclut les trials dont les fenêtres débordent des signaux si les durées sont fournies."""
    keep = np.ones(len(trials), dtype=bool)
    if micro_duration_s is not None:
        keep &= trials["pre_start_micro"].to_numpy(float) >= 0
        keep &= trials["post_end_micro"].to_numpy(float) <= float(micro_duration_s)
    if macro_duration_s is not None:
        keep &= trials["pre_start_macro"].to_numpy(float) >= 0
        keep &= trials["post_end_macro"].to_numpy(float) <= float(macro_duration_s)
    return trials.loc[keep].reset_index(drop=True)


# =============================================================================
# HILBERT EXPORTS : CHARGEMENT / INDEXAGE
# =============================================================================


def load_hilbert_exports_index(hilbert_root: str | Path,
                               session_name: str,
                               bands: Optional[Sequence[str]] = None,
                               mmap_mode: Optional[str] = "r") -> Dict[str, Any]:
    """
    Charge l'index des exports Hilbert et, si demandé, les tenseurs par bande.

    Retour :
        {
          session, session_dir, metadata, trial_table, times, bands, epochs_by_band
        }
    """
    session_dir = find_hilbert_session_dir(hilbert_root, session_name)
    meta_file = session_dir / f"{session_name}_metadata.json"
    trial_file = session_dir / f"{session_name}_trial_table.csv"
    times_file = session_dir / f"{session_name}_times.npy"

    if not meta_file.exists():
        raise FileNotFoundError(meta_file)
    if not trial_file.exists():
        raise FileNotFoundError(trial_file)
    if not times_file.exists():
        raise FileNotFoundError(times_file)

    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    times = np.load(times_file)
    trial_table = pd.read_csv(trial_file)

    if bands is None:
        band_files = sorted(session_dir.glob(f"{session_name}_hilbert_*.npy"))
        bands = [fp.stem.replace(f"{session_name}_hilbert_", "") for fp in band_files]

    epochs_by_band: Dict[str, np.ndarray] = {}
    for band in bands:
        fp = session_dir / f"{session_name}_hilbert_{safe_name(band)}.npy"
        if fp.exists():
            arr = np.load(fp, mmap_mode=mmap_mode)
            if arr.ndim != 3:
                raise ValueError(f"{fp.name}: shape attendue (n_trials, n_channels, n_times), reçue {arr.shape}")
            epochs_by_band[band] = arr

    return {
        "session": session_name,
        "session_dir": str(session_dir),
        "metadata": meta,
        "trial_table": trial_table,
        "times": times,
        "bands": list(bands),
        "epochs_by_band": epochs_by_band,
    }


def split_hilbert_times(times: np.ndarray) -> Dict[str, np.ndarray]:
    """Retourne masques pré/post sur l'axe temps concaténé Hilbert."""
    times = np.asarray(times, dtype=float)
    return {
        "pre_mask": times < 0,
        "post_mask": times >= 0,
        "times_pre": times[times < 0],
        "times_post": times[times >= 0],
    }


def get_hilbert_time_indices(times: np.ndarray,
                             t0_rel: float,
                             t1_rel: float,
                             side: Literal["closed", "left_closed"] = "left_closed") -> np.ndarray:
    """Indices de l'axe Hilbert dans une fenêtre relative."""
    times = np.asarray(times, dtype=float)
    if side == "closed":
        return np.where((times >= t0_rel) & (times <= t1_rel))[0]
    return np.where((times >= t0_rel) & (times < t1_rel))[0]


# =============================================================================
# SPIKES : HELPERS GÉNÉRIQUES SANS DÉPENDANCE FORTE À pynapple/NWB
# =============================================================================


def as_spike_times_array(unit_obj: Any) -> np.ndarray:
    """
    Convertit un objet unité en array de temps de spikes en secondes.

    Compatible avec :
    - array-like ;
    - objets ayant `.index.values` ;
    - objets ayant `.times()`.
    """
    if unit_obj is None:
        return np.asarray([], dtype=float)
    if hasattr(unit_obj, "index") and hasattr(unit_obj.index, "values"):
        arr = np.asarray(unit_obj.index.values, dtype=float)
    elif hasattr(unit_obj, "times") and callable(unit_obj.times):
        arr = np.asarray(unit_obj.times(), dtype=float)
    else:
        arr = np.asarray(unit_obj, dtype=float)
    arr = arr[np.isfinite(arr)]
    return np.sort(arr)


def get_spikes_in_interval(spike_times: Sequence[float],
                           t0: float,
                           t1: float,
                           dead_intervals: Optional[np.ndarray | pd.DataFrame] = None) -> np.ndarray:
    """Retourne les spikes dans [t0, t1), en retirant les dead intervals si fournis."""
    spk = np.asarray(spike_times, dtype=float)
    spk = spk[(spk >= t0) & (spk < t1)]
    if dead_intervals is None or len(spk) == 0:
        return spk

    if isinstance(dead_intervals, pd.DataFrame):
        di = dead_intervals.iloc[:, :2].to_numpy(float)
    else:
        di = np.asarray(dead_intervals, dtype=float).reshape(-1, 2)
    if di.size == 0:
        return spk

    keep = np.ones(len(spk), dtype=bool)
    for s, e in di:
        if not np.isfinite(s) or not np.isfinite(e):
            continue
        keep &= ~((spk >= s) & (spk < e))
    return spk[keep]


def get_trial_spikes(spike_times: Sequence[float],
                     trials: pd.DataFrame,
                     trial_idx: int,
                     window: Literal["pre", "post", "full"] = "post",
                     reference: Literal["micro", "macro"] = "micro",
                     dead_intervals: Optional[np.ndarray | pd.DataFrame] = None,
                     relative_to: Literal["stim_start", "stim_end", "window_start", "none"] = "stim_end") -> np.ndarray:
    """
    Extrait les spikes d'un trial dans le référentiel choisi.

    Pour les spikes unitaires, `reference='micro'` est le cas normal.
    """
    row = trials.iloc[int(trial_idx)]
    if window == "pre":
        t0, t1 = float(row[f"pre_start_{reference}"]), float(row[f"pre_end_{reference}"])
    elif window == "post":
        t0, t1 = float(row[f"post_start_{reference}"]), float(row[f"post_end_{reference}"])
    elif window == "full":
        t0, t1 = float(row[f"pre_start_{reference}"]), float(row[f"post_end_{reference}"])
    else:
        raise ValueError(f"window invalide: {window}")

    spk = get_spikes_in_interval(spike_times, t0, t1, dead_intervals=dead_intervals)
    if relative_to == "none":
        return spk
    if relative_to == "stim_start":
        return spk - float(row[f"stim_start_{reference}"])
    if relative_to == "stim_end":
        return spk - float(row[f"stim_end_{reference}"])
    if relative_to == "window_start":
        return spk - t0
    raise ValueError(f"relative_to invalide: {relative_to}")


def bin_spikes_relative(spike_times_rel: Sequence[float],
                        times: np.ndarray,
                        bin_edges: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Bin des spikes relatifs sur l'axe Hilbert ou sur des bin_edges explicites.

    Si bin_edges est None, on construit des bords depuis l'axe `times` en utilisant
    le pas médian.
    """
    times = np.asarray(times, dtype=float)
    if bin_edges is None:
        if len(times) < 2:
            raise ValueError("times doit contenir au moins 2 points")
        dt = float(np.median(np.diff(times)))
        bin_edges = np.r_[times - dt / 2.0, times[-1] + dt / 2.0]
    counts, _ = np.histogram(np.asarray(spike_times_rel, dtype=float), bins=bin_edges)
    return counts.astype(float)


# =============================================================================
# CANAUX / LOCALITÉ
# =============================================================================


def parse_bipolar_shaft(bp_name: str) -> str:
    """Extrait le shaft d'un canal bipolaire, ex. CU1-CU2 -> CU."""
    left = str(bp_name).split("-")[0]
    m = re.match(r"^([A-Za-zÀ-ÿ_'`]+_?)(\d+)$", left.replace(" ", ""))
    if m is None:
        return normalize_channel_name(left)
    return normalize_channel_name(m.group(1))


def channel_locality_for_trial(channel_name: str, trial_row: pd.Series) -> str:
    """local si shaft canal == stim_shaft, distant sinon."""
    ch_shaft = parse_bipolar_shaft(channel_name)
    stim_shaft = trial_row.get("stim_shaft", np.nan)
    if pd.isna(stim_shaft):
        return "unknown"
    stim_shaft = normalize_channel_name(str(stim_shaft))
    if not stim_shaft:
        return "unknown"
    return "local" if ch_shaft == stim_shaft else "distant"


def build_channel_trial_locality_table(trials: pd.DataFrame,
                                       bp_names: Sequence[str]) -> pd.DataFrame:
    """Table trial × channel avec localité local/distant."""
    rows: List[Dict[str, Any]] = []
    for ti, row in trials.iterrows():
        for ci, ch in enumerate(bp_names):
            rows.append({
                "stim_index": int(row["stim_index"]),
                "trial_idx": int(ti),
                "channel_idx": int(ci),
                "channel_name": ch,
                "channel_shaft": parse_bipolar_shaft(ch),
                "stim_shaft": row.get("stim_shaft", np.nan),
                "locality": channel_locality_for_trial(ch, row),
            })
    return pd.DataFrame(rows)


# =============================================================================
# ORCHESTRATEUR COMMUN
# =============================================================================


def prepare_common_session(patient: Optional[str] = None,
                           session: Optional[str | int] = None,
                           session_name: Optional[str] = None,
                           cfg: Optional[CommonPreprocessConfig] = None,
                           micro_root: Optional[str | Path] = None,
                           macro_root: Optional[str | Path] = None,
                           hilbert_root: Optional[str | Path] = None,
                           hilbert_bands: Optional[Sequence[str]] = None) -> CommonSessionBundle:
    """
    Point d'entrée principal.

    Peut être appelé soit avec cfg, soit avec les racines en arguments directs.
    """
    patient_id, session_num, sess_name = parse_patient_session(session_name, patient, session)

    if cfg is None:
        if micro_root is None or macro_root is None:
            raise ValueError("Fournir cfg ou micro_root + macro_root")
        cfg = CommonPreprocessConfig(
            micro_root=str(micro_root),
            macro_root=str(macro_root),
            hilbert_root=str(hilbert_root) if hilbert_root is not None else None,
        )
    else:
        if micro_root is not None:
            cfg.micro_root = str(micro_root)
        if macro_root is not None:
            cfg.macro_root = str(macro_root)
        if hilbert_root is not None:
            cfg.hilbert_root = str(hilbert_root)

    log(f"\n=== Base commune spike-LFP | {sess_name} ===", cfg.verbose)

    micro_trials, micro_paths = load_micro_trial_table(
        micro_root=cfg.micro_root,
        patient=patient_id,
        session_num=session_num,
        session_name=sess_name,
    )
    macro_trials, macro_paths = load_macro_trial_table(
        macro_root=cfg.macro_root,
        session_name=sess_name,
        allow_missing_macro_cog=True,
    )

    if cfg.require_same_n_events and len(micro_trials) != len(macro_trials):
        raise ValueError(f"Nombre d'événements différent: micro={len(micro_trials)}, macro={len(macro_trials)}")

    trials = build_common_trial_table(
        micro_trials=micro_trials,
        macro_trials=macro_trials,
        session_name=sess_name,
        strategy=cfg.event_match_strategy,
        max_label_mismatch_fraction_for_order_fallback=cfg.max_label_mismatch_fraction_for_order_fallback,
    )

    offset = None
    if cfg.compute_micro_macro_offset_qc:
        offset = estimate_micro_macro_offset_from_trials(
            trials,
            residual_outlier_mad_thresh=cfg.offset_outlier_mad_thresh,
        )
        trials = add_offset_qc_columns(trials, offset)
        log(
            f"[INFO] offset micro↔macro: offset={offset.offset_s:.6f}s, "
            f"MAD résidus={offset.residual_mad_s:.6g}s, "
            f"max|résidu|={offset.residual_max_abs_s:.6g}s, "
            f"n_used={offset.n_used}/{offset.n_points}",
            cfg.verbose,
        )

    trials = add_common_trial_windows(
        trials,
        pre_length=cfg.pre_length,
        post_length=cfg.post_length,
        epsilon=cfg.epsilon,
    )

    hilbert = None
    hroot = cfg.hilbert_root or (str(hilbert_root) if hilbert_root is not None else None)
    if hroot is not None:
        try:
            hilbert = load_hilbert_exports_index(hroot, sess_name, bands=hilbert_bands, mmap_mode="r")
            log(f"[INFO] Hilbert exports chargés: {list(hilbert['epochs_by_band'].keys())}", cfg.verbose)
        except FileNotFoundError as exc:
            log(f"[WARN] exports Hilbert non chargés: {exc}", cfg.verbose)
            hilbert = None

    paths: Dict[str, Optional[str]] = {
        **micro_paths,
        **macro_paths,
        "hilbert_root": hroot,
        "hilbert_session_dir": hilbert["session_dir"] if hilbert is not None else None,
    }

    return CommonSessionBundle(
        patient=patient_id,
        session_num=session_num,
        session_name=sess_name,
        trials=trials,
        offset=offset,
        paths=paths,
        hilbert=hilbert,
    )


# =============================================================================
# SAUVEGARDE / RECHARGEMENT DE LA BASE COMMUNE
# =============================================================================


def save_common_session_bundle(bundle: CommonSessionBundle, out_dir: str | Path) -> Path:
    """Sauvegarde trials + metadata/offset dans un dossier session."""
    out_root = ensure_dir(out_dir)
    session_out = ensure_dir(out_root / bundle.session_name)
    trials_to_save = bundle.trials.copy()
    if "cog_labels" in trials_to_save.columns:
        trials_to_save["cog_labels"] = trials_to_save["cog_labels"].apply(lambda x: repr(x) if isinstance(x, list) else x)
    trials_to_save.to_csv(session_out / f"{bundle.session_name}_common_trials.csv", index=False)

    meta = {
        "patient": bundle.patient,
        "session_num": bundle.session_num,
        "session_name": bundle.session_name,
        "paths": bundle.paths,
        "offset": asdict(bundle.offset) if bundle.offset is not None else None,
        "n_trials": int(len(bundle.trials)),
        "columns": list(bundle.trials.columns),
        "hilbert_loaded": bundle.hilbert is not None,
        "hilbert_bands_loaded": list(bundle.hilbert["epochs_by_band"].keys()) if bundle.hilbert is not None else [],
    }
    with open(session_out / f"{bundle.session_name}_common_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return session_out


def load_common_session_bundle(session_dir: str | Path,
                               load_hilbert: bool = False,
                               hilbert_bands: Optional[Sequence[str]] = None) -> CommonSessionBundle:
    """Recharge un bundle sauvegardé par save_common_session_bundle."""
    session_dir = Path(session_dir)
    session_name = session_dir.name
    trials_fp = session_dir / f"{session_name}_common_trials.csv"
    meta_fp = session_dir / f"{session_name}_common_metadata.json"
    if not trials_fp.exists():
        raise FileNotFoundError(trials_fp)
    if not meta_fp.exists():
        raise FileNotFoundError(meta_fp)

    trials = pd.read_csv(trials_fp)
    if "cog_labels" in trials.columns:
        trials["cog_labels"] = trials["cog_labels"].apply(parse_list_cell)

    with open(meta_fp, "r", encoding="utf-8") as f:
        meta = json.load(f)

    offset = None
    if meta.get("offset") is not None:
        offset = OffsetMapping(**meta["offset"])

    hilbert = None
    if load_hilbert and meta.get("paths", {}).get("hilbert_root") is not None:
        hilbert = load_hilbert_exports_index(
            meta["paths"]["hilbert_root"],
            session_name,
            bands=hilbert_bands,
            mmap_mode="r",
        )

    return CommonSessionBundle(
        patient=meta["patient"],
        session_num=meta["session_num"],
        session_name=meta["session_name"],
        trials=trials,
        offset=offset,
        paths=meta.get("paths", {}),
        hilbert=hilbert,
    )


# =============================================================================
# VALIDATION RAPIDE
# =============================================================================


def validate_common_trials(trials: pd.DataFrame) -> Dict[str, Any]:
    """Résumé QC rapide d'une table commune."""
    required = [
        "stim_index", "label_stim", "t_start_micro", "t_end_micro",
        "t_start_macro", "t_end_macro", "stim_shaft", "group_label",
        "pre_start_micro", "post_end_micro", "pre_start_macro", "post_end_macro",
    ]
    missing = [c for c in required if c not in trials.columns]
    numeric_cols = ["t_start_micro", "t_end_micro", "t_start_macro", "t_end_macro"]
    nan_counts = {c: int(pd.to_numeric(trials[c], errors="coerce").isna().sum()) for c in numeric_cols if c in trials.columns}

    out = {
        "n_trials": int(len(trials)),
        "missing_required_columns": missing,
        "nan_counts": nan_counts,
        "n_label_order_mismatch": int((~trials.get("label_match_order", pd.Series([True] * len(trials))).astype(bool)).sum()) if len(trials) else 0,
        "group_counts": trials["group_label"].value_counts(dropna=False).to_dict() if "group_label" in trials.columns else {},
        "n_unparsed_stim_labels": int(trials["stim_shaft"].isna().sum()) if "stim_shaft" in trials.columns else np.nan,
    }
    return out


__all__ = [
    "CommonPreprocessConfig",
    "OffsetMapping",
    "CommonSessionBundle",
    "parse_patient_session",
    "normalize_channel_name",
    "normalize_stim_label_for_matching",
    "parse_stim_label_robust",
    "parse_list_cell",
    "classify_group_and_cog_labels",
    "add_stim_and_cog_metadata",
    "read_cog_file_robust",
    "read_corrected_macro_events_robust",
    "find_micro_session_dir",
    "find_micro_cog_file",
    "find_macro_corrected_file",
    "find_macro_cog_file",
    "find_hilbert_session_dir",
    "load_micro_trial_table",
    "load_macro_trial_table",
    "build_common_trial_table",
    "estimate_micro_macro_offset_from_trials",
    "add_offset_qc_columns",
    "add_common_trial_windows",
    "keep_trials_with_valid_windows",
    "load_hilbert_exports_index",
    "split_hilbert_times",
    "get_hilbert_time_indices",
    "as_spike_times_array",
    "get_spikes_in_interval",
    "get_trial_spikes",
    "bin_spikes_relative",
    "parse_bipolar_shaft",
    "channel_locality_for_trial",
    "build_channel_trial_locality_table",
    "prepare_common_session",
    "save_common_session_bundle",
    "load_common_session_bundle",
    "validate_common_trials",
]
