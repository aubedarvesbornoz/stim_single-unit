#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spike_lfp_common_batch.py
=========================

Orchestrateur multi-sessions pour construire la base commune spike–LFP.

Ce module évite de lancer manuellement `prepare_common_session` session par session.
Il parcourt les sessions disponibles dans les exports Hilbert, ou dans les fichiers
macro corrigés, vérifie que les fichiers micro/macro nécessaires existent, vérifie
optionnellement la présence du NWB côté micro, puis sauvegarde les common bundles :

    common_root/SESSION/SESSION_common_trials.csv
    common_root/SESSION/SESSION_common_metadata.json

Le NWB n'est pas nécessaire pour construire la base commune, mais l'option
`require_existing_nwb=True` permet de ne préparer que les sessions qui pourront
ensuite être analysées en spike–power / spike–phase.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Literal
import numpy as np
import pandas as pd

from utils_spike_lfp.spike_lfp_common_preprocess_session import (
    CommonPreprocessConfig,
    prepare_common_session,
    save_common_session_bundle,
    validate_common_trials,
)

SessionSource = Literal["hilbert", "macro", "intersection", "union"]


@dataclass
class CommonBatchPreprocessConfig:
    """Configuration du batch common-preprocess."""

    micro_root: str
    macro_root: str
    common_root: str
    hilbert_root: Optional[str] = None

    # Source de la liste des sessions :
    # - hilbert : dossiers présents dans results_hilbert ; recommandé si les Hilbert sont déjà calculés.
    # - macro : fichiers *_stim_events_TRC_corrected.txt présents dans macro_root.
    # - intersection : sessions présentes dans Hilbert ET macro.
    # - union : sessions présentes dans Hilbert OU macro.
    session_source: SessionSource = "hilbert"

    # Optionnel : restreindre explicitement à certaines sessions, ex. ('P119_FM71_stim4', ...).
    include_sessions: Optional[Tuple[str, ...]] = None
    exclude_sessions: Tuple[str, ...] = ()

    # Fenêtres communes.
    pre_length: float = 3.0
    post_length: float = 3.0
    epsilon: float = 0.1

    # Fusion événements.
    event_match_strategy: str = "auto"
    require_same_n_events: bool = True
    max_label_mismatch_fraction_for_order_fallback: float = 1.0

    # Offset QC.
    compute_micro_macro_offset_qc: bool = True
    offset_outlier_mad_thresh: float = 6.0

    # Hilbert.
    hilbert_bands: Tuple[str, ...] = ("theta", "alpha", "beta", "low_gamma", "high_gamma")
    require_hilbert_exports: bool = True

    # NWB : utile pour ne préparer que les sessions exploitables ensuite.
    require_existing_nwb: bool = False

    # Sorties.
    skip_existing: bool = False
    must_be_absolute_output: bool = True
    verbose: bool = True


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


def parse_session_name(session_name: str) -> Tuple[str, str]:
    """Parse 'P119_FM71_stim4' -> ('P119_FM71', '4')."""
    m = re.match(r"^(?P<patient>.+?)_stim(?P<session>\d+[A-Za-z]?)$", str(session_name))
    if not m:
        raise ValueError(f"Nom de session non reconnu: {session_name}")
    return m.group("patient"), m.group("session")


def list_hilbert_sessions(hilbert_root: str | Path) -> List[str]:
    """Liste les dossiers SESSION contenant au minimum metadata/trial_table/times."""
    root = Path(hilbert_root).expanduser().resolve()
    if not root.exists():
        return []
    out: List[str] = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        s = d.name
        if (d / f"{s}_metadata.json").exists() and (d / f"{s}_trial_table.csv").exists() and (d / f"{s}_times.npy").exists():
            out.append(s)
    return out


def list_macro_corrected_sessions(macro_root: str | Path) -> List[str]:
    """Liste les sessions avec un fichier *_stim_events_TRC_corrected.txt."""
    root = Path(macro_root).expanduser().resolve()
    if not root.exists():
        return []
    out: List[str] = []
    for fp in sorted(root.glob("*_stim_events_TRC_corrected.txt")):
        out.append(fp.name.replace("_stim_events_TRC_corrected.txt", ""))
    return sorted(set(out))


def select_sessions_for_batch(cfg: CommonBatchPreprocessConfig) -> List[str]:
    hilb = set(list_hilbert_sessions(cfg.hilbert_root)) if cfg.hilbert_root is not None else set()
    macro = set(list_macro_corrected_sessions(cfg.macro_root))

    if cfg.session_source == "hilbert":
        sessions = hilb
    elif cfg.session_source == "macro":
        sessions = macro
    elif cfg.session_source == "intersection":
        sessions = hilb & macro
    elif cfg.session_source == "union":
        sessions = hilb | macro
    else:
        raise ValueError(f"session_source inconnu: {cfg.session_source}")

    if cfg.include_sessions is not None:
        sessions &= set(cfg.include_sessions)
    sessions -= set(cfg.exclude_sessions)
    return sorted(sessions)


def find_nwb_file(root_micro: str | Path, patient: str, session: str | int) -> Optional[Path]:
    """Recherche PATIENT_stimN.nwb dans les structures micro habituelles."""
    root = Path(root_micro).expanduser()
    session = str(session).replace("stim", "")
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


def common_output_exists(common_root: str | Path, session_name: str) -> bool:
    root = Path(common_root).expanduser()
    d = root / session_name
    return (d / f"{session_name}_common_trials.csv").exists() and (d / f"{session_name}_common_metadata.json").exists()


def make_common_preprocess_config(cfg: CommonBatchPreprocessConfig) -> CommonPreprocessConfig:
    return CommonPreprocessConfig(
        micro_root=cfg.micro_root,
        macro_root=cfg.macro_root,
        hilbert_root=cfg.hilbert_root,
        pre_length=cfg.pre_length,
        post_length=cfg.post_length,
        epsilon=cfg.epsilon,
        event_match_strategy=cfg.event_match_strategy,  # type: ignore[arg-type]
        require_same_n_events=cfg.require_same_n_events,
        max_label_mismatch_fraction_for_order_fallback=cfg.max_label_mismatch_fraction_for_order_fallback,
        compute_micro_macro_offset_qc=cfg.compute_micro_macro_offset_qc,
        offset_outlier_mad_thresh=cfg.offset_outlier_mad_thresh,
        verbose=cfg.verbose,
    )


def run_all_common_preprocess(cfg: CommonBatchPreprocessConfig) -> Dict[str, Any]:
    """Construit/sauvegarde les common bundles pour toutes les sessions sélectionnées."""
    common_root = ensure_dir(cfg.common_root)
    sessions = select_sessions_for_batch(cfg)
    if len(sessions) == 0:
        raise RuntimeError("Aucune session candidate trouvée pour le common preprocess")

    rows: List[Dict[str, Any]] = []
    errors: List[Tuple[str, str]] = []
    skipped: List[Tuple[str, str]] = []
    out_dirs: List[str] = []

    common_cfg = make_common_preprocess_config(cfg)

    for session_name in sessions:
        try:
            patient, sess = parse_session_name(session_name)
        except Exception as exc:
            skipped.append((session_name, f"parse_failed: {exc}"))
            continue

        if cfg.skip_existing and common_output_exists(cfg.common_root, session_name):
            d = str((Path(cfg.common_root).expanduser().resolve() / session_name))
            out_dirs.append(d)
            rows.append({"session": session_name, "patient": patient, "session_num": sess, "status": "skipped_existing", "out_dir": d})
            continue

        nwb_fp = find_nwb_file(cfg.micro_root, patient, sess)
        if cfg.require_existing_nwb and nwb_fp is None:
            skipped.append((session_name, "NWB absent"))
            rows.append({"session": session_name, "patient": patient, "session_num": sess, "status": "skipped_nwb_absent"})
            continue

        if cfg.require_hilbert_exports and cfg.hilbert_root is not None:
            hdir = Path(cfg.hilbert_root).expanduser().resolve() / session_name
            if not hdir.exists():
                skipped.append((session_name, "Hilbert exports absents"))
                rows.append({"session": session_name, "patient": patient, "session_num": sess, "status": "skipped_hilbert_absent"})
                continue

        try:
            log(f"\n=== Common batch | {session_name} ===", cfg.verbose)
            bundle = prepare_common_session(
                patient=patient,
                session=sess,
                cfg=common_cfg,
                hilbert_bands=cfg.hilbert_bands,
            )
            out_dir = save_common_session_bundle(
                bundle,
                common_root,
                must_be_absolute=cfg.must_be_absolute_output,
            )
            qc = validate_common_trials(bundle.trials)
            out_dirs.append(str(out_dir))
            rows.append({
                "session": session_name,
                "patient": patient,
                "session_num": sess,
                "status": "ok",
                "nwb_file": str(nwb_fp) if nwb_fp is not None else None,
                "out_dir": str(out_dir),
                "n_trials": int(len(bundle.trials)),
                "n_unparsed_stim_labels": qc.get("n_unparsed_stim_labels"),
                "n_label_order_mismatch": qc.get("n_label_order_mismatch"),
                "group_counts": qc.get("group_counts"),
                "offset_s": bundle.offset.offset_s if bundle.offset is not None else np.nan,
                "offset_residual_max_abs_s": bundle.offset.residual_max_abs_s if bundle.offset is not None else np.nan,
                "hilbert_loaded": bundle.hilbert is not None,
            })
        except Exception as exc:
            errors.append((session_name, repr(exc)))
            rows.append({"session": session_name, "patient": patient, "session_num": sess, "status": "error", "error": repr(exc)})
            log(f"[ERROR] {session_name}: {exc}", cfg.verbose)

    summary_df = pd.DataFrame(rows)
    summary_csv = common_root / "run_all_common_preprocess_sessions.csv"
    summary_df.to_csv(summary_csv, index=False)

    summary = {
        "config": _jsonify(asdict(cfg)),
        "n_sessions_candidate": len(sessions),
        "n_ok": int((summary_df.get("status") == "ok").sum()) if not summary_df.empty and "status" in summary_df.columns else 0,
        "n_skipped": len(skipped),
        "skipped": skipped,
        "n_errors": len(errors),
        "errors": errors,
        "session_output_dirs": out_dirs,
        "summary_csv": str(summary_csv),
    }
    with open(common_root / "run_all_common_preprocess_summary.json", "w", encoding="utf-8") as f:
        json.dump(_jsonify(summary), f, ensure_ascii=False, indent=2)

    return {"summary": summary, "sessions": summary_df, "common_root": common_root}


__all__ = [
    "CommonBatchPreprocessConfig",
    "run_all_common_preprocess",
    "select_sessions_for_batch",
    "list_hilbert_sessions",
    "list_macro_corrected_sessions",
    "find_nwb_file",
]
