import pandas as pd
import numpy as np
import spikeinterface as si
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from pathlib import Path

from functions_Rasters_DF import (
    get_SR, 
    get_nwb,
    get_dict_deadfiles, 
    get_dict_tetrodeName_from_tetrodeIndex,
)

TEMPORAL_FEATURE_COLUMNS = [
    "n_spikes_isi",
    "n_isi_total",
    "n_isi_valid",
    "isi_mean_ms",
    "isi_median_ms",
    "isi_std_ms",
    "cv_isi",
    "cv2_isi",
    "refractory_violations_count",
    "refractory_violations_ratio",
    "burst_index_6ms",
    "burst_index_10ms",
    "acg_bin_ms",
    "acg_window_ms",
    "acg_n_lags",
    "acg_peak_lag_ms",
    "acg_peak_count",
    "acg_peak_rate_hz",
    "acg_mean_3_10ms_rate_hz",
    "acg_mean_10_50ms_rate_hz",
    "acg_refractory_count_2ms",
    "acg_refractory_ratio_2ms",
    "acg_burst_ratio_3_10_over_3_50ms",
]


CELLTYPE_LABEL_COLUMNS = [
    "patient",
    "session",
    "clu",
    "unit_key",
    "putative_cell_type",
    "gmm_cluster",
    "gmm_confidence",
    "quality_ok_celltype",
    "waveform_feature_ok",
    "cell_type_classifier",
    "cell_type_interpretation",
    "trough_to_peak_ms",
    "peak_half_width_ms",
    "trough_half_width_ms",
    "waveform_ptp_uV",
    "trough_amp_uV",
    "peak_amp_uV",
    "peak_trough_ratio",
    "repolarization_slope_uV_per_ms",
    "sampling_rate_hz",
    "source_polarity_multiplier",
    "source_polarity_corrected",
    "waveform_sign_flipped_after_source_correction",
    "interp_out_of_bounds_fraction",
    "waveform_cache_ok",
    "num_spikes",
    "firing_rate",
    "amplitude_median",
    "presence_ratio",
    "amplitude_cutoff",
    "snr",
    "isi_violations_ratio",
    "wavemap_umap_1",
    "wavemap_umap_2",
    "wavemap_cluster",
]

##################################################
# Neuronal Metrics functions
##################################################

############### Utils loading waveforms/recordings ###############

def RecFiltered_Sort_SI_elec(patient, session, clus, mapping_anat, root):
    """
    Renvoie un objet Recording filtré et un objet Sorting spikeinterface, a partir des fichiers neuroscope et klusters d'une session
    """
    import probeinterface as pi
    from spikeinterface.extractors import read_neuroscope
    from spikeinterface.preprocessing import bandpass_filter

    path_folder = root + f'Spike-sorting/Data_folders/{patient}/{patient}_stim{session}/'
    
## 1) Charger recording + sorting depuis neuroscope
    # (read_neuroscope suppose que les .res/.clu sont dans le même dossier que le .xml)
    xml_path = f"{path_folder}/{patient}_stim{session}.xml"  
    recording, sorting = read_neuroscope(xml_path, load_recording=True, load_sorting=True)

## 2) charger probe geometry
    # n_tetrodes = mapping_anat.shape[0]
    # # positions: chaque tetrode = carré 2x2, tétrodes espacées
    # spacing_tetrode = 500.0  # µm
    # local = np.array([[0,0],[0,20],[20,0],[20,20]], dtype=float)  # 4 contacts proches
    # positions = np.zeros((n_tetrodes * 4, 2), dtype=float)
    # for t in range(n_tetrodes):
    #     base = np.array([t * spacing_tetrode, 0.0])
    #     positions[t*4:(t+1)*4] = local + base
    # probe = pi.Probe(ndim=2)
    # probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 5})
    # probe.set_device_channel_indices(np.arange(n_tetrodes * 4))
    # recording = recording.set_probe(probe, in_place=False) # on associe le mapping au recording
    
    ## 2) charger probe geometry robuste aux canaux absents
    n_tetrodes = mapping_anat.shape[0]
    spacing_tetrode = 500.0
    local = np.array([[0, 0], [0, 20], [20, 0], [20, 20]], dtype=float)

    # Canaux réellement présents dans le recording
    channel_ids = list(recording.get_channel_ids())        # ex: ['0', '1', ..., '52']
    channel_ids = np.array([int(ch) for ch in channel_ids])

    # Position théorique de chaque canal restant
    positions = np.zeros((len(channel_ids), 2), dtype=float)

    for i, ch in enumerate(channel_ids):
        t = ch // 4          # tétrode 0-index
        k = ch % 4           # canal local dans la tétrode
        base = np.array([t * spacing_tetrode, 0.0])
        positions[i] = local[k] + base

    probe = pi.Probe(ndim=2)
    probe.set_contacts(
        positions=positions,
        shapes="circle",
        shape_params={"radius": 5})

    # Ici les indices doivent correspondre aux indices internes du recording, pas forcément aux IDs originaux des canaux
    probe.set_device_channel_indices(np.arange(len(channel_ids)))

    recording = recording.set_probe(probe, in_place=False)


## 3) For one electrode: rec + sorting 

    # sous-sorting (unités dont group ∈ clus)
    def select_units_by_group(sorting, groups):
        groups = set(groups)
        g = sorting.get_property("group")
        unit_ids = [u for u, gg in zip(sorting.unit_ids, g) if int(gg) in groups]
        return sorting.select_units(unit_ids)

    sorting_elec = select_units_by_group(sorting, clus)

    # sous-recording : canaux de l'electrode seulement
    # ch=[]
    # for tt in clus :
    #     ch.append([str(tt*4-4), str(tt*4-3), str(tt*4-2), str(tt*4-1)])
    # ch = [x for xs in ch for x in xs] # on applatit la liste de listes en liste
    # recording_elec = recording.select_channels(ch) 
    ch = []
    for tt in clus:
        ch.extend([
            str(tt * 4 - 4),
            str(tt * 4 - 3),
            str(tt * 4 - 2),
            str(tt * 4 - 1)])
    # garder uniquement les canaux réellement présents
    available_ch = set(recording.get_channel_ids())
    ch = [c for c in ch if c in available_ch]
    if len(ch) == 0:
        raise ValueError(f"Aucun canal disponible pour clus={clus}")
    recording_elec = recording.select_channels(ch)

    # filtre signal (300-3000 Hz)
    recording_elec_f = bandpass_filter(recording=recording_elec, freq_min=300, freq_max=3000)

    return recording_elec_f, sorting_elec


def RecFiltered_Sort_SI_remapped_elec(patient, session, elec, clus, mapping_anat, dict_elec2deadfile, root):
    """
    Re-construct periods outside dead periods, for one electrode: returns recording filtré et sorting re-mappés avec good chunks concaténés
    """
    import spikeinterface as si
    recording_elec_f, sorting_e = RecFiltered_Sort_SI_elec(patient, session, clus, mapping_anat, root)

    deadfile_elec = dict_elec2deadfile[elec]
    sr = get_SR(patient)
    
    # Utilitary functions:
    def load_dead_intervals_ts(deadF, n_frames):
        """Depuis deadfile, retourne des intervalles bad (start,end) en frames int64, triés, mergés, clipés."""
        bad = np.rint(deadF[[0, 1]].to_numpy(dtype=float) * sr).astype(np.int64)
        # bad = deadF[[0, 1]].to_numpy(dtype=np.int64)
        # tri + filtre
        bad = bad[np.argsort(bad[:, 0])]
        bad = bad[bad[:, 1] > bad[:, 0]]
        # clip
        bad[:, 0] = np.clip(bad[:, 0], 0, n_frames)
        bad[:, 1] = np.clip(bad[:, 1], 0, n_frames)
        bad = bad[bad[:, 1] > bad[:, 0]]
        # merge overlap
        merged = []
        for s, e in bad:
            if not merged or s > merged[-1][1]:
                merged.append([int(s), int(e)])
            else:
                merged[-1][1] = max(merged[-1][1], int(e))
        return np.array(merged, dtype=np.int64)

    def invert_intervals_to_good(bad: np.ndarray, n_frames: int):
        """Depuis les bad frames, retourne les good frames. 
        bad: (N,2) frames, retourne good_chunks list[(s,e)] frames."""
        good = []
        start = 0
        for s, e in bad:
            if start < s:
                good.append((start, int(s)))
            start = max(start, int(e))
        if start < n_frames:
            good.append((start, n_frames))
        return good

    def build_concat_recording(recording, good_chunks):
        """Depuis les good frames, retourne un recording concaténé."""
        rec_list = [recording.frame_slice(start_frame=int(s), end_frame=int(e)) for s, e in good_chunks]
        return si.concatenate_recordings(rec_list)

    def remap_sorting_to_concat(sorting, good_chunks, fs):
        """
        Crée un sorting compressé correspondant au recording concaténé.
        Suppose 1 segment.
        """
        old_starts = np.array([s for s, e in good_chunks], dtype=np.int64)
        old_ends   = np.array([e for s, e in good_chunks], dtype=np.int64)
        lengths    = old_ends - old_starts
        new_starts = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.int64)

        def map_old_to_new(x):
            x = np.asarray(x, dtype=np.int64)
            idx = np.searchsorted(old_starts, x, side="right") - 1
            valid = (idx >= 0) & (x < old_ends[idx])
            x = x[valid]
            idx = idx[valid]
            return new_starts[idx] + (x - old_starts[idx])

        unit_dict = {}
        for u in sorting.unit_ids:
            st_old = sorting.get_unit_spike_train(u)  # frames
            st_new = map_old_to_new(st_old)
            unit_dict[u] = st_new

        return si.NumpySorting.from_unit_dict(unit_dict, sampling_frequency=fs)

    # recording concaténé selon deadfile
    n = recording_elec_f.get_num_frames()
    bad = load_dead_intervals_ts(deadfile_elec, n_frames=n)
    good_chunks = invert_intervals_to_good(bad, n_frames=n)
    recording_e_remapped = build_concat_recording(recording_elec_f, good_chunks)

    # remap spikes vers périodes concaténées
    sorting_remapped = remap_sorting_to_concat(sorting_e, good_chunks, sr)
    
    return recording_e_remapped, sorting_remapped


############### Waveforms & features extraction ###############

def _width_at_level(w, center_idx, level, mode, fs):
    """
    Largeur autour d'un point central au niveau donné.

    mode='above' : largeur où w >= level
    mode='below' : largeur où w <= level
    """
    n = len(w)

    if mode == "above":
        cond = lambda x: x >= level
    elif mode == "below":
        cond = lambda x: x <= level
    else:
        raise ValueError("mode doit être 'above' ou 'below'")

    if not cond(w[center_idx]):
        return np.nan

    left = center_idx
    while left > 0 and cond(w[left]):
        left -= 1

    right = center_idx
    while right < n - 1 and cond(w[right]):
        right += 1

    return (right - left) / fs * 1000.0


def _get_templates_array_from_analyzer(analyzer, n_channels):
    """
    Récupère les templates moyens depuis un SortingAnalyzer SpikeInterface
    en gérant quelques différences de versions.

    Sortie attendue :
        templates : n_units x n_samples x n_channels
    """

    templates_ext = analyzer.get_extension("templates")

    try:
        templates = templates_ext.get_data(operator="average")
    except TypeError:
        templates = templates_ext.get_data()

    templates = np.asarray(templates)

    if templates.ndim != 3:
        raise ValueError(f"Templates inattendus, shape={templates.shape}")

    # Cas standard : units x samples x channels
    if templates.shape[-1] == n_channels:
        return templates

    # Cas alternatif : units x channels x samples
    if templates.shape[1] == n_channels:
        return np.transpose(templates, (0, 2, 1))

    raise ValueError(
        f"Shape templates incompatible avec n_channels={n_channels}: {templates.shape}"
    )

    return templates

def get_source_polarity_multiplier(patient, session):
    """
    Corrige la polarité connue des systèmes d'acquisition.

    Convention cible :
    - Blackrock : déjà dans le bon sens
    - Neuralynx : inversé, sauf quelques sessions passées par MEDD

    Retour :
        +1 : ne pas inverser
        -1 : inverser
    """
    sr = get_SR(patient)
    # Blackrock
    if sr == 30000:
        return 1
    # Exception connue : Neuralynx passé par format intermédiaire MEDD, déjà dans le bon sens.
    if patient == "P106_LL59" and str(session) in ["1", "3", "5"]:
        return 1

    # Neuralynx ancien ou récent
    if sr in [16384, 32768]:
        return -1

    # Par prudence
    return 1


def _extract_features_and_waveforms_from_templates(
    analyzer,
    sorting_clean,
    recording_elec_f,
    patient,
    session,
    elec,
    unit_id_to_clu,
    ms_before=1.0,
    target_time_ms=None,
):
    """
    Extrait, pour une électrode :
    - features scalaires waveform
    - waveforms standardisées en µV
    - waveforms normalisées pour WaveMAP

    Retour :
        features_df,
        waveforms_standardized_uV,
        waveforms_normalized,
        unit_keys
    """

    if target_time_ms is None:
        target_time_ms = np.linspace(-1.0, 2.0, 121)

    fs = recording_elec_f.get_sampling_frequency()
    n_channels = recording_elec_f.get_num_channels()

    source_polarity_multiplier = get_source_polarity_multiplier(patient, session)
    
    templates = _get_templates_array_from_analyzer(
        analyzer=analyzer,
        n_channels=n_channels,
    )

    unit_ids = list(sorting_clean.unit_ids)

    if len(unit_ids) != templates.shape[0]:
        raise ValueError(
            f"Mismatch unit_ids/templates : "
            f"{len(unit_ids)} unit_ids vs {templates.shape[0]} templates"
        )

    rows = []
    waveforms_standardized_uV = []
    waveforms_normalized = []
    unit_keys = []

    for i, unit_id in enumerate(unit_ids):
        if unit_id not in unit_id_to_clu:
            continue

        clu = unit_id_to_clu[unit_id]
        unit_key = f"{patient}|stimic{session}|clu{clu}"

        wf_all_ch = templates[i]  # samples x channels

        # Best channel = plus grand peak-to-peak.
        ptp_by_ch = np.nanmax(wf_all_ch, axis=0) - np.nanmin(wf_all_ch, axis=0)
        best_ch_idx = int(np.nanargmax(ptp_by_ch))

        w_native = wf_all_ch[:, best_ch_idx].astype(float) * source_polarity_multiplier  # Facteur de correction de polarité liée au système d'acquisition.

        # Baseline simple sur début de fenêtre.
        n_base = max(3, int((0.20 / 1000.0) * fs))
        baseline = np.nanmedian(w_native[:n_base])
        w = w_native - baseline

        # Convention : trough principal négatif.
        waveform_sign_flipped = False
        if abs(np.nanmax(w)) > abs(np.nanmin(w)):
            w = -w
            waveform_sign_flipped = True

        trough_idx = int(np.nanargmin(w))
        trough_amp_uV = float(w[trough_idx])

        # Pic post-trough.
        if trough_idx < len(w) - 2:
            peak_idx = trough_idx + int(np.nanargmax(w[trough_idx:]))
            peak_amp_uV = float(w[peak_idx])
            trough_to_peak_ms = (peak_idx - trough_idx) / fs * 1000.0
        else:
            peak_idx = np.nan
            peak_amp_uV = np.nan
            trough_to_peak_ms = np.nan

        waveform_ptp_uV = float(np.nanmax(w) - np.nanmin(w))

        # Half-width trough.
        trough_half_width_ms = _width_at_level(
            w=w,
            center_idx=trough_idx,
            level=trough_amp_uV / 2.0,
            mode="below",
            fs=fs,
        )

        # Half-width peak.
        if np.isfinite(peak_amp_uV) and peak_amp_uV > 0:
            peak_half_width_ms = _width_at_level(
                w=w,
                center_idx=int(peak_idx),
                level=peak_amp_uV / 2.0,
                mode="above",
                fs=fs,
            )
        else:
            peak_half_width_ms = np.nan

        if trough_amp_uV < 0 and np.isfinite(peak_amp_uV):
            peak_trough_ratio = peak_amp_uV / abs(trough_amp_uV)
        else:
            peak_trough_ratio = np.nan

        if np.isfinite(trough_to_peak_ms) and trough_to_peak_ms > 0:
            repolarization_slope_uV_per_ms = (
                peak_amp_uV - trough_amp_uV
            ) / trough_to_peak_ms
        else:
            repolarization_slope_uV_per_ms = np.nan

        # Standardisation pour WaveMAP.
        (
            w_std_uV,
            w_norm,
            sign_flipped_for_wavemap,
            trough_time_shift_ms,
            interp_out_of_bounds_fraction,
            waveform_cache_ok,
        ) = standardize_waveform_for_wavemap(
            waveform=w,
            fs=fs,
            ms_before=ms_before,
            target_time_ms=target_time_ms,
            force_negative_trough=False,  # déjà fait juste au-dessus
        )

        waveform_cache_ok = (
            np.all(np.isfinite(w_norm))
            and np.nanmax(np.abs(w_norm)) > 0
        )

        rows.append(
            {
                "patient": patient,
                "session": f"stimic{session}",
                "session_num": str(session),
                "clu": clu,
                "unit_key": unit_key,
                "unit_id_si": unit_id,
                "electrode": elec,
                "sampling_rate_hz": fs,
                "best_channel_index_local": best_ch_idx,
                "waveform_sign_flipped": waveform_sign_flipped,
                "source_polarity_multiplier": source_polarity_multiplier,
                "source_polarity_corrected": source_polarity_multiplier == -1,
                "sign_flipped_for_wavemap": sign_flipped_for_wavemap,
                "interp_out_of_bounds_fraction": interp_out_of_bounds_fraction,
                "waveform_cache_ok": waveform_cache_ok,
                "trough_idx_native": trough_idx,
                "peak_idx_native": peak_idx,
                "trough_time_shift_ms": trough_time_shift_ms,
                "trough_to_peak_ms": trough_to_peak_ms,
                "peak_half_width_ms": peak_half_width_ms,
                "trough_half_width_ms": trough_half_width_ms,
                "waveform_ptp_uV": waveform_ptp_uV,
                "trough_amp_uV": trough_amp_uV,
                "peak_amp_uV": peak_amp_uV,
                "peak_trough_ratio": peak_trough_ratio,
                "repolarization_slope_uV_per_ms": repolarization_slope_uV_per_ms,
                "waveform_standard_n_samples": len(target_time_ms),
                "waveform_standard_t_start_ms": float(target_time_ms[0]),
                "waveform_standard_t_end_ms": float(target_time_ms[-1]),
            }
        )

        waveforms_standardized_uV.append(w_std_uV)
        waveforms_normalized.append(w_norm)
        unit_keys.append(unit_key)

    features_df = pd.DataFrame(rows)

    if len(features_df) == 0:
        return (
            features_df,
            np.empty((0, len(target_time_ms))),
            np.empty((0, len(target_time_ms))),
            np.asarray([], dtype=str),
        )

    return (
        features_df,
        np.vstack(waveforms_standardized_uV),
        np.vstack(waveforms_normalized),
        np.asarray(unit_keys, dtype=str),
    )


def standardize_waveform_for_wavemap(
    waveform,
    fs,
    ms_before,
    target_time_ms=None,
    force_negative_trough=True,
    eps=1e-12,
):
    """
    Standardise une waveform moyenne pour WaveMAP.
    (soustraction de sa médiane, interpolation si maque, renversement si positif, trough centré sur zéro)

    Garantie importante :
    - w_std_uV et w_norm ne contiennent jamais NaN/inf.
    - les waveforms invalides sont mises à zéro et marquées waveform_cache_ok=False.

    Retour :
        w_std_uV
        w_norm
        sign_flipped_for_analysis
        trough_time_ms
        interp_out_of_bounds_fraction
        waveform_cache_ok
    """

    if target_time_ms is None:
        target_time_ms = np.linspace(-1.0, 2.0, 121)

    target_time_ms = np.asarray(target_time_ms, dtype=float)
    n_target = len(target_time_ms)

    waveform = np.asarray(waveform, dtype=float).ravel()

    sign_flipped_for_analysis = False
    trough_time_ms = np.nan
    interp_out_of_bounds_fraction = 1.0
    waveform_cache_ok = False

    # Cas waveform absente ou trop courte.
    if waveform.size < 5:
        return (
            np.zeros(n_target, dtype=float),
            np.zeros(n_target, dtype=float),
            sign_flipped_for_analysis,
            trough_time_ms,
            interp_out_of_bounds_fraction,
            waveform_cache_ok,
        )

    finite0 = np.isfinite(waveform)

    if finite0.sum() < 5:
        return (
            np.zeros(n_target, dtype=float),
            np.zeros(n_target, dtype=float),
            sign_flipped_for_analysis,
            trough_time_ms,
            interp_out_of_bounds_fraction,
            waveform_cache_ok,
        )

    # Remplace les valeurs non finies par interpolation simple sur l'index.
    idx = np.arange(waveform.size)
    waveform_clean = waveform.copy()

    if not np.all(finite0):
        waveform_clean[~finite0] = np.interp(
            idx[~finite0],
            idx[finite0],
            waveform[finite0],
        )

    # Temps natif relatif au début de fenêtre.
    native_time_ms = (np.arange(len(waveform_clean)) / float(fs)) * 1000.0 - float(ms_before)

    # Baseline sur les premières 0.2 ms.
    n_base = max(3, int((0.20 / 1000.0) * float(fs)))
    n_base = min(n_base, len(waveform_clean))

    baseline = np.median(waveform_clean[:n_base])
    waveform_clean = waveform_clean - baseline

    # Si tout est plat après baseline.
    amp0 = np.max(np.abs(waveform_clean))
    if not np.isfinite(amp0) or amp0 <= eps:
        return (
            np.zeros(n_target, dtype=float),
            np.zeros(n_target, dtype=float),
            sign_flipped_for_analysis,
            trough_time_ms,
            interp_out_of_bounds_fraction,
            waveform_cache_ok,
        )

    # Orientation analytique : trough négatif.
    if force_negative_trough and abs(np.max(waveform_clean)) > abs(np.min(waveform_clean)):
        waveform_clean = -waveform_clean
        sign_flipped_for_analysis = True

    trough_idx = int(np.argmin(waveform_clean))
    trough_time_ms = float(native_time_ms[trough_idx])

    # Aligne le trough à 0 ms.
    aligned_time_ms = native_time_ms - trough_time_ms

    t_min = float(np.min(aligned_time_ms))
    t_max = float(np.max(aligned_time_ms))

    out_of_bounds = (target_time_ms < t_min) | (target_time_ms > t_max)
    interp_out_of_bounds_fraction = float(np.mean(out_of_bounds))

    # Interpolation sans NaN aux bords.
    order = np.argsort(aligned_time_ms)
    x = aligned_time_ms[order]
    y = waveform_clean[order]

    w_std_uV = np.interp(
        target_time_ms,
        x,
        y,
        left=y[0],
        right=y[-1],
    )

    # Sécurité absolue.
    w_std_uV = np.nan_to_num(
        w_std_uV,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    denom = np.max(np.abs(w_std_uV))

    if not np.isfinite(denom) or denom <= eps:
        w_norm = np.zeros_like(w_std_uV)
        waveform_cache_ok = False
    else:
        w_norm = w_std_uV / denom
        w_norm = np.nan_to_num(
            w_norm,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        waveform_cache_ok = bool(
            np.all(np.isfinite(w_norm))
            and np.max(np.abs(w_norm)) > 0
            and interp_out_of_bounds_fraction <= 0.10
        )

    return (
        w_std_uV,
        w_norm,
        sign_flipped_for_analysis,
        trough_time_ms,
        interp_out_of_bounds_fraction,
        waveform_cache_ok,
    )


############### Exporting/storing  features & waveforms ###############

def save_celltype_cache_session(
    patient,
    session,
    features_df,
    waveforms_standardized_uV,
    waveforms_normalized,
    unit_keys,
    time_ms,
    root,
):
    """
    Sauvegarde le cache cell type d'une session.

    features_df :
        Table une ligne par unit.
    waveforms_standardized_uV :
        Waveforms alignées/rééchantillonnées, en µV.
    waveforms_normalized :
        Même chose, mais normalisées en amplitude pour WaveMAP.
    unit_keys :
        Identifiants stables, ex: P64_BR34|stimic3|clu12.
    time_ms :
        Grille temporelle commune.
    """

    paths = get_celltype_cache_paths(root)
    session_name = f"{patient}_stim{session}"

    features_path = (
        paths["features_per_session"]
        / f"{session_name}_celltype_features.parquet"
    )

    features_xlsx_path = (
        paths["features_per_session"]
        / f"{session_name}_celltype_features.xlsx"
    )

    waveforms_path = (
        paths["waveforms_per_session"]
        / f"{session_name}_waveforms_standardized.npz"
    )

    # Stored in two formats: parquet for quicker reading etc; xlsx for visual inspection
    features_df.to_parquet(features_path, index=False)
    features_df.to_excel(features_xlsx_path, index=False)

    np.savez_compressed(
        waveforms_path,
        unit_keys=np.asarray(unit_keys, dtype=str),
        time_ms=np.asarray(time_ms, dtype=float),
        waveforms_standardized_uV=np.asarray(waveforms_standardized_uV, dtype=float),
        waveforms_normalized=np.asarray(waveforms_normalized, dtype=float),
    )

    print(f"[celltype] saved features: {features_path}")
    print(f"[celltype] saved waveforms: {waveforms_path}")

    return features_path, waveforms_path


def get_celltype_cache_paths(root):
    # pour éviter d'ecrire les chemins partout 

    cache_root = Path(root) / "Spike-sorting" / "CellTypeCache"

    paths = {
        "cache_root": cache_root,
        "features_per_session": cache_root / "features" / "per_session",
        "waveforms_per_session": cache_root / "waveforms" / "per_session",
        "features_all": cache_root / "features" / "celltype_features_all_sessions.parquet",
        "features_all_xlsx": cache_root / "features" / "celltype_features_all_sessions.xlsx",
        "waveforms_all": cache_root / "waveforms" / "waveforms_standardized_all_units.npz",
        "manifest": cache_root / "manifest_celltype_extraction.csv",
        "models": cache_root / "models",
        "qc": cache_root / "qc",
    }

    for p in paths.values():
        if p.suffix == "":
            p.mkdir(parents=True, exist_ok=True)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)

    return paths


############### Calcul et sauvegarde des features et waveforms par session ###############

def celltype_features_session(
    patient,
    session,
    root="/media/aube/Aube/",
    ms_before=1.5,
    ms_after=2.5,
    max_spikes_per_unit=1000,
    target_time_ms=None,
    save=True,
):
    """
    Extraction lourde à faire une seule fois depuis le disque avec .dat.

    Pour une session :
    - charge recording + sorting par électrode
    - retire les dead periods
    - calcule waveforms/templates/quality metrics
    - extrait métriques waveform
    - sauvegarde waveforms standardisées pour WaveMAP
    """
    mapping_anat = pd.read_csv(root + 'Spike-sorting/Data_folders/'+patient+'/mapping_anat_'+patient+'.txt', sep=',', engine='python')
    dict_elec2deadfile = get_dict_deadfiles(mapping_anat, patient, session, root + 'Spike-sorting/Data_folders/'+patient+'/'+patient+'_stim'+session+'/', get_SR(patient))
    spikes = get_nwb(patient, session, root)
    dict_clu2tt = get_dict_tetrodeName_from_tetrodeIndex(spikes, mapping_anat)

    if target_time_ms is None:
        target_time_ms = np.linspace(-1.0, 2.0, 121)

    mapping_local = mapping_anat.copy()
    mapping_local["electrode"] = [tt[:-1] for tt in mapping_local["tt"].tolist()]

    elec_to_tt = (
        mapping_local
        .groupby("electrode")["clu"]
        .apply(lambda x: sorted(set(x.astype(int))))
        .to_dict()
    )

    all_features = []
    all_waveforms_standardized_uV = []
    all_waveforms_normalized = []
    all_unit_keys = []

    mapping_check_rows = []

    for elec, ind_tt in elec_to_tt.items():
        print(f"[celltype] {patient} stim{session} | electrode={elec} | tt={ind_tt}")

        try:
            recording_elec_f, sorting_clean = RecFiltered_Sort_SI_remapped_elec(
                patient=patient,
                session=session,
                elec=elec,
                clus=ind_tt,
                mapping_anat=mapping_anat,
                dict_elec2deadfile=dict_elec2deadfile,
                root=root,
            )
        except Exception as e:
            print(f"WARNING: loading failed for {patient} stim{session} {elec}: {e}")
            mapping_check_rows.append(
                {
                    "patient": patient,
                    "session": f"stimic{session}",
                    "electrode": elec,
                    "status": "loading_failed",
                    "error": str(e),
                }
            )
            continue

        if len(sorting_clean.unit_ids) == 0:
            mapping_check_rows.append(
                {
                    "patient": patient,
                    "session": f"stimic{session}",
                    "electrode": elec,
                    "status": "no_units",
                    "n_units_si": 0,
                    "n_clu_spikes": 0,
                }
            )
            continue

        # Remapping SI unit_id -> ton clu, par ordre dans l'électrode.
        spikes_clu_this_elec = sorted(
            [clu for clu, tt in dict_clu2tt.items() if tt[:-1] == elec]
        )
        si_unit_ids = sorted(list(sorting_clean.unit_ids))

        match_ok = len(spikes_clu_this_elec) == len(si_unit_ids)

        if not match_ok:
            print(
                f"WARNING: mismatch {patient} stim{session} {elec}: "
                f"{len(spikes_clu_this_elec)} clu vs {len(si_unit_ids)} SI units"
            )
            n = min(len(spikes_clu_this_elec), len(si_unit_ids))
            spikes_clu_this_elec = spikes_clu_this_elec[:n]
            si_unit_ids = si_unit_ids[:n]

        unit_id_to_clu = dict(zip(si_unit_ids, spikes_clu_this_elec))

        mapping_check_rows.append(
            {
                "patient": patient,
                "session": f"stimic{session}",
                "electrode": elec,
                "status": "ok" if match_ok else "warning_unit_mapping",
                "n_units_si": len(sorting_clean.unit_ids),
                "n_clu_spikes": len([clu for clu, tt in dict_clu2tt.items() if tt[:-1] == elec]),
                "n_used": len(unit_id_to_clu),
                "si_unit_ids": str(si_unit_ids),
                "spikes_clu_ids": str(spikes_clu_this_elec),
            }
        )

        # SortingAnalyzer.
        try:
            analyzer = si.create_sorting_analyzer(
                sorting_clean,
                recording_elec_f,
                format="memory",
                sparse=False,
                return_in_uV=True,
            )

            analyzer.compute(
                "random_spikes",
                method="uniform",
                max_spikes_per_unit=max_spikes_per_unit,
                seed=0,
            )
            analyzer.compute("waveforms", ms_before=ms_before, ms_after=ms_after, n_jobs=1)
            analyzer.compute("templates")
            analyzer.compute("noise_levels")

            try:
                qm = analyzer.compute("quality_metrics").get_data()
            except Exception as e:
                print(f"WARNING: quality_metrics failed for {elec}: {e}")
                qm = pd.DataFrame(index=sorting_clean.unit_ids)

        except Exception as e:
            print(f"WARNING: analyzer failed for {patient} stim{session} {elec}: {e}")
            continue

        # Features + waveforms.
        try:
            (
                feat_elec,
                W_std_elec,
                W_norm_elec,
                unit_keys_elec,
            ) = _extract_features_and_waveforms_from_templates(
                analyzer=analyzer,
                sorting_clean=sorting_clean,
                recording_elec_f=recording_elec_f,
                patient=patient,
                session=session,
                elec=elec,
                unit_id_to_clu=unit_id_to_clu,
                ms_before=ms_before,
                target_time_ms=target_time_ms,
            )
        except Exception as e:
            print(f"WARNING: waveform extraction failed for {elec}: {e}")
            continue

        if len(feat_elec) == 0:
            continue

        # Ajout quality metrics.
        qm = qm.copy()
        qm["unit_id_si"] = qm.index
        qm = qm[qm["unit_id_si"].isin(unit_id_to_clu.keys())].copy()
        qm["clu"] = qm["unit_id_si"].map(unit_id_to_clu)

        feat_elec = feat_elec.merge(
            qm,
            on=["clu", "unit_id_si"],
            how="left",
            suffixes=("", "_qm"),
        )

        feat_elec["tetrode"] = feat_elec["clu"].map(dict_clu2tt)

        all_features.append(feat_elec)
        all_waveforms_standardized_uV.append(W_std_elec)
        all_waveforms_normalized.append(W_norm_elec)
        all_unit_keys.append(unit_keys_elec)

    if len(all_features) == 0:
        features_df = pd.DataFrame()
        W_std = np.empty((0, len(target_time_ms)))
        W_norm = np.empty((0, len(target_time_ms)))
        unit_keys = np.asarray([], dtype=str)
    else:
        features_df = pd.concat(all_features, ignore_index=True)
        W_std = np.vstack(all_waveforms_standardized_uV)
        W_norm = np.vstack(all_waveforms_normalized)
        unit_keys = np.concatenate(all_unit_keys)

    # Vérification cruciale : ordre table == ordre waveforms.
    if len(features_df) != W_norm.shape[0]:
        raise RuntimeError(
            f"Mismatch features/waveforms: "
            f"{len(features_df)} rows vs {W_norm.shape[0]} waveforms"
        )

    if len(features_df) > 0:
        if not np.all(features_df["unit_key"].astype(str).values == unit_keys.astype(str)):
            raise RuntimeError("Mismatch unit_key entre features_df et waveforms.")

    if save:
        save_celltype_cache_session(
            patient=patient,
            session=session,
            features_df=features_df,
            waveforms_standardized_uV=W_std,
            waveforms_normalized=W_norm,
            unit_keys=unit_keys,
            time_ms=target_time_ms,
            root=root,
        )

        # Sauvegarde du contrôle de mapping.
        paths = get_celltype_cache_paths(root)
        mapping_check = pd.DataFrame(mapping_check_rows)
        mapping_check_path = (
            paths["features_per_session"]
            / f"{patient}_stim{session}_celltype_mapping_check.csv"
        )
        mapping_check.to_csv(mapping_check_path, index=False)
        print(f"[celltype] saved mapping check: {mapping_check_path}")

    return features_df



############### Lancement par session ###############

def celltype_session_cache_exists(patient, session, root): # Manifest, pour ne pas retraiter une session lourde
    paths = get_celltype_cache_paths(root)

    session_name = f"{patient}_stim{session}"

    feature_path = paths["features_per_session"] / f"{session_name}_celltype_features.parquet"
    waveform_path = paths["waveforms_per_session"] / f"{session_name}_waveforms_standardized.npz"

    return feature_path.exists() and waveform_path.exists()


def extract_celltype_cache_session_if_needed(
    patient,
    session,
    root,
    overwrite=False,
):
    """
    À appeler dans le notebook session par session pour calculer/stocker les features/waveforms.
    Si overwrite=False et cache déjà présent :
        ne rouvre pas le .dat.
    Si overwrite=True :
        refait l'extraction lourde.
    """
    if celltype_session_cache_exists(patient, session, root) and not overwrite:
        print(f"[celltype] cache déjà présent : {patient} stim{session}")
        return None

    print(f"[celltype] extraction lourde : {patient} stim{session}")

    return celltype_features_session(
        patient=patient,
        session=session,
        root=root,
        save=True,
    )



############### Merge des résultats des différentes sessions ###############

def build_celltype_global_cache(root): 
    # concatène tous les caches session. se lance seulement qd besoin de 
    # reconstruire la base globale après plusieurs sessions nouvellement extraites.
    # Crée/met à jour les fichiers: 
        # "celltype_features_all_sessions.parquet" et "celltype_features_all_sessions.xlsx", dans cache_root/features  
        # "waveforms_standardized_all_units.npz", dans cache_root/waveforms
    # à partir des concaténations des fichiers : 
        # ...celltype_features.parquet, dans cache_root/features/per_session
        # ..._waveforms_standardized.npz, dans cache_root/waveforms/per_session

    paths = get_celltype_cache_paths(root)

    feature_files = sorted(paths["features_per_session"].glob("*_celltype_features.parquet"))
    waveform_files = sorted(paths["waveforms_per_session"].glob("*_waveforms_standardized.npz"))

    all_features = []
    all_W = []
    all_keys = []
    time_ms_ref = None

    for fp in feature_files:
        all_features.append(pd.read_parquet(fp))

    for wp in waveform_files:
        z = np.load(wp, allow_pickle=True)

        W = z["waveforms_normalized"]
        keys = z["unit_keys"]
        time_ms = z["time_ms"]

        if time_ms_ref is None:
            time_ms_ref = time_ms
        else:
            if not np.allclose(time_ms_ref, time_ms):
                raise ValueError(f"Grille temporelle différente dans {wp}")

        all_W.append(W)
        all_keys.append(keys)

    features_all = pd.concat(all_features, ignore_index=True)
    W_all = np.vstack(all_W)
    unit_keys_all = np.concatenate(all_keys)

    features_all.to_parquet(paths["features_all"], index=False)
    features_all.to_excel(paths["features_all_xlsx"], index=False)

    np.savez_compressed(
        paths["waveforms_all"],
        waveforms_normalized=W_all,
        unit_keys=unit_keys_all,
        time_ms=time_ms_ref,
    )

    print(f"[celltype] features globales : {paths['features_all']}")
    print(f"[celltype] waveforms globales : {paths['waveforms_all']}")

    return features_all, W_all, unit_keys_all, time_ms_ref


##################################################
# Cell-type labelling
##################################################


def classify_celltypes_gmm_global(
    features_all,
    min_num_spikes=100,
    min_snr=3.0,
    min_presence_ratio=0.5,
    max_isi_violations_ratio=0.5,
    confidence_threshold=0.70,
):
    df = features_all.copy()

    required = ["trough_to_peak_ms", "peak_half_width_ms"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante : {col}")

    df["quality_ok_celltype"] = True

    if "num_spikes" in df.columns:
        df["quality_ok_celltype"] &= df["num_spikes"].fillna(0) >= min_num_spikes
    if "snr" in df.columns:
        df["quality_ok_celltype"] &= df["snr"].fillna(0) >= min_snr
    if "presence_ratio" in df.columns:
        df["quality_ok_celltype"] &= df["presence_ratio"].fillna(0) >= min_presence_ratio
    if "isi_violations_ratio" in df.columns:
        df["quality_ok_celltype"] &= df["isi_violations_ratio"].fillna(np.inf) <= max_isi_violations_ratio

    df["waveform_feature_ok"] = (
        df["trough_to_peak_ms"].between(0.05, 2.0)
        & df["peak_half_width_ms"].between(0.02, 2.0)
    )

    mask = (
        df["quality_ok_celltype"]
        & df["waveform_feature_ok"]
        & df["trough_to_peak_ms"].notna()
        & df["peak_half_width_ms"].notna()
    )

    df["putative_cell_type"] = "unclassified_low_quality"

    X = df.loc[mask, ["trough_to_peak_ms", "peak_half_width_ms"]].to_numpy()

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
    cluster = gmm.fit_predict(Xz)
    proba = gmm.predict_proba(Xz)

    df.loc[mask, "gmm_cluster"] = cluster
    df.loc[mask, "gmm_confidence"] = proba.max(axis=1)

    tmp = df.loc[mask].copy()
    fs_cluster = tmp.groupby("gmm_cluster")["trough_to_peak_ms"].median().idxmin()

    df.loc[mask, "putative_cell_type"] = np.where(
        df.loc[mask, "gmm_cluster"] == fs_cluster,
        "putative_FS_inhibitory",
        "putative_RS_excitatory",
    )

    df.loc[mask & (df["gmm_confidence"] < confidence_threshold), "putative_cell_type"] = "uncertain_boundary"

    df["cell_type_classifier"] = "GMM_2D_trough_to_peak_peak_half_width"

    return df, gmm, scaler


def run_and_save_celltype_gmm_global(root): # sauvegarde les labels globaux

    import pickle

    paths = get_celltype_cache_paths(root)

    features_all = pd.read_parquet(paths["features_all"])

    labels_all, gmm, scaler = classify_celltypes_gmm_global(features_all)

    labels_path = paths["cache_root"] / "features" / "celltype_labels_all_sessions.parquet"
    labels_xlsx = paths["cache_root"] / "features" / "celltype_labels_all_sessions.xlsx"

    labels_all.to_parquet(labels_path, index=False)
    labels_all.to_excel(labels_xlsx, index=False)

    with open(paths["models"] / "gmm_celltype_model.pkl", "wb") as f:
        pickle.dump(gmm, f)

    with open(paths["models"] / "gmm_celltype_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(labels_all["putative_cell_type"].value_counts(dropna=False))

    return labels_all


##################################################
# Plot waveform/features
##################################################

def plot_waveform_metric_matrix(labels_all, root):
    import numpy as np
    import matplotlib.pyplot as plt

    paths = get_celltype_cache_paths(root)

    metrics = [
        "trough_to_peak_ms",
        "peak_half_width_ms",
        "trough_half_width_ms",
        "waveform_ptp_uV",
        "peak_trough_ratio",
        "repolarization_slope_uV_per_ms",
        "snr",
        "num_spikes",
    ]

    metrics = [m for m in metrics if m in labels_all.columns]

    n = len(metrics)
    if n == 0:
        raise ValueError("Aucune métrique disponible.")

    df = labels_all.copy()

    fig, axes = plt.subplots(n, n, figsize=(2.2 * n, 2.2 * n))

    labels = (
        df["putative_cell_type"].fillna("unlabelled").unique()
        if "putative_cell_type" in df.columns
        else ["all"]
    )

    for i, y in enumerate(metrics):
        for j, x in enumerate(metrics):
            ax = axes[i, j]

            if i == j:
                vals = df[x].replace([np.inf, -np.inf], np.nan).dropna()
                ax.hist(vals, bins=30)
            else:
                for lab in labels:
                    if "putative_cell_type" in df.columns:
                        sub = df[df["putative_cell_type"].fillna("unlabelled") == lab]
                    else:
                        sub = df

                    ax.scatter(
                        sub[x],
                        sub[y],
                        s=8,
                        alpha=0.5,
                        label=lab if (i == 0 and j == 1) else None,
                    )

            if i == n - 1:
                ax.set_xlabel(x, fontsize=8)
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(y, fontsize=8)
            else:
                ax.set_yticklabels([])

    handles, lab_names = axes[0, 1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, lab_names, loc="upper right", fontsize=8)

    plt.tight_layout()
    out = paths["qc"] / "waveform_metric_matrix.png"
    plt.savefig(out, dpi=300)
    plt.show()

    print(f"Saved: {out}")

def plot_peyrache_scatter(labels_all, root): # Scatter valley-to-peak × half-peak width
    import matplotlib.pyplot as plt

    paths = get_celltype_cache_paths(root)

    df = labels_all.copy()
    df = df[
        df["trough_to_peak_ms"].notna()
        & df["peak_half_width_ms"].notna()
    ]

    plt.figure(figsize=(6, 5))

    if "putative_cell_type" in df.columns:
        for lab, sub in df.groupby("putative_cell_type"):
            plt.scatter(
                sub["trough_to_peak_ms"],
                sub["peak_half_width_ms"],
                s=15,
                alpha=0.7,
                label=lab,
            )
    else:
        plt.scatter(
            df["trough_to_peak_ms"],
            df["peak_half_width_ms"],
            s=15,
            alpha=0.7,
        )

    plt.xlabel("Valley/trough-to-peak width (ms)")
    plt.ylabel("Peak half-width (ms)")
    plt.legend(fontsize=8)
    plt.tight_layout()

    out = paths["qc"] / "peyrache_AB_scatter_vp_phw.png"
    plt.savefig(out, dpi=300)
    plt.show()

    print(f"Saved: {out}")

def plot_peyrache_waveforms_by_class(root):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    paths = get_celltype_cache_paths(root)

    labels_path = paths["cache_root"] / "features" / "celltype_labels_all_sessions.parquet"
    labels = pd.read_parquet(labels_path)

    z = np.load(paths["waveforms_all"], allow_pickle=True)
    W = z["waveforms_normalized"]
    time_ms = z["time_ms"]
    keys = z["unit_keys"]

    if "unit_key" not in labels.columns:
        raise ValueError("unit_key absent de labels.")

    # Vérification ordre.
    if not np.all(labels["unit_key"].astype(str).values == keys.astype(str)):
        raise ValueError("Mismatch ordre labels / waveforms_all.")

    plt.figure(figsize=(6, 4))

    for lab, sub in labels.groupby("putative_cell_type"):
        idx = sub.index.values
        W_lab = W[idx]

        valid = np.all(np.isfinite(W_lab), axis=1)
        W_lab = W_lab[valid]

        if W_lab.shape[0] == 0:
            continue

        mean = np.nanmean(W_lab, axis=0)
        sd = np.nanstd(W_lab, axis=0)

        plt.plot(time_ms, mean, label=f"{lab} n={W_lab.shape[0]}")
        plt.fill_between(time_ms, mean - sd, mean + sd, alpha=0.2)

    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Time from trough (ms)")
    plt.ylabel("Normalized waveform")
    plt.legend(fontsize=8)
    plt.tight_layout()

    out = paths["qc"] / "peyrache_C_average_waveforms_by_class.png"
    plt.savefig(out, dpi=300)
    plt.show()

    print(f"Saved: {out}")

def plot_peyrache_firing_rate(labels_all, root):
    import numpy as np
    import matplotlib.pyplot as plt

    paths = get_celltype_cache_paths(root)

    df = labels_all.copy()

    fr_col = None
    for c in ["firing_rate", "fr_global", "fr_baseline"]:
        if c in df.columns:
            fr_col = c
            break

    if fr_col is None:
        raise ValueError("Aucune colonne firing rate trouvée.")

    classes = list(df["putative_cell_type"].dropna().unique())

    plt.figure(figsize=(6, 4))

    data = []
    labs = []

    for lab in classes:
        vals = df.loc[df["putative_cell_type"] == lab, fr_col]
        vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
        vals = vals[vals >= 0]

        if len(vals) > 0:
            data.append(vals.values)
            labs.append(lab)

    plt.boxplot(data, labels=labs, showfliers=False)
    plt.yscale("log")
    plt.ylabel(f"{fr_col} (Hz, log scale)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out = paths["qc"] / "peyrache_D_firing_rate_by_class.png"
    plt.savefig(out, dpi=300)
    plt.show()

    print(f"Saved: {out}")


def run_celltype_qc_plots(root):

    paths = get_celltype_cache_paths(root)

    labels_path = paths["cache_root"] / "features" / "celltype_labels_all_sessions.parquet"
    labels_all = pd.read_parquet(labels_path)
    plot_peyrache_scatter(labels_all, root)
    plot_waveform_metric_matrix(labels_all, root)
    plot_peyrache_waveforms_by_class(root)
    plot_peyrache_firing_rate(labels_all, root)

    print("[celltype QC] done")


##################################################
# Merge cell types/features with summary tables
##################################################

# def add_celltypes_to_summary_tables(root): # DOUBLON AVEC merge_celltypes_into_neuronal_df etc ?
#     paths = get_celltype_cache_paths(root)

#     labels_path = paths["cache_root"] / "features" / "celltype_labels_all_sessions.parquet"
#     labels = pd.read_parquet(labels_path)

#     tables_dir = Path(root) / "Spike-sorting" / "Tables"

#     label_cols = [
#         "patient",
#         "session",
#         "clu",
#         "putative_cell_type",
#         "gmm_cluster",
#         "gmm_confidence",
#         "quality_ok_celltype",
#         "trough_to_peak_ms",
#         "peak_half_width_ms",
#         "trough_half_width_ms",
#         "waveform_ptp_uV",
#         "snr",
#         "num_spikes",
#         "presence_ratio",
#         "isi_violations_ratio",
#         "cell_type_classifier",
#         "cell_type_interpretation",
#     ]

#     label_cols = [c for c in label_cols if c in labels.columns]
#     labels_small = labels[label_cols].copy()

#     summary_path = tables_dir / "summary_by_nrn_all_sessions.xlsx"
#     general_path = tables_dir / "general_summary_all_sessions.xlsx"

#     if summary_path.exists():
#         summary = pd.read_excel(summary_path)
#         summary2 = summary.merge(labels_small, on=["patient", "session", "clu"], how="left")
#         summary2.to_excel(tables_dir / "summary_by_nrn_all_sessions_with_celltypes.xlsx", index=False)

#     if general_path.exists():
#         general = pd.read_excel(general_path)
#         general2 = general.merge(labels_small, on=["patient", "session", "clu"], how="left")
#         general2.to_excel(tables_dir / "general_summary_all_sessions_with_celltypes.xlsx", index=False)


def _normalize_session_label_for_merge(x):
    """
    Normalise les sessions pour merger :
    - '3'       -> 'stimic3'
    - 3         -> 'stimic3'
    - 'stim3'   -> 'stimic3'
    - 'stimic3' -> 'stimic3'
    """
    import pandas as pd

    if pd.isna(x):
        return x

    s = str(x).strip()

    if s.startswith("stimic"):
        return s

    if s.startswith("stim"):
        return "stimic" + s.replace("stim", "", 1)

    return "stimic" + s


def merge_celltypes_into_neuronal_df( # fonction noyau, merge en mémoire sur un DataFrame
    neuronal_df,
    celltype_labels,
    keep_celltype_cols=None,
    remove_existing_celltype_cols=True,
    verbose=True,
):
    """
    Merge une table neuronale avec les labels cell type.

    Clés :
        patient, session, clu

    neuronal_df :
        DataFrame avec au moins patient/session/clu.

    celltype_labels :
        DataFrame globale issue de run_and_save_celltype_gmm_global().

    Retour :
        neuronal_df enrichie.
    """

    df = neuronal_df.copy()
    labels = celltype_labels.copy()

    required_keys = ["patient", "session", "clu"]

    for k in required_keys:
        if k not in df.columns:
            raise ValueError(f"Colonne absente de neuronal_df : {k}")
        if k not in labels.columns:
            raise ValueError(f"Colonne absente de celltype_labels : {k}")

    df["session"] = df["session"].apply(_normalize_session_label_for_merge)
    labels["session"] = labels["session"].apply(_normalize_session_label_for_merge)

    # Harmonise patient et clu.
    df["patient"] = df["patient"].astype(str)
    labels["patient"] = labels["patient"].astype(str)

    df["clu"] = pd.to_numeric(df["clu"], errors="coerce")
    labels["clu"] = pd.to_numeric(labels["clu"], errors="coerce")

    if keep_celltype_cols is None:
        keep_celltype_cols = [
            "patient",
            "session",
            "clu",
            "unit_key",
            "putative_cell_type",
            "gmm_cluster",
            "gmm_confidence",
            "quality_ok_celltype",
            "waveform_feature_ok",
            "cell_type_classifier",
            "cell_type_interpretation",

            # waveform metrics
            "trough_to_peak_ms",
            "peak_half_width_ms",
            "trough_half_width_ms",
            "waveform_ptp_uV",
            "trough_amp_uV",
            "peak_amp_uV",
            "peak_trough_ratio",
            "repolarization_slope_uV_per_ms",

            # acquisition / cache
            "sampling_rate_hz",
            "source_polarity_multiplier",
            "source_polarity_corrected",
            "waveform_sign_flipped_after_source_correction",
            "interp_out_of_bounds_fraction",
            "waveform_cache_ok",

            # quality metrics sauvegardées dans celltype cache
            "num_spikes",
            "firing_rate",
            "presence_ratio",
            "snr",
            "isi_violations_ratio",

            # wavemap, si déjà calculé et mergé dans labels
            "wavemap_umap_1",
            "wavemap_umap_2",
            "wavemap_cluster",
        ]

    keep_celltype_cols = [c for c in keep_celltype_cols if c in labels.columns]
    labels_small = labels[keep_celltype_cols].copy()

    # Une seule ligne par patient/session/clu côté labels.
    dup = labels_small.duplicated(["patient", "session", "clu"], keep=False)

    if dup.any() and verbose:
        print(
            "[merge_celltypes] WARNING: doublons dans celltype_labels pour "
            f"{dup.sum()} lignes. Conservation de la dernière occurrence."
        )

    labels_small = (
        labels_small
        .dropna(subset=["patient", "session", "clu"])
        .drop_duplicates(["patient", "session", "clu"], keep="last")
    )

    celltype_value_cols = [
        c for c in labels_small.columns
        if c not in ["patient", "session", "clu"]
    ]

    if remove_existing_celltype_cols:
        cols_to_drop = [c for c in celltype_value_cols if c in df.columns]
        if len(cols_to_drop) > 0 and verbose:
            print(f"[merge_celltypes] Colonnes remplacées : {cols_to_drop}")
        df = df.drop(columns=cols_to_drop, errors="ignore")

    merged = df.merge(
        labels_small,
        on=["patient", "session", "clu"],
        how="left",
        validate="many_to_one",
    )

    if verbose:
        n_rows = len(merged)
        n_matched = merged["putative_cell_type"].notna().sum() if "putative_cell_type" in merged.columns else 0
        print(
            f"[merge_celltypes] matched rows: {n_matched}/{n_rows} "
            f"({100 * n_matched / n_rows:.1f}%)"
        )

        if "putative_cell_type" in merged.columns:
            print(merged["putative_cell_type"].value_counts(dropna=False).to_string())

    return merged


def load_global_celltype_labels(root):
    """
    Charge la table globale des cell types depuis CellTypeCache.
    Essaie parquet puis xlsx.
    """

    paths = get_celltype_cache_paths(root)

    parquet_path = paths["cache_root"] / "features" / "celltype_labels_all_sessions.parquet"
    xlsx_path = paths["cache_root"] / "features" / "celltype_labels_all_sessions.xlsx"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    if xlsx_path.exists():
        return pd.read_excel(xlsx_path)

    raise FileNotFoundError(
        f"Aucune table de labels trouvée :\n"
        f"- {parquet_path}\n"
        f"- {xlsx_path}"
    )


def merge_global_neuronal_summaries_with_celltypes( # wrapper disque : lit les tables globales, appelle la fonction noyau, sauvegarde
    root,
    summary_filename="summary_by_nrn_all_sessions.xlsx",
    general_filename="general_summary_all_sessions.xlsx",
    save_parquet=True,
    verbose=True,
):
    """
    Merge les tables globales summary_by_nrn et general_summary
    avec la table globale des cell types.

    À lancer après :
        - build_celltype_global_cache(root)
        - run_and_save_celltype_gmm_global(root)
    """

    tables_dir = Path(root) / "Spike-sorting" / "Tables"

    summary_path = tables_dir / summary_filename
    general_path = tables_dir / general_filename

    labels = load_global_celltype_labels(root)

    outputs = {}

    if summary_path.exists():
        summary = pd.read_excel(summary_path)

        summary_merged = merge_celltypes_into_neuronal_df(
            neuronal_df=summary,
            celltype_labels=labels,
            verbose=verbose,
        )

        out_xlsx = tables_dir / summary_filename.replace(".xlsx", "_with_celltypes.xlsx")
        summary_merged.to_excel(out_xlsx, index=False)
        outputs["summary_xlsx"] = out_xlsx

        if save_parquet:
            out_parquet = tables_dir / summary_filename.replace(".xlsx", "_with_celltypes.parquet")
            summary_merged.to_parquet(out_parquet, index=False)
            outputs["summary_parquet"] = out_parquet

        if verbose:
            print(f"[merge_celltypes] Saved: {out_xlsx}")
    else:
        if verbose:
            print(f"[merge_celltypes] WARNING: fichier absent : {summary_path}")

    if general_path.exists():
        general = pd.read_excel(general_path)

        general_merged = merge_celltypes_into_neuronal_df(
            neuronal_df=general,
            celltype_labels=labels,
            verbose=verbose,
        )

        out_xlsx = tables_dir / general_filename.replace(".xlsx", "_with_celltypes.xlsx")
        general_merged.to_excel(out_xlsx, index=False)
        outputs["general_xlsx"] = out_xlsx

        if save_parquet:
            out_parquet = tables_dir / general_filename.replace(".xlsx", "_with_celltypes.parquet")
            general_merged.to_parquet(out_parquet, index=False)
            outputs["general_parquet"] = out_parquet

        if verbose:
            print(f"[merge_celltypes] Saved: {out_xlsx}")
    else:
        if verbose:
            print(f"[merge_celltypes] WARNING: fichier absent : {general_path}")

    return outputs


def run_wavemap_from_cache(root):
    import umap
    import networkx as nx
    import community as community_louvain

    paths = get_celltype_cache_paths(root)

    features = pd.read_parquet(paths["features_all"])
    z = np.load(paths["waveforms_all"], allow_pickle=True)

    W = z["waveforms_normalized"]
    unit_keys = z["unit_keys"]

    mask = (
        features["num_spikes"].fillna(0).ge(100)
        & features["snr"].fillna(0).ge(3.0)
        & features["trough_to_peak_ms"].notna()
    )

    W_good = W[mask.values]
    features_good = features.loc[mask].copy()
    unit_keys_good = unit_keys[mask.values]

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.0,
        metric="euclidean",
        random_state=0,
    )

    emb = reducer.fit_transform(W_good)

    G = nx.from_scipy_sparse_array(reducer.graph_)
    partition = community_louvain.best_partition(
        G,
        weight="weight",
        resolution=1.5,
        random_state=0,
    )

    wavemap_cluster = np.array([partition[i] for i in range(len(unit_keys_good))])

    out = features_good.copy()
    out["unit_key"] = unit_keys_good
    out["wavemap_umap_1"] = emb[:, 0]
    out["wavemap_umap_2"] = emb[:, 1]
    out["wavemap_cluster"] = wavemap_cluster

    out_path = paths["models"] / "wavemap_labels.parquet"
    out.to_parquet(out_path, index=False)

    return out, reducer


# def quality_metrics_session(patient, session, mapping_anat, dict_elec2deadfile, dict_clu2tt, root='/media/aube/Aube/'):
#     """
#     Renvoie un tableau avec les quality metrics de SpikeInterface par unit. Besoin de Recording filtré et Sorting, re-mappé avec retrait des dead periods

#     Les quality metrics sont calculées électrode par électrode, sur :
#     - un recording déjà filtré
#     - un sorting déjà "remappé" après retrait / concaténation des dead periods

#     Paramètres
#     ----------
#     patient : str. Identifiant patient.
#     session : str. Identifiant de session.
#     mapping_anat : pd.DataFrame
#         Table de correspondance anatomique / électrodes. NB : la colonne `clu` de ce fichier désigne l'indice de tétrode dans le connect.
#     dict_elec2deadfile : dict
#         Dictionnaire electrode -> deadfile.
#         Ici les deadfiles sont utilisés par `RecFiltered_Sort_SI_remapped_elec`
#         pour reconstruire un recording/sorting sans périodes mortes.
#     dict_clu2tt : dict
#         Mapping : clu (ID neurone dans `spikes`) -> nom de tétrode (ex. 'vgc2')
#     root : str. Racine du projet.

#     Retour
#     ------
#     qm_all : pd.DataFrame
#         DataFrame concaténé pour toutes les électrodes.
#         L'index final de ce DataFrame est remappé sur les IDs `clu` de `spikes`, pour permettre ensuite un accès direct de type : qm.loc[clu, metric]

#     NB :
#     ----
#     Le remapping `qm.index -> clu_spikes` repose sur l'hypothèse suivante : à l'intérieur d'une même électrode, l'ordre des unités renvoyées par 
#     SpikeInterface (`sorting_clean.unit_ids` / `qm.index`) correspond à l'ordre des neurones dans `spikes` sur cette même électrode.
#     Si cette hypothèse devient fausse dans une future version du pipeline, l'appariement sera incorrect.
#     """
#     import spikeinterface as si 

#     list_col_qm = ['firing_rate', 'amplitude_median', 'num_spikes', 'presence_ratio', 'amplitude_cutoff', 'snr', 'isi_violations_ratio'] # liste des qm a exporter

# ## 1) run per electrode:
#     all_qm = [] # ce sera une liste de dataframes de QM (un par electrode)

#     mapping_anat["electrode"] = [tt[:-1] for tt in mapping_anat["tt"].tolist()] # depuis dgl2 renvoie dgl
#     # Pour chaque électrode (ex. 'dtp', 'vgc', 'vof'), on récupère la liste des indices des tétrodes associés. Ex: 'dtp' -> [3, 4, 5]
#     elec_to_tt = mapping_anat.groupby("electrode")["clu"].apply(lambda x: sorted(set(x.astype(int)))).to_dict()

#     for elec, ind_tt in elec_to_tt.items(): #  par ex. : dtp, [3, 4]
#         # import d'un rec_filtré et sorting re-mappés avec good chunks concaténés
#         recording_elec_f, sorting_clean = RecFiltered_Sort_SI_remapped_elec(patient, session, elec, ind_tt, mapping_anat, dict_elec2deadfile, root)
        
#         if len(sorting_clean.unit_ids) == 0: # Si aucune unité n'est présente sur cette électrode, on passe à l'électrode suivante.
#             continue

#         # Construction du SortingAnalyzer SpikeInterface :
#         # format="memory" : toutes les données intermédiaires sont stockées en mémoire.
#         analyzer = si.create_sorting_analyzer(sorting_clean, recording_elec_f, format="memory", sparse=False)
#         # Calculs nécessaires avant les quality metrics :
#         analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500, seed=0)
#         analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0, n_jobs=1)
#         analyzer.compute("templates")
#         analyzer.compute("noise_levels")
        
#         qm = analyzer.compute("quality_metrics").get_data() # Récupération du tableau de QM
#         list_col_qm_exported = [col for col in list_col_qm if col in list(qm.columns)]
#         qm = qm[list_col_qm_exported]

#         # -----------------------------
#         # Remapping des IDs d'unités
#         # -----------------------------
#         # `qm.index` correspond aux unit_ids de `sorting_clean`.
#         # Ces IDs ne correspondent pas directement aux IDs de neurones utilisés dans `spikes` / `dict_clu2tt`.
#         # On reconstruit donc un mapping par ordre, à l'intérieur de l'électrode.
#         # Liste des IDs `clu` (côté `spikes`) qui appartiennent à cette électrode.
#         # Exemple :
#         #   pour elec='vgc', récupérer tous les neurones dont dict_clu2tt[clu] est
#         #   'vgc1', 'vgc2', ou 'vgc3' -> après troncature tt[:-1] == 'vgc'

#         spikes_clu_this_elec = sorted([clu for clu, tt in dict_clu2tt.items() if tt[:-1] == elec]) # liste des clu sur l'electrode
#         qm_unit_ids = sorted(qm.index.tolist()) # unités présentes dans sorting_clean
#         # Vérification de cohérence :
#         # si le nombre d'unités n'est pas le même entre les neurones trouvés dans `spikes` VS les unités trouvées dans `qm`,
#         # alors on tronque par prudence à la plus petite longueur.
#         # Cela évite un crash, mais il peut y avoir ici une discordance réelle entre les deux représentations.
#         if len(spikes_clu_this_elec) != len(qm_unit_ids):
#             print(f"WARNING: mismatch on {elec}: {len(spikes_clu_this_elec)} neurones spikes vs {len(qm_unit_ids)} unités QM")
#             n = min(len(spikes_clu_this_elec), len(qm_unit_ids))
#             spikes_clu_this_elec = spikes_clu_this_elec[:n]
#             qm_unit_ids = qm_unit_ids[:n]
#             qm = qm.loc[qm_unit_ids].copy()

#         # mapping ordre par ordre : unit_id_qm -> clu_spikes
#         # Exemple :
#         #   qm_unit_ids         = [12, 14, 15]
#         #   spikes_clu_this_elec = [7, 8, 9]
#         #   => 12->7, 14->8, 15->9
#         dict_qm_to_spikes = dict(zip(qm_unit_ids, spikes_clu_this_elec))
#         # On garde explicitement la correspondance comme colonne informative :
#         qm["clu_spikes"] = [dict_qm_to_spikes[u] for u in qm.index]
#         # On remplace l'index par les IDs `clu` de `spikes`. Permettra plus tard de faire qm.loc[clu, metric]
#         qm.index = qm["clu_spikes"].values
#         all_qm.append(qm)

# ## 2) fusion des QM de toutes les electrodes
#     qm_all = pd.concat(all_qm, axis=0)
    
#     return qm_all