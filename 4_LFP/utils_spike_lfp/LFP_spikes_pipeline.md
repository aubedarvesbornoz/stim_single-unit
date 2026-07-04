# Pipeline LFP–spikes SEIC : fichiers requis et fichiers générés

Ce document récapitule les fichiers nécessaires et les fichiers générés pour les quatre étapes du pipeline :

1. Hilbert run : calcul des enveloppes Hilbert par bande ;
2. Hilbert stats : statistiques LFP (modulation selon bande de fréquence et conditions) ;
3. Common preprocess : construction de la base commune micro–macro ;
4. Power correlations : corrélations firing rate single-unit × power/enveloppe Hilbert.

Les noms ci-dessous utilisent la convention :

```text
PATIENT = P119_FM71
SESSION = P119_FM71_stim4
session_num = 4
```

---

## Résumé minimal des dépendances entre étapes

```text
Hilbert run
    nécessite : TRC + txt macro corrected + txt micro COG + tableur bad channels
    génère   : results_hilbert/SESSION/SESSION_hilbert_<band>.npy

Hilbert stats
    nécessite : results_hilbert/SESSION/
    génère   : results_hilbert_stats/

Common preprocess
    nécessite : txt micro COG + txt macro corrected + hilbert exports + NWB 
    génère   : LFP_spikes/common_preprocess/SESSION/

Power correlations
    nécessite : common_preprocess/SESSION + hilbert exports + NWB + mapping_anat + deadfiles
    génère   : results_fr_power_corr/<variant>/per_session/SESSION/
               results_fr_power_corr/<variant>/pooled_across_sessions/
```

---

## 1. Hilbert run

### Objectif

Calculer, pour chaque session macro LFP, les enveloppes Hilbert par bande fréquentielle, puis les époker autour des stimulations. Les sorties servent ensuite aux stats Hilbert et aux corrélations spike–power.

### Dossier d’entrée macro recommandé

```text
LFP/
├── SESSION.TRC
├── SESSION_stim_events_TRC_corrected.txt                nécessaire, sinon créé, à partir de *stim_events_TRC + *stim_events_TRC_shifted + *stim_events_TRC_re-shifted_loca_COG
├── SESSION_stim_events_TRC_re-shifted_loca_COG.txt
└── TRC_bad_channels.xlsx                                
```

### Fichiers nécessaires (contenu)

#### 1. Fichier TRC

```text
LFP/SESSION.TRC
```
Exemple :

```text
LFP/P119_FM71_stim4.TRC
```
Contient le signal LFP macro brut.

#### 2. Fichier d’événements macro corrigé

```text
LFP/SESSION_stim_events_TRC_corrected.txt
```
Colonnes attendues :
```text
label_stim	t_start	duration	t_end	correction_start	macro_part	macro_part_index	macro_event_index
```
Exemple :
```text
Tp3-Tp42.0mA7.0Hz1025µsec	307.656	9.86499	317.52099	0.0	P119_FM71_stim4	0	0
```
Ce fichier fournit les temps macro fiables utilisés pour l’extraction LFP.
Créé tout seul si absent, dans utils_time_frequency.lfp_preprocess_utils, fonction recover_precise_macro_stim_events()


#### 3. Fichier COG / localisation

```text
LFP/SESSION_stim_events_TRC_re-shifted_loca_COG.txt
```
Colonnes/logique attendues :
```text
stim;t_start;duration;lobe;cog
```
Exemple :
```text
CU_8-CU_92.0mA10.0Hz1025µsec,1125.78,9.90005,R Occipital;['aphasie']
```
Ce fichier fournit les labels cognitifs et la localisation/lobe des stimulations.

#### 4. Table des mauvais canaux macro

```text
LFP/TRC_bad_channels.xlsx
```
ou format équivalent attendu par les fonctions de preprocessing LFP.

Sert à exclure les contacts macro défectueux avant bipolarisation et analyse.

### Fichiers générés par session Hilbert

Dossier de sortie :

```text
results_hilbert/
└── SESSION/
```

Contenu attendu :

```text
results_hilbert/SESSION/
├── SESSION_metadata.json
├── SESSION_trial_table.csv
├── SESSION_times.npy
├── SESSION_hilbert_theta.npy
├── SESSION_hilbert_alpha.npy
├── SESSION_hilbert_beta.npy
├── SESSION_hilbert_low_gamma.npy
└── SESSION_hilbert_high_gamma.npy
```

#### SESSION_metadata.json

Contient notamment :

```text
session
config
n_trials
raw_ch_names
bad_channels
bipolar_names
main_band_to_subbands
```

La clé `bipolar_names` est importante pour les analyses locales/distantes et les corrélations spike–LFP.

#### SESSION_trial_table.csv

Table des stimulations utilisée par le pipeline Hilbert.

Contient au minimum :

```text
label_stim
t_start
duration
t_end
lobe
cog
group_label
cog_labels
stim_shaft
```

#### SESSION_times.npy

Axe temporel relatif des epochs Hilbert.

Convention :

```text
times < 0 : fenêtre pré-stimulation
times >= 0 : fenêtre post-stimulation, alignée après fin de stimulation + epsilon
```

#### SESSION_hilbert_<band>.npy

Tenseur par bande :

```text
shape = (n_trials, n_channels, n_times)
```

Ces fichiers sont les entrées directes des stats Hilbert et des corrélations firing rate × LFP power.

### Fichiers globaux générés par le run Hilbert

```text
results_hilbert/
└── run_summary_hilbert.json
```

Contient la config utilisée, le nombre de sessions traitées et les erreurs éventuelles.

---

## 2. Hilbert stats

### Objectif

Tester les modulations LFP par bande, condition cognitive et localité, à partir des exports Hilbert déjà générés.

L’unité d’observation est :
```text
session × stim × channel
```
Les tests portent sur l’axe temporel, car les fréquences ont déjà été agrégées par bande.

### Dossier d’entrée

```text
results_hilbert/
└── SESSION/
    ├── SESSION_metadata.json
    ├── SESSION_trial_table.csv
    ├── SESSION_times.npy
    └── SESSION_hilbert_<band>.npy
```

### Fichiers nécessaires

Pour chaque session :

```text
SESSION_metadata.json
SESSION_stim_table.csv
SESSION_times.npy
SESSION_hilbert_theta.npy
SESSION_hilbert_alpha.npy
SESSION_hilbert_beta.npy
SESSION_hilbert_low_gamma.npy
SESSION_hilbert_high_gamma.npy
```

### Fichiers générés : stats pooled

Structure :

```text
results_hilbert_stats/
└── pooled_across_sessions/
    ├── hilbert_stats_config.json
    ├── summary_hilbert_stats.csv
    ├── condition_main/
    │   ├── cog+/
    │   │   ├── local/
    │   │   │   └── <band>/
    │   │   └── distant/
    │   │       └── <band>/
    │   ├── negatif/
    │   └── controle/
    └── condition_subcategories/
        ├── cog__aphasie/
        ├── cog__souvenir/
        └── ...
```

Dans chaque dossier condition × localité × bande :

```text
mean_trace.npy
median_trace.npy
observations_table.csv
stat_wilcoxon.npy
pvals_wilcoxon.npy
pvals_wilcoxon_fdr.npy
sig_mask_wilcoxon_fdr.npy
T_obs_cluster.npy
sig_mask_cluster.npy
cluster_pvals.npy
figure_wilcoxon_fdr.png
figure_cluster.png
```

Selon la config, certains fichiers peuvent être absents si une méthode statistique est désactivée.

### Fichiers générés : stats par session

Si `also_run_per_session=True` :

```text
results_hilbert_stats/
├── pooled_across_sessions/
...
└── SESSION/
    ├── hilbert_stats_config.json
    ├── summary_hilbert_stats.csv
    ├── condition_main/
    └── condition_subcategories/
```

---

## 3. Common preprocess (single-unit / LFP)

### Objectif

Construire une base commune micro–macro par session, utilisée ensuite par les analyses spike–power et spike–phase.

Cette étape garde deux référentiels temporels séparés :

```text
micro : temps des spikes et fichiers unitaires
macro : temps TRC/Hilbert
```

### Dossiers d’entrée

#### Micro

```text
root_micro/
└── Spike-sorting/
    └── Data_folders/
        └── PATIENT/
            └── SESSION/
                ├── SESSION_stim_events_TRC_re-shifted_loca_COG.txt
                ├── SESSION.nwb 
                └── derivatives/
                    ├── SESSION_deadfile_<elec>_in_ts.txt
                    └── ...
```

ou structure alternative acceptée par les fonctions de recherche :

```text
root_micro/
└── Data_folders/
    └── PATIENT/
        └── SESSION/
```

#### Macro

```text
macro_root/
└── SESSION_stim_events_TRC_corrected.txt
```

#### Hilbert

```text
hilbert_root/
└── SESSION/
    ├── SESSION_metadata.json
    ├── SESSION_trial_table.csv
    ├── SESSION_times.npy
    └── SESSION_hilbert_<band>.npy
```

### Fichiers nécessaires côté micro

#### 1. Fichier COG micro

```text
SESSION_stim_events_TRC_re-shifted_loca_COG.txt
```

Colonnes/logique :
```text
stim;t_start;duration;lobe;cog
```

Utilisé pour :

```text
t_start_micro
duration_micro
t_end_micro
lobe
cog
group_label
cog_labels
```

#### 2. Fichier NWB

```text
SESSION.nwb
```

Non strictement nécessaire pour construire le common preprocess, mais recommandé si `require_existing_nwb=True` afin de ne préparer que les sessions exploitables pour les spikes.

#### 3. Mapping anatomique micro

```text
root_micro/Spike-sorting/Data_folders/PATIENT/mapping_anat_PATIENT.txt
```

Utilisé ensuite pour associer unités, tétrondes, électrodes hybrides et deadfiles.

#### 4. Deadfiles micro

```text
root_micro/Spike-sorting/Data_folders/PATIENT/SESSION/derivatives/SESSION_deadfile_<elec>_in_ts.txt
```

Exemples :

```text
P63_AK33_stim1_deadfile_vcr_in_ts.txt
P112_MV65_stim3_deadfile_dr_in_ts.txt
```

Ces fichiers contiennent des intervalles morts en indices de samples. Ils sont convertis en secondes via le sampling rate micro.

### Fichiers nécessaires côté macro

```text
macro_root/SESSION_stim_events_TRC_corrected.txt
```

Colonnes :

```text
label_stim
t_start
duration
t_end
correction_start
macro_part
macro_part_index
macro_event_index
```

### Fichiers générés par session

Dossier de sortie recommandé :

```text
LFP_spikes/
└── common_preprocess/
    └── SESSION/
        ├── SESSION_common_trials.csv
        └── SESSION_common_metadata.json
```

#### SESSION_common_trials.csv

Table maître. Une ligne = une stimulation.

Colonnes principales :

```text
session
stim_index
label_stim
t_start_macro
duration_macro
t_end_macro
t_start_micro
duration_micro
t_end_micro
lobe
cog
group_label
cog_labels
stim_bipolar_label
stim_shaft
stim_contact_pair
stim_contact_1
stim_contact_2
stim_intensity
stim_frequency
macro_part
macro_part_index
macro_event_index
micro_event_index
micro_macro_offset_s
t_start_macro_predicted_from_micro_offset
offset_residual_start_s
offset_used_for_qc
pre_start_micro
pre_end_micro
post_start_micro
post_end_micro
pre_start_macro
pre_end_macro
post_start_macro
post_end_macro
pre_length
post_length
epsilon
```

#### SESSION_common_metadata.json

Contient :

```text
patient
session_num
session_name
paths
offset
n_trials
columns
hilbert_loaded
hilbert_bands_loaded
```

### Fichiers globaux générés par le batch common preprocess

```text
LFP_spikes/
└── common_preprocess/
    ├── run_all_common_preprocess_sessions.csv
    └── run_all_common_preprocess_summary.json
```

#### run_all_common_preprocess_sessions.csv

Une ligne par session candidate :

```text
session
patient
session_num
status
nwb_file
out_dir
n_trials
group_counts
offset_s
offset_residual_max_abs_s
hilbert_loaded
error
```

#### run_all_common_preprocess_summary.json

Résumé global : config, sessions OK, sessions skipped, erreurs, dossiers générés.

---

## 4. Power correlations : firing rate × Hilbert power

### Objectif

Corréler, pour chaque unité, contact LFP, bande, bin temporel et condition :

```text
FR normalisé par baseline pré-stim
vs
variation de power/enveloppe Hilbert normalisée par baseline pré-stim
```

Analyse principale recommandée :

```text
FR : logratio post/pré
LFP : lfp_power_pre_logratio
corrélation : Spearman
bins : 100 ms et 500 ms
```

### Dossiers d’entrée

#### Common preprocess

```text
LFP_spikes/common_preprocess/SESSION/
├── SESSION_common_trials.csv
└── SESSION_common_metadata.json
```

#### Hilbert exports

Référencés dans `SESSION_common_metadata.json`, clé :

```text
paths.hilbert_root
```

Fichiers utilisés :

```text
results_hilbert/SESSION/
├── SESSION_metadata.json
├── SESSION_trial_table.csv
├── SESSION_times.npy
└── SESSION_hilbert_<band>.npy
```

#### NWB micro

```text
root_micro/Spike-sorting/Data_folders/PATIENT/SESSION/SESSION.nwb
```

#### Deadfiles micro

```text
root_micro/Spike-sorting/Data_folders/PATIENT/SESSION/derivatives/SESSION_deadfile_<elec>_in_ts.txt
```

#### Mapping anatomique micro

```text
root_micro/Spike-sorting/Data_folders/PATIENT/mapping_anat_PATIENT.txt
```

Utilisé pour relier unité/tétrode à l’électrode hybride, puis au deadfile correspondant.

### Fichiers générés par variante

Dossier racine :

```text
results_fr_power_corr/
```

Chaque combinaison FR × LFP est séparée :

```text
results_fr_power_corr/
├── fr_logratio__lfp_pre_logratio/
├── fr_zscore__lfp_pre_zscore/
└── run_all_fr_power_variants_summary.csv
```

### Fichiers générés par session

```text
results_fr_power_corr/
└── fr_logratio__lfp_pre_logratio/
    └── per_session/
        └── SESSION/
            ├── figures_significant_histograms/
            ├── SESSION_fr_power_corr_config.json
            ├── SESSION_fr_bins.csv
            ├── SESSION_fr_power_trial_bins_theta.csv
            ├── SESSION_fr_power_trial_bins_alpha.csv
            ├── SESSION_fr_power_trial_bins_beta.csv
            ├── SESSION_fr_power_trial_bins_low_gamma.csv
            ├── SESSION_fr_power_trial_bins_high_gamma.csv
            ├── SESSION_fr_power_correlations.csv
            ├── SESSION_fr_power_correlation_signif.csv
            └── SESSION_fr_power_corr_run_summary.json
```

#### SESSION_fr_bins.csv

Table FR seule.

Une ligne = unité × trial × bin temporel.

Colonnes importantes :

```text
session
unit_id
trial_idx
stim_index
group_label
cog_labels
label_stim
stim_shaft
bin_width_ms
time_bin_id
window
bin_start_rel
bin_end_rel
bin_center_rel
bin_abs_start_micro
bin_abs_end_micro
n_spikes
fr_hz
fr_hz_smooth
fr_pre_mean_hz
fr_pre_sd_hz
fr_norm
fr_norm_method
dead_overlap_s
valid_bin_fraction
bin_valid_after_deadfile
```

Si un bin chevauche une dead period de la tétrade du neurone et que l’option stricte est activée, alors `fr_hz` et `fr_norm` sont NaN et le bin est exclu des corrélations.

#### SESSION_fr_power_trial_bins_*band*.csv

Table longue FR × LFP pour une bande.

Une ligne = unité × trial × channel × band × bin.

Colonnes importantes :

```text
unit_id
trial_idx
channel_name
channel_idx
band
locality
group_label
cog_labels
bin_width_ms
window
bin_center_rel
fr_norm
lfp_power_bin_mean
lfp_power_pre_subtract
lfp_power_pre_percent
lfp_power_pre_logratio
lfp_power_pre_zscore
```

Cette table peut être volumineuse. Elle est optionnelle via `save_trial_bin_tables`.

#### SESSION_fr_power_correlations.csv

Table principale des résultats.

Une ligne = une corrélation calculée pour :

```text
session
unit_id
band
channel
bin_width
window
time_bin
corr_grouping
corr_condition
method
```

Colonnes principales :

```text
session
unit_id
band
channel_idx
channel_name
channel_shaft
bin_width_ms
window
time_bin_id
bin_center_rel
corr_grouping
corr_condition
method
fr_norm_method
lfp_value_col
n_trials_total
n_finite_pairs
rho_or_r
p_value
q_value_fdr_bh
fr_mean
fr_sd
lfp_mean
lfp_sd
```

`corr_grouping` indique quel masque statistique a été utilisé :

```text
all
locality
group_label
group_label_x_locality
cog_subcategory
cog_subcategory_x_locality
```

`corr_condition` indique la modalité spécifique dans ce grouping :

```text
all
local
distant
cog+
negatif
controle
cog+::local
negatif::distant
cog::aphasie
cog::aphasie::local
```

#### SESSION_fr_power_correlation_signif.csv

Sous-ensemble de la table principale selon :

```text
p_value < alpha
```

ou selon la colonne configurée, par exemple `q_value_fdr_bh < alpha`.

### Fichiers pooled across sessions

```text
results_fr_power_corr/
└── fr_logratio__lfp_pre_logratio/
    └── pooled_across_sessions/
        ├── figures_significant_histograms/
        ├── ALL_SESSIONS_fr_power_correlations_pooled.csv
        ├── ALL_SESSIONS_fr_power_correlations_pooled_SIGNIF.csv
        ├── ALL_SESSIONS_fr_power_pool_summary.json
        ├── run_all_fr_power_correlations_summary.json
        └── run_all_fr_power_correlations_sessions.csv
```

#### ALL_SESSIONS_fr_power_correlations_pooled.csv

Concaténation des tables `SESSION_fr_power_correlations.csv` de toutes les sessions incluses.

Peut contenir en plus :

```text
q_value_fdr_bh_pooled
```

si la FDR est recalculée après pooling.

#### ALL_SESSIONS_fr_power_correlations_pooled_SIGNIF.csv

Sous-ensemble significatif du pooled.

#### ALL_SESSIONS_fr_power_pool_summary.json

Résumé du pooling : fichiers utilisés, fichiers manquants, nombre de lignes pooled, nombre de lignes significatives, figures produites.

#### run_all_fr_power_correlations_sessions.csv

Une ligne par session :

```text
session
patient
session_num
status
nwb_file
out_dir
n_units
n_correlation_rows
n_significant_rows
error
```

#### run_all_fr_power_variants_summary.csv

Une ligne par variante FR × LFP :

```text
variant
fr_norm_method
lfp_value_col
status
output_root
n_session_rows
n_pooled_rows
n_pooled_significant_rows
error
```

---
