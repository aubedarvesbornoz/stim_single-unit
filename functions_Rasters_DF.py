import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec # affichage rasters
import pynapple as nap
from pynapple.io import interface_nwb
from pynapple.io.interface_nwb import NWBFile
from pathlib import Path
import ast
import seaborn as sb

############################################################
# Get general info on recording
############################################################

def get_SR(patient):
    if int(patient[1:-5]) <= 75: # sampling rate (Hz)
        if int(patient[1:-5]) < 65: # blackrock
            return 30000 
        else: # nlx ancien
            return 16384
    else: # nlx recent
        return 32768
    
def nb_channels(mapping_anat, patient, session, root):
    if os.path.exists(root + 'Spike-sorting/Data_folders/'+patient+'/deadCh_'+patient+'.txt'):
        list_deadCh = pd.read_csv(root + 'Spike-sorting/Data_folders/'+patient+'/deadCh_'+patient+'.txt', sep=';', engine='python').iloc[int(session)-1, 1]
        list_deadCh = ast.literal_eval(list_deadCh)
    else:
        list_deadCh = []
    return 4*len(mapping_anat.index)-len(list_deadCh) # nombre de canaux micros (donc pas le max s'il y a des dead channels)

def get_total_duration(path_folder, patient, session, nCh, dtype = np.int16): # renvoie la durée totale en secondes
    if os.path.exists(path_folder+patient+'_stim'+session+'.dat'): # si on a le dat
        return int(os.path.getsize(path_folder+patient+'_stim'+session+'.dat')/(nCh*np.dtype(dtype).itemsize) / get_SR(patient))
    else: # si on a plus le dat, on va chercher dans l'excel
        metadata = pd.read_excel('C:/Users/darves-bornoz/Documents/article_neuronal_stimic/duration.xlsx')
        return metadata[metadata['patient']==patient][int(session)].tolist()[0]
    
############################################################
# Create or import session-specific files = nwb, stims, deadfiles.
############################################################

def get_nwb(patient, session, root='D:/'):
    '''
    Renvoie le nwb associe a la session
    root = 'D:/' ou 'C:/Users/darves-bornoz/Documents/'
    '''
    path_folder = root + 'Spike-sorting/Data_folders/'+patient+'/'+patient+'_stim'+session+'/'
    files_basename = patient+'_stim'+session
    nwbfile_path = path_folder + files_basename + ".nwb"
    print(path_folder)
    if not os.path.exists(nwbfile_path): # creation du nwb s'il n'existe pas encore
        from datetime import datetime
        from dateutil import tz
        from neuroconv.datainterfaces import NeuroScopeRecordingInterface
        from neuroconv.datainterfaces import NeuroScopeSortingInterface
        xml_path = Path(path_folder + files_basename + ".xml")
        interface = NeuroScopeSortingInterface(folder_path = Path(path_folder), xml_file_path=xml_path, verbose=False)
        metadata = interface.get_metadata()
        session_start_time = datetime(2023, 4, 4, 12, 30, 0, tzinfo=tz.gettz("US/Pacific"))
        metadata["NWBFile"].update(session_start_time=session_start_time)
        interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata)

    spikes = NWBFile(nwbfile_path)["units"]
    
    return spikes

############### Import stimulation characteristics ###############

def import_stims(path_df_stims): # called in get_stims()
    """
    import stim data from .txt file
    """
    stims = pd.read_csv(path_df_stims, header=None)
    stims.columns = [['paramètres', 't', 'durée'] if stims.shape[1]==3 else ['paramètres', 't', 'durée', 'lobe']][0]
    stims['t + durée'] = stims['t'] + stims['durée']
    list_caracs = stims['paramètres']
    list_elec, list_plots, list_freq, list_int = [], [], [], []
    for s in list_caracs:
        list_int = np.append(list_int, s[s.index('mA')-3:s.index('mA')]+' '+s[s.index('mA'):s.index('mA')+2]) # toujours mm nb de caracteres
        list_freq = np.append(list_freq, s[s.index('mA')+2:s.index('Hz')-2]+' '+s[s.index('Hz'):s.index('Hz')+2]  )
        elec_plot = s[:s.index('mA')-3]
        list_elec = np.append(list_elec, re.match(r"([A-Za-z'_]+)(\d+)-([A-Za-z'_]+)(\d+)", elec_plot).group(1))
        list_plots = np.append(list_plots,re.match(r"([A-Za-z'_]+)(\d+)-([A-Za-z'_]+)(\d+)", elec_plot).group(2) +'-'+ re.match(r"([A-Za-z'_]+)(\d+)-([A-Za-z'_]+)(\d+)", elec_plot).group(4))
    stims['electrode'], stims['plots'], stims['frequence'], stims['intensite'] = list_elec,list_plots,list_freq,list_int
    return stims

def get_stims(patient, session, root): 
    """
    return DF with characteritics of each stim (electrode, contacts, time, duration, intensity, frequency, location...)
    """
    path_folder = root + f'Spike-sorting/Data_folders/{patient}/{patient}_stim{session}/'
    if os.path.exists(path_folder+patient+'_stim'+session+'_stim_events_TRC_re-shifted_loca.txt'):
        stims_loca = import_stims(path_folder+patient+'_stim'+session+'_stim_events_TRC_re-shifted_loca.txt')
    else:
        stims_loca = import_stims(path_folder+patient+'_stim'+session+'_stim_events_TRC_shifted_loca.txt')
    return stims_loca

############### Import artefacts characteristics ###############

def merge_overlapping_events(df):
    # Parfois il y a des chevauchements d'evenements dans les deadfiles, donc cette fct harmonise ce format
    df = df.sort_values(by=0).reset_index(drop=True) # pour reset dans ordre croissant au cas ou
    merged = []
    current_start, current_end = df.iloc[0, 0], df.iloc[0, 1] # Pour parcourir les événements

    for i in range(1, len(df)):
        start, end = df.iloc[i, 0], df.iloc[i, 1]
        if start <= current_end: # si chevauchement ou inclusion, fusionner
            current_end = max(current_end, end) # et on garde la fin la plus longue
        else:
            merged.append([current_start, current_end]) # sinon on laisse
            current_start, current_end = start, end

    merged.append([current_start, current_end])
    return pd.DataFrame(merged, columns=[0, 1])

def get_dict_deadfiles(mapping_anat, patient, session, path_folder, sr):
    '''
    Renvoie un dict avec pour chaque electrode le deadfile associe (liste de paires) 
    '''
    dict_elec2deadfile = {} # va chercher tous les deadfiles dans le sous-dossier "derivatives" de la session
    dict_ttInd2tt = {i+1:mapping_anat.loc[i, 'tt'] for i in range(mapping_anat.shape[0])} # numero de fichier klusters vers nom de tt
    for elec in np.unique([tt[:-1] for tt in list(dict_ttInd2tt.values())]):
        dict_elec2deadfile[elec] = merge_overlapping_events(pd.read_csv(path_folder+'derivatives/'+patient+'_stim'+session+'_deadfile_'+elec+'_in_ts.txt', header = None, sep='\t') / sr)
    return dict_elec2deadfile

def get_dict_tetrodeName_from_tetrodeIndex(spikes, mapping_anat):
    '''Renvoie un dict avec pour chaque neuron_index la tetrode associee'''
    dict_clu2tt = {} # indice de neurone vers nom de tt
    dict_ttInd2tt = {i+1:mapping_anat.loc[i, 'tt'] for i in range(mapping_anat.shape[0])} # numero de fichier klusters vers nom de tt (e.g. clu.3, '3' => 'tb1')
    spikes_by_location = spikes.getby_category("group")  # dict : ind_tt vers ensemble des nrn en format TsGroup
    for ttInd in spikes_by_location.keys(): # pr chq ind de tt qui a des neurones
        for clu in spikes_by_location[int(ttInd)].index:
            dict_clu2tt[clu] = dict_ttInd2tt[int(ttInd)]
    return dict_clu2tt


##################################################
# Metrics functions
##################################################

############### Manipulations de noms d'electrodes ###############

def normalize_name(name: str) -> str:
    """
    Normalise les noms d’électrodes.
    Règles :
    - Micro :
        'va', 'va1' → 'A_L'
        'dtp', 'dtp2' → 'TP_R'
    - Macro :
        'A' → 'A_R'
        "B'" → 'B_L'
        'TPp' ou 'TP_p' → 'TP_L'
        'TP' ou 'TP_' → 'TP_R'
        'ABCp' → 'AB_L' (exclusion de la 3e lettre du radical)
    """
    name = name.strip()
    # --- Cas micro (minuscules avec préfixe v/d) ---
    if name.islower():
        side = 'L' if name.startswith('v') else 'R' if name.startswith('d') else None
        if not side:
            raise ValueError(f"Nom micro invalide : {name}")

        core = re.sub(r'^[vd]', '', name)      # enlève le préfixe v/d
        core = re.sub(r'\d+$', '', core)       # enlève le numéro à la fin
        return f"{core.upper()}_{side}"

    # --- Cas macro (majuscules, avec suffixe éventuel) ---
    # On tolère formats : "ABCp", "ABC_p", "ABC'", "ABC", "ABC_"
    if re.match(r"^[A-Z]{1,3}(_?p|_|'|)$", name):
        # Détection du côté
        if name.endswith("p") or name.endswith("_p") or name.endswith("'"):
            side = "L"
            core = re.sub(r"(_?p|'$)", "", name)   # on enlève le suffixe gauche
        else:
            side = "R"
            core = re.sub(r"_$", "", name)         # on enlève "_" final éventuel

        # Si radical fait 3 lettres, on garde seulement les 2 premières
        if len(core) == 3:
            core = core[:2]

        return f"{core}_{side}"

    raise ValueError(f"Nom d’électrode non reconnu : {name}")

def electrodes_equal(name1, name2): # on vérifie si le nom de la tetrode du neurone est le même que celui de la stim
    return normalize_name(name1) == normalize_name(name2)

############### Fonctions pour calcul distance euclidienne ###############

def find_back_macrocontacts_from_tt(norm_name, coord_MNI_pat, verb=False):
    """
    Renvoie les macroplots de l'hybride entre lesquels est situee la tetrode, et la macro selon nomenclature de table_MNI.
    Retrouve l'électrode macro dans list_elec_MNI à partir du nom normalisé. Gère :
    - Côté gauche (suffixes p, ' ou _p)
    - Côté droit (sans suffixe ou avec "_")
    - Radical avec éventuellement une 3e lettre (joker).
    """
    base = norm_name[:-2]  # le radical (sans le suffixe _L/_R)
    list_elec_MNI = coord_MNI_pat['Electrode_name'].unique()
    macro_tt = None
    if norm_name[-1] == 'L':
        # Recherche stricte d'abord
        candidates = [f"{base}p", f"{base}'", f"{base}_p"]
        for c in candidates:
            if c in list_elec_MNI:
                macro_tt =  c
                break # on arrete de parcourir candidates
        # Recherche avec une 3e lettre quelconque
        if macro_tt is None: # si on a toujours pas trouvé c'est peut-etre car il y a une 3e lettre
            pattern = re.compile(rf"^{base}[A-Z]?(p|'|_p)$")
            matches = [elec for elec in list_elec_MNI if pattern.match(elec)] # au cas ou plusieurs electrodes ont leurs 2 premieres lettres identiques
            if len(matches) == 1:
                macro_tt = matches[0]
            elif len(matches) > 1:
                if verb:
                    print(f"⚠️ Plusieurs macros possibles pour {norm_name} : {matches}, on garde {matches[0]}")
                macro_tt = matches[0]
            else:
                if verb:
                    print(NameError(f"Pas de macro correspondant à {norm_name} dans la liste MNI"))
                return [np.nan, np.nan]

    else:  # côté droit
        # Recherche stricte
        candidates = [base, f"{base}_"]
        for c in candidates:
            if c in list_elec_MNI:
                macro_tt =  c
                break # on arrete de parcourir candidates
        # Recherche avec une 3e lettre quelconque
        if macro_tt is None: # si on a toujours pas trouvé c'est peut-etre car il y a une 3e lettre
            pattern = re.compile(rf"^{base}[A-Z]?(_)?$")
            matches = [elec for elec in list_elec_MNI if pattern.match(elec)] # au cas ou plusieurs electrodes ont leurs 2 premieres lettres identiques
            if len(matches) == 1:
                macro_tt = matches[0]
            elif len(matches) > 1:
                if verb:
                    print(f"⚠️ Plusieurs macros possibles pour {norm_name} : {matches}, on garde {matches[0]}")
                macro_tt = matches[0]
            else:
                if verb:
                    print(NameError(f"Pas de macro correspondant à {norm_name} dans la liste MNI"))
                return [np.nan, np.nan]
                
    # Maintenant qu'on a macro_tt, on trouve les plots associés à la tetrode de cette hybride :
    if coord_MNI_pat[coord_MNI_pat['Electrode_name']==macro_tt]['Electrode_model'].unique().tolist() == ['hybride']:
        plots_asso = '1-2'
    elif coord_MNI_pat[coord_MNI_pat['Electrode_name']==macro_tt]['Electrode_model'].unique().tolist() == ['hybride latérale']:
        if coord_MNI_pat[coord_MNI_pat['Electrode_name']==macro_tt]['Number_of_tetrodes'].unique().tolist() == [2]:
            plots_asso = '8-9'
        elif coord_MNI_pat[coord_MNI_pat['Electrode_name']==macro_tt]['Number_of_tetrodes'].unique().tolist() == [3]:
            plots_asso = '5-6'
        else: 
            if verb:
                print('Erreur dans nombre de tétrodes pour la MME')
            return [np.nan, np.nan]
    else: 
        if verb:
            print('Electrode model not recognized:', coord_MNI_pat[coord_MNI_pat['Electrode_name']==macro_tt]['Electrode_model'].unique())
        return np.nan
    return [macro_tt, plots_asso]


def find_back_XYZ_from_macro(coord_MNI_pat, macro, plots, verb=False): 
    '''Renvoie liste des coordonnées MNI des plots demandés. Renvoie np.nan si pas de MNI'''
    if type(macro)==float or type(plots)==float : # parfois des NaN sont fournis en argument
        return np.nan
    norm_name = normalize_name(macro)
    base = norm_name[:-2]  # le radical (sans le suffixe _L/_R)
    list_elec_MNI = coord_MNI_pat['Electrode_name'].unique()
    macro_mni = None
    if norm_name[-1] == 'L':
        # Recherche stricte d'abord
        candidates = [f"{base}p", f"{base}'", f"{base}_p"]
        for c in candidates:
            if c in list_elec_MNI:
                macro_mni =  c
                break # on arrete de parcourir candidates
        # Recherche avec une 3e lettre quelconque
        if macro_mni is None: # si on a toujours pas trouvé c'est peut-etre car il y a une 3e lettre
            pattern = re.compile(rf"^{base}[A-Z]?(p|'|_p)$")
            matches = [elec for elec in list_elec_MNI if pattern.match(elec)] # au cas ou plusieurs electrodes ont leurs 2 premieres lettres identiques
            if len(matches) == 1:
                macro_mni = matches[0]
            elif len(matches) > 1:
                if verb:
                    print(f"⚠️ Plusieurs macros possibles pour {norm_name} : {matches}, on garde {matches[0]}")
                macro_mni = matches[0]
            else:
                if verb:
                    print(NameError(f"Pas de macro correspondant à {norm_name} dans la liste MNI"))
                return np.nan

    else:  # côté droit
        # Recherche stricte
        candidates = [base, f"{base}_"]
        for c in candidates:
            if c in list_elec_MNI:
                macro_mni =  c
                break # on arrete de parcourir candidates
        # Recherche avec une 3e lettre quelconque
        if macro_mni is None: # si on a toujours pas trouvé c'est peut-etre car il y a une 3e lettre
            pattern = re.compile(rf"^{base}[A-Z]?(_)?$")
            matches = [elec for elec in list_elec_MNI if pattern.match(elec)] # au cas ou plusieurs electrodes ont leurs 2 premieres lettres identiques
            if len(matches) == 1:
                macro_mni = matches[0]
            elif len(matches) > 1:
                if verb:
                    print(f"⚠️ Plusieurs macros possibles pour {norm_name} : {matches}, on garde {matches[0]}")
                macro_mni = matches[0]
            else:
                if verb:
                    print(NameError(f"Pas de macro correspondant à {norm_name} dans la liste MNI"))
                return np.nan


    if macro_mni not in coord_MNI_pat['Electrode_name'].unique(): 
        if verb:
            print(ValueError(f"L'electrode {macro_mni} n'existe pas pour ce patient."))
        return np.nan
    if coord_MNI_pat[coord_MNI_pat['Electrode_name']==macro_mni][coord_MNI_pat[coord_MNI_pat['Electrode_name']==macro_mni]['Channel_num']==plots].empty: 
        if verb:
            print(ValueError(f"Le plot {plots} n'existe pas pour l'électrode {macro_mni} dans les données MNI pour ce patient."))
        return np.nan
    elif coord_MNI_pat[coord_MNI_pat['Electrode_name']==macro_mni][coord_MNI_pat[coord_MNI_pat['Electrode_name']==macro_mni]['Channel_num']==plots].shape[0]>1:
        if verb:
            print(ValueError(f"Le plot {plots} pour l'électrode {macro_mni} a plusieurs entrées dans les données MNI pour ce patient."))
        return np.nan
    else:
        return coord_MNI_pat[coord_MNI_pat['Electrode_name']==macro_mni][coord_MNI_pat[coord_MNI_pat['Electrode_name']==macro_mni]['Channel_num']==plots][['X','Y','Z']].to_numpy().tolist()[0]


def euclidian_distance(xyz_tt, xyz_stim,verb=False):
    """
    Renvoie la distance euclidienne en mm entre deux triplets [x, y, z].
    Si un des inputs est NaN/None ou contient un NaN -> renvoie np.nan
    """
    # Cas None
    if xyz_tt is None or xyz_stim is None:
        if verb:
            print("No distance computable: one input is None")
        return np.nan
    
    # Conversion en array numpy
    xyz_tt = np.array(xyz_tt, dtype=float) if not isinstance(xyz_tt, float) else np.array([np.nan])
    xyz_stim = np.array(xyz_stim, dtype=float) if not isinstance(xyz_stim, float) else np.array([np.nan])
    
    # Si un des vecteurs contient un NaN
    if np.any(np.isnan(xyz_tt)) or np.any(np.isnan(xyz_stim)):
        # print("No distance computable because xyz_tt or xyz_stim contains NaN.",
        #       "xyz_stim =", xyz_stim, "xyz_tt =", xyz_tt)
        return np.nan
    
    return np.linalg.norm(xyz_tt - xyz_stim) # Distance euclidienne


############### ZE ou hors ZE ###############

def is_in_ZEZIZPZLNI(patient, elec, plots):
    '''
    Determine si une paire de plots est en ZE/ZI/ZP/ZL/Non-involved ou pas. Renvoie une liste de 5 bool.
    Args:
        patient (str): nom du patient
        elec (str): nom de l'électrode macro
        plots (str): plots de l'électrode macro
    '''
    DF_EZPZIZ = pd.read_excel('C:/Users/darves-bornoz/Documents/Data/ZeZiZpZl_23.10.25_anonyme.xlsx')
    dict_index_zone = {'ZE':1, 'ZI':3, 'ZP':5, 'ZL':7} # indice de colonne avec electrodes de chaque zone
    dict_zone2boolean = {}
    elec = normalize_name(elec) # pour etre sur de pas passer acoté d'une elec si les nomenclaures diffèrent ('p' au lieu d'une apostrophe...)

    if patient in DF_EZPZIZ['Patient'].unique(): # si on a les ZE etc pour ce patient
        table_pat = DF_EZPZIZ[DF_EZPZIZ['Patient']==patient]
        for zone in dict_index_zone.keys() : 
            ind_zone = dict_index_zone[zone]
            if isinstance(table_pat.iloc[:,ind_zone].to_numpy().tolist()[0], float) : # pas de donnée pour cette zone OK
                dict_zone2boolean[zone] = np.nan
            elif table_pat.iloc[:,ind_zone].to_numpy().tolist()[0] =='-': # aucune zone pour ce patient
                dict_zone2boolean[zone] = False
            else : # la liste n'est pas vide
                list_elec_zone = [normalize_name(x.strip()) for x in table_pat.iloc[:,ind_zone].to_numpy().tolist()[0].split(";")] # liste de chq electrode de la zone
                if list_elec_zone == [0]: # pas l'info pour cette zone (et donc pas info sur non-involvement)
                    dict_zone2boolean[zone] = np.nan
                    dict_zone2boolean['NI'] = np.nan
                elif elec in list_elec_zone: # si elec dans liste 
                    elec_ind = list_elec_zone.index(elec) # on recupere l'ind de l'elec dans sa liste pr obtenir les plots associes
                    if ";" in table_pat.iloc[:,ind_zone+1].to_numpy().tolist()[0]: # au moins 2 sets de plots, de 2 electrodes, on va prendre le set correspondant
                        # liste des intervalles des plots de la zone => on prend l'intervalle à l'indice de l'electrode correspondante, on le stocke dans list_plots_zone, qui est donc un str.
                        list_plots_zone = [x.strip() for x in table_pat.iloc[:,ind_zone+1].to_numpy().tolist()[0].split(";")][elec_ind] 
                    else: # un seul ensemble de plots, pour une electrode
                        list_plots_zone = table_pat.iloc[:,ind_zone+1].to_numpy().tolist()[0]
                    # maintenant qu'on a les plots associes a l'elec dans la zone :
                    if list_plots_zone == 'Full': # qq soit la valeur des plots, ils sont dans la zone
                        dict_zone2boolean[zone] = True 
                    else:
                        if "_" in list_plots_zone : # si 2 intervalles de plots :
                            list_plots_zone = list_plots_zone.split("_") # liste de 2 listes, chacune contenant
                            l_p = list_plots_zone[0] # on tente le 1er des 2 invervalles
                            plot_min, plot_max = l_p.split('-')[0], l_p.split('-')[1]
                            if plots.split('-')[0] >= plot_min and plots.split('-')[1] <= plot_max: # si les plots sont dans le 1er intervalle
                                dict_zone2boolean[zone] = True 
                            else : 
                                l_p = list_plots_zone[1]
                                plot_min, plot_max = l_p.split('-')[0], l_p.split('-')[1]
                                if plots.split('-')[0] >= plot_min and plots.split('-')[1] <= plot_max: # si les plots sont dans le 2nd intervalle
                                    dict_zone2boolean[zone] = True 
                                else :  # les plots ne sont dans aucun des 2 intervalles
                                    dict_zone2boolean[zone] = False

                        else: # si un seul intervalle de plots : 
                            plot_min, plot_max = list_plots_zone.split('-')[0], list_plots_zone.split('-')[1]
                            if plots.split('-')[0] >= plot_min and plots.split('-')[1] <= plot_max:
                                dict_zone2boolean[zone] = True 
                            else:
                                dict_zone2boolean[zone] = False
                    
                else: # l'electrode n'est pas dans la liste pour cette zone
                    dict_zone2boolean[zone] = False

        if [dict_zone2boolean[zone] for zone in dict_index_zone.keys()] == [False, False, False, False] : # si on sait qu'on n'est dans aucune des 4 zones OK
            dict_zone2boolean['NI'] = True
        elif True in [dict_zone2boolean[zone] for zone in dict_index_zone.keys()] : # s'il y a au moins un True, alors NI est forcement False OK
            dict_zone2boolean['NI'] = False
        else: # au moins un NaN donc incertain OK
            dict_zone2boolean['NI'] = np.nan
            
    else:  # si on n'a pas les ZE etc pour ce patient OK
        print('patient', patient, 'absent du tableau ZE')
        for zone in dict_index_zone.keys() : 
            dict_zone2boolean[zone] = np.nan
        dict_zone2boolean['NI'] = np.nan

    return list(dict_zone2boolean.values())


############### Creation des tables de resultats par session ###############

def Rec_Sort_SI_remapped_filt_elec(patient, session, elec, clus, mapping_anat, dict_elec2deadfile, root='/media/aube/Aube/'):
    """
    Renvoie un objet Recording et un objet Sorting spikeinterface, a partir des fichiers neuroscope et klusters d'une session
    """
    import probeinterface as pi
    from spikeinterface.extractors import read_neuroscope
    import spikeinterface as si 
    from spikeinterface.preprocessing import bandpass_filter

    path_folder = root + f'Spike-sorting/Data_folders/{patient}/{patient}_stim{session}/'
    
## 1) Charger recording + sorting depuis neuroscope
    # (read_neuroscope suppose que les .res/.clu sont dans le même dossier que le .xml)
    xml_path = f"{path_folder}/{patient}_stim{session}.xml"  
    recording, sorting = read_neuroscope(xml_path, load_recording=True, load_sorting=True)

## 2) charger probe geometry
    n_tetrodes = mapping_anat.shape[0]
    # positions: chaque tetrode = carré 2x2, tétrodes espacées
    spacing_tetrode = 500.0  # µm
    local = np.array([[0,0],[0,20],[20,0],[20,20]], dtype=float)  # 4 contacts proches
    positions = np.zeros((n_tetrodes * 4, 2), dtype=float)
    for t in range(n_tetrodes):
        base = np.array([t * spacing_tetrode, 0.0])
        positions[t*4:(t+1)*4] = local + base
    probe = pi.Probe(ndim=2)
    probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 5})
    probe.set_device_channel_indices(np.arange(n_tetrodes * 4))
    recording = recording.set_probe(probe, in_place=False) # on associe le mapping au recording
    
## 3) Re-construct periods outside deadperiods, for one electrode
# For each electrode: rec + deadfile + re-mapped sorting + QM

# 3.A) utilitary functions:
    def load_dead_intervals_ts(deadfile_path: str, n_frames: int):
        """Depuis deadfile, retourne des intervalles bad (start,end) en frames int64, triés, mergés, clipés."""
        deadF = pd.read_csv(deadfile_path, sep="\t", header=None, names=["start", "end"])
        bad = deadF[["start", "end"]].to_numpy(dtype=np.int64)
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

    def select_units_by_group(sorting, groups):
        groups = set(groups)
        g = sorting.get_property("group")
        unit_ids = [u for u, gg in zip(sorting.unit_ids, g) if int(gg) in groups]
        return sorting.select_units(unit_ids)

# 3.B) run for this electrode:
    deadfile_elec = dict_elec2deadfile[elec]

    # sous-sorting (unités dont group ∈ clus)
    sorting_e = select_units_by_group(sorting, clus)

    # sous-recording : canaux de l'electrode seulement
    ch=[]
    for tt in clus :
        ch.append([str(tt*4-4), str(tt*4-3), str(tt*4-2), str(tt*4-1)])
    ch = [x for xs in ch for x in xs] # on applatit la liste de listes en liste
    recording_e = recording.select_channels(ch) 

    # recording concaténé selon deadfile
    n = recording_e.get_num_frames()
    bad = load_dead_intervals_ts(deadfile_elec, n_frames=n)
    good_chunks = invert_intervals_to_good(bad, n_frames=n)
    recording_e_clean = build_concat_recording(recording_e, good_chunks)

    # filtre signal (300-3000 Hz) avant calcul des metriques
    recording_elec_f = bandpass_filter(recording=recording_e_clean, freq_min=300, freq_max=3000)

    # remap spikes vers périodes concaténées
    sr = get_SR(patient)
    sorting_clean = remap_sorting_to_concat(sorting_e, good_chunks, sr)

    return recording_elec_f, sorting_clean


def quality_metrics_session(patient, session, elec, clus, mapping_anat, dict_elec2deadfile, root='/media/aube/Aube/'):
    """
    Renvoie un tableau avec les quality metrics par unit. Besoin de Recording filtré et Sorting, sans les dead periods
    """
    import spikeinterface as si 

    list_col_qm = ['firing_rate', 'amplitude_median', 'num_spikes', 'presence_ratio', 'amplitude_cutoff' 'snr', 'isi_violations_ratio'] # liste des qm a exporter

## 1) run per electrode:
    all_qm = [] # ce sera une liste de dataframes de QM (un par electrode)

    mapping_anat["electrode"] = [tt[:-1] for tt in mapping_anat["tt"].tolist()] # depuis dgl2 renvoie dgl
    elec_to_clus = mapping_anat.groupby("electrode")["clu"].apply(lambda x: sorted(set(x.astype(int)))).to_dict()

    for elec, clus in elec_to_clus.items():

        recording_elec_f, sorting_clean = Rec_Sort_SI_remapped_filt_elec(patient, session, elec, clus, mapping_anat, dict_elec2deadfile, root='/media/aube/Aube/')

        if len(sorting_clean.unit_ids) == 0:
            continue

        # analyzer + qm
        analyzer = si.create_sorting_analyzer(sorting_clean, recording_elec_f, format="memory", sparse=False)
        analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500, seed=0)
        analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0, n_jobs=1)
        analyzer.compute("templates")
        analyzer.compute("noise_levels")
        qm = analyzer.compute("quality_metrics").get_data()
        qm = qm[list_col_qm]
        all_qm.append(qm)

## 2) fusion des QM de toutes les electrodes
    qm_all = pd.concat(all_qm, axis=0)
    
    return qm_all


def compute_neuronal_summary(spikes, stims_loca, dict_clu2tt, dict_elec2deadfile, mpg, patient, session, root, mapping_anat, start_baseline=0, end_baseline=300, epsilon=0.1, verb=False, bin_z=0.05, bin_resp=[0.05, 0.075, 0.1]):
    '''Cree un summary_session_df et general_session_df pour une session : summary_by_neuron et general_summary_by_neuron_and_stim
    epsilon : valeur tres petite pour calculer le log-ratio

    '''
    # Initialisation
    path_folder = root + f'Spike-sorting/Data_folders/{patient}/{patient}_stim{session}/'
    stimic_session='stimic'+session
    all_clu_ids = list(dict_clu2tt.keys()) # liste des indices de tous les neurones
    data, general_data = [], []
    labels_stims = stims_loca['paramètres'] # labels des stimulations
    coord_MNI = pd.read_excel('C:/Users/darves-bornoz/Documents/Data/MNI_all_patients.xlsx')
    coord_MNI_pat = coord_MNI[coord_MNI['Patient']==patient] # coordonnées MNI pour ce patient

    # variables pour obtenir dynamiques temporelles par U de temps
    pre_duration, post_duration = 10, 10  # sec
    
    # on charge les quality metrics de chaque SU de la session
    qm = quality_metrics_session(patient, session, mapping_anat, dict_elec2deadfile, root)
    qm.reset_index()

    for ind_clu, clu in enumerate(all_clu_ids): # pour chaque neurone clu
        
        spk_times = spikes[clu].index.dropna().values  # Liste des t des spikes de ce neurone en sec
        
        if len(spk_times) == 0: # si aucun spike, on passe au neurone suivant
            continue
        
        # général :
        row = {'patient': patient, 'session':stimic_session, 'clu': clu, 'tetrode': dict_clu2tt[clu],
               'lobe_tt':mpg.loc[mpg['tt'] == dict_clu2tt[clu], 'lobe'].values[0],
               'loca_tt':mpg.loc[mpg['tt'] == dict_clu2tt[clu], 'loca'].values[0]} # Initialisation de la ligne de ce neurone
        
        # quality metrics :
        for metric in qm.columns.tolist():
            row[metric] = qm.loc[ind_clu, metric]

        # fr_global / Taux de décharge global :
        deadfile_elec = dict_elec2deadfile[dict_clu2tt[clu][:-1]]  # deadfile de l'electrode correspondante
        total_duration = get_total_duration(path_folder, patient, session, nb_channels(mapping_anat, patient, session, root)) - np.sum(deadfile_elec[1] - deadfile_elec[0]) # on soustrait deadperiods
        row['fr_global'] = len(spk_times) / (total_duration ) if total_duration > 0 else np.nan # nb de spikes / durée totale
        # print('global: total_duration,dead,',total_duration,np.sum(deadfile_elec[1] - deadfile_elec[0]))

        # fr_baseline / Taux de décharge sur une baseline des premières minutes (0 à 300s par défaut) :
        artefacts_filtered_baseline = deadfile_elec[(deadfile_elec[1] >= start_baseline) & (deadfile_elec[0] <= end_baseline)]
        dead_baseline = np.sum(np.minimum(artefacts_filtered_baseline[1], end_baseline) - np.maximum(artefacts_filtered_baseline[0], start_baseline))  
        spk_in_baseline = spk_times[spk_times <= end_baseline] # tous les spikes avant la fin de la baseline
        spk_in_baseline = spk_in_baseline[spk_in_baseline >= start_baseline] # tous les spikes après le début de la baseline
        row['fr_baseline'] = len(spk_in_baseline) / (end_baseline-start_baseline-dead_baseline) if (end_baseline-start_baseline-dead_baseline) > 0 else np.nan
        # print('baseline: nb spikes, dead ', len(spk_in_baseline), dead_baseline)

        # Responsiveness: est-ce que le clu repond a au moins une stim sur toute la session ? bool
        responsive_clu = [False for _ in range(len(bin_resp))]
        # inhib_only, excit_only, inhib_then_excit, excit_then_inhib = [], [], [], []

        # tt_in_ZE, etc 
        ttZE, ttZI, ttZP, ttZL, ttNI = is_in_ZEZIZPZLNI(patient, dict_clu2tt[clu][:-1], find_back_macrocontacts_from_tt(normalize_name(dict_clu2tt[clu][:-1]), coord_MNI_pat,verb=verb)[1])
        row['tt_en_ZE'],row['tt_en_ZI'],row['tt_en_ZP'],row['tt_en_ZL'],row['tt_en_NI'] = ttZE, ttZI, ttZP, ttZL, ttNI
            
        for i, stim in stims_loca.iterrows(): # Pour chaque stim:
            
            # fr_pre, fr_post / Taux de décharge pré-/post-stim et % de variation :
            t_pre_start = stim['t'] - pre_duration
            t_pre_end = stim['t']
            t_post_start = stim['t'] + stim['durée']
            t_post_end = stim['t'] + stim['durée'] + post_duration
            pre_spike_times = spk_times[(spk_times >= t_pre_start) & (spk_times < t_pre_end)]
            post_spike_times = spk_times[(spk_times >= t_post_start) & (spk_times < t_post_end)]

            artefacts_filtered_pre = deadfile_elec[(deadfile_elec[1] >= t_pre_start) & (deadfile_elec[0] <= t_pre_end)]
            artefacts_filtered_post = deadfile_elec[(deadfile_elec[1] >= t_post_start) & (deadfile_elec[0] <= t_post_end)]
            dead_pre = np.sum(np.minimum(artefacts_filtered_pre[1], t_pre_end) - np.maximum(artefacts_filtered_pre[0], t_pre_start)) 
            dead_post = np.sum(np.minimum(artefacts_filtered_post[1], t_post_end) - np.maximum(artefacts_filtered_post[0], t_post_start))  
            fr_pre = len(pre_spike_times) / (pre_duration - dead_pre) # les 10 sec sont diminuees avec quantité d'artefact dans deadfile
            fr_post = len(post_spike_times) / (post_duration - dead_post) # les 10 sec sont diminuees avec quantité d'artefact dans deadfile
            # print('pre: nb spikes, pre_duration, dead_pre, pre_duration - dead_pre ',len(pre_spike_times), pre_duration, dead_pre, pre_duration - dead_pre)
            # print('post: nb spikes, post_duration, dead_pre, post_duration - dead_post',len(post_spike_times), post_duration, dead_pre, post_duration - dead_post)


            # delta_pre_post, delta_baseline_post / % de variation :
            if fr_pre == 0:
                delta_pre_post = np.nan
            else:
                delta_pre_post = 100 * (fr_post - fr_pre) / fr_pre
            if row['fr_baseline'] == 0:
                delta_baseline_post = np.nan
            else:  
                delta_baseline_post = 100 * (fr_post - row['fr_baseline']) / row['fr_baseline']
            
            # Log-ratio : variation symétrique (hausse/diminution comparables). epsilon pr eviter division par 0
            log_ratio = np.log((fr_post + epsilon) / (fr_pre + epsilon)) if fr_pre >= 0 else np.nan
            
            # Z-score pre : ( FRpost - mean_pre ) / std_pre
            # attention ds vanderPlas prennent comme baseline la baseline de tous les trials
            if len(pre_spike_times) > 0:
                bins = np.arange(t_pre_start, t_pre_end + bin_z, bin_z)
                counts, _ = np.histogram(pre_spike_times, bins=bins)
                fr_bins = counts / bin_z
                mu_pre, sigma_pre = fr_bins.mean(), fr_bins.std(ddof=1)
                z_score_pre = (fr_post - mu_pre) / sigma_pre if sigma_pre > 0 else np.nan
            else:
                z_score_pre = np.nan

            # Modulation Index : borné entre –1 et +1
            # MI = 0 ➝ pas de changement / MI → +1 ➝ fort gain (FRpost ≫ FRpre) / MI → –1 ➝ forte suppression (FRpost ≪ FRpre)
            modulation_index = (fr_post - fr_pre) / (fr_post + fr_pre) \
                               if (fr_post + fr_pre) > 0 else np.nan
            
            # Distance euclidienne entre la stim et la tt :
            xyz_tt = find_back_XYZ_from_macro(coord_MNI_pat, find_back_macrocontacts_from_tt(normalize_name(dict_clu2tt[clu][:-1]), coord_MNI_pat,verb=verb)[0], find_back_macrocontacts_from_tt(normalize_name(dict_clu2tt[clu][:-1]), coord_MNI_pat,verb=verb)[1], verb=verb)
            xyz_stim = find_back_XYZ_from_macro(coord_MNI_pat, stims_loca.loc[i, 'electrode'], stims_loca.loc[i, 'plots'], verb=verb)
            distance_tt_stim = euclidian_distance(xyz_tt, xyz_stim, verb=verb)
            
            # Stim en ZE, ZI, etc / Tetrode en ZE, ZI, etc :
            stimZE, stimZI, stimZP, stimZL, stimNI = is_in_ZEZIZPZLNI(patient, stims_loca.loc[i, 'electrode'], stims_loca.loc[i, 'plots'])
            
            # Stockage de ces variable dans data (summary_by_neuron)
            label_stim = labels_stims[i][:-8]
            row[f'{i}_{label_stim}_pre'] = fr_pre
            row[f'{i}_{label_stim}_post'] = fr_post
            row[f'{i}_{label_stim}_delta_pre_post'] = delta_pre_post
            row[f'{i}_{label_stim}_delta_baseline_post'] = delta_baseline_post
            row[f'{i}_{label_stim}_log_ratio'] = log_ratio
            row[f'{i}_{label_stim}_zscore_pre'] = z_score_pre
            row[f'{i}_{label_stim}_modulation_index'] = modulation_index
            row[f'{i}_{label_stim}_distance_tt_stim'] = distance_tt_stim
            row[f'{i}_{label_stim}_stim_en_ZE'],row[f'{i}_{label_stim}_stim_en_ZI'],row[f'{i}_{label_stim}_stim_en_ZP'],row[f'{i}_{label_stim}_stim_en_ZL'],row[f'{i}_{label_stim}_stim_en_NI'] = stimZE, stimZI, stimZP, stimZL, stimNI
            
            # Topographie : sameElec, sameLobe / Distance avec la stim / stim ou tt en ZE,ZI,ZP,ZL,NI 
            row[f'{i}_{label_stim}_sameElec'] = electrodes_equal(dict_clu2tt[clu][:-1],stims_loca.loc[i,'electrode']) # localité du neurone par rapport aux stimulations
            row[f'{i}_{label_stim}_sameLobe'] = (mpg.loc[mpg['tt'] == dict_clu2tt[clu], 'lobe'].values[0] == stim['lobe'].strip())
            
            # et pour chaque stim et chaque neurone, on veut aussi une ligne entiere dans le df general_data
            row_general = {'patient': patient, 'session':stimic_session, 'clu': clu, 'tetrode': dict_clu2tt[clu], 
                           'lobe_tt':mpg.loc[mpg['tt'] == dict_clu2tt[clu], 'lobe'].values[0],
                            'loca_tt':mpg.loc[mpg['tt'] == dict_clu2tt[clu], 'loca'].values[0],
                           'fr_global': row['fr_global'], 'fr_baseline': row['fr_baseline'],
                           'delta_pre_post':delta_pre_post, 'delta_baseline_post':delta_baseline_post, 
                           'fr_pre': fr_pre, 'fr_post': fr_post,  'log_ratio': log_ratio,
                           'zscore_pre': z_score_pre, 'modulation_index': modulation_index, 
                        #    'responsive_clu':responsive_clu, 'inhib_only': inhib_only, 'excit_only': excit_only, 
                        #    'inhib_then_excit': inhib_then_excit, 'excit_then_inhib': excit_then_inhib, 
                        #    'inhib_general': inhib_general, 'excit_general':excit_general, 
                           'stim_Lobe': stim['lobe'].strip(), 'sameLobe': row[f'{i}_{label_stim}_sameLobe'], 'sameElec': row[f'{i}_{label_stim}_sameElec'],
                            'freq_stim': int(stim['frequence'].strip()[:-3]), 'intensity_stim': float(stim['intensite'].strip()[:-3]), 
                            'stim_in_ZE': stimZE, 'stim_in_ZI': stimZI, 'stim_in_ZP': stimZP, 'stim_in_ZL': stimZL, 'stim_in_NI': stimNI,
                            'tt_in_ZE': ttZE, 'tt_in_ZI': ttZI, 'tt_in_ZP': ttZP, 'tt_in_ZL': ttZL, 'tt_in_NI': ttNI,
                            'distance_tt_stim': distance_tt_stim}
            
            # quality metrics :
            for metric in qm.columns.tolist():
                row_general[metric] = qm.loc[ind_clu, metric]
            
            # Time binned to get AP per time unit
            for ind_bin, bin_r_i in enumerate(bin_resp):
                bins_edges_i = np.arange(0, post_duration + bin_r_i, bin_r_i)
                post_counts_i, _ = np.histogram(post_spike_times - t_post_start, bins=bins_edges_i)
                pre_counts_i, _ = np.histogram(pre_spike_times - t_pre_start, bins=bins_edges_i)
            
            # calcul seuil de depassement reponse significative : mean_pre +/- 2*std_pre
                mean_pre = pre_counts_i.mean()
                std_pre = pre_counts_i.std(ddof=1)
                upper_thr = mean_pre + 2*std_pre
                lower_thr = mean_pre - 2*std_pre
                above = np.where(post_counts_i > upper_thr)[0] # depassement du seuil superieur
                below = np.where(post_counts_i < lower_thr)[0] # depassement du seuil inferieur
                # inhib_only = np.append(inhib_only, int(len(below) > 0 and len(above) == 0))
                # excit_only = np.append(excit_only, int(len(above) > 0 and len(below) == 0))
                # inhib_then_excit = np.append(inhib_then_excit, int(len(below) > 0 and len(above) > 0 and below[0] < above[0]))
                # excit_then_inhib = np.append(excit_then_inhib, int(len(below) > 0 and len(above) > 0 and above[0] < below[0]))
                # inhib_general = np.append(inhib_general, inhib_only[ind_bin] + inhib_then_excit[ind_bin])
                # excit_general = np.append(excit_general, excit_only[ind_bin] + excit_then_inhib[ind_bin])
                
                # Ajout a data puis a data_general : 
                row[f'{i}_{label_stim}_inhib_only_{bin_r_i}s_bins'] = int(len(below) > 0 and len(above) == 0) #inhib_only[ind_bin]
                row[f'{i}_{label_stim}_excit_only_{bin_r_i}s_bins'] = int(len(above) > 0 and len(below) == 0) #excit_only[ind_bin]
                row[f'{i}_{label_stim}_inhib_then_excit_{bin_r_i}s_bins'] = int(len(below) > 0 and len(above) > 0 and below[0] < above[0]) #inhib_then_excit[ind_bin]
                row[f'{i}_{label_stim}_excit_then_inhib_{bin_r_i}s_bins'] = int(len(below) > 0 and len(above) > 0 and above[0] < below[0]) #excit_then_inhib[ind_bin]
                row[f'{i}_{label_stim}_inhib_general_{bin_r_i}s_bins'] = row[f'{i}_{label_stim}_inhib_only_{bin_r_i}s_bins'] + row[f'{i}_{label_stim}_inhib_then_excit_{bin_r_i}s_bins'] + row[f'{i}_{label_stim}_excit_then_inhib_{bin_r_i}s_bins'] #inhib_general[ind_bin]
                row[f'{i}_{label_stim}_excit_general_{bin_r_i}s_bins'] = row[f'{i}_{label_stim}_excit_only_{bin_r_i}s_bins'] + row[f'{i}_{label_stim}_inhib_then_excit_{bin_r_i}s_bins'] + row[f'{i}_{label_stim}_excit_then_inhib_{bin_r_i}s_bins'] #excit_general[ind_bin]
                if row[f'{i}_{label_stim}_inhib_general_{bin_r_i}s_bins']>0 or row[f'{i}_{label_stim}_excit_general_{bin_r_i}s_bins']>0: # si une reponse a lieu, le clu est classé responsive, sinon il reste non-responsive par defaut
                    responsive_clu[ind_bin] = True
                
                row_general[f'inhib_only_{bin_r_i}s_bins'] = row[f'{i}_{label_stim}_inhib_only_{bin_r_i}s_bins']
                row_general[f'excit_only_{bin_r_i}s_bins'] = row[f'{i}_{label_stim}_excit_only_{bin_r_i}s_bins']
                row_general[f'inhib_then_excit_{bin_r_i}s_bins'] = row[f'{i}_{label_stim}_inhib_then_excit_{bin_r_i}s_bins']
                row_general[f'excit_then_inhib_{bin_r_i}s_bins'] = row[f'{i}_{label_stim}_excit_then_inhib_{bin_r_i}s_bins']
                row_general[f'inhib_general_{bin_r_i}s_bins'] = row[f'{i}_{label_stim}_inhib_general_{bin_r_i}s_bins']
                row_general[f'excit_general_{bin_r_i}s_bins'] = row[f'{i}_{label_stim}_excit_general_{bin_r_i}s_bins']

            general_data.append(row_general) # une ligne par neurone et par stim
        for ind_bin in range(len(bin_resp)):
            row[f'responsive_clu_{bin_resp[ind_bin]}s_bins'] = responsive_clu[ind_bin]
            # row_general.loc[-stims_loca.shape[0]:-1, f'responsive_clu_{bin_resp[ind_bin]}s_bins'] = [responsive_clu[ind_bin] for _ in range(stims_loca.shape[0])]
        data.append(row) # une ligne par neurone

    return (pd.DataFrame(data), pd.DataFrame(general_data))


def create_or_update_session_summary(patient, session, start_baseline=0, end_baseline=300, root='D:/',verb=False, bin_z=0.05, bin_resp=[0.05, 0.075, 0.1]):
    ''' Cree le tableau récapitulatif pour une session: summary_by_neuron et general_summary_by_neuron_and_stim, a partir du patient et session
    Tourne pendant environ 30 sec/1 min par session. '''
    path_folder = root + 'Spike-sorting/Data_folders/'+patient+'/'+patient+'_stim'+session+'/'
    sr = get_SR(patient)
    spikes = get_nwb(patient, session)
    stims_loca = get_stims(patient, session, root)
    mapping_anat = pd.read_csv(root + 'Spike-sorting/Data_folders/'+patient+'/mapping_anat_'+patient+'.txt', sep=',', engine='python')
    dict_elec2deadfile = get_dict_deadfiles(mapping_anat, patient, session, path_folder, sr)
    dict_clu2tt = get_dict_tetrodeName_from_tetrodeIndex(spikes, mapping_anat)

    # summary tables for the session:
    summary_df, general_df = compute_neuronal_summary(spikes, stims_loca, dict_clu2tt, dict_elec2deadfile, mapping_anat, patient, session, root, mapping_anat, start_baseline=0, end_baseline=300, verb=verb, bin_z=bin_z, bin_resp=bin_resp)
    summary_df.to_csv(path_folder+patient+'_stim'+session+"_summary_by_neuron.csv", index=False)
    summary_df.to_excel(path_folder+patient+'_stim'+session+"_summary_by_neuron.xlsx", index=False)
    general_df.to_csv(path_folder+patient+'_stim'+session+"_general_summary_by_neuron_and_stim.csv", index=False)
    general_df.to_excel(path_folder+patient+'_stim'+session+"_general_summary_by_neuron_and_stim.xlsx", index=False)
    print('Summary tables saved')

    return summary_df, general_df


def update_all_existing_session_summaries(root='D:/', verb=False, bin_z=0.05, bin_resp=[0.05, 0.075, 0.1]):
    ''' Met à jour tous les session_summaries, au cas où ajout de nouvelles metriques, ou modifs de neurones'''
    base = Path(root, "Spike-sorting/Data_folders")
    for path in base.rglob("*.nwb"): # pour chaque session traitée, donc pour laquelle on a un .nwb :
        relative = path.relative_to(base) # On garde les fichiers qui sont exactement à 2 niveaux sous le dossier racine
        if len(relative.parts) == 3:  # Patient / Session / fichier.nwb
            patient = relative.parts[0]
            session = relative.parts[1]
            print('Update started for ', session)
            _,_  = create_or_update_session_summary(patient, session[-1], start_baseline=0, end_baseline=300, root=root, verb=verb, bin_z=bin_z, bin_resp=bin_resp)
            print('Update done for ', session)


############### Big dataframe functions ###############

def update_general_summary_on_all_sessions(root='D:/'):
    '''Tourne sur tous les session_summaries et recrée et renvoie big_df
    Tourne eviron 40 sec pour une vingtaine de sessions.'''
    base = Path(root, "Spike-sorting/Data_folders")
    
    nwb_files = []
    all_gen_summaries, all_summaries_by_nrn, all_stims = [], [], [] # Liste de DF, pour stocker tous les csv

    for path in base.rglob("*.nwb"): # pour chaque session traitée, donc pour laquelle on a un .nwb :
        relative = path.relative_to(base) # On garde les fichiers qui sont exactement à 2 niveaux sous le dossier racine
        if len(relative.parts) == 3:  # Patient / Session / fichier.nwb
            nwb_files.append(path)
            # print(path) # Affichage des nwb d'intéret

            patient = relative.parts[0]
            session = relative.parts[1]
            gen_summary_path = base / patient / session / f"{session}_general_summary_by_neuron_and_stim.csv"
            summary_by_neuron_path = base / patient / session / f"{session}_summary_by_neuron.csv"

            if gen_summary_path.exists():
                # on récupère general_summary (par neurone et par stimulation)
                # print(f"existe: {gen_summary_path}")
                general_summary = pd.read_csv(gen_summary_path)
                # On crée une colonne 'global_clu' unique pour s'y retrouver dans l'indexation de l'ensemble des neurones
                general_summary["patient"] = patient
                general_summary["session"] = session
                general_summary["global_clu"] = (general_summary["clu"].astype(str).apply(lambda x: f"{session}_{x}"))
                
                all_gen_summaries.append(general_summary)

                # on récupère summary_by_neuron (par neurone)
                summary_by_nrn = pd.read_csv(summary_by_neuron_path)
                all_summaries_by_nrn.append(summary_by_nrn)

                # on récupère les stims
                if os.path.exists(base / patient / session / f"{session}_stim_events_TRC_re-shifted.txt"):
                    stims_loca = pd.read_csv(base / patient / session / f"{session}_stim_events_TRC_re-shifted_loca.txt", header=None)
                else:
                    stims_loca = pd.read_csv(base / patient / session / f"{session}_stim_events_TRC_shifted_loca.txt", header=None)
                all_stims.append(stims_loca)

    # Empile tous les DataFrames de type summary
    big_df = pd.concat(all_gen_summaries, ignore_index=True)
    big_df_by_nrn = pd.concat(all_summaries_by_nrn, ignore_index=True)
    big_df_stims = pd.concat(all_stims, ignore_index=True)
    big_df_stims.columns = ["label", "t_start", "duration", "lobe"]

    # Ajout de qqs colonnes utiles: stim_Lobe_noLat, lobe_tt_noLat, ind_stim (intra-session)
    big_df['stim_Lobe_noLat'] = big_df['stim_Lobe'].str[2:]
    big_df['lobe_tt_noLat'] = big_df['lobe_tt'].str[2:]
    big_df['loca_tt_noLat'] = big_df['loca_tt'].str[2:]
    list_stim_indices = np.zeros(big_df.shape[0])
    ind_trial = 0
    for p in big_df['patient'].unique().tolist():
        for sess in big_df[big_df['patient']==p]['session'].unique().tolist():
            df_session = big_df[big_df['patient']==p][big_df[big_df['patient']==p]['session']==sess]
            nb_stims = int(big_df[big_df['session']==sess].shape[0]/len(big_df[big_df['session']==sess]['clu'].unique())) # normalement, tombe pile sur un entier
            for _ in df_session['clu'].unique():
                for ind_stim in range(nb_stims):
                    list_stim_indices[ind_trial] = int(ind_stim)
                    ind_trial += 1
    big_df['ind_stim'] = list_stim_indices

    # Export des dataframes generaux
    big_df.to_excel("C:/Users/darves-bornoz/Documents/article_neuronal_stimic/stats/general_summary_all_sessions.xlsx", index=False)
    big_df.to_csv("C:/Users/darves-bornoz/Documents/article_neuronal_stimic/stats/general_summary_all_sessions.csv", index=False)
    
    big_df_by_nrn.to_excel("C:/Users/darves-bornoz/Documents/article_neuronal_stimic/stats/summary_by_nrn_all_sessions.xlsx", index=False)
    big_df_by_nrn.to_csv("C:/Users/darves-bornoz/Documents/article_neuronal_stimic/stats/summary_by_nrn_all_sessions.csv", index=False)
    
    # big_df_stims.to_excel("C:/Users/darves-bornoz/Documents/article_neuronal_stimic/stats/stims_all_sessions.xlsx", index=False)
    # big_df_stims.to_csv("C:/Users/darves-bornoz/Documents/article_neuronal_stimic/stats/stims_all_sessions.csv", index=False)
    
    return big_df


##################################################
# Raster display functions
##################################################

def plot_artefact_patches(ax, deadfile, stim_start, line_index, duration_out_array, side='pre', type_raster='one_neuron', stim_duration=None, window=10, color='black', alpha=0.3, display=True):
    """
    Dessine un patch d'artefact sur l'axe donné.
    Args:
        ax (matplotlib.axes): Axe matplotlib sur lequel dessiner.
        deadfile (pd.DataFrame): DataFrame contenant les artefacts, avec début (col 0) et fin (col 1).
        stim_start (float): Début de la stimulation.
        line_index (int): Position verticale (ligne raster) du neurone ou de la stimulation.
        duration_out_array (np.array): Tableau accumulant la durée supprimée pour le neurone.
        side (str): 'pre' ou 'post'.
        type_raster (str): Si 'one_neuron', centre la période autour de 0 (soustrait stim_start aux temps). Si 'one_stim' on ne change rien, car les temps restent bruts, ne sont pas relatifs.
        stim_duration (float): Durée de la stimulation, nécessaire pour les cas post-stim alignés.
        window (float): Durée de la fenêtre avant ou après la stimulation (en secondes).
        color (str): Couleur du patch.
        alpha (float): Transparence du patch.
    """
    if side == 'pre':
        t_start, t_end = stim_start - window, stim_start
    elif side == 'post':
        t_start, t_end = stim_start + stim_duration, stim_start + stim_duration + window
    else:
        raise ValueError("side must be 'pre' or 'post'")

    artefacts_filtered = deadfile[(deadfile[1] >= t_start) & (deadfile[0] <= t_end)]

    for _, row in artefacts_filtered.iterrows():
        artefact_start, artefact_end = row[0], row[1]
        artefact_start_clipped = max(artefact_start, t_start)
        artefact_end_clipped = min(artefact_end, t_end)
        width = artefact_end_clipped - artefact_start_clipped

        if width > 0:
            duration_out_array[line_index] += width 
            if display:
                if side == 'post' and stim_duration is not None:
                    display_start = artefact_start_clipped - stim_duration  
                else:
                    display_start = artefact_start_clipped # a gauche/pre-stim: on prend la valeur brute de debut artefact ou debut window
                if type_raster == 'one_neuron': # on décale parce que qd on plot pr ttes les stims, l'absisse est centrée sur le début de la stimulation (-10 à +10 s)
                    display_start -= stim_start
                rect = plt.Rectangle((display_start, line_index - 0.23), width, 0.5,
                                    color=color, alpha=alpha, edgecolor='none')
                ax.add_patch(rect)


def rasters_OneNeuron_allStims(patient, session, spikes, stims, dict_clu2tt, dict_elec2deadfile, path_folder, display_patches = True, plafond_inhib_100 = True):
    '''
    Returns and saves one raster per neuron, for all the EBS of the session. Runs on all neurons of the recording.
    Args:
        display_patches = True # si on veut afficher les patches gris des dead periods
        plafond_inhib_100 = True # si on veut que la variation de fréquence de décharge soit plafonnée à 100% qd inhibition (sinon, on change le calcul de % variation)
    '''
    print('in raster')
    if not os.path.exists(Path(path_folder+"\Rasters")): # on cree le dossier "Rasters" pour stock images s'il n'existe pas encore
        os.makedirs(Path(path_folder+"\Rasters"))

    for ind_neuron in range(len(spikes)):
        print('in neuron ',ind_neuron)
        # ind_neuron=0
        deadfile_elec = dict_elec2deadfile[dict_clu2tt[ind_neuron][:-1]]  # deadfile de l'electrode correspondante
        spike_times_before = [] # on veut une liste par epoch
        spike_times_after = []
        for ind_stim in range(stims.shape[0]):
            epoch_stim_i = nap.IntervalSet(start=[stims['t'][ind_stim]-10, stims['t + durée'][ind_stim]], end=[stims['t'][ind_stim], stims['t + durée'][ind_stim]+10], time_units="s") # 
            spikes_restricted_i = spikes.restrict(epoch_stim_i)
            spike_times_i = list(spikes_restricted_i[ind_neuron].index) # tous les spikes du neurone avant + apres la stim
            spike_times_before.append([t-stims['t'][ind_stim] for t in spike_times_i if t<=stims['t'][ind_stim]] ) # avant la stim
            spike_times_after.append([t-stims['durée'][ind_stim]-stims['t'][ind_stim] for t in spike_times_i if t>stims['t'][ind_stim]]) # apres la stim

        # affichage
        fig = plt.figure(figsize=(15, max(4, stims.shape[0]/4))) # grille de sous-graphiques 1x3
        gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1])
        plt.style.use('ggplot')
        fig.suptitle(patient+", stimic"+session+": all stimulations for neuron "+str(ind_neuron)+', recorded by tetrode '+ dict_clu2tt[ind_neuron], fontsize=14, y=0.95) 
        ax1 = plt.subplot(gs[0,0])
        ax2 = plt.subplot(gs[0,1])
        ax4 = plt.subplot(gs[1,0])

        # 1. raster plot for the specific neuron
        ax1.eventplot(spike_times_before, colors="green", linelengths=0.5)
        ax1.axvline(0, color="blue", linestyle="-", linewidth=1, label="Stimulation")
        ax1.eventplot(spike_times_after, colors="red", linelengths=0.5)
        ax1.set_xlim([-10, 10])
        x = np.arange(-10,11,1)
        ax1.set_xticks(x, np.repeat(' ', len(x)))# ax1.set_xticks([stim_start+x_i for x_i in x], x)
        ax1.set_ylim(-1, stims.shape[0])
        stim_names = [stims['electrode'][ind_stim]+stims['plots'][ind_stim]+', '+stims['frequence'][ind_stim]+', '+stims['intensite'][ind_stim] for ind_stim in range(stims.shape[0])]
        ax1.set_yticks(range(0,stims.shape[0]), stim_names) # selon nombre total de nrn pour cette session
        ax1.set_ylabel("Stimulations")

        # 1.bis. Ajout des periodes qui sont dans le deadfile :
        durationOut_preStim, durationOut_postStim = np.zeros(stims.shape[0]), np.zeros(stims.shape[0]) # duree cumulee par neurone qui a ete retiree
        for ind_stim in range(stims.shape[0]):
            print('in stim',ind_stim)
            stim_start = stims['t'][ind_stim]
            stim_end = stims['t + durée'][ind_stim]
            # on met d'abord a jour la somme des periodes enlevées sur ce trial
            artefacts_filtered_pre = deadfile_elec[(deadfile_elec[1] >= stim_start-10) & (deadfile_elec[0] <= stim_start)]
            artefacts_filtered_post = deadfile_elec[(deadfile_elec[1] >= stim_end) & (deadfile_elec[0] <= stim_end+10)]
            durationOut_preStim[ind_stim] = np.sum(np.minimum(artefacts_filtered_pre[1], stim_start) - np.maximum(artefacts_filtered_pre[0], stim_start-10)) 
            durationOut_postStim[ind_stim] = np.sum(np.minimum(artefacts_filtered_post[1], stim_end+10) - np.maximum(artefacts_filtered_post[0], stim_end))  
            # puis on affiche les dead periods
            plot_artefact_patches(ax1, deadfile_elec, stim_start, ind_stim, durationOut_preStim, side='pre', type_raster='one_neuron', display=display_patches)
            plot_artefact_patches(ax1, deadfile_elec, stim_start, ind_stim, durationOut_postStim, side='post', type_raster='one_neuron', stim_duration=stims['durée'][ind_stim], display=display_patches)

        # 2. variation de fréquence de décharge, > ou < 0
        if plafond_inhib_100:
            firing_rate_var = [100 * ((len(spike_times_after[i]) / (10 - durationOut_postStim[i])) -
                (len(spike_times_before[i]) / (10 - durationOut_preStim[i]))) /
            (len(spike_times_before[i]) / (10 - durationOut_preStim[i]))
            if len(spike_times_before[i]) > 0 else np.nan for i in range(len(spike_times_before))]
        else:
            firing_rate_var = [np.nan for i in range(stims.shape[0])]
            for i in range(len(spike_times_before)) :
                if len(spike_times_before[i]) > 0:
                    if (len(spike_times_after[i]) / (10 - durationOut_postStim[i])) < (len(spike_times_before[i]) / (10 - durationOut_preStim[i])) : 
                        if len(spike_times_after[i]) > 0:
                            firing_rate_var[i] =  - 100 * ((len(spike_times_before[i]) / (10 - durationOut_preStim[i])) - (len(spike_times_after[i]) / (10 - durationOut_postStim[i]))) / (len(spike_times_after[i]) / (10 - durationOut_postStim[i]))
                        else:
                            firing_rate_var[i] = np.nan
                    else:
                        firing_rate_var[i] = 100 * ((len(spike_times_after[i]) / (10 - durationOut_postStim[i])) - (len(spike_times_before[i]) / (10 - durationOut_preStim[i]))) / (len(spike_times_before[i]) / (10 - durationOut_preStim[i]))
                else:
                    firing_rate_var[i] = np.nan
        ax2.barh([str(i) for i in range(stims.shape[0])], firing_rate_var, color='orange')
        ax2.axvline(0, linewidth=1, color="black")
        ax2.set_ylim(-1, stims.shape[0])
        ax2.set_yticks(list(range(stims.shape[0])), [' ' for s in stim_names])
        ax2.set_xlabel("Variation of firing rate:\nafter-before EBS (in %)", fontsize=12)

        # 3. nombre de spikes par unité de temps
        nb_time_bins=100
        ax4.hist(np.sort([item for sublist in spike_times_before for item in sublist]), nb_time_bins, color='green')
        ax4.axvline(0, color="blue", linestyle="-", linewidth=1, label="Stimulation")
        ax4.hist(np.sort([item for sublist in spike_times_after for item in sublist]), nb_time_bins, color='red')
        ax4.set_xlim([-10, 10])
        ax4.set_xticks(x)
        ax4.set_xlabel("Time before the start (s)                              Time after the end (s)", fontsize=12)
        ax4.set_ylabel("Number of\nspikes per\ntime unit", fontsize=12)

        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        plt.savefig(path_folder+"/Rasters/Raster - all stimulations for neuron "+str(ind_neuron)+" from "+dict_clu2tt[ind_neuron]+".png", dpi=300)
        plt.show()


def rasters_OneStim_allNeurons(patient, session, spikes, stims, dict_clu2tt, dict_elec2deadfile, path_folder, display_patches = True, plafond_inhib_100 = True):

    if not os.path.exists(Path(path_folder+"/Rasters")):
        os.makedirs(Path(path_folder+"/Rasters"))

    # D'abord determiner si certaines stims avec mm caracs/mm label exactement => car besoin 
    # si label identique a autres: ajout suffixe _1, _2,... au fichier exporté / ‘_’+i for i in range(nb de fois que repet)
    list_suffix_stim_rep = ['' for _ in range(stims.shape[0])] # par défaut, suffixe vide
    for s in np.unique(stims['paramètres'].tolist()):
        if stims['paramètres'].tolist().count(s) > 1 : # si au moins 2 stims avec meme label
            list_ind_stim_repeated = [i for i, x in enumerate(stims['paramètres'].tolist()) if x == s]  # liste des indices de stims avec ce label
            for ind_stim_rep in list_ind_stim_repeated:
                list_suffix_stim_rep[ind_stim_rep] = '_' + str(ind_stim_rep + 1)

    for ind_stim in range(stims.shape[0]):
        stim_start, stim_end = stims['t'][ind_stim], stims['t + durée'][ind_stim]
        epoch_test = nap.IntervalSet(start=[stim_start-10, stim_end], end=[stim_start, stim_end+10], time_units="s") # df avec debut/fin de chaque epoch

        spikes_restricted = spikes.restrict(epoch_test) # activité totale sur ces epochs seulement
        spike_times = [spikes_restricted[nrn].index for nrn in dict_clu2tt.keys()] # liste de listes : temps de décharge pour chaque neurone, dans un ordre tel que fourni ds mapping 
        spike_times_before, spike_times_after = [], []
        for spike_times_i in spike_times:
            spike_times_before.append([t for t in spike_times_i if t<=stim_start])
            spike_times_after.append([t-stims['durée'][ind_stim] for t in spike_times_i if t>stim_start])

        # affichage general
        fig = plt.figure(figsize=(15, max(4, int(len(spikes)/4)))) # grille de sous-graphiques 1x3
        gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1])
        plt.style.use('ggplot')
        stim_title = stims['electrode'][ind_stim]+' '+stims['plots'][ind_stim]+', at '+stims['frequence'][ind_stim]+' and '+stims['intensite'][ind_stim]
        fig.suptitle(patient+", stimic"+session+": all neurons when stimulating in "+stim_title, fontsize=14, y=0.95)
        ax1 = plt.subplot(gs[0,0])
        ax2 = plt.subplot(gs[0,1])
        ax4 = plt.subplot(gs[1,0])

        # 1. raster plot for the specific stimulation
        ax1.eventplot(spike_times_before, colors="green", linelengths=0.5)
        ax1.axvline(stim_start, color="blue", linestyle="-", linewidth=1, label="Stimulation")
        ax1.eventplot(spike_times_after, colors="red", linelengths=0.5)
        ax1.set_xlim([stim_start-10, stim_start+10])
        x = np.arange(-10,11,1)
        ax1.set_xticks([stim_start+x_i for x_i in x], np.repeat(' ', len(x)))
        ax1.set_ylim(-2, len(spikes.index)+1)
        ax1.set_yticks(range(0,len(spikes.index)), dict_clu2tt.values()) # selon nombre total de nrn pour cette session
        ax1.set_ylabel("Neuron per tetrode")

        # 1.bis. Ajout des periodes qui sont dans le deadfile :
        durationOut_preStim, durationOut_postStim = np.zeros(len(spikes.index)), np.zeros(len(spikes.index)) # duree cumulee par neurone qui a ete retiree
        for ind_ligne_raster, tt in enumerate(dict_clu2tt.values()):
            deadfile_elec = dict_elec2deadfile[tt[:-1]]
            # on met d'abord a jour la somme des periodes enlevées sur ce trial
            artefacts_filtered_pre = deadfile_elec[(deadfile_elec[1] >= stim_start-10) & (deadfile_elec[0] <= stim_start)]
            artefacts_filtered_post = deadfile_elec[(deadfile_elec[1] >= stim_end) & (deadfile_elec[0] <= stim_end+10)]
            durationOut_preStim[ind_ligne_raster] = np.sum(np.minimum(artefacts_filtered_pre[1], stim_start) - np.maximum(artefacts_filtered_pre[0], stim_start-10)) 
            durationOut_postStim[ind_ligne_raster] = np.sum(np.minimum(artefacts_filtered_post[1], stim_end+10) - np.maximum(artefacts_filtered_post[0], stim_end))  
            # puis on affiche les dead periods
            plot_artefact_patches(ax1, deadfile_elec, stim_start, ind_ligne_raster, durationOut_preStim, side='pre', type_raster='one_stim', display=display_patches)
            plot_artefact_patches(ax1, deadfile_elec, stim_start, ind_ligne_raster, durationOut_postStim, side='post', type_raster='one_stim', stim_duration=stims['durée'][ind_stim], display=display_patches)

        # 2. variation de fréquence de décharge, > ou < 0 (corrigee par les portions de temps dans deadfile)
        if plafond_inhib_100:
            firing_rate_var = [100*(len(spike_times_after[i])/(10-durationOut_postStim[i])-len(spike_times_before[i])/(10-durationOut_preStim[i]))/(len(spike_times_before[i])/(10-durationOut_preStim[i])) for i in range(len(spikes.index))] 
        else:
            firing_rate_var = [np.nan for i in range(stims.shape[0])]
            for i in range(len(spikes.index)) :
                if len(spike_times_before[i]) > 0:
                    if (len(spike_times_after[i]) / (10 - durationOut_postStim[i])) < (len(spike_times_before[i]) / (10 - durationOut_preStim[i])) : 
                        if len(spike_times_after[i]) > 0:
                            firing_rate_var[i] =  - 100 * ((len(spike_times_before[i]) / (10 - durationOut_preStim[i])) - (len(spike_times_after[i]) / (10 - durationOut_postStim[i]))) / (len(spike_times_after[i]) / (10 - durationOut_postStim[i]))
                        else:
                            firing_rate_var[i] = np.nan
                    else:
                        firing_rate_var[i] = 100 * ((len(spike_times_after[i]) / (10 - durationOut_postStim[i])) - (len(spike_times_before[i]) / (10 - durationOut_preStim[i]))) / (len(spike_times_before[i]) / (10 - durationOut_preStim[i]))
                else:
                    firing_rate_var[i] = np.nan    
        ax2.barh([str(i) for i in range(len(spikes.index))], firing_rate_var, color='orange') 
        ax2.axvline(0, linewidth=1, color="black")
        ax2.set_ylim(-2, len(spikes.index)+1)
        ax2.set_yticks(range(0,len(spikes.index)), np.repeat('', len(spikes.index)))
        ax2.set_xlabel("Variation of firing rate:\nafter-before EBS (in %)", fontsize=10)

        # 3. nombre de spikes par unité de temps
        nb_time_bins=100
        ax4.hist(np.sort([item for sublist in spike_times_before for item in sublist]), nb_time_bins, color='green')
        ax4.axvline(stim_start, color="blue", linestyle="-", linewidth=1, label="Stimulation")
        ax4.hist(np.sort([item for sublist in spike_times_after for item in sublist]), nb_time_bins, color='red')
        ax4.set_xlim([stim_start-10, stim_start+10])
        ax4.set_xticks([stim_start+x_i for x_i in x], x)
        ax4.set_xlabel("Time before the start (s)                                               Time after the end (s)", fontsize=10)
        ax4.set_ylabel("Number of spikes\nper time unit", fontsize=10)

        plt.subplots_adjust(hspace=0.05, wspace=0.15)
        patch_title = ['' if display_patches else '_noDeadPeriods'][0]
        title_fig = path_folder+"/Rasters/Raster - all neurons for stim "+stims['electrode'][ind_stim]+' '+stims['plots'][ind_stim]+', at '+stims['frequence'][ind_stim]+', '+stims['intensite'][ind_stim]+patch_title+".png"
        plt.savefig(title_fig, dpi=300)

        plt.show()


def create_or_update_rasters(patient, session, overwrite_rasters=True, root='D:/'):
    '''
    Cree tous les rasters pour une session
    overwrite_rasters = False si les rasters ont déjà été faits
    '''
    path_folder = root + 'Spike-sorting/Data_folders/'+patient+'/'+patient+'_stim'+session+'/'

    sr = get_SR(patient)
    spikes = get_nwb(patient, session)

    stims_loca = get_stims(patient, session, root)
    mapping_anat = pd.read_csv(root + 'Spike-sorting/Data_folders/'+patient+'/mapping_anat_'+patient+'.txt', sep=',', engine='python')

    dict_elec2deadfile = get_dict_deadfiles(mapping_anat, patient, session, path_folder, sr)
    dict_clu2tt = get_dict_tetrodeName_from_tetrodeIndex(spikes, mapping_anat)

    # rasters: 
    if os.path.exists(path_folder+"/Rasters"): 
        if overwrite_rasters :
            if os.listdir(Path(path_folder+"/Rasters")) != []: # si deja fichiers 
                for f in os.listdir(Path(path_folder+"/Rasters")): # alors on efface tous les fichiers du dossier Rasters
                    Path(path_folder+"/Rasters"+'/'+f).unlink()
                rasters_OneNeuron_allStims(patient, session, spikes, stims_loca, dict_clu2tt, dict_elec2deadfile, path_folder, display_patches = True, plafond_inhib_100 = True)
                rasters_OneStim_allNeurons(patient, session, spikes, stims_loca, dict_clu2tt, dict_elec2deadfile, path_folder, display_patches = True, plafond_inhib_100 = True)
    else:
        rasters_OneNeuron_allStims(patient, session, spikes, stims_loca, dict_clu2tt, dict_elec2deadfile, path_folder, display_patches = True, plafond_inhib_100 = True)
        rasters_OneStim_allNeurons(patient, session, spikes, stims_loca, dict_clu2tt, dict_elec2deadfile, path_folder, display_patches = True, plafond_inhib_100 = True)

##################################################
# Signal display functions
##################################################

def load_eeg(eegfname, nCh, order = 'F', dtype = np.int16):
    """
    return lfps. shape = (nb channels, time*sr in sec)
    """
    size = get_total_duration(eegfname, nCh, np.int16)
    lfps = np.memmap(eegfname, mode='r', dtype=dtype, order=order, shape=(nCh, int(size/nCh)))
    
    return lfps 

def get_tickLocsNLabels_centered(tickMin, tickMax, nTicks, dtype = float, conversion = 1): # called in plot_chTraces()
    """
    return tickLocs, tickLabels 
    
    convert will be multiplied to tickLabels 
    to convert sample_rate to secs: 1/sr
    to convert sample_rate to ms: 1000/sr
    & vice versa
    
    ex:
    get_tickLocsNLabels(0, 4*1250, 5, convert = 1000/sr)
    return
    (array([   0., 1250., 2500., 3750., 5000.]),
     array([-2000., -1000.,    -0.,  1000.,  2000.]))
    """
    tickLocs = np.array(np.linspace(tickMin, tickMax, nTicks))
    m = np.mean([tickMin, tickMax])
    tickLabels = np.array(np.array([-(tickMax-m)+(x/(nTicks-1))*2*(tickMax-m) for x in range(nTicks)])*conversion, dtype = dtype)
    
    return tickLocs, tickLabels

# Plot    
def plot_chTraces(chTraces, sr, chInds = None, t_sample = None, chCols = None, win_s = 1, hspace = -200, title = False, legend = True, chLegend = None, xlabel = True, fontsize_legend = 20, lw=1, nTicks=5, roundTickLabels=2, locLegend='upper right', titleLegend=None, title_fontsizeLegend=20, lwLegend=1, alpha=1, xTickFS=20,labFS=30, yTickFS=12):
    """
    chInds is a np.array
    Advised size: plt.figure(figsize=(30,nCh*1.5))
    hspace : in µV, space between channels
    legendLabels 
    """
    try:
        chTraces.shape[1]
        multipleCh = 1 
    except:
        multipleCh = 0
    
    if chInds is None:
        if multipleCh:
            inds = np.arange(chTraces.shape[0])
        else:
            inds = np.array([0])
    else:
        inds = chInds
        
    nCh = inds.shape[0]
    
    if chCols is None:
        colors = sb.color_palette('tab10', nCh) ######### pb avec import seaborn => trouver une autre palette
    else:
        colors = chCols
        
    if t_sample is None:
        if multipleCh:
            t = np.random.choice(chTraces.shape[1])
        else:
            t = np.random.choice(chTraces.shape[0])
    else:
        t = t_sample
        
    if chLegend is None:
        chLegend = np.copy(inds)
    # 
    start = int(t-win_s*sr/2)
    end = int(t+win_s*sr/2)
    
    xLoc, xLabels=get_tickLocsNLabels_centered(0, win_s*sr, nTicks, conversion=1/sr)
    xLabels=np.round(xLabels ,roundTickLabels)
    
    for chii, chi in enumerate(inds):
        if multipleCh:
            plt.plot(chTraces[chi, start:end]+hspace*chii, color = colors[chii], label = chLegend[chii], lw=lw, alpha=alpha)
        else:
            plt.plot(chTraces[start:end]+hspace*chii, color = colors[chii], label = chLegend[chii], lw=lw, alpha=alpha)
    
#     plt.xlim(0, win_s*sr)
#     plt.xticks(xLoc, xLabels, fontsize = xTickFS)
    plt.yticks(fontsize=yTickFS)
#     plt.spines['top'].set_visible(False)
#     plt.spines['right'].set_visible(False)
#     plt.spines['left'].set_visible(False)
#     plt.spines['bottom'].set_visible(False)
    
    if legend:
        leg = plt.legend(fontsize=fontsize_legend, loc = locLegend, title=titleLegend, title_fontsize=title_fontsizeLegend)
        for line in leg.get_lines():
            line.set_linewidth(lwLegend)
    if title:
        plt.title('t = '+str(t)+' sample points <=> '+str(t*1000/sr)+' ms')
    if xlabel:
        plt.xlabel('Time (secs)', fontsize = labFS)