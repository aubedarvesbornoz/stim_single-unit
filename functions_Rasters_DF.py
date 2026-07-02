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
    print(path_folder, '.nwb existe deja ?', os.path.exists(nwbfile_path))
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
        try:
            dict_elec2deadfile[elec] = merge_overlapping_events(pd.read_csv(path_folder+'derivatives/'+patient+'_stim'+session+'_deadfile_'+elec+'_in_ts.txt', header = None, sep='\t') / sr)
        except TypeError:
            print(TypeError, 'deadfile',elec, pd.read_csv(path_folder+'derivatives/'+patient+'_stim'+session+'_deadfile_'+elec+'_in_ts.txt', header = None, sep='\t'))
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
    if re.match(r"^[A-Z]{1,3}(_p|_|p_|p|'|)$", name):
        # Détection du côté
        if name.endswith("p") or name.endswith("_p") or name.endswith("'") or name.endswith("p_"):
            side = "L"
            core = re.sub(r"(_?p|'$)", "", name)   # on enlève le suffixe gauche
            core = re.sub(r"_$", "", name)         # on enlève "_" final éventuel
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
        print(f"{macro_tt} = lateral tetrodes")
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

def distance_semi_qualitative(distance_tt_stim, sameElec, sameLobe, stim_Lobe, lobe_tt):
    if distance_tt_stim <= 40:
        distance_semi_quali = 'local_inf4cm'
    elif sameElec == True:
        distance_semi_quali = 'local_elec'
    elif sameLobe == True:
        distance_semi_quali = 'local_lobe'
    elif stim_Lobe[0] != lobe_tt[0]: # controlat
        distance_semi_quali = 'controlat'
    else: # np.nan si aucune des conditions précédentes ne convient
        distance_semi_quali = np.nan
    return distance_semi_quali

def remove_laterality(s): # enleve R, R., L, L. au debut d'une loca ou lobe
    return re.sub(r"^[LR]\.?\s+", "", s)

############### ZE ou hors ZE ###############

def is_in_ZEZIZPZLNI(patient, elec, plots, root):
    '''
    Determine si une paire de plots est en ZE/ZI/ZP/ZL/Non-involved ou pas. Renvoie une liste de 5 bool.
    Args:
        patient (str): nom du patient
        elec (str): nom de l'électrode macro
        plots (str): plots de l'électrode macro
    '''
    DF_EZPZIZ = pd.read_excel(root+'Spike-sorting/Tables/ZeZiZpZl.xlsx')
    dict_index_zone = {'ZE':list(DF_EZPZIZ.columns).index('ZE_Electrode'), 'ZI':list(DF_EZPZIZ.columns).index('ZI_Electrode'), 'ZP':list(DF_EZPZIZ.columns).index('ZP_Electrode'), 'ZL':list(DF_EZPZIZ.columns).index('ZL_Electrode')} # indice de colonne avec electrodes de chaque zone
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



##################################################
# Neuronal Metrics functions
##################################################

############### Quality Metrics ###############

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


############### Waveform features + quality metrics extraction ###############


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

    # Exception connue : Neuralynx passé par format intermédiaire MEDD,
    # déjà dans le bon sens.
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


from scipy.interpolate import interp1d

# def standardize_waveform_for_wavemap(
#     waveform,
#     fs,
#     ms_before,
#     target_time_ms=np.linspace(-1.0, 2.0, 121),
#     force_negative_trough=True,
# ):
#     """
#     Prend une waveform moyenne native et renvoie une waveform standardisée
#     utilisable par WaveMAP.

#     waveform : 1D array, best channel, en µV
#     fs : sampling rate en Hz
#     ms_before : durée pré-spike utilisée lors de l'extraction SI
#     target_time_ms : grille commune, en ms
#     """

#     waveform = np.asarray(waveform, dtype=float)

#     # Temps natif, relatif au début de la fenêtre.
#     native_time_ms = (np.arange(len(waveform)) / fs) * 1000.0 - ms_before

#     # Baseline : début de fenêtre.
#     n_base = max(3, int(0.2 / 1000 * fs))
#     waveform = waveform - np.nanmedian(waveform[:n_base])

#     # Polarité commune : trough négatif.
#     sign_flipped = False
#     if force_negative_trough and abs(np.nanmax(waveform)) > abs(np.nanmin(waveform)):
#         waveform = -waveform
#         sign_flipped = True

#     # Alignement sur trough.
#     trough_idx = int(np.nanargmin(waveform))
#     trough_time_ms = native_time_ms[trough_idx]
#     aligned_time_ms = native_time_ms - trough_time_ms

#     # Interpolation sur grille commune.
#     f = interp1d(
#         aligned_time_ms,
#         waveform,
#         kind="linear",
#         bounds_error=False,
#         fill_value=np.nan,
#     )
#     w_std = f(target_time_ms)

#     # Normalisation amplitude.
#     denom = np.nanmax(np.abs(w_std))
#     if denom > 0:
#         w_norm = w_std / denom
#     else:
#         w_norm = w_std * np.nan

#     return w_std, w_norm, sign_flipped, trough_time_ms

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
    import numpy as np

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


import numpy as np
import pandas as pd
import spikeinterface as si


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


############### Creation des tables de resultats par session ###############

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
    À appeler dans le notebook session par session.
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


def build_celltype_global_cache(root): 
    # concatène tous les caches session. se lance seulement qd besoin de 
    # reconstruire la base globale après plusieurs sessions nouvellement extraites.

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

##################### qqs metriques de comportement neuronal ################

def _deadfile_to_intervals(deadfile):
    """
    Convertit un deadfile en array (n_intervals, 2), en secondes.

    Accepte :
    - pd.DataFrame avec colonnes 0 et 1
    - array-like n x 2
    - None
    """
    import numpy as np
    import pandas as pd

    if deadfile is None:
        return np.empty((0, 2), dtype=float)

    if isinstance(deadfile, pd.DataFrame):
        if 0 in deadfile.columns and 1 in deadfile.columns:
            arr = deadfile[[0, 1]].to_numpy(dtype=float)
        else:
            arr = deadfile.iloc[:, :2].to_numpy(dtype=float)
    else:
        arr = np.asarray(deadfile, dtype=float)

    if arr.size == 0:
        return np.empty((0, 2), dtype=float)

    arr = np.asarray(arr, dtype=float).reshape(-1, 2)
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    arr = arr[arr[:, 1] > arr[:, 0]]

    if len(arr) == 0:
        return np.empty((0, 2), dtype=float)

    arr = arr[np.argsort(arr[:, 0])]

    # Merge des intervalles qui se chevauchent.
    merged = []
    for s, e in arr:
        if not merged or s > merged[-1][1]:
            merged.append([float(s), float(e)])
        else:
            merged[-1][1] = max(merged[-1][1], float(e))

    return np.asarray(merged, dtype=float)

def _intervals_cross_dead(t0, t1, dead_intervals):
    """
    Retourne un booléen array indiquant si chaque intervalle [t0, t1]
    croise un dead interval.

    t0, t1 : arrays de même longueur, en secondes.
    dead_intervals : array n x 2, en secondes.
    """
    import numpy as np

    t0 = np.asarray(t0, dtype=float)
    t1 = np.asarray(t1, dtype=float)

    if dead_intervals is None or len(dead_intervals) == 0:
        return np.zeros(len(t0), dtype=bool)

    starts = dead_intervals[:, 0]
    ends = dead_intervals[:, 1]

    # Pour chaque intervalle [t0, t1], on cherche s'il existe un dead interval
    # tel que dead_start < t1 et dead_end > t0.
    prefix_max_ends = np.maximum.accumulate(ends)

    last_possible = np.searchsorted(starts, t1, side="left")
    crosses = np.zeros(len(t0), dtype=bool)

    valid = last_possible > 0
    crosses[valid] = prefix_max_ends[last_possible[valid] - 1] > t0[valid]

    return crosses

def compute_isi_features(
    spk_times,
    dead_intervals=None,
    refractory_ms=2.0,
    burst_ms_1=6.0,
    burst_ms_2=10.0,
):
    """
    Calcule des features ISI simples pour un neurone.

    spk_times : array-like, temps des spikes en secondes.
    dead_intervals : array-like n x 2, intervalles morts en secondes.

    Retourne des colonnes scalaires :
    - n_spikes_isi
    - n_isi_total
    - n_isi_valid
    - isi_mean_ms
    - isi_median_ms
    - isi_std_ms
    - cv_isi
    - cv2_isi
    - refractory_violations_count
    - refractory_violations_ratio
    - burst_index_6ms
    - burst_index_10ms
    """
    import numpy as np

    spk_times = np.asarray(spk_times, dtype=float)
    spk_times = spk_times[np.isfinite(spk_times)]
    spk_times = np.sort(spk_times)

    out = {
        "n_spikes_isi": len(spk_times),
        "n_isi_total": np.nan,
        "n_isi_valid": np.nan,
        "isi_mean_ms": np.nan,
        "isi_median_ms": np.nan,
        "isi_std_ms": np.nan,
        "cv_isi": np.nan,
        "cv2_isi": np.nan,
        "refractory_violations_count": np.nan,
        "refractory_violations_ratio": np.nan,
        "burst_index_6ms": np.nan,
        "burst_index_10ms": np.nan,
    }

    if len(spk_times) < 2:
        out["n_isi_total"] = 0
        out["n_isi_valid"] = 0
        return out

    dead_intervals = _deadfile_to_intervals(dead_intervals)

    t0 = spk_times[:-1]
    t1 = spk_times[1:]
    isi_s = t1 - t0

    valid = isi_s > 0

    if len(dead_intervals) > 0:
        crosses_dead = _intervals_cross_dead(t0, t1, dead_intervals)
        valid &= ~crosses_dead

    isi_s_valid = isi_s[valid]
    isi_ms = isi_s_valid * 1000.0

    out["n_isi_total"] = len(isi_s)
    out["n_isi_valid"] = len(isi_ms)

    if len(isi_ms) == 0:
        return out

    mean_isi = np.nanmean(isi_ms)

    out["isi_mean_ms"] = mean_isi
    out["isi_median_ms"] = np.nanmedian(isi_ms)
    out["isi_std_ms"] = np.nanstd(isi_ms, ddof=1) if len(isi_ms) > 1 else np.nan
    out["cv_isi"] = out["isi_std_ms"] / mean_isi if mean_isi > 0 and np.isfinite(out["isi_std_ms"]) else np.nan

    # CV2 : mesure locale de variabilité, moins sensible aux changements lents de firing rate.
    if len(isi_ms) >= 2:
        denom = isi_ms[1:] + isi_ms[:-1]
        valid_cv2 = denom > 0
        if np.any(valid_cv2):
            cv2 = 2.0 * np.abs(isi_ms[1:] - isi_ms[:-1]) / denom
            out["cv2_isi"] = np.nanmean(cv2[valid_cv2])

    refractory = isi_ms < refractory_ms
    out["refractory_violations_count"] = int(np.sum(refractory))
    out["refractory_violations_ratio"] = float(np.mean(refractory)) if len(isi_ms) > 0 else np.nan

    out["burst_index_6ms"] = float(np.mean(isi_ms < burst_ms_1))
    out["burst_index_10ms"] = float(np.mean(isi_ms < burst_ms_2))

    return out

def compute_acg_features(
    spk_times,
    dead_intervals=None,
    bin_ms=1.0,
    window_ms=100.0,
    refractory_ms=2.0,
    burst_start_ms=3.0,
    burst_end_ms=10.0,
    mid_start_ms=10.0,
    mid_end_ms=50.0,
):
    """
    Calcule des features d'autocorrélogramme positives-lags.

    Retourne :
    - acg_bin_ms
    - acg_window_ms
    - acg_n_lags
    - acg_peak_lag_ms
    - acg_peak_count
    - acg_peak_rate_hz
    - acg_mean_3_10ms_rate_hz
    - acg_mean_10_50ms_rate_hz
    - acg_refractory_count_2ms
    - acg_refractory_ratio_2ms
    - acg_burst_ratio_3_10_over_3_50ms
    """
    import numpy as np

    spk_times = np.asarray(spk_times, dtype=float)
    spk_times = spk_times[np.isfinite(spk_times)]
    spk_times = np.sort(spk_times)

    out = {
        "acg_bin_ms": bin_ms,
        "acg_window_ms": window_ms,
        "acg_n_lags": np.nan,
        "acg_peak_lag_ms": np.nan,
        "acg_peak_count": np.nan,
        "acg_peak_rate_hz": np.nan,
        "acg_mean_3_10ms_rate_hz": np.nan,
        "acg_mean_10_50ms_rate_hz": np.nan,
        "acg_refractory_count_2ms": np.nan,
        "acg_refractory_ratio_2ms": np.nan,
        "acg_burst_ratio_3_10_over_3_50ms": np.nan,
    }

    if len(spk_times) < 2:
        out["acg_n_lags"] = 0
        return out

    dead_intervals = _deadfile_to_intervals(dead_intervals)

    max_lag_s = window_ms / 1000.0
    lags = []

    # Sliding window simple.
    for i, t in enumerate(spk_times[:-1]):
        j_end = np.searchsorted(spk_times, t + max_lag_s, side="right")

        if j_end <= i + 1:
            continue

        candidate_times = spk_times[i + 1:j_end]
        candidate_lags = candidate_times - t

        if len(dead_intervals) > 0:
            t0 = np.full(len(candidate_times), t)
            t1 = candidate_times
            crosses_dead = _intervals_cross_dead(t0, t1, dead_intervals)
            candidate_lags = candidate_lags[~crosses_dead]

        if len(candidate_lags) > 0:
            lags.append(candidate_lags)

    if len(lags) == 0:
        out["acg_n_lags"] = 0
        return out

    lags_s = np.concatenate(lags)
    lags_ms = lags_s * 1000.0

    out["acg_n_lags"] = len(lags_ms)

    bins = np.arange(0, window_ms + bin_ms, bin_ms)
    counts, edges = np.histogram(lags_ms, bins=bins)
    centers = edges[:-1] + bin_ms / 2.0

    # Normalisation approximative en taux par spike déclencheur.
    # counts / n_spikes / bin_s
    bin_s = bin_ms / 1000.0
    acg_rate_hz = counts / max(len(spk_times), 1) / bin_s

    # Peak hors réfractaire.
    peak_mask = centers >= refractory_ms

    if np.any(peak_mask):
        idx_local = np.argmax(counts[peak_mask])
        idx = np.where(peak_mask)[0][idx_local]

        out["acg_peak_lag_ms"] = float(centers[idx])
        out["acg_peak_count"] = int(counts[idx])
        out["acg_peak_rate_hz"] = float(acg_rate_hz[idx])

    # Fenêtres utiles.
    burst_mask = (centers >= burst_start_ms) & (centers < burst_end_ms)
    mid_mask = (centers >= mid_start_ms) & (centers < mid_end_ms)
    broad_burst_mask = (centers >= burst_start_ms) & (centers < mid_end_ms)
    refractory_mask = centers < refractory_ms

    if np.any(burst_mask):
        out["acg_mean_3_10ms_rate_hz"] = float(np.nanmean(acg_rate_hz[burst_mask]))

    if np.any(mid_mask):
        out["acg_mean_10_50ms_rate_hz"] = float(np.nanmean(acg_rate_hz[mid_mask]))

    out["acg_refractory_count_2ms"] = int(np.sum(counts[refractory_mask]))
    out["acg_refractory_ratio_2ms"] = (
        float(np.sum(counts[refractory_mask]) / len(lags_ms))
        if len(lags_ms) > 0 else np.nan
    )

    denom = np.sum(counts[broad_burst_mask])
    if denom > 0:
        out["acg_burst_ratio_3_10_over_3_50ms"] = float(np.sum(counts[burst_mask]) / denom)

    return out

def compute_spiketrain_temporal_features(
    spk_times,
    dead_intervals=None,
    acg_bin_ms=1.0,
    acg_window_ms=100.0,
):
    """
    Wrapper unique pour les features temporelles d'un neurone.
    """
    out = {}

    out.update(
        compute_isi_features(
            spk_times=spk_times,
            dead_intervals=dead_intervals,
        )
    )

    out.update(
        compute_acg_features(
            spk_times=spk_times,
            dead_intervals=dead_intervals,
            bin_ms=acg_bin_ms,
            window_ms=acg_window_ms,
        )
    )

    return out


def compute_neuronal_summary(spikes, stims_loca, dict_clu2tt, dict_elec2deadfile, mpg, patient, session, root, mapping_anat, epsilon=0.1, verb=False, bin_z=0.05, bin_resp=[0.05, 0.075, 0.1]):
    """
    Construit deux tables de résumé pour une session :
    1) summary_by_neuron
       -> une ligne par neurone
    2) summary_by_neuron_and_stim
       -> une ligne par trial (neurone, stimulation)

    Contenu général
    ---------------
    Pour chaque neurone, on stocke :
    - ses quality metrics
    - son firing rate global
    - son firing rate sur une baseline
    - sa localisation anatomique et dans les zones ZE/ZI/ZP/ZL/NI

    Pour chaque couple (neurone, stimulation), on ajoute aussi :
    - firing rate pré-stim, post-stim
    - variations relatives
    - log-ratio
    - z-score basé sur la période pré-stim
    - modulation index
    - distance stimulation / tétrode
    - effets cognitifs s'ils sont renseignés
    - indicateurs binaires de réponse inhibitrice / excitatrice à plusieurs tailles de bins

    Paramètres
    ----------
    spikes : TsGroup-like
        Activité de tous les neurones de la session, avec temps de spikes (en s)
    stims_loca : pd.DataFrame
        Tableau des stimulations, contenant 't', 'durée', 'paramètres', 'electrode', 'plots', 'frequence', 'intensite', 'lobe'
    dict_clu2tt : dict
        Mapping :
            clu (ID neurone dans spikes) -> nom de tétrade
    dict_elec2deadfile : dict
        Mapping : electrode -> deadfile
    mpg : pd.DataFrame. Table d'information anatomique (tt, lobe, loca, etc.).
    patient, session, root, mapping_anat :
        Paramètres de contexte / I/O.
    epsilon : float
        Petite constante pour stabiliser le calcul du log-ratio.
    verb : bool
        Passe à certaines fonctions auxiliaires de localisation.
    bin_z : float
        Taille des bins (en s) pour estimer le z-score pré-stim.
    bin_resp : list[float]
        Tailles de bins testées pour classifier les réponses (inhib/excit).

    Retour
    ------
    (summary_df, general_df) : tuple[pd.DataFrame, pd.DataFrame]
        - summary_df : une ligne par neurone
        - general_df : une ligne par couple neurone x stimulation
    """
    # --------------
    # Initialisation
    # --------------
    import re
    path_folder = root + f'Spike-sorting/Data_folders/{patient}/{patient}_stim{session}/'
    stimic_session='stimic'+session
    all_clu_ids = list(dict_clu2tt.keys()) # liste des indices de tous les neurones
    data, general_data = [], []
    labels_stims = stims_loca['paramètres'] # labels des stimulations
    coord_MNI = pd.read_excel(root+'Spike-sorting/Tables/MNI_all_patients.xlsx')
    coord_MNI_pat = coord_MNI[coord_MNI['Patient']==patient] # coordonnées MNI pour ce patient

    # variables pour obtenir dynamiques temporelles par U de temps
    pre_duration, post_duration = 10, 10  # sec
    
    # on charge les quality metrics de chaque SU de la session
    # qm = quality_metrics_session(patient, session, mapping_anat, dict_elec2deadfile, dict_clu2tt, root)#[list_col_qm]
    # list_col_qm = ['amplitude_median', 'num_spikes', 'presence_ratio', 'amplitude_cutoff', 'snr', 'isi_violations_ratio']
    # list_col_qm_exported = [col for col in list_col_qm if col in list(qm.columns)]

    # --- Chargement éventuel d'un fichier de stim avec colonne cognitive ("cog") ---
    # Ce fichier, s'il existe, correspond ligne à ligne à stims_loca
    # avec une colonne supplémentaire 'cog' en fin de table.
    path_cog = (path_folder + f"{patient}_stim{session}_stim_events_TRC_re-shifted_loca_COG.txt")
    if os.path.exists(path_cog):
        stims_with_cog = pd.read_csv(path_cog, sep=';')
        cog_by_stim = stims_with_cog['cog'].apply(lambda x: np.nan if pd.isna(x) or str(x).strip() == '' else ast.literal_eval(x))
    else:
        cog_by_stim = pd.DataFrame([np.nan for _ in range(stims_loca.shape[0])],columns=['cog'])['cog']
    # --- Chargement éventuel d'un fichier de stim avec colonne "post_decharge" ---
    # Ce fichier, s'il existe, correspond ligne à ligne à stims_loca
    # avec une colonne supplémentaire 'post_decharge' en fin de table, avec liste de tetrodes avec AD par stim
    path_AD = (path_folder + f"{patient}_stim{session}_stim_events_TRC_re-shifted_loca_AD.txt")
    if os.path.exists(path_AD):
        stims_with_AD = pd.read_csv(path_AD, sep=';')
        AD_by_stim = stims_with_AD['cog'].apply(lambda x: np.nan if pd.isna(x) or str(x).strip() == '' else ast.literal_eval(x))
    else:
        AD_by_stim = pd.DataFrame([np.nan for _ in range(stims_loca.shape[0])])

    for _, clu in enumerate(all_clu_ids): # pour chaque neurone
        
        spk_times = spikes[clu].index.values  # Liste des t des spikes de ce neurone en sec
        if len(spk_times) == 0: # si aucun spike, on passe au neurone suivant
            continue
        
        # Initialisation de la ligne "par neurone" 
        lobe_tt = mpg.loc[mpg['tt'] == dict_clu2tt[clu], 'lobe'].values[0] # aussi utilisé après
        row_unit = {'patient': patient, 'session':stimic_session, 'clu': clu, 'tetrode': dict_clu2tt[clu],
                    'lobe_tt' : lobe_tt,
                    'loca_tt' : mpg.loc[mpg['tt'] == dict_clu2tt[clu], 'loca'].values[0],
                    'lobe_tt_noLat' : remove_laterality(lobe_tt),
                    'loca_tt_noLat' : remove_laterality(mpg.loc[mpg['tt'] == dict_clu2tt[clu], 'loca'].values[0])} # Initialisation de la ligne de ce neurone
        
        # ajout des quality metrics :
        # if clu in list(qm.index):
        #     for metric in list_col_qm_exported:
        #         row_unit[metric] = qm.loc[clu, metric]
        # else:
        #     for metric in list_col_qm_exported:
        #         row_unit[metric] = np.nan

        # fr_global / Taux de décharge global :
        deadfile_elec = dict_elec2deadfile[dict_clu2tt[clu][:-1]]  # deadfile de l'electrode correspondante
        total_duration = get_total_duration(path_folder, patient, session, nb_channels(mapping_anat, patient, session, root)) - np.sum(deadfile_elec[1] - deadfile_elec[0]) # on soustrait deadperiods
        row_unit['fr_global'] = len(spk_times) / (total_duration ) if total_duration > 0 else np.nan # nb de spikes / durée totale
        
        # Features temporelles spike-train : ISI + ACG 
        temporal_features = compute_spiketrain_temporal_features(
            spk_times=spk_times,
            dead_intervals=deadfile_elec,
            acg_bin_ms=1.0,
            acg_window_ms=100.0,)
        row_unit.update(temporal_features)

        # fr_baseline / Taux de décharge sur une baseline des premières minutes (0 à la stim 1, en retirant les dead periods) :
        start_baseline = 0
        end_baseline = stims_loca.loc[0,'t']
        artefacts_filtered_baseline = deadfile_elec[(deadfile_elec[1] >= start_baseline) & (deadfile_elec[0] <= end_baseline)]
        dead_baseline = np.sum(np.minimum(artefacts_filtered_baseline[1], end_baseline) - np.maximum(artefacts_filtered_baseline[0], start_baseline))  
        spk_in_baseline = spk_times[spk_times <= end_baseline] # tous les spikes avant la fin de la baseline
        spk_in_baseline = spk_in_baseline[spk_in_baseline >= start_baseline] # tous les spikes après le début de la baseline
        row_unit['fr_baseline'] = len(spk_in_baseline) / (end_baseline-start_baseline-dead_baseline) if (end_baseline-start_baseline-dead_baseline) > 0 else np.nan

        # tt_in_ZE, etc 
        ttZE, ttZI, ttZP, ttZL, ttNI = is_in_ZEZIZPZLNI(patient, dict_clu2tt[clu][:-1], find_back_macrocontacts_from_tt(normalize_name(dict_clu2tt[clu][:-1]), coord_MNI_pat,verb=verb)[1], root)
        row_unit['tt_en_ZE'],row_unit['tt_en_ZI'],row_unit['tt_en_ZP'],row_unit['tt_en_ZL'],row_unit['tt_en_NI'] = ttZE, ttZI, ttZP, ttZL, ttNI
            
        for i, stim in stims_loca.iterrows(): # Pour chaque stim:
            # fr_pre, fr_post / Taux de décharge pré-/post-stim et % de variation :
            t_pre_start = stim['t'] - pre_duration
            t_pre_end = stim['t']
            t_post_start = stim['t'] + stim['durée']
            t_post_end = stim['t + durée'] + post_duration
            pre_spike_times = spk_times[(spk_times >= t_pre_start) & (spk_times < t_pre_end)]
            post_spike_times = spk_times[(spk_times >= t_post_start) & (spk_times < t_post_end)]

            artefacts_filtered_pre = deadfile_elec[(deadfile_elec[1] >= t_pre_start) & (deadfile_elec[0] <= t_pre_end)]
            artefacts_filtered_post = deadfile_elec[(deadfile_elec[1] >= t_post_start) & (deadfile_elec[0] <= t_post_end)]
            dead_pre = np.sum(np.minimum(artefacts_filtered_pre[1], t_pre_end) - np.maximum(artefacts_filtered_pre[0], t_pre_start)) 
            dead_post = np.sum(np.minimum(artefacts_filtered_post[1], t_post_end) - np.maximum(artefacts_filtered_post[0], t_post_start))  
            fr_pre = len(pre_spike_times) / (pre_duration - dead_pre) # les 10 sec sont diminuees avec quantité d'artefact dans deadfile
            fr_post = len(post_spike_times) / (post_duration - dead_post) # les 10 sec sont diminuees avec quantité d'artefact dans deadfile

            # Quantifier la reponse post-stim :
            # Wilcoxon pre/post :
            # on met d'abord a jour la somme des periodes enlevées sur ce trial
            artefacts_filtered_pre = deadfile_elec[(deadfile_elec[1] >= stim['t'] -10) & (deadfile_elec[0] <= stim['t'])]
            artefacts_filtered_post = deadfile_elec[(deadfile_elec[1] >= stim['t + durée']) & (deadfile_elec[0] <= stim['t + durée']+10)]
            
            _, pval_wilcoxon, _, _, _ = _compute_binned_rates_and_wilcoxon(
                pre_spikes_rel=pre_spike_times - stim['t'],   # temps relatifs à stim_start
                post_spikes_rel= post_spike_times - stim['t + durée'],   # temps relatifs à stim_end
                stim_start=stim['t'],
                stim_end=stim['t + durée'],
                deadfile_elec=deadfile_elec,
                bin_size=0.1,     
                window=10,
                min_valid_bin_frac=0.8)
                
            # delta_pre_post, delta_baseline_post / % de variation :
            if fr_pre == 0:
                delta_pre_post = np.nan
            else:
                delta_pre_post = 100 * (fr_post - fr_pre) / fr_pre
            if row_unit['fr_baseline'] == 0:
                delta_baseline_post = np.nan
            else:  
                delta_baseline_post = 100 * (fr_post - row_unit['fr_baseline']) / row_unit['fr_baseline']
            
            # Log-ratio : variation symétrique (hausse/diminution comparables). epsilon pr eviter division par 0
            log_ratio = np.log((fr_post + epsilon) / (fr_pre + epsilon)) if fr_pre >= 0 else np.nan
            
            # Z-score pre : ( FRpost - mean_pre ) / std_pre
            # attention dans vanderPlas prennent comme baseline la baseline de tous les trials
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

            # Distance semi-qualitative (inf4cm ; meme electrode ; meme lobe ; controlatéral) :
            sameElec = electrodes_equal(dict_clu2tt[clu][:-1],stims_loca.loc[i,'electrode']) # localité du neurone par rapport aux stimulations
            sameLobe = (mpg.loc[mpg['tt'] == dict_clu2tt[clu], 'lobe'].values[0] == stim['lobe'].strip())
            stim_Lobe = stim['lobe'].strip()
            distance_semi_quali = distance_semi_qualitative(distance_tt_stim, sameElec, sameLobe, stim_Lobe, lobe_tt)

            # Stim en ZE, ZI, etc / Tetrode en ZE, ZI, etc :
            stimZE, stimZI, stimZP, stimZL, stimNI = is_in_ZEZIZPZLNI(patient, stims_loca.loc[i, 'electrode'], stims_loca.loc[i, 'plots'], root)
            
            # Stockage de ces variable dans general_data (summary_by_neuron_and_stim)
            # avec une ligne entiere pour chaque stim et chaque neurone
            row_trial = {'patient': patient, 'session':stimic_session, 'clu': clu, 
                        # infos sur tetrode :
                        'tetrode': dict_clu2tt[clu], 
                        'lobe_tt': lobe_tt,
                        'loca_tt': mpg.loc[mpg['tt'] == dict_clu2tt[clu], 'loca'].values[0],
                        'lobe_tt_noLat' : remove_laterality(lobe_tt),
                        'loca_tt_noLat' : remove_laterality(mpg.loc[mpg['tt'] == dict_clu2tt[clu], 'loca'].values[0]),
                        'tt_in_ZE': ttZE, 'tt_in_ZI': ttZI, 'tt_in_ZP': ttZP, 'tt_in_ZL': ttZL, 'tt_in_NI': ttNI,

                        # infos sur stim :
                        'stim_label':labels_stims[i][:-8], 'ind_stim' : i,
                        'stim_Lobe': stim_Lobe, 'stim_Lobe_noLat' : remove_laterality(stim_Lobe),
                        'stim_in_ZE': stimZE, 'stim_in_ZI': stimZI, 'stim_in_ZP': stimZP, 'stim_in_ZL': stimZL, 'stim_in_NI': stimNI,
                        'freq_stim': int(stim['frequence'].strip()[:-3]), 'intensity_stim': float(stim['intensite'].strip()[:-3]), 
                        'cog': cog_by_stim.loc[i], #'after_discharge':[True if type(AD_by_stim.loc[i])==list else False][0], # AD vrai s'il y a une AD avec cette stim
                        'after_discharge_loc': [True if dict_clu2tt[clu] in AD_by_stim.loc[i] else False][0], # AD local vrai si tetrode dans liste de tetrodes concernées par une AD 

                        # Topographie : sameElec, sameLobe / Distance avec la stim / stim ou tt en ZE,ZI,ZP,ZL,NI 
                        'sameElec': sameElec, 'sameLobe': sameLobe, 
                        'distance_tt_stim': distance_tt_stim, 'distance_semi_quali':distance_semi_quali,

                        # infos sur dynamique neuronale :
                        'fr_global': row_unit['fr_global'], 'fr_baseline': row_unit['fr_baseline'],
                        'pval_wilcoxon':pval_wilcoxon, 'wilcoxon_signif':[True if pval_wilcoxon < 0.05 else False][0],
                        'delta_pre_post':delta_pre_post, 'delta_baseline_post':delta_baseline_post, 
                        'fr_pre': fr_pre, 'fr_post': fr_post,  'log_ratio': log_ratio,
                        'zscore_pre': z_score_pre, 'modulation_index': modulation_index}
            
            for col, val in temporal_features.items():
                row_trial[col] = val
            # quality metrics :
            # if clu in list(qm.index):
            #     for metric in qm.columns.tolist():
            #         row_trial[metric] = qm.loc[clu, metric]
            # else:
            #     for metric in qm.columns.tolist():
            #         row_trial[metric] = np.nan

            # FR_pre(t), FR_post(t): pre_counts_i and post_counts_i with bin_r size = [0.05, 0.075, 0.1]
            for bin_r_i in bin_resp:
                bins_edges_i = np.arange(0, post_duration + bin_r_i, bin_r_i)
                post_counts_i, _ = np.histogram(post_spike_times - t_post_start, bins=bins_edges_i)
                pre_counts_i, _ = np.histogram(pre_spike_times - t_pre_start, bins=bins_edges_i)
            
            # Seuil de reponse significative : mean_pre +/- 2*std_pre
                mean_pre = pre_counts_i.mean()
                std_pre = pre_counts_i.std(ddof=1)
                upper_thr = mean_pre + 2*std_pre
                lower_thr = mean_pre - 2*std_pre
                above = np.where(post_counts_i > upper_thr)[0] # depassement du seuil superieur
                below = np.where(post_counts_i < lower_thr)[0] # depassement du seuil inferieur
                
                row_trial[f'inhib_only_{bin_r_i}s_bins'] = int(len(below) > 0 and len(above) == 0) 
                row_trial[f'excit_only_{bin_r_i}s_bins'] = int(len(above) > 0 and len(below) == 0) 
                row_trial[f'inhib_then_excit_{bin_r_i}s_bins'] = int(len(below) > 0 and len(above) > 0 and below[0] < above[0])
                row_trial[f'excit_then_inhib_{bin_r_i}s_bins'] = int(len(below) > 0 and len(above) > 0 and above[0] < below[0])
                row_trial[f'inhib_general_{bin_r_i}s_bins'] = row_trial[f'inhib_only_{bin_r_i}s_bins'] + row_trial[f'inhib_then_excit_{bin_r_i}s_bins'] + row_trial[f'excit_then_inhib_{bin_r_i}s_bins']
                row_trial[f'excit_general_{bin_r_i}s_bins'] = row_trial[f'excit_only_{bin_r_i}s_bins'] + row_trial[f'inhib_then_excit_{bin_r_i}s_bins'] + row_trial[f'excit_then_inhib_{bin_r_i}s_bins']

            general_data.append(row_trial) # une ligne par neurone et par stim
        data.append(row_unit) # une ligne par neurone

    return (pd.DataFrame(data), pd.DataFrame(general_data))


def create_or_update_session_summary(patient, session, root='D:/',verb=False, bin_z=0.05, bin_resp=[0.05, 0.075, 0.1]):
    ''' Cree le tableau récapitulatif pour une session: summary_by_neuron et general_summary_by_neuron_and_stim, a partir du patient et session
    Tourne pendant environ 30 sec/1 min par session. '''
    print(f"=== Create or update session summary for {patient}, session {session}. ===")
    path_folder = root + 'Spike-sorting/Data_folders/'+patient+'/'+patient+'_stim'+session+'/'
    sr = get_SR(patient)
    spikes = get_nwb(patient, session, root)
    stims_loca = get_stims(patient, session, root)
    mapping_anat = pd.read_csv(root + 'Spike-sorting/Data_folders/'+patient+'/mapping_anat_'+patient+'.txt', sep=',', engine='python')
    dict_elec2deadfile = get_dict_deadfiles(mapping_anat, patient, session, path_folder, sr)
    dict_clu2tt = get_dict_tetrodeName_from_tetrodeIndex(spikes, mapping_anat)

    # summary tables for the session:
    tables_folder = root + 'Spike-sorting/Tables'+'/'
    if not os.path.exists(tables_folder): # si le dossier de Tables n'existe pas encore, alors on le crée
        os.makedirs(tables_folder)
    summary_df, general_df = compute_neuronal_summary(spikes, stims_loca, dict_clu2tt, dict_elec2deadfile, mapping_anat, patient, session, root, mapping_anat, verb=verb, bin_z=bin_z, bin_resp=bin_resp)
    summary_df.to_csv(tables_folder+patient+'_stim'+session+"_summary_by_neuron.csv", index=False)
    summary_df.to_excel(tables_folder+patient+'_stim'+session+"_summary_by_neuron.xlsx", index=False)
    general_df.to_csv(tables_folder+patient+'_stim'+session+"_general_summary_by_neuron_and_stim.csv", index=False)
    general_df.to_excel(tables_folder+patient+'_stim'+session+"_general_summary_by_neuron_and_stim.xlsx", index=False)
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
            _,_  = create_or_update_session_summary(patient, session[-1], root=root, verb=verb, bin_z=bin_z, bin_resp=bin_resp)
            print('Update done for ', session)


############### Big dataframe functions ###############

def update_general_summary_on_all_sessions(root='D:/'):
    '''Tourne sur tous les session_summaries du dossier Spike-sorting/Tables et recrée et renvoie big_df_trials
    Tourne eviron 40 sec pour une vingtaine de sessions.'''
    path_tables = Path(root, "Spike-sorting/Tables")
    
    all_summaries_trials, all_summaries_nrn = [], [] # Liste de DF, pour stocker tous les csv, qui seront concaténés

    path_summaries_trials = path_tables.rglob("*general_summary_by_neuron_and_stim.csv")
    path_summaries_nrn = path_tables.rglob("*_summary_by_neuron.csv")
    
    for path in path_summaries_trials:
        df_summary_trials = pd.read_csv(path)
        # On crée une colonne 'global_clu' unique pour s'y retrouver dans l'indexation de l'ensemble des neurones
        df_summary_trials["global_clu"] = (df_summary_trials["clu"].astype(str).apply(lambda x: f"{df_summary_trials.loc[0,"patient"]}_{df_summary_trials.loc[0,"session"]}_{x}"))
        all_summaries_trials.append(df_summary_trials)

    for path in path_summaries_nrn:
        df_summary_nrn = pd.read_csv(path)
        all_summaries_nrn.append(df_summary_nrn)

    # Empile tous les dataframes de type summary
    big_df_trials = pd.concat(all_summaries_trials, axis=0, ignore_index=True)
    big_df_by_nrn = pd.concat(all_summaries_nrn, axis=0, ignore_index=True)

    # Export des dataframes generaux
    big_df_trials.to_excel(root+"Spike-sorting/Tables/general_summary_all_sessions.xlsx", index=False)
    big_df_trials.to_csv(root+"Spike-sorting/Tables/general_summary_all_sessions.csv", index=False)
    
    big_df_by_nrn.to_excel(root+"Spike-sorting/Tables/summary_by_nrn_all_sessions.xlsx", index=False)
    big_df_by_nrn.to_csv(root+"Spike-sorting/Tables/summary_by_nrn_all_sessions.csv", index=False)
    
    return big_df_trials


##################################################
# Raster display functions
##################################################

def _interval_overlap_length(a_start, a_end, b_start, b_end):
    return max(0, min(a_end, b_end) - max(a_start, b_start))

def _compute_binned_rates_and_wilcoxon(
    pre_spikes_rel,
    post_spikes_rel,
    stim_start,
    stim_end,
    deadfile_elec,
    bin_size=0.1,              # 100 ms conseillé comme défaut
    window=10,
    min_valid_bin_frac=0.8     # on garde le bin si au moins 80% du bin est exploitable
):
    """
    pre_spikes_rel : spikes pré, en temps relatifs à stim_start, donc entre -window et 0
    post_spikes_rel : spikes post, en temps relatifs à stim_end, donc entre 0 et +window
    stim_start, stim_end : temps absolus
    deadfile_elec : array-like Nx2 avec [start, end] en temps absolus
    """
    deadfile_elec = np.asarray(deadfile_elec) # df to array
    pre_spikes_rel = np.asarray(pre_spikes_rel)
    post_spikes_rel = np.asarray(post_spikes_rel)

    from scipy.stats import wilcoxon
    edges = np.arange(0, window + bin_size, bin_size)
    n_bins = len(edges) - 1

    pre_rates = []
    post_rates = []

    pre_spikes_rel = np.asarray(pre_spikes_rel)
    post_spikes_rel = np.asarray(post_spikes_rel)

    for k in range(n_bins):
        # bornes relatives
        pre_rel_start = -window + k * bin_size
        pre_rel_end   = pre_rel_start + bin_size

        post_rel_start = k * bin_size
        post_rel_end   = post_rel_start + bin_size

        # bornes absolues
        pre_abs_start = stim_start + pre_rel_start
        pre_abs_end   = stim_start + pre_rel_end

        post_abs_start = stim_end + post_rel_start
        post_abs_end   = stim_end + post_rel_end

        # durée valide du bin pré
        removed_pre = 0.0
        artefacts_pre = deadfile_elec[
            (deadfile_elec[:, 1] > pre_abs_start) & (deadfile_elec[:, 0] < pre_abs_end)
        ]
        for a0, a1 in artefacts_pre:
            removed_pre += _interval_overlap_length(pre_abs_start, pre_abs_end, a0, a1)
        valid_pre = bin_size - removed_pre

        # durée valide du bin post
        removed_post = 0.0
        artefacts_post = deadfile_elec[
            (deadfile_elec[:, 1] > post_abs_start) & (deadfile_elec[:, 0] < post_abs_end)
        ]
        for a0, a1 in artefacts_post:
            removed_post += _interval_overlap_length(post_abs_start, post_abs_end, a0, a1)
        valid_post = bin_size - removed_post

        # on ne garde que les bins suffisamment exploitables des deux côtés
        if valid_pre < min_valid_bin_frac * bin_size or valid_post < min_valid_bin_frac * bin_size:
            continue

        # comptage des spikes dans le bin
        n_pre = np.sum((pre_spikes_rel >= pre_rel_start) & (pre_spikes_rel < pre_rel_end))
        n_post = np.sum((post_spikes_rel >= post_rel_start) & (post_spikes_rel < post_rel_end))

        # conversion en taux (Hz)
        pre_rates.append(n_pre / valid_pre)
        post_rates.append(n_post / valid_post)

    pre_rates = np.asarray(pre_rates)
    post_rates = np.asarray(post_rates)

    # pas assez de bins valides
    if len(pre_rates) < 5:
        return np.nan, np.nan, len(pre_rates), pre_rates, post_rates

    # Wilcoxon échoue si toutes les différences sont nulles
    if np.allclose(pre_rates - post_rates, 0):
        return np.nan, 1.0, len(pre_rates), pre_rates, post_rates
    try:
        stat, pval = wilcoxon(pre_rates, post_rates, zero_method='pratt', alternative='two-sided')
    except ValueError:
        stat, pval = np.nan, np.nan
    return stat, pval, len(pre_rates), pre_rates, post_rates

def plot_artefact_patches(ax, deadfile, stim_start, line_index, duration_out_array, side='pre', time_mode='relative', 
                          stim_duration=None, window=10, color='black', alpha=0.15, display=True):
    """
    Dessine un patch d'artefact sur l'axe donné.
    Args:
        ax (matplotlib.axes): Axe matplotlib sur lequel dessiner.
        deadfile (pd.DataFrame): DataFrame contenant les artefacts, avec début (col 0) et fin (col 1).
        stim_start (float): Début de la stimulation.
        line_index (int): Position verticale (ligne raster) du neurone ou de la stimulation.
        duration_out_array (np.array): Tableau accumulant la durée supprimée pour le neurone.
        side (str): 'pre' ou 'post'.
        time_mode (str): Si 'relative', centre la période autour de 0 (soustrait stim_start aux temps). Si 'absolute' on ne change rien, car les temps restent bruts, ne sont pas relatifs.
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
                if time_mode == 'relative': # on décale parce que qd on plot pr ttes les stims, l'absisse est centrée sur le début de la stimulation (-10 à +10 s)
                    display_start -= stim_start
                rect = plt.Rectangle((display_start, line_index - 0.23), width, 0.5,
                                    color=color, alpha=alpha, edgecolor='none')
                ax.add_patch(rect)


def rasters_OneNeuron_allStims(patient, session, spikes, stims, dict_clu2tt, dict_elec2deadfile, path_rasters, display_patches = True, plafond_inhib_100 = True):
    '''
    Returns and saves one raster per neuron, for all the EBS of the session. Runs on all neurons of the recording.
    Args:
        display_patches = True # si on veut afficher les patches gris des dead periods
        plafond_inhib_100 = True # si on veut que la variation de fréquence de décharge soit plafonnée à 100% qd inhibition (sinon, on change le calcul de % variation)
    '''
    from matplotlib.transforms import blended_transform_factory

    for ind_neuron in range(len(spikes)):
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
        wilcoxon_pvals, wilcoxon_labels, wilcoxon_nbins = [], [], []
        for ind_stim in range(stims.shape[0]):
            stim_start = stims['t'][ind_stim]
            stim_end = stims['t + durée'][ind_stim]
            # on met d'abord a jour la somme des periodes enlevées sur ce trial
            artefacts_filtered_pre = deadfile_elec[(deadfile_elec[1] >= stim_start-10) & (deadfile_elec[0] <= stim_start)]
            artefacts_filtered_post = deadfile_elec[(deadfile_elec[1] >= stim_end) & (deadfile_elec[0] <= stim_end+10)]
            durationOut_preStim[ind_stim] = np.sum(np.minimum(artefacts_filtered_pre[1], stim_start) - np.maximum(artefacts_filtered_pre[0], stim_start-10)) 
            durationOut_postStim[ind_stim] = np.sum(np.minimum(artefacts_filtered_post[1], stim_end+10) - np.maximum(artefacts_filtered_post[0], stim_end))  
            # puis on affiche les dead periods
            plot_artefact_patches(ax1, deadfile_elec, stim_start, ind_stim, durationOut_preStim, side='pre', time_mode='relative', display=display_patches)
            plot_artefact_patches(ax1, deadfile_elec, stim_start, ind_stim, durationOut_postStim, side='post', time_mode='relative', stim_duration=stims['durée'][ind_stim], display=display_patches)

            # Ajout Wilcoxon pré vs post par bins, pour chaque stimulation
            stat, pval, n_bins_used, pre_rates, post_rates = _compute_binned_rates_and_wilcoxon(
                pre_spikes_rel=spike_times_before[ind_stim],   # temps relatifs à stim_start
                post_spikes_rel=spike_times_after[ind_stim],   # temps relatifs à stim_end
                stim_start=stim_start,
                stim_end=stim_end,
                deadfile_elec=deadfile_elec,
                bin_size=0.1,          # 100 ms ; mets 0.05 si tu veux tester 50 ms
                window=10,
                min_valid_bin_frac=0.8)
            wilcoxon_pvals.append(pval)
            wilcoxon_nbins.append(n_bins_used)
            if np.isnan(pval):
                label = "NA"
            elif pval < 0.001:
                label = "***"
            elif pval < 0.01:
                label = "**"
            elif pval < 0.05:
                label = "*"
            else:
                label = ""#"ns"
            wilcoxon_labels.append(label)
        # modif couleur des ylabels selon résultat stat
        yticklabels = ax1.get_yticklabels()
        for i, label in enumerate(yticklabels):
            pval = wilcoxon_pvals[i]
            if not np.isnan(pval) and pval < 0.05:
                label.set_color('black')
            else:
                label.set_color('gray')

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
        # couleur de variation variable selon wilcoxon significatif ou pas (orange foncé ou clair)
        colors_var_wilcoxon = []
        for pval in wilcoxon_pvals:
            if np.isnan(pval):
                colors_var_wilcoxon.append('lightgray')  # option pour NA
            elif pval < 0.05:
                colors_var_wilcoxon.append('orange')     # jaune foncé (actuel)
            else:
                colors_var_wilcoxon.append("#E4CEA3")    # jaune clair

        ypos = np.arange(stims.shape[0])
        ax2.barh(ypos, firing_rate_var, color=colors_var_wilcoxon)
        ax2.axvline(0, linewidth=1, color="black")
        ax2.set_ylim(-1, stims.shape[0])
        ax2.set_yticks(ypos)
        ax2.set_yticklabels([' ' for _ in stim_names])

        ax2.set_xlabel("Variation of firing rate:\nafter-before EBS (in %)", fontsize=12)
        # affichage du Wilcoxon 
        trans = blended_transform_factory(ax1.transAxes, ax1.transData) # a mettre en argument du text a ajouter pour que les coordonnées soient dans le referentiel de ax1 plutot
        for i, (lab, pval, nb) in enumerate(zip(wilcoxon_labels, wilcoxon_pvals, wilcoxon_nbins)):
            txt = "NA" if np.isnan(pval) else lab
            ax2.text(
                1.02, ypos[i]-0.2, txt,
                transform=trans,
                va='center',
                ha='left',
                fontsize=9,
                clip_on=False)

        # 3. nombre de spikes par unité de temps
        nb_time_bins=100
        ax4.hist(np.sort([item for sublist in spike_times_before for item in sublist]), nb_time_bins, color='green')
        ax4.axvline(0, color="blue", linestyle="-", linewidth=1, label="Stimulation")
        ax4.hist(np.sort([item for sublist in spike_times_after for item in sublist]), nb_time_bins, color='red')
        ax4.set_xlim([-10, 10])
        ax4.set_xticks(x)
        ax4.set_xlabel("Time before the start (s)                              Time after the end (s)", fontsize=12)
        ax4.set_ylabel("Number of\nspikes per\ntime unit", fontsize=12)

        plt.subplots_adjust(hspace=0.05, wspace=0.1)        
        patch_title = '' if display_patches else '_noDead'
        plt.savefig(path_rasters + "Raster - all stims for unit "+str(ind_neuron)+" from "+dict_clu2tt[ind_neuron]+patch_title+".png", dpi=300)
        plt.show()


def rasters_OneStim_allNeurons(patient, session, spikes, stims, dict_clu2tt, dict_elec2deadfile, path_rasters, display_patches=True, plafond_inhib_100=True):
    from matplotlib.transforms import blended_transform_factory

    # D'abord determiner si certaines stims avec mm caracs/mm label exactement (pour export des figures)
    list_suffix_stim_rep = ['' for _ in range(stims.shape[0])]
    params_list = stims['paramètres'].tolist()

    for param in set(params_list):
        list_ind_stim_repeated = [i for i, x in enumerate(params_list) if x == param]
        if len(list_ind_stim_repeated) > 1:
            for rank, ind_stim_rep in enumerate(list_ind_stim_repeated, start=1):
                list_suffix_stim_rep[ind_stim_rep] = f'_{rank}'

    for ind_stim in range(stims.shape[0]):
        stim_start = stims['t'][ind_stim]
        stim_end = stims['t + durée'][ind_stim]

        epoch_test = nap.IntervalSet(
            start=[stim_start-10, stim_end],
            end=[stim_start, stim_end+10],
            time_units="s"
        )

        spikes_restricted = spikes.restrict(epoch_test)

        # liste des spikes pour chaque neurone
        spike_times = [spikes_restricted[nrn].index for nrn in dict_clu2tt.keys()]
        spike_times_before, spike_times_after = [], []

        for spike_times_i in spike_times:
            # temps relatifs: pre vs stim_start, post vs stim_end
            spike_times_before.append([t - stim_start for t in spike_times_i if t <= stim_start])
            spike_times_after.append([t - stim_end for t in spike_times_i if t > stim_start])

        # affichage general
        fig = plt.figure(figsize=(15, max(4, int(len(spikes)/4))))
        gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1])
        plt.style.use('ggplot')

        stim_title = stims['electrode'][ind_stim] + ' ' + stims['plots'][ind_stim] + ', at ' + stims['frequence'][ind_stim] + ' and ' + stims['intensite'][ind_stim]
        fig.suptitle(patient + ", stimic" + session + ": all neurons when stimulating in " + stim_title, fontsize=14, y=0.95)

        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax4 = plt.subplot(gs[1, 0])

        # 1. raster plot for the specific stimulation
        ax1.eventplot(spike_times_before, colors="green", linelengths=0.5)
        ax1.axvline(0, color="blue", linestyle="-", linewidth=1, label="Stimulation end-aligned comparison")
        ax1.eventplot(spike_times_after, colors="red", linelengths=0.5)
        ax1.set_xlim([-10, 10])

        x = np.arange(-10, 11, 1)
        ax1.set_xticks(x, np.repeat(' ', len(x)))
        ax1.set_ylim(-2, len(spikes.index) + 1)
        ax1.set_yticks(range(0, len(spikes.index)))
        ax1.set_yticklabels(list(dict_clu2tt.values()))
        ax1.set_ylabel("Neuron per tetrode")

        # 1.bis. dead periods + Wilcoxon
        durationOut_preStim = np.zeros(len(spikes.index))
        durationOut_postStim = np.zeros(len(spikes.index))
        wilcoxon_pvals, wilcoxon_labels, wilcoxon_nbins = [], [], []

        for ind_ligne_raster, tt in enumerate(dict_clu2tt.values()):
            deadfile_elec = dict_elec2deadfile[tt[:-1]]

            artefacts_filtered_pre = deadfile_elec[(deadfile_elec[1] >= stim_start-10) & (deadfile_elec[0] <= stim_start)]
            artefacts_filtered_post = deadfile_elec[(deadfile_elec[1] >= stim_end) & (deadfile_elec[0] <= stim_end+10)]

            durationOut_preStim[ind_ligne_raster] = np.sum(
                np.minimum(artefacts_filtered_pre[1], stim_start) - np.maximum(artefacts_filtered_pre[0], stim_start-10))
            durationOut_postStim[ind_ligne_raster] = np.sum(
                np.minimum(artefacts_filtered_post[1], stim_end+10) - np.maximum(artefacts_filtered_post[0], stim_end))

            plot_artefact_patches(
                ax1, deadfile_elec, stim_start, ind_ligne_raster,
                durationOut_preStim, side='pre', time_mode='relative',
                display=display_patches)
            plot_artefact_patches(
                ax1, deadfile_elec, stim_start, ind_ligne_raster,
                durationOut_postStim, side='post', time_mode='relative',
                stim_duration=stims['durée'][ind_stim], display=display_patches)

            # Wilcoxon par neurone
            stat, pval, n_bins_used, pre_rates, post_rates = _compute_binned_rates_and_wilcoxon(
                pre_spikes_rel=spike_times_before[ind_ligne_raster],
                post_spikes_rel=spike_times_after[ind_ligne_raster],
                stim_start=stim_start,
                stim_end=stim_end,
                deadfile_elec=deadfile_elec,
                bin_size=0.1,
                window=10,
                min_valid_bin_frac=0.8)

            wilcoxon_pvals.append(pval)
            wilcoxon_nbins.append(n_bins_used)

            if np.isnan(pval):
                label = "NA"
            elif pval < 0.001:
                label = "***"
            elif pval < 0.01:
                label = "**"
            elif pval < 0.05:
                label = "*"
            else:
                label = ""

            wilcoxon_labels.append(label)
        # couleur noms de tetrode sur ax1
        yticklabels = ax1.get_yticklabels()
        for i, label in enumerate(yticklabels):
            pval = wilcoxon_pvals[i]
            if not np.isnan(pval) and pval < 0.05:
                label.set_color('black')
            else:
                label.set_color('gray')

        # 2. variation de fréquence de décharge
        if plafond_inhib_100:
            firing_rate_var = [
                100 * (
                    (len(spike_times_after[i]) / (10 - durationOut_postStim[i])) -
                    (len(spike_times_before[i]) / (10 - durationOut_preStim[i]))
                ) / (len(spike_times_before[i]) / (10 - durationOut_preStim[i]))
                if len(spike_times_before[i]) > 0 else np.nan
                for i in range(len(spikes.index))
            ]
        else:
            firing_rate_var = [np.nan for _ in range(len(spikes.index))]
            for i in range(len(spikes.index)):
                if len(spike_times_before[i]) > 0:
                    pre_rate = len(spike_times_before[i]) / (10 - durationOut_preStim[i])
                    post_rate = len(spike_times_after[i]) / (10 - durationOut_postStim[i])

                    if post_rate < pre_rate:
                        if len(spike_times_after[i]) > 0:
                            firing_rate_var[i] = -100 * (pre_rate - post_rate) / post_rate
                        else:
                            firing_rate_var[i] = np.nan
                    else:
                        firing_rate_var[i] = 100 * (post_rate - pre_rate) / pre_rate
                else:
                    firing_rate_var[i] = np.nan

        # couleur selon significativité du Wilcoxon
        colors_var_wilcoxon = []
        for pval in wilcoxon_pvals:
            if np.isnan(pval):
                colors_var_wilcoxon.append('lightgray')
            elif pval < 0.05:
                colors_var_wilcoxon.append('orange')
            else:
                colors_var_wilcoxon.append("#E4CEA3")

        ypos = np.arange(len(spikes.index))
        ax2.barh(ypos, firing_rate_var, color=colors_var_wilcoxon)
        ax2.axvline(0, linewidth=1, color="black")
        ax2.set_ylim(-2, len(spikes.index) + 1)
        ax2.set_yticks(ypos)
        ax2.set_yticklabels(np.repeat('', len(spikes.index)))
        ax2.set_xlabel("Variation of firing rate:\nafter-before EBS (in %)", fontsize=10)

        # affichage du Wilcoxon entre ax1 et ax2
        trans = blended_transform_factory(ax1.transAxes, ax1.transData)
        for i, (lab, pval, nb) in enumerate(zip(wilcoxon_labels, wilcoxon_pvals, wilcoxon_nbins)):
            txt = "NA" if np.isnan(pval) else lab
            ax2.text(
                1.02, ypos[i] - 0.2, txt,
                transform=trans,
                va='center',
                ha='left',
                fontsize=9,
                clip_on=False
            )

        # 3. nombre de spikes par unité de temps
        nb_time_bins = 100
        ax4.hist(np.sort([item for sublist in spike_times_before for item in sublist]), nb_time_bins, color='green')
        ax4.axvline(0, color="blue", linestyle="-", linewidth=1, label="Stimulation")
        ax4.hist(np.sort([item for sublist in spike_times_after for item in sublist]), nb_time_bins, color='red')
        ax4.set_xlim([-10, 10])
        ax4.set_xticks(x)
        ax4.set_xlabel("Time before the start (s)                              Time after the end (s)", fontsize=10)
        ax4.set_ylabel("Number of spikes\nper time unit", fontsize=10)

        plt.subplots_adjust(hspace=0.05, wspace=0.1)

        patch_title = '' if display_patches else '_noDead'
        suffix_rep = list_suffix_stim_rep[ind_stim] # export : ajout suffixe si plusieurs stims identiques 
        title_fig = (
            path_rasters + "Raster - all units for stim " +
            stims['electrode'][ind_stim] + ' ' + stims['plots'][ind_stim] +
            ', ' + stims['frequence'][ind_stim] + ', ' + stims['intensite'][ind_stim] +
            suffix_rep + patch_title + ".png")
        plt.savefig(title_fig, dpi=300)
        plt.show()

def create_or_update_rasters(patient, session, overwrite_rasters=True, root='D:/'):
    '''
    Cree tous les rasters pour une session
    overwrite_rasters = False si les rasters ont déjà été faits
    disp = True pour afficher les zones artéfactées
    '''
    print(f"=== Create or update raster plots for {patient}, session {session}. ===")
    path_folder = root + 'Spike-sorting/Data_folders/'+patient+'/'+patient+'_stim'+session+'/'
    rasters_folder_artefacts = root + 'Spike-sorting/Rasters'+'/'+patient+'_stim'+session+'/'
    rasters_folder_NoArtefacts = root + 'Spike-sorting/Rasters_noPatch'+'/'+patient+'_stim'+session+'/'
    disp_patch = [True, False] # affichage patches artefacts pour rasters_folder_artefacts et rasters_folder_NoArtefacts

    sr = get_SR(patient)
    spikes = get_nwb(patient, session, root)

    stims_loca = get_stims(patient, session, root)
    mapping_anat = pd.read_csv(root + 'Spike-sorting/Data_folders/'+patient+'/mapping_anat_'+patient+'.txt', sep=',', engine='python')

    dict_elec2deadfile = get_dict_deadfiles(mapping_anat, patient, session, path_folder, sr)
    dict_clu2tt = get_dict_tetrodeName_from_tetrodeIndex(spikes, mapping_anat)
    
    for ind, rasters_folder in enumerate([rasters_folder_artefacts, rasters_folder_NoArtefacts]):
        if not os.path.exists(rasters_folder): # si le dossier de Rasters n'existe pas encore, alors on le crée
            os.makedirs(rasters_folder)
        folder_is_empty = (len(os.listdir(rasters_folder)) == 0) # True si dossier Rasters vide, False sinon (pour savoir si on efface des fichiers ou pas)
        if folder_is_empty:
            rasters_OneNeuron_allStims(patient, session, spikes, stims_loca, dict_clu2tt, dict_elec2deadfile, path_rasters=rasters_folder, display_patches = disp_patch[ind], plafond_inhib_100 = True)
            rasters_OneStim_allNeurons(patient, session, spikes, stims_loca, dict_clu2tt, dict_elec2deadfile, path_rasters=rasters_folder, display_patches = disp_patch[ind], plafond_inhib_100 = True)
        else: # folder deja rempli
            if overwrite_rasters: # il y a bien des fichiers, on les supprime d'abord 
                for f in os.listdir(rasters_folder):
                    Path(rasters_folder + '/' + f).unlink()
                rasters_OneNeuron_allStims(patient, session, spikes, stims_loca, dict_clu2tt, dict_elec2deadfile, path_rasters=rasters_folder, display_patches = disp_patch[ind], plafond_inhib_100 = True)
                rasters_OneStim_allNeurons(patient, session, spikes, stims_loca, dict_clu2tt, dict_elec2deadfile, path_rasters=rasters_folder, display_patches = disp_patch[ind], plafond_inhib_100 = True)
            else:
                print('Raster plots already exist for this session, and overwrite_rasters is False')

##################################################
# Cell-type labelling
##################################################

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import numpy as np

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

def add_celltypes_to_summary_tables(root):
    paths = get_celltype_cache_paths(root)

    labels_path = paths["cache_root"] / "features" / "celltype_labels_all_sessions.parquet"
    labels = pd.read_parquet(labels_path)

    tables_dir = Path(root) / "Spike-sorting" / "Tables"

    label_cols = [
        "patient",
        "session",
        "clu",
        "putative_cell_type",
        "gmm_cluster",
        "gmm_confidence",
        "quality_ok_celltype",
        "trough_to_peak_ms",
        "peak_half_width_ms",
        "trough_half_width_ms",
        "waveform_ptp_uV",
        "snr",
        "num_spikes",
        "presence_ratio",
        "isi_violations_ratio",
        "cell_type_classifier",
        "cell_type_interpretation",
    ]

    label_cols = [c for c in label_cols if c in labels.columns]
    labels_small = labels[label_cols].copy()

    summary_path = tables_dir / "summary_by_nrn_all_sessions.xlsx"
    general_path = tables_dir / "general_summary_all_sessions.xlsx"

    if summary_path.exists():
        summary = pd.read_excel(summary_path)
        summary2 = summary.merge(labels_small, on=["patient", "session", "clu"], how="left")
        summary2.to_excel(tables_dir / "summary_by_nrn_all_sessions_with_celltypes.xlsx", index=False)

    if general_path.exists():
        general = pd.read_excel(general_path)
        general2 = general.merge(labels_small, on=["patient", "session", "clu"], how="left")
        general2.to_excel(tables_dir / "general_summary_all_sessions_with_celltypes.xlsx", index=False)


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


def merge_celltypes_into_neuronal_df(
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
    import numpy as np
    import pandas as pd

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
            "amplitude_median",
            "presence_ratio",
            "amplitude_cutoff",
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
    import pandas as pd
    from pathlib import Path

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

def merge_global_neuronal_summaries_with_celltypes(
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
    import pandas as pd
    from pathlib import Path

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