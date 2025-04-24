import MDAnalysis
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from .FastAtomsClasses import Frame

def trajectory_analysis(traj_file_path, do_hydrogen_analysis=True, save=False, step_min=0, Nstep=1, stepsList=None, analyze_whole_trajectory=True, do_molecular_analysis=True, **kwargs):
    """
    Analyzes every frame of the provided trajectory file.
    Builds (and optionally saves) a list of analyzed Frame objects.

    Args :
        traj_file_path (string) : path of the trajectory file.

    Kwargs :
        do_hydrogen_analysis (bool) : if True, builds a list of hydrogen identification dataframes.
        save (bool) : if True, saves both the list of Frame objects and pandas dfs.
    """
    if stepsList is None:
        stepsList = np.arange(step_min, step_min+Nstep)
    u = MDAnalysis.Universe(traj_file_path)
    if analyze_whole_trajectory:
        stepsList = slice(None)
    h_identification_DFs_list = []
    frames_list = []
    
    for i, ts in enumerate(u.trajectory[stepsList]):
        currentFrame = Frame(positions=u.atoms.positions, labels=u.atoms.types, **kwargs)
        currentFrame.analyze()
        if do_hydrogen_analysis:
            h_id = currentFrame.build_hydrogen_identification_array()
            h_identification_DFs_list.append(pd.DataFrame(h_id))
        frames_list.append(currentFrame)

    if save:
        #with open(f'{os.path.basename(traj_file_path)}_frame_objects.pkl', 'wb') as f:
         #   pickle.dump(frames_list, f)
         # Apparently having to pickle that makes the program crash.
        if do_hydrogen_analysis:
            with open(f'{os.path.basename(traj_file_path)}_h_ids_dfs.pkl', 'wb') as f:
                pickle.dump(h_identification_DFs_list, f)
    if do_molecular_analysis:
        timeHistoDF = molecular_analysis(frames_list)
        timeHistoDF.to_csv(f'{os.path.basename(traj_file_path)}_time_molecule_histogram.csv', sep=' ', index=False) 
    return h_identification_DFs_list, frames_list

def molecular_analysis(frames_list, stepsList=None):
    # if first arg is a path : load pickled frames_list object
    if isinstance(frames_list, str):
        with open(f'{os.path(frames_list)}_frame_objects.pkl', 'rb') as f:
            frames_list = pickle.load(frames_list, f)
    # initialize stepsList if not given
    if stepsList is None:
        stepsList = np.arange(len(frames_list))
    histoList = []
    moleculesList = []
    timeHisto = {'Step' : stepsList}
    for currentFrame in frames_list:
        currentHisto = currentFrame.molecule_histogram()
        histoList.append(currentHisto)
        for mol in [*currentHisto]:
            if mol not in moleculesList:
                moleculesList.append(mol)               
    for mol in moleculesList:
        timeHisto[mol] = []
    for histogram in histoList:
        for mol in moleculesList:
            if mol not in [*histogram]:
                histogram[mol]=0
            timeHisto[mol] += [histogram[mol]]
    timeHistoDF = pd.DataFrame.from_dict(timeHisto,  dtype='int')
    return timeHistoDF
        

def free_energy_proton_hopping(h_id_list, mol1, mol2, mol3, mol4, T = 3000, 
                               num_bins=100, bins_range = (-1.5, 1.5), plot=False, label='', save=True):
    """
    Builds the free energy profile for a given proton transfer reaction.
    The reaction coordinate is of the form d[mol1-H] - d [mol2-H].
    Optionally plots the profile (if plot == True).
    
    Arguments : 
        h_id_list (list of pandas DataFrames) : list of H identification dfs built using the Frame class.
        mol1, mol2, mol3, mol4 (strings) : molecules involved in the reaction.
        mol1 and mol2 are the reagents ; mol3 and mol4 are the products.
        mol3/mol1 and mol2/mol4 should be conjugate acid/base pairs
        (i.e. mol3/mol1 = AH/A- & mol2/mol4 = BH/B- for instance).

    Kwargs :
        T (scalar) : temperature in K.
        bins_range (tuple of size 2) : range over which the RC histogram is computed (in A).
        plot (bool) : if True, will plot the free energy profile.
        label (string) : label for the plot.
        save (bool) : if True will save the bins and free energy profile into a two-column txt file.

    Returns : 
        free energy profile in kJ/mol (np array), bin centers in A (np array)

    Example :
        free_energy_proton_hopping(h_id_list, 'CH4', 'H3O', 'CH5', 'H2O')
        This will output the free energy profile and bins for the reaction :
        CH4 + H3O^+ -> CH5^+ + H2O ; reaction coordinate is d[C-H] - d[O-H].
    """
    # nested list of RC values
    RC_list = []
    for h_id in h_id_list:
        RC_list.append((h_id['d1'] - h_id['d2'])[(h_id['mol1']==mol2) & (h_id['mol2']==mol1)].to_numpy())
        RC_list.append((h_id['d2'] - h_id['d1'])[(h_id['mol1']==mol3) & (h_id['mol2']==mol4)].to_numpy())

    # Histogram of the flattened collection of all RC values.
    counts, bins = np.histogram([rc for ds in RC_list for rc in ds ], bins=num_bins, range=bins_range)
    
    kT = 8.314463*T/1000  # kT in kJ/mol
    bin_centers = (bins[1:] + bins[:-1])/2
    free_energy = -kT*np.log(counts)
    if plot:
        plt.plot(bin_centers, free_energy, label=label)       
        plt.grid()
        if label:
            plt.legend()
        plt.show()
    if save:
        np.savetxt(f'free_energy_{mol1}+{mol2}.txt', np.vstack((bin_centers, free_energy)).T, header='RC[A] F[kJ/mol]')
    return free_energy, bin_centers
