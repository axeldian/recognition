import MDAnalysis
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Optional, Union, Sequence
from pathlib import Path
from .FastAtomsClasses import Frame

@dataclass
class TrajectoryAnalysisResult:
    """
    A container for the results of the analysis (frames_list, h dfs list, mol histogram).
    Access a component by calling corresponding attribute (e.g., TrajectoryAnalysisResult.frames is the frames list).
    """
    frames: List[Frame]
    hydrogen_dfs: List[pd.DataFrame]
    time_histogram: Optional[pd.DataFrame] = None

def trajectory_analysis(
    traj_file_path: str,
    step_min: int = 0,
    step_max: Optional[int] = None,
    stride: int = 1,
    steps_list: Optional[Union[slice, Sequence[int], np.ndarray]] = None,
    do_molecular_analysis: bool = True,
    do_hydrogen_analysis: bool = True,
    save: bool = False,
    **kwargs
) -> TrajectoryAnalysisResult:
    """
    Analyzes every frame of the provided trajectory file.

    Args :
        traj_file_path (string) : path of the trajectory file.

    Kwargs :
        step_min (int) : initial step (default: 0)
        step_max (int) : final step (default: 1)
        stride (int) : step between every analyzed Frame.
        steps_list (array-like) : integer or boolean array specifying the list of frames to analyze.
        do_molecular_analysis (bool) : if True, builds and saves the molecular population histogram over the analyzed trajectory as a CSV.
        do_hydrogen_analysis (bool) : if True, builds a list of hydrogen identification dataframes.
        save (bool) : if True, saves pandas hydrogen_analysis list of dfs.


    Returns:
        TrajectoryAnalysisResult : a container for the Frame list, the H DFs list and the molecule histogram DF.
    """
    if steps_list is None: 
    # initialize the list of steps to analyze
        steps_list = slice(step_min, step_max, stride)
        steps_list_to_print = []
    elif isinstance(steps_list, (list, np.ndarray)):    
    # initialize list of steps to be printed in the molecule histgram
        steps_list_to_print = steps_list
    else:
        step_min = steps_list.start if steps_list.start is not None else 0
        stride = steps_list.step if steps_list.step is not None else 1
        steps_list_to_print = []
    
    # create an MDA Universe for the trajectory
    u = MDAnalysis.Universe(traj_file_path)
    # initialize list of Frames and H identification DFs
    frames_list = []
    h_identification_DFs_list = []
    
    # loop over trajectory, keeping only the steps in steps_list
    for i, ts in enumerate(tqdm(u.trajectory[steps_list], desc="Analyzing frames")):
        # create and analyze Frame object from positions and types arrays
        # kwargs typically include cutoffs
        currentFrame = Frame(positions=u.atoms.positions, labels=u.atoms.types, **kwargs)
        currentFrame.analyze()
        frames_list.append(currentFrame)
        if do_hydrogen_analysis:
            # build H identification DF
            h_id = currentFrame.build_hydrogen_identification_array()
            h_identification_DFs_list.append(pd.DataFrame(h_id))
    
    if do_molecular_analysis:
        # build and save per-step molecular histogram
        if len(steps_list_to_print)==0:
            step_max = step_min + (i+1) * stride
            steps_list_to_print = np.arange(step_min, step_max, stride)
        timeHistoDF = molecular_analysis(frames_list, steps_list=steps_list_to_print)
        timeHistoDF.to_csv(f'{os.path.basename(traj_file_path)}_time_molecule_histogram.csv', sep=',', index=False)
    
    if save and do_hydrogen_analysis:
        # save list of H identification DFs
        with open(f'{os.path.basename(traj_file_path)}_h_ids_dfs.pkl', 'wb') as f:
            pickle.dump(h_identification_DFs_list, f)

    return TrajectoryAnalysisResult(
        frames=frames_list,
        hydrogen_dfs=h_identification_DFs_list,
        time_histogram=timeHistoDF if do_molecular_analysis else None
    )

def molecular_analysis(frames_list, steps_list=None):
    # if first arg is a path : load pickled frames_list object
    if isinstance(frames_list, str):
        with open(f'{os.path(frames_list)}_frame_objects.pkl', 'rb') as f:
            frames_list = pickle.load(frames_list, f)
    # if steps_list not given, defaults to 0, 1, ..., len(frames_list)-1
    if steps_list is None:
        steps_list = np.arange(len(frames_list))
    histoList = []
    moleculesList = []
    timeHisto = {'Step' : steps_list}
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
