import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
from scipy.spatial import cKDTree
from itertools import combinations_with_replacement
from collections import Counter
from rdkit import Chem

class Hydrogen():
    """
    A class for Hydrogen atoms.
    """
    
    def __init__(self, position, index, isCoordinated=None, neighAtomType = None, neighAtomIndx = None, neighAtomDist = None, neighAtomVect=None):
        self.position = position
        self.index = index
        self.type_='H'
        self.molecule=None
        self.moleculeFormula = None
        if isCoordinated is None:
            self.isCoordinated = False
        else:
            self.isCoordinated = isCoordinated
        if neighAtomType is None:
            self.neighAtomType = [None]*2
        else:
            self.neighAtomType = neighAtomType
        if neighAtomIndx is None:
            self.neighAtomIndx = [None]*2
        else:
            self.neighAtomIndx = neighAtomIndx
        if neighAtomDist is None:
            self.neighAtomDist = [float('inf')]*2
        else:
            self.neighAtomDist = neighAtomDist
        if neighAtomVect is None:
            self.neighAtomVect = [None]*2
        else:
            self.neighAtomVect = neighAtomVect

        
    def coordinate(self):
        self.isCoordinated = True

    def attribute_first_neighbor(self, type_, index, distance, vector):
        self.neighAtomType[0] = type_
        self.neighAtomIndx[0] = index
        self.neighAtomDist[0] = distance
        self.neighAtomVect[0] = vector
        self.coordinate()
        
    def attribute_second_neighbor(self, type_, index, distance, vector):
        self.neighAtomType[1] = type_
        self.neighAtomIndx[1] = index
        self.neighAtomDist[1] = distance
        self.neighAtomVect[1] = vector

class HeavyAtom():
    """
    A class for all atoms that are not hydrogens.
    """ 
    def __init__(self, position, type_, index, neighAtomType = None, neighAtomIndx = None, neighAtomDist = None, molecule=None):
        self.position = position
        self.type_ = type_
        self.index = index
        self.moleculeFormula = None
        
        if neighAtomType is None:
            self.neighAtomType = []
        else:
            self.neighAtomType = neighAtomType
        if neighAtomIndx is None:
            self.neighAtomIndx = []
        else:
            self.neighAtomIndx = neighAtomIndx
        if neighAtomDist is None:
            self.neighAtomDist = []
        else:
            self.neighAtomDist = neighAtomDist
        if molecule is None:
            self.molecule = None
        else:
            self.molecule = molecule
            
    def add_neighbor(self, type_, index, distance):
        self.neighAtomType.append(type_)
        self.neighAtomIndx.append(index)
        self.neighAtomDist.append(distance)

class Frame():
    def __init__(self, filename=None, box_size=None, types_list = None, N_atom=0, 
                 labels=None, positions=None, hydrogenList=None, 
                 heavyList=None, allAtomsList=None, N_H=0, N_Heavy=0, heavyTrees = None, 
                 H_cutoffs={'C':1.5, 'O':1.3, 'H':1.2}, 
                 heavy_cutoffs={'C': {'C' : 1.95, 'O' : 1.82}, 'O': {'C':1.82, 'O':1.52 }}):
        if filename is None:
            self.filename = None
        else:
            self.filename = filename
        if box_size is None:
            self.box_size = 12.885920493
        else:
            self.box_size = box_size
        if types_list is None:
            self.types_list = []
        else :
            self.types_list = types_list
        self.N_atom = N_atom
        if labels is None:
            self.labels = []
        else:
            self.labels = labels
        if positions is None:
            self.positions = []
        else:
            self.positions = positions
        if hydrogenList is None:
            self.hydrogenList = []
        else:
            self.hydrogenList = hydrogenList
        if heavyList is None:
            self.heavyList = []
        else:
            self.heavyList = heavyList
        if allAtomsList is None:
            self.allAtomsList = []
        else:
            self.allAtomsList = allAtomsList
        if heavyTrees is None:
            self.heavyTrees = {}
        else:
            self.heavyTrees = heavyTrees
        self.N_H = N_H
        self.N_Heavy = N_Heavy
        self.H_cutoffs = H_cutoffs
        self.heavy_cutoffs = heavy_cutoffs
        self.analyzed = False

    def __len__(self):
        """Redéfinition de len."""
        return self.N_atom

    def __getitem__(self, key):
        """Est appelé quand on fait objet[index] ou objet[key]."""
        return self.allAtomsList[key]

    def __iter__(self):
        """Est appelé quand on fait un iter(objet)."""
        return iter(self.allAtomsList)

    def __reversed__(self):
        """Est appelé quand on fait reversed(objt)."""
        return reversed(self.allAtomsList)

    def __contains__(self, item):
        """Est appelé quand "in objet"."""
        return item in self.allAtomsList
        
    
    def read_frame(self):   # this function only reads files composed of single frames
        if isinstance(self.filename, str):
            with open(self.filename) as traj:
                self.N_atom = [int(s) for s in traj.readline().split() if s.isdigit()][0]
                frame = traj.readlines()[1:]
        else:
            frame = self.filename
        return frame

    
    def initialize_labels_and_positions(self):
        """
        Initializes positions, labels, N_atom, types_list, heavy_types.
        Wraps positions if needed (ie if they are not all in [0, box_size]
        as prescribed by scipy.spatial.KDTree)
        """
        if len(self.positions) == 0 and len(self.labels) == 0:
            f = self.read_frame()
            frame_arr0 = np.genfromtxt(f, dtype=None, encoding=None)
            self.labels = [line[0] for line in frame_arr0]
            frame_arr1 = np.genfromtxt(f)
            self.positions = frame_arr1[:,1:4]
            
        self.N_atom = len(self.positions)
        self.positions = np.array(self.positions)
        self.labels = np.array(self.labels)
        self.types_list = np.sort(np.unique(self.labels))
        self.heavy_types = np.array([typ for typ in self.types_list if typ != 'H'])
        
        if not len(np.where( (self.positions<0) | (self.positions>=self.box_size) )[0]) == 0:
            # wrap positions if they need to be wrapped
            self.positions = wrap(self.positions, self.box_size)

    def initialize_objects_lists(self):
        """
        self.hydrogenList = [ Hydrogen(position=self.positions[k], index=k) 
                             for k in range(self.N_atom) if self.labels[k]=='H' ]
        self.heavyList = [ HeavyAtom(position=self.positions[k], type_=self.labels[k], index=k)  
                          for k in range(self.N_atom) if self.labels[k]!='H' ]
        self.allAtomsList = [ self.heavyList[k] if self.labels[k]!='H' else self.hydrogenList[k] 
                             for k in range(self.N_atom) ]
        """
        for k in range(self.N_atom):
            if self.labels[k]=='H':
                H=Hydrogen(position=self.positions[k], index=k)
                self.allAtomsList.append(H)
                self.hydrogenList.append(H)
            else:
                X=HeavyAtom(position=self.positions[k], type_=self.labels[k], index=k)
                self.allAtomsList.append(X)
                self.heavyList.append(X)
        self.N_H = len(self.hydrogenList)
        self.N_Heavy = len(self.heavyList)

    def hydrogenNeighborAnalysis_with_KDTree(self):
        """
        Perform neighbor search with type_-specific cutoffs for hydrogen-heavy atom interactions.
        
        Args:
            hydrogen_positions (ndarray): Positions of hydrogen atoms (N_H x 3).
            heavy_positions (ndarray): Positions of heavy atoms (N_Heavy x 3).
            heavy_types (ndarray): Types of heavy atoms (N_Heavy,).
            box_size (float): Size of the periodic simulation box.
            H_cutoffs (dict): Dictionary of type_-specific cutoff distances.
    
        Returns:
            neighbors (list of list): Each hydrogen's neighbors, filtered by type_-specific cutoffs.
        """
        heavy_indices = self.labels != 'H'
        h_positions = self.positions[~heavy_indices]
        #heavy_positions = self.positions[self.labels!='H']
        # Get the unique types of heavy atoms
        heavy_labels = self.labels[ heavy_indices ]
        
        distances_to_neighbors = np.empty((len(self.heavy_types), self.N_H))
        neighbors_indexes = np.empty((len(self.heavy_types), self.N_H))
        #type_indexes = []
        # Build KDTree for each heavy atom type_
        for i, atom_type in enumerate(self.heavy_types):
            # Select positions corresponding to the current type_
            type_indexes = np.append( np.arange(self.N_atom)[self.labels==atom_type] , self.N_atom )
            type_positions = self.positions[self.labels == atom_type]
            self.heavyTrees[atom_type] = cKDTree(type_positions, boxsize=self.box_size)
            distances_to_neighbors[i], neighbors_indexes[i] = self.heavyTrees[atom_type].query(h_positions,k=1,distance_upper_bound=self.H_cutoffs[atom_type]
                                                                                              )
            neighbors_indexes[i] = type_indexes[neighbors_indexes[i].astype(int)]

        first_neighbor_index = neighbors_indexes[np.argmin(distances_to_neighbors, axis=0), np.arange(neighbors_indexes.shape[1])]
        first_neighbor_distance = np.min(distances_to_neighbors, axis=0)

        for k in range(self.N_H):
            if first_neighbor_index[k] < self.N_atom:
                neighbor = self.allAtomsList[first_neighbor_index[k].astype(int)] 
                H = self.hydrogenList[k]
                H.attribute_first_neighbor(neighbor.type_, neighbor.index, first_neighbor_distance[k], correct_vector(neighbor.position - H.position, self.box_size))
            # BEWARE for cKDTree, atomic cordinates should all be wrapped in [0,boxsize[
    
    def find_H2_with_KDTree(self):
        """ 
        To apply : after hydrogenNeighborAnalysis, once most hydrogens have been coordinate
        with stringent cutoffs beyond which H2 molecules never penetrate.
        
        Here we focus on uncoordinated (lonely) hydrogens. We build a KDTree for them and query
        it for lonely hydrogen pairs within (user-defined) H-H cutoff (H_cutoffs['H']).
        Once potential pairs have been identified, we compute corresponding H-H distances to 
        perform a filtering operation, coded in the function forbid_Hn_chains. We know Hn, n>=3 
        molecules do not exist. However, an hydrogen atom could be involved in more than one 
        potential_pair, which means it could form two H-H covalent bonds. To prevent this, we 
        apply forbid_Hn_chains which selects the pair with smallest distance for all hydrogen 
        indexes in the flattened potential_pairs array. This gives us the actual indexes of 
        hydrogens in H2 molecules. The final for loop updates the Hydrogen objects of H's in H2
        molecules by adding corresponding first neighbor info.
        
        After : H's involved in H2 molecules have been identified and their neighbors attributed.
        """
        # Building lonely hydrogen positions and indexes arrays (lonely == ~H.isCoordinated)
        zipped_pos_ind = list(zip(
            *[(H.position, H.index) for H in self.hydrogenList if not H.isCoordinated]
        ))
        if len(zipped_pos_ind)==0:
            return []   # if no lonely H's, no H2s.
        else:
            lonely_hydrogen_pos, lonely_hydrogen_indexes = zipped_pos_ind
        lonely_hydrogen_pos = np.array(lonely_hydrogen_pos)
        lonely_hydrogen_indexes = np.array(lonely_hydrogen_indexes)
        
        # Building lonely hydrogen KDTree and identifying pairs within cutoff.
        h_tree = cKDTree(lonely_hydrogen_pos, boxsize=self.box_size)
        potential_H2_pairs_indexes = h_tree.query_pairs(
            r=self.H_cutoffs['H'], output_type='ndarray') # size (number_of_pairs, 2)

        if not len(potential_H2_pairs_indexes)==0:
            # if there are no pairs, there's nothing to do. Otherwise :
            # Building H-H vector and distance arrays
            diff = correct_vector(lonely_hydrogen_pos[potential_H2_pairs_indexes][:, 0, :] - lonely_hydrogen_pos[potential_H2_pairs_indexes][:, 1, :], self.box_size)
            distances = np.linalg.norm(diff, axis=1)
            # Get the list of indexes in allAtomsList (usual atomic indexes) 
            # instead of indexes in lonely_hydrogen_pos.
            potential_H2_pairs_indexes = lonely_hydrogen_indexes[potential_H2_pairs_indexes]

            # Prevent Hn (n>=3) chains strictly forbidding an H to appear in two different pairs
            # (if an H appears in two different potential_pairs, only the one with smallest H-H
            # distance is kept in actual_pairs)
            actual_H2_pair_indexes, actual_distances, actual_vectors = forbid_H_n_chains(potential_H2_pairs_indexes, distances, diff)
             
            for pair_num, pair in enumerate(actual_H2_pair_indexes):
                H1, H2 = self.allAtomsList[pair[0]], self.allAtomsList[pair[1]]
                H1.attribute_first_neighbor('H', pair[1], actual_distances[pair_num], actual_vectors[pair_num])
                H2.attribute_first_neighbor('H', pair[0], actual_distances[pair_num], -actual_vectors[pair_num])
            
            return actual_H2_pair_indexes.flatten() # returns indexes of H's in H2 for attribute_remaining_H
        
        else:
            return [] # if there are no H2, we return an empty list.
    
    def attribute_remaining_hydrogens_KDTree(self, H_in_H2_indexes):
        """
        To apply : Once H2 molecules have been identified (follows find_H2)
        Here we finish hydrogen neighbor analysis :
        - if H is not yet coordinated (lonely) : a neighbor search among heavy atoms
          is performed to find the two closest heavies. They are attributed as 1st and 2d
          neighbor of H.
        - if H is already coordinated (coordinated) : we also perform a neighbor search,
          but only for the second neighbor (k=[2]) and attribute it.
        After this method : all hydrogens have first and second neighbors.
        """
        """ Interrogations about performance : 
         - should we perform two distinct neighbor searches or only one and filter hydrogens
           depending on their coordinence afterwards ? (probably option 2, contrary to current
           implementation...)
         - we should probably add a distance_upper_bound to speed up neighbor search(es).
           We must analyze a statistical ensemble of frames and determine the maximum distance
           of a second neighbor (actually this might be overkill for very little perf gain)
        """
        lonely_hydrogen_pos, lonely_hydrogen_indexes = [] , []
        for H in self.hydrogenList:
            if not H.isCoordinated:
                lonely_hydrogen_indexes.append(H.index)
                lonely_hydrogen_pos.append(H.position)
        
        lonely_hydrogen_pos = np.array(lonely_hydrogen_pos)
        lonely_hydrogen_indexes = np.array(lonely_hydrogen_indexes)
        hydrogen_pos = np.asarray([H.position for H in self.hydrogenList])

        heavy_indices = self.labels!='H'
        heavy_pos = self.positions[heavy_indices]
        heavy_indexes = np.nonzero(heavy_indices)[0]
        heavy_tree = cKDTree(heavy_pos, boxsize=self.box_size)

        if len(lonely_hydrogen_pos) != 0:
            lonely_distances_to_neighbors, lonely_neighbors_indexes = heavy_tree.query(lonely_hydrogen_pos, k=[1])
            lonely_neighbors_indexes = heavy_indexes[lonely_neighbors_indexes.astype(int)]
            # queried arrays have size (len(x=lonely_hydrogen_pos), k=1).
    
            for i, neighbor_index in enumerate(lonely_neighbors_indexes):
                neighbor_1 = self.allAtomsList[neighbor_index[0]]
                H = self.allAtomsList[lonely_hydrogen_indexes[i]]
                H.attribute_first_neighbor(neighbor_1.type_, neighbor_1.index, lonely_distances_to_neighbors[i][0], correct_vector(neighbor_1.position - H.position, self.box_size))

        if len(H_in_H2_indexes) > 0:
            heavy_indices[H_in_H2_indexes] = True
            heavy_pos = self.positions[heavy_indices]
            heavy_indexes = np.nonzero(heavy_indices)[0]
            heavy_tree = cKDTree(heavy_pos, boxsize=self.box_size)

        h_distances_to_neighbors, h_neighbors_indexes = heavy_tree.query(hydrogen_pos, k=[2])
        h_neighbors_indexes = heavy_indexes[h_neighbors_indexes.astype(int)]
        
        for i, neighbor_index in enumerate(h_neighbors_indexes):
            second_neighbor = self.allAtomsList[neighbor_index[0]]
            H = self.hydrogenList[i]
            H.attribute_second_neighbor(second_neighbor.type_, second_neighbor.index, h_distances_to_neighbors[i][0], correct_vector(second_neighbor.position - H.position, self.box_size))

    def coordinate_heavies_to_hydrogens(self):
        """ 
        To apply : Once all hydrogens have been coordinated (follows attribute_remaining_hydrogens).
        Here we reciprocally coordinate their heavy first neighbors. 
        After this method : all heavy atoms will have all their H neighbors.
        """
        for H in self.hydrogenList:
            if H.neighAtomType[0] != 'H':
                neighbor_of_H = self.allAtomsList[H.neighAtomIndx[0]]
                neighbor_of_H.add_neighbor('H', H.index, H.neighAtomDist[0])


    def heavyAtomsNeighborAnalysis_KDTree(self):
        """
        To apply : anytime after initialization
        Here we perform neighbor searches for heavy-heavy interactions, with 
        user-defined cutoffs, and update HeavyAtoms accordingly.
        After : all heavy atoms have their heavy neighbors. If Hydrogens have already
        been processed, molecules can be built afterwards.
        """
        type_indexes_dict = {type_ : index_list 
                             for type_, index_list in zip(
                                 self.heavy_types, [np.arange(self.N_atom)[self.labels==typ] 
                                               for typ in self.heavy_types])}
        for pair in combinations_with_replacement(self.heavy_types, 2):
            type1, type2 = pair
            if type1!=type2:
                indexes = self.heavyTrees[type1].query_ball_tree(
                    self.heavyTrees[type2], r=self.heavy_cutoffs[type1][type2])
                neighbor_pairs = np.array(
                    [[i, j] for i in range(len(indexes)) for j in indexes[i]])
            else:
                neighbor_pairs = self.heavyTrees[type1].query_pairs(
                    r=self.heavy_cutoffs[type1][type1], output_type='ndarray')
            if len(neighbor_pairs) != 0:
                
                neighbor_pairs[:,0] = type_indexes_dict[type1][neighbor_pairs[:,0]]
                neighbor_pairs[:,1] = type_indexes_dict[type2][neighbor_pairs[:,1]]
                    
                for neighbor_ind in neighbor_pairs:
                    ind1, ind2 = neighbor_ind
                    X1 = self.allAtomsList[ind1]
                    X2 = self.allAtomsList[ind2]
                    dist = np.linalg.norm(correct_vector(X2.position - X1.position, self.box_size))
                    X1.add_neighbor(type2, ind2, dist)
                    X2.add_neighbor(type1, ind1, dist)
                
    def build_molecule(self, start_index, seen=None,molecule=None):
        if seen is None:
            seen=set()
            seen.add(None)
        if molecule is None:
            molecule = nx.Graph()
        if start_index in seen:
            return molecule
        seen.add(start_index)
        X = self.allAtomsList[start_index]
        if X.type_ == 'H':
            if X.neighAtomIndx[0] not in seen:
                molecule.add_edge(start_index, X.neighAtomIndx[0])
                self.build_molecule(X.neighAtomIndx[0], seen, molecule)
            elif X.neighAtomIndx[0] is None:
                molecule.add_node(start_index)
                return molecule
        else:
            for neighbor_index in X.neighAtomIndx:
                if neighbor_index not in seen:
                    molecule.add_edge(start_index, neighbor_index)
                    self.build_molecule(neighbor_index, seen, molecule)
            if len(X.neighAtomIndx) == 0:
                molecule.add_node(start_index)
                return molecule
        return molecule

    def assign_molecules_to_atoms(self):
        for X in self.allAtomsList:
            if X.molecule is None:
                molecule = self.build_molecule(X.index)
                molecule_form = molecule_formula(molecule, self.labels, self.types_list)
                for index in molecule.nodes:
                    self.allAtomsList[index].molecule = molecule
                    self.allAtomsList[index].moleculeFormula = molecule_form
        
    def analyze(self):
        if not self.analyzed:
            self.initialize_labels_and_positions()
            self.initialize_objects_lists()
            self.hydrogenNeighborAnalysis_with_KDTree()
            H_in_H2_indexes = self.find_H2_with_KDTree()
            self.attribute_remaining_hydrogens_KDTree(H_in_H2_indexes)
            self.coordinate_heavies_to_hydrogens()
            self.heavyAtomsNeighborAnalysis_KDTree()
            self.assign_molecules_to_atoms()
        self.analyzed=True    
    
    def molecule_histogram(self):
        seen = set()
        hist = {}
        for X in self.allAtomsList:
            if X.molecule not in seen:
                seen.add(X.molecule)
                formula = molecule_formula(X.molecule, self.labels, self.types_list)
                if formula not in hist.keys():
                    hist[formula] = 1
                else:
                    hist[formula] += 1
        return hist

    def get_molecule_list(self):
        """
        Returns list of each atom's molecular networkx graph.
        """
        return [X.molecule for X in self.allAtomsList]

    def get_molecule_formulas(self):
        """
        Returns list of each atom's molecular formula (str).
        """
        return np.asarray([molecule_formula(mol, self.labels, self.types_list) for mol in self.get_molecule_list()])
    
    def build_hydrogen_identification_array(self):
        
        # Preallocate lists
        dict_ = {'index': [], 'mol1': [], 'mol2': [], 'atom1_type': [], 'atom1_index': [],
                 'atom2_type': [], 'atom2_index': [], 'd1': [], 'd2': [], 'cos(angle)': []}
        
        # Iterate through hydrogens
        for H in self.hydrogenList:
            i1, i2 = H.neighAtomIndx
            mol1 = H.moleculeFormula
            mol2 = self.allAtomsList[i2].moleculeFormula
    
            # Cache neighbor attributes
            v1, v2 = H.neighAtomVect
            d1, d2 = H.neighAtomDist
    
            # Fill dictionary
            dict_['index'].append(H.index)
            dict_['mol1'].append(mol1)
            dict_['mol2'].append(mol2)
            dict_['atom1_type'].append(H.neighAtomType[0])
            dict_['atom2_type'].append(H.neighAtomType[1])
            dict_['atom1_index'].append(i1)
            dict_['atom2_index'].append(i2)
            dict_['d1'].append(d1)
            dict_['d2'].append(d2)
            dict_['cos(angle)'].append(np.dot(v1, v2) / (d1 * d2))
        
        return dict_ 

@njit
def correct_vector(vector, box_size):
    """
    Shifts vector components to account for pbcs.
    Beware : input vectors must be have been built from positions 
    wrapped in [0, box_size]^3 !
    """
    vector = np.where(vector > box_size / 2, vector - box_size, vector)
    vector = np.where(vector < - box_size / 2, vector + box_size, vector)
    return vector

def wrap(positions, box_size):    # supposes the center of the CUBIC box is at (0.5) 
    return positions % box_size

def forbid_H_n_chains(pairs, distances, vectors):    # tested : it works.
    sorted_indices = np.argsort(distances)
    sorted_pairs = pairs[sorted_indices]

    used_indices = set()
    filtered_pairs = []
    unique_mask = []

    for i, pair in enumerate(sorted_pairs):
        if pair[0] not in used_indices and pair[1] not in used_indices:
            filtered_pairs.append(pair)
            unique_mask.append(True)
            used_indices.update(pair)
        else:
            unique_mask.append(False)

    return np.asarray(filtered_pairs), distances[sorted_indices][unique_mask], vectors[sorted_indices][unique_mask]

def formula_dict_to_string(dict_):   # returns the usual molecular formula as a string from a dictionary input
    string = ''                     # ex : {'C':3, 'O':0, 'H':9, 'N':1} --> 'C3H9N'
    for type_ in np.sort([*dict_]):
        if dict_[type_] > 0: 
            string += type_
            if dict_[type_]>1:
                string += f'{dict_[type_]}' 
    return string

def molecule_formula(graph, labels, types_list=None):
    """
    Builds the molecular formula string directly from a networkx graph.
    
    Args:
        graph (nx.Graph): The molecular graph.
        labels (list): A list of atom labels indexed by graph nodes.
        types_list (list): A list of all possible atom types (e.g., ['H', 'C', 'O', 'N']). Order sets formula order.
        
    Returns:
        str: Molecular formula as a string.
    """
    # Count atom types directly from node labels
    atom_counts = Counter(labels[index] for index in graph.nodes)

    # If order is not specified, default is alphabetic.
    if types_list is None:
        types_list = np.sort(np.unique(labels))
        
    # Build the formula string in the order of `types_list`
    return ''.join(
        f"{type_}{atom_counts[type_] if atom_counts[type_] > 1 else ''}" 
        for type_ in types_list if atom_counts.get(type_, 0) > 0
    )

def graph_to_rdkit(graph, labels):
    """
    Convert a NetworkX graph into an RDKit molecule, using `labels[node]`
    to get the atom symbol for each node.

    Parameters
    ----------
    graph : networkx.Graph
        Undirected graph whose nodes are integer IDs (not necessarily 0…n-1).
    labels : sequence or dict
        A mapping (e.g. list or dict) from node ID → atom symbol (string).

    Returns
    -------
    rdkit.Chem.Mol
        An RDKit molecule with atoms and single bonds as specified.
    """
    # 1) Create a mutable molecule
    rw_mol = Chem.RWMol()

    # 2) Map each graph node to its new atom index in rw_mol
    node_to_idx = {}
    for node in graph.nodes():
        # Retrieve atom symbol (e.g. 'C', 'O', 'N') for this node
        atom_symbol = labels[node]
        atom = Chem.Atom(atom_symbol)
        atom.SetNoImplicit(True)
        idx = rw_mol.AddAtom(atom)
        node_to_idx[node] = idx

    # 3) Add a single bond for every edge in the graph
    for u, v in graph.edges():
        idx_u = node_to_idx[u]
        idx_v = node_to_idx[v]
        # Use SINGLE bonds by default
        rw_mol.AddBond(idx_u, idx_v, Chem.BondType.SINGLE)

    # 4) Freeze to an immutable Mol and return
    return rw_mol.GetMol()
    