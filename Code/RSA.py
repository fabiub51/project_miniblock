from os.path import join
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image
import glob
from itertools import combinations
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from numpy.linalg import eigh, svd
from tqdm import tqdm
import os

def RSA_between(project_dir, spearman=True):
    """
    Function that calculates RSA between particpants for every ROI. 
    Set spearman to False to use Pearson's r instead. 
    Returns nothing but saves a dataframe of results to the RSA/ROI_between folder.
    """

    # Set up directories 
    outdir = join(project_dir, 'miniblock/Outputs/')
    datadir = join(project_dir, 'miniblock/')

    # Set up paris dictionary 
    # Since participants 9 and 16 had to be excluded, 21 and 22 are taking their place 
    pairs_dict = {"pair_1" : ["01", "02"], 
                "pair_2" : ["03", "04"], 
                "pair_3" : ["05", "06"], 
                "pair_4" : ["07", "08"], 
                "pair_5" : ["11", "12"], 
                "pair_6" : ["13", "14"], 
                "pair_7" : ["17", "18"],
                "pair 8" : ["19", "20"],
                "pair 9" : ["10", "21"],
                "pair 10": ["15", "22"]}
    smooths = ['sm_2_vox'] # only smoothed data
    presdir = join(project_dir, 'Behavior', 'designmats')
    runtypes = ['sus', 'miniblock', 'er']
    ROIs = ["EBA","FFA", "PPA", "EVC" ]

    correlation_results = [] # store results here
    for ROI in ROIs: 
        for pair_name, subjects in pairs_dict.items():
            #print(f"Now working on {pair_name} in ROI: {ROI}")
            for runtype in runtypes: 
                    for smoothing in smooths:
                        upper_triangles_flattened = [] # store flattened upper triangles 
                        # loop over both subjects
                        for sub in subjects: 
                            # Load GLMsingle Outputs
                            results_glmsingle = dict()
                            results_glmsingle['typed'] = np.load(join(outdir,"GLMSingle_Outputs",f'{smoothing}_sub-{sub}_{runtype}_TYPED_FITHRF_GLMDENOISE_RR.npy'), allow_pickle=True).item()
                            betas = results_glmsingle['typed']['betasmd']

                            # load the participant-specific ROI mask 
                            brain_mask_path = join(datadir, 'derivatives', f'sub-{sub}', 'anat', f'{ROI}_mask_sm_2_vox.nii')
                            brain_mask = image.load_img(brain_mask_path)
                            mask = brain_mask.get_fdata()   
                            # Mask betas 
                            masked_betas = betas[mask.astype(bool)]
                            # Get design matrix from GLMsingle
                            pattern = presdir + f'/P0{sub}_ConditionRich_Run*_{runtype}.csv'
                            matches = glob.glob(pattern)
                            matches.sort()
                            
                            design = []
                            for i in range(len(matches)):
                                designMat = pd.read_csv(matches[i], header=None)
                                design.append(designMat)

                            all_design = np.vstack((design[0], design[1], design[2]))
                            condition_mask = all_design.sum(axis=1) > 0
                            # a vector assigning conditions to trials
                            condition_vector = np.argmax(all_design[condition_mask], axis=1)
                            # Get the beta values into a vector 
                            n_conditions = 40
                            max_reps = 6

                            # create a matrix of shape trials by condition containing the indeces 
                            repindices = np.full((max_reps, n_conditions), np.nan)
                            for p in range(n_conditions):  
                                inds = np.where(condition_vector == p)[0]  
                                repindices[:len(inds), p] = inds  
                            
                            # prepare empty array to store betas per repitition per condition
                            X, T = masked_betas.shape
                            n_reps, n_conds = repindices.shape
                            betas_per_condition = np.full((X, n_reps, n_conds), np.nan)

                            # loop over conditions and fill the betas_per_condition matrix
                            for cond in range(n_conds):
                                trial_indices = repindices[:, cond]
                                for rep, trial_idx in enumerate(trial_indices):
                                    if not np.isnan(trial_idx):
                                        trial_idx = int(trial_idx)
                                        betas_per_condition[:, rep, cond] = masked_betas[:, trial_idx]

                            # result: number of voxels by repetitions by conditions
                            # calculate mean over all trials
                            all_betas_means = betas_per_condition.mean(axis=-2)
                            all_betas = all_betas_means.T #transpose 
                            upper_triangle = pdist(all_betas, metric='correlation') # get similarity matrix and extract upper triangle
                            upper_triangles_flattened.append(upper_triangle)
                        # Calculate correlation between the two subjects
                        if spearman: 
                            cor, p_value = spearmanr(upper_triangles_flattened[0], upper_triangles_flattened[1])
                        else: 
                            cor, p_value = pearsonr(upper_triangles_flattened[0], upper_triangles_flattened[1])
                        # append to results 
                        correlation_results.append({
                                    "pair": pair_name,
                                    "ROI": ROI,
                                    "runtype": runtype,
                                    "smoothing": smoothing,
                                    "correlation": cor})
    # store as csv file after doing all pairs
    if spearman:                        
        os.makedirs(join(outdir,"RSA/ROI_between"),exist_ok=True)
        pd.DataFrame(correlation_results).to_csv(join(outdir,"RSA/ROI_between","rsa_results_spearman_between.csv"), index=False)
    else: 
        pd.DataFrame(correlation_results).to_csv(join(outdir,"RSA/ROI_between","rsa_results_pearson_between.csv"), index=False)

def RSA_within(project_dir, subjects, spearman=True):
    """
    Function that calculates RSA within every particpant for every ROI. 
    Set spearman to False to use Pearson's r instead. 
    Returns nothing but saves a dataframe of results to the RSA/ROI_within folder.
    """
    # Set up directories 
    outdir = join(project_dir, 'miniblock/Outputs/')
    datadir = join(project_dir, 'miniblock/')
    smooths = ['sm_2_vox']
    presdir = join(project_dir, 'Behavior', 'designmats')
    runtypes = ['sus', 'miniblock', 'er']
    ROIs = ["EBA_mask", "FFA_mask", "PPA_mask", "EVC_mask"]

    # we want to calculate all splits of the 6 betas
    # this part creates all possible combinations when splitting in two groups
    elements = [0, 1, 2, 3, 4, 5]
    # Get all combinations of 3 elements
    group1_list = list(combinations(elements, 3))

    # To avoid duplicates (like (group1, group2) and (group2, group1)), only keep half
    splits = []
    seen = set()

    for group1 in group1_list:
        group2 = tuple(sorted(set(elements) - set(group1)))
        # Make sure we haven't already seen this partition
        key = tuple(sorted([group1, group2]))
        if key not in seen:
            seen.add(key)
            splits.append((group1, group2))

    correlation_results = [] # store results here
    for ROI in ROIs: 
        for sub in subjects:
            for split in splits:
                split_idx = splits.index(split) + 1
                for runtype in runtypes: 
                        for smoothing in smooths:
                            # Load GLMsingle Outputs
                            results_glmsingle = dict()
                            results_glmsingle['typed'] = np.load(join(outdir,"GLMSingle_Outputs",f'{smoothing}_sub-{sub}_{runtype}_TYPED_FITHRF_GLMDENOISE_RR.npy'), allow_pickle=True).item()
                            betas = results_glmsingle['typed']['betasmd']
                            # load the participant-specific ROI mask 
                            brain_mask_path = join(datadir, 'derivatives', f'sub-{sub}', 'anat', f'{ROI}_sm_2_vox.nii')
                            brain_mask = image.load_img(brain_mask_path)
                            mask = brain_mask.get_fdata()   
                            # Mask betas 
                            masked_betas = betas[mask.astype(bool)]
                            # Get design matrix from GLMsingle
                            pattern = presdir + f'/P0{sub}_ConditionRich_Run*_{runtype}.csv'
                            matches = glob.glob(pattern)
                            matches.sort()
                            
                            design = []
                            for i in range(len(matches)):
                                designMat = pd.read_csv(matches[i], header=None)
                                design.append(designMat)

                            all_design = np.vstack((design[0], design[1], design[2]))
                            condition_mask = all_design.sum(axis=1) > 0
                            # a vector assigning conditions to trials
                            condition_vector = np.argmax(all_design[condition_mask], axis=1)
                            # Get the beta values into a vector 
                            n_conditions = 40
                            max_reps = 6

                            # create a matrix of shape trials by condition containing the indices 
                            repindices = np.full((max_reps, n_conditions), np.nan)
                            for p in range(n_conditions):  
                                inds = np.where(condition_vector == p)[0]  
                                repindices[:len(inds), p] = inds  
                            
                            # prepare empty array to store betas per repitition per condition
                            X, T = masked_betas.shape
                            n_reps, n_conds = repindices.shape
                            betas_per_condition = np.full((X, n_reps, n_conds), np.nan)

                            # loop over conditions and fill the betas_per_condition matrix
                            for cond in range(n_conds):
                                trial_indices = repindices[:, cond]
                                for rep, trial_idx in enumerate(trial_indices):
                                    if not np.isnan(trial_idx):
                                        trial_idx = int(trial_idx)
                                        betas_per_condition[:, rep, cond] = masked_betas[:, trial_idx]
                            # result: number of voxels by repetitions by conditions
                            
                            # for each split, calculate the mean over the 3 respective betas and transpose
                            first_split = betas_per_condition[:,list(split[0]),:].mean(axis = -2)
                            second_split = betas_per_condition[:,list(split[1]),:].mean(axis = -2)
                            first_split_betas = first_split.T
                            second_split_betas = second_split.T
                            first_upper = pdist(first_split_betas, metric='correlation') # get similarity matrix and extract upper triangle
                            second_upper = pdist(second_split_betas, metric='correlation') # get similarity matrix and extract upper triangle
                            # Calculate correlation between the two splits
                            if spearman:
                                cor, p_value = spearmanr(first_upper, second_upper)

                            else:
                                cor, p_value = pearsonr(first_upper, second_upper)

                            # store results 
                            correlation_results.append({
                                    "subject": sub,
                                    "ROI": ROI,
                                    "runtype": runtype,
                                    "smoothing": smoothing,
                                    "correlation": cor,
                                    "split": split_idx})
    # save end result as csv
    if spearman:                        
        os.makedirs(join(outdir,"RSA/ROI_within"),exist_ok=True)
        pd.DataFrame(correlation_results).to_csv(join(outdir,"RSA/ROI_within","rsa_results_spearman_within.csv"), index=False)
    else: 
        pd.DataFrame(correlation_results).to_csv(join(outdir,"RSA/ROI_within","rsa_results_pearson_within.csv"), index=False)

def PCA_voxels(mean_betas):
    """
    PCA function using SVD to extract number of principle components of matrix shaped 
    number of conditions by number of voxels. 
    Maximum number of components is 39. 
    """
    # Standardize (assuming data is shape conditions (40) by voxels (whatever size the ROI is))
    X_standard = (mean_betas - np.mean(mean_betas, axis=0)) / np.std(mean_betas, axis = 0)
    n_samples = X_standard.shape[0]
    n_features = X_standard.shape[1]

    M = X_standard/n_features**.5

    # Compute eigenvalues
    U, L, V = np.linalg.svd(M, full_matrices=False)

    # Sort eigenvalues 
    eigvals = L
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = V[:,idx]

    # Return % of explained variance for each PC
    explained_variance = (L**2) / (n_samples - 1)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)

    return eigvecs, explained_variance_ratio

def make_sphere(radius_voxels):
    """
    Function later used in the searchlight RSA to extract the sphere of voxels around each voxel. 
    Returns the offsets of all adjacent voxels, if within the radius (2mm our case) from the center voxel.
    """
    r = radius_voxels
    offsets = []
    # loop over all voxels 
    for x in range(-r, r+1):
        for y in range(-r, r+1):
            for z in range(-r, r+1):
                if x**2 + y**2 + z**2 <= r**2: # check if that voxel lies within the euclidean distance of the input radius 
                    offsets.append((x, y, z))
    return np.array(offsets)  

def RSA_searchlight(project_dir, subjects):
    """
    Calculates searchlight RSA for every participant. Loops over all subjects, designs and splits of betas (10).
    """
    # Set up directories 
    outdir = join(project_dir, 'miniblock/Outputs/')
    datadir = join(project_dir, 'miniblock/')
    smooths = ['sm_2_vox'] # only smoothed data
    presdir = join(project_dir, 'Behavior', 'designmats')
    runtypes = ['sus', 'miniblock', 'er']

    sphere_offsets = make_sphere(radius_voxels=2) # use custom function to extract spheres 
    # we want to calculate all splits of the 6 betas
    # this part creates all possible combinations when splitting in two groups
    elements = [0, 1, 2, 3, 4, 5]
    # Get all combinations of 3 elements
    group1_list = list(combinations(elements, 3))

    # To avoid duplicates (like (group1, group2) and (group2, group1)), only keep half
    splits = []
    seen = set()

    # Function that generates all the splits of 6 betas into two groups without order mattering 
    for group1 in group1_list:
        group2 = tuple(sorted(set(elements) - set(group1)))
        # Make sure we haven't already seen this partition
        key = tuple(sorted([group1, group2]))
        if key not in seen:
            seen.add(key)
            splits.append((group1, group2))

    for sub in subjects:
        # load the whole-brain mask just once 
        brain_mask_path = join(datadir, 'derivatives', f'sub-{sub}', 'anat', f'sub-{sub}_space-MNI152NLin2009cAsym_desc-brain_mask_resampled.nii.gz')
        brain_mask = image.load_img(brain_mask_path)
        mask = brain_mask.get_fdata() 
        for split in tqdm(splits, desc=f"Subject {sub}", position=0): # track progress
            for runtype in runtypes: 
                # Get the design matrix from GLMSingle to get the condition order 
                pattern = presdir + f'/P0{sub}_ConditionRich_Run*_{runtype}.csv'
                matches = glob.glob(pattern)
                matches.sort()
                
                design = []
                for i in range(len(matches)):
                    designMat = pd.read_csv(matches[i], header=None)
                    design.append(designMat)

                all_design = np.vstack((design[0], design[1], design[2]))
                condition_mask = all_design.sum(axis=1) > 0
                # a vector assigning conditions to trials
                condition_vector = np.argmax(all_design[condition_mask], axis=1)
                # Get the beta values into a vector 
                n_conditions = 40
                max_reps = 6

                # create a matrix of shape trials by condition containing the indices 
                repindices = np.full((max_reps, n_conditions), np.nan)
                for p in range(n_conditions):  
                    inds = np.where(condition_vector == p)[0]  
                    repindices[:len(inds), p] = inds  
                
                n_reps, n_conds = repindices.shape
                for smoothing in smooths: 

                    print(f"Working on: subject {sub}, runtype {runtype}, split {split}")
                    # Load the GLMsingle outputs 
                    results_glmsingle = dict()
                    results_glmsingle['typed'] = np.load(join(outdir,"GLMSingle_Outputs",f'{smoothing}_sub-{sub}_{runtype}_TYPED_FITHRF_GLMDENOISE_RR.npy'), allow_pickle=True).item()
                    betas = results_glmsingle['typed']['betasmd']

                    X, Y, Z, T = betas.shape
                    empty_array = np.zeros((X,Y,Z))

                    # Loop over all voxels 
                    voxel_count = 0
                    for x in range(X): 
                        for y in range(Y): 
                            for z in range(Z):
                                if not mask[x,y,z]: # check if this voxel is part of the whole brain mask 
                                    continue

                                # get the neighboring voxels 
                                neighbors = []
                                for dx, dy, dz in sphere_offsets:
                                    nx, ny, nz = x + dx, y + dy, z + dz
                                    if 0 <= nx < X and 0 <= ny < Y and 0 <= nz <Z:
                                        if mask[nx,ny,nz]:
                                            neighbors.append((nx,ny,nz))
                                # skip if too few neighboring voxels 
                                if len(neighbors) < 5: 
                                    continue
    
                                
                                searchlight_data = np.array([
                                    betas[nx, ny, nz, :] for (nx, ny, nz) in neighbors
                                ])  # shape: [n_voxels_in_sphere, n_trials]

                                # prepare empty array to store betas per repitition per condition
                                betas_per_condition = np.full((searchlight_data.shape[0], n_reps, n_conds), np.nan)

                                #  loop over conditions and fill the betas_per_condition matrix
                                for cond in range(n_conds):
                                    trial_indices = repindices[:, cond]
                                    for rep, trial_idx in enumerate(trial_indices):
                                        if not np.isnan(trial_idx):
                                            trial_idx = int(trial_idx)
                                            betas_per_condition[:, rep, cond] = searchlight_data[:, trial_idx]

                                # split betas into two groups - according to the current split
                                # Shape: X,Y,Z * 6 * 40 
                                even_betas_mean = betas_per_condition[:,list(split[0]),:].mean(axis=-2)
                                odd_betas_mean = betas_per_condition[:,list(split[1]),:].mean(axis=-2)

                                # calculate RSM based on correlation (distance)
                                cor_matrix_odd = np.corrcoef(odd_betas_mean.T)
                                cor_matrix_even = np.corrcoef(even_betas_mean.T)

                                # get the upper triangles of both RSMs
                                extract_even = cor_matrix_even[np.triu_indices(40, k = 1)]
                                extract_odd = cor_matrix_odd[np.triu_indices(40, k = 1)]

                                # get the correlation of both 
                                cor, p_value = pearsonr(extract_even, extract_odd)

                                # store at that voxel's location
                                empty_array[x,y,z] = cor
                    
                    # create one nifti image per split, per participant, per smoothing option
                    out_img = nib.Nifti1Image(empty_array, affine=brain_mask.affine)
                    split_idx = splits.index(split) + 1
                    out_subdir = join(outdir, 'RSA', 'searchlight', f"sub-{sub}")
                    os.makedirs(out_subdir, exist_ok=True)
                    nib.save(out_img, join(outdir, 'RSA', 'searchlight', f"sub-{sub}", f'{smoothing}_sub-{sub}_rsa_map_{runtype}_split_{split_idx}.nii.gz'))

def cv_pca_curves(betas_per_run):
    """
    betas_per_run: (voxels, runs=3, conds=40)
    returns: folds x Kmax array of R² curves
    """
    V, n_runs, C = betas_per_run.shape
    Kmax = C - 1
    curves = np.zeros((n_runs, Kmax))

    for test_run in range(n_runs):
        train_runs = [r for r in range(n_runs) if r != test_run]
        X_train = betas_per_run[:,train_runs,:].mean(axis=1)
        X_test  = betas_per_run[:,test_run,:]

        # Standardize
        mu = X_train.mean(axis=0)
        sd = X_train.std(axis=0, ddof=1)
        sd[sd==0] = 1.0
        Xtr = (X_train - mu) / sd
        Xte = (X_test  - mu) / sd
        Xtr = Xtr.T
        Xte = Xte.T

        # PCA
        U, S, Vt = np.linalg.svd(Xtr, full_matrices=False)
        Vvox = Vt.T

        # R² curve
        for k in range(1, Kmax+1):
            Uk = Vvox[:, :k]
            scores_te = Xte @ Uk
            Xte_hat = scores_te @ Uk.T
            num = np.sum((Xte - Xte_hat)**2)
            den = np.sum(Xte**2)
            curves[test_run, k-1] = 1 - num/den

    return curves

def PCA_CV(project_dir, subjects, ROIs):
    """
    Applies cross-validation to the PCA in a leave-one-run-out cross-validation scheme. A PCA is done using 2 of the 3 runs. 
    The left out matrix created with the left out run is then projected onto the eigenvectors created by the PCA on the 2 
    runs. The amount of variance explained by each of the principal components from the test data in the train data 
    is then stored in an array and saved to the results as a csv-file. 
    """    
    # set up directories
    outdir = join(project_dir, "miniblock/Outputs")
    datadir = join(project_dir, "miniblock")
    smooths = ['sm_2_vox']
    presdir = join(project_dir, 'Behavior', 'designmats')
    runtypes = ['sus', 'miniblock', 'er']
    # store results
    # track explained variance by each component in the train data
    all_R2 = np.zeros(shape=(20,3,4,39))

    for sub in range(len(subjects)):
        for runtype in range(len(runtypes)): 
                for smoothing in range(len(smooths)):
                    for ROI in range(len(ROIs)):
                        # load GLMsingle outputs
                        results_glmsingle = dict()
                        results_glmsingle['typed'] = np.load(join(outdir,"GLMSingle_Outputs",f'{smooths[smoothing]}_sub-{subjects[sub]}_{runtypes[runtype]}_TYPED_FITHRF_GLMDENOISE_RR.npy'), allow_pickle=True).item()
                        betas = results_glmsingle['typed']['betasmd']

                        brain_mask_path = join(datadir, 'derivatives', f'sub-{subjects[sub]}', 'anat', f'{ROIs[ROI]}_mask_sm_2_vox.nii')
                        brain_mask = image.load_img(brain_mask_path)
                        mask = brain_mask.get_fdata()   
                        # apply mask
                        masked_betas = betas[mask.astype(bool)]

                        # get design matrix from GLMsingle
                        pattern = presdir + f'/P0{subjects[sub]}_ConditionRich_Run*_{runtypes[runtype]}.csv'
                        matches = glob.glob(pattern)
                        matches.sort()
                        
                        design = []
                        for i in range(len(matches)):
                            designMat = pd.read_csv(matches[i], header=None)
                            design.append(designMat)

                        all_design = np.vstack((design[0], design[1], design[2]))
                        condition_mask = all_design.sum(axis=1) > 0
                        # a vector assigning condtions to trials
                        condition_vector = np.argmax(all_design[condition_mask], axis=1)
                        # Get the beta values into a vector 
                        n_conditions = 40
                        max_reps = 6

                        # create a matrix of shape trials by condition containing the indices 
                        repindices = np.full((max_reps, n_conditions), np.nan)
                        for p in range(n_conditions):  
                            inds = np.where(condition_vector == p)[0]  
                            repindices[:len(inds), p] = inds  
                        
                        # prepare empty array to store betas per repitition per condition
                        X, T = masked_betas.shape
                        n_reps, n_conds = repindices.shape
                        betas_per_condition = np.full((X, n_reps, n_conds), np.nan)

                        # loop over conditions and fill the betas_per_condition matrix
                        for cond in range(n_conds):
                            trial_indices = repindices[:, cond]
                            for rep, trial_idx in enumerate(trial_indices):
                                if not np.isnan(trial_idx):
                                    trial_idx = int(trial_idx)
                                    betas_per_condition[:, rep, cond] = masked_betas[:, trial_idx]
                        
                        betas_per_run = np.zeros(shape=(betas_per_condition.shape[0],3,betas_per_condition.shape[2]))
                        betas_per_run[:,0,:] = betas_per_condition[:,:2,:].mean(axis=1)
                        betas_per_run[:,1,:] = betas_per_condition[:,2:4,:].mean(axis=1)
                        betas_per_run[:,2,:] = betas_per_condition[:,4:,:].mean(axis=1)
                        
                        subject_curves = cv_pca_curves(betas_per_run)
                        mean_R2 = subject_curves.mean(axis=0)
                        all_R2[sub,runtype,ROI,:] = mean_R2
    mean_over_part = all_R2.mean(axis=0)
    make_2_dimension = []
    for runtype in range(3):
        for ROI in range(4):
            for n_component in range(39):
                make_2_dimension.append({
                    "ROI": ROIs[ROI],
                    "runtype": runtypes[runtype],
                    "component": n_component+1,
                    "value": mean_over_part[runtype,ROI,n_component]
                })

    make_2_dimension_subs = []
    for sub in range(20):
        for runtype in range(3):
            for ROI in range(4):
                for n_component in range(39):
                    make_2_dimension_subs.append({
                        "subject": subjects[sub],
                        "ROI": ROIs[ROI],
                        "runtype": runtypes[runtype],
                        "component": n_component+1,
                        "value": all_R2[sub,runtype,ROI,n_component]
                    })

    R2_df = pd.DataFrame(make_2_dimension)
    os.makedirs(join(outdir, "RSA/CV_PCA"), exist_ok=True)
    R2_df.to_csv(join(outdir, "RSA/CV_PCA/R2_df.csv"))

    R2_subs = pd.DataFrame(make_2_dimension_subs)
    R2_subs.to_csv(join(outdir, "RSA/CV_PCA/R2_subs.csv"))

def EVC_analysis(project_dir, subjects):
    """
    Calculates the correlation of the upper triangle of the RDM in EVC with each of the other ROIs' RDM for every subject. 
    Returns the results in the end with one value per subject, design and ROI.
    """
    # set up directories
    outdir = join(project_dir, "miniblock/Outputs")
    datadir = join(project_dir, "miniblock")
    smooths = ['sm_2_vox', "unsmoothed"]
    presdir = join(project_dir, 'Behavior', 'designmats')
    runtypes = ['sus', 'miniblock', 'er']
    ROIs = ["EBA_mask", "FFA_mask", "PPA_mask", "EVC_mask"]

    # store results here
    store_matrix = np.zeros((len(runtypes), len(subjects), len(smooths), len(ROIs), 780))

    for sub in range(len(subjects)):
        for runtype in range(len(runtypes)): 
                for smoothing in range(len(smooths)):
                    for ROI in range(len(ROIs)):
                        # Load GLMsingle outputs
                        results_glmsingle = dict()
                        results_glmsingle['typed'] = np.load(join(outdir,"GLMSingle_Outputs",f'{smooths[smoothing]}_sub-{subjects[sub]}_{runtypes[runtype]}_TYPED_FITHRF_GLMDENOISE_RR.npy'), allow_pickle=True).item()
                        betas = results_glmsingle['typed']['betasmd']
                        # load the participant-specific ROI mask 
                        brain_mask_path = join(datadir, 'derivatives', f'sub-{subjects[sub]}', 'anat', f'{ROIs[ROI]}_sm_2_vox.nii')
                        brain_mask = image.load_img(brain_mask_path)
                        mask = brain_mask.get_fdata()   

                        # Mask betas
                        masked_betas = betas[mask.astype(bool)]

                        # Get design matrix from GLMsingle
                        pattern = presdir + f'/P0{subjects[sub]}_ConditionRich_Run*_{runtypes[runtype]}.csv'
                        matches = glob.glob(pattern)
                        matches.sort()
                        
                        design = []
                        for i in range(len(matches)):
                            designMat = pd.read_csv(matches[i], header=None)
                            design.append(designMat)

                        all_design = np.vstack((design[0], design[1], design[2]))
                        condition_mask = all_design.sum(axis=1) > 0
                        # a vector assigning conditions to trials
                        condition_vector = np.argmax(all_design[condition_mask], axis=1)
                        # Get the beta values into a vector 
                        n_conditions = 40
                        max_reps = 6
                        
                        # create a matrix of shape trials by condition containing the indices 
                        repindices = np.full((max_reps, n_conditions), np.nan)
                        for p in range(n_conditions):  
                            inds = np.where(condition_vector == p)[0]  
                            repindices[:len(inds), p] = inds  
                        
                        # prepare empty array to store betas per repitition per condition
                        X, T = masked_betas.shape
                        n_reps, n_conds = repindices.shape
                        betas_per_condition = np.full((X, n_reps, n_conds), np.nan)

                        for cond in range(n_conds):
                            trial_indices = repindices[:, cond]
                            for rep, trial_idx in enumerate(trial_indices):
                                if not np.isnan(trial_idx):
                                    trial_idx = int(trial_idx)
                                    betas_per_condition[:, rep, cond] = masked_betas[:, trial_idx]
                        # result: number of voxels by repetitions by conditions
                        # calculte mean over all betas 
                        all_betas_means = betas_per_condition.mean(axis=-2)
                        all_betas = all_betas_means.T
                        upper_triangle = pdist(all_betas, metric='correlation') # get similarity matrix and extract upper triangle
                        
                        store_matrix[runtype, sub, smoothing, ROI, :] = upper_triangle # store upper triangle

    matrix_smoothed = store_matrix[:,:,0,:,:] # filter for smoothed data
    results_smoothed = []

    for sub in range(20):
        for design in range(3):
            for ROI in range(3):
                # for every subject, design and ROI calculate the correlation with the EVC
                cor,_ = spearmanr(matrix_smoothed[design,sub,ROI,:], matrix_smoothed[design,sub,3,:]) 
                # store results
                results_smoothed.append({
                    "design" : runtypes[design],
                    "subject" : subjects[sub],
                    "ROI" : ROIs[ROI],
                    "correlation": cor
        
                })
    # return as dataframe
    results_smoothed = pd.DataFrame(results_smoothed)
    os.makedirs(join(outdir,"RSA/ROI_within/EVC_analysis"),exist_ok=True)
    pd.DataFrame(results_smoothed).to_csv(join(outdir,"RSA/ROI_within/EVC_analysis","evc_results.csv"), index=False)
    
    return results_smoothed


