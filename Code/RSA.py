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

def RSA_between(project_dir, subjects, spearman=True):
    outdir = join(project_dir, 'miniblock/Outputs/')
    datadir = join(project_dir, 'miniblock/')

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
    smooths = ['sm_2_vox']
    presdir = join(project_dir, 'Behavior', 'designmats')
    runtypes = ['sus', 'miniblock', 'er']
    ROIs = ["EBA_mask", "FFA_mask", "PPA_mask", "EVC_mask"]

    correlation_results = []
    for ROI in ROIs: 
        for pair_name, subjects in pairs_dict.items():
            #print(f"Now working on {pair_name} in ROI: {ROI}")
            for runtype in runtypes: 
                    for smoothing in smooths:
                        upper_triangles_flattened = []
                        for sub in subjects: 
                            results_glmsingle = dict()
                            results_glmsingle['typed'] = np.load(join(outdir,"GLMSingle_Outputs",f'{smoothing}_sub-{sub}_{runtype}_TYPED_FITHRF_GLMDENOISE_RR.npy'), allow_pickle=True).item()
                            betas = results_glmsingle['typed']['betasmd']


                            brain_mask_path = join(datadir, 'derivatives', f'sub-{sub}', 'anat', f'{ROI}_sm_2_vox.nii.gz')
                            brain_mask = image.load_img(brain_mask_path)
                            mask = brain_mask.get_fdata()   

                            masked_betas = betas[mask.astype(bool)]

                            pattern = presdir + f'/P0{sub}_ConditionRich_Run*_{runtype}.csv'
                            matches = glob.glob(pattern)
                            matches.sort()
                            
                            design = []
                            for i in range(len(matches)):
                                designMat = pd.read_csv(matches[i], header=None)
                                design.append(designMat)

                            all_design = np.vstack((design[0], design[1], design[2]))
                            condition_mask = all_design.sum(axis=1) > 0
                            condition_vector = np.argmax(all_design[condition_mask], axis=1)
                            n_conditions = 40
                            max_reps = 6

                            repindices = np.full((max_reps, n_conditions), np.nan)
                            for p in range(n_conditions):  
                                inds = np.where(condition_vector == p)[0]  
                                repindices[:len(inds), p] = inds  
                            
                            X, T = masked_betas.shape
                            n_reps, n_conds = repindices.shape
                            betas_per_condition = np.full((X, n_reps, n_conds), np.nan)

                            for cond in range(n_conds):
                                trial_indices = repindices[:, cond]
                                for rep, trial_idx in enumerate(trial_indices):
                                    if not np.isnan(trial_idx):
                                        trial_idx = int(trial_idx)
                                        betas_per_condition[:, rep, cond] = masked_betas[:, trial_idx]

                            all_betas_means = betas_per_condition.mean(axis=-2)
                            all_betas = all_betas_means.T
                            upper_triangle = pdist(all_betas, metric='correlation')
                            upper_triangles_flattened.append(upper_triangle)

                        if spearman: 
                            cor, p_value = spearmanr(upper_triangles_flattened[0], upper_triangles_flattened[1])
                        else: 
                            cor, p_value = pearsonr(upper_triangles_flattened[0], upper_triangles_flattened[1])


                        correlation_results.append({
                                    "pair": pair_name,
                                    "ROI": ROI,
                                    "runtype": runtype,
                                    "smoothing": smoothing,
                                    "correlation": cor})

    if spearman:                        
        os.makedirs(join(outdir,"RSA/ROI_between"),exist_ok=True)
        pd.DataFrame(correlation_results).to_csv(join(outdir,"RSA/ROI_between","rsa_results_spearman_between.csv"), index=False)
    else: 
        pd.DataFrame(correlation_results).to_csv(join(outdir,"RSA/ROI_between","rsa_results_pearson_between.csv"), index=False)

def RSA_within(project_dir, subjects, spearman=True):
    outdir = join(project_dir, 'miniblock/Outputs/')
    datadir = join(project_dir, 'miniblock/')
    smooths = ['sm_2_vox']
    presdir = join(project_dir, 'Behavior', 'designmats')
    runtypes = ['sus', 'miniblock', 'er']
    ROIs = ["EBA_mask", "FFA_mask", "PPA_mask", "EVC_mask"]
    subjects = [f"{i:02d}" for i in range(1, 23) if i not in [9, 16]]

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

    correlation_results = []
    for ROI in ROIs: 
        for sub in subjects:
            for split in splits:
                split_idx = splits.index(split) + 1
                for runtype in runtypes: 
                        for smoothing in smooths:
                            
                            results_glmsingle = dict()
                            results_glmsingle['typed'] = np.load(join(outdir,"GLMSingle_Outputs",f'{smoothing}_sub-{sub}_{runtype}_TYPED_FITHRF_GLMDENOISE_RR.npy'), allow_pickle=True).item()
                            betas = results_glmsingle['typed']['betasmd']

                            brain_mask_path = join(datadir, 'derivatives', f'sub-{sub}', 'anat', f'{ROI}_sm_2_vox.nii.gz')
                            brain_mask = image.load_img(brain_mask_path)
                            mask = brain_mask.get_fdata()   

                            masked_betas = betas[mask.astype(bool)]

                            pattern = presdir + f'/P0{sub}_ConditionRich_Run*_{runtype}.csv'
                            matches = glob.glob(pattern)
                            matches.sort()
                            
                            design = []
                            for i in range(len(matches)):
                                designMat = pd.read_csv(matches[i], header=None)
                                design.append(designMat)

                            all_design = np.vstack((design[0], design[1], design[2]))
                            condition_mask = all_design.sum(axis=1) > 0
                            condition_vector = np.argmax(all_design[condition_mask], axis=1)
                            n_conditions = 40
                            max_reps = 6

                            repindices = np.full((max_reps, n_conditions), np.nan)
                            for p in range(n_conditions):  
                                inds = np.where(condition_vector == p)[0]  
                                repindices[:len(inds), p] = inds  
                            
                            X, T = masked_betas.shape
                            n_reps, n_conds = repindices.shape
                            betas_per_condition = np.full((X, n_reps, n_conds), np.nan)

                            for cond in range(n_conds):
                                trial_indices = repindices[:, cond]
                                for rep, trial_idx in enumerate(trial_indices):
                                    if not np.isnan(trial_idx):
                                        trial_idx = int(trial_idx)
                                        betas_per_condition[:, rep, cond] = masked_betas[:, trial_idx]

                            first_split = betas_per_condition[:,list(split[0]),:].mean(axis = -2)
                            second_split = betas_per_condition[:,list(split[1]),:].mean(axis = -2)
                            first_split_betas = first_split.T
                            second_split_betas = second_split.T
                            first_upper = pdist(first_split_betas, metric='correlation')
                            second_upper = pdist(second_split_betas, metric='correlation')
                            if spearman:
                                cor, p_value = spearmanr(first_upper, second_upper)

                            else:
                                cor, p_value = pearsonr(first_upper, second_upper)

                            correlation_results.append({
                                    "subject": sub,
                                    "ROI": ROI,
                                    "runtype": runtype,
                                    "smoothing": smoothing,
                                    "correlation": cor,
                                    "split": split_idx})

    if spearman:                        
        os.makedirs(join(outdir,"RSA/ROI_within"),exist_ok=True)
        pd.DataFrame(correlation_results).to_csv(join(outdir,"RSA/ROI_within","rsa_results_spearman_within.csv"), index=False)
    else: 
        pd.DataFrame(correlation_results).to_csv(join(outdir,"RSA/ROI_within","rsa_results_pearson_within.csv"), index=False)


def PCA_voxels(mean_betas):
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
    r = radius_voxels
    offsets = []
    for x in range(-r, r+1):
        for y in range(-r, r+1):
            for z in range(-r, r+1):
                if x**2 + y**2 + z**2 <= r**2: # check if that voxel lies within the euclidean distance of the input radius 
                    offsets.append((x, y, z))
    return np.array(offsets)  

def RSA_searchlight(project_dir, subjects):
    outdir = join(project_dir, 'miniblock/Outputs/')
    datadir = join(project_dir, 'miniblock/')
    smooths = ['sm_2_vox']
    presdir = join(project_dir, 'Behavior', 'designmats')
    runtypes = ['sus', 'miniblock', 'er']

    sphere_offsets = make_sphere(radius_voxels=2)  

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
        for split in tqdm(splits, desc=f"Subject {sub}", position=0):
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
                condition_vector = np.argmax(all_design[condition_mask], axis=1)
                n_conditions = 40
                max_reps = 6

                # Get the conditions in order 
                repindices = np.full((max_reps, n_conditions), np.nan)
                for p in range(n_conditions):  
                    inds = np.where(condition_vector == p)[0]  
                    repindices[:len(inds), p] = inds  
                
                n_reps, n_conds = repindices.shape
                for smoothing in smooths: 

                    print(f"Working on: subject {sub}, runtype {runtype}, split {split}")
                    # Load the betas 
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

                            
                                betas_per_condition = np.full((searchlight_data.shape[0], n_reps, n_conds), np.nan)

                                # get the corresponding beta values into the correct shape
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

                                # Get the correlation of both 
                                cor, p_value = pearsonr(extract_even, extract_odd)

                                # store at that voxel's location
                                empty_array[x,y,z] = cor
                    
                    # Create one nifti image per split, per participant, per smoothing option
                    out_img = nib.Nifti1Image(empty_array, affine=brain_mask.affine)
                    split_idx = splits.index(split) + 1
                    out_subdir = join(outdir, 'RSA', 'searchlight', f"sub-{sub}")
                    os.makedirs(out_subdir, exist_ok=True)
                    nib.save(out_img, join(outdir, 'RSA', 'searchlight', f"sub-{sub}", f'{smoothing}_sub-{sub}_rsa_map_{runtype}_split_{split_idx}.nii.gz'))

def PCA_all_trials(project_dir, subjects):
 
    outdir = join(project_dir, "miniblock/Outputs")
    datadir = join(project_dir, "miniblock")
    smooths = ['unsmoothed']
    presdir = join(project_dir, 'Behavior', 'designmats')
    runtypes = ['sus', 'miniblock', 'er']
    ROIs  = ["FFA", "PPA", "EBA", "EVC"]

    explained_variance = np.zeros(shape=(20,3,4,40))

    for sub in range(len(subjects)):
        for runtype in range(len(runtypes)): 
                for smoothing in range(len(smooths)):
                    for ROI in range(len(ROIs)):

                        results_glmsingle = dict()
                        results_glmsingle['typed'] = np.load(join(outdir,"GLMSingle_Outputs",f'{smooths[smoothing]}_sub-{subjects[sub]}_{runtypes[runtype]}_TYPED_FITHRF_GLMDENOISE_RR.npy'), allow_pickle=True).item()
                        betas = results_glmsingle['typed']['betasmd']

                        if ROI == "visually_responsive_voxels":
                            brain_mask_path = join(datadir, 'derivatives', f'sub-{subjects[sub]}', 'anat', f'{ROIs[ROI]}_sm_2_vox_gm.nii')
                        else: 
                            brain_mask_path = join(datadir, 'derivatives', f'sub-{subjects[sub]}', 'anat', f'{ROIs[ROI]}_mask_sm_2_vox.nii')
                        brain_mask = image.load_img(brain_mask_path)
                        mask = brain_mask.get_fdata()   

                        masked_betas = betas[mask.astype(bool)]

                        pattern = presdir + f'/P0{subjects[sub]}_ConditionRich_Run*_{runtypes[runtype]}.csv'
                        matches = glob.glob(pattern)
                        matches.sort()
                        
                        design = []
                        for i in range(len(matches)):
                            designMat = pd.read_csv(matches[i], header=None)
                            design.append(designMat)

                        all_design = np.vstack((design[0], design[1], design[2]))
                        condition_mask = all_design.sum(axis=1) > 0
                        condition_vector = np.argmax(all_design[condition_mask], axis=1)
                        n_conditions = 40
                        max_reps = 6

                        repindices = np.full((max_reps, n_conditions), np.nan)
                        for p in range(n_conditions):  
                            inds = np.where(condition_vector == p)[0]  
                            repindices[:len(inds), p] = inds  
                        
                        X, T = masked_betas.shape
                        n_reps, n_conds = repindices.shape
                        betas_per_condition = np.full((X, n_reps, n_conds), np.nan)

                        for cond in range(n_conds):
                            trial_indices = repindices[:, cond]
                            for rep, trial_idx in enumerate(trial_indices):
                                if not np.isnan(trial_idx):
                                    trial_idx = int(trial_idx)
                                    betas_per_condition[:, rep, cond] = masked_betas[:, trial_idx]
                        
                        mean_betas = betas_per_condition.mean(axis=1)
                        eigvecs, explained_variance_ratio = PCA_voxels(mean_betas.T)

                        explained_variance[sub, runtype, ROI, :] = explained_variance_ratio[:40]

    return explained_variance

def PCA_CV(project_dir, subjects, ROIs):
    outdir = join(project_dir, "miniblock/Outputs")
    datadir = join(project_dir, "miniblock")
    smooths = ['sm_2_vox']
    presdir = join(project_dir, 'Behavior', 'designmats')
    runtypes = ['sus', 'miniblock', 'er']
    explained_variance_train = np.zeros(shape=(20,3,4,6,40))
    all_eigenvectors = np.zeros(shape=(20,3,4,6,40,40))
    explained_variance_test = np.zeros(shape=(20,3,4,6,40))
    first_39_components = np.zeros(shape=(20,3,4,6,39))


    for sub in range(len(subjects)):
        for runtype in range(len(runtypes)): 
                for smoothing in range(len(smooths)):
                    for ROI in range(len(ROIs)):
                        
                        results_glmsingle = dict()
                        results_glmsingle['typed'] = np.load(join(outdir,"GLMSingle_Outputs",f'{smooths[smoothing]}_sub-{subjects[sub]}_{runtypes[runtype]}_TYPED_FITHRF_GLMDENOISE_RR.npy'), allow_pickle=True).item()
                        betas = results_glmsingle['typed']['betasmd']

                        if ROI == "visually_responsive_voxels":
                            brain_mask_path = join(datadir, 'derivatives', f'sub-{subjects[sub]}', 'anat', f'{ROIs[ROI]}_sm_2_vox_gm.nii')
                        else: 
                            brain_mask_path = join(datadir, 'derivatives', f'sub-{subjects[sub]}', 'anat', f'{ROIs[ROI]}_mask_sm_2_vox.nii')
                        brain_mask = image.load_img(brain_mask_path)
                        mask = brain_mask.get_fdata()   

                        masked_betas = betas[mask.astype(bool)]

                        pattern = presdir + f'/P0{subjects[sub]}_ConditionRich_Run*_{runtypes[runtype]}.csv'
                        matches = glob.glob(pattern)
                        matches.sort()
                        
                        design = []
                        for i in range(len(matches)):
                            designMat = pd.read_csv(matches[i], header=None)
                            design.append(designMat)

                        all_design = np.vstack((design[0], design[1], design[2]))
                        condition_mask = all_design.sum(axis=1) > 0
                        condition_vector = np.argmax(all_design[condition_mask], axis=1)
                        n_conditions = 40
                        max_reps = 6

                        repindices = np.full((max_reps, n_conditions), np.nan)
                        for p in range(n_conditions):  
                            inds = np.where(condition_vector == p)[0]  
                            repindices[:len(inds), p] = inds  
                        
                        X, T = masked_betas.shape
                        n_reps, n_conds = repindices.shape
                        betas_per_condition = np.full((X, n_reps, n_conds), np.nan)

                        for cond in range(n_conds):
                            trial_indices = repindices[:, cond]
                            for rep, trial_idx in enumerate(trial_indices):
                                if not np.isnan(trial_idx):
                                    trial_idx = int(trial_idx)
                                    betas_per_condition[:, rep, cond] = masked_betas[:, trial_idx]


                        # First for no cross-validation
                        for i in range(6):
                            train_idx = np.arange(6) != i
                            beta_filtered = betas_per_condition[:,train_idx,:]
                            beta_filtered_test = betas_per_condition[:,i,:]
                            beta_filtered = beta_filtered.mean(axis=1)
                            eigvecs, explained_variance_ratio = PCA_voxels(beta_filtered.T)

                            # Project hold-out betas onto the fitted PCs
                            beta_filtered_test_centered = (beta_filtered_test - np.mean(beta_filtered_test, axis=0)) / np.std(beta_filtered_test, axis = 0)
                            beta_test_projected = beta_filtered_test_centered @ eigvecs.T

                            # Store eigenvectors and explained variance in the train set
                            explained_variance_train[sub, runtype, ROI, :] = explained_variance_ratio
                            all_eigenvectors[sub, runtype, ROI, :, :] = eigvecs

                            # Variance captured along each PC
                            variance_along_pcs = np.var(beta_test_projected, axis=0, ddof=1)

                            # Total variance in hold-out data
                            total_variance_new_data = np.sum(np.var(beta_test_projected, axis=0, ddof=1))

                            # Fraction explained by each PC
                            explained_fraction_per_pc = variance_along_pcs / total_variance_new_data

                            explained_variance_test[sub, runtype, ROI, :] = explained_fraction_per_pc
                            non_noise_variance = explained_fraction_per_pc[:39].sum()
                            first_39_components[sub, runtype, ROI, :] = non_noise_variance

    return explained_variance_train, explained_variance_test, all_eigenvectors, first_39_components



