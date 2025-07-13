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
from numpy.linalg import eigh


def PCA_all_trials(project_dir, subjects, ROIs):
 
    outdir = join(project_dir, "miniblock/Outputs")
    datadir = join(project_dir, "miniblock")
    smooths = ['sm_2_vox']
    presdir = join(project_dir, 'Behavior', 'designmats')
    runtypes = ['sus', 'miniblock', 'er']

    explained_variance = np.zeros(shape=(20,3,4,40))

    for sub in range(len(subjects)):
        for runtype in range(len(runtypes)): 
                for smoothing in range(len(smooths)):
                    for ROI in range(len(ROIs)):
                        
                        results_glmsingle = dict()
                        results_glmsingle['typed'] = np.load(join(outdir,"GLMSingle_Outputs",f'{smooths[smoothing]}_sub-{subjects[sub]}_{runtypes[runtype]}_TYPED_FITHRF_GLMDENOISE_RR.npy'), allow_pickle=True).item()
                        betas = results_glmsingle['typed']['betasmd']

                        if ROI == "visually_responsive_voxels":
                            brain_mask_path = join(datadir, 'derivatives', f'sub-{subjects[sub]}', 'anat', f'{ROIs[ROI]}_sm_2_vox_gm.nii.gz')
                        else: 
                            brain_mask_path = join(datadir, 'derivatives', f'sub-{subjects[sub]}', 'anat', f'{ROIs[ROI]}_mask_sm_2_vox.nii.gz')
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
                        mean_betas = betas_per_condition.mean(axis=1)

                        pca = PCA()
                        pca.fit(mean_betas)
                        explained_variance[sub, runtype, ROI, :] = pca.explained_variance_ratio_

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


    for sub in range(len(subjects)):
        for runtype in range(len(runtypes)): 
                for smoothing in range(len(smooths)):
                    for ROI in range(len(ROIs)):
                        
                        results_glmsingle = dict()
                        results_glmsingle['typed'] = np.load(join(outdir,"GLMSingle_Outputs",f'{smooths[smoothing]}_sub-{subjects[sub]}_{runtypes[runtype]}_TYPED_FITHRF_GLMDENOISE_RR.npy'), allow_pickle=True).item()
                        betas = results_glmsingle['typed']['betasmd']

                        if ROI == "visually_responsive_voxels":
                            brain_mask_path = join(datadir, 'derivatives', f'sub-{subjects[sub]}', 'anat', f'{ROIs[ROI]}_sm_2_vox_gm.nii.gz')
                        else: 
                            brain_mask_path = join(datadir, 'derivatives', f'sub-{subjects[sub]}', 'anat', f'{ROIs[ROI]}_mask_sm_2_vox.nii.gz')
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
                            pca = PCA()
                            pca.fit(beta_filtered)
                            # Project hold-out betas onto the fitted PCs
                            beta_test_projected = pca.transform(beta_filtered_test)

                            # Store eigenvectors and explained variance in the train set
                            explained_variance_train[sub, runtype, ROI, :] = pca.explained_variance_ratio_
                            all_eigenvectors[sub, runtype, ROI, :, :] = pca.components_

                            # Variance captured along each PC
                            variance_along_pcs = np.var(beta_test_projected, axis=0, ddof=1)

                            # Total variance in hold-out data
                            total_variance_new_data = np.sum(np.var(beta_test_projected, axis=0, ddof=1))

                            # Fraction explained by each PC
                            explained_fraction_per_pc = variance_along_pcs / total_variance_new_data

                            explained_variance_test[sub, runtype, ROI, :] = explained_fraction_per_pc

    return explained_variance_train, explained_variance_test, all_eigenvectors



