import numpy as np
import nibabel as nib
from nilearn import plotting, image
import pandas as pd
import os
import glob
from os.path import join, exists, split
from scipy.stats import pearsonr
import re
from itertools import combinations
import warnings

def get_whole_brain_rel_maps(project_dir, subjects):

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=scipy.stats.ConstantInputWarning)

    outdir = join(project_dir, 'miniblock/Outputs')
    datadir = join(project_dir, 'miniblock')
    presdir = join(project_dir, 'Behavior', 'designmats')
    smooths = ["unsmoothed", "sm_2_vox"]
    runtypes = ['miniblock', 'sus', 'er']

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

    for sub in subjects: 
        counter = 0
        print(f"Now working on subject {sub}")
        for split in splits: 
            counter += 1
            for runtype in runtypes:
                for smoothing in smooths: 

                    results_glmsingle = dict()
                    results_glmsingle['typed'] = np.load(join(outdir,f'sub-{sub}',f'{smoothing}_sub-{sub}_{runtype}/TYPED_FITHRF_GLMDENOISE_RR.npy'), allow_pickle=True).item()
                    betas = results_glmsingle['typed']['betasmd']

                    brain_mask = image.load_img(join(datadir, 'derivatives', f'sub-{sub}', 'anat', f'sub-{sub}_space-MNI152NLin2009cAsym_desc-brain_mask_resampled.nii.gz'))
                    mask = brain_mask.get_fdata()

                    masked_betas = betas[mask.astype(bool)]
                    unmasked_betas = np.zeros(betas.shape)
                    unmasked_betas[mask.astype(bool)] = masked_betas

                    # Load and process design matrix
                    pattern = presdir + f'/P0{sub}_ConditionRich_Run*_{runtype}.csv'
                    matches = glob.glob(pattern)
                    matches.sort()
                    design = []
                    for i in range(len(matches)):
                        designMat = pd.read_csv(matches[i], header=None)
                        num = re.search(r'Run_(\d+)', matches[i])
                        runNum = int(num.group(1))
                        # Adjust runNum for interspersed localizer runs
                        if (runNum > 3) & (runNum < 7) & (sub != '01'): 
                            runNum += 1
                        elif (runNum >= 7):
                            runNum += 2
                        elif (sub == '01') & (runNum > 4) & (runNum < 7):
                            runNum += 1
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

                    # Preallocate betas_per_condition array
                    X, Y, Z, T = unmasked_betas.shape
                    n_reps, n_conds = repindices.shape
                    betas_per_condition = np.full((X, Y, Z, n_reps, n_conds), np.nan)

                    # Populate betas_per_condition array
                    for cond in range(n_conds):
                        trial_indices = repindices[:, cond]
                        for rep, trial_idx in enumerate(trial_indices):
                            if not np.isnan(trial_idx):
                                trial_idx = int(trial_idx)
                                betas_per_condition[:, :, :, rep, cond] = unmasked_betas[:, :, :, trial_idx]

                    # Compute reliability map
                    reliability_map = np.full((X, Y, Z), np.nan)
                    even_betas_mean = betas_per_condition[:,:,:,list(split[0]),:].mean(axis=3)
                    odd_betas_mean = betas_per_condition[:,:,:,list(split[1]),:].mean(axis=3)

                    rel_map,_ = pearsonr(even_betas_mean, odd_betas_mean, axis=-1)

                    # Save reliability map as NIfTI file
                    reliability_img = nib.Nifti1Image(rel_map, brain_mask.affine)
                    reliability_filename = join(outdir, 'reliability', f'sub-{sub}', f'{smoothing}_sub-{sub}_{runtype}_reliability_map_split_{counter}.nii.gz')
                    sub_outdir = join(outdir, 'reliability', f'sub-{sub}')
                    os.makedirs(sub_outdir, exist_ok=True)
                    nib.save(reliability_img, reliability_filename)