import numpy as np
import nibabel as nib
from nilearn import plotting, image
import pandas as pd
import os
import glob
from os.path import join, exists, split
import scipy
from scipy.stats import pearsonr
import re
from itertools import combinations
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.anova import AnovaRM
from scipy.stats import ttest_rel
import statsmodels.stats.multitest as smm

def get_whole_brain_rel_maps(project_dir, subjects):
    """
    Creates whole-brain reliability maps for all participants for the following specifications: 
    - design (er, miniblock, sus)
    - smoothin option (sm_2_vox, unsmoothed)
    - splits (1-10)

    All maps are stored in Outputs/sub-xx.
    """
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
                    results_glmsingle['typed'] = np.load(join(outdir,"GLMSingle_Outputs",f'{smoothing}_sub-{sub}_{runtype}_TYPED_FITHRF_GLMDENOISE_RR.npy'), allow_pickle=True).item()
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

def gather_reliability_maps(project_dir, subjects, ROIs):
    """
    Gathers reliability maps to a dataframe and stores it, if all participants are included (20). 
    For every participant and smoothing option, the maps are averaged for every voxel. 
    Lastly, for every ROI, the median, mean and maximum reliability is stored in the dataframe. 
    """
    outdir = join(project_dir, 'miniblock/Outputs/reliability')
    anatdir = join(project_dir, 'miniblock/derivatives')
    smooths = ["unsmoothed", "sm_2_vox"]
    runtypes = ['miniblock', 'sus', 'er']

    splits = range(1,11)

    results = []

    for sub in subjects:
        for ROI in ROIs:
            # Load brain mask
            if ROI in ["FFA" , "PPA" , "EBA", "EVC"]:
                brain_mask_path = join(anatdir, f'sub-{sub}', 'anat', f'{ROI}_mask_sm_2_vox.nii.gz')
                brain_mask = image.load_img(brain_mask_path)
                mask = brain_mask.get_fdata()
            elif ROI in ["visually_responsive_voxels", "occipital_mask"]:
                brain_mask_path = join(anatdir, f'sub-{sub}', 'anat', f'{ROI}_sm_2_vox_gm.nii.gz')
                brain_mask = image.load_img(brain_mask_path)
                mask = brain_mask.get_fdata()

            for runtype in runtypes:
                for smoothing in smooths:
                    for split_num in splits:
                        # Construct filename
                        reliability_filename = join(
                            outdir, f'sub-{sub}', f'{smoothing}_sub-{sub}_{runtype}_reliability_map_split_{split_num}.nii.gz'
                        )

                        if os.path.exists(reliability_filename):
                            reliability_img = nib.load(reliability_filename)
                            reliability_data = reliability_img.get_fdata()


                            masked_values = reliability_data[mask.astype(bool)]
                            median_reliability = np.nanmedian(masked_values)
                            mean_reliability = np.nanmean(masked_values)
                            max_reliability = masked_values[np.nanargmax(masked_values)]
                    

                            results.append({
                                "subject": sub,
                                "runtype": runtype,
                                "smoothing": smoothing,
                                "split": split_num,
                                "median_reliability": median_reliability,
                                "mean_reliability": mean_reliability,
                                "maximum_reliability": max_reliability,
                                "ROI": ROI
                            })
                        else:
                            print(f"Missing file: {reliability_filename}")

    # Build DataFrame
    df = pd.DataFrame(results)
    print("Data collected for", len(df), "split maps.")

    if len(subjects) == 20:
        df.to_csv(join(outdir, 'reliability_results_all.csv'), index=False)

    return(df)

def make_reliability_plots(dataframe):
    """
    Simple plotting function that visualizes results by ROI, smoothing option and design. 
    Creates one plot per ROI.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    # Optional: custom palette
    palette = sns.color_palette("Set1")

    ROI_dir = {
        "EVC": "Early Visual Cortex",
        "visually_responsive_voxels": "Visually Responsive Voxels",
        "EBA": "Extrastriate Body Area",
        "FFA": "Fusiform Face Area",
        "PPA": "Parahippocampal Place Area"
    }

    # Create one plot per ROI
    for roi, roi_name in ROI_dir.items():
        roi_df = dataframe[dataframe["ROI"] == roi]

        g = sns.catplot(
            data=roi_df,
            kind="bar",
            x="runtype",
            y="median_reliability",
            hue="smoothing",
            height=4,
            aspect=1.2,
            palette=palette,
            errorbar="se",
            capsize=0.1,
            err_kws={'linewidth': 1.5}
        )

            # Set y-axis limits and ticks on the main axis
        ax = g.ax
        ax.set_ylim(0, 0.5)  # set min and max y-axis limits, adjust to your needs
        ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4,0.5])  # set custom tick marks

        # Optional: format y-axis labels
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

        g.set_titles(f"{roi}", size=16, weight="bold")
        g.set_axis_labels("Design", "Median Reliability")
        g.set_xticklabels()
        g._legend.remove()


        plt.title(f"Median Reliability by Design and Smoothing Option -\n{roi_name}", fontsize=18, fontweight="bold")
        custom_labels = ["Event-Related", "Sustained", "Miniblock"]
        plt.xticks(ticks=np.arange(len(custom_labels)), 
            labels=custom_labels)
        plt.legend(title="Smoothing")
        plt.tight_layout()
        plt.show()

def reliability_progression_between_runs(project_dir, subjects):
    """
    Calculates reliability between runs for every participant (per design and smoothing option). 3 Comparisons are made: 
    - 1: run 1 vs. run 2
    - 2: run 2 vs. run 3
    - 3: run 1 vs. run 3
    """

    outdir = join(project_dir,'miniblock/Outputs/')
    datadir = join(project_dir,'miniblock/')
    presdir = join(project_dir, 'Behavior', 'designmats')
    smooths = ["unsmoothed", "sm_2_vox"]
    runtypes = ['miniblock', 'sus', 'er']


    for sub in subjects: 
        print(f"Now working on subject {sub}")
        for runtype in runtypes:
            for smoothing in smooths: 

                results_glmsingle = dict()
                results_glmsingle['typed'] = np.load(join(outdir,"GLMSingle_Outputs",f'{smoothing}_sub-{sub}_{runtype}_TYPED_FITHRF_GLMDENOISE_RR.npy'), allow_pickle=True).item()
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
                first_betas_mean = betas_per_condition[:,:,:,:2,:].mean(axis=3)
                middle_betas_mean = betas_per_condition[:,:,:,2:4,:].mean(axis=3)
                last_betas_mean = betas_per_condition[:,:,:,4:,:].mean(axis=3)

                rel_map_1,_ = pearsonr(first_betas_mean, middle_betas_mean, axis=-1)
                rel_map_2,_ = pearsonr(middle_betas_mean, last_betas_mean, axis=-1)
                rel_map_3,_ = pearsonr(first_betas_mean, last_betas_mean, axis=-1)

                # Save reliability map as NIfTI file
                sub_outdir = join(outdir, 'reliability', 'progression_analysis_between',f'sub-{sub}')
                os.makedirs(sub_outdir, exist_ok=True)
                reliability_img_1 = nib.Nifti1Image(rel_map_1, brain_mask.affine)
                reliability_filename_1 = join(outdir, 'reliability', 'progression_analysis_between', f'sub-{sub}', f'{smoothing}_sub-{sub}_{runtype}_reliability_map_comp_1.nii.gz')
                nib.save(reliability_img_1, reliability_filename_1)

                reliability_img_2 = nib.Nifti1Image(rel_map_2, brain_mask.affine)
                reliability_filename_2 = join(outdir, 'reliability', 'progression_analysis_between', f'sub-{sub}', f'{smoothing}_sub-{sub}_{runtype}_reliability_map_comp_2.nii.gz')
                nib.save(reliability_img_2, reliability_filename_2)

                reliability_img_3 = nib.Nifti1Image(rel_map_3, brain_mask.affine)
                reliability_filename_3 = join(outdir, 'reliability', 'progression_analysis_between', f'sub-{sub}', f'{smoothing}_sub-{sub}_{runtype}_reliability_map_comp_3.nii.gz')
                nib.save(reliability_img_3, reliability_filename_3)

def gather_progession_between(project_dir, subjects, ROIs):
    """
    Gathers between run reliabilities for every ROI. Outputs a dataframe if all participants are included. 
    """
    outdir = join(project_dir, 'miniblock/Outputs/reliability/progression_analysis_between')
    anatdir = join(project_dir, 'miniblock/derivatives')
    smooths = ["unsmoothed", "sm_2_vox"]
    runtypes = ['miniblock', 'sus', 'er']
    comparisons = range(1,4)

    results = []

    for sub in subjects:
        for ROI in ROIs:
            # Load brain mask
            if ROI in ["FFA" , "PPA" , "EBA", "EVC"]:
                brain_mask_path = join(anatdir, f'sub-{sub}', 'anat', f'{ROI}_mask_sm_2_vox.nii.gz')
                brain_mask = image.load_img(brain_mask_path)
                mask = brain_mask.get_fdata()
            elif ROI in ["visually_responsive_voxels", "occipital_mask"]:
                brain_mask_path = join(anatdir, f'sub-{sub}', 'anat', f'{ROI}_sm_2_vox_gm.nii.gz')
                brain_mask = image.load_img(brain_mask_path)
                mask = brain_mask.get_fdata()

            for runtype in runtypes:
                for smoothing in smooths:
                    for comp in comparisons:
                        # Construct filename
                        reliability_filename = join(
                            outdir, f'sub-{sub}', f'{smoothing}_sub-{sub}_{runtype}_reliability_map_comp_{comp}.nii.gz'
                        )

                        if os.path.exists(reliability_filename):
                            reliability_img = nib.load(reliability_filename)
                            reliability_data = reliability_img.get_fdata()


                            masked_values = reliability_data[mask.astype(bool)]
                            median_reliability = np.nanmedian(masked_values)
                            mean_reliability = np.nanmean(masked_values)
                    

                            results.append({
                            "subject": sub,
                            "runtype": runtype,
                            "smoothing": smoothing,
                            "median_rel": median_reliability,
                            "mean_rel": mean_reliability,
                            "ROI": ROI,
                            "comparison": comp})
                        else:
                            print(f"Missing file: {reliability_filename}")

    # Build DataFrame
    df = pd.DataFrame(results)

    if len(subjects) == 20:
        df.to_csv(join(outdir, 'reliability_progression_between.csv'), index=False)

    return(df)

def reliability_progression_within_runs(project_dir, subjects):
    """
    Calculates reliability within runs for every participant (per design and smoothing option). 
    3 runs are analyzed. 
    """

    outdir = join(project_dir,'miniblock/Outputs/')
    datadir = join(project_dir,'miniblock/')
    presdir = join(project_dir, 'Behavior', 'designmats')
    smooths = ["unsmoothed", "sm_2_vox"]
    runtypes = ['miniblock', 'sus', 'er']


    for sub in subjects: 
        print(f"Now working on subject {sub}")
        for runtype in runtypes:
            for smoothing in smooths: 

                results_glmsingle = dict()
                results_glmsingle['typed'] = np.load(join(outdir,"GLMSingle_Outputs",f'{smoothing}_sub-{sub}_{runtype}_TYPED_FITHRF_GLMDENOISE_RR.npy'), allow_pickle=True).item()
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
                rel_map_1,_ = pearsonr(betas_per_condition[:,:,:,0,:], betas_per_condition[:,:,:,1,:], axis=-1)
                rel_map_2,_ = pearsonr(betas_per_condition[:,:,:,2,:], betas_per_condition[:,:,:,3,:], axis=-1)
                rel_map_3,_ = pearsonr(betas_per_condition[:,:,:,4,:], betas_per_condition[:,:,:,5,:], axis=-1)

                # Save reliability map as NIfTI file
                sub_outdir = join(outdir, 'reliability', 'progression_analysis_within',f'sub-{sub}')
                os.makedirs(sub_outdir, exist_ok=True)
                reliability_img_1 = nib.Nifti1Image(rel_map_1, brain_mask.affine)
                reliability_filename_1 = join(outdir, 'reliability', 'progression_analysis_within', f'sub-{sub}', f'{smoothing}_sub-{sub}_{runtype}_reliability_map_run_1.nii.gz')
                nib.save(reliability_img_1, reliability_filename_1)

                reliability_img_2 = nib.Nifti1Image(rel_map_2, brain_mask.affine)
                reliability_filename_2 = join(outdir, 'reliability', 'progression_analysis_within', f'sub-{sub}', f'{smoothing}_sub-{sub}_{runtype}_reliability_map_run_2.nii.gz')
                nib.save(reliability_img_2, reliability_filename_2)

                reliability_img_3 = nib.Nifti1Image(rel_map_3, brain_mask.affine)
                reliability_filename_3 = join(outdir, 'reliability', 'progression_analysis_within', f'sub-{sub}', f'{smoothing}_sub-{sub}_{runtype}_reliability_map_run_3.nii.gz')
                nib.save(reliability_img_3, reliability_filename_3)

def gather_progession_within(project_dir, subjects, ROIs):
    """
    Gathers within run reliabilities for every ROI. Outputs a dataframe if all participants are included. 
    """

    outdir = join(project_dir, 'miniblock/Outputs/reliability/progression_analysis_within')
    anatdir = join(project_dir, 'miniblock/derivatives')
    smooths = ["unsmoothed", "sm_2_vox"]
    runtypes = ['miniblock', 'sus', 'er']
    runs = range(1,4)

    results = []

    for sub in subjects:
        for ROI in ROIs:
            # Load brain mask
            if ROI in ["FFA" , "PPA" , "EBA", "EVC"]:
                brain_mask_path = join(anatdir, f'sub-{sub}', 'anat', f'{ROI}_mask_sm_2_vox.nii.gz')
                brain_mask = image.load_img(brain_mask_path)
                mask = brain_mask.get_fdata()
            elif ROI in ["visually_responsive_voxels", "occipital_mask"]:
                brain_mask_path = join(anatdir, f'sub-{sub}', 'anat', f'{ROI}_sm_2_vox_gm.nii.gz')
                brain_mask = image.load_img(brain_mask_path)
                mask = brain_mask.get_fdata()

            for runtype in runtypes:
                for smoothing in smooths:
                    for run in runs:
                        # Construct filename
                        reliability_filename = join(
                            outdir, f'sub-{sub}', f'{smoothing}_sub-{sub}_{runtype}_reliability_map_run_{run}.nii.gz'
                        )

                        if os.path.exists(reliability_filename):
                            reliability_img = nib.load(reliability_filename)
                            reliability_data = reliability_img.get_fdata()


                            masked_values = reliability_data[mask.astype(bool)]
                            median_reliability = np.nanmedian(masked_values)
                            mean_reliability = np.nanmean(masked_values)
                    

                            results.append({
                            "subject": sub,
                            "runtype": runtype,
                            "smoothing": smoothing,
                            "median_rel": median_reliability,
                            "mean_rel": mean_reliability,
                            "ROI": ROI,
                            "run": run})
                        else:
                            print(f"Missing file: {reliability_filename}")

    # Build DataFrame
    df = pd.DataFrame(results)

    if len(subjects) == 20:
        df.to_csv(join(outdir, 'reliability_progression_within.csv'), index=False)

    return(df)

def noise_ceilings(project_dir, subjects):
    """
    Calculates the noise ceiling for every voxel similar to Allen et al. (2021) in the NSD paper. Again, for each participant,
    each design, and both smoothing options one value per voxel is calculated. 
    """        
    outdir = join(project_dir, 'miniblock/Outputs')
    datadir = join(project_dir, 'miniblock')
    presdir = join(project_dir, 'Behavior', 'designmats')
    smooths = ["unsmoothed", "sm_2_vox"]
    runtypes = ['miniblock', 'sus', 'er']


    for sub in subjects: 
        for runtype in runtypes:
            for smoothing in smooths: 

                results_glmsingle = dict()
                results_glmsingle['typed'] = np.load(join(outdir,"GLMSingle_Outputs",f'{smoothing}_sub-{sub}_{runtype}_TYPED_FITHRF_GLMDENOISE_RR.npy'), allow_pickle=True).item()
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

                # 1. Noise variance: average within-image variance
                variance_per_image = np.var(betas_per_condition, axis=3, ddof=1)  # over 6 repeats
                variance_noise = np.mean(variance_per_image, axis=3)              # over 40 images

                # 2. Total variance: variance across image means
                mean_betas_per_image = np.mean(betas_per_condition, axis=3)       # over 6 repeats
                variance_tot = np.var(mean_betas_per_image, axis=3, ddof=1)       # over 40 images

                # 3. Estimate signal variance
                variance_signal = variance_tot - (variance_noise / 6)
                variance_signal = np.clip(variance_signal, 0, None)               # avoid sqrt negatives

                # 4. Noise ceiling
                NC = np.sqrt(variance_signal) / np.sqrt(variance_tot)


                # Save reliability map as NIfTI file
                NC_img = nib.Nifti1Image(NC, brain_mask.affine)
                NC_filename = join(outdir, 'reliability', 'noise_ceilings', f'sub-{sub}', f'{smoothing}_sub-{sub}_{runtype}_noise_ceilings.nii.gz')
                sub_outdir = join(outdir, 'reliability','noise_ceilings', f'sub-{sub}')
                os.makedirs(sub_outdir, exist_ok=True)
                nib.save(NC_img, NC_filename)

def gather_noise_ceilings(project_dir, subjects, ROIs):
    """
    Gathers noise ceilings for every ROI. Outputs a dataframe if all participants are included. 
    """
    outdir = join(project_dir, 'miniblock/Outputs/reliability/noise_ceilings')
    anatdir = join(project_dir, 'miniblock/derivatives')
    smooths = ["unsmoothed", "sm_2_vox"]
    runtypes = ['miniblock', 'sus', 'er']

    results = []
    for sub in subjects: 
        for runtype in runtypes:
            for smoothing in smooths: 
                for ROI in ROIs:
                    if ROI == "visually_responsive_voxels":
                        mask_path = join(anatdir, f"sub-{sub}", "anat", f"{ROI}_sm_2_vox_gm.nii")
                    else:
                        mask_path = join(anatdir, f"sub-{sub}", "anat", f"{ROI}_mask_sm_2_vox.nii")
                    mask = image.load_img(mask_path).get_fdata()
                    
                    NC_filename = join(outdir, f'sub-{sub}', f'{smoothing}_sub-{sub}_{runtype}_noise_ceilings.nii.gz')
                    nc_data = image.load_img(NC_filename).get_fdata()

                    if nc_data.shape[:3] == mask.shape:
                        masked_values = nc_data[mask.astype(bool)]
                    else: 
                        print("Data shapes not compatible.")

                    median_nc = np.nanmedian(masked_values)
                    mean_nc = np.nanmean(masked_values)

                    results.append({
                                "subject": sub,
                                "runtype": runtype,
                                "smoothing": smoothing,
                                "median_nc": median_nc,
                                "mean_nc": mean_nc,
                                "ROI": ROI
                            })
                    
    results_df = pd.DataFrame(results)

    if len(subjects) == 20:
        results_df.to_csv(join(outdir, "df_noise_ceilings.csv"))

    return(results_df)

def group_results(dataframe):
    """
    Does repeated measures ANOVAs for every ROI and smoothing option the dataframe contains over the designs. 
    Afterwards, FDR-corrected post-hoc paired t-tests are calculated.
    A binary dataframe of the ROIs and smoothing options as rows and the pairwise comparisons between designs as colummns
    is returned:
    0 indicates not significant, 
    1 indicates significance
    """
    ROIs = ["visually_responsive_voxels", "FFA", "PPA", "EBA", "EVC"]
    smooths = ["sm_2_vox", "unsmoothed"]

    significant_df = []

    for ROI in ROIs:
        for smoothing in smooths:
            df = dataframe[dataframe["ROI"] == ROI]

            df = (
                df[df["smoothing"] == smoothing]
                .groupby(["subject", "runtype"], as_index=False)["median_nc"]
                .mean()
            )

            # Run repeated-measures ANOVA
            aov = AnovaRM(data=df, depvar='median_nc', subject='subject', within=["runtype"])
            aov_res = aov.fit()

            # Pairwise comparisons
            runtype_conditions = df['runtype'].unique()
            comparisons = list(combinations(runtype_conditions, 2))

            pvals = []
            results = []

            df_wide = df.pivot(index='subject', columns='runtype', values='median_nc')

            for cond1, cond2 in comparisons:
                t_stat, p_val = ttest_rel(df_wide[cond1], df_wide[cond2])
                pvals.append(p_val)
                results.append((cond1, cond2, t_stat, p_val))

            _, pvals_corrected, _, _ = smm.multipletests(pvals, method='fdr_bh')

            # Build row dict
            row = {"ROI": ROI, "smoothing": smoothing}

            for i, (cond1, cond2, t_stat, p_val) in enumerate(results):
                label = f"{cond1}_vs_{cond2}"
                row[f"{label}_t"] = t_stat
                row[f"{label}_p_uncorrected"] = p_val
                row[f"{label}_p_corrected"] = pvals_corrected[i]

            significant_df.append(row)

    # Convert to DataFrame
    significant_df = pd.DataFrame(significant_df)
    significant_df = significant_df[["ROI", "smoothing", "er_vs_miniblock_p_corrected", "er_vs_sus_p_corrected", "miniblock_vs_sus_p_corrected"]]
    # List of p-value columns
    pval_cols = ["er_vs_miniblock_p_corrected", "er_vs_sus_p_corrected", "miniblock_vs_sus_p_corrected"]

    # Create a copy to avoid modifying the original
    binary_df = significant_df.copy()

    # Replace p-value columns with 1 if p < 0.05 else 0
    binary_df[pval_cols] = (binary_df[pval_cols] < 0.05).astype(int)

    return binary_df