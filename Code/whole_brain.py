import nibabel as nib
import numpy as np
import os
from nilearn import plotting 
import matplotlib.pyplot as plt
from os.path import join
from nilearn import datasets, surface, plotting
import seaborn as sns
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import ttest_rel, ttest_ind

def create_participant_images(analysis, subjects, project_dir):
    """
    For the whole-brain reliability and RSA analysis the 10 different whole-brain maps have to be averaged and saved first 
    to access these files later and use them in voxel-wise t-tests.
    """
    smooths = ["unsmoothed", "sm_2_vox"]
    designs = ["sus", "miniblock", "er"]

    if analysis == "reliability":
        splits = range(1,11)
        outdir = join(project_dir, 'miniblock/Outputs/whole_brain/reliability')
        base_dir = join(project_dir, 'miniblock/Outputs/reliability')
        # Load the NIfTI files
        for smoothing in smooths: 
            for subj in subjects:
                subj_maps = {'sus': [], 'miniblock': [], 'er': []}
                for design in designs:
                    for split_num in splits: 
                        path = os.path.join(base_dir, f'sub-{subj}', f'{smoothing}_sub-{subj}_{design}_reliability_map_split_{split_num}.nii.gz')
                        nii = nib.load(path)
                        subj_maps[design].append(nii.get_fdata())  

                subj_maps_sus = np.stack(subj_maps['sus'], axis=-1)         # (x, y, z, nSubjects)
                subj_maps_miniblock = np.stack(subj_maps['miniblock'], axis=-1)
                subj_maps_er = np.stack(subj_maps['er'], axis=-1)

                mean_sus = subj_maps_sus.mean(axis=-1)
                mean_miniblock = subj_maps_miniblock.mean(axis=-1)
                mean_er = subj_maps_er.mean(axis=-1)

                miniblock_nifti = nib.Nifti1Image(mean_miniblock, affine=nii.affine)
                sus_nifti = nib.Nifti1Image(mean_sus, affine=nii.affine)
                er_nifti = nib.Nifti1Image(mean_er, affine=nii.affine)

                miniblock_outdir = join(outdir, f'{smoothing}', f'sub-{subj}')
                os.makedirs(miniblock_outdir, exist_ok=True)

                sus_outdir = join(outdir, f'{smoothing}', f'sub-{subj}')
                os.makedirs(sus_outdir, exist_ok=True)

                er_outdir = join(outdir, f'{smoothing}', f'sub-{subj}')
                os.makedirs(er_outdir, exist_ok=True)
                nib.save(miniblock_nifti, join(outdir, f'{smoothing}', f'sub-{subj}', f'{smoothing}_sub-{subj}_miniblock.nii'))
                nib.save(sus_nifti, join(outdir, f'{smoothing}', f'sub-{subj}', f'{smoothing}_sub-{subj}_sus.nii'))
                nib.save(er_nifti, join(outdir, f'{smoothing}', f'sub-{subj}', f'{smoothing}_sub-{subj}_er.nii'))

    elif analysis == "RSA":
        # Load the NIfTI files
        splits = range(1,11)
        outdir = join(project_dir, 'miniblock/Outputs/RSA/searchlight')
        base_dir = join(project_dir, 'miniblock/Outputs/RSA/searchlight')

        for smoothing in smooths: 
            for subj in subjects:
                subj_maps = {'sus': [], 'miniblock': [], 'er': []}
                for design in designs:
                    for split_num in splits: 
                        path = os.path.join(base_dir, f'sub-{subj}', f'{smoothing}_sub-{subj}_rsa_map_{design}_split_{split_num}.nii.gz')
                        nii = nib.load(path)
                        subj_maps[design].append(nii.get_fdata())  

                subj_maps_sus = np.stack(subj_maps['sus'], axis=-1)         # (x, y, z, nSubjects)
                subj_maps_miniblock = np.stack(subj_maps['miniblock'], axis=-1)
                subj_maps_er = np.stack(subj_maps['er'], axis=-1)

                mean_sus = subj_maps_sus.mean(axis=-1)
                mean_miniblock = subj_maps_miniblock.mean(axis=-1)
                mean_er = subj_maps_er.mean(axis=-1)

                miniblock_nifti = nib.Nifti1Image(mean_miniblock, affine=nii.affine)
                sus_nifti = nib.Nifti1Image(mean_sus, affine=nii.affine)
                er_nifti = nib.Nifti1Image(mean_er, affine=nii.affine)

                miniblock_outdir = join(outdir,  f'sub-{subj}')
                os.makedirs(miniblock_outdir, exist_ok=True)

                sus_outdir = join(outdir,  f'sub-{subj}')
                os.makedirs(sus_outdir, exist_ok=True)

                er_outdir = join(outdir,  f'sub-{subj}')
                os.makedirs(er_outdir, exist_ok=True)
                nib.save(miniblock_nifti, join(outdir, f'sub-{subj}', f'{smoothing}_sub-{subj}_miniblock.nii'))
                nib.save(sus_nifti, join(outdir, f'sub-{subj}', f'{smoothing}_sub-{subj}_sus.nii'))
                nib.save(er_nifti, join(outdir, f'sub-{subj}', f'{smoothing}_sub-{subj}_er.nii'))

    else: 
        print("Please specify an analysis: reliability, RSA")

def whole_brain_reliability(subjects, project_dir, design1, design2, alternative = "greater"):
    """
    Function that runs a whole-brain reliability analysis between two specified design (design1, design2). These must be either sus, er or miniblock. 
    The participant-wise maps created by create_participants_images are averaged over all participants per voxel. Finally, 
    voxel-wise t-tests are calculated between the two specified designs and FDR-corrected. Please speficy the direction of the comparison.
    If nothing is specified, the alternative hypothesis is set to greater, meaning the first sample (design) is assumed to be
    greater than the secondThe t-values are then saved as a whole-brain map, if any voxels survived FDR-correction. Does not
    return something, except for printing the number of significant 
    voxels for this comparison. 
    """
    designs = ["er", "miniblock", "sus"]
    smooths = ["unsmoothed", "sm_2_vox"]
    base_dir = join(project_dir, 'miniblock/Outputs/whole_brain/reliability')
    design_maps = {'sus': [], 'miniblock': [], 'er': []}
    # Load the NIfTI files
    for smoothing in smooths: 
        for subj in subjects:
            for design in designs:
                path = os.path.join(base_dir, smoothing,f'sub-{subj}', f'{smoothing}_sub-{subj}_{design}.nii')
                nii = nib.load(path)
                affine = nib.load(path).affine
                design_maps[design].append(nii.get_fdata())  

    maps1 = np.stack(design_maps[design1], axis=0)  # shape: (n_subj, x, y, z)
    maps2 = np.stack(design_maps[design2], axis=0)  # same shape

    t_vals, p_vals = ttest_rel(maps1, maps2, axis=0, alternative = alternative)

    # Flatten the p-values (excluding NaNs)
    p_vals_flat = p_vals.flatten()
    valid_mask = ~np.isnan(p_vals_flat)

    # Apply FDR correction only on valid voxels
    fdr_pass, fdr_corrected_pvals = fdrcorrection(p_vals_flat[valid_mask], alpha=0.05)

    # Create a full-size array of zeros
    fdr_mask_flat = np.zeros_like(p_vals_flat, dtype=np.uint8)

    # Set passing voxels to 1
    fdr_mask_flat[valid_mask] = fdr_pass.astype(np.uint8)

    # Reshape back to 3D
    affine = nib.load(path).affine  # path from one of your NIfTI files
    fdr_mask = fdr_mask_flat.reshape(p_vals.shape)
    fdr_mask_img = nib.Nifti1Image(fdr_mask, affine)
    t_img = nib.Nifti1Image(t_vals, affine)

    # Get the highest uncorrected p-value that survives FDR
    significant_pvals = p_vals_flat[valid_mask][fdr_pass]
    fdr_threshold_p = np.max(significant_pvals)

    if np.sum(fdr_mask) > 0:
        plotting.view_img(fdr_mask_img)
        print(np.sum(fdr_mask))
    else: 
        print("No significant voxels")

    fsaverage = datasets.fetch_surf_fsaverage()
    surf_data = surface.vol_to_surf(nii, fsaverage['pial_left'], interpolation = "nearest")  
    bg_maps = {
        "left": fsaverage['sulc_left'],
        "right": fsaverage['sulc_right']
    }
    views = ['medial', 'ventral', 'lateral']
    hemispheres = ['left', 'right']

    design_dict = {
        "er": "Event-Related",
        "miniblock": "Miniblock",
        "sus": "Sustained"
    }

    cmap = "RdBu_r"
    # Set up matplotlib figure
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 7),
                            subplot_kw={'projection': '3d'})
    fig.suptitle(f"Surface Reliability Maps: {design_dict[design1]} vs {design_dict[design2]}", fontsize=16)
    plt.rcParams['font.family'] = 'Helvetica'
    for row, hemi in enumerate(hemispheres):
        for col, view in enumerate(views):
            ax = axes[row, col]

            mesh = fsaverage[f'infl_{hemi}']
            surf_values = surface.vol_to_surf(t_img, fsaverage[f'pial_{hemi}'])

            plotting.plot_surf_stat_map(
                mesh,
                surf_values,
                hemi=hemi,
                view=view,
                bg_map=bg_maps[hemi],
                colorbar=True,
                figure=fig,
                axes=ax,
                vmax = 10, 
                #vmin = fdr_threshold_t,
                title=f'{view.capitalize()} - {hemi.capitalize()}',
                cmap = cmap, 
                threshold = 2
            )

    if np.sum(fdr_mask) > 0:
        # Save figure
        output_path = join(project_dir, f"miniblock/Outputs/whole_brain/reliability/surface_plot_{design1}_vs_{design2}.png")
        plt.savefig(output_path, dpi=300)

def whole_brain_decoding(subjects, project_dir, design1, design2, alternative = "greater"):
    smooths = ["unsmoothed", "sm_2_vox"]
    designs = ["sus", "miniblock", "er"]
    design_maps = {'sus': [], 'miniblock': [], 'er': []}
    base_dir = join(project_dir, "miniblock/Outputs/decoding/searchlight/pairwise")
    # Load the NIfTI files
    for smoothing in smooths: 
        for subj in subjects:
            for design in designs:
                path = os.path.join(base_dir, design, f'sub-{subj}', f'res_accuracy_pairwise_minus_chance.nii')
                nii = nib.load(path)
                affine = nib.load(path).affine
                design_maps[design].append(nii.get_fdata())  

    
    maps1 = np.stack(design_maps[design1], axis=0)  # shape: (n_subj, x, y, z)
    maps2 = np.stack(design_maps[design2], axis=0)  # same shape

    t_vals, p_vals = ttest_rel(maps1, maps2, axis=0, alternative = alternative)

    # Flatten the p-values (excluding NaNs)
    p_vals_flat = p_vals.flatten()
    valid_mask = ~np.isnan(p_vals_flat)

    # Apply FDR correction only on valid voxels
    fdr_pass, fdr_corrected_pvals = fdrcorrection(p_vals_flat[valid_mask], alpha=0.05)

    # Create a full-size array of zeros
    fdr_mask_flat = np.zeros_like(p_vals_flat, dtype=np.uint8)

    # Set passing voxels to 1
    fdr_mask_flat[valid_mask] = fdr_pass.astype(np.uint8)

    # Reshape back to 3D
    affine = nib.load(path).affine  # path from one of your NIfTI files
    fdr_mask = fdr_mask_flat.reshape(p_vals.shape)
    fdr_mask_img = nib.Nifti1Image(fdr_mask, affine)
    t_img = nib.Nifti1Image(t_vals, affine)

    # Get the highest uncorrected p-value that survives FDR
    significant_pvals = p_vals_flat[valid_mask][fdr_pass]
    fdr_threshold_p = np.max(significant_pvals)

    np.sum(fdr_mask)
    if np.sum(fdr_mask) > 0:
        plotting.view_img(fdr_mask_img)
        print(np.sum(fdr_mask))
    else: 
        print("No significant voxels")
    
    fsaverage = datasets.fetch_surf_fsaverage()
    surf_data = surface.vol_to_surf(nii, fsaverage['pial_left'], interpolation = "nearest")  
    bg_maps = {
        "left": fsaverage['sulc_left'],
        "right": fsaverage['sulc_right']
    }
    views = ['medial', 'ventral', 'lateral']
    hemispheres = ['left', 'right']

    design_dict = {
        "er": "Event-Related",
        "miniblock": "Miniblock",
        "sus": "Sustained"
    }

    cmap = "RdBu_r"
    # Set up matplotlib figure
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 7),
                            subplot_kw={'projection': '3d'})
    fig.suptitle(f"Surface Decoding Accuracy t-value Maps: {design_dict[design1]} vs {design_dict[design2]}", fontsize=16)
    plt.rcParams['font.family'] = 'Helvetica'
    for row, hemi in enumerate(hemispheres):
        for col, view in enumerate(views):
            ax = axes[row, col]

            mesh = fsaverage[f'infl_{hemi}']
            surf_values = surface.vol_to_surf(t_img, fsaverage[f'pial_{hemi}'])

            plotting.plot_surf_stat_map(
                mesh,
                surf_values,
                hemi=hemi,
                view=view,
                bg_map=bg_maps[hemi],
                colorbar=True,
                figure=fig,
                axes=ax,
                vmax = 14, 
                #vmin = fdr_threshold_t,
                title=f'{view.capitalize()} - {hemi.capitalize()}',
                cmap = cmap, 
                threshold = 3
            )

    if np.sum(fdr_mask) > 0:
        # Save figure
        os.makedirs(join(project_dir, f"miniblock/Outputs/whole_brain/decoding"), exist_ok=True)
        output_path = join(project_dir, f"miniblock/Outputs/whole_brain/decoding/decoding_surface_plot_{design1}_vs_{design2}.png")
        plt.savefig(output_path, dpi=300)

def whole_brain_RSA(subjects, project_dir, design1, design2, alternative = "greater"):
    smooths = ["unsmoothed", "sm_2_vox"]
    designs = ["sus", "miniblock", "er"]
    base_dir = join(project_dir, "miniblock/Outputs/RSA/searchlight")

    design_maps = {'sus': [], 'miniblock': [], 'er': []}
    # Load the NIfTI files
    for smoothing in smooths: 
        for subj in subjects:
            for design in designs:
                path = os.path.join(base_dir, f'sub-{subj}', f'{smoothing}_sub-{subj}_{design}.nii')
                nii = nib.load(path)
                affine = nib.load(path).affine
                design_maps[design].append(nii.get_fdata())  

    maps1 = np.stack(design_maps[design1], axis=0)  # shape: (n_subj, x, y, z)
    maps2 = np.stack(design_maps[design2], axis=0)  # same shape

    t_vals, p_vals = ttest_rel(maps1, maps2, axis=0, alternative = alternative)

    # Flatten the p-values (excluding NaNs)
    p_vals_flat = p_vals.flatten()
    valid_mask = ~np.isnan(p_vals_flat)

    # Apply FDR correction only on valid voxels
    fdr_pass, fdr_corrected_pvals = fdrcorrection(p_vals_flat[valid_mask], alpha=0.05)

    # Create a full-size array of zeros
    fdr_mask_flat = np.zeros_like(p_vals_flat, dtype=np.uint8)

    # Set passing voxels to 1
    fdr_mask_flat[valid_mask] = fdr_pass.astype(np.uint8)

    # Reshape back to 3D
    affine = nib.load(path).affine  # path from one of your NIfTI files
    fdr_mask = fdr_mask_flat.reshape(p_vals.shape)
    fdr_mask_img = nib.Nifti1Image(fdr_mask, affine)
    t_img = nib.Nifti1Image(t_vals, affine)

    # Get the highest uncorrected p-value that survives FDR
    significant_pvals = p_vals_flat[valid_mask][fdr_pass]
    #fdr_threshold_p = np.max(significant_pvals)

    np.sum(fdr_mask)
    if np.sum(fdr_mask) > 0:
        plotting.view_img(fdr_mask_img)
        print(np.sum(fdr_mask))
    else: 
        print("No significant voxels")
    
    fsaverage = datasets.fetch_surf_fsaverage()
    surf_data = surface.vol_to_surf(nii, fsaverage['pial_left'], interpolation = "nearest")  
    bg_maps = {
        "left": fsaverage['sulc_left'],
        "right": fsaverage['sulc_right']
    }
    views = ['medial', 'ventral', 'lateral']
    hemispheres = ['left', 'right']

    design_dict = {
        "er": "Event-Related",
        "miniblock": "Miniblock",
        "sus": "Sustained"
    }

    cmap = "RdBu_r"
    # Set up matplotlib figure
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 7),
                         subplot_kw={'projection': '3d'})
    fig.suptitle(f"Surface RSA Maps: {design_dict[design1]} vs {design_dict[design2]}", fontsize=16)
    plt.rcParams['font.family'] = 'Helvetica'
    for row, hemi in enumerate(hemispheres):
        for col, view in enumerate(views):
            ax = axes[row, col]

            mesh = fsaverage[f'infl_{hemi}']
            surf_values = surface.vol_to_surf(t_img, fsaverage[f'pial_{hemi}'])

            plotting.plot_surf_stat_map(
                mesh,
                surf_values,
                hemi=hemi,
                view=view,
                bg_map=bg_maps[hemi],
                colorbar=True,
                figure=fig,
                axes=ax,
                vmax = 9, 
                #vmin = fdr_threshold_t,
                title=f'{view.capitalize()} - {hemi.capitalize()}',
                cmap = cmap, 
                threshold = 2
            )

    if np.sum(fdr_mask) > 0:
        # Save figure
        output_path = join(project_dir, f"miniblock/Outputs/whole_brain/RSA/rsa_surface_plot_{design1}_vs_{design2}.png")
        #plt.tight_layout(rect=[0.2, 0.03, 0.8, 0.95])
        plt.savefig(output_path, dpi=300)
