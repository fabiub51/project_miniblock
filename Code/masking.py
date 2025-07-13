import os
from os.path import join
import gzip
import shutil
from nilearn.image import load_img, resample_img
from nilearn import image
import numpy as np
from statsmodels.stats.multitest import fdrcorrection
import pandas as pd
import nibabel as nib
import glob
from scipy.stats import ttest_1samp, false_discovery_control, ttest_rel
from IPython.display import display
from statsmodels.stats.multitest import fdrcorrection
from nilearn.image import math_img, new_img_like
from nilearn.regions import connected_regions
from nilearn.image import index_img
from scipy.ndimage import center_of_mass

def unzip_file(gz_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    print(f"Unzipped: {gz_path} -> {output_path}")

def resample_and_save_mask(
    project_dir,
    input_mask_path, 
    output_mask_path,
    target_affine=None,
    target_shape=(77, 95, 82),
    interpolation='nearest'
):
    """
    Resample a binary brain mask to a new resolution and save it.

    Parameters
    ----------
    input_mask_path : str
        Path to the input NIfTI mask file.
    output_mask_path : str
        Path where the resampled mask will be saved.
    target_affine : np.ndarray, optional
        New affine matrix. If None, defaults to MNI 2mm.
    target_shape : tuple of int
        Desired shape of the output image.
    interpolation : str
        Interpolation type ('nearest' for masks).
    """
    if target_affine is None:
        target_affine = image.load_img(join(project_dir, 'miniblock/derivatives/sub-01/func/sub-01_task-func_run-01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')).affine

    # Load the mask
    mask_img = load_img(input_mask_path)

    # Resample the mask
    resampled_mask = resample_img(
        mask_img,
        target_affine=target_affine,
        target_shape=target_shape,
        interpolation=interpolation
    )

    # Save the resampled mask
    resampled_mask.to_filename(output_mask_path)
    print(f"Resampled mask saved to: {output_mask_path}")

def get_group_mask(project_dir):
    subs = ["02", "03","04", "05","06", "07", "08", "10", "11", "12", "13", "14", "15", "17", "18", "19", "20", "21", "22"]
    outdir = join(project_dir, 'miniblock/Outputs/')
    datadir = join(project_dir, "miniblock")
    brain_mask = image.load_img(join(datadir, 'derivatives', 'sub-01', 'anat', 'sub-01_space-MNI152NLin2009cAsym_desc-brain_mask_resampled.nii.gz'))
    mask = brain_mask.get_fdata()

    for sub in subs: 
        new_brain_mask = image.load_img(join(datadir, 'derivatives', f'sub-{sub}', 'anat', f'sub-{sub}_space-MNI152NLin2009cAsym_desc-brain_mask_resampled.nii.gz'))
        new_mask = new_brain_mask.get_fdata()
        mask = np.logical_and(mask, new_mask)

    group_mask = nib.Nifti1Image(mask.astype(np.float32), brain_mask.affine)
    os.makedirs(join(outdir, "masking"),exist_ok=True)
    nib.save(group_mask, join(outdir,"masking/group_mask.nii"))

def get_visually_responsive_voxels(project_dir, subjects):
    outdir = join(project_dir, 'miniblock/Outputs')
    datadir = join(project_dir, 'miniblock')
    smooths = ['sm_2_vox']

    for sub in subjects: 
        for smoothing in smooths: 
            results_glmsingle = dict()
            results_glmsingle['typed'] = np.load(join(outdir,'GLMSingle_Outputs','localizer',f'{smoothing}_sub-{sub}_TYPED_FITHRF_GLMDENOISE_RR.npy'), allow_pickle=True).item()
            betas = results_glmsingle['typed']['betasmd']

            gm_data = nib.load(join(datadir, 'derivatives', f'sub-{sub}', 'anat', f'sub-{sub}_space-MNI152NLin2009cAsym_label-GM_probseg_resampled.nii.gz')).get_fdata()
            whole_brain_mask = image.load_img(join(datadir, 'derivatives', f'sub-{sub}', 'anat', f'sub-{sub}_space-MNI152NLin2009cAsym_desc-brain_mask_resampled.nii.gz')).get_fdata()
            
            # Threshold GM mask (e.g., > 0.3 probability of being gray matter)
            gm_mask = gm_data > 0.3

            # Combine both masks: only keep voxels within the brain *and* in GM
            combined_mask = np.logical_and(whole_brain_mask, gm_mask)
            betas_masked = betas[combined_mask, :]  
            t_vals, p_vals = ttest_1samp(betas_masked, axis=1, popmean=0)

            corrected_p_vals = false_discovery_control(p_vals, method = "bh")

            significant_voxels = (corrected_p_vals < 0.05)
            significant_whole_brain = np.zeros_like(whole_brain_mask, dtype=bool)
            significant_whole_brain[combined_mask] = significant_voxels

            visual_mask_path = "/Users/danieljanini/Documents/Thesis/Code/masking/visually_responsive_voxels.nii.gz"
            visual_mask_path = join(project_dir, "Code/JuBrain_masks/VRV_resampled.nii")
            visual_mask = image.load_img(visual_mask_path)
            visual_mask = visual_mask.get_fdata()

            # Combine occipital and t-value-mask
            filtered_mask = significant_whole_brain & (visual_mask > 0)

            filtered_mask_img = nib.Nifti1Image(filtered_mask.astype(np.uint8), image.load_img(join(datadir, 'derivatives', f'sub-{sub}', 'func', f'sub-{sub}_task-func_run-01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')).affine)
            nib.save(filtered_mask_img, join(datadir, 'derivatives', f'sub-{sub}','anat',f"visually_responsive_voxels_{smoothing}_gm.nii"))

def get_evc_mask(project_dir, subjects):
    outdir = join(project_dir, 'miniblock/Outputs')
    datadir = join(project_dir, 'miniblock')
    smooths = ['sm_2_vox']

    for sub in subjects: 
        for smoothing in smooths: 
            results_glmsingle = dict()
            results_glmsingle['typed'] = np.load(join(outdir,'GLMSingle_Outputs','localizer',f'{smoothing}_sub-{sub}_TYPED_FITHRF_GLMDENOISE_RR.npy'), allow_pickle=True).item()
            betas = results_glmsingle['typed']['betasmd']

            whole_brain_mask = image.load_img(join(datadir, 'derivatives', f'sub-{sub}', 'anat', f'sub-{sub}_space-MNI152NLin2009cAsym_desc-brain_mask_resampled.nii.gz')).get_fdata()
            whole_brain_mask = whole_brain_mask > 0  
            betas_masked = betas[whole_brain_mask, :]  
            t_vals, p_vals = ttest_1samp(betas_masked, axis=1, popmean=0)

            corrected_p_vals = false_discovery_control(p_vals, method = "bh")

            significant_voxels = (corrected_p_vals < 0.05)
            significant_whole_brain = np.zeros_like(whole_brain_mask, dtype=bool)
            significant_whole_brain[whole_brain_mask] = significant_voxels

            evc_mask_path = join(project_dir, "Code/JuBrain_masks/early_visual_cortex_resampled.nii")
            evc_mask = image.load_img(evc_mask_path)
            evc_mask = evc_mask.get_fdata()

            # Combine occipital and t-value-mask
            filtered_mask = significant_whole_brain & (evc_mask > 0)

            filtered_mask_img = nib.Nifti1Image(filtered_mask.astype(np.uint8), image.load_img(join(datadir, 'derivatives', f'sub-{sub}', 'func', f'sub-{sub}_task-func_run-01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')).affine)
            nib.save(filtered_mask_img, join(datadir, 'derivatives', f'sub-{sub}','anat',f"EVC_mask_{smoothing}.nii"))

def get_FFA_mask(project_dir, subjects):
    outdir = join(project_dir, 'miniblock/Outputs')
    datadir = join(project_dir, 'miniblock')
    presdir = join(project_dir, "Behavior/designmats")
    smooths = ['sm_2_vox']
    categories = ["Faces", "Objects", "Scenes", "Bodies"]
    ROI = "FFA"

    for sub in subjects: 
        for smoothing in smooths: 
            results_glmsingle = dict()
            results_glmsingle['typed'] = np.load(join(outdir,'GLMSingle_Outputs','localizer',f'{smoothing}_sub-{sub}_TYPED_FITHRF_GLMDENOISE_RR.npy'), allow_pickle=True).item()
            betas = results_glmsingle['typed']['betasmd']

            whole_brain_mask_initial = image.load_img(join(datadir, 'derivatives', f'sub-{sub}', 'anat', f'sub-{sub}_space-MNI152NLin2009cAsym_desc-brain_mask_resampled.nii.gz'))
            whole_brain_mask = whole_brain_mask_initial.get_fdata()         

            masked_betas = betas[whole_brain_mask.astype(bool)]
            unmasked_betas = np.zeros(betas.shape)
            unmasked_betas[whole_brain_mask.astype(bool)] = masked_betas

            pattern = presdir + f'/localizer/P0{sub}_CategoryLocalizer_Run*.csv'
            matches = glob.glob(pattern)
            matches.sort()
            
            design = []
            for i in range(len(matches)):
                designMat = pd.read_csv(matches[i], header=None)
                design.append(designMat)

            full_design = np.vstack(design)
            faces_idx = np.where(full_design[:, categories.index("Faces")] == 1)[0]
            objects_idx = np.where(full_design[:, categories.index("Objects")] == 1)[0]
            scenes_idx = np.where(full_design[:, categories.index("Scenes")] == 1)[0]
            bodies_idx = np.where(full_design[:, categories.index("Bodies")] == 1)[0]


            event_list = []
            event_list += [(time, 'Faces') for time in faces_idx]
            event_list += [(time, 'Objects') for time in objects_idx]
            event_list += [(time, 'Scenes') for time in scenes_idx]
            event_list += [(time, 'Bodies') for time in bodies_idx]

            # Sort them by time
            event_list.sort(key=lambda x: x[0])

            # Now get the condition labels in presentation order
            conditions_in_order = [condition for (_, condition) in event_list]

            conditions_in_order = np.array(conditions_in_order)

            trial_indices = {
                'Faces': np.where(conditions_in_order == 'Faces')[0],
                'Objects': np.where(conditions_in_order == 'Objects')[0],
                'Scenes': np.where(conditions_in_order == 'Scenes')[0],
                'Bodies': np.where(conditions_in_order == 'Bodies')[0],
            }

            # Define contrast depending on ROI
            contrast_def = {
                "FFA": ("Faces", "Objects")
            }

            cond1, cond2 = contrast_def[ROI]
            cond1_betas = unmasked_betas[..., trial_indices[cond1]]
            cond2_betas = unmasked_betas[..., trial_indices[cond2]]

            # Paired t-test
            tvals, pvals = ttest_rel(cond1_betas, cond2_betas, axis=-1, alternative="greater")
            pvals_flat = pvals.flatten()
            not_nan_mask = ~np.isnan(pvals_flat)

            # Run FDR only on valid values
            _, pvals_corrected_flat = fdrcorrection(pvals_flat[not_nan_mask])

            # Prepare an array filled with nan, then fill the valid entries
            pvals_corrected_full = np.full_like(pvals_flat, np.nan)
            pvals_corrected_full[not_nan_mask] = pvals_corrected_flat

            # Reshape back
            pvals_corrected = pvals_corrected_full.reshape(pvals.shape)
            significant_mask = pvals_corrected < 0.05
            tvals_thresholded = tvals * significant_mask

            
            fusiform_mask = image.load_img(join(project_dir, 'Code/JuBrain_masks/fusiform_gyrus_resampled.nii')).get_fdata()
            masked_tvals = tvals_thresholded * fusiform_mask

            binary_mask = (masked_tvals > 0).astype(np.uint8)
            contrast_img = nib.Nifti1Image(binary_mask, affine=whole_brain_mask_initial.affine)

            nib.save(contrast_img, join(datadir, "derivatives", f"sub-{sub}", "anat", f"{ROI}_mask_{smoothing}.nii"))
            print("Saved mask.")

def get_PPA_mask(project_dir, subjects):
    outdir = join(project_dir, 'miniblock/Outputs')
    datadir = join(project_dir, 'miniblock')
    presdir = join(project_dir, "Behavior/designmats")
    smooths = ['sm_2_vox']
    categories = ["Faces", "Objects", "Scenes", "Bodies"]
    ROI = "PPA"

    for sub in subjects: 
        for smoothing in smooths: 
            results_glmsingle = dict()
            results_glmsingle['typed'] = np.load(join(outdir,'GLMSingle_Outputs','localizer',f'{smoothing}_sub-{sub}_TYPED_FITHRF_GLMDENOISE_RR.npy'), allow_pickle=True).item()
            betas = results_glmsingle['typed']['betasmd']

            whole_brain_mask_initial = image.load_img(join(datadir, 'derivatives', f'sub-{sub}', 'anat', f'sub-{sub}_space-MNI152NLin2009cAsym_desc-brain_mask_resampled.nii.gz'))
            whole_brain_mask = whole_brain_mask_initial.get_fdata()         

            masked_betas = betas[whole_brain_mask.astype(bool)]
            unmasked_betas = np.zeros(betas.shape)
            unmasked_betas[whole_brain_mask.astype(bool)] = masked_betas

            pattern = presdir + f'/localizer/P0{sub}_CategoryLocalizer_Run*.csv'
            matches = glob.glob(pattern)
            matches.sort()
            
            design = []
            for i in range(len(matches)):
                designMat = pd.read_csv(matches[i], header=None)
                design.append(designMat)

            full_design = np.vstack(design)
            faces_idx = np.where(full_design[:, categories.index("Faces")] == 1)[0]
            objects_idx = np.where(full_design[:, categories.index("Objects")] == 1)[0]
            scenes_idx = np.where(full_design[:, categories.index("Scenes")] == 1)[0]
            bodies_idx = np.where(full_design[:, categories.index("Bodies")] == 1)[0]


            event_list = []
            event_list += [(time, 'Faces') for time in faces_idx]
            event_list += [(time, 'Objects') for time in objects_idx]
            event_list += [(time, 'Scenes') for time in scenes_idx]
            event_list += [(time, 'Bodies') for time in bodies_idx]

            # Sort them by time
            event_list.sort(key=lambda x: x[0])

            # Now get the condition labels in presentation order
            conditions_in_order = [condition for (_, condition) in event_list]

            conditions_in_order = np.array(conditions_in_order)

            trial_indices = {
                'Faces': np.where(conditions_in_order == 'Faces')[0],
                'Objects': np.where(conditions_in_order == 'Objects')[0],
                'Scenes': np.where(conditions_in_order == 'Scenes')[0],
                'Bodies': np.where(conditions_in_order == 'Bodies')[0],
            }

            # Define contrast depending on ROI
            contrast_def = {
                "PPA": ("Scenes", "Objects")
            }

            cond1, cond2 = contrast_def[ROI]
            cond1_betas = unmasked_betas[..., trial_indices[cond1]]
            cond2_betas = unmasked_betas[..., trial_indices[cond2]]

            # Paired t-test
            tvals, pvals = ttest_rel(cond1_betas, cond2_betas, axis=-1, alternative="greater")
            pvals_flat = pvals.flatten()
            not_nan_mask = ~np.isnan(pvals_flat)

            # Run FDR only on valid values
            _, pvals_corrected_flat = fdrcorrection(pvals_flat[not_nan_mask])

            # Prepare an array filled with nan, then fill the valid entries
            pvals_corrected_full = np.full_like(pvals_flat, np.nan)
            pvals_corrected_full[not_nan_mask] = pvals_corrected_flat

            # Reshape back
            pvals_corrected = pvals_corrected_full.reshape(pvals.shape)
            significant_mask = pvals_corrected < 0.05
            tvals_thresholded = tvals * significant_mask

            
            parahippocampal_mask = image.load_img(join(project_dir, 'Code/JuBrain_masks/parahippocampal_mask_resampled.nii')).get_fdata()
            masked_tvals = tvals_thresholded * parahippocampal_mask

            binary_mask = (masked_tvals > 0).astype(np.uint8)
            contrast_img = nib.Nifti1Image(binary_mask, affine=whole_brain_mask_initial.affine)

            nib.save(contrast_img, join(datadir, "derivatives", f"sub-{sub}", "anat", f"{ROI}_mask_{smoothing}.nii"))

def get_EBA_mask(project_dir, subjects):
    outdir = join(project_dir, 'miniblock/Outputs')
    datadir = join(project_dir, 'miniblock')
    presdir = join(project_dir, "Behavior/designmats")
    smooths = ['sm_2_vox']
    categories = ["Faces", "Objects", "Scenes", "Bodies"]
    ROI = "EBA"

    for sub in subjects: 
        for smoothing in smooths: 
            results_glmsingle = dict()
            results_glmsingle['typed'] = np.load(join(outdir,'GLMSingle_Outputs','localizer',f'{smoothing}_sub-{sub}_TYPED_FITHRF_GLMDENOISE_RR.npy'), allow_pickle=True).item()
            betas = results_glmsingle['typed']['betasmd']

            whole_brain_mask_initial = image.load_img(join(datadir, 'derivatives', f'sub-{sub}', 'anat', f'sub-{sub}_space-MNI152NLin2009cAsym_desc-brain_mask_resampled.nii.gz'))
            whole_brain_mask = whole_brain_mask_initial.get_fdata()         

            masked_betas = betas[whole_brain_mask.astype(bool)]
            unmasked_betas = np.zeros(betas.shape)
            unmasked_betas[whole_brain_mask.astype(bool)] = masked_betas

            pattern = presdir + f'/localizer/P0{sub}_CategoryLocalizer_Run*.csv'
            matches = glob.glob(pattern)
            matches.sort()
            
            design = []
            for i in range(len(matches)):
                designMat = pd.read_csv(matches[i], header=None)
                design.append(designMat)

            full_design = np.vstack(design)
            faces_idx = np.where(full_design[:, categories.index("Faces")] == 1)[0]
            objects_idx = np.where(full_design[:, categories.index("Objects")] == 1)[0]
            scenes_idx = np.where(full_design[:, categories.index("Scenes")] == 1)[0]
            bodies_idx = np.where(full_design[:, categories.index("Bodies")] == 1)[0]


            event_list = []
            event_list += [(time, 'Faces') for time in faces_idx]
            event_list += [(time, 'Objects') for time in objects_idx]
            event_list += [(time, 'Scenes') for time in scenes_idx]
            event_list += [(time, 'Bodies') for time in bodies_idx]

            # Sort them by time
            event_list.sort(key=lambda x: x[0])

            # Now get the condition labels in presentation order
            conditions_in_order = [condition for (_, condition) in event_list]

            conditions_in_order = np.array(conditions_in_order)

            trial_indices = {
                'Faces': np.where(conditions_in_order == 'Faces')[0],
                'Objects': np.where(conditions_in_order == 'Objects')[0],
                'Scenes': np.where(conditions_in_order == 'Scenes')[0],
                'Bodies': np.where(conditions_in_order == 'Bodies')[0],
            }

            # Define contrast depending on ROI
            contrast_def = {
                "EBA": ("Bodies", "Objects")
            }

            cond1, cond2 = contrast_def[ROI]
            cond1_betas = unmasked_betas[..., trial_indices[cond1]]
            cond2_betas = unmasked_betas[..., trial_indices[cond2]]

            # Paired t-test
            tvals, pvals = ttest_rel(cond1_betas, cond2_betas, axis=-1, alternative="greater")
            pvals_flat = pvals.flatten()
            not_nan_mask = ~np.isnan(pvals_flat)

            # Run FDR only on valid values
            _, pvals_corrected_flat = fdrcorrection(pvals_flat[not_nan_mask])

            # Prepare an array filled with nan, then fill the valid entries
            pvals_corrected_full = np.full_like(pvals_flat, np.nan)
            pvals_corrected_full[not_nan_mask] = pvals_corrected_flat

            # Reshape back
            pvals_corrected = pvals_corrected_full.reshape(pvals.shape)

            # Now threshold
            significant_mask = pvals_corrected < 0.05
            tvals_thresholded = tvals * significant_mask
            t_val_img = nib.Nifti1Image(tvals_thresholded, affine=whole_brain_mask_initial.affine)

            fusiform_mask_img = image.load_img(join(project_dir, 'Code/JuBrain_masks/fusiform_gyrus_resampled.nii'))
            inverse_fusiform_mask = math_img("~img.astype(bool)", img=fusiform_mask_img)
            masked_regions_img = math_img("regions * mask", regions=t_val_img, mask=inverse_fusiform_mask)

            regions_extracted_img, idx = connected_regions(
                                            maps_img=masked_regions_img,
                                            extract_type='connected_components',
                                            mask_img=whole_brain_mask_initial
                                        )
            
            regions_data = regions_extracted_img.get_fdata()
            affine = regions_extracted_img.affine

            n_regions = regions_data.shape[-1]
            region_sizes = []
            region_sides = [] 

            for i in range(n_regions):
                region = regions_data[..., i]
                region_sizes.append(np.count_nonzero(region))

                # Compute center of mass in voxel space
                com_voxel = center_of_mass(region)
                # Convert to MNI space
                com_mni = nib.affines.apply_affine(affine, com_voxel)

                # Determine hemisphere based on x-coordinate in MNI
                region_sides.append('left' if com_mni[0] < 0 else 'right')

            # Find largest region for each hemisphere
            left_indices = [i for i in range(n_regions) if region_sides[i] == 'left']
            right_indices = [i for i in range(n_regions) if region_sides[i] == 'right']

            left_largest = max(left_indices, key=lambda i: region_sizes[i]) if left_indices else None
            right_largest = max(right_indices, key=lambda i: region_sizes[i]) if right_indices else None

            selected_indices = [i for i in [left_largest, right_largest] if i is not None]
            largest_hemisphere_regions = index_img(regions_extracted_img, selected_indices)

            binary_mask_img = math_img("np.sum(img, axis=-1) > 0", img=largest_hemisphere_regions)

            binary_data = binary_mask_img.get_fdata().astype("uint8")
            binary_mask_img = new_img_like(binary_mask_img, binary_data)

            nib.save(binary_mask_img, join(datadir, "derivatives", f"sub-{sub}", "anat", f"{ROI}_mask_{smoothing}.nii"))




