import numpy as np
import nibabel as nib
from nilearn import plotting, image
import pandas as pd
import os
import glob
from os.path import join
from scipy.stats import pearsonr
import re


def create_nifti_betas(project_dir, subjects):
    """
    Function that creates individual beta images for every stimulus presentation (240 total = 6x40)
    """

    outdir = join(project_dir, "miniblock/Outputs")
    smooths = ["sm_2_vox"]
    runtypes = ['miniblock', "sus", "er"]

    # Affine transformation of data
    target_affine = np.array([
        [2., 0., 0., -76.],
        [0., 2., 0., -112.],
        [0., 0., 2., -76.],
        [0., 0., 0., 1.]
    ])
    affine = target_affine

    for sub in subjects: 
        for smoothing in smooths: 
            for runtype in runtypes: 

                results_glmsingle = dict()
                results_glmsingle['typed'] = np.load(join(outdir,"GLMSingle_Outputs",f'{smoothing}_sub-{sub}_{runtype}_TYPED_FITHRF_GLMDENOISE_RR.npy'), allow_pickle=True).item()
                betas = results_glmsingle['typed']['betasmd']

                # Output folder for NIfTIs
                save_dir = join(outdir, 'GLMSingle_Outputs/nifti_betas')
                os.makedirs(save_dir, exist_ok=True)

                # Define affine (identity or use a real affine if you have one)
                affine = target_affine

                # Loop over trials and save each one
                n_trials = betas.shape[-1]
                for i in range(n_trials):
                    beta_data = betas[..., i]  # Take 3D volume for trial i
                    nii = nib.Nifti1Image(beta_data, affine)
                    nib.save(nii, join(save_dir, f'beta_{i+1:04d}_{smoothing}_sub-{sub}_{runtype}.nii'))

def extract_object_name(filepath):
    # Get just the filename, e.g. "things_accordion_03s.jpg"
    filename = os.path.basename(filepath)
    # Remove extension
    name = os.path.splitext(filename)[0]
    # Split by underscore
    parts = name.split('_')
    # Get the second part (index 1)
    if len(parts) >= 2:
        return parts[1]
    else:
        return None

def get_animate(project_dir):
    presdir = join(project_dir, 'Behavior', 'CondRichData')

    subjects = [f"{i:02d}" for i in range(1, 23) if i not in [9, 16]]
    max_runs = 9

    animate = []
    inanimate = []

    for sub in subjects: 
        for run in range(1,max_runs+1):
            pattern = presdir + f'/P0{sub}_ConditionRich_Run{run}*.csv'
            matches = glob.glob(pattern)
            match = matches[0]
            df = pd.read_csv(match)
            df = df.drop([0, 1]).reset_index(drop=True)

            # Check condition
            if df["imFile"][5] == df ["imFile"][7]:
                runtype = "miniblock"
                duration = 4
            elif (df["eventEndTime"][3] - df["eventStartTime"][3]) == 3.75 :
                runtype = "sus"
                duration = 4
            else: 
                runtype = "er"
                duration = 1

            file_names = df["imFile"].unique()
            file_names = file_names[file_names != 'Stimuli/Blank.png']
            file_names.sort()



            for file in list(file_names)[20:]:
                object = extract_object_name(file)
                if (object not in animate) and (object not in inanimate):
                    user_check = input(f"{file}: Animate? A Inanimate? L")
                    if user_check == "a":
                        animate.append(object)
                    else: 
                        inanimate.append(object)

    return animate, inanimate

def get_orders_by_design(project_dir, animate):
    presdir = join(project_dir, 'Behavior', 'CondRichData')
    subjects = [f"{i:02d}" for i in range(1, 23) if i not in [9, 16]]
    max_runs = 9

    for sub in subjects: 
        orders = {
        "miniblock" : [],
        "er": [],
        "sus": []
        }
        print(f"Working on subject {sub}")
        for run in range(1,max_runs+1):
            pattern = presdir + f'/P0{sub}_ConditionRich_Run{run}*.csv'
            matches = glob.glob(pattern)
            match = matches[0]
            df = pd.read_csv(match)
            df = df.drop([0, 1]).reset_index(drop=True)

            # Check condition
            if df["imFile"][5] == df ["imFile"][7]:
                runtype = "miniblock"
                duration = 4
            elif (df["eventEndTime"][3] - df["eventStartTime"][3]) == 3.75 :
                runtype = "sus"
                duration = 4
            else: 
                runtype = "er"
                duration = 1

            file_names = df["imFile"].unique()
            file_names = file_names[file_names != 'Stimuli/Blank.png']
            file_names.sort()

            scene_counter = 0
            animate_counter = 0
            inanimate_counter = 0
            file_names_new = list(file_names)
            for file_idx in range(40):
                object = extract_object_name(file_names[file_idx])
                if "sun_" in file_names[file_idx]: 
                    file_names_new[file_idx] = f"scene{scene_counter}"
                    scene_counter += 1
                
                elif object in animate.values: 
                    file_names_new[file_idx] = f"isanimate{animate_counter}"
                    animate_counter += 1

                else: 
                    file_names_new[file_idx] = f"inanimate{inanimate_counter}"
                    inanimate_counter += 1

            if animate_counter != inanimate_counter:
                print("Lengths do not match!")
                break

            empty_array = np.zeros((40))

            df['two_before'] = df['imFile'].shift(2)
            for row in range(len(df["imFile"])):
                for name_idx in range(40):
                    if (file_names[name_idx] == df["imFile"][row]) & (df["imFile"][row] != df["two_before"][row]): 
                        orders[runtype].append(file_names_new[name_idx])


        order_df = pd.DataFrame(orders)
        pd.DataFrame.to_csv(order_df, path_or_buf=join(presdir, "group_decoding", f"sub-{sub}_orders.csv"))
            
                
