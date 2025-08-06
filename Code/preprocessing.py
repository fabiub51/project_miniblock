import numpy as np
import os
from os.path import join, exists
import glob
import pandas as pd
import matplotlib.pyplot as plt
import re
import nibabel
import scipy
import scipy.io as sio
from scipy.ndimage import gaussian_filter
import time
import warnings
from tqdm import tqdm
from pprint import pprint
warnings.filterwarnings('ignore')
from scipy.interpolate import PchipInterpolator
from glmsingle.glmsingle import GLM_single

def get_design_matrices(project_dir, subjects):
    """
    This function takes in the .csv files from the presentation data and transforms them into the design matrix files required 
    for running GLMSingle. 
    For the miniblock and sustained design only the second of the onset is coded as a one to later set the stimulus duration accordingly. 
    The presentation files (.csv files) should be located at:
    ~/project_miniblock/Behavior/CondRichData or  
    ~/project_miniblock/Behavior/LocData 
    for the functional and localizer data, respectively. They will be stored in: 
    ~/project_miniblock/Behavior/designmats
    """

    # Set directories
    presdir = join(project_dir, 'Behavior', 'CondRichData')
    outdir  = join(project_dir, 'Behavior', 'designmats')
    if exists(presdir):
        print(f"Saving files to the following directory: {outdir}")
        os.makedirs(outdir, exist_ok=True)

        max_runs = 9

        for sub in subjects: 
            for run in range(1,max_runs+1):
                # load the presentation file of that run
                pattern = presdir + f'/P0{sub}_ConditionRich_Run{run}*.csv'
                matches = glob.glob(pattern)
                match = matches[0]
                df = pd.read_csv(match)
                df = df.drop([0, 1]).reset_index(drop=True)
                # Check condition
                if df["imFile"][5] == df ["imFile"][7]: # if the same file is repeated --> miniblock
                    runtype = "miniblock"
                elif (df["eventEndTime"][3] - df["eventStartTime"][3]) == 3.75 : # if presentation time is 3.75 --> sustained
                    runtype = "sus"
                    duration = 4
                else: # else event-related
                    runtype = "er"
                    duration = 1
                # Get all filenames, remove the fixation dot presentations
                file_names = df["imFile"].unique()
                file_names = file_names[file_names != 'Stimuli/Blank.png']
                file_names.sort() # Sort alphabetically
                
                # Make empty array of shape seconds by conditions 
                empty_array = np.zeros((388,40))
                
                # shift the files names back two places (to only get the onsets of the stimulus presentation for miniblock as GLMsingle requires )
                df['two_before'] = df['imFile'].shift(2)

                # loop over all conditions
                for name in range(empty_array.shape[1]):
                    for row in range(len(df["imFile"])):
                        if (file_names[name] == df["imFile"][row]) & (df["imFile"][row] != df["two_before"][row]): # control for repeated presentation in miniblock
                            empty_array[int(df["eventStartTime"][row]), name] = 1 # set to 1 of stimulus presentation started here

                # store design matrix as csv file 
                path = outdir + f'/P0{sub}_ConditionRich_Run_0{run}_{runtype}.csv' 
                np.savetxt(path, empty_array, delimiter=",", fmt="%d")

    else: # check if the presentation files exist 
        print("It seems like the functional presentation files are not present. Did you save them under ~project_miniblock/Behavior/CondRichData?")
    
    # similar logic for the localizer data
    # Set up directories 
    locpresdir = join(project_dir, 'Behavior', 'LocData')
    locoutdir  = join(project_dir, 'Behavior', 'designmats', 'localizer')

    if exists(locpresdir): # check if directory exists
        print(f"Saving files to the following directory: {locoutdir}")
        os.makedirs(locoutdir, exist_ok=True)

        subjects = [f"{i:02d}" for i in range(1, 23)] 
        max_loc_runs = 3 

        for sub in subjects: 
            for run in range(1,max_loc_runs+1):
                # Load presentation file of that run
                pattern = locpresdir + f'/P0{sub}_CategoryLocalizer_Run{run}*.csv'
                matches = glob.glob(pattern)
                if matches == []: # if no match is found, the subject probably had no 3 localizer runs
                    print(f"Run {run} not found for subject {sub}. Skipping.")
                    continue
                match = matches[0]
                df = pd.read_csv(match)
                df = df.drop([0, 1]).reset_index(drop=True)
                # make new columns for face-, object-, scene- and body-images 
                df["Faces"]= df["imFile"].str.contains("Faces")
                df["Objects"]= df["imFile"].str.contains("Objects")
                df["Scenes"] = df["imFile"].str.contains("Scenes")
                df["Bodies"]= df["imFile"].str.contains("Bodies")

                categories = ["Faces", "Objects", "Scenes", "Bodies"]
                
                # make empty array with max time point of the presentation file 
                empty_array = np.zeros((int(max(df["eventEndTime"])),4))
                
                # loop over condition names and mark every presentation of an image within a condition with a one, all others with 0
                for name in range(4):
                    for row in range(len(df["imFile"])):
                        if df[categories[name]][row] and df[categories[name]][row-1] == False: 
                            empty_array[int(df["eventStartTime"][row]), name] = 1 

                rows, cols = empty_array.shape

                new_array = np.zeros(empty_array.shape)
                # make sure that only the start of a new condition gets a 1 (as needed for GLMsingle)
                for col in range(cols):
                    for row in range(rows):
                        if empty_array[row][col] == 1:
                            if empty_array[row-1][col] == 1:
                                new_array[row][col] = 0
                            else: 
                                new_array[row][col] = 1
                        else: 
                            new_array[row][col] = 0

                # Save csv.file for every localizer run
                path = locoutdir + f'/P0{sub}_CategoryLocalizer_Run_0{run}.csv'
                np.savetxt(path, new_array, delimiter=",", fmt="%d")

    else: 
        print("It seems like the localizer presentation files are not present. Did you save them under ~project_miniblock/Behavior/CondRichData?")

def get_motion_parameters(project_dir, subjects):
    """
    This function takes in the desc-confounds_timeseries.tsv files from fMRIPrep and checks, whether there was excessive
    motion from a participant within a run. If that was the case, a plot is printed into the console. 
    Please not that this function depends on the outputs from fMRIPrep. 
    The function saves a motion plot for every subject and every file into a folder within the behavior directory. 
    """

    # Define subjects and maximum number of runs
    max_runs = 12
    output_dir = join(project_dir, "Behavior/motion_plots")
    datadir = join(project_dir, "miniblock/derivatives")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for sub in subjects:
        sub_dir = os.path.join(output_dir, f'sub-{sub}')
        os.makedirs(sub_dir, exist_ok=True)

        for run in range(1, max_runs + 1):
            path = join(datadir, f'sub-{sub}/func/sub-{sub}_task-func_run-{run:02d}_desc-confounds_timeseries.tsv')

            if not os.path.exists(path):
                print(f"Run {run} not found for subject {sub}. Skipping.")
                continue

            # Load and select motion parameters
            data = pd.read_csv(path, sep='\t')
            motion_params = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
            motion = data[motion_params].dropna()

            excessive_motion = 0
            for column in motion.columns:
                if np.max(motion[f"{column}"]) > 1.5:
                    print(f"Excessive movement detected in subject {sub}, run {run}. Check the motion plots.")
                    excessive_motion = 1

            # Plot
            plt.figure(figsize=(12, 6))
            for param in motion_params:
                plt.plot(motion[param], label=param)

            plt.title(f'Sub-{sub} Run-{run:02d} Head Motion Parameters')
            plt.xlabel('Timepoints (TRs)')
            plt.ylabel('Displacement (mm or radians)')
            plt.ylim(-3, 3)
            plt.legend()
            plt.tight_layout()

            # Save plot
            plot_path = os.path.join(sub_dir, f'sub-{sub}_run-{run:02d}_motion.png')
            plt.savefig(plot_path)
            if excessive_motion == 1:
                plt.show()
            plt.close()

def accuracies_functional(project_dir, subjects):

    """
    This function takes in the .csv files from the presentation data and checks, the accuracy of each participant within each run. 
    Returns a dataframe with columns participant_run, accuracy, and design.
    This is done only for the functional data. See below the function for the localizer data. The tasks were slightly different, 
    therefore we opted for two different functions. 
    """
 
    # Initialize a dictionary to store the accuracies
    accuracies = {}

    # Loop over subjects and runs
    for sub in subjects: 
        for run in range(1,10):  
            # Get the presentation file's path
            pattern = join(project_dir, f'Behavior/CondRichData/P0{sub}_ConditionRich_Run{run}_*.csv')
            matches = glob.glob(pattern)
            path = matches[0]  

            # Read the file
            df = pd.read_csv(path)

            # Whenever an event occurred
            condition = (df['responseEvent'] == 1)
            # filter the dataframe
            df_filtered = df[condition]

            # filter for conditions
            conditions = np.array(df_filtered["eventEndTime"] - df_filtered["eventStartTime"])
            values, counts = np.unique(conditions, return_counts=True)
            most_frequent = values[np.argmax(counts)]

            if most_frequent == 3.75:
                runtype = "sus"
            elif df["imFile"][5] == df ["imFile"][7]:
                runtype = "miniblock"
            else: 
                runtype = "er"
            
            df_filtered = df_filtered.copy()
            # Create a new column "responseWindow" with the given formula
            df_filtered[ "responseWindow"] = (df_filtered["eventStartTime"].fillna(0) + df_filtered["fixRespStart"].fillna(0) + 5)

            # Calculate accuracy based on participant response time
            correct_responses = (df_filtered["eventStartTime"] + df_filtered["participantResponse.rt"] <= df_filtered["responseWindow"])
            accuracy = 100 * correct_responses.sum() / len(df_filtered)

            # Store the accuracy in the dictionary using participant and run as the key
            accuracies[f"sub{sub}_run{run}"] = {
                "accuracy": accuracy,
                "runtype": runtype
            }

    # Store one dataframe for all subjects
    accuracies_df = pd.DataFrame.from_dict(accuracies, orient='index').reset_index()
    accuracies_df.columns = ["Participant_Run", "Accuracy", "RunType"]
    accuracies_df.to_csv(join(project_dir,"Behavior/accuracy_results_func.csv"), index=False)
    return accuracies_df

def accuracies_localizer(project_dir, subjects):
    """
    This function takes in the .csv files from the presentation data and checks, the accuracy of each participant within each run. 
    Returns a dataframe with columns participant_run, accuracy, and design.
    This is done only for the localizer data. 
    """

    localizer_accuracies = {}

    # Loop over subjects and runs
    for sub in subjects:  

        for run in range(1,4):  
            
            # Get the presentation file's path
            pattern = join(project_dir, f'Behavior/LocData/P0{sub}_CategoryLocalizer_Run{run}_*.csv')
            matches = glob.glob(pattern)

            # skip if participant had only 2 localizer runs
            if matches == []:
                print(f"Run {run} not found for subject {sub}. Skipping.")
                continue

            # Read the file
            df = pd.read_csv(matches[0])

            # detect repeats manually since the column repeatedImage from the csv-file does not seem to correspond 
            repeats = []
            image = ['start']
            
            for i in range(len(df)):
                if image == df["imFile"][i]:
                    repeats.append(1)
                else: 
                    repeats.append(0)
                if df["imFile"][i] != r"Stimuli\Blank.png":
                    image = df["imFile"][i]


            df["repeats"] = repeats
            condition = (df['repeats'] == 1) 
            # filter for the repeated stimuli only
            df_filtered = df[condition]

            df_filtered = df_filtered.copy()
            # Create a new column "responseWindow" by giving a 5 second interval to the start time of each presentation
            df_filtered[ "responseWindow"] = (df_filtered["eventStartTime"].fillna(0)  + 5)

            # Calculate accuracy based on participant response time
            correct_responses = (df_filtered["eventStartTime"] + df_filtered["participantResponse.rt"] <= df_filtered["responseWindow"])
            accuracy = 100 * correct_responses.sum() / len(df_filtered)
            localizer_accuracies[f"sub{sub}_run{run}"] = {
                    "accuracy": accuracy
                }

    # save as a csv file 
    localizer_accuracies_df = pd.DataFrame.from_dict(localizer_accuracies, orient='index').reset_index()
    localizer_accuracies_df.columns = ["Participant_Run", "Accuracy"]
    localizer_accuracies_df.to_csv(join(project_dir, "Behavior/accuracy_results_loc.csv"), index=False)
    return localizer_accuracies_df

def fmri_interpolate(input_mat, tr_orig=2.0, tr_new=0.5):
    """
    Interpolates 4D fMRI data (X, Y, Z, T) to double the temporal resolution.
    Assumes slice time correction was done to the middle of the TR.

    Parameters:
    - input_mat: 4D numpy array (X, Y, Z, T)
    - tr_orig: Original TR (default 1.0s)
    - tr_new: New TR (default 0.5s)

    Returns:
    - output_mat: 4D numpy array (X, Y, Z, new T)
    """
    assert input_mat.ndim == 4, "Input must be a 4D array."

    num_x, num_y, num_z, num_t = input_mat.shape

    # Create original and new time axes
    original_time = np.arange(0.5, num_t, 1.0) * tr_orig
    new_time = np.arange(0.5, original_time[-1] + tr_new, tr_new)

    # Preallocate the output matrix
    output_mat = np.zeros((num_x, num_y, num_z, len(new_time)), dtype=input_mat.dtype)

    # Interpolate for each voxel
    for x in range(num_x):
        for y in range(num_y):
            for z in range(num_z):
                time_series = input_mat[x, y, z, :]
                if np.any(time_series):  # Only interpolate if there is non-zero data
                    interpolator = PchipInterpolator(original_time, time_series)
                    output_mat[x, y, z, :] = interpolator(new_time)

    return output_mat

def glm_single_func(project_dir, subjects, outputs):

    """
    This function runs GLM Single for all subjects specified in the function's input for all designs and smoothing option for the 
    FUNCTIONAL DATA. Please find below the function for the localizer data as well.  
    It takes in the space-MNI152NLin2009cAsym_desc-preproc_bold.nii output from fMRIPrep.
    It stores the all outputs that you specify in the outputs input. 
    The function also calls an interpolation function that is required to align the TRs. This function is part of the preprocessing
    script, too. 
    """

    runtypes = [ 'er', 'miniblock', 'sus']
    sm_fwhm = 2 # smoothing
    tr_old = 2 # TR before resampling
    tr_new = 0.5 # TR after resampling
    datadir = join(project_dir,'miniblock','derivatives')
    presdir = join(project_dir, 'Behavior', 'designmats')

    for sub in subjects: 
        subString = f'sub-{sub}'
        for runTypeIdx in runtypes:
            for smoothing in range(2):
                print(f'These are {runTypeIdx} - runs and smoothing is {smoothing}')

                # Specify stimulus duration (1 second for the ER, 4 secongs for MB and SUS)
                if runTypeIdx == 'er':
                    stimdur = 1
                else: 
                    stimdur = 4

                # get files for design matrices and fmri data
                pattern = presdir + f'/P0{sub}_ConditionRich_Run*_{runTypeIdx}.csv'
                matches = glob.glob(pattern)
                matches.sort()
                
                # store data and designs
                data = []
                design = []
                for i in range(len(matches)):
                    designMat = pd.read_csv(matches[i], header=None)
                    print(f"Size of Design Matrix before upsampling: {designMat.shape}")
                    num = re.search(r'Run_(\d+)', matches[i])
                    # reminder: the runNum of the designmats is not the same as for the functional data as localizer runs were interspersed 
                    runNum = int(num.group(1))
                    if (runNum >3) & (runNum < 7) & (sub != '01'): 
                        runNum += 1 # localizer run after 3rd functional run
                    elif (runNum >= 7):
                        runNum += 2 # localizer run after 6th functional run (7th overall)
                    elif (sub == '01') & (runNum >4) & (runNum < 7):
                        runNum +=1
                    
                    # get the fully preprocessed file from fMRIprep
                    if runNum < 10: 
                        file_path = join(datadir, subString, 'func', subString + f'_task-func_run-0{runNum}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
                    else: 
                        file_path = join(datadir, subString, 'func', subString + f'_task-func_run-{runNum}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
                    
                    # Load the file 
                    origData = nibabel.load(file_path)
                    # Interpolation with custom function 
                    interpData = fmri_interpolate(origData.get_fdata(), tr_old, tr_new)
                    print(interpData.shape[-1])

                    # Resampling the Design Matrix 
                    upsample_factor = 2
                    time_points, conditions = designMat.shape

                    upsampled_matrix = np.zeros((time_points * upsample_factor, conditions))

                    # Fill design matrix with values and remove the first row 
                    upsampled_matrix[::upsample_factor, :] = designMat
                    design.append(upsampled_matrix[1:-1, :])
                    print(f"Size of Design Matrix after upsampling: {design[i].shape}")

                    # smoothing the data 
                    if smoothing == 1: 
                        print('doing smoothing')
                        sigma = sm_fwhm/np.sqrt(8 * np.log(2))
                        numX,numY,numz,numT = interpData.shape
                        smoothedData = np.full(interpData.shape, np.nan)
                        for tIdx in range(numT):
                            cData = interpData[:,:,:,tIdx]
                            smoothedData[:,:,:,tIdx] = gaussian_filter(cData, sigma)
                        data.append(smoothedData)

                        outputdir = join(project_dir, 'miniblock', 'Outputs', subString, f'sm_{sm_fwhm}_vox_{subString}_{runTypeIdx}')
                    
                    else: # skip smoothing
                        data.append(interpData)
                        outputdir = join(project_dir, 'miniblock', 'Outputs', subString, f'unsmoothed_{subString}_{runTypeIdx}')
                    
                    assert design[i].shape[0] == interpData.shape[-1], "Design matrix and fMRI timepoints do not match!"
                
                # Some print statements to check data (sizes, stimuli per design...)
                print(f'There are {len(data)} runs in total\n')
                print(f'N = {data[0].shape[3]} TRs per run\n')
                print(f'The dimensions of the data for each run are: {data[0].shape}\n')
                print(f'The stimulus duration is {stimdur} seconds\n')
                print(f'XYZ dimensionality is: {data[0].shape[:3]} (one slice only in this example)\n')
                print(f'Numeric precision of data is: {type(data[0][0,0,0,0])}\n')    

                opt = dict()
                opt['wantlibrary'] = 1
                opt['wantglmdenoise'] = 1
                opt['wantfracridge'] = 1
                opt['wantfileoutputs'] = outputs
                opt['wantmemoryoutputs'] = [0, 0, 0, 0] # no memory outputs needed


                glmsingle_obj = GLM_single(opt) # make GLMsingle object

                pprint(glmsingle_obj.params)

                start_time = time.time() # check duration

                os.makedirs(outputdir, exist_ok=True)
                # run GLMsingle
                results_glmsingle = glmsingle_obj.fit(
                    design, 
                    data, 
                    stimdur, 
                    tr_new, 
                    outputdir=outputdir
                )

                elapsed_time = time.time() - start_time
                print('\telapsed_time: '
                f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
                )

def glm_single_loc(project_dir, subjects, outputs):
    """
    This function runs GLM Single for all subjects specified in the function's input for the 
    LOCALIZER DATA. Please find below the function for the localizer data as well.  
    It takes in the space-MNI152NLin2009cAsym_desc-preproc_bold.nii output from fMRIPrep.
    It stores the all outputs that you specify in the outputs input (list of 4 integers). 
    The function also calls an interpolation function that is required to align the TRs. This function is part of the preprocessing
    script, too. 
    """
    sm_fwhm = 2 # smoothing 
    tr_old = 2 # before resampling
    tr_new = 0.5 # after resampling
    stimdur = 16 # since all localizer runs have the same stimulus length
    datadir = join(project_dir,'miniblock','derivatives')
    presdir = join(project_dir, 'Behavior', 'designmats', 'localizer')
    for sub in subjects: 
        subString = f'sub-{sub}'

        for smoothing in range(2):
            print(f"Now working on subject {sub}, smoothing: {smoothing}")
            # get files for design matrices and fmri data
            pattern = presdir + f'/P0{sub}_CategoryLocalizer_Run*.csv'
            matches = glob.glob(pattern)
            matches.sort()
            
            # store data and designs
            data = []
            design = []
            for i in range(len(matches)):
                designMat = pd.read_csv(matches[i], header=None)
                print(f"Size of Design Matrix before upsampling: {designMat.shape}")
                num = re.search(r'Run_(\d+)', matches[i])
                runNum = int(num.group(1))
                if (runNum == 1) & (sub != '01'): 
                    runNum += 3 # localizer run after 3rd functional run
                elif (runNum == 2):
                    runNum += 6 # localizer run after 6th functional run (7th overall)
                elif (runNum == 3):
                    runNum += 9
                elif (sub == '01') & (runNum == 1):
                    runNum +=4
                
                # get the nii.gz file 
                if runNum < 10: 
                    file_path = join(datadir, subString, 'func', subString + f'_task-func_run-0{runNum}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
                else: 
                    file_path = join(datadir, subString, 'func', subString + f'_task-func_run-{runNum}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
                
                # Load the file 
                origData = nibabel.load(file_path)
                # Interpolation 
                interpData = fmri_interpolate(origData.get_fdata(), tr_old, tr_new)
                print(interpData.shape[-1])

                # Resampling the Design Matrix 
                upsample_factor = 2
                time_points, conditions = designMat.shape

                upsampled_matrix = np.zeros((time_points * upsample_factor, conditions))

                # Fill design matrix with values and remove the first row 
                upsampled_matrix[::upsample_factor, :] = designMat
                design.append(upsampled_matrix[1:-1, :])
                print(f"Size of Design Matrix after upsampling: {design[i].shape}")

                if smoothing == 1: # Apply smoothing
                    print('doing smoothing')
                    sigma = sm_fwhm/np.sqrt(8 * np.log(2))
                    numX,numY,numz,numT = interpData.shape
                    smoothedData = np.full(interpData.shape, np.nan)
                    for tIdx in range(numT):
                        cData = interpData[:,:,:,tIdx]
                        smoothedData[:,:,:,tIdx] = gaussian_filter(cData, sigma)
                    data.append(smoothedData)

                    outputdir = join(project_dir, 'miniblock', 'Outputs','localizer',f'sub-{sub}', f'sm_{sm_fwhm}_vox_{subString}_localizer')
                
                else: # skip smoothing
                    data.append(interpData)
                    outputdir = join(project_dir, 'miniblock', 'Outputs','localizer',f'sub-{sub}', f'unsmoothed_{subString}_localizer')
                
                assert design[i].shape[0] == interpData.shape[-1], "Design matrix and fMRI timepoints do not match!"
            
            # some print statements to check data specifications
            print(f'There are {len(data)} runs in total\n')
            print(f'N = {data[0].shape[3]} TRs per run\n')
            print(f'The dimensions of the data for each run are: {data[0].shape}\n')
            print(f'The stimulus duration is {stimdur} seconds\n')
            print(f'XYZ dimensionality is: {data[0].shape[:3]} (one slice only in this example)\n')
            print(f'Numeric precision of data is: {type(data[0][0,0,0,0])}\n')    

            opt = dict()
            opt['wantlibrary'] = 1
            opt['wantglmdenoise'] = 1
            opt['wantfracridge'] = 1
            opt['wantfileoutputs'] = outputs
            opt['wantmemoryoutputs'] = [0, 0, 0, 0] # no memory outputs needed


            glmsingle_obj = GLM_single(opt) # create GLMsingle object

            pprint(glmsingle_obj.params)

            start_time = time.time() # check durations

            os.makedirs(outputdir, exist_ok=True)
            # Run GLMsingle
            results_glmsingle = glmsingle_obj.fit(
                design, 
                data, 
                stimdur, 
                tr_new, 
                outputdir=outputdir
            )

            elapsed_time = time.time() - start_time
            print('\telapsed_time: '
            f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
            )                       