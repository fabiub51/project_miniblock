# project_miniblock
## Start here - if you want to start at preprocessing - 
- Step 1: Once you cloned the repo, store all BIDS-formatted data in the "new_data" folder. This way, you can run the run_fmriprep.txt file in bash using docker.
- Step 2: After running fMRIPrep, you can start analyzing the data in the main script. Just make sure that you set up glmsingle somewhere on your machine.
## Start - if you want to skip preprocessing - 
- Step 3: Start with the GLMSingle outputs and create the masks for all ROIs using the script.
## Reliability
- Step 4: Run the function to get all whole-brain reliability maps (all subjects, all designs, both smoothing options, all splits) - takes approx. 1 hour
- Step 5: Gather all reliability maps into a dataframe.
- Step 6: Make a quick plot for each ROI by using the code available or make a custom plot using the dataframe.
- Step 7: Run additional analyses (Reliability over the progress of the experiment, Noise Ceilings as in NSD, 
  
## Decoding 
... to be added ...
## Representational Similarity Analysis
... to be added ...
