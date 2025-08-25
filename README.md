# project_miniblock
## Start here - if you want to start at preprocessing - 
- Step 1: Once you cloned the repo, store all BIDS-formatted data in the "new_data" folder. This way, you can run the run_fmriprep.txt file in bash using docker.
- Step 2: After running fMRIPrep, you can start analyzing the data in the main script. Just make sure that you set up GLMsingle somewhere on your machine.
## Start here - if you want to skip preprocessing - 
- Step 3: Start with the GLMSingle outputs
## Step by step analysis 
- main.ipynb takes you through all of the analyses
- Preprocessing is a required step since it creates design matrices used for condition order later on
- you can skip the GLMSingle part if you downloaded the outputs to your machine and in the correct folder
- all functions called in the main script are defined in .py files that fit the name of the current analysis-step in lower case
## Group analyses 
- group analyses were conducted in R
- all the necessary individual files are contained in statistical_analyses
- there is an R-project as well as environment you can use 
