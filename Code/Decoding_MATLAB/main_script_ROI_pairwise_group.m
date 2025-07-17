% Base directories
fullPath = pwd;

targetDir = 'project_miniblock';
[fileDir,~,~] = fileparts(fullPath);
idx = strfind(fileDir, targetDir);

if ~isempty(idx)
    % get the path up to the end of targetDir
    pathUpToDir = fileDir(1 : idx(1) + length(targetDir) - 1);
    fprintf('Path up to %s: %s\n', targetDir, pathUpToDir);
else
    disp('Directory not found in path.');
end
project_dir = pathUpToDir;
output_base_dir = fullfile(project_dir, 'miniblock/Outputs/decoding');

% Subjects to loop over
all_subjects = 1:1;
skip_subjects = [9, 16];
subjects = setdiff(all_subjects, skip_subjects);

% Designs and ROIs
designs = {'miniblock', 'er', 'sus'};
rois = {'EBA', 'PPA', 'FFA', 'EVC', 'visually_responsive_voxels'};
groups = {'scene', 'isanimate', 'inanimate'};

% Loop over subjects
for s = subjects
    subj = sprintf('sub-%02d', s);
    
    % Loop over designs
    for d = 1:length(designs)
        design = designs{d};
        
        % Loop over ROIs
        for r = 1:length(rois)
            roi = rois{r};
            % Set mask path
            if strcmp(roi, 'visually_responsive_voxels')
                mask = fullfile(project_dir, 'miniblock', 'derivatives', subj, 'anat', 'visually_responsive_voxels_sm_2_vox_gm.nii');
            else
                mask = fullfile(project_dir, 'miniblock', 'derivatives', subj, 'anat', sprintf('%s_mask_sm_2_vox.nii', roi));
            end

            for g = 1:length(groups)
                group = groups{g};
                output_dir = roi;
            
                run_ROI_decoding_pairwise_group(project_dir, subj, design, mask, output_dir, group)
                close all
            end
        end
    end
end
close all;