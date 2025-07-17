% Base directories
fullPath = pwd;

targetDir = 'project_miniblock';
idx = strfind(fileDir, targetDir);

if ~isempty(idx)
    % get the path up to the end of targetDir
    pathUpToDir = fileDir(1 : idx(1) + length(targetDir) - 1);
    fprintf('Path up to %s: %s\n', targetDir, pathUpToDir);
else
    disp('Directory not found in path.');
end
project_dir = pathUpToDir;

% Subjects to loop over
all_subjects = 1:22;
skip_subjects = [9, 16];
subjects = setdiff(all_subjects, skip_subjects);

% Designs and ROIs
designs = {'miniblock', 'er', 'sus'};

% Loop over subjects
for s = subjects
    subj = sprintf('sub-%02d', s);
    % Loop over designs
    for d = 1:length(designs)
        design = designs{d};
        run_searchlight_decoding(project_dir, subj, design)

    end
end
