function run_ROI_decoding_pairwise(project_dir,subj, design, mask, output_dir)
    % This function runs the decoding analysis with specified subject, design, and smoothing for one specific ROI.

    %% Initialize config
    cfg = decoding_defaults;
    cfg.analysis = 'roi';  % specify ROI-based decoding 
    cfg.decoding.software = 'libsvm';  % SVM library
    cfg.decoding.method = 'classification';
    cfg.scale.method = 'min0max1global';
    % Setting up directories 
    basedir = fullfile(strcat(project_dir, '/miniblock/Outputs/'));
    glmdir = fullfile(basedir, 'GLMSingle_Outputs');
    output_subfolder = output_dir;
    % Create new folder if does not exist 
    if ~exist(fullfile(basedir, 'decoding', 'ROI','pairwise', output_subfolder, design, subj), 'dir')
        mkdir(fullfile(basedir, 'decoding', 'ROI','pairwise', output_subfolder, design, subj));
    end

    beta_folder = fullfile(glmdir, 'nifti_betas');
    cfg.results.dir = fullfile(basedir, 'decoding', 'ROI','pairwise', output_subfolder, design, subj);

    %% Beta images
    % filter for subject and design
    pattern = fullfile(beta_folder, sprintf('beta_*%s*%s*.nii', subj, design));
    files = dir(pattern);
    
    % Sort by name (since named 1 - 240)
    [~, idx] = sort({files.name});
    files = files(idx);
    
    % Get full paths
    beta_files = arrayfun(@(f) fullfile(f.folder, f.name), files, 'UniformOutput', false);
    
    % Assign to config
    cfg.files.name = beta_files;
    
    % Mask
    cfg.files.mask = mask;
    cfg.results.output = {'confusion_matrix', 'accuracy_pairwise'};
    
    %% Set labels
    % Path to presentation files 
    pres_dir = fullfile(project_dir, "/Behavior/designmats/");
    
    pattern = strcat('^P0',subj(end-1:end),'_.*_',design,'\.csv$');  % Filename pattern for the CSV files
    files = dir(fullfile(pres_dir, '*.csv'));
    filenames = {files.name};
    % Find matching files 
    matches = ~cellfun('isempty', regexp(filenames, pattern));
    design_files = fullfile(pres_dir, filenames(matches));
    
    % Initialize labels array
    labels = [];
    for i = 1:length(design_files)
        data = readtable(design_files{i});
        [rows, cols] = size(data);
        for r = 1:rows
            for c = 1:cols
                if data{r,c} == 1
                    labels(end+1) = c;  % Condition number = column index
                end
            end
        end
    end
    cfg.files.label = labels(:);  % Ensure labels are a column vector
    
    %% Define chunks for cross-validation 
    unique_labels = unique(labels);
    chunks = zeros(size(labels));
    
    for i = 1:length(unique_labels)
        condition_idx = find(labels == unique_labels(i));
        % Assign chunk numbers 1-6 to each repetition
        chunks(condition_idx) = 1:length(condition_idx);
    end
    
    cfg.files.chunk = chunks';
    
    %% Decoding design
    cfg.design.function = 'make_design_cv';  % Define function to use for design creation
    cfg.design.label = 'leave_one_chunk_out';  % Cross-validation type (in this case betas)
    cfg.design = make_design_cv(cfg); 
    cfg.results.overwrite = 1;  % Overwrite existing results
    
    %% Run decoding
    decoding(cfg);  % Run the decoding analysis
end
