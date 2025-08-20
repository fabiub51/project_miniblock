function run_ROI_decoding_pairwise_group(project_dir,subj, design, mask, output_dir, group)
    % This function runs the decoding analysis with specified subject, design, and smoothing for one specific ROI and 
    % within one of the groups of images: scenes, animate/inanimate objects.

    %% Initialize config
    cfg = decoding_defaults;
    cfg.analysis = 'roi';  % specify ROI-based decoding 
    cfg.decoding.software = 'libsvm';  % SVM library
    cfg.decoding.method = 'classification';
    cfg.scale.method = 'min0max1global';
    % Setting up directories 
    basedir = fullfile(project_dir, '/miniblock/Outputs/');
    pres_dir = fullfile(project_dir, "Behavior/CondRichData/group_decoding");
    glmdir = fullfile(basedir, 'GLMSingle_Outputs');
    output_subfolder = output_dir;
    % Create new folder if does not exist yet
    if ~exist(fullfile(basedir, 'decoding', 'ROI','pairwise','group', output_subfolder, group, design, subj), 'dir')
        mkdir(fullfile(basedir, 'decoding', 'ROI','pairwise','group', output_subfolder, group, design, subj));
    end
    % Set up  Path to beta values
    beta_folder = fullfile(glmdir, 'nifti_betas');
    cfg.results.dir = fullfile(basedir, 'decoding', 'ROI','pairwise', 'group', output_subfolder, group, design, subj);
    
    % Mask
    cfg.files.mask = mask;
    cfg.results.output = {'confusion_matrix', 'accuracy_pairwise'};
    
    %% Presentation files to extract groups
    pattern = strcat('^',subj,'_.*.csv$');  
    files = dir(fullfile(pres_dir, '*.csv'));
    filenames = {files.name};
    matches = ~cellfun('isempty', regexp(filenames, pattern));
    design_files = fullfile(pres_dir, filenames(matches)); 
    
    data = readtable(design_files{1}); % Load the subject's file that contains the order of the betas by group
    binary_rows = contains(data.(design), group); % filter the specified design and group
    filtered_table = data(binary_rows, :); % select only the indices for that group and design
    labels = filtered_table.(design);
    labels_num = cellfun(@(x) erase(x, group), labels, 'UniformOutput', false);
    numeric_labels = cellfun(@str2double, labels_num); % final labels of betas
    
    %% Beta images - select beta images corresponding to group
    pattern = fullfile(beta_folder, sprintf('beta_*%s*%s*.nii', subj, design));
    files = dir(pattern);
    
    % Sort by name
    [~, idx] = sort({files.name});
    files = files(idx);
    
    % Get full paths
    beta_files = arrayfun(@(f) fullfile(f.folder, f.name), files, 'UniformOutput', false);
    betas = beta_files;
    filtered_betas = betas(binary_rows); % Apply the filter
    
    % Assign to config
    cfg.files.name = filtered_betas;
    
    %% Specify chunks
    unique_labels = unique(labels_num);
    chunks = zeros(size(labels_num));
    
    for i = 1:length(unique_labels)
        condition_idx = find(strcmp(labels_num, unique_labels{i}));
        % Assign chunk numbers 1-6 to each repetition
        chunks(condition_idx) = 1:length(condition_idx);
    end
    
     %% Set up cfg
    cfg.files.mask = mask;
    cfg.results.output = {'accuracy_pairwise'};
    cfg.files.label = numeric_labels(:);
    cfg.files.chunk = chunks;
    %% Decoding design
    cfg.design.function = 'make_design_cv';   % Define function to use for design creation
    cfg.design.label = 'leave_one_chunk_out';  % Cross-validation type
    cfg.design = make_design_cv(cfg);  
    cfg.results.overwrite = 1;  % Overwrite existing results
    %% Run decoding
    decoding(cfg);  % Run the decoding analysis
end 
