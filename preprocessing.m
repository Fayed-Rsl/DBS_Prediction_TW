% =============================================================================
% MEG-LFP Feature Extraction Pipeline
% Computes power spectra and source-space connectivity features across
% frequency bands for FOOOF analysis and machine learning
% =============================================================================

clearvars;
clc;
restoredefaultpath;
clear RESTOREDEFAULTPATH_EXECUTED;

% =============================================================================
% PATH CONFIGURATION
% =============================================================================
matlab_toolbox = '/data/user/rasfay01/matlab_toolbox/';
script_path = '/home/rasfay01/scripts_compute_features/';
ft_path = [matlab_toolbox, 'fieldtrip-20201229'];

mpath = '/data/user/rasfay01/ContactPrediction/'; 
grid_and_parcel_path = [mpath, 'utils/cortical_grid_and_parcel.mat']; 
headmodel_folder = [mpath, 'headmodel/'];
sourcemodel_folder = [mpath, 'sourcemodel/'];
info_folder = [mpath, 'info/'];

% Initialize FieldTrip toolbox
addpath(ft_path, script_path);
ft_defaults;

% =============================================================================
% SUBJECT AND DATA CONFIGURATION
% =============================================================================
subj = 'ID';  % Subject ID (currently single subject, can be looped later)

% Load subject information and anatomical templates
load([info_folder, subj, '_info.mat']);
load(grid_and_parcel_path);  % Cortical grid and AAL parcellation
load([headmodel_folder, subj, '/', subj, '_headmodel.mat']);
load([sourcemodel_folder, subj, '/', subj, '_sourcemodel.mat']);

% =============================================================================
% FREQUENCY BAND DEFINITIONS
% =============================================================================
freq_bands.label      = {'theta', 'alpha', 'low-beta', 'high-beta', 'low-gamma', 'high-gamma', 'sHFO', 'fHFO'};
freq_bands.centerfreq = [5,       10,      17,         28,          48,          75,           250,    350];
freq_bands.bandwidth  = [2,       2,       4,          7,           12,          15,           48,     48];

foi = 1:400;  % Frequencies of interest for full power spectrum (1-400 Hz)

% =============================================================================
% FEATURE SPACE DIMENSIONS
% =============================================================================
N_lfp = subj_info.(subj).number_lfp_chans;  % Number of LFP channels
N_groups = numel(parcel.aallabel);  % Number of AAL parcels
N_bands = numel(freq_bands.label);  % Number of frequency bands
N_band_feat = (N_lfp * N_groups + N_lfp);  % Features per band: connectivity + power
N_feat = N_band_feat * N_bands;  % Total features across all bands

conds = subj_info.(subj).conds;  % Conditions (typically medication on/off)
ref_scheme = 'average';  % Referencing scheme: 'average' or 'bipolar'

% =============================================================================
% MAIN PROCESSING LOOP - ITERATE OVER CONDITIONS
% =============================================================================
for c = 1:numel(conds)
    cond = conds{c};
    
    % Create output directories
    foof_dir = [mpath, 'pow_for_fooof/', cond, '/', ref_scheme, '/'];
    feat_dir = [mpath, 'features/', cond, '/', ref_scheme, '/'];
    
    if ~exist(foof_dir, 'dir'), mkdir(foof_dir); end
    if ~exist(feat_dir, 'dir'), mkdir(feat_dir); end
    
    % list for combining data across files
    feats_file = [];
    bad_lfp_avgref = [];
    weights = zeros(1, numel(subj_info.(subj).(cond).file));
    lfp_pows = cell(1, numel(subj_info.(subj).(cond).file));
    
    % =========================================================================
    % FILE PROCESSING LOOP - PROCESS EACH RECORDING FILE
    % =========================================================================
    for f = 1:numel(subj_info.(subj).(cond).file)
        features = [];
        feat_labels = [];
        
        % Get file information
        file = subj_info.(subj).(cond).file{f};
        dataset = [subj_info.(subj).file_path, file];
        bad_meg = subj_info.(subj).(cond).bad_meg_channels{f};
        bad_lfp = subj_info.(subj).(cond).bad_lfp_channels{f};
        
        % ---------------------------------------------------------------------
        % Define trial structure from selected epochs
        % ---------------------------------------------------------------------
        hdr = ft_read_header(dataset);
        trl = subj_info.(subj).(cond).good_lfp_epochs{f};
        trl = [trl, zeros(size(trl, 1), 1)] .* hdr.Fs;
        if trl(1, 1) == 0, trl(1, 1) = 1; end
        
        % ---------------------------------------------------------------------
        % Load and preprocess MEG data
        % ---------------------------------------------------------------------
        cfg = [];
        cfg.dataset = dataset;
        cfg.dftfilter = 'yes';  % Remove powerline noise (50/60 Hz)
        cfg.trl = trl;
        cfg.channel = ['MEG***2', 'MEG***3', bad_meg];
        meg = ft_preprocessing(cfg);
        
        % ---------------------------------------------------------------------
        % Load and re-reference LFP data
        % ---------------------------------------------------------------------
        montage_avg_ref = subj_info.(subj).(cond).montage{f};
        montage_all.old = [montage_avg_ref.right_contacts_old, montage_avg_ref.left_contacts_old];
        montage_all.new = [montage_avg_ref.right_contacts_new, montage_avg_ref.left_contacts_new];
        
        cfg.channel = montage_all.old;
        cfg.reref = 'yes';
        cfg.refchannel = 'all';  % Average reference (left and right sides separately)
        lfp = ft_preprocessing(cfg);
        lfp.label = montage_all.new;
        
        % Track bad LFP channels after re-referencing
        bad_lfp_ind = false(1, numel(lfp.label));
        if ~isempty(bad_lfp)
            bad_ind = comp_cell_of_strings(bad_lfp, [montage_avg_ref.right_contacts_old, montage_avg_ref.left_contacts_old]);
            bad_lfp_ind(bad_ind) = true;
        end
        bad_lfp_avgref = [bad_lfp_avgref; lfp.label(bad_lfp_ind)];
        
        % ---------------------------------------------------------------------
        % Combine MEG and LFP data
        % ---------------------------------------------------------------------
        cfg = [];
        cfg.keepsampleinfo = 'no';
        data = ft_appenddata(cfg, meg, lfp);
        
        % Segment data into overlapping windows
        cfg = [];
        cfg.length = 2;  % 2-second segments
        cfg.overlap = 0.5;  % 50% overlap
        data = ft_redefinetrial(cfg, data);
        weights(f) = numel(data.trial);  % Number of trials for weighted averaging
        
        % ---------------------------------------------------------------------
        % Compute full-spectrum LFP power (1-400 Hz) for FOOOF analysis
        % ---------------------------------------------------------------------
        cfg = [];
        cfg.foi = foi;
        cfg.keeptrials = 'no';
        cfg.channel = lfp.label;
        cfg.method = 'mtmfft';
        cfg.output = 'pow';
        cfg.taper = 'hanning';
        lfp_pow = ft_freqanalysis(cfg, data);
        lfp_pows{f} = lfp_pow;
        
        % =====================================================================
        % FREQUENCY BAND LOOP - COMPUTE FEATURES FOR EACH BAND
        % =====================================================================
        for fr = 1:numel(freq_bands.label)
            band_label = freq_bands.label{fr};
            center_freq = freq_bands.centerfreq(fr);
            bandwidth = freq_bands.bandwidth(fr);
            
            % -----------------------------------------------------------------
            % Extract LFP power in current frequency band
            % -----------------------------------------------------------------
            cfg = [];
            cfg.foi = center_freq;
            cfg.tapsmofrq = bandwidth;
            cfg.keeptrials = 'no';
            cfg.channel = lfp.label;
            cfg.method = 'mtmfft';
            cfg.output = 'pow';
            cfg.taper = 'dpss';  % Discrete prolate spheroidal sequences for smoothing
            lfp_pow_band = ft_freqanalysis(cfg, data);
            
            % Store power features
            pow_feat = lfp_pow_band.powspctrm;
            pow_feat_labels = [[lfp.label', lfp.label'], repmat({band_label}, [numel(lfp.label), 1])];
            features = [features; pow_feat];
            feat_labels = [feat_labels; pow_feat_labels];
            
            % -----------------------------------------------------------------
            % Compute power and cross-spectral density for connectivity
            % -----------------------------------------------------------------
            cfg = [];
            cfg.foilim = [center_freq - bandwidth, center_freq + bandwidth];
            cfg.taper = 'hanning';
            cfg.keeptrials = 'no';
            cfg.method = 'mtmfft';
            cfg.output = 'powandcsd';  % Power and cross-spectral density
            freq = ft_freqanalysis(cfg, data);
            
            % -----------------------------------------------------------------
            % Handle special frequency bands to avoid line noise artifacts
            % Split bands around powerline harmonics (50, 250, 350 Hz)
            % -----------------------------------------------------------------
            if strcmp(band_label, 'low-gamma')
                % Split around 50 Hz line noise (35-48 Hz and 52-60 Hz)
                cfg = [];
                cfg.frequency = [35 48];
                cfg.avgoverfreq = 'yes';
                first = ft_selectdata(cfg, freq);
                
                cfg.frequency = [52 60];
                second = ft_selectdata(cfg, freq);
                
                % Average the two segments
                freq_avg = freq;
                freq_avg.powspctrm = (first.powspctrm + second.powspctrm) / 2;
                freq_avg.crsspctrm = (first.crsspctrm + second.crsspctrm) / 2;
                freq_avg.freq = center_freq;
                
            elseif strcmp(band_label, 'sHFO')
                % Split around 250 Hz harmonic (202-248 Hz and 252-298 Hz)
                cfg = [];
                cfg.frequency = [202 248];
                cfg.avgoverfreq = 'yes';
                first = ft_selectdata(cfg, freq);
                
                cfg.frequency = [252 298];
                second = ft_selectdata(cfg, freq);
                
                freq_avg = freq;
                freq_avg.powspctrm = (first.powspctrm + second.powspctrm) / 2;
                freq_avg.crsspctrm = (first.crsspctrm + second.crsspctrm) / 2;
                freq_avg.freq = center_freq;
                
            elseif strcmp(band_label, 'fHFO')
                % Split around 350 Hz harmonic (302-348 Hz and 352-398 Hz)
                cfg = [];
                cfg.frequency = [302 348];
                cfg.avgoverfreq = 'yes';
                first = ft_selectdata(cfg, freq);
                
                cfg.frequency = [352 398];
                second = ft_selectdata(cfg, freq);
                
                freq_avg = freq;
                freq_avg.powspctrm = (first.powspctrm + second.powspctrm) / 2;
                freq_avg.crsspctrm = (first.crsspctrm + second.crsspctrm) / 2;
                freq_avg.freq = center_freq;
                
            else
                % Standard bands without line noise issues
                cfg = [];
                cfg.frequency = [center_freq - bandwidth, center_freq + bandwidth];
                cfg.avgoverfreq = 'yes';
                freq_avg = ft_selectdata(cfg, freq);
            end

            % -----------------------------------------------------------------
            % Compute source-space connectivity for each LFP channel
            % -----------------------------------------------------------------
            for l = 1:numel(lfp.label)
                % DICS beamformer for source coherence estimation
                cfg = [];
                cfg.method = 'dics';  % Dynamic Imaging of Coherent Sources
                cfg.dics.lambda = '5%';  % Regularization parameter
                cfg.headmodel = hdm;
                cfg.grid = grid;
                cfg.frequency = center_freq;
                cfg.reducerank = 2;  % Rank reduction for numerical stability
                cfg.refchannel = lfp.label{l};  % Reference channel for coherence
                source = ft_sourceanalysis(cfg, freq_avg);
                source.pos = template_grid.pos;
                
                % Parcellate source coherence to AAL atlas regions
                cfg = [];
                cfg.parcellation = 'aal';
                cfg.method = 'mean';  % Average coherence within each parcel
                cfg.parameter = 'coh';
                source_parcel = ft_sourceparcellate(cfg, source, parcel);
                
                % Store connectivity features with labels
                conn_labels = [source_parcel.label', repmat(lfp.label(l), [numel(source_parcel.label), 1]), repmat({band_label}, [numel(source_parcel.label), 1])];
                features = [features; source_parcel.coh];
                feat_labels = [feat_labels; conn_labels];
            end
            
            % -----------------------------------------------------------------
            % Mark features from bad LFP channels as NaN
            % -----------------------------------------------------------------
            bad_lfp_new = unique(bad_lfp_avgref);
            has_bad_lfp_first_col = comp_cell_of_strings(bad_lfp_new, feat_labels(:, 1));
            has_bad_lfp_sec_col = comp_cell_of_strings(bad_lfp_new, feat_labels(:, 2));
            has_bad_lfp = unique(sort([has_bad_lfp_first_col(:); has_bad_lfp_sec_col(:)]));
            features(has_bad_lfp) = nan;
        end
        
        % Accumulate features from current file
        feats_file = [feats_file, features];
    end
    
    % =========================================================================
    % COMPUTE WEIGHTED AVERAGE ACROSS FILES
    % =========================================================================
    
    % Weighted average of features based on number of trials per file
    feat = zeros(size(feats_file, 1), 1);
    for j = 1:size(feats_file, 2)
        feat = feat + feats_file(:, j) * weights(j) / sum(weights);
    end
    save([feat_dir, subj], 'feat', 'feat_labels');
    
    % Weighted average of full-spectrum power
    lfp_power = zeros(size(lfp_pows{1}.powspctrm));
    for j = 1:numel(lfp_pows)
        lfp_power = lfp_power + lfp_pows{j}.powspctrm * weights(j) / sum(weights);
    end
    
    % Remove bad channels from power spectra
    lfp_labels = lfp_pows{1}.label;
    bad_lfp_new = unique(bad_lfp_avgref);
    has_bad_lfp = comp_cell_of_strings(bad_lfp_new, lfp_pows{1}.label);
    lfp_power(has_bad_lfp, :) = [];
    lfp_labels(has_bad_lfp) = [];
    
    % Save power spectra for FOOOF spectral parameterization
    save([foof_dir, subj], 'lfp_power', 'lfp_labels');
end
