import numpy as np
import pandas as pd
import json
import re
import pickle
import os
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, LeaveOneOut
from hyperopt import fmin, tpe, hp, Trials
from joblib import parallel_backend
import xgboost as xgb
from boruta import BorutaPy
from scipy.stats import zscore

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# create LOOelectrode scheme
def folds_sep_electrode(S):
    df = pd.DataFrame(S, columns=['S'])
    
    # parse the S to get the subject
    df['subject'] = df['S'].str.split('-').str[0]
    
    # parse the S to get the hemisphere
    df['hemisphere'] = df['S'].str.split('-').str[1]
    
    # parse the S to get the contact
    df['contact'] = df['S'].str.split('-').str[2]
    
    # create electrode column 
    df['electrode'] = df['subject'] + '-' + df['hemisphere']
    
    # get all electrode_labels
    all_electrode_labels = df['electrode'].values
    
    # get all unique electrode
    electrode = df['electrode'].unique()
    
    # get size of electrodes
    n_electrode = len(electrode)
    
    # partition 
    splits_at = np.arange(0, n_electrode, np.floor(n_electrode / n_electrode))
    splits_at = np.hstack([splits_at,np.array(n_electrode)])
    
    cv_inds = []
    for sp in np.arange(len(splits_at) - 1):
        split_ind1 = int(splits_at[sp])
        split_ind2 = int(splits_at[sp + 1])
        subjects_test = electrode[split_ind1:split_ind2]
        subjects_train = electrode[np.setdiff1d(np.arange(len(electrode)), np.arange(split_ind1, split_ind2))]
        inds_test_cv = np.nonzero(np.in1d(all_electrode_labels, subjects_test))[0]
        inds_train_cv = np.nonzero(np.in1d(all_electrode_labels, subjects_train))[0]
        cv_inds.append((inds_train_cv, inds_test_cv))
    return cv_inds

# create either LOOsubject scheme or Kfolds
def folds_sep_subjects(S,folds,shuffle=True):
    '''
    creates folds for cross-validation such that right and left hemisphere of any
    given subject is always in the same fold
    input:
    S - N_obseravationsx1 array of str - observation labels
    N_folds - int - number of folds
    shuffle - bool - shuffle observations before assigning to folds to eliminate order effects
    returns:
    cv_inds - list of len N_folds. contains train and test indices for each fold
    '''
    all_subj_labels = []
    for i in np.arange(len(S)):
        subj_match = re.findall(r'(S\d+|dbsai\d+)', S[i][0])  # Match S### or dbsai###
        all_subj_labels.append(subj_match[0] if subj_match else None)  # Avoid index error
        
    subjects = np.array(sorted(set(all_subj_labels)))
    N_subj = len(subjects)
    if shuffle:
        rand_ind = np.random.permutation(np.arange(0, len(subjects)))
        subjects = subjects[rand_ind]
    if isinstance(folds, int):
        N_folds = folds
    elif 'LOO' in folds:
        N_folds=len(subjects)
    # partition
    splits_at = np.arange(0, N_subj, np.floor(N_subj / N_folds))
    remain = np.mod(N_subj,N_folds)
    #if a remaining piece is at least 80% of the size of a usual fold it can become its own fold
    if remain==0 or remain>(0.8*np.floor(N_subj / N_folds)):
        splits_at = np.hstack([splits_at,np.array(N_subj)])
    #otherwise incorporate the piece into the last fold
    else:
        splits_at[-1] = N_subj
    cv_inds = []
    for sp in np.arange(len(splits_at) - 1):
        split_ind1 = int(splits_at[sp])
        split_ind2 = int(splits_at[sp + 1])
        subjects_test = subjects[split_ind1:split_ind2]
        subjects_train = subjects[np.setdiff1d(np.arange(len(subjects)), np.arange(split_ind1, split_ind2))]
        inds_test_cv = np.nonzero(np.in1d(all_subj_labels, subjects_test))[0]
        inds_train_cv = np.nonzero(np.in1d(all_subj_labels, subjects_train))[0]
        cv_inds.append((inds_train_cv, inds_test_cv))
    return cv_inds

def figure_out_best_params(model, X,y, search_spaces, search_space_names, score_metric,cv_inds, max_evals=100):
    '''
    input:
    model - instance of XGBRegressor
    X - N_obsxN_feat numpy array - features
    y - N_obsx1 numpy array - target
    search_spaces - list - search spaces (dictionaries) as used in hyperopt
    cv_inds - list of len N_folds. contains train and test indices for each fold
    score_metric - str - score metric used to evaluate fit
    search_space_names - list - names for search spaces
    returns:
    optimized_hyperparams - dictionary - optimized hyperparameters for XGBRegressor
    '''
    #this is the function to optimize
    def xgb_loss_cv(params):
        if 'max_depth' in params.keys():
            params['max_depth'] = int(params['max_depth'])
        if 'n_estimators' in params.keys():
            params['n_estimators'] = int(params['n_estimators'])
        model.set_params(**params)
        score = -cross_val_score(model, X, y, cv=cv_inds, scoring=score_metric, n_jobs=-1).mean()
        return score
    
    #search spaces simultaneously
    big_space = {key: value for sp in search_spaces for (key, value) in sp.items()}
    print(f'searching space: {search_space_names}')
        
    trials = Trials()
    params = fmin(fn=xgb_loss_cv,
                space=big_space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)
    if 'max_depth' in params.keys():
        params['max_depth'] = int(params['max_depth'])
    if 'n_estimators' in params.keys():
        params['n_estimators'] = int(params['n_estimators'])
    return params

def zscore_electrode(X, S, feat_labels_plot, axis=0, norm='all'):
    """
    Applies z-scoring to X based on subject-electrode groups and frequency bands.
    
    - Standardizes **power and coherence separately** per **frequency and subject-electrode**.
    - Handles test/train subject-electrode naming differences.
    - Skips standardization if an electrode has only **one contact**.
    - Allows selecting normalization mode: 'all', 'coh' (coherence only), 'pow' (power only).
    
    Parameters:
    - X: ndarray of shape (N_contacts, N_features) → Input data matrix.
    - S: ndarray of shape (N_contacts, 1) → Subject-electrode labels.
    - feat_labels_plot: list of str → Feature labels.
    - norm: str → 'all' (default), 'coh' (coherence only), 'pow' (power only).

    Returns:
    - X_scaled: ndarray of shape (N_contacts, N_features) → Z-scored data matrix.
    """

    # Extract subject-electrode combinations
    pattern = r'(dbsai\d+-(?:left|right)|S\d+-(?:left|right))'
    subject_electrodes = np.unique([re.match(pattern, s[0]).group(1) for s in S if re.match(pattern, s[0])])
    
    # Copy X to avoid modifying the original data
    X_scaled = np.copy(X)
    ALL_FREQUENCY = config['FEATURES']['ALL_FREQUENCY']

    # Separate indices for power and coherence features
    power_idx = {freq: [i for i, feat in enumerate(feat_labels_plot) if 'pow' in feat and freq in feat] for freq in ALL_FREQUENCY}
    coherence_idx = {freq: [i for i, feat in enumerate(feat_labels_plot) if 'pow' not in feat and freq in feat] for freq in ALL_FREQUENCY}

    # Standardize within each subject-electrode group
    for subj_el in subject_electrodes:
        mask = np.array([s[0].startswith(subj_el) for s in S])  # Select rows for this subject-electrode
        num_contacts = np.sum(mask)  # Count number of contacts

        for freq in ALL_FREQUENCY:  # Process each frequency separately
            # Standardize power separately if selected
            if norm in ['all', 'pow'] and power_idx[freq] and num_contacts > 1:
                X_scaled[np.ix_(mask, power_idx[freq])] = zscore(X[mask][:, power_idx[freq]], axis=0) 

            # Standardize coherence separately if selected
            if norm in ['all', 'coh'] and coherence_idx[freq] and num_contacts > 1:
                X_scaled[np.ix_(mask, coherence_idx[freq])] = zscore(X[mask][:, coherence_idx[freq]], axis=axis)
                
    return X_scaled  # Return transformed data


def select_feature_boruta_nested(X, y, feature_names, alpha, max_iter, n_splits, threshold, verbose=False):
    """Nested feature selection using Boruta with consensus across sub-folds."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    selected_features_per_fold = []
    
    # Random Forest estimator with fixed params for stability
    forest = RandomForestRegressor(random_state=42, n_jobs=-1, max_depth=5)
    
    for fold_idx, (train_idx, _) in enumerate(kf.split(X)):
        if verbose: print(f'Running Boruta for fold {fold_idx + 1}')
        
        X_train, y_train = X[train_idx], y[train_idx].ravel()
        
        with parallel_backend('loky', n_jobs=-1):
            boruta = BorutaPy(estimator=forest, n_estimators='auto', verbose=verbose, 
                              random_state=42, alpha=alpha, max_iter=max_iter)
            
            # Boruta needs a 1D array for y, so we use y_train.ravel()
            boruta.fit(X_train, y_train)
        
        selected_features_per_fold.append(boruta.support_)
    
    selected_features_per_fold = np.array(selected_features_per_fold)
    
    # Calculate selection frequency
    feature_selection_frequency = np.mean(selected_features_per_fold, axis=0)
    
    df_freq = pd.DataFrame({'features': feature_names, 'freq': feature_selection_frequency})
    
    # Consensus mask
    consensus_mask = feature_selection_frequency >= threshold
    
    n_consensus_features = np.sum(consensus_mask)
    print(f'Boruta Nested: Consensus threshold {threshold}. Kept {n_consensus_features} features.')
    
    return df_freq, consensus_mask

# --- PREDICTOR CORE CLASS ---
class Predictor:
    def __init__(self, config):
        self.config = config
        self.model_cfg = config['MODEL']
        self.feat_cfg = config['FEATURES']
        self.search_space = self.model_cfg['SEARCH_SPACE']
        self.results_dir = config['PATHS']['PREDICTOR_OUTPUT']

    def _get_cv_indices(self, S, cv_scheme):
        """Returns CV indices based on the chosen scheme."""
        if cv_scheme == 'LOOelectrode':
            return folds_sep_electrode(S)
        elif cv_scheme == 'LOOsubject':
            return folds_sep_subjects(S, cv_scheme, shuffle=False)
        elif cv_scheme == 'LOO':
            return list(LeaveOneOut().split(S))
        elif isinstance(cv_scheme, int): # for Kfolds
            return folds_sep_subjects(S, cv_scheme, shuffle=True)
        else:
            raise ValueError(f"Unknown CV scheme: {cv_scheme}")

    def run_prediction(self, data, run_label):
        """Runs the full nested cross-validation, feature selection, and prediction pipeline."""
        X, y, S, feat_labels_plot = data
        
        # Initialize storage
        cv_scheme = self.model_cfg['DEFAULT_CV']
        idx_folds = self._get_cv_indices(S, cv_scheme)
        
        # Initialize arrays for stacking results
        y_pred_all = np.array([])
        y_test_all = np.array([])
        S_test_all = np.array([])
        feat_labels_selected_all = []
        
        # LOO
        for f, (ind_train, ind_test) in enumerate(idx_folds):
            print(f"--- Running Fold {f+1}/{len(idx_folds)} ({cv_scheme}) ---")

            X_train = X[ind_train,:]
            y_train = y[ind_train]
            S_train = S[ind_train]
            X_test = X[ind_test,:]
            y_test = y[ind_test]
            S_test = S[ind_test]
            
            # Reshape X_test if it's a single sample in the case of LOO (not subject or electrode)
            if np.ndim(X_test) == 1: X_test = X_test.reshape(1, X_test.size)
            
            # Nested CV indices for tuning/selection on train fold
            cv_inds_tune = self._get_cv_indices(S_train, 3) # Use Kfold for internal tuning stability

            # 1. Z-Score Coherence Features
            # zscore the features before the selection (here we are using the X_train afterwards wthout zscoring but possible to zscore the x before)
            X_feat_find = zscore_electrode(X_train, S_train, feat_labels_plot, axis=1, norm='coh')
            print("Feature Selection (Boruta Nested)...")
            df_freq, consensus_mask = select_feature_boruta_nested(X_feat_find, y_train, feat_labels_plot,             
                          n_splits=self.model_cfg['BORUTA_N_SPLITS'], 
                          threshold=self.model_cfg['BORUTA_THRESHOLD'],
                          alpha=0.05, max_iter=100, verbose=False)

            # Save feature selection frequency for analyser
            feat_labels_selected_all.append(df_freq) 
            
            # Apply mask to data and feature labels
            X_train = X_train[:, consensus_mask]
            X_test = X_test[:, consensus_mask]
            best_feat_labels = feat_labels_plot[consensus_mask]
            print(f"Features selected: {len(best_feat_labels)}")

            # 2. Create naive xgboost model with basic parameter
            model = XGBRegressor(learning_rate=0.05, n_estimators=2000, max_depth=6, n_jobs=-1,
                                objective='reg:squarederror', tree_method='exact', eval_metric='rmse',
                                random_state=42, seed=42)
        
            # 3. Find best number of trees 
            xgtrain = xgb.DMatrix(X_train, label=y_train.ravel())
            cv_results = xgb.cv(model.get_xgb_params(), xgtrain, nfold=10,
                                num_boost_round=model.n_estimators, early_stopping_rounds=50)
            print('Best number of trees found after CV: ', cv_results.shape[0])
            model.set_params(n_estimators=cv_results.shape[0])
        
            # 4. Tune hyperparameters
            print("Hyperparameter Tuning ...")
            space1 = {'min_child_weight': hp.quniform('min_child_weight', 1, 15, 1)}
            space2 = {'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1)}
            space3 = {'subsample': hp.uniform('subsample', 0.5, 1)}            
            space4 = {'gamma': hp.uniform('gamma', 0, 0.1)}
            space5 = {'reg_alpha': hp.uniform('reg_alpha', 0, 1)}
            space6 = {'reg_lambda': hp.uniform('reg_lambda', 0, 1)}
            
            search_spaces = [space1, space2, space3, space4, space5, space6]
            search_space_names = ['min_child_weight', 'colsample_bytree',
            'subsample', 'gamma', 'reg_alpha', 'reg_lambda']
            
            best_params = figure_out_best_params(
                model=model, X=X_train, y=y_train,
                search_spaces=search_spaces, search_space_names=search_space_names,
                score_metric='neg_mean_squared_error', cv_inds=cv_inds_tune, max_evals=self.model_cfg['MAX_EVALS'])
            
            model.set_params(**best_params)
            print(f"Best parameters: {best_params}")
            
            # Fit the model with the best hyperparameters and features and then make prediction 
            model.fit(X_train, y_train.ravel())
            model.get_booster().feature_names = best_feat_labels.tolist()
            y_pred_fold = model.predict(X_test)
             
            # Stack results
            y_pred_all = np.hstack([y_pred_all, y_pred_fold.flatten()])
            y_test_all = np.hstack([y_test_all, y_test.flatten()])
            S_test_all = np.hstack([S_test_all, S_test.flatten()])
            
        # 6. Save final stacked results
        stacked_results = {
            'y_pred': y_pred_all,
            'y_test': y_test_all,
            'S_test': S_test_all,
            'feat_labels_all': feat_labels_plot, # The original set of all features
            'feat_selection_data': pd.concat(feat_labels_selected_all, ignore_index=True),
        }
        
        # Create unique run label based on fixed parameters
        save_path = os.path.join(self.results_dir, run_label, 'predict.pkl')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, "wb") as f:
            pickle.dump(stacked_results, f)
            
        print("\n--- Prediction Complete ---")
        print(f"Results saved to: {save_path}")
        
        return stacked_results

