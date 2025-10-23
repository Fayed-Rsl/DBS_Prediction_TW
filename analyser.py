import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from scipy.stats import pearsonr
import pickle

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Define constants from config for convenience
ALL_FREQUENCY = config['FEATURES']['ALL_FREQUENCY']
ALL_ROIS = config['FEATURES']['ALL_ROIS']
ALL_FREQUENCY_SHORT2 = ['θ', 'α', 'Low-β', 'High-β', 'Low-γ', 'High-γ', 'sHFO', 'fHFO']

# --- ANALYSER CORE CLASS ---
class Analyser:
    """Handles loading prediction results, computing performance metrics,
    and generating key figures based on Boruta feature selection frequency,
    including the Cumulative Hit Ratio analysis."""

    def __init__(self, config):
        self.config = config
        self.results_dir = config['PATHS']['PREDICTOR_OUTPUT']
        self.min_contacts = config['PROJECT']['MIN_CONTACTS_FOR_HIT']

    # --- CORE METRIC METHODS ---
    def _process_results(self, stacked_results):
        """Computes correlation, RMSE, and error metrics from stacked data."""
        y_pred = stacked_results['y_pred']
        y_test = stacked_results['y_test']
        
        r, p = pearsonr(y_pred, y_test)
        
        error_fold = y_pred - y_test
        error_null_fold = np.mean(y_test) - y_test # Null model is mean of test set (for comparison)

        rmse = np.sqrt(np.mean(np.square(error_fold)))
        rmse_c = np.sqrt(np.mean(np.square(error_null_fold)))
        
        return {'r': r, 'p': p, 'rmse': rmse, 'rmse_c': rmse_c}

    # --- FEATURE IMPORTANCE METHODS ---
    def _get_feature_ranking_boruta(self, stacked_results):
        """Calculates feature importance based *only* on Boruta selection frequency."""
        
        feat_labels_plot = stacked_results['feat_labels_all']
        feat_sel_data = stacked_results['feat_selection_data']
        
        selection_frequency = feat_sel_data.groupby('features')['freq'].mean().reset_index()
        selection_frequency.rename(columns={'features': 'features_label', 'freq': 'selection_frequency'}, inplace=True)
        
        full_features_df = pd.DataFrame({'features_label': feat_labels_plot})
        ranking = full_features_df.merge(selection_frequency, on='features_label', how='left')
        ranking['selection_frequency'] = ranking['selection_frequency'].fillna(0)
        
        def clean_feature(df):
            df = df.copy()
            df['rois'] = ''
            df['freq'] = ''
            
            for i, row in df.iterrows():
                label_elements = row['features_label'].split(' ')
                
                if 'pow' in row['features_label']:
                    this_roi = 'LFP' 
                    this_freq = label_elements[-1]
                else:
                    this_roi = label_elements[3] 
                    this_freq = label_elements[-1]
                
                df.loc[i, 'rois'] = this_roi.replace('Senorimotor', 'Sensorimotor')
                df.loc[i, 'freq'] = this_freq
            return df

        ranking = clean_feature(ranking)
        ranking['rois'] = ranking['rois'].replace({'Angular': 'Parietal', 'SupraMarginal': 'Parietal', 'LFP': 'STN'})
        ranking['freq'] = pd.Categorical(ranking['freq'], categories=ALL_FREQUENCY, ordered=True)
        ranking = ranking.sort_values(by='selection_frequency', ascending=False).reset_index(drop=True)        
        return ranking

    # --- CUMULATIVE HIT RATIO METHODS ---
    def _create_contact_ranking(self, S, pred, true):
        """Converts raw prediction/true data into a DataFrame structured by electrode contact."""
        contact_ranking = pd.DataFrame({'S': S, 'pred': pred, 'true': true})
        
        # Parse the sample label S (Subject-Hemisphere-Contact)
        contact_ranking['subject'] = contact_ranking['S'].str.split('-').str[0]
        contact_ranking['hemisphere'] = contact_ranking['S'].str.split('-').str[1]
        contact_ranking['contact'] = contact_ranking['S'].str.split('-').str[2]
        
        # Calculate the error for RMSE visualization (if needed, though not used in hit ratio)
        contact_ranking['error'] = contact_ranking['pred'] - contact_ranking['true']
        
        return contact_ranking.sort_values(by=['subject', 'hemisphere', 'S']).reset_index(drop=True)

    def _load_active_contact(self):
        """
        Loads the ground truth active contacts.
        """
        # read the excel file that hold the active contact
        active_contact = pd.read_excel('/home/rasfay01/DBS_Prediction_TW/active_contact.xlsx')
        # transform the ID 1 into S001 and so on
        active_contact['subject'] = active_contact['ID'].apply(lambda x: f'S{x:03d}')
        # drop the ID column
        active_contact = active_contact.drop('ID', axis=1)
        # create the Sample label column (subject-hemisphere-contact)
        active_contact['S'] = active_contact['subject'] + '-' + active_contact['hemisphere'] + '-' + active_contact['contact'].astype(str)
        # put the S column at the beginning of the dataframe
        active_contact = active_contact[['S', 'subject', 'hemisphere', 'contact']]
        # create a column named active_contact and set it to 1
        active_contact['active_contact'] = 1  
        return active_contact


    def _run_hit_ratio(self, contact_ranking_df, ascending_sort, n_perm, ranking_source='model'):
        """
        Performs the ranking and permutation test to calculate hit ratio.
        Used internally by hit_model.
        """
        
        if ranking_source == 'model':
            sort_col = 'pred'
        elif ranking_source == 'random':
            sort_col = 'shuffled_pred'
            
        # 1. Rank contacts based on prediction/shuffled prediction (within electrode)
        # TW is best when prediction is high, so ascending=False.
        ranking_df = contact_ranking_df.copy()
        ranking_df['rank'] = ranking_df.groupby(['subject', 'hemisphere'])[sort_col].rank(
            method='first', # Breaking ties consistently
            ascending=ascending_sort
        ).astype(int)
        
        # 2. Get the list of unique electrodes that have an active contact
        keep_electrode = ranking_df[ranking_df['active_contact'] == 1].drop_duplicates(subset=['subject', 'hemisphere'])
        total_electrodes = len(keep_electrode)        

        # 3. Filter for only electrodes containing an active contact
        ranking_df['electrode_id'] = ranking_df['subject'] + '-' + ranking_df['hemisphere']
        keep_electrode_ids = keep_electrode['subject'] + '-' + keep_electrode['hemisphere']
        
        ranking_filtered = ranking_df[ranking_df['electrode_id'].isin(keep_electrode_ids)].copy()
        
        max_rank = ranking_filtered['rank'].max()
        cumulative_hit = np.zeros(max_rank)

        for r in range(1, max_rank + 1):
            # Count how many active contacts were found at this rank or better
            hits_at_rank = ranking_filtered[(ranking_filtered['active_contact'] == 1) & (ranking_filtered['rank'] <= r)]
            # Count unique electrodes hit
            unique_electrodes_hit = hits_at_rank.drop_duplicates(subset=['subject', 'hemisphere'])
            
            # Cumulative hits up to rank r
            cumulative_hit[r-1] = len(unique_electrodes_hit)
            
        # Convert to cumulative percentage
        cumulative_hit_percentage = (cumulative_hit / total_electrodes) * 100
        
        return cumulative_hit_percentage, ranking_filtered


    def hit_model(self, stacked_results, n_perm=1000):
        """
        Calculates the model's cumulative hit ratio and compares it against random chance
        using permutation testing.
        """
        
        # TW: Higher TW is better, so the best contact has the highest prediction/true value.
        ascending_sort = False 

        # 1. Prepare base ranking dataframe
        base_df = self._create_contact_ranking(
            stacked_results['S_test'], stacked_results['y_pred'], stacked_results['y_test']
        )
        
        # 2. Load ground truth active contacts
        active_contacts = self._load_active_contact()
        
        # 3. Merge ground truth back into the ranking data. 1 the active contact, 0 non active contact.
        ranking_data = base_df.merge(active_contacts[['S', 'active_contact']], on='S', how='left')
        ranking_data['active_contact'] = ranking_data['active_contact'].fillna(0).astype(int)
        # keep only a minimum of 4 contacts for thte permutation
        ranking_data = ranking_data.groupby(['subject', 'hemisphere']).filter(lambda x: len(x) >= self.min_contacts) 

        # 4. Calculate Model Hit Ratio
        model_hit_percentage, ranking_filtered = self._run_hit_ratio(
            ranking_data, ascending_sort, n_perm, ranking_source='model'
        )
        
        max_rank = len(model_hit_percentage)
        random_results_matrix = np.zeros((n_perm, max_rank))
        
        # 5. Permutation Test (Random Hit Ratio)
        print(f"Starting permutation test with {n_perm} iterations...")
        
        for i in range(n_perm):
            shuffled_df = ranking_data.copy()
            
            # Shuffle predictions PER-ELECTRODE to simulate random chance within the available options
            shuffled_df['shuffled_pred'] = shuffled_df.groupby(['subject', 'hemisphere'])['pred'].transform(
                lambda x: np.random.permutation(x.values)
            )
            
            random_hit, _ = self._run_hit_ratio(
                shuffled_df, ascending_sort, n_perm, ranking_source='random'
            )
            
            random_results_matrix[i, :] = random_hit


        return model_hit_percentage, random_results_matrix, ranking_filtered

    # --- PLOTTING UTILITIES ---
    def plot_hit_model(self, model_hit_percentage, random_results_matrix):
        """Plots the cumulative hit ratio with permutation confidence bounds."""
        
        max_rank = len(model_hit_percentage)
        ranks = np.arange(1, max_rank + 1)
        
        # Calculate mean and confidence intervals for random chance
        mean_random = np.mean(random_results_matrix, axis=0)
        # Use 95% confidence interval (2.5th and 97.5th percentiles)
        ci_lower = np.percentile(random_results_matrix, 2.5, axis=0)
        ci_upper = np.percentile(random_results_matrix, 97.5, axis=0)

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot Random Confidence Interval (Shaded area)
        ax.fill_between(ranks, ci_lower, ci_upper, color='gray', alpha=0.3, label='Random Chance (95% CI)')
        
        # Plot Random Mean (Dashed line)
        ax.plot(ranks, mean_random, color='gray', linestyle='--', linewidth=2, label='Random Chance (Mean)')
        
        # Plot Model Performance (Solid line)
        ax.plot(ranks, model_hit_percentage, color='darkblue', linewidth=3, marker='o', label='Model Prediction')

        # Identify significant ranks (where model exceeds 97.5th percentile of random chance)
        for r in range(max_rank):
            if model_hit_percentage[r] > ci_upper[r]:
                # Add star above the model line if significant
                ax.text(ranks[r], model_hit_percentage[r] + 1, '*', 
                        ha='center', va='bottom', fontsize=12, fontweight='bold', color='darkblue')

        # Customize plot
        ax.set_xticks(ranks)
        ax.set_xlabel('Contact Rank (1=Best Predicted)')
        ax.set_ylabel('Cumulative Hit Ratio (%)')
        ax.set_title('Model Performance: Cumulative Hit Ratio vs. Random Chance')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend()
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, feature_ranking):
        """Plots feature importance as a stacked bar chart (Region vs Frequency)."""
        # (Content unchanged, using feature_ranking for plotting)
        
        importance_df = feature_ranking.copy()
        importance_grouped = importance_df.groupby(['rois', 'freq'])['selection_frequency'].sum().unstack(fill_value=0)
        importance_plot = importance_grouped * 100
        
        display_rois = ['STN', 'Sensorimotor', 'Frontal', 'Parietal', 'Temporal', 'Occipital', 'Cerebellum']
        importance_plot = importance_plot.reindex(display_rois, fill_value=0).T
        importance_plot.index = ALL_FREQUENCY_SHORT2
        
        fig, ax = plt.subplots(figsize=(10, 6))
        importance_plot.plot(kind='bar', stacked=True, ax=ax, cmap='viridis', rot=45)

        ax.set_title('Feature Importance (Boruta Selection Frequency)')
        ax.set_ylabel('Selection Frequency Score (%)')
        ax.set_xlabel('Frequency Band')
        plt.tight_layout()
        return fig

    def plot_prediction_vs_true(self, stacked_results):
        """Plots predicted TW vs. True TW with."""
        data = pd.DataFrame({
            'True': stacked_results['y_test'].flatten(),
            'Pred': stacked_results['y_pred'].flatten()
        })
                
        return sns.lmplot(data, x='True', y='Pred')

# --- ANALYSER EXECUTION FUNCTION ---
def run_analyser(config, run_label, n_perm=10000):
    """Loads prediction results and generates plots."""
    results_dir = config['PATHS']['PREDICTOR_OUTPUT']
    load_path = os.path.join(results_dir, run_label, 'predict.pkl')
    
    try:
        with open(load_path, 'rb') as f:
            stacked_results = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Results file not found at {load_path}. Run predictor.py first.")
        return

    analyser = Analyser(config)
    
    # 1. Performance Metrics
    metrics = analyser._process_results(stacked_results)
    print("\n--- Model Performance Metrics ---")
    print(f"Pearson R: {metrics['r']:.4f} (p={metrics['p']:.4f})")
    print(f"RMSE (Model): {metrics['rmse']:.4f}")
    print(f"RMSE (Null Model): {metrics['rmse_c']:.4f}")
    
    # 2. Feature Importance
    feature_ranking = analyser._get_feature_ranking_boruta(stacked_results)
    print("\n--- Top 10 Features (Boruta Frequency) ---")
    print(feature_ranking[['features_label', 'selection_frequency']].head(10))
    
    # 3. Cumulative Hit Ratio Calculation
    model_hit, random_matrix, ranking_data = analyser.hit_model(stacked_results, n_perm=n_perm)
    
    print("\n--- Cumulative Hit Ratio ---")
    df_hit = pd.DataFrame({
        'Rank': np.arange(1, len(model_hit) + 1),
        'Model Hit (%)': model_hit,
        'Random Mean (%)': np.mean(random_matrix, axis=0),
        'Electrodes (N)': len(ranking_data.drop_duplicates(subset=['subject', 'hemisphere']))
    })
    print(df_hit)
    
    # 4. Plotting
    fig_feat = analyser.plot_feature_importance(feature_ranking)
    fig_pred = analyser.plot_prediction_vs_true(stacked_results)
    fig_hit = analyser.plot_hit_model(model_hit, random_matrix)
    
    # Save figures
    plot_dir = os.path.join(results_dir, run_label, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    fig_feat.savefig(os.path.join(plot_dir, 'feature_importance.png'))
    fig_pred.savefig(os.path.join(plot_dir, 'prediction_vs_true.png'))
    fig_hit.savefig(os.path.join(plot_dir, 'cumulative_hit_ratio.png'))
    
    print(f"\nPlots saved to: {plot_dir}")
