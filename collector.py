import numpy as np
import re
import os
import pandas as pd
from scipy.io import loadmat
import pickle
from collector_utils import select_features, get_plot_labels, get_side_inds, apply_side_inds, replace_pow_with_fooof 

# --- TW CALCULATION ---
class MonopolarTWCalculator:
    """Handles loading MPR data, cleaning, creating ring/segment contacts,
    and calculating the Therapeutic Window (TW) normalized per-subject.
    Focuses only on TW for 'More' MPR date and 'Substantial' effect.
    Ignore SET/TET collection and other parameter for simpler implementation.
    """
    def __init__(self, config):
        self.config = config
        self.mpr_path = self.config['PATHS']['MPR_FILENAME']
        self._max_tested = self.config['COLLECTION']['MAX_TESTED_MA']
        self._min_tested = self.config['COLLECTION']['MIN_TESTED_MA']

    def _load_and_clean_data(self):
        """Load MPR data and perform initial cleaning and contact normalization.
        Already cleaned the MPR data for simpler implementation. """
        df = pd.read_excel(self.mpr_path)
        return df

    def _get_set_tet_param(self, df):
        """Filter for Substantial effect and select the 'More' contact available MPR date."""
        # Filter for Substantial Effect (Transient/Partial Side Effects are removed)
        set_df = df[df['Effect Kind'] == 'Side Effect'].copy()
        tet_df = df[df['Effect Kind'] == 'Benefit'].copy()
        
        set_df = set_df[set_df['Effect Time'] != 'Transient']
        tet_df = tet_df[tet_df['Effect Time'] != 'Partial']

        # Select 'More' MPR Date (date with the highest number of entries per subject)
        def filter_by_mpr_strategy(sub_df):
            if sub_df.empty: return sub_df
            # Count contacts per date for each subject
            counts = sub_df.groupby(['ID', 'Monopolar Date']).size().reset_index(name='count')
            # Get the date with the highest count per subject
            best_dates = counts.loc[counts.groupby('ID')['count'].idxmax()]
            
            # Merge to filter the original DataFrame
            return sub_df.merge(best_dates[['ID', 'Monopolar Date']], on=['ID', 'Monopolar Date'])

        set_df = filter_by_mpr_strategy(set_df)
        tet_df = filter_by_mpr_strategy(tet_df)
        
        set_df = set_df.groupby(['ID', 'Hemisphere', 'Contact', 'Monopolar Date', 'Date Surgery'], as_index=False).first()
        tet_df = tet_df.groupby(['ID', 'Hemisphere', 'Contact', 'Monopolar Date', 'Date Surgery'], as_index=False).first()
        
        # return the SET and TET df to compute TW later on 
        return set_df, tet_df

    def _tw_clean(self, tw):
        '''Avoid the case where a subject has the same contact but not the same monopolar review date 
    for example:
    ID       left    3ABC  ...  Side Effect     2018-03-28   2018-03-22
    ID       left    3ABC  ...      Benefit     2019-05-15   2018-03-22
    we always want to compute TW within the same monopolar review
    '''
        # create a small dataframe with ID, Date and Contact and count of contact for each monopolar review
        small = tw.groupby(['ID', 'Monopolar Date'])['Contact'].count().reset_index()
        
        # at this stage, we need to only have one monopolar review to compute the therapeutic window
        small = small[small.duplicated('ID', keep=False)]
        
        # check if the small df is not empty
        if not small.empty:
            # sort the small dataframe by ID contact and Monopolar date
            monopolar_sorted = small.sort_values(by=['ID', 'Contact', 'Monopolar Date'], ascending=[True, False, True]).reset_index(drop=True)
            
            # keep the monopolar that is: the earliest and that has the most contact
            monopolar_sorted.drop_duplicates(subset=['ID'], keep='first', inplace=True)
            
            # find the index from each subject in the tw that are not in the monopolar sorted selection
            index = tw[(tw.ID.isin(monopolar_sorted.ID) & 
                           ~(tw['Monopolar Date'].isin(monopolar_sorted['Monopolar Date'])))].index
            
            # remove the index from the _tw that where duplicated in the monopolar
            tw = tw.drop(index).reset_index(drop=True)
            return tw
        
        else:
            return tw 

    def _tw_normalize_per_subject(self, set_df, tet_df):
        """
        Calculates the normalized TW: (SET - TET) / (max_tested_subj - min_tested_subj).
        Handles NaNs by replacing missing SET/TET with subject's max/min tested mA respectively.
        This is equivalent to the original _tw_rel_sub logic.
        """
        mpr_combined = pd.concat([set_df, tet_df], axis=0).sort_values(by=['ID', 'Hemisphere', 'Contact']).reset_index(drop=True)
        mpr_combined = self._tw_clean(mpr_combined)
        
        # 1. Determine Max/Min Tested mA per subject        
        # Get actual tested mA values for each subject
        tested_values = mpr_combined.dropna(subset=['mA']).groupby(['ID'])['mA']
        max_tested_per_subj = tested_values.max()
        min_tested_per_subj = tested_values.min()

        max_ma = max_tested_per_subj.reindex(mpr_combined['ID']).values
        min_ma = min_tested_per_subj.reindex(mpr_combined['ID']).values

        mpr_combined['max_tested'] = max_ma
        mpr_combined['min_tested'] = min_ma
        
        # Calculate TW
        tw_wide = (mpr_combined
            .pivot_table(index=['ID', 'Hemisphere', 'Contact', 'Monopolar Date', 'Date Surgery', 'max_tested', 'min_tested'], 
                         columns='Effect Kind', 
                         values='mA')
            .rename(columns={'Side Effect': 'SET', 'Benefit': 'TET'})
            .reset_index()
            .rename_axis(None, axis=1)
        )
        
        # Fill NaN SET with subject's max_tested (worst case scenario)
        tw_wide['SET'] = tw_wide.apply(lambda row: row['max_tested'] if pd.isna(row['SET']) else row['SET'], axis=1)
        # Fill NaN TET with subject's min_tested (worst case scenario)
        tw_wide['TET'] = tw_wide.apply(lambda row: row['min_tested'] if pd.isna(row['TET']) else row['TET'], axis=1)

        # Normalize TW: (SET - TET) / (max_tested - min_tested)
        tw_wide['TW'] = (tw_wide['SET'] - tw_wide['TET']) / tw_wide['max_tested'] - tw_wide['min_tested']
    
        return tw_wide[['ID', 'Hemisphere', 'Contact', 'Monopolar Date', 'Date Surgery', 'SET', 'TET', 'TW']]

    def get_therapeutic_window(self):
        """Main method to load, clean, filter, and calculate the TW."""
        df = self._load_and_clean_data()
        set_df, tet_df = self._get_set_tet_param(df)
        tw_df = self._tw_normalize_per_subject(set_df, tet_df)
        
        # Rename ID to match feature file format 'S001': matching the ID pattern with leading 0 and S in front
        tw_df['ID'] = tw_df['ID'].apply(lambda x: 'S' + str(int(x)).zfill(3))
        
        return tw_df


def _create_ring_averages(features, feat_labels, chans_this_side, side, target_contacts, all_frequency):
    """
    Creates averaged features for ring contacts (e.g., 2ABC) if they are present in the
    target_contacts and the individual segments (2A, 2B, 2C) are present in the features.
    """
    
    # Identify ring contacts in the TW target list
    pattern = re.compile(r'[A-Z]{2,3}')
    chans_short_ring = [c for c in target_contacts if pattern.search(c)]
    
    if not chans_short_ring:
        return features, feat_labels, chans_this_side, target_contacts
    
    # Convert feature data to DataFrame for easier manipulation
    feat_df = pd.DataFrame(features.flatten(), columns=['coherence'])
    feat_df['source'] = [t[0] for t in feat_labels]
    feat_df['lfpchannels'] = [t[1] for t in feat_labels]
    feat_df['frequency'] = [t[2] for t in feat_labels]

    # Process each ring contact
    for ring_id in chans_short_ring:
        
        # 1. Determine the component segments (e.g., '2ABC' -> ['2A', '2B', '2C'])
        number = ring_id[0]
        letters = list(ring_id[1:])
        segments_short = [number + l for l in letters]
        
        # 2. Check if component segments are available in the current MEG/LFP features
        segments_full = [f'LFP-{side}-{s}' for s in segments_short]
        available_segments = [s for s in segments_full if s in chans_this_side]
        
        if len(available_segments) == len(segments_full): # All parts present
            
            # Filter the DataFrame to include only features from the constituent segments
            data_segments = feat_df[feat_df['lfpchannels'].isin(available_segments)]
            
            # create a new dataframe with the average of the ring by source and frequency, the average coherence
            data_ring = data_segments.groupby(['source', 'frequency'])['coherence'].mean().reset_index()

            # average only for the LFP power --> when the source is LFP
            lfp_avg = data_ring[data_ring['source'].str.contains('LFP')]
            lfp_avg = lfp_avg.groupby(['frequency'])['coherence'].mean().reset_index()
            lfp_avg['source'] = 'LFP-' + side + '-' + ring_id # 'LFP-right-3ABC'

            # drop the rows where the source is LFP to remove the power and only keep coherence.
            data_ring = data_ring[~data_ring['source'].str.contains('LFP')]

            # add the average of the LFP to the data_ring
            data_ring = pd.concat([data_ring, lfp_avg])

            # add the lfpchannels label to the data_ring
            new_lfp_channel_label = f'LFP-{side}-{ring_id}'
            data_ring['lfpchannels'] = new_lfp_channel_label
            
            # add the data_ring to the data
            feat_df = pd.concat([feat_df, data_ring])
    
            # Update available channels and target contacts to include the new averaged ring.
            chans_this_side.append(new_lfp_channel_label)
            
        else:
            # If the ring components are missing, remove the ring from target_contacts so it's not processed later
            # Find index in target_contacts and remove
            target_contacts.remove(ring_id) 

    feat_df = feat_df.reset_index(drop=True)

    # extract the feat_labels and features from the data and make it the same format as before
    new_feat_labels = []
    for i in range(len(feat_df)):
        new_feat_labels.append((feat_df['source'][i], feat_df['lfpchannels'][i], feat_df['frequency'][i]))
    # features is a numpy array, containing the values in a numpy array of shape (n, 1)
    new_features = feat_df['coherence'].values.reshape(-1, 1)

    # replace the feat_labels and features by the new_feat_labels and new_features
    feat_labels = new_feat_labels
    features = new_features
    
    return features, feat_labels, chans_this_side, target_contacts



def collect_data(config):
    """
    Collects Monopolar Review data and corresponding MEG-LFP features.
    Hardcodes all parameters based on the project scope (TW prediction).
    """
    cfg = config['PROJECT']
    paths = config['PATHS']
    
    # Parameters
    med = 'med_off'
    lfp_ref = 'monopolar'
    beamformer = cfg['BEAMFORMER']
    coupling_to = ['ipsi', 'contra']
    sig_type = config['FEATURES']['SIG_TYPES']
    all_frequency = config['FEATURES']['ALL_FREQUENCY'] # Pass all frequencies for ring averaging

    # Paths Construction
    feat_dir_base = paths['FEAT_DIR']
    data_dir = os.path.join(feat_dir_base, paths['BEAMFORMER_DIR'])
    
    # 1. Load TW Labels
    tw_calc = MonopolarTWCalculator(config)
    tw_df = tw_calc.get_therapeutic_window()
    
    # 2. Prepare Feature Data Structures
    X = np.empty((0, 0))
    labels = np.empty((1, 0))
    S = np.empty((0, 1))

    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    monop_sub = list(set(tw_df['ID']))
    monop_sub.sort()
    
    # Filter feature files to match subjects with TW data
    files = [f for f in files if f[0:4].upper() in monop_sub] # filter the files to only have the files that are in the monopolar subject list
    files.sort()
    
    # Update TW dataframe to match only available feature files
    tw_df = tw_df[tw_df['ID'].isin([f.split('.')[0].upper() for f in files])].reset_index(drop=True)
    
    # --- Feature Collection Loop ---
    for f_name in files:
        subj = f_name.split('.')[0].upper()
        
        x = loadmat(os.path.join(data_dir, f_name))
        features_raw = x['feat']
        feat_labels_raw = [(x['feat_labels'][i, 0][0], x['feat_labels'][i, 1][0], x['feat_labels'][i, 2][0]) for i in range(len(features_raw))]

        # Replace LFP pow with FOOOFed version
        fooof_dir = os.path.join(feat_dir_base, beamformer, 'avg', 'fooof', med, lfp_ref) 
        has_fooof, features, feat_labels = replace_pow_with_fooof(features_raw, feat_labels_raw, subj, med, lfp_ref, fooof_dir)
             
        # Determine hemispheres available (must have TW data)
        mpr_sides = list(tw_df.loc[tw_df['ID'] == subj]['Hemisphere'].unique()) 
        lfp_sides = list(sorted(set(re.findall('left|right', t[1])[0] for t in feat_labels if 'LFP' in t[1])))
        sides = list(set(mpr_sides) & set(lfp_sides))
        
        # Loop over hemispheres
        for side in sides:
            # get contact channels (full LFP-side-channel name)
            chans_this_side = list(sorted(set([t[1] for t in feat_labels if 'LFP' in t[1] and side in t[1]])))
            
            # get target contacts (short name) from TW dataframe for this subj/side
            target_contacts = list(tw_df.loc[(tw_df['ID'] == subj) & (tw_df['Hemisphere'] == side), 'Contact'])
            
            # create Ring Contact Averages and update features/labels/targets
            features, feat_labels, chans_this_side, target_contacts = _create_ring_averages(
                features, feat_labels, chans_this_side, side, target_contacts, all_frequency
            )
            
            try:               
                # feats_side: (N_selected_features x N_lfp_channels)
                # feat_labels_plot_side: (N_selected_features,)
                side_inds = get_side_inds(feat_labels, chans_this_side, ipsi_contra=True)
                d = apply_side_inds(features, feat_labels, side_inds['lab2ind'], side_inds['lab2lab'], len(chans_this_side))
                feats_side = d['data']
                feat_labels_side = d['labels']

                # General feature selection (LFP-MEG COH and LFP POW only)
                sel_ind_feat = select_features(feat_labels_side[:, 0], ipsi_contra=True,
                                               freq_bands=config['FEATURES']['ALL_FREQUENCY'],
                                               chpair_partner=config['FEATURES']['ALL_ROIS'],
                                               src_hemi=coupling_to, sig_types=sig_type)
                
                # Apply selection
                feats_side = np.delete(feats_side, sel_ind_feat['del'], axis=0)
                feat_labels_side = np.delete(feat_labels_side, sel_ind_feat['del'], axis=0)
                feat_labels_plot_side = get_plot_labels(feat_labels_side)

                # Channel selection: mark channels not in TW list as NaN
                short_chans = [c[re.search(r'left-|right-', c).span()[1]:] for c in chans_this_side]
                
                # Identify channels in feature file but NOT in the target list, mark them for removal
                chans_to_remove = [chans_this_side[i] for i, c in enumerate(short_chans) if c not in target_contacts]
                del_ind_chan = [i for i, ch in enumerate(chans_this_side) if ch in chans_to_remove]

                feats_side[:, del_ind_chan] = np.nan
                feat_labels_side[:, del_ind_chan] = np.nan
                
                # Remove NaN columns (channels not in monopolar review)
                inds_missing = np.where(np.isnan(feats_side[0, :]))[0]
                feats_side = np.delete(feats_side, inds_missing, axis=1)
                feat_labels_side = np.delete(feat_labels_side, inds_missing, axis=1)
                short_chans = np.delete(np.array(short_chans), inds_missing, axis=0)
                
                # Final set of contacts used from the feature file
                final_contacts = list(short_chans)
                
                # Stack features and labels for each channel as one example
                for i in range(feats_side.shape[1]):
                    feat_vector = feats_side[:, i].reshape(-1, 1) # N_features x 1
                    contact = final_contacts[i]
                    
                    sample_label = np.array([f'{subj}-{side}-{contact}'])
                    
                    if X.size == 0:
                        X = feat_vector
                        S = sample_label
                    else:
                        X = np.hstack((X, feat_vector))
                        S = np.hstack((S, sample_label))
                    
                    # Look up TW value
                    tw_value = tw_df.loc[(tw_df['ID'] == subj) & 
                                         (tw_df['Hemisphere'] == side) & 
                                         (tw_df['Contact'] == contact), 'TW'].values
                    
                    if tw_value.size > 0:
                        labels_subj = tw_value.reshape((1,1))
                        labels = np.hstack((labels, labels_subj))
                    else:
                        print(f"Warning: Missing TW for {subj}-{side}-{contact}. Skipping.")
                        
            except Exception as e:
                print(f"Error processing {subj}-{side}: {e}. Skipping.")
                continue

    # Final reshape
    y = np.reshape(labels, (labels.size, 1))
    X = np.transpose(X)
    S = np.reshape(S, (S.size, 1))
    
    # Feature labels are consistent across all examples since they are the same per channel
    feat_labels_plot = feat_labels_plot_side

    print(f'Collected {len(y)} labels from {len(S)} examples.')
    print(f'The size of the feature matrix is {X.shape[0]}x{X.shape[1]}.')
    
    # Store the feature labels separately as they are uniform
    with open(os.path.join(paths['RESULTS_DIR'], 'ALL_FEATURE_LABELS.pkl'), "wb") as f:
        pickle.dump(feat_labels_plot, f)

    return (X, y, S, feat_labels_plot)
