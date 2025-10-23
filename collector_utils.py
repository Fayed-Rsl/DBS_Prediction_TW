import numpy as np
import re
import os
from scipy.io import loadmat

def select_features(feat_labels, ipsi_contra,
                    freq_bands=["theta", "alpha", "low-beta", "high-beta", "low-gamma", "high-gamma", "sHFO", "fHFO"],
                    chpair_partner=["LFP", "Angular", "Cerebellum", "Frontal", "Occipital", "Parietal", "Senorimotor",
                                    "SupraMarginal", "Temporal"], src_hemi=["ipsi", "contra"],
                    sig_types=["lfp_pow", "meg_pow", "lfp_meg_coh"]):
    """
    function to select features by freauency band, brain area, hemisphere and signal type
    input
    feat_labels - list of 3xtuples - original format of feature labels
    ipsi_contra - boolean - are you using ipsi_contra scheme?
    freq_bands - list - frequency bands
    chpair_partner - list - list of areas STN is coupled to. include "LFP" when needing STN power
    src_hemi - list - hemispheres to include. must be a sub-list of ["ipsi", "contra"] if ipsi_contra is True
    sig_types - list - sub-list of ["lfp_pow", "meg_pow", "lfp_meg_coh"]
    returns:
    sel_ind - array - indices into feat_labels of features to be kept
    """
    assert set(freq_bands).issubset(set(["theta", "alpha", "low-beta", "high-beta", "low-gamma", "high-gamma", "sHFO",
                                         "fHFO"])), 'freq_bands must be in ["theta","alpha","low-beta","high-beta","low-gamma","high-gamma","sHFO","fHFO"]'
    assert set(chpair_partner).issubset(
        set(["LFP", "Angular", "Cerebellum", "Frontal", "Occipital", "Parietal", "Sensorimotor", "SupraMarginal",
             "Temporal"])), 'chpair_partner must be in ["LFP","Angular","Cerebellum","Frontal","Occipital","Parietal","Sensorimotor","SupraMarginal","Temporal"]'
    if ipsi_contra:
        assert set(src_hemi).issubset(
            set(["ipsi", "contra"])), 'src_hemi must be "ipsi" or"contra" if ispi_contra is True'
    else:
        assert set(src_hemi).issubset(
            set(["left", "right"])), 'src_hemi must be "left" or "right" if ispi_contra is False'
    assert set(sig_types).issubset(
        set(["lfp_pow", "meg_pow", "lfp_meg_coh"])), 'sig_type must be in ["lfp_pow","meg_pow","lfp_meg_coh"]'

    # due to a typo in feature computation script
    chpair_partner_tmp = np.array(chpair_partner)
    has_senmot = np.where(chpair_partner_tmp == 'Sensorimotor')
    if has_senmot:
        chpair_partner_tmp[has_senmot[0]] = "Senorimotor"
        chpair_partner = chpair_partner_tmp

    ind_sel_del = []
    for t in feat_labels:
        if isinstance(t, np.chararray):
            t = t[0]

        idx = np.where(feat_labels == t)[0][0]
        label_elements = re.split(' ', t)

        if label_elements[1] == 'pow':
            if "LFP" in label_elements[0]:
                this_sig_type = "lfp_pow"
                this_area = "LFP"
            else:
                this_sig_type = "meg_pow"
                this_area = label_elements[0]
            this_freq = label_elements[2]
            coh_tag = False

        else:
            this_sig_type = "lfp_meg_coh"
            this_area = label_elements[2]
            this_freq = label_elements[4]
            this_hemi = label_elements[3]
            coh_tag = True

        keep = False
        # is the freq_band in the selection
        freq_in_sel = np.any([label in this_freq for label in freq_bands])
        # is the signal type in the selection
        sigtype_in_sel = np.any([label in this_sig_type for label in sig_types])
        # does this ROI label equal or contain any labels of the selection
        area_in_sel = np.any([label in this_area for label in chpair_partner])

        if freq_in_sel and sigtype_in_sel and area_in_sel:
            if not coh_tag:
                keep = True
            else:
                hemi_in_sel = np.any([label in this_hemi for label in src_hemi])
                if hemi_in_sel:
                    keep = True
        if not keep:
            ind_sel_del.append(idx)

    sel_ind = {'del': ind_sel_del}
    return sel_ind


def select_channels(lfpchans, lfpchan_sel=['0', '1', '2', '3', '4', '5', '6', '7', '8'], lfpsides=["left", "right"]):
    """
    function for taggin unwanted LFP channels for removal
    input:
    lfpchans - list - channels to pick from
    lfpchan_sel - list - channels you want to keep. note that all channels having any of the items in this list
    in their name will be kept, e.g. when lfpchan_sel = ['0','1'] any chan with a '0' or a '1' in its name will
    be kept
    lfpside - list - hemispheres you want to keep
    returns:
    del_ind - int array - indices into lfpchans of channels which shall be removed
    """
    global_lfp_set = set(
    ["0", "1", "2", "3", "4", "5", "6", "7", "8",
     '234', '567', # boston ring
     '2ABC', '2A', '2B', '2C', '2AB', '2AC', '2BC',
     '3A', '3ABC', '3B', '3C', '3AB', '3AC', '3BC']) 
    
    assert set(lfpchan_sel).issubset(global_lfp_set), f"lfpchans must be a subset of {global_lfp_set}"
    
    assert set(lfpsides).issubset(set(["left", "right"])), 'lfpsides must be in ["left","right"]'
    del_ind = []
    for ch in lfpchans:

        # does this channel label equal or contain any channels of the selection
        lfpchan_in_sel = np.any([label in ch for label in lfpchan_sel])

        # does this LFP side equal or contain any side of the selection
        this_lfpside = re.findall('-.*-', ch)[0][1:-1]
        lfpside_in_sel = np.any([label in this_lfpside for label in lfpsides])

        if not (lfpside_in_sel and lfpchan_in_sel):
            del_ind.append(lfpchans.index(ch))
    return del_ind


def get_side_inds(feat_labels, channels, ipsi_contra):
    """
    computes indices for re-organizing features into a matrix, with one column per LFP channel
    (size: N_feat/N_chan x N_chan). in addition, the function changes the labels from
    3xtuple to string. e.g. ('LFP-right-12', 'LFP-right-12', 'theta') -> LFP-right-12 pow theta
    ('ParietalInfL', 'LFP-right-12', 'alpha') -> LFP-right-12 coh ParietalInf contra alpha
    contra/ipsi are used in the label when ipsi_contra is True, left/right otherwise. labels
    get organized in the same way as features, such that labels[x,y] corresponds to features[x,y]
    input:
    feat_labels - list of 3xtuples - original feature labels
    channels - list - lfp channels of the current hemisphere
    ipsi_contra - boolean - are you using ipsi_contra scheme?
    returns:
    indices - dict - for left and right hemisphere, contains 1) lab2ind (label to index) which maps
    old label to index in matrix and 2) lab2lab (label to label) which maps old feature labels to new feature
    labels
    """
    label_to_ind = {}
    label_to_label = {}
    side = list(sorted(set(re.findall('left|right', t)[0] for t in channels)))

    if ipsi_contra:

        # get new labels and indices into new format
        for c in range(len(channels)):
            chan = channels[c]

            cnt_pow = 0
            cnt_ipsi_megpow = 0
            cnt_ipsi = 0
            cnt_contra = 0
            label_to_ind_pow = {}
            label_to_ind_ipsi = {}
            label_to_ind_contra = {}

            for k in range(len(feat_labels)):
                if feat_labels[k][1] == chan:
                    if feat_labels[k][0] == feat_labels[k][1]:
                        new_label = chan + ' pow ' + feat_labels[k][2]
                        label_to_ind_pow[feat_labels[k]] = (int(cnt_pow), int(c))
                        cnt_pow = cnt_pow + 1
                    else:
                        l = feat_labels[k][1]
                        is_ipsi = ('left' in l and feat_labels[k][0][-1] == 'L') or (
                                    'right' in l and feat_labels[k][0][-1] == 'R')
                        if is_ipsi:
                            new_label = chan + ' coh ' + feat_labels[k][0][:-1] + ' ipsi ' + feat_labels[k][2]
                            label_to_ind_ipsi[feat_labels[k]] = (int(cnt_ipsi), int(c))
                            cnt_ipsi = cnt_ipsi + 1
                        else:
                            new_label = chan + ' coh ' + feat_labels[k][0][:-1] + ' contra ' + feat_labels[k][2]
                            label_to_ind_contra[feat_labels[k]] = (int(cnt_contra), int(c))
                            cnt_contra = cnt_contra + 1
                    label_to_label[feat_labels[k]] = new_label

            for j in label_to_ind_ipsi.keys():
                label_to_ind_ipsi[j] = (label_to_ind_ipsi[j][0] + cnt_pow, label_to_ind_ipsi[j][1])
            for h in label_to_ind_contra.keys():
                label_to_ind_contra[h] = (label_to_ind_contra[h][0] + cnt_pow + cnt_ipsi, label_to_ind_contra[h][1])

            label_to_ind.update(label_to_ind_pow)
            label_to_ind.update(label_to_ind_ipsi)
            label_to_ind.update(label_to_ind_contra)
    else:
        for c in range(len(channels)):
            chan = channels[c]
            cnt = 0
            for k in range(len(feat_labels)):
                if feat_labels[k][1] == chan:
                    if feat_labels[k][0] == feat_labels[k][1]:
                        new_label = chan + ' pow ' + feat_labels[k][2]
                    else:
                        if feat_labels[k][0][-1] == 'L':
                            new_label = chan + ' coh ' + feat_labels[k][0][:-1] + ' left ' + feat_labels[k][2]
                        else:
                            new_label = chan + ' coh ' + feat_labels[k][0][:-1] + ' right ' + feat_labels[k][2]
                    label_to_ind[feat_labels[k]] = (int(cnt), int(c))
                    label_to_label[feat_labels[k]] = new_label
                    cnt = cnt + 1
    indices = {'lab2ind': label_to_ind, 'lab2lab': label_to_label}
    return indices

def apply_side_inds(features, feat_labels, label_to_ind, label_to_label, N_chans):
    """
    applies indices computed by get_side_inds to re-organize features and labels into matrix
    input:
    features - array - Nx1 feature vector
    feat_labels - list of 3xtuples - original feature labels
    label_to_ind - dict - maps original label to index into matrix
    label_to_label dict - map original label to new label
    returns:
    matrices - dict - contains feature matrix in field data and label matrix in field labels
    """
    # organize the data and the labels using the indices computed above
    data_arr = np.zeros(shape=(int(len(label_to_ind) / N_chans), N_chans))
    data_arr.fill(np.nan)
    label_arr = np.chararray(data_arr.shape, itemsize=150, unicode=True)
    for k in label_to_ind.keys():
        label_arr[label_to_ind[k]] = label_to_label[k]
        data_arr[label_to_ind[k]] = features[feat_labels.index(k)]

    matrices = {'data': data_arr, 'labels': label_arr}
    return matrices

def get_plot_labels(feat_labels):
    """
    function for getting feature labels for plots.py by removing the first column of label matrix and removing the leading LFP-xxx
    input:
    feat_labels - feature labels organized in matrix
    returns:
    feat_labels_new - numpy.chararray - modified feature labels
    """
    feat_labels = feat_labels[:, 0]
    feat_labels_new = feat_labels.copy()
    for k in np.arange(feat_labels.size):
        f = str(feat_labels[k])
        f2 = re.findall('pow.*|coh.*', f)[0]
        if "pow" in f2:
            if "LFP" in f:
                f2 = "LFP " + f2
            else:
                f2 = f
        feat_labels_new[k] = f2
    return feat_labels_new

def replace_pow_with_fooof(features, feat_labels, subj, med, lfp_ref, fooof_dir):
    '''
    replaces LFP power features by their FOOOFed version; only channels which have a FOOOFed version remain
    input:
    features - array - Nx1 feature vector
    feat_labels - list of 3xtuples - original feature labels
    subj - str - pseudonym of subject
    med - str - "off" or "on"
    lfp_ref - str - type of LFP referencing; 'average' or 'bipolar'
    fooof_dir - str - path to FOOOFed features
    returns:
    tuple (has_fooof, modified features, corresponding feature labels)
    if a FOOOF file has been found for the given subject, the modified features are LFP power and LFP-MEG coh
    of the LFP channels which have a FOOOFed version of their LFP power. LFP power features have been replaced
    by their FOOOFed version. if a FOOOF file has not been found for the given subject, has_fooof is false and
    features and feature labels are returned unchanged
    '''

    file = fooof_dir + '/' + subj.lower() + '_pow_fooofed.mat'
    new_features = features.copy()

    if not os.path.exists(file):
        raise FileExistsError ('FOOOF file does not exist for ' + subj)
    else:
        x = loadmat(file) # load the FOOOFed features
        fooof_features = x['powfooofed'] # power features
        N_feats = fooof_features.size # number of features
        fooof_labels = [
            (x['powlabels'][i, 0][0],
            x['powlabels'][i, 1][0], 
            x['powlabels'][i, 2][0])
            for i in range(N_feats)]
        
        lfpchans_orig = set([f[1] for f in feat_labels]) # get the LFP channels of the original features
        lfpchans_with_fooof = set([f[1] for f in fooof_labels]) # get the LFP channels of the FOOOFed features
        
        if not lfpchans_with_fooof.issubset(lfpchans_orig):
            raise ValueError ('orginal LFP chans and FOOOF LFP chans do not match for ' + subj)

        #for each original feature, see if it needs to be replaced by a fooofed feature and which
        inds_replace = []
        inds_replaced = []
        for k in np.arange(new_features.size):
            idx_replace = [i for i in np.arange(N_feats) if
                           feat_labels[k][0] == fooof_labels[i][0] and feat_labels[k][1] == fooof_labels[i][1] and
                           feat_labels[k][2] == fooof_labels[i][2]]
            #if so, replace
            if np.size(idx_replace) > 0:
                new_features[k] = fooof_features[idx_replace[0]]
                inds_replace.append(idx_replace[0])
                inds_replaced.append(k)
                
        # keep only features present in FOOOFed data, i.e. only the features of selected channels
        keep = [feat_labels.index(f) for f in feat_labels if f[1] in lfpchans_with_fooof]
        feat_labels = [f for f in feat_labels if f[1] in lfpchans_with_fooof]
        new_features = new_features[keep]
        return (True, new_features, feat_labels)
