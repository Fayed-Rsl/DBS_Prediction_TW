import numpy as np
import scipy.io as sio
import os
from pathlib import Path
import matplotlib.pyplot as plt
from fooof import FOOOF
from numpy.polynomial.polynomial import polyfit
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


# =============================================================================
# FOOOF WRAPPER CLASS
# =============================================================================
class FOOOFProcessor:
    """
    Wrapper for FOOOF algorithm with support for:
    - Multi-trough fitting (low vs. high frequency ranges)
    - Power line noise interpolation
    - Visualization of fitted components
    """
    
    def __init__(self, peak_width_limits, troughs):
        """
        Parameters
        ----------
        peak_width_limits : tuple of (float, float)
            Limits on possible peak width in Hz as (lower_bound, upper_bound).
        troughs : list of [float, float]
            Frequency ranges for fitting. First N-1 ranges use FOOOF, 
            last range uses linear detrending.
            Example: [[1, 90], [91, 400]] fits 1-90 Hz with FOOOF, 91-400 Hz linearly.
        """
        self.peak_width_limits = peak_width_limits
        self.troughs = troughs
        
        assert len(self.troughs) >= 2, 'At least 2 frequency ranges (troughs) required'
        
    def set_original_data(self, freqs, spectrum):
        """Store original frequency and power spectrum data."""
        self.orig_freqs = freqs
        self.orig_spectrum = spectrum
        
    def fit(self, freqs, spectrum, interest, axs, mode='fixed'):
        """
        Fit power spectrum using FOOOF and linear detrending.
        
        Parameters
        ----------
        freqs : np.ndarray
            Frequency bins (1-400 Hz).
        spectrum : np.ndarray
            Power spectrum values.
        interest : list of [frequency_ranges, labels]
            Frequency bands of interest for visualization.
        axs : matplotlib axes
            Axes for plotting (expects 3 subplots).
        mode : str, default='fixed'
            FOOOF aperiodic mode: 'fixed' or 'knee'.
            
        Returns
        -------
        spectrum_flat : np.ndarray
            Flattened (detrended) power spectrum.
        """
        frequency_of_interest, label_of_interest = interest
        
        self.set_original_data(freqs, spectrum)
        
        spectrum_flat = []
        models = []
        
        # Fit each frequency range
        for trough in self.troughs:
            start = np.where(freqs == trough[0])[0][0]
            stop = np.where(freqs == trough[1])[0][0]
            
            freq_range = freqs[start:stop + 1]
            spec = spectrum[start:stop + 1]
            
            # FOOOF fitting for low-frequency ranges
            if trough != self.troughs[-1]:
                model = FOOOF(
                    aperiodic_mode=mode, 
                    peak_width_limits=self.peak_width_limits,
                    peak_threshold=2.0, 
                    max_n_peaks=4
                )
                model.fit(np.arange(1, len(freq_range) + 1), spec)
                models.append(model)
                
            # Linear detrending for high-frequency range
            else:
                intercept, slope = polyfit(x=freq_range, y=np.log10(spec), deg=1)
                spectrum_fit_linear = intercept + slope * freq_range
                spectrum_flat_linear = np.log10(spec) - spectrum_fit_linear
        
        # Combine FOOOF and linear components
        for mod in models:
            spectrum_flat.append(mod._spectrum_flat)
        spectrum_flat.append(spectrum_flat_linear)
        spectrum_flat = np.concatenate(spectrum_flat)
        
        # Visualization
        self._plot_fitting(freqs, spectrum, models, spectrum_fit_linear, 
                          spectrum_flat, axs, frequency_of_interest, label_of_interest)
        
        return spectrum_flat
    
    def _plot_fitting(self, freqs, spectrum, models, spectrum_fit_linear, 
                      spectrum_flat, axs, frequency_of_interest, label_of_interest):
        """Plot original spectrum, fitted components, and flattened spectrum."""
        
        linewidth = 2
        
        # Plot fitted components
        for n, trough in enumerate(self.troughs):
            start = np.where(freqs == trough[0])[0][0]
            stop = np.where(freqs == trough[1])[0][0]
            
            freq_range = freqs[start:stop + 1]
            spec = spectrum[start:stop + 1]
            
            if trough != self.troughs[-1]:
                # FOOOF fit
                axs[1].plot(freq_range, np.log10(spec), '-', 
                           color='black', label='Original spectrum', linewidth=linewidth)
                axs[1].plot(freq_range, models[n]._ap_fit, '--', 
                           color='C1', label='Aperiodic component', linewidth=linewidth)
            else:
                # Linear fit
                axs[1].plot(freq_range, np.log10(spec), '-', 
                           color='black', linewidth=linewidth)
                axs[1].plot(freq_range, spectrum_fit_linear, '--', 
                           color='C3', label='Linear fit', linewidth=linewidth)
                axs[1].set_ylabel('Log10 (Power)')
        
        # Plot flattened spectrum
        axs[2].plot(freqs, spectrum_flat, linewidth=2, color='black', 
                   label='Flattened spectrum')
        axs[2].set_ylabel('Log10 (Power)')
        
        # Highlight frequency bands of interest
        for n, (f_interest, label) in enumerate(zip(frequency_of_interest, label_of_interest)):
            f_interest = freqs[(freqs >= f_interest[0]) & (freqs <= f_interest[1])]
            color = f'C{n}'
            axs[2].fill_between(f_interest, spectrum_flat.min(), spectrum_flat.max(), 
                               color=color, alpha=0.4, label=label)
    
    def interpolate(self, freqs, spec, label, ax, thresh=(49, 349), distance=35, width=7):
        """
        Interpolate power spectrum to remove power line noise artifacts.
        
        Identifies peaks (likely line noise harmonics) and replaces them with 
        interpolated values based on surrounding median power.
        
        Parameters
        ----------
        freqs : np.ndarray
            Frequency bins.
        spec : np.ndarray
            Power spectrum.
        label : str
            Channel label for logging.
        ax : matplotlib axis or None
            Axis for plotting (if None, no plot generated).
        thresh : tuple of (float, float), default=(49, 349)
            Frequency range to search for peaks (avoids edge artifacts).
        distance : int, default=35
            Minimum distance between detected peaks in samples.
        width : int, default=7
            Number of samples before/after peak to interpolate.
            
        Returns
        -------
        x : np.ndarray
            Interpolated power spectrum.
        """
        
        # Detect peaks in log-space
        peaks, _ = find_peaks(np.log10(spec), prominence=0.05, distance=distance)
        peaks = np.array([p for p in peaks if thresh[0] <= p <= thresh[1]])
        
        if peaks.size == 0:
            return spec
        
        x = spec.copy()
        
        # Interpolate around each peak
        for peak in peaks:
            lower_bound = np.arange(peak - width, peak + width)
            bound = [peak - width, peak + width]
            
            # Use median of surrounding regions as interpolation bounds
            lower_median = np.median(x[peak - width:peak])
            upper_median = np.median(x[peak:peak + width])
            
            interp_func = interp1d(bound, [lower_median, upper_median], kind='linear')
            x[lower_bound] = interp_func(lower_bound)
        
        # Flatten high-frequency tail (395-400 Hz) to avoid edge artifacts
        x[395:] = np.median(x[395:])
        
        # Visualization
        if ax:
            ax.plot(freqs, np.log10(spec), '-', color='black', 
                   label='Original spectrum', linewidth=2, alpha=0.8)
            ax.plot(freqs, np.log10(x), '--', color='deepskyblue', 
                   label='Interpolated spectrum', linewidth=2, alpha=1.0)
            ax.set_ylabel('Log10 (Power)')
            ax.legend(frameon=False, loc='upper right', fontsize=10)
        
        return x


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def process_fooof(spectra_dir, output_path, subjects=None, 
                  save=True, show=False, verbose=True):
    """
    Process LFP power spectra using FOOOF parameterization.
    
    Parameters
    ----------
    spectra_dir : str
        Directory containing input .mat files with LFP power spectra.
    output_path : str
        Directory to save FOOOFed power features and plots.
    subjects : list of str, optional
        Specific subject files to process. If None, processes all .mat files.
    save : bool, default=True
        Save output .mat files and figures.
    show : bool, default=False
        Display matplotlib figures during processing.
    verbose : bool, default=True
        Print processing status.
        
    Returns
    -------
    None
        Saves FOOOFed features to {output_path}/average/{subject}_pow_fooofed.mat
    """
    
    # Configuration
    freqs = np.arange(1, 401)  # 1-400 Hz
    
    frequency_labels = [
        'theta', 'alpha', 'low-beta', 'high-beta',
        'low-gamma', 'high-gamma', 'sHFO', 'fHFO'
    ]
    
    frequency_ranges = [
        [3, 7], [8, 12], [13, 20], [21, 35],      # theta, alpha, low-beta, high-beta
        [36, 60], [61, 90],                        # low-gamma, high-gamma
        [200, 300], [301, 400]                     # sHFO, fHFO
    ]
    
    # FOOOF parameters
    troughs = [[1, 90], [91, 400]]  # Low-freq FOOOF, high-freq linear
    gaussian_limits = (2, 100)       # Peak width limits in Hz
    modes = ['fixed']                # Aperiodic mode
    
    # Noise interpolation parameters
    width = 7           # Interpolation window (samples)
    thresh = (49, 349)  # Peak detection range (Hz)
    distance = 35       # Minimum peak distance (samples)
    
    # File discovery
    if subjects is None:
        subjects = sorted([f for f in os.listdir(spectra_dir) if f.endswith('.mat')])
    
    # Output directories
    path_pics = Path(output_path) / 'pics'
    path_results = Path(output_path) / 'average'
    path_pics.mkdir(parents=True, exist_ok=True)
    path_results.mkdir(parents=True, exist_ok=True)
    
    # Initialize FOOOF processor
    fm = FOOOFProcessor(peak_width_limits=gaussian_limits, troughs=troughs)
    
    # Process each subject
    for subject_file in subjects:
        if verbose:
            print(f'\nProcessing: {subject_file}')
        
        subject_id = subject_file[:-4]  # Remove .mat extension
        (path_pics / subject_id).mkdir(parents=True, exist_ok=True)
        
        # Load data
        data = sio.loadmat(os.path.join(spectra_dir, subject_file))
        
        # Storage for subject results
        lfp_labels = []
        lfp_powers = []
        lfp_freq_labels = []
        
        # Process each LFP channel
        for spec, label in zip(data['lfp_power'], data['lfp_labels']):
            
            # Extract label
            label = label[0] if isinstance(label, np.ndarray) else label[0][0]
            
            # Skip bad channels (all zeros)
            if np.all(spec == 0):
                if verbose:
                    print(f'  {label}: Bad channel (skipped)')
                continue
            
            # Setup figure
            fig, ax = plt.subplots(3, 1, sharex=True, figsize=(5.5, 4.5))
            
            # Step 1: Interpolate power line noise
            spectrum_interp = fm.interpolate(freqs, spec, label, ax=ax[0], 
                                            thresh=thresh, distance=distance, width=width)
            
            # Step 2: FOOOF fitting
            spectrum_flat = fm.fit(freqs, spectrum_interp, 
                                   interest=[frequency_ranges, frequency_labels], 
                                   axs=ax, mode=modes[0])
            
            # Format plot
            plt.xlabel('Frequency (Hz)')
            xticks = np.arange(0, freqs[-1] + 1, 50)
            xticks[0] = 1
            plt.xticks(xticks)
            plt.tight_layout()
            plt.subplots_adjust(right=0.9)
            
            # Save figure
            if save:
                fig.savefig(path_pics / subject_id / f'{label}_fooof.png', 
                           dpi=300, bbox_inches='tight')
            
            if show:
                plt.show()
            else:
                plt.close()
            
            # Step 3: Extract band-limited power
            band_powers = []
            band_labels = []
            channel_labels = []
            
            for freq_label, freq_range in zip(frequency_labels, frequency_ranges):
                idx = np.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))[0]
                band_powers.append(np.mean(spectrum_flat[idx]))
                band_labels.append(freq_label)
                channel_labels.append(label)
            
            lfp_powers.append(band_powers)
            lfp_freq_labels.append(band_labels)
            lfp_labels.append(channel_labels)
        
        # Structure output data
        lfp_labels = np.array(lfp_labels, dtype=object).flatten()
        lfp_freq_labels = np.array(lfp_freq_labels, dtype=object).flatten()
        lfp_powers = np.array(lfp_powers, dtype=float).flatten()
        
        # Reshape to match collector.py expected format
        power_structure = np.reshape(lfp_powers, (len(lfp_powers), 1))
        label_structure = np.dstack((lfp_labels, lfp_labels, lfp_freq_labels)).squeeze()
        
        output_dict = {
            'powfooofed': power_structure,  # (N_features, 1)
            'powlabels': label_structure     # (N_features, 3): [channel, channel, freq_band]
        }
        
        # Save
        if save:
            output_file = path_results / f'{subject_id}_pow_fooofed.mat'
            sio.savemat(output_file, output_dict)
            if verbose:
                print(f'  Saved: {output_file}')
    
    if verbose:
        print('\n FOOOF processing complete!')


if __name__ == '__main__':
    process_fooof(
        spectra_dir='...',
        output_path='...',
        subjects='...',
        save='...',
        show=True,
        verbose=True,
    )