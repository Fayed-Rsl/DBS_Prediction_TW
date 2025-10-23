# DBS Therapeutic Window Prediction
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MEG-LFP](https://img.shields.io/badge/data-MEG--LFP-orange.svg)]()


> Machine learning framework for predicting therapeutic windows in Deep Brain Stimulation for Parkinson's Disease using magnetoencephalography and local field potentials.
> <br> <br> Article: Electrophysiological signatures predict the therapeutic window of deep brain stimulation electrode contacts 

## 🧠 Overview

This repository implements a machine learning pipeline to predict the **therapeutic window** (TW) of electrode contacts in Deep Brain Stimulation (DBS) for Parkinson's Disease patients. By analyzing resting-state neural oscillations from the subthalamic nucleus (STN) and STN-cortex coherence patterns, the model helps identify optimal contacts for chronic stimulation.

### Key Features

- **Multimodal Analysis**: Combines MEG and LFP recordings for comprehensive neural signatures
- **Advanced Feature Engineering**: Extracts power spectra and coherence features across frequency bands
- **Prediction**: XGBoost-based regression with nested cross-validation
- **Automated Contact Ranking**: Aim is to speed up monopolar review procedures

## Architecture
```
DBS_Prediction_TW/
├── main.py                    # Main pipeline orchestrator
├── collector.py               # Data collection and TW calculation
├── collector_utils.py         # Helper for collector utilities
├── preprocessing.m            # MEG-LFP feature extraction (MATLAB)
├── fooof_lfp.m                # Apply FOOOF to LFP
├── predictor.py               # XGBoost model training & prediction
├── analyser.py                # Performance metrics & visualization
├── config.json                # Configuration parameters

Prerequisites
- Python 3.8+
- MATLAB R2019b+ (for preprocessing)
- FieldTrip toolbox
```

## 📄 Citation
If you use this code, please cite:
```bibtex
Electrophysiological signatures predict the therapeutic window of deep brain stimulation electrode contacts.
```

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
