# CropLEM: Foundation Model for Cropland Detection

**Self-Supervised Multi-Sensor Foundation Model for Global Agricultural Monitoring**


## Table of Contents

- Overview

- Motivation

- Key Features

- Project Structure

- Data

- System Architecture

- Installation

- Quick Start

- Results and Evaluation

- Research Context

- Future Plan



## Overview

CropLEM is a foundation model for global cropland detection that addresses critical challenges in agricultural remote sensing,

- **Geographic generalization:** Learn representations that transfer across continents and climate zones.

- **Multi-sensor robustness:** Work with Sentinel-1 SAR and Sentinel-2 optical for monitoring in all weather (through SEN12MS dataset).

- **Label efficiency:** Leverage self-supervised pretraining on vast unlabeled satellite imagery, and apply contextual bandits for active learning of sample selection for efficient labeling.

- **Scalability:** Foundation model approach enables rapid adaptation to new regions with minimal or no labeled data.

CropLEM learns generalizable feature representations from unlabeled multi-sensor satellite data using Masked Autoencoders (MAE), then adapts to downstream cropland detection tasks with minimal supervision. This approach mirrors recent successes in NLP with Large Language Models (LLMs), but specifically designed for multi-modal Earth observation data, with the vision of creating a component for Large Earth Models (LEMs).


**Why would we use foundation models for agriculture and Earth observation?**

Current crop mapping approaches face three critical bottlenecks,

- **Geographic bias:** Models trained on US or European data fail in regions like Sub-Saharan Africa, Bangladesh, and data-scarce regions.

- **Labeling costs:** Expert annotation of satellite imagery is expensive and slow, so typical supervised approaches fail despite having petabytes of unlabeled data.

- **Sensor limitations:** Cloud cover renders optical-only systems unusable for weeks, so a multi-sensor approach is required.

Foundation models solve these issues by,

- Learning universal spatial, spectral, and temporal patterns that generalize globally.

- Requiring only a fraction of the labels (10 to 15%) compared to supervised approaches.

- Integrating complementary modalities or features such as temporal data from multiple seasons, combining SAR with optical sensor data, etc.









## Motivation

The Agricultural Monitoring Challenge

Global food security is a critical challenge of our time. According to the World Health Organization, about two billion people experience moderate or severe food insecurity worldwide, with Africa particularly hard hit. Agricultural monitoring systems must,

- **Operate globally:** Substantial work should be done in regions like sub-Saharan Africa and Bangladesh, where food security has suffered greatly due to disasters like cyclones, droughts, and extreme flooding.

- **Scale to smallholder farms:** Traditional analysis can be insufficient for typical smallholder farm sizes and compositions.

- **Transfer across regions:** Models trained in data-rich regions (US, Europe) often fail when deployed in data-scarce regions (South Asia, sub-Saharan Africa).

**Current Limitations**

Supervised learning approaches require,

- Thousands of labeled training samples per region, which is often times impractical

- Expert annotators with local agricultural knowledge

- Months of data collection and validation

- Separate models for each geographic area

**Why This Doesn’t Scale?**

- Africa alone has 54 countries with diverse agroecological zones

- Labeling costs make it unfeasible to use labeled data

- Climate change rapidly shifts growing patterns, outdating trained models

- Cloud cover in tropical regions limits optical-only approaches

**The Foundation Model Solution**

Foundation models that are pretrained on massive unlabeled data offer a paradigm shift, as self-supervised pretraining on petabytes of unlabeled data can be utilized. The generated embeddings of these models can be used with downstream classifiers later, which ensure that these models transfer globally with minimal labels required.











## Key Features

**Multi-Sensor Foundation Model**

This approach combines Sentinel-1 SAR and Sentinel-2 optical satellite imagery,

- Sentinel-1 SAR: All weather, day and night monitoring. Uses VV, VH polarization.

- Sentinel-2 optical: Rich spectral information (13 bands).

- Cross-modal feature learning: Model learns complementary patterns.

- Robust to missing modalities: Works even if SAR or optical is unavailable.

**Self-Supervised Pretraining using Masked Autoencoders (MAE)**

- Masked autoencoder (MAE) architecture adapted for multi-spectral imagery

- 75% masking ratio forces learning of spatial and spectral context

- Pretraining on around 10,000 patches (approximately 30% of SEN12MS summer data due to computational constraints)

- Transferable embeddings for downstream classification or detection tasks

**Geographic Generalization**

- Pretrained on globally diverse data (SEN12MS covers all continents)

- Leave-one-out evaluation protocol

- Fine-tuning with few labeled (400 to 500) samples per new region (e.g. – I use Kenya Crop Type Classification dataset for downstream classification in this project)

- Addresses geographic bias through pretraining diversity

**Active Learning Through Contextual Bandits**

- Reinforcement learning agent, or specifically, contextual bandit selects most informative samples

- Has multi-objective reward system which includes – Uncertainty, Diversity, and Under-representation of regions

- Reduces labeling requirements significantly, and potentially much more effective than random sampling (sampling based on rewards)

**Comprehensive Evaluation**

Beyond accuracy metrics, the following evaluation criteria will be used,

- Cross-region transfer learning

- Label efficiency curves

- Geographic fairness metrics (performance variance across regions)

- Calibration analysis (confidence reliability)



## Project Structure

Under construction….





## Data

**SEN12MS**

SEN12MS is the ideal dataset for developing multi-sensor cropland foundation models because of its features,

- Global coverage: 180,662 patches distributed across all inhabited continents

- Multi-sensor: Combines Sentinel-1 SAR, Sentinel-2 optical, and MODIS Land Cover data

- Multi-temporal: Four seasons - spring, summer, fall, winter (though I use only part of the summer data due to computational constraints)

- Resolution: 256x256 pixels at 10m ground sampling distance covering 2.56km x 2.56km areas

- Rich labels: Land cover annotations following IGBP classification scheme with 17 classes

The dataset high-level statistics are provided below,
|       Modality      |       Channels      |       Value   Range      |       Description            |
|---------------------|---------------------|--------------------------|------------------------------|
|     Sentinel-1      |     2               |                          |     VV,   VH polarization    |
|     Sentinel-2      |     13              |     0   to 10,000        |     B01   to B12, B8A        |
|     Land   Cover    |     1               |     0   to 16            |     IGBP   classification    |


The dataset can be downloaded [here](https://mediatum.ub.tum.de/1474000)

[Paper](https://isprs-annals.copernicus.org/articles/IV-2-W7/153/2019/) on this dataset

More information on the dataset available in this [GitHub repo](https://github.com/schmitt-muc/SEN12MS)













## System Architecture

A high-level (and somewhat unpolished) system architecture diagram is provided below. As I continue to develop the project, I will update the pieces of this diagram as required, and possibly have it more polished and detailed.

<p align="center">
  <img src="CropLEM - System Architecture.png" width="700">
</p>

<p align="center">Fig.1. CropLEM High-Level System Architecture</p>






## Installation

Under construction….



## Quick Start

Under construction….



## Results and Evaluation

Under construction….



