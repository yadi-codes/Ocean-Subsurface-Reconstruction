# Ocean Reconstruction using Deep Learning

Reproduction of the paper **"Ocean Subsurface Temperature and Salinity Reconstruction Using ConvFormer"** using publicly available satellite observations and BOA-Argo data.

The objective is to reconstruct **3D ocean temperature and salinity profiles** from surface observations using a deep learning model based on **ConvLSTM** and **Transformer (ConvFormer)**.

---

# Features

- Automated dataset downloading
- Data validation and integrity checks
- Monthly averaging of daily datasets
- Spatial alignment to a common 1° × 1° grid
- Min-Max normalization
- Creation of training tensors
- ConvFormer implementation
- Model training and evaluation

---

# Datasets

| Dataset | Source | Purpose |
|---------|--------|---------|
| NOAA OISST v2.1 | https://www.ncei.noaa.gov/products/optimum-interpolation-sst | Sea Surface Temperature (SST) |
| Copernicus DUACS SSH | https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047/description | Sea Surface Height (SSH) |
| NASA CCMP V3.1 | https://www.remss.com/measurements/ccmp/ | Surface Wind Speed (U & V) |
| ESA CCI Sea Surface Salinity | https://climate.esa.int/en/projects/sea-surface-salinity/data/ | Sea Surface Salinity (SSS) |
| BOA-Argo | http://www.argo.org.cn | Subsurface Temperature & Salinity (Ground Truth) |

---

# Project Pipeline

```
Raw Data
    │
    ▼
Download
    │
    ▼
Validation
    │
    ▼
Monthly Averaging
    │
    ▼
Preprocessing
    │
    ▼
Spatial Alignment (1° × 1°)
    │
    ▼
Normalization
    │
    ▼
Dataset Creation
    │
    ▼
ConvFormer Training
    │
    ▼
Evaluation
```

---

# Current Progress

- [x] Dataset collection
- [x] Download automation
- [x] Dataset validation
- [x] Monthly averaging
- [x] Data preprocessing
- [x] Spatial alignment
- [ ] Normalization
- [ ] Training dataset creation
- [ ] ConvFormer implementation
- [ ] Model training
- [ ] Evaluation

---

# Folder Structure

```
Ocean-Reconstruction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│
├── results/
│
├── scripts/
│   ├── download/
│   ├── preprocess/
│   ├── check/
│   └── utils/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Tech Stack

- Python
- PyTorch
- Xarray
- NumPy
- NetCDF4
- Pandas
- SciPy

---

# Paper

This repository reproduces the methodology proposed in:

> **Ocean Subsurface Temperature and Salinity Reconstruction Using ConvFormer**

The implementation follows the preprocessing pipeline and model architecture described in the paper.

---

# Future Work

- Implement ConvLSTM encoder
- Implement ConvFormer encoder
- Train using 2010–2020 data
- Compare reconstructed fields with BOA-Argo observations
- Evaluate RMSE, MAE, and correlation against the published results

---

# License

This project is intended for research and educational purposes only.
Dataset licenses remain with their respective providers.