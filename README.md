# 📊 CMIE Data Imputation Project

🔍 **Understanding Missing Data in Large-Scale Surveys**  
This project explores the impact of **missing data** in the **CMIE (Centre for Monitoring Indian Economy) Consumer Pyramids dataset**, applying **statistical, machine learning, and deep learning imputation methods** to recover missing values and improve predictive modeling.

## 🚀 Project Overview

The **CMIE Consumer Pyramids dataset** is a structured longitudinal survey covering **~240,000 households across India** over multiple waves. It includes diverse sociodemographic variables such as **income, consumption, and household demographics**.

Our goal is to:
- **Simulate missingness** (MCAR, MAR, MNAR) in **spatial and temporal dimensions**.
- **Apply different imputation techniques** (statistical, ML, deep learning).
- **Evaluate imputation quality** using RMSE (for continuous) & accuracy (for categorical).
- **Analyze economic impact** of missing data on policy decisions.

## 🛠 Methods

| 📌 Approach  | 🔬 Techniques |
|-------------|----------------------------------|
| **Baseline** | Mean/Mode |
| **Statistical** | MICE |
| **ML-Based** | MissForest, k-NN |
| **Deep Learning** | GAIN (GAN-based), MIDAS/VAE (Autoencoders), DSAN (Attention-based) |

## Environment

To set up a dedicated environment for this project, follow these steps:

1. **Create a new conda environment (Python 3.7):**
   ```bash
   conda create -n tf115_env python=3.7
   ```

2. **Activate the environment:**
   ```bash
   conda activate tf115_env
   ```
   
3. **Install the required packages:**
   ```bash
   conda install tensorflow==1.15.0
   conda install tensorflow-probability==0.8.0
   conda install numpy
   conda install pandas
   conda install matplotlib
   conda install scikit-learn
   ```

## 📈 Evaluation Metrics
- **Imputation Performance**: RMSE (continuous variables) & accuracy (categorical variables).
- **Downstream Task Impact**: Effect on predictive models (poverty classification, health trends).
- **Economic Cost Mapping**: Assess impact of imputation errors on policy decisions.

## 🔮 Future Work
Integrate Bayesian approaches for uncertainty quantification.
Extend analysis to other large-scale survey datasets.
Investigate causal impact of imputation errors on policy simulations.

## ⚠️ License & Usage Restrictions
This project is proprietary and cannot be copied, modified, or used without explicit permission from the author.
If you wish to reuse or collaborate on this project, please contact me directly.

## ❌ Contributing Policy
This repository does not accept public contributions.

- If you have feedback or suggestions, please open an issue instead of submitting a pull request.
- Direct collaboration is by invitation only.
- If you are interested in working on related research, please contact me directly.
