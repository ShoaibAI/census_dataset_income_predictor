# 📊 Census Income Prediction Pipeline

An end-to-end, production-optimized Machine Learning pipeline designed to classify individual income levels (>50K vs <=50K) using structural census data. This system features a robust `scikit-learn` preprocessing architecture coupled with a real-time predictive dashboard built using `Gradio`.

## ⚡ Core Performance Benchmarks
Rather than optimizing blindly for raw accuracy (which conceals data skew issues), this model was tuned specifically to maintain strict operational control over classification errors:

| Metric | Performance Value | Context |
| :--- | :--- | :--- |
| **True Positive Rate (TPR / Recall)** | **71%** | Successfully captures high-earning demographics |
| **False Positive Rate (FPR)** | **10%** | Exceptionally low false-alarm rate |
| **Balanced F1-Score** | **> 70%** | Robust harmonic mean across both target classes |

---

## 🛠️ The Optimization Journey: How I Got Here

The baseline model suffered from a common data science bottleneck: high overall accuracy but terrible minority-class recall. The model continuously guessed the lower income bracket because it dominated the dataset. 

I broke past this using a three-stage engineering framework:

### 1. Robust Type-Forced Preprocessing Pipeline
I constructed a clean `ColumnTransformer` network to isolate and handle features deterministically. String-based categorical arrays are structurally locked using explicit string type-coercion, preventing standard runtime errors when running modern Python (3.12+) and NumPy environments. 
* **Categorical Handling:** Handled via custom structured string encoding rules (`OneHotEncoder(handle_unknown='ignore')`).
* **Feature Engineering:** Implemented native structural trackers like `works-overtime` derived instantly from operational work-hour parameters.

### 2. Algorithmic Grid Sweeps (`GridSearchCV`)
Instead of guessing hyperparameters, I executed exhaustive cross-validated grid sweeps over the model's ensemble methods. I optimized tree depth boundaries and `min_samples_split` thresholds to enforce generic decision paths across sparse census traits (such as distinct native countries or occupations).

### 3. Precision Threshold Calibration (The Game Changer)
Standard models split predictions blindly at a `0.50` probability limit. By plotting the ROC curve and analyzing the operational trade-off space, I discovered that dropping the decision threshold to **0.40** completely unlocked the model's utility. This specific modification shot the True Positive Rate up to **71%** while ensuring the False Positive Rate never crossed the **10%** ceiling.

---

## 🖥️ Production Dashboard Features
The application contains a standalone web interface built on `Gradio` featuring:
* **Position-vs-Name Fallback Engine:** The inference backend contains an automated fallback block. If the underlying model was fit on structural array index positions rather than strict pandas namespaces, it seamlessly unpacks dataframes down to raw target matrices without crashing.
* **NumPy Monkey-Patch Safety Net:** Contains a runtime interceptor for `np.isnan` tracking over object string vectors, bypassing environment version disparities between legacy training sets and modern infrastructure.

---

## 🚀 Quick Start & Deployment

### Prerequisites
```bash
pip install pandas numpy scikit-learn gradio
