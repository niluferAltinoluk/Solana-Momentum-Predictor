#  Solana Pumpfun Alpha Hunter: 30s Predictive Model

This project is an advanced machine learning solution designed to identify high-potential tokens on the **Solana Pumpfun** platform. The core challenge is making a "buy" decision within the **first 30 seconds** of a token's lifecycle, where volatility is highest and data is scarcest.

---

##  Data Architecture & Pipeline

The model processes a massive dataset (9.5M+ rows), transforming raw sub-minute events into actionable intelligence.

### 1. Time-Critical Transformation
Standardized non-standard timestamps into UTC datetime objects to enable precise filtering for the critical **30-second prediction window**.



### 2. Feature Engineering Strategy
The features are designed to capture instantaneous momentum and "rug-pull" risk signals:
*   **Creator Trust Signals:** Analyzing creator balance and historical sell behavior.
*   **Velocity & Momentum:** Measuring the rate of buy/sell counts within the first 30s.
*   **Holder Dynamics:** Monitoring the `top10_percent_total` and `holder_ratio` to detect centralization.
*   **Technical Indicators:** Custom RSI, Bollinger Relative Position, and Money Flow Index (MFI) calculated on micro-intervals.

### 3. Preprocessing & Stability
*   **Winsorization:** Numerical features are clipped at the $1^{st}$ and $99^{th}$ percentiles to ensure stability against extreme price swings.
*   **Imputation:** Handling tokens with zero activity in the initial window to prevent data bias.

---

##  Model Architecture: CatBoost

A **CatBoost Classifier** was selected for its superior handling of categorical features (like `mint_token_id` and `creator`) and its robustness against noisy financial data.



### Optimization Strategy
*   **Validation:** Stratified 5-Fold Cross-Validation to maintain the class ratio of rare "needle in a haystack" profitable tokens.
*   **Recall-First Approach:** The model is strictly tuned to maintain **Recall ≥ 0.75**, capturing at least 75% of all successful tokens.
*   **Jaccard Index (IoU) Maximization:** Instead of a default $0.5$ threshold, we iterate from $0.005$ to $0.99$ to find the optimal point that maximizes IoU while satisfying the recall constraint.

---

## Tech Stack

*   **Languages:** Python (Pandas, NumPy, SciPy)
*   **ML Frameworks:** CatBoost, Scikit-Learn
*   **Environment:** Jupyter / Kaggle / CUDA Support

---

# Project Structure
├── data/                   # Transaction logs (9.5M rows)
├── notebooks/              # Feature Engineering & CatBoost Training
├── models/                 # Saved CatBoost weights (5-Fold)
├── outputs/                # submission.csv & detailed_report.csv
└── README.md               # Documentation



---

Disclaimer: This tool is for educational and research purposes in blockchain data science. High-frequency trading on Solana involves significant financial risk.

---

## Results & Evaluation

The final output is optimized for near real-time execution, delivering:
1.  **High Recall:** Minimizing missed opportunities in the high-frequency Pumpfun environment.
2.  **Precision via IoU:** Balancing sensitivity with decision quality through threshold optimization.

```python
# Strategic Optimization Snippet
# Iterating thresholds to maximize Jaccard Index (IoU) while Recall >= 0.75
best_threshold = 0.5
best_iou = 0

for threshold in np.arange(0.005, 0.99, 0.005):
    preds = (oof_probs >= threshold).astype(int)
    current_recall = recall_score(y_train, preds)
    if current_recall >= 0.75:
        current_iou = jaccard_score(y_train, preds)
        if current_iou > best_iou:
            best_iou = current_iou
            best_threshold = threshold
