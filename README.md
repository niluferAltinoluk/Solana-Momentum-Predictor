# Project Overview
This project is a machine learning solution designed to identify high-potential and profitable tokens on Pumpfun, the meme token platform within the Solana ecosystem, 
at the most critical moment of a token's lifecycle.

# Data
The provided data set consists of 30 days of transaction records for tokens launched on the Pumpfun platform. Each row represents a specific event related to a token, identified by its mint_token_id.

Data Columns Summary
The dataset is rich in features, including:

Identifiers and Timestamps: Row ID / sequence, timestamp (Event time in UTC), mint_token_id, holder, and creator.

Transaction Metrics: trade_mode, token_quantity, token_delta, sol_delta, consumed_gas, and fee.

Token Activity and Volume: buy_count, sell_count, total_count, token_volume, and sol_volume.

Liquidity and Reserves: liquidity_ratio, virtual_sol_reserves, and virtual_token_reserves.

Market and Creator State: market_cap_usd, creator_balance, creator_sold, creator_fee, and creator_fee_pump.

Holder and Distribution Metrics: total_holders, current_holders, top10_percent_total, and holder_ratio.

Pre-calculated Technical Indicators: relative_strength_index, bollinger_relative_position, volume_oscillator, rate_of_change, money_flow_index, and buy_sell_ratio.

Preprocessing and Target Integration
Crucial steps were taken to prepare the data for the CatBoost classification model, ensuring adherence to the competition rules:

Target Label Creation: The separate CSV file containing the identified "target tokens" was merged with the training dataset. A new binary target column (e.g., is_target_token) was created, transforming the task into a binary classification problem.

Timestamp Alignment: Critical synchronization and alignment of the timestamp column across both the training and testing datasets were performed to resolve any inconsistencies. This ensures that the model can be accurately tested against the requirement of making predictions within the first 30 seconds of a token's lifespan based on the event time.

Data Partitioning: The final data structure includes a stabilized feature set ready for Feature Engineering and subsequent training/testing of the predictive model.


# Code structure
1. Data Loading & Concatenation: All training and evaluation chunks are loaded and merged into df_master and df_test. The target_df is loaded and used to create the binary is_target column in df_master.
2. Timestamp Standardization: The mixed timestamp format is converted to consistent UTC datetime objects.
3. Feature Engineering
4. Data Cleaning & Imputation: Missing tokens (those with no activity in 30s) are imputed back into df_test_features_final.
5. Outlier Handling (apply_winsorization_v2): Numerical features are Winsorized (clipped at the $1^{st}$ and $99^{th}$ percentiles of the training data) to prevent outliers from overly influencing the model, ensuring no data leakage to the test set.
6. Model Training & Prediction: CatBoost is trained using Stratified 5-Fold CV. OOF predictions are generated for threshold optimization, and the final test predictions (test_preds) are calculated as the average of the 5 folds.
7. Final Output Generation: The optimal threshold is applied to the final test_preds to create the final binary predictions, which are then exported into the required submission.csv and detailed_report.csv formats.



## Feature Engineering Strategy

The core challenge is predicting token success within the first 30 seconds. Therefore, features must capture instantaneous momentum, holder dynamics, and potential "rug pull" signals.
* Creator Trust Signals
* Outlier Mitigation
* Efficiency & Quality
* Velocity & Momentum

## CatBoost Model Architecture and Optimization

* Hyperparameter and Weight Strategy (Focus on Recall & Precision)
* Cross-Validation: Stratified K-Fold (N=5) was used to ensure each fold maintains the same ratio of rare target tokens, leading to a more robust OOF (Out-Of-Fold) score.
* Iterations: Increased to 3000 with early_stopping_rounds=100 to allow the model sufficient time to converge to the optimal solution.
* Threshold Optimization : nstead of using the default $0.5$ threshold, an optimized threshold was found on the OOF prediction scores:Method: Iterating through thresholds ($0.005$ to $0.99$) and selecting the threshold that maximizes the Jaccard Index (Intersection over Union) while strictly maintaining the Recall score $\ge 0.75$.Result: This ensures the final model output meets the minimum success baseline while prioritizing precision for the final predicted subset.

# Results and Evaluation
* Final Recall : The rate of profitable token detection (surpassed the $\ge 75\%$ baseline).
* Jaccard Index (IoU) : The optimization target after achieving the minimum Recall threshold (balance between Precision and Recall).
* Optimal Threshold : The threshold value that maximizes Jaccard while ensuring Recall $\ge 0.75$.
* Prediction Time : Feature engineering was exclusively performed on data filtered to a token's first 30 seconds of lifespan.


