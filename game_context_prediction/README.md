# NFL Game Outcome Prediction
## Overview
This project utilizes machine learning to predict the probability of a "successful" play outcome based on pre-snap game context. By analyzing the relationship between offensive formations, defensive alignment, and game-state variables, the model provides insights into which scenarios favor the offense versus the defense. 

## Data Sources
[nfl-data-py] https://pypi.org/project/nfl-data-py/
- Play by Play data from the 2024 NFL Season. 

## Methodology
- Gradient Boosted Decision Tree: Selected for the efficiency with large-scale data and ability to handle missing values and mixed data types.
- Categorical Encoding: Processes categorical features to avoid the dimensionality challenge of one-hot encodings.
- Segmented Modeling: Distinct model pipelines for Passing and Rushing play families to capture the unique drivers of success for each play type.
- Feature Engineering: Transformed raw game metrics into normalized features for model awareness and mapped categorical score states to numerical proxies.
- Model Deployment: Implemented 'joblib' for model serialization, enabling instant loading and real-time inference without retraining.
- Inference: Output a "success probability" percentage, providing more granular insights for analysis.

## Results
- The models were evaluated using a 80/20 train-test split. Both models had accuracy over 50%, with the Rush model being higher with over 60% accuracy.
- The models had a higher precision score and lower recall, reflecting that the models are conservative. It only predicts success when the pre-snap indicators are overwhelmingly favorable.
- The lower accuracy in the Passing model reflects the volatility of the NFL passing fame compared to rushing. 
- The model is strong at identifying play failure, indicating a high proficiency in recognizing defensive advantages.

## File
- 'game_context_prediction.py'


