# NFL Game Outcome Prediction
## Overview
This project predicts NFL game outcomes using the current week team performance 
metrics, depth chart, injury report, home field, by weeks, and power rankings. The goal is to explore how
position specific assumptions, such as play strength, can be translated into structured
features for predictive modeling. 

## Data Sources
[NFL] https://www.nfl.com/stats/team-stats/
- Current week team statistics for offense, defense, and special teams
- Power ranking
- Weekly schedule 
[ESPN] https://www.espn.com/nfl/
- Current week depth chart
- Power ranking

## Methodology
- Engineered offensive and defensive efficiency metric from team data
- Weighted injury impact by position groups
- Adjusted team strength based on expected started availability
- Incorporated power ranking of team overall strength
- Created matchup-level features sets for model input

## Results
- Predictions aligned reasonable with observed outcomes compared to opposing team
- Injury-related features had the greatest impact for QB and WR positons

## File
- 'weekly_matchup_predictions.py'

