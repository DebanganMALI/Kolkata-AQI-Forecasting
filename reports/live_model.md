# Live GRU (for the website)

Inputs: past 14 days of city AQI + 13 weather variables + day of year (16 per day). Chosen target mode on validation: **change** (level 15.87, change 15.77).

| test period | days | live GRU MAE | persistence MAE | difference [95%] | verdict | RMSE | R2 | 80% range coverage | alert F1 (GRU / persistence) |
|---|---|---|---|---|---|---|---|---|---|
| 2025-01-01 to 2026-10-31 | 669 | 12.9 | 13.8 | -0.9 [-1.6, -0.2] | better than persistence | 18.8 | 0.893 | 86% | 0.638 / 0.585 |
