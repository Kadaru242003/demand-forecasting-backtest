# Backtest results: store CA_1, 3049 items, 6 folds x 8 weeks

## Summary (mean over evaluation folds)

| model | MAE | RMSE | MASE | MAE vs seasonal naive | MASE min | MASE max | folds beating baseline (MASE) |
|---|---|---|---|---|---|---|---|
| LightGBM | 4.129 | 9.039 | 0.963 | 0.633 | 0.914 | 1.027 | 6/6 |
| ETS | 4.330 | 10.042 | 0.996 | 0.664 | 0.949 | 1.038 | 6/6 |
| Moving average (8w) | 4.562 | 10.958 | 1.027 | 0.700 | 0.978 | 1.067 | 6/6 |
| Seasonal naive | 6.531 | 14.606 | 1.437 | 1.000 | 1.373 | 1.506 | 0/6 |

## MASE per fold

| fold | ETS | LightGBM | Moving average (8w) | Seasonal naive |
|---|---|---|---|---|
| 1 | 1.038 | 1.027 | 1.066 | 1.506 |
| 2 | 1.013 | 0.980 | 1.033 | 1.495 |
| 3 | 1.019 | 0.968 | 1.067 | 1.393 |
| 4 | 0.949 | 0.914 | 0.978 | 1.373 |
| 5 | 0.969 | 0.946 | 1.004 | 1.406 |
| 6 | 0.985 | 0.941 | 1.013 | 1.448 |

## Full per-fold metrics

| model | fold | MAE | RMSE | MASE | MAE vs seasonal naive |
|---|---|---|---|---|---|
| ETS | 1 | 4.503 | 10.895 | 1.038 | 0.654 |
| ETS | 2 | 4.433 | 10.166 | 1.013 | 0.639 |
| ETS | 3 | 4.614 | 11.632 | 1.019 | 0.750 |
| ETS | 4 | 4.117 | 9.679 | 0.949 | 0.658 |
| ETS | 5 | 4.102 | 8.778 | 0.969 | 0.634 |
| ETS | 6 | 4.210 | 9.102 | 0.985 | 0.650 |
| LightGBM | 1 | 4.419 | 9.529 | 1.027 | 0.641 |
| LightGBM | 2 | 4.194 | 9.034 | 0.980 | 0.604 |
| LightGBM | 3 | 4.208 | 9.634 | 0.968 | 0.684 |
| LightGBM | 4 | 3.880 | 8.228 | 0.914 | 0.620 |
| LightGBM | 5 | 4.108 | 9.443 | 0.946 | 0.635 |
| LightGBM | 6 | 3.963 | 8.365 | 0.941 | 0.612 |
| Moving average (8w) | 1 | 4.873 | 12.283 | 1.066 | 0.707 |
| Moving average (8w) | 2 | 4.544 | 10.539 | 1.033 | 0.655 |
| Moving average (8w) | 3 | 4.874 | 12.765 | 1.067 | 0.792 |
| Moving average (8w) | 4 | 4.337 | 10.775 | 0.978 | 0.694 |
| Moving average (8w) | 5 | 4.369 | 10.051 | 1.004 | 0.675 |
| Moving average (8w) | 6 | 4.375 | 9.335 | 1.013 | 0.675 |
| Seasonal naive | 1 | 6.889 | 16.069 | 1.506 | 1.000 |
| Seasonal naive | 2 | 6.942 | 15.757 | 1.495 | 1.000 |
| Seasonal naive | 3 | 6.153 | 12.954 | 1.393 | 1.000 |
| Seasonal naive | 4 | 6.254 | 13.703 | 1.373 | 1.000 |
| Seasonal naive | 5 | 6.473 | 14.986 | 1.406 | 1.000 |
| Seasonal naive | 6 | 6.477 | 14.169 | 1.448 | 1.000 |

## Mean scaled error by week type

| week_type | ETS | LightGBM | Moving average (8w) | Seasonal naive | n_rows |
|---|---|---|---|---|---|
| regular | 0.978 | 0.950 | 1.008 | 1.397 | 111699 |
| promo week | 1.428 | 1.314 | 1.460 | 2.174 | 7014 |
| holiday week | 0.958 | 0.927 | 0.992 | 1.410 | 27399 |

## Mean scaled error by volume segment

| volume | ETS | LightGBM | Moving average (8w) | Seasonal naive | n_rows |
|---|---|---|---|---|---|
| low (<2/wk) | 1.028 | 1.040 | 1.036 | 1.337 | 27512 |
| medium (2-10/wk) | 0.989 | 0.947 | 1.017 | 1.418 | 79720 |
| high (>10/wk) | 0.986 | 0.940 | 1.041 | 1.547 | 38880 |

## Mean scaled error by holiday

| holiday | ETS | LightGBM | Moving average (8w) | Seasonal naive | n_rows |
|---|---|---|---|---|---|
| Christmas | 0.960 | 0.881 | 0.995 | 1.311 | 3045 |
| Easter | 0.826 | 0.803 | 0.883 | 1.456 | 3049 |
| Halloween | 1.014 | 0.985 | 1.051 | 1.441 | 3043 |
| IndependenceDay | 0.969 | 0.982 | 0.991 | 1.525 | 3039 |
| LaborDay | 1.017 | 1.003 | 1.038 | 1.532 | 3041 |
| NewYear | 0.910 | 0.887 | 0.956 | 1.335 | 3045 |
| SuperBowl | 0.907 | 0.897 | 0.941 | 1.417 | 3047 |
| Thanksgiving | 1.111 | 1.003 | 1.142 | 1.333 | 3043 |
| ValentinesDay | 0.910 | 0.904 | 0.932 | 1.340 | 3047 |

## Relative bias (sum forecast / sum actual - 1)

| level_0 | level_1 | ETS | LightGBM | Moving average (8w) | Seasonal naive |
|---|---|---|---|---|---|
| week_type | holiday week | 0.027 | -0.003 | 0.019 | 0.041 |
| week_type | promo week | -0.118 | -0.036 | -0.157 | -0.407 |
| week_type | regular | 0.009 | -0.021 | 0.003 | 0.025 |
| volume | low (<2/wk) | -0.034 | 0.042 | -0.055 | -0.112 |
| volume | medium (2-10/wk) | -0.010 | -0.011 | -0.020 | -0.079 |
| volume | high (>10/wk) | 0.014 | -0.024 | 0.006 | 0.043 |

## Store-level weekly total (sum of item forecasts): MAPE / bias per fold

| fold | ETS | LightGBM | Moving average (8w) | Seasonal naive |
|---|---|---|---|---|
| 1 | 0.040 | 0.029 | 0.042 | 0.018 |
| 2 | 0.046 | 0.037 | 0.042 | 0.035 |
| 3 | 0.121 | 0.046 | 0.084 | 0.060 |
| 4 | 0.030 | 0.055 | 0.041 | 0.027 |
| 5 | 0.065 | 0.061 | 0.081 | 0.056 |
| 6 | 0.038 | 0.021 | 0.046 | 0.039 |


| fold | ETS | LightGBM | Moving average (8w) | Seasonal naive |
|---|---|---|---|---|
| 1 | -0.016 | -0.009 | -0.024 | 0.008 |
| 2 | 0.031 | -0.022 | 0.024 | 0.016 |
| 3 | 0.118 | 0.039 | 0.080 | 0.050 |
| 4 | -0.001 | -0.055 | 0.038 | 0.019 |
| 5 | -0.066 | -0.061 | -0.082 | -0.039 |
| 6 | -0.026 | -0.006 | -0.046 | -0.023 |

## Interval coverage (LightGBM)

| level | by | group | coverage | mean_width | n |
|---|---|---|---|---|---|
| 0.800 | overall | all | 0.812 | 12.811 | 146120 |
| 0.800 | fold | 1 | 0.804 | 13.217 | 24312 |
| 0.800 | fold | 2 | 0.809 | 13.080 | 24328 |
| 0.800 | fold | 3 | 0.808 | 12.919 | 24344 |
| 0.800 | fold | 4 | 0.827 | 12.437 | 24360 |
| 0.800 | fold | 5 | 0.813 | 12.587 | 24384 |
| 0.800 | fold | 6 | 0.814 | 12.627 | 24392 |
| 0.950 | overall | all | 0.954 | 21.645 | 146120 |
| 0.950 | fold | 1 | 0.946 | 22.210 | 24312 |
| 0.950 | fold | 2 | 0.953 | 22.221 | 24328 |
| 0.950 | fold | 3 | 0.952 | 21.900 | 24344 |
| 0.950 | fold | 4 | 0.962 | 20.955 | 24360 |
| 0.950 | fold | 5 | 0.955 | 21.211 | 24384 |
| 0.950 | fold | 6 | 0.958 | 21.379 | 24392 |
