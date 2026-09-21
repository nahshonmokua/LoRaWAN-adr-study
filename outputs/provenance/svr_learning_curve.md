# SVR learning curve

RBF SVR (C 10, gamma 0.1, epsilon 0.1, StandardScaler) fitted on random subsamples of the released, DHT22-cleaned training set (seed 42), scored on the full test split of that data (ANN reference on the same data: RMSE 1.438). libsvm is O(n^3): 250 k rows took 18 min to fit and 11 min to predict. RMSE moves 1.454 -> 1.439 from 50 k to 250 k rows; the final run uses 50 k. Recorded 18 Sep 2026; the script that produced it was not retained.
