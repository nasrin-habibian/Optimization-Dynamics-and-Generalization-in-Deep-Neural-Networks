# Methodology

The benchmark compares four optimizers on a two-moons binary classification task across configurable noise levels and random seeds. A fixed MLP is used. The train/test split is stratified; StandardScaler is fitted only on training features.

Each epoch records train/test cross-entropy, train/test accuracy, gradient L2 norm, parameter update L2 norm, and elapsed time. Run-level results are summarized by mean and standard deviation across seeds.

L-BFGS uses full-batch closure-based steps, while SGD, Momentum, and AdamW use mini-batches. Epochs are therefore not equal computational budgets. Interpret wall-clock time and optimizer protocol alongside epoch curves.

This is a small synthetic empirical benchmark. It does not establish transfer to large-scale models, statistical significance, or theoretical convergence. Hyperparameter tuning and broader datasets are future work.
