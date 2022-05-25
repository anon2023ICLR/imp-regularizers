# imp-regularizers
Code for NeurIPS 2022 Submission: Disentangling the Mechanisms Behind Implicit Regularization in SGD

All the regularization experiments are run via `train_regloss.py` through the following:

* `Vanilla` - Normal small-batch SGD (batch size controlled by the `--batch-size` argument)
* `PseudoGD` - Accumulated, large-batch SGD (batch size controlled by the `--large-batch-size` argument, and `--batch-size` sets the size of the accumulated micro-batch)
* `RegLoss` - Large-batch SGD + average microbatch gradient norm regularization
* `FishLoss` - Large-batch SGD + average microbatch Fisher trace regularization
* `AvgJacLoss` - Large-batch SGD + average Jacobian regularization
* `UnitJacLoss` - Large-batch SGD + Unit Jacobian regularization

All hyperparameters are set via the `--learning-rate, --batch-size, --large-batch-size` and `--exter-lambda` (which controls the regularization strength) arguments. In order to recreate the experiments, the optimal learning rate and lambda values are listed below:

TODO: insert table
