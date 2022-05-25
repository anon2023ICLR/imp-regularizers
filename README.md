# imp-regularizers
Code for NeurIPS 2022 Submission: Disentangling the Mechanisms Behind Implicit Regularization in SGD

All the regularization experiments are run via `train_regloss.py` through the following:

* `Vanilla` - Normal small-batch SGD (batch size controlled by the `--micro-batch-size` argument)
* `PseudoGD` - Accumulated, large-batch SGD (batch size controlled by the `--batch-size` argument, and `--micro-batch-size` sets the size of the accumulated micro-batch)
* `RegLoss` - Large-batch SGD + average microbatch gradient norm regularization
* `FishLoss` - Large-batch SGD + average microbatch Fisher trace regularization
* `AvgJacLoss` - Large-batch SGD + average Jacobian regularization
* `UnitJacLoss` - Large-batch SGD + Unit Jacobian regularization

All hyperparameters are set via the `--learning-rate, --micro-batch-size, --batch-size` and `--exter-lambda` (which controls the regularization strength) arguments. In order to recreate the experiments, the optimal learning rate and lambda values (as η and λ respectively) are listed below:

| Model/Dataset      | SB SGD | LB SGD | LB + GN       | LB + FT       | LB + AJ        | LB + UJ        |
|--------------------|--------|--------|---------------|---------------|----------------|----------------|
| ResNet-18/CIFAR10  | η=0.1  | η=0.1  | η=0.1, λ=0.01 | η=0.1, λ=0.01 | η=0.1, λ=0.001 | η=0.1, λ=0.001 |
| ResNet-18/CIFAR100 | η=0.1  | η=0.5  | η=0.1, λ=0.01 | η=0.1, λ=0.01 | η=0.1, λ=5e-5  | η=0.1, λ=0.001 |
| VGG11/CIFAR10      |        |        |               |               |                |                |
