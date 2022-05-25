# Disentangling the Mechanisms Behind Implicit Regularization in SGD
Official code for NeurIPS 2022 Submission: Disentangling the Mechanisms Behind Implicit Regularization in SGD
​
## Requirements
To install requirements:
​
```setup
conda env create -f environment.yml
```
​
## Training
All the regularization experiments are run via `train.py` through the following:
​
* `Vanilla` - Normal small-batch SGD (batch size controlled by the `--micro-batch-size` argument)
* `PseudoGD` - Accumulated, large-batch SGD (batch size controlled by the `--batch-size` argument, and `--micro-batch-size` sets the size of the accumulated micro-batch)
* `RegLoss` - Large-batch SGD + average microbatch gradient norm regularization
* `FishLoss` - Large-batch SGD + average microbatch Fisher trace regularization
* `AvgJacLoss` - Large-batch SGD + average Jacobian regularization
* `UnitJacLoss` - Large-batch SGD + Unit Jacobian regularization
​
All hyperparameters are set via the `--learning-rate, --micro-batch-size, --batch-size` and `--exter-lambda` (which controls the regularization strength) arguments. In order to recreate the experiments, the optimal learning rate (η) and lambda values (λ) are listed in the table below:


| Model/Dataset      | SB SGD | LB SGD | LB + GN       | LB + FT       | LB + AJ        | LB + UJ        |
|--------------------|--------|--------|---------------|---------------|----------------|----------------|
| ResNet-18/CIFAR10  | η=0.1  | η=0.1  | η=0.1, λ=0.01 | η=0.1, λ=0.01 | η=0.1, λ=0.001 | η=0.1, λ=0.001 |
| ResNet-18/CIFAR100 | η=0.1  | η=0.5  | η=0.1, λ=0.01 | η=0.1, λ=0.01 | η=0.1, λ=5e-5  | η=0.1, λ=0.001 |
| VGG-11/CIFAR10     | η=0.15 | η=0.01 | η=0.01, λ=0.5 | η=0.01, λ=0.5 | η=0.01, λ=2e-5 | N/A            |


For example, running the following command trains a Resnet-18 on CIFAR-10 with average micro-batch gradient norm regularization (where batch size is 5120, learning rate is 0.1, regularization penalty is 0.01, and micro-batch size is 128)
​
```setup
python train.py --model='resnet' --dataset='cifar10' --batch-size=5120 --learning-rate=0.1 --exter-lambda=0.01 --micro-batch_size=128 --test='RegLoss'
```
​
## Evaluation
After training is complete, the model can be evaluated using `eval.py`. As long as the `--no-logging` flag is not turned on during training, the best performing model (in terms of validation accuracy) will be saved within a `saved_models/run_name` directory as `checkpoint_best.pth`. To evaluate the model, we must provide the path to this file in the `--path` argument to `eval.py`.
​
Building off of our Resnet-18 example earlier, we can run the following command to obtain the final test accuracy:
​
```setup
python eval.py --model='resnet' --dataset='cifar10' --batch-size=5120 --lr=0.1 --exter-lambda=0.01 --micro-batch_size=128 --test='RegLoss' --path='saved_models/run_name/checkpoint_best.pth'
```
​
## Results
​
Our models achieves the following test accuracies for various regularization penalties:
| Model/Dataset      | SB SGD | LB SGD | LB + GN | LB + FT | LB + AJ | LB + UJ |
|--------------------|--------|--------|---------|---------|---------|---------|
| ResNet-18/CIFAR10  | 92.64  | 89.83  | 91.75   | 91.50   | 90.13   | 90.15   |
| ResNet-18/CIFAR100 | 71.31  | 67.27  | 70.65   | 71.20   | 66.08   | 66.26   |
| VGG-11/CIFAR10      | 78.19  | 73.90  | 77.62   | 78.40   | 74.09   | N/A    |
