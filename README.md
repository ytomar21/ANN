# ANN Model
This repository highlights the usecase of a simple Artificial Neural Network against the [Predict Calorie Expenditure Dataset](https://www.kaggle.com/competitions/playground-series-s5e5/overview)

![ANN](ann.png)

## Code
1) ANN Model: ann.py
2) Datasets: train.csv, test.csv
3) Notebook: main.ipynb

## Evaluation

Root Mean Squared Logarithmic Error: 0.06042

## Dataset Overview

The csv dataset comprises of id, Sex, Age, Height, Weight, Duration, Heart_Rate, Body_Temp, and Calories. The target variable is calories, with the rest of columns acting as predictors. ID was removed as a predictor column.

## Model Overview

The ANN comprises of a single hidden layer with a Rectified Linear Unit (ReLU) activation function. The hyperparameters are defined as follows:

1) learning_rate = $5 ^ {-5}$
2) weight_decay = $1 ^ {-4}$
3) num_epochs = 100

### Learning Rate
Learning rate dictates the impact each gradient step has on the steps.

Gradient Rule:
$$ \Delta w_i = \eta \sum_{d \in D} (t_d - o_d)x_{id} $$

Where:

1) $t_d$ is the target output for training example $d$
2) $o_d$ is the ANN output
3) $x_{id}$ is the linear unit inputs
4) $\eta$ is the learning rate

### Weight Decay (L2 Regularization)
Penalizes larger weights and limits the freedom of the model.

Gradient Rule with Weight Decay:
$$ \Delta w_i = \eta \sum_{d \in D} (t_d - o_d)x_{id} - \eta \lambda w_i $$

Where:
1) $\lambda$ is the regularization parameter

### ReLU
ReLU is utilized as the activation function after the hidden layer is applied.

$$ f(x) = max(0, x) $$

