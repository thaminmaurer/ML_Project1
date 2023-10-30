# Project 1 of EPFL Machine Learning Course CS-433

## Authors

- Jonas Affentranger, 312045 (jonas.affentranger@epfl.ch)
- Ramon Huber, 367456 (ramon.huber@epfl.ch)
- Thamin Maurer, 313387 (thamin.maurer@epfl.ch)

## Introduction

This README provides an overview of the Python code we've written for the Machine Learning Project 1. The code is designed to implement various machine learning methods, preprocess the dataset, and generate predictions for a competition. Below, I'll break down the code's structure and functionality.

## The file implementations.py
In the file implementations.py are the implementations of all the machine learning methods implemeneted for this project including the methods seen in class and in the labs.

## The file run.ipynb
The file run.ipynb is the main file where the Machine Model is trained and predictions are created for the test data.

The following tasks are solved by the file run.ipynb
<ul>
<li>Load Dataset</li>
<li>Feature Selection</li>
<li>Dataset Preprocessing</li>
<li>Dataset Balancing</li>
<li>Dataset Splitting</li>
<li>Training and Evaluation</li>
<li>Visualization</li>
<li>Prediction</li>
<li>Plots Creation</li>
</ul>

1. **Load Dataset**<br>  The dataset is loaded from the csv files and the data is split into training and test data. The training data is further split into training and validation data. The data is split into 4 parts: <br>
    - **y_train** : The labels of the training data <br>
    - **x_train** : The features of the training data <br>
    - **y_test** : The labels of the test data <br>
    - **x_test** : The features of the test data <br>
```python	
x_train, x_test, y_train, train_ids, test_ids = helpers.load_csv_data("data/dataset")
```	

2. **Feature Selection**<br>  The code performs feature selection using the chi-squared test. The chi2_features function calculates chi-squared values for all features, and the top-k features are selected based on these scores. The indices of the selected features are stored in `selected_feature_indices`. The features from the chi-squared test did not give us a better score than the original features. Therefore, we decided to use other features that seemed to be more promising. <br>
The following features are used for the final submission: <br>
`20 (CADULT), 26 (GENHLTH), 27 (PHYSHLTH), 32 (MEDCOST), 33 (CHECKUP1), 34 (BPHIGH4), 35 (BPMEDS), 36 (BLOODCHO), 38 (TOLDHI2), 39 (CVDSTRK3), 42 (CHCSCNCR), 43 (CHCOCNCR), 45 (HAVARTH3), 48 (DIABETE3), 50 (SEX), 65 (QLACTLM2), 66 (USEEQUIP), 69 (DIFFWALK), 72 (SMOKE100), 73 (SMOKDAY2), 77 (ALCDAY5)`

3. **Dataset Preprocessing**<br>
A subset of the training dataset is created, including only the selected features, which is stored in best_x_train. The code also handles missing values (NaN) for specific columns, replacing them with appropriate values.

```python
best_x_train = x_train[:, chosen_indices]
best_x_train = change_nan(best_x_train)
```

4. **Dataset Balancing**<br>
The code rebalances the dataset to address class imbalance issues. It randomly undersamples the majority class to achieve a desired class proportion.

5. **Dataset Splitting**<br>
The dataset is split into training and validation sets. The first 80% of the data is used for training, and the remaining 20% for validation.
```python
X_train = X_balanced[:int(0.8 * len(X_balanced)]
X_val = X_balanced[int(0.8 * len(X_balanced))
Y_train = y_balanced[:int(0.8 * len(y_balanced)]
Y_val = y_balanced[int(0.8 * len(y_balanced)]
```

6. **Training and Evaluation**<br>
The machine learning methods are trained and evaluated. The code iterates through multiple training steps, storing the results and statistics such as accuracy, loss, F1 score, and more.

7. **Visualization**<br>
The code provides visualizations of accuracy, loss, and F1 score over the training iterations.

8. **Prediction**<br>
Finally, the code creates a submission file by applying the trained model to the test data. The best features are selected and missing values are handled.

9. **Plots Creation**<br>
The code creates plots for the report.

## Submissions
This file `AI_crowd_submission` is the final and best submission file for the AI Crowd competition. It contains the predictions for the test data. The file `baseline` is the baseline submission file to compare the results with the baseline.

## The file helpers.py
This file was provided in the Project Description

## Usage
To use this code, follow these steps:
1. Download the dataset from the [project page](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs-2019/dataset_files)
2. Place the dataset in the `data/dataset` folder
3. Ensure you have the necessary libraties installed, including NumPy and Matplotlib
4. Adjust the feature selection, dataset preprocessing, and balancing as needed based on your specific dataset and requirements
5. Select the machine learning method you want to use an duncomment the corresponding section of code for traning and evaluation
6. Run the `run.ipynb` file to train the model and generate predictions
7. Visualize the results and analyze the model's perfomance based on the provided visualizations and statistics

## Conclusion
This code serves as the foundation for implementing and experimenting with machine learning methods for Project 1. You can customize it further to fine-tune your model and adapt it to your specific dataset.

## Further Explanation
For further explanation and interpretation of the code we refer to the PDF report.