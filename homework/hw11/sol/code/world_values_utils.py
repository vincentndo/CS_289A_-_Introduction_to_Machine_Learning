import pandas as pd
from math import sqrt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


def import_world_values_data():
    """
    Reads the world values data into data frames.

    Returns:
        values_train: world_values responses on the training set
        hdi_train: HDI (human development index) on the training set
        values_test: world_values responses on the testing set
    """
    values_train = pd.read_csv('world-values-train2.csv')
    values_train = values_train.drop(['Country'], axis=1)
    values_test = pd.read_csv('world-values-test.csv')
    values_test = values_test.drop(['Country'], axis=1)
    hdi_train = pd.read_csv('world-values-hdi-train2.csv')
    hdi_train = hdi_train.drop(['Country'], axis=1)
    return values_train, hdi_train, values_test


def plot_hdi_vs_feature(training_features, training_labels, feature, color, title):
    """
    Input:
    training_features: world_values responses on the training set
    training_labels: HDI (human development index) on the training set
    feature: name of one selected feature from training_features
    color: color to plot selected feature
    title: title of plot to display

    Output:
    Displays plot of HDI vs one selected feature.
    """
    plt.scatter(training_features[feature],
    training_labels['2015'],
    c=color)
    plt.title(title)
    plt.show()


def calculate_correlations(training_features,
                           training_labels):
    """
    Input:
        training_features: world_values responses on the training set
        training_labels: HDI (human development index) on the training set

    Output:
        Prints correlations between HDI and each feature, separately.
        Displays plot of HDI vs one selected feature.
    """
    # Calculate correlations between HDI and each feature
    correlations = []
    for column in training_features.columns:
        print(column, training_features[column].corr(training_labels['2015']))
        correlations.append(round(training_features[column].corr(training_labels['2015']), 4))
    print(correlations)
    print()

    # Identify three features
    feature_list = list(training_features.columns)
    feature_correlation = dict( zip(feature_list, correlations) )
    positive_correlation = max( feature_correlation, key=feature_correlation.get )
    negative_correlation = min( feature_correlation, key=feature_correlation.get )
    least_correlation = min( feature_correlation, key=lambda x: abs(feature_correlation.get(x)) )
    print("Most positively correlated:")
    print(positive_correlation + ": " + str(feature_correlation.get(positive_correlation)))
    plot_hdi_vs_feature(training_features, training_labels, positive_correlation,
                        'green', 'HDI versus ' + positive_correlation)
    print()
    print("Most negatively correlated:")
    print(negative_correlation + ": " + str(feature_correlation.get(negative_correlation)))
    plot_hdi_vs_feature(training_features, training_labels, negative_correlation,
                        'magenta', 'HDI versus ' + negative_correlation)
    print()
    print("Least correlated:")
    print(least_correlation + ": " + str(feature_correlation.get(least_correlation)))
    plot_hdi_vs_feature(training_features, training_labels, least_correlation,
                        'blue', 'HDI versus ' + least_correlation)
    print()
    print("Observation: For most positively correlated HDI-feature, the points spread in forward flash shape (/)\n" +
        "For most negatively correlated HDI-feature, the points spread in backward flash shape (/)\n" +
        "For least correlated HDI-feature, the points spread in C shape")
    print()


def plot_pca(training_features,
             training_labels,
             training_classes):
    """
    Input:
        training_features: world_values responses on the training set
        training_labels: HDI (human development index) on the training set
        training_classes: HDI class, determined by hdi_classification(), on the training set

    Output:
        Displays plot of first two PCA dimensions vs HDI
        Displays plot of first two PCA dimensions vs HDI, colored by class
    """
    # Run PCA on training_features
    pca = PCA()
    transformed_features = pca.fit_transform(training_features)

    # Plot countries by first two PCA dimensions
    plt.scatter(transformed_features[:, 0],     # Select first column
                transformed_features[:, 1],     # Select second column
                c=training_labels["2015"])
    plt.colorbar(label='Human Development Index')
    plt.title('Countries by World Values Responses after PCA')
    plt.show()

    # Plot countries by first two PCA dimensions, color by class
    # training_colors = training_classes.apply(lambda x: 'green' if x else 'red')
    # plt.scatter(transformed_features[:, 0],     # Select first column
    #             transformed_features[:, 1],     # Select second column
    #             c=training_colors)
    # plt.title('Countries by World Values Responses after PCA')
    # plt.show()


def plot_pca_2(training_features,
             training_labels,
             training_classes):
    # Run PCA on training_features
    pca = PCA()
    transformed_features = pca.fit_transform(training_features)
    
    # Plot countries by first two PCA dimensions, color by class
    training_colors = training_classes.apply(lambda x: 'green' if x else 'red')
    plt.scatter(transformed_features[:, 0],     # Select first column
                transformed_features[:, 1],     # Select second column
                c=training_colors)
    plt.title('Countries by World Values Responses after PCA (Low-High HDI)')
    plt.show()


def hdi_classification(hdi):
    """
    Input:
        hdi: HDI (human development index) value

    Output:
        high HDI vs low HDI class identification
    """
    if 1.0 > hdi >= 0.7:
        return 1.0
    elif 0.7 > hdi >= 0.30:
        return 0.0
    else:
        raise ValueError('Invalid HDI')
