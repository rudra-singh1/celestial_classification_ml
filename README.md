# Celestial Object Classification using Machine Learning

This project implements various machine learning techniques to classify celestial objects as stars, galaxies, or quasars using data from the Sloan Digital Sky Survey (SDSS). The project showcases a comprehensive machine learning workflow, including data preprocessing, feature engineering, dimensionality reduction, unsupervised clustering, and supervised classification.

## Project Overview

The Sloan Digital Sky Survey is a major multi-spectral imaging and spectroscopic redshift survey using a dedicated 2.5-meter wide-angle optical telescope at Apache Point Observatory in New Mexico, United States. This project utilizes SDSS data to build and compare different machine learning models for celestial object classification.

## Features

- Data preprocessing and cleaning
- Exploratory data analysis with visualizations
- Feature correlation analysis
- Dimensionality reduction using Principal Component Analysis (PCA)
- Unsupervised clustering with K-Means
- Implementation of multiple classification algorithms:
  - K-Nearest Neighbors
  - Naive Bayes
  - Random Forest
  - Support Vector Machine
  - Multi-Layer Perceptron
- Model evaluation and comparison
- Feature importance analysis

## Dataset

The dataset contains various features of celestial objects, including:

- Right Ascension (ra) and Declination (dec)
- Magnitude measurements in different optical filters (u, g, r, i, z)
- Redshift
- Other spectroscopic and photometric measurements

## Technologies Used

- Python
- Pandas for data manipulation
- NumPy for numerical operations
- Matplotlib and Seaborn for data visualization
- Scikit-learn for machine learning algorithms and preprocessing

## Project Structure

1. **Data Preparation**
   - Data cleaning and initial exploration.
   - Correlation analysis.
   - Feature distribution analysis.
   - Dimensionality reduction with PCA.

2. **Unsupervised Clustering**
   - K-Means clustering.
   - Optimal cluster number determination.
   - Clustering performance evaluation.

3. **Supervised Machine Learning Models**
   - Feature scaling.
   - Train/test split.
   - Model implementation and evaluation for multiple algorithms.
   - Cross-validation.
   - Model comparison.

4. **Feature Importance Analysis**
   - Using Random Forest to determine key predictive features.

## Data Preparation

### Data Cleaning

The initial step involves reading the dataset into a Pandas DataFrame and performing basic cleaning operations. We drop unnecessary columns such as `objid`, `specobjid`, `run`, `rerun`, `camcol`, and `field`, which do not contribute to our predictive model.

```python
import pandas as pd

# Load the dataset
sdss_df = pd.read_csv('Skyserver_SQL2_27_2018_6_51_39_PM.csv', skiprows=1)

# Dropping irrelevant columns
sdss_df.drop(['objid', 'specobjid', 'run', 'rereun', 'camcol', 'field'], axis=1, inplace=True)
```

### Correlation Analysis

To understand the relationships between features, we compute the correlation matrix. This helps identify which features are strongly correlated.

#### Correlation Matrix Heatmap

![Correlation Heatmap](https://github.com/rudra-singh1/celestial_classification_ml/blob/main/correlation_heatmap.png)

The correlation matrix heatmap reveals strong relationships between magnitude measurements (u, g, r, i, z), while redshift shows distinct patterns for different celestial objects.

### Distribution of Redshift

Next, we analyze the distribution of redshift values across different classes of celestial objects.

#### Histograms of Redshift for Each Class

![Redshift Distribution](https://github.com/rudra-singh1/celestial_classification_ml/blob/main/redshift.png)

The redshift distributions show distinct patterns: stars typically have lower redshift values, while quasars show the highest redshift values, indicating their vast distances from Earth.

### Right Ascension vs. Declination Scatter Plot

We visualize the spatial distribution of celestial objects using right ascension and declination coordinates.

![Right Ascension and Declination](https://github.com/rudra-singh1/celestial_classification_ml/blob/main/ra.png)

The spatial distribution of objects across right ascension and declination coordinates reveals the survey's coverage pattern and object clustering.

## Dimensionality Reduction with PCA

We apply Principal Component Analysis (PCA) to reduce dimensionality while retaining variance in our dataset. After fitting PCA on our feature set (redshift, u, g, r, i, z), we analyze the explained variance ratio to determine how many components to retain.

#### PCA Explained Variance Plot

![PCA Analysis](https://github.com/rudra-singh1/celestial_classification_ml/blob/main/pca.png)

Principal Component Analysis of the five magnitude measurements (u, g, r, i, z) shows that the first two components capture most of the variance in the data.

## Unsupervised Clustering with K-Means

We perform K-Means clustering to explore whether our features are sufficient for classification.

### Preliminary K-Means Results

After implementing K-Means clustering on selected features:

![KMeans Clustering](https://github.com/rudra-singh1/celestial_classification_ml/blob/main/kmeans.png)

K-means clustering analysis reveals natural groupings in the data that largely correspond to the three celestial object classes.

### Optimal Number of Clusters

Using methods such as the Elbow method or Silhouette analysis helps us determine the optimal number of clusters for our K-Means model.

## Supervised Machine Learning Models

We implement several classification algorithms on our cleaned dataset:

1. **K-Nearest Neighbors**
2. **Naive Bayes**
3. **Random Forest**
4. **Support Vector Machine**
5. **Multi-Layer Perceptron**

Each model is evaluated based on accuracy, precision, recall, and F1-score metrics.

### Confusion Matrix Visualization

![Confusion Matrix](https://github.com/rudra-singh1/celestial_classification_ml/blob/main/confusionmatrix.png)

The confusion matrix demonstrates the classification performance across the three classes, with particularly strong performance in distinguishing stars from other celestial objects.

### Feature Importance Analysis with Random Forest

Using Random Forest allows us to rank features based on their contribution to model predictions.

#### Feature Importance Bar Plot

![Feature Importance](https://github.com/rudra-singh1/celestial_classification_ml/blob/main/featureplot.png)

Random Forest feature importance analysis reveals that redshift is the most significant predictor, followed by magnitude measurements in different wavelength bands.

## Results Summary

The project demonstrates the effectiveness of different machine learning algorithms in classifying celestial objects based on SDSS data. Key findings include:

- Comparison of model performances across different metrics.
- Insights into the most important features for classification.
- Analysis of clustering effectiveness for this astronomical dataset.

## Future Improvements

- Implement more advanced deep learning models.
- Explore additional feature engineering techniques.
- Incorporate more diverse datasets from other astronomical surveys.
- Optimize hyperparameters for better model performance.

## Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.

## Acknowledgments

- Sloan Digital Sky Survey for providing the dataset.
---
