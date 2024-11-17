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

## Visualizations

### Correlation Matrix Heatmap
Correlation Matrix
**Description**: This heatmap visualizes the correlations between different features in the dataset. It helps identify which features are strongly correlated, which can inform feature selection and dimensionality reduction.

### Histograms of Redshift for Each Class
Redshift Histograms
**Description**: The histograms display the distribution of redshift values for each class of celestial objects. Stars generally show lower redshift values compared to galaxies and quasars, indicating their proximity.

### Right Ascension vs. Declination Scatter Plot
RA vs Dec Scatter Plot
**Description**: The scatter plot of right ascension versus declination reveals spatial distributions of stars, galaxies, and quasars in the celestial sphere. Clear clustering patterns suggest that certain classes occupy distinct regions.

### PCA Explained Variance Plot
PCA Explained Variance
**Description**: The PCA explained variance plot indicates how much information is captured by each principal component. The first few components capture a significant portion of the variance.

### Confusion Matrix
Confusion Matrix
**Description**: The confusion matrix summarizes the performance of our best classifier, illustrating how well it distinguishes between stars, galaxies, and quasars.

### Feature Importance Bar Plot
Feature Importance
**Description**: The feature importance bar plot ranks features based on their contribution to model predictions. Redshift emerged as the most significant predictor.

## Results

The project demonstrates the effectiveness of different machine learning algorithms in classifying celestial objects. Key findings include:

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

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Sloan Digital Sky Survey for providing the dataset.
- University of Washington eScience Institute for guidance throughout this project.

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/36324829/8d2b9826-b304-4b63-bcc7-e495857fa99c/paste.txt
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/36324829/31bd796d-e2ed-414e-9ea5-2a2052ee8c94/paste-2.txt
