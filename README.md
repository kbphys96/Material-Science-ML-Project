# Predicting Material Properties from Composition using Machine Learning

## Project overview

This project explores the use of machine learning to predict the properties of materials, such as bandgap, based on their elemental composition. The goal is to accelerate the process of materials discovery and design by offering a faster and more cost-effective method than traditional experimental or high-fidelity simulation techniques (e.g., DFT). By training predictive models on existing data, this system aims to aid in identifying promising materials for various applications, ranging from semiconductors to energy storage.

## Problem statement

Traditional materials discovery processes are often labor-intensive, time-consuming, and expensive. They struggle to efficiently navigate the vast space of possible material compositions and structures. This project addresses this bottleneck by leveraging machine learning to rapidly predict material properties, thereby streamlining the initial screening phase of materials research.

## Proposed solution

The proposed system utilizes a supervised machine learning approach to predict specific material properties (e.g., bandgap) from their composition. It involves the following steps:

1.  **Data Collection & Featurization:** Gathering existing datasets of materials with known properties and transforming their compositions into numerical features suitable for machine learning.
2.  **Model Training:** Implementing and training a regression model on the featurized data to learn the relationship between composition and material properties.
3.  **Prediction & Evaluation:**  The model predicts properties for new, unseen material compositions. The model's performance is then rigorously evaluated using appropriate metrics.

## System development approach (technology used)

*   **Programming Language:** Python
*   **Key Libraries:**
    *   `pandas` for data manipulation.
    *   `pymatgen` for representing chemical compositions and material data handling.
    *   `scikit-learn` for machine learning algorithms (e.g., `RandomForestRegressor`) and model evaluation.
    *   `numpy` for numerical operations.
    *   `matplotlib` and `seaborn` for data visualization.
*   **Development Environment:** Jupyter Notebooks are used for interactive development and analysis.
*   
## Algorithm and deployment

### Algorithm selection

A **Random Forest Regressor** is chosen for its robustness, ability to handle non-linear relationships, and capacity to handle relatively high-dimensional feature spaces, which are common in materials science datasets. This ensemble method builds multiple decision trees and averages their predictions to improve accuracy and reduce overfitting. 

### Data input

Computational Materials Data Sources
- Materials Project (MP)
  [1] A. Jain et al., *APL Materials*, 2013  
  https://materialsproject.org  
- Open Quantum Materials Database (OQMD)  
  [2] S. Kirklin et al., *npj Computational Materials*, 2015  
  http://oqmd.org  


All DFT calculations performed using PBE function.


### Training process

The model is trained using a split of the dataset (e.g., 80% for training, 20% for testing). k-fold cross-validation are employed to tune hyperparameters and ensure the model generalizes well to new, unseen material compositions.

### Prediction process

Once trained, the model predicts the target property for new material compositions. 


## Result (output image)

The model's performance is visually represented through a scatter plot comparing the actual vs. predicted values for the target material property from the test dataset. A line representing perfect prediction (actual = predicted) is also typically included for comparison.


 Scatter Plot: Actual vs. Predicted Material Property_

## Conclusion

This project successfully demonstrates the application of machine learning for predicting material properties from their composition, offering a faster alternative to traditional characterization and simulation methods. The chosen Random Forest regression model effectively learns complex relationships within the data, showing promise for practical applications in materials discovery.

## Future scope

*   **Expand Features:** Incorporate more advanced features such as structural parameters, bonding environment descriptors, or crystal graph representations.
*   **Advanced Algorithms:** Explore deep learning techniques, including {Link: Graph Neural Networks (GNNs) or Convolutional Neural Networks (CNNs) for capturing complex relationships, to potentially improve predictive power, especially with larger and more diverse datasets.
*   **Uncertainty Quantification:** Implement methods to quantify the uncertainty associated with predictions, providing more robust guidance for material design.
*   **Inverse Design:** Develop strategies to suggest material compositions given desired properties.
*   **Integration:** Incorporate the predictive model into a high-throughput screening or materials design workflow.



---
