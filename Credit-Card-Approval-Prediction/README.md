# Credit Card Approval Prediction

A machine learning project that predicts credit card approval decisions using Random Forest classification. The model analyzes various applicant features to determine the likelihood of credit card approval.

## Project Overview

This project implements a binary classification model to predict whether a credit card application will be approved or rejected. It includes comprehensive data preprocessing, exploratory data analysis, feature engineering, and model evaluation with multiple performance metrics.

## Features

- **Data Preprocessing**: Handles missing values through intelligent imputation
- **Feature Engineering**: One-hot encoding for categorical variables
- **Exploratory Data Analysis**: Comprehensive visualizations including histograms and correlation heatmaps
- **Model Implementation**: Random Forest Classifier with MinMax scaling
- **Performance Evaluation**: 
  - Accuracy metrics
  - Confusion matrix visualization
  - ROC curve analysis with AUC score

## Dataset

The project uses credit card application data stored in `Approval.xlsx`. The dataset includes various applicant features with a binary target variable `Approved` indicating approval status.

## Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms and tools
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization
- **scipy** - Scientific computing
- **Jupyter Notebook** - Interactive development environment

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AnjaliiD/Credit-Card-Approval-Prediction.git
cd Credit-Card-Approval-Prediction
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Ensure your data file `Approval.xlsx` is in the project directory

2. Open the Jupyter notebook:
```bash
jupyter notebook credit_card_approval.ipynb
```

3. Run all cells in the notebook to:
   - Load and preprocess the data
   - Generate exploratory visualizations
   - Train the Random Forest model
   - Display performance metrics and visualizations

## Model Performance

The model is evaluated using:
- **Accuracy Score**: Overall prediction accuracy
- **Confusion Matrix**: Detailed breakdown of predictions vs actual values
- **ROC Curve**: Model's ability to distinguish between classes with AUC score

## Project Structure

```
Credit-Card-Approval-Prediction/
│
├── credit_card_approval.ipynb # Main Jupyter notebook with complete pipeline
├── Approval.xlsx              # Dataset (included in the repo)
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── images/                    # Generated visualizations
```

## Workflow

1. **Data Loading**: Import data from Excel file
2. **Data Exploration**: Examine dataset structure and statistics
3. **Missing Value Handling**: 
   - Numeric features: Mean imputation
   - Categorical features: Mode imputation
4. **Feature Encoding**: One-hot encoding for categorical variables
5. **Visualization**: Generate histograms and correlation heatmap
6. **Data Splitting**: 80-20 train-test split
7. **Feature Scaling**: MinMax normalization
8. **Model Training**: Random Forest Classifier
9. **Evaluation**: Multiple metrics and visualizations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

For questions or feedback, please open an issue or contact anjalidesai0111@gmail.com.
