# House Price Prediction using Linear Regression

This project implements a house price prediction model using linear regression from scratch. It uses gradient descent optimization and includes data normalization for better model performance.

## Features

- Custom implementation of linear regression without using scikit-learn
- Gradient descent optimization with configurable learning rate
- Data normalization for better model performance
- Train-test split for model evaluation
- CSV data handling for both input and output
- Model evaluation metrics

## Requirements

- Python 3.x
- NumPy
- scikit-learn (for train_test_split)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ovais007/HousePrizePredictionusingLinearRegression.git
cd HousePrizePredictionusingLinearRegression
```

2. Install the required packages:
```bash
pip install numpy scikit-learn
```

## Data Format

The project expects two CSV files:

1. `train.csv`: Training data with features and target prices
2. `test.csv`: Test data with features only
3. `submission_example.csv`: Example submission format

## Usage

1. Place your data files in the project directory:
   - `train.csv`
   - `test.csv`
   - `submission_example.csv`

2. Run the main script:
```bash
python main_code.py
```

3. The script will:
   - Load and preprocess the data
   - Normalize the features
   - Train the linear regression model
   - Make predictions on the test set
   - Output predictions to `predictions.csv`

## Model Details

- The model uses gradient descent optimization
- Learning rate (alpha) is set to 0.1
- Features are normalized using mean and standard deviation
- The model continues training until the cost change is less than 0.00001

## Output

The script generates a `predictions.csv` file containing:
- ID: Identifier for each prediction
- Prediction: Predicted house price

## Performance

The model's performance is evaluated using the mean absolute percentage error (MAPE) on the test set. The accuracy is displayed as a percentage.

## License

MIT License

## Author

Mohammad Ovais 