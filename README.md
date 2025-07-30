# California Housing Price Predictor

This project is a web application that predicts the median house value in California using a linear regression model trained on the California housing dataset. The app is built with Streamlit and allows users to input housing features to get a price prediction.

## Features

- Predict median house value based on user input features
- Displays model evaluation metrics including MAE, MSE, and R² score
- Interactive and user-friendly web interface using Streamlit

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Californiahousingdataset
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirement.txt
   ```

## Usage

Run the Streamlit app with the following command:
```bash
streamlit run app.py
```

This will open a web interface where you can input housing features and get a predicted median house value.

## Model Details

- Model: Linear Regression
- Dataset: California Housing Dataset from scikit-learn
- Features used: Median Income, House Age, Average Rooms, Average Bedrooms, Population, Average Occupancy, Latitude, Longitude

## Evaluation Metrics

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

These metrics are displayed in the app interface to provide insight into the model's performance.

## License

This project is open source and available under the MIT License. (Modify as needed)

## Contact

For questions or feedback, 
please contact 
Harshitha D K
9008016268
