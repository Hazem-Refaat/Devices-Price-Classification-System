from catboost import CatBoostClassifier
import pandas as pd

class PhonePricePredictor:
    def __init__(self, model_path):
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)

    def load_and_preprocess_data(self, input_csv):
        # load the test dataset
        test_df = pd.read_csv(input_csv)
        # drop rows with missing values
        test_df.dropna(inplace=True)
        # extract features
        X_test = test_df.drop(['id'], axis=1).values  # convert to NumPy array, exclude 'id' column if present
        return test_df, X_test

    def make_predictions(self, X_test):
        # make predictions
        return self.model.predict(X_test)

    def save_predictions(self, test_df, predictions, output_csv):
        # add the predictions as a new column to the test DataFrame
        test_df['price_range'] = predictions
        # save the updated DataFrame to a new CSV file
        test_df.to_csv(output_csv, index=False)

    def predict_and_save(self, input_csv, output_csv):
        test_df, X_test = self.load_and_preprocess_data(input_csv)
        predictions = self.make_predictions(X_test)
        self.save_predictions(test_df, predictions, output_csv)

# Usage
predictor = PhonePricePredictor('Python/Model/catboost_phone_price_prediction.cbm')
predictor.predict_and_save('Python/Dataset/test.csv', 'Python/Dataset/test_with_predictions.csv')