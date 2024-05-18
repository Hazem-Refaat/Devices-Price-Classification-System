import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier

class PhonePriceModelTrainer:
    def __init__(self, iterations=1000, depth=10, learning_rate=0.1, loss_function='MultiClass'):
        self.model = CatBoostClassifier(iterations=iterations, depth=depth, learning_rate=learning_rate, loss_function=loss_function)

    def load_and_preprocess_data(self, input_csv):
        # load the dataset
        train_df = pd.read_csv(input_csv)
        # drop rows with missing values
        train_df.dropna(inplace=True)
        # prepare features and target
        X = train_df.drop(['price_range'], axis=1)
        y = train_df['price_range']
        # train-test split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_val, y_train, y_val

    def train_model(self, X_train, y_train, X_val, y_val):
        # train the model
        self.model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False, plot=False, save_snapshot=False)
        # evaluate the model
        y_pred = self.model.predict(X_val)
        print("Accuracy:", accuracy_score(y_val, y_pred))
        print(classification_report(y_val, y_pred))

    def save_model(self, model_path):
        # save the model
        self.model.save_model(model_path)

    def train_and_save(self, input_csv, model_path):
        X_train, X_val, y_train, y_val = self.load_and_preprocess_data(input_csv)
        self.train_model(X_train, y_train, X_val, y_val)
        self.save_model(model_path)

# usage
trainer = PhonePriceModelTrainer()
trainer.train_and_save('Python/Dataset/train.csv', 'Python/Model/catboost_phone_price_prediction.cbm')
