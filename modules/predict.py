from datetime import datetime
import json
import pandas as pd
import dill
import os

path = os.environ.get('PROJECT_PATH', '.')


def load_model(model_path: str):
    with open(model_path, 'rb') as file:
        return dill.load(file)


def read_json_to_df(file_path: str):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return pd.DataFrame([data]), os.path.basename(file_path)


def predict_for_dataframe(model, df: pd.DataFrame, filename: str) -> pd.DataFrame:
    predictions = model.predict(df)
    return pd.DataFrame({'filename': filename, 'predictions': predictions})


def predict():
    model = load_model(f'{path}/data/models/cars_pipe_{datetime.now().strftime("%Y%m%d%H%M")}.pkl')

    predictions_list = []

    for filename in os.listdir(f'{path}/data/test'):
        if filename.endswith('.json'):
            file_path = os.path.join(f'{path}/data/test', filename)
            df, file_name = read_json_to_df(file_path)
            predictions = predict_for_dataframe(model, df, file_name)
            predictions_list.append(predictions)

    final_predictions = pd.concat(predictions_list)
    final_predictions.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)


if __name__ == '__main__':
    predict()
