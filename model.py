import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import logging
import argparse
import os
import pickle
import warnings


logging.basicConfig(filename='./data/log_file.log', level=logging.INFO)
logger = logging.getLogger(__name__)

# Функция для записи предупреждений в файл логов
def warn_with_log(message, category, filename, lineno, file=None, line=None):
    log = logging.getLogger(__name__)
    log.warning(f'{filename}:{lineno}: {category.__name__}: {message}')

# Перенаправление предупреждений в файл логов
warnings.showwarning = warn_with_log
warnings.filterwarnings('always')  # Всегда выводить предупреждения

class My_Classifier_Model:
    def __init__(self):
        self.base_model = RandomForestClassifier()
        self.model = CatBoostClassifier(verbose=0)
        self.logger = logging.getLogger(__name__)

    def preprocess_data(self, df):
        df['HomePlanet'] = df['HomePlanet'].fillna(pd.Series(np.where(df['VIP'] == True, 'Europa', 'Earth'), index=df.index))
        cols_for_cryo_sleep = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        df['CryoSleep'] = df['CryoSleep'].fillna((df[cols_for_cryo_sleep] != 0.0).any(axis=1))

        imputer = SimpleImputer(strategy='mean')
        df[['Age']] = imputer.fit_transform(df[['Age']])

        imputer = SimpleImputer(strategy='most_frequent')
        df[['Cabin', 'VIP']] = imputer.fit_transform(df[['Cabin', 'VIP']])

        df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
        df = df.drop(columns=["Name", "Cabin", "Num"])

        df['Destination'] = df['Destination'].fillna('Unknown')

        df_VIP = df.loc[((df["VIP"] == True) & (df["CryoSleep"] == False))]
        df_com = df.loc[((df["VIP"] == False) & (df["CryoSleep"] == False))]
        
        for c in cols_for_cryo_sleep:
            mean_com = df_com[c].mean()
            mean_VIP = df_VIP[c].mean()
            df[c] = df.apply(lambda row: 0 if row['CryoSleep'] else (mean_com if not(row['VIP']) else mean_VIP) if pd.isnull(row[c]) else row[c], axis=1)

        le = LabelEncoder()
        df['Deck'] = le.fit_transform(df['Deck'])
        df['Side'] = le.fit_transform(df['Side'])
        df['HomePlanet'] = le.fit_transform(df['HomePlanet'])
        df['Destination'] = le.fit_transform(df['Destination'])

        df['VIP'] = df['VIP'].astype(int)
        df['CryoSleep'] = df['CryoSleep'].astype(int)

        return df

    def train(self, train_dataset):
        try:
             df_train = pd.read_csv(train_dataset)
             df_train = self.preprocess_data(df_train)
             listTransported = [False, True]
             mapTransported = {i: listTransported.index(i) for i in listTransported}
             df_train['Transported'] = df_train['Transported'].map(mapTransported)

             df_train = df_train.drop(columns=["PassengerId"])

             X = df_train.drop('Transported', axis=1)
             y = df_train['Transported']

             X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

             self.base_model.fit(X_train, y_train)

             cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
             self.model.fit(X_train, y_train, cat_features=cat_features)

             base_predictions = self.base_model.predict(X_val)
             base_accuracy = accuracy_score(y_val, base_predictions)

             predictions = self.model.predict(X_val)
             accuracy = accuracy_score(y_val, predictions)


             self.logger.info(f'Точность базового классификатора на валидационных данных: {base_accuracy}')
             self.logger.info(f'Точность модели CatBoost на валидационных данных: {accuracy}')

             with open('./data/model/base_model.pkl', 'wb') as f:
                 pickle.dump(self.base_model, f)

             with open('./data/model/catboost_model.pkl', 'wb') as f:
                 pickle.dump(self.model, f)

             self.logger.info("Модель успешно обучена и артефакты сохранены в ./data/model/")
        except Exception as e:
             self.logger.error(f"Ошибка при обучении модели: {str(e)}")

    def make_prediction(self, test_dataset):
        
        try: 
             df_test = pd.read_csv(test_dataset)

             df_test = self.preprocess_data(df_test)

             passenger_ids = df_test['PassengerId'].copy()

             df_test = df_test.drop(columns=["PassengerId"])
        
             base_model_path = './data/model/base_model.pkl'
             catboost_model_path = './data/model/catboost_model.pkl'
             if not os.path.exists(base_model_path) or not os.path.exists(catboost_model_path):
                 self.logger.error(f'Ошибка: Файлы моделей не найдены ({base_model_path}, {catboost_model_path})')
                 return
        
             with open('./data/model/base_model.pkl', 'rb') as f:
                 self.base_model = pickle.load(f)

             with open('./data/model/catboost_model.pkl', 'rb') as f:
                 self.model = pickle.load(f)

             base_test_predictions = self.base_model.predict(df_test)
             model_test_predictions = self.model.predict(df_test)

             base_test_predictions = (base_test_predictions == 1)
             model_test_predictions = (model_test_predictions == 1)
        
             model_submission = pd.DataFrame({'PassengerId': passenger_ids, 'Transported': model_test_predictions})
             model_submission.to_csv('./data/results.csv', index=False)
             
             self.logger.info("Прогнозы успешно сохранены в ./data/results.csv")
        except Exception as e: 
             self.logger.error(f"Ошибка при выполнении предсказаний: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Обучение и предсказание с использованием модели")
    parser.add_argument("action", type=str, choices=["train", "predict"], help="Выберите действие: обучение (train) или предсказание (predict)")
    parser.add_argument("--dataset", type=str, help="Путь к файлу набора данных")
    args = parser.parse_args()

    model = My_Classifier_Model()

    if args.action == "train":
        model.train(args.dataset)
    elif args.action == "predict":
        model.make_prediction(args.dataset)

if __name__ == "__main__":
    main()