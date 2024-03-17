# ml_hits

Понтак Мирон, группа 972203

## Запуск скрипта Python из командной строки

1. Убедитесь, что у вас установлен Python
2. Установите необходимые библиотеки, выполнив команду:
   ```bash
   poetry install
   ```
3. Перейдите в корень проекта
4. Запустите скрипт с помощью следующей команды:
   ```bash
   python model.py train --dataset=data/train.csv
   ```
   Для предсказаний используйте команду:
   ```bash
   python model.py predict --dataset data/test.csv
   ```

## Отправка запроса в Flask

Для отправки запросов на обучение и предсказание используйте curl или Invoke-RestMethod (для PowerShell) с указанием URL-адреса Flask-приложения и соответствующими данными.

   Примеры для отправки запроса на обучение:
   ```bash
   curl -X POST -H "Content-Type: application/json" -d "{\"dataset\":\"data/train.csv\"}" http://localhost:5000/train
   ```
   ```powershell
   $body = @{ dataset="data/train.csv" } | ConvertTo-Json
   Invoke-RestMethod -Method Post -Uri "http://localhost:5000/train" -ContentType "application/json" -Body $body
   ```

   Примеры для отправки запроса на предсказание:
   ```bash
   curl -X POST -H "Content-Type: application/json" -d "{\"dataset\":\"data/test.csv\"}" http://localhost:5000/predict
   ```
   ```powershell
   $body = @{ dataset="data/test.csv" } | ConvertTo-Json
   Invoke-RestMethod -Method Post -Uri "http://localhost:5000/predict" -ContentType "application/json" -Body $body
   ```

## Запуск с использованием Docker

1. Установите Docker, если он еще не установлен.
2. Перейдите в корень проекта
3. Запустите приложение с помощью следующей команды:
   ```bash
   docker-compose up --build
   ```
