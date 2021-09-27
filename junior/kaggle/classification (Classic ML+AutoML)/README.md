Задача на Kaggle по предсказанию дождя в Автралии на основе метеоданных

Датасет содержит как категориальные данные (например, направление ветра), так и колличественные данные (например, температура воздуха)

Решение состоит из следующих этапов:

- Обработка датасета (обработка пустых значений, OneHotEncoding для категориальных значенние, компоновка данных в спарс матрицу)
- Random Forest Classifier + scoring
- Logistic Regression + Grid Search CV + scoring
- Применение Sberbank LightAutoML + scoring

[ссылка на страницу задачи](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)

[ссылка на оргинальный notebook](https://www.kaggle.com/andreychubin/2-simple-models-automl-australia-rain)
