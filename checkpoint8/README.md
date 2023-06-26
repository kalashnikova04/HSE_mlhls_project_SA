Сервис реализован на FastAPI и предоставляет два эндпоинта:


/predict_item - принимает объект типа Item, содержащий текст твита, и возвращает предсказание сентимента в виде числа типа float.


/predict_items - принимает список объектов типа Item, содержащих тексты твитов, и возвращает список предсказаний сентимента в виде чисел типа float.

/retrain_model - принимает файл с данными для дообучения


Для использования сервиса необходимо выполнить POST-запрос на один из эндпоинтов, передав в теле запроса соответствующий объект или список объектов в формате JSON.

POST http://0.0.0.0:8000/predict_item 


Request body


{"text": "string"}


POST http://0.0.0.0:8000/predict_items 


Request body


[
  {
    "text": "string"
  },
  {
    "text": "string"
  }
]

POST http://0.0.0.0:8000/retrain_model


Request body

uploaded_file=@dop_data.csv;type=text/csv