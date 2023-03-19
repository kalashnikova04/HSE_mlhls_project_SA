## Запуск проекта
---
Для успешного запуска проекта необходимо проделать следующее:

1. Клонировать репозиторий `https://github.com/kalashnikova04/HSE_mlhls_project_SA.git`

2. Перейти в папку `code_review`

3. Создать виртуальное окружение и установить зависимости - `pip install -r requirements.txt`

4. Выполнить команду `dvc repro`

Код выполнится по стадиям:

`download`: загрузка датасета с Google Disk

`split_test`: разбиение датасета на train и test

`split_val`: разбиение train на train и val

`objective`: оптимизация catboost с помощью optuna

`train`: обучение catboost на лучших параметрах
