[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Promotion Predictor

**Студент:** Сергеев Матвей Вячеславович

**Группа:** БИВ-235


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
7. [Отчёт](#отчёт)


## Описание задачи

Необходимо предсказать, получит ли сотрудник повышение, основываясь на данных о его работе, таких как количество рабочих часов, зарплата, оценка качества и др.

**Задача:** Классификация

**Датасет:** https://www.kaggle.com/datasets/rohit8527kmr7518/employee-promtion-prediction

**Целевая метрика:** PR_AUC


## Структура репозитория
Опишите структуру проекта, сохранив при этом верхнеуровневые папки. Можно добавить новые при необходимости.
```
.
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходные файлы
├── models                      # Сохранённые модели 
├── notebooks
│   ├── 01_eda.ipynb            # EDA
│   ├── 02_baseline.ipynb       # Baseline-модель
│   └── 03_experiments.ipynb    # Эксперименты и ablation study
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Изображения для отчёта
│   └── report.md               # Финальный отчёт
├── src
│   ├── data_preprocessor.py        # Предобработка данных
│   └── machine_learner.py             # Обучение и оценка моделей
├── tests
│   └── test.py                 # Тесты пайплайна
├── requirements.txt
└── README.md
```

## Запуск

Этот блок замените способом запуска вашего сервиса.
```bash
# 1. Клонировать репозиторий
git clone https://github.com/hsemlcourse/hseml-group-project-mvsergeev
cd hseml-group-project-mvsergeev

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt
```

## Данные
- `data/raw/` — исходные файлы
- `data/processed/` — предобработанные данные


## Результаты
Здесь коротко выпишите результаты.
| Модель | [Метрика 1] | [Метрика 2] | Примечание |
|--------|-------------|-------------|------------|
| Baseline | — | — | |
| Лучшая модель | — | — | |


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
