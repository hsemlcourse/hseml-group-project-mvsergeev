import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import chi2


def preprocess_data(raw_file_path:str="../data/raw/employee_promotion_prediction.csv",
                    processed_file_path:str="../data/processed/employee_promotion_prediction.csv",
                    size:int = 11000,
                    force_create:bool = False):

    if not force_create and os.path.exists(processed_file_path):
        print("Препроцессинг уже выполнен.")
        return

    sns.set_theme(style='darkgrid')

    full_df = pd.read_csv(raw_file_path)
    processed_df = full_df.sample(n=size)
    processed_df = processed_df.drop(columns=["employee_id"])

    print("Информация о наборе данных:", processed_df.info(), sep="\n", end="\n\n")

    num_cols = processed_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    stats = processed_df[num_cols].agg(['min', 'mean', 'max']).T
    print("Числовые признаки:", stats, sep="\n", end="\n\n")

    processed_df['avg_monthly_hours'].plot.hist()
    plt.title("Среднее количество рабочих часов в неделю")

    processed_df = filter_outliers_zscore(processed_df, num_cols)

    stats = processed_df[num_cols].agg(['min', 'mean', 'max']).T
    print("Статистика после удаления выбросов:", stats, sep="\n", end="\n\n")
    print("Информация о наборе данных после удаления выбросов:", processed_df.info(), sep="\n", end="\n\n")

    corr_matrix = processed_df[num_cols].corr()

    target_corr = corr_matrix['promoted'].drop('promoted')
    top_features = target_corr.abs().sort_values(ascending=False).head(15).index.tolist()

    cols_for_heatmap = top_features + ['promoted']
    corr_matrix_top = processed_df[cols_for_heatmap].corr()

    plt.figure(figsize=(12, 10), constrained_layout=True)

    sns.heatmap(
        corr_matrix_top,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        vmin=-1, vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )

    plt.title('Корреляция числовых признаков с promoted')

    Y = processed_df["promoted"]
    X = processed_df.drop(columns=["promoted"])

    cat_cols = X.select_dtypes(include=['str', 'category']).columns.tolist()
    #Среди важных числовых признаков выберем множество наиболее независимых
    num_cols = ["performance_score", "projects_completed", "salary", "leadership_score", "job_satisfaction_score"]

    cat_features_importance = calculate_categorial_features_importance(X, Y, cat_cols)
    print("Важность категориальных признаков:", end="\n\n")
    print(cat_features_importance.head(6))

    cat_cols = cat_features_importance["feature"].tolist()[:4]
    all_cols = num_cols + cat_cols + ["promoted"]

    print(X.head(10))

    plt.show()
    processed_df[all_cols].to_csv(processed_file_path, index=False)


def filter_outliers_zscore(df: pd.DataFrame, columns: list, z_thresh: int = 3.5):
    mask = pd.Series(True, index=df.index)
    for col in columns:
        mean_c = df[col].mean()
        std_c = df[col].std()
        upper_bound = mean_c + z_thresh * std_c
        mask &= (df[col] <= upper_bound)
    return df[mask]


def calculate_categorial_features_importance(features: pd.DataFrame, target: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    cat_importance_agg = {}

    for col in cat_cols:
        dummies = pd.get_dummies(features[col], prefix=col, drop_first=True)

        chi2_scores, p_values = chi2(dummies, target)

        total_chi2 = np.sum(chi2_scores)
        cat_importance_agg[col] = total_chi2

    cat_importance_df = pd.DataFrame({
        'feature': list(cat_importance_agg.keys()),
        'chi2_sum': list(cat_importance_agg.values())
    }).sort_values('chi2_sum', ascending=False)
    return cat_importance_df