import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    balanced_accuracy_score, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns

from tools import create_pipeline, get_param_grid, get_display_name


class ModelTrainer:
    def __init__(
        self,
        data_path:str='../data/processed/employee_promotion_prediction.csv',
        target_col:str='promoted',
        test_size:float=0.2,
        val_size:float=0.25,
        random_state:int=42,
        cv:int=5,
        scoring:str='average_precision',
        n_jobs:int=-1
    ):

        self.data_path = data_path
        self.target_col = target_col
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs

        self._load_data()

        self.best_estimators_ = {}
        self.results_ = None

    def _load_data(self):
        df = pd.read_csv(self.data_path)
        self.X = df.drop(self.target_col, axis=1)
        self.y = df[self.target_col]

        x_temp, self.x_test, y_temp, self.y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y
        )

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            x_temp, y_temp,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=y_temp
        )

        self.x_train_val = pd.concat([self.x_train, self.x_val])
        self.y_train_val = pd.concat([self.y_train, self.y_val])

        self.num_cols = self.X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.cat_cols = self.X.select_dtypes(include=['object', 'category']).columns.tolist()

    def train_models(self, model_keys, verbose=True):
        for key in model_keys:
            display_name = get_display_name(key)
            if verbose:
                print(f"Обучение: {display_name}", end="\n\n")

            pipeline = create_pipeline(self.num_cols, self.cat_cols, key)
            param_grid = get_param_grid(key)

            grid = GridSearchCV(
                pipeline,
                param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=1 if verbose else 0
            )
            grid.fit(self.x_train_val, self.y_train_val)

            self.best_estimators_[key] = grid.best_estimator_

            if verbose:
                print(f"Лучшие параметры: {grid.best_params_}")
                print(f"Лучший {self.scoring} на CV: {grid.best_score_:.4f}")

    def evaluate_on_test(self, model_keys=None):
        if model_keys is None:
            model_keys = list(self.best_estimators_.keys())

        results = []
        for key in model_keys:
            if key not in self.best_estimators_:
                raise ValueError(f"Модель '{key}' не была обучена.")

            model = self.best_estimators_[key]
            display_name = get_display_name(key)

            y_pred = model.predict(self.x_test)
            y_proba = model.predict_proba(self.x_test)[:, 1]

            roc_auc = roc_auc_score(self.y_test, y_proba)
            pr_auc = average_precision_score(self.y_test, y_proba)
            f1 = f1_score(self.y_test, y_pred)
            balanced_acc = balanced_accuracy_score(self.y_test, y_pred)

            results.append({
                'model': display_name,
                'model_key': key,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'f1_score': f1,
                'balanced_accuracy': balanced_acc
            })

        scoring_name = 'pr_auc' if self.scoring == 'average_precision' else self.scoring
        self.results_ = pd.DataFrame(results).sort_values(scoring_name, ascending=False)
        return self.results_

    def print_results(self, sort_by=None):
        if self.results_ is None:
            print("Результаты ещё не вычислены.")
            return

        scoring_name = 'pr_auc' if self.scoring == 'average_precision' else self.scoring
        sort_col = sort_by if sort_by is not None else scoring_name
        results = self.results_.sort_values(sort_col, ascending=False)

        print()
        print("Сводная таблица результатов на тестовой выборке.")
        print(results[['model', 'roc_auc', 'pr_auc', 'f1_score', 'balanced_accuracy']]
              .to_string(index=False, float_format="%.4f"))

    def plot_confusion_matrices(self, model_keys=None):
        if model_keys is None:
            model_keys = list(self.best_estimators_.keys())

        n_models = len(model_keys)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        if n_models == 1:
            axes = [axes]

        for ax, key in zip(axes, model_keys):
            model = self.best_estimators_[key]
            y_pred = model.predict(self.x_test)
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(get_display_name(key))
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

        plt.tight_layout()
        plt.show()

    def plot_roc_curves(self, model_keys=None):
        if model_keys is None:
            model_keys = list(self.best_estimators_.keys())

        fig, ax = plt.subplots(figsize=(8, 6))
        for key in model_keys:
            model = self.best_estimators_[key]
            RocCurveDisplay.from_estimator(
                model, self.x_test, self.y_test,
                name=get_display_name(key), ax=ax
            )
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_title('ROC Curves')
        ax.legend()
        plt.show()

    def plot_pr_curves(self, model_keys=None):
        if model_keys is None:
            model_keys = list(self.best_estimators_.keys())

        fig, ax = plt.subplots(figsize=(8, 6))
        for key in model_keys:
            model = self.best_estimators_[key]
            PrecisionRecallDisplay.from_estimator(
                model, self.x_test, self.y_test,
                name=get_display_name(key), ax=ax
            )
        ax.set_title('Precision-Recall Curves')
        ax.legend()
        plt.show()