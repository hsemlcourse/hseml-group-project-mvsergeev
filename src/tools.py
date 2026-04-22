from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from models import MODELS


def create_pipeline(num_cols, cat_cols, model_key):
    if model_key not in MODELS:
        raise ValueError(f"Модель '{model_key}' не найдена. Доступные: {list(MODELS.keys())}")

    num_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline([
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    classifier = MODELS[model_key]['model']

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    return pipeline


def get_param_grid(model_key):
    return MODELS[model_key]['params']


def get_display_name(model_key):
    return MODELS[model_key]['display_name']