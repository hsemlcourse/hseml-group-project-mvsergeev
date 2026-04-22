from data_preprocessor import preprocess_data
from machine_learner import ModelTrainer


def main():
    preprocess_data()
    trainer = ModelTrainer(scoring='average_precision')

    models_to_train = ['logreg', 'rf', 'gb', 'knn']
    trainer.train_models(models_to_train)

    trainer.evaluate_on_test()
    trainer.print_results()

    trainer.plot_roc_curves()
    trainer.plot_confusion_matrices()


if __name__ == "__main__":
    main()