import lime
from lime import lime_tabular
from data_split import X_test, X_train, y_train, y_test
import matplotlib.pyplot as plt
from model_random_forest import clf as model_rand_for
from model_SVC import model as model_SVC
# from model_keras import model as model_keras
# from model_keras import X_train_ker, X_test_ker, y_test_ker, y_train_ker
from model_MLP import model as model_MLP
import numpy as np


def run_lime_sklearn(classifier):
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns,
        class_names=['Fail', 'Pass'],
        mode='classification'
    )

    exp = explainer.explain_instance(
        data_row=X_test.iloc[1],
        predict_fn=classifier.predict_proba
    )

    plt.close()
    exp.as_pyplot_figure()
    plt.tight_layout()
    plt.show()


# run_lime_sklearn(model_rand_for)
# run_lime_sklearn(model_SVC)
# run_lime_sklearn(model_MLP)