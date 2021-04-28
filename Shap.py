from data_split import X_test, X_train, y_train, y_test
from model_random_forest import clf as model_rand_for
from model_SVC import model as model_SVC
import matplotlib.pyplot as plt
from model_keras import model as model_keras
import numpy as np
import shap

classifier = model_SVC
#classifier = model_rand_for

#classifier = model_keras (needs fixing)


#We will use SHAP KernelExplainer to explain the model.
explainer = shap.KernelExplainer(model=classifier.predict_proba, data=X_train.iloc[0:100,:])

#Next, we compute the SHAP values
shap_values= explainer.shap_values(X=X_test.iloc[0:50,:])

#Since is binary classification, len = 2
print(len(shap_values))
#(50,6) - 50 objects, 6 features
print(shap_values[0].shape)

#Explaining a single prediction for passing
shap.initjs()
plot = shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_test.iloc[0,:])
shap.save_html('plot_1_instances.html', plot)

#Explaining a single prediction for failing
plot = shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], X_test.iloc[0,:])
shap.save_html('plot_2_instances.html', plot)

#Explaining predictions for passing for 50 instances of X_test
plot = shap.force_plot(explainer.expected_value[1], shap_values[1], X_test)
shap.save_html('plot_X_test_instances.html', plot)

#Shap summary plot
print(shap.summary_plot(shap_values, X_test))
