from data_split import X_test, X_train, y_train, y_test
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import shap

#model = SVC(kernel='linear', probability=True)
model = RandomForestClassifier()
fitted_model = model.fit(X_train, y_train)

data_for_prediction = X_test.iloc[0:50,:]
data_for_prediction2 = X_test.iloc[6]

print(fitted_model.predict_proba(data_for_prediction))

explainer = shap.TreeExplainer(fitted_model)

#Analysis for 50 instances
shap_values = explainer.shap_values(X = X_test.iloc[0:50,:])
#Analysis for an instance
shap_values2 = explainer.shap_values(data_for_prediction2)

shap.initjs()
plot = shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
shap.save_html('plot_50_instances.html', plot)

plot2 = shap.force_plot(explainer.expected_value[1], shap_values2[1], data_for_prediction2)
shap.save_html('plot_1_instance.html', plot2)