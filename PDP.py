from pdpbox import pdp, get_dataset, info_plots
from model_SVC import model as model_SVC
from model_random_forest import clf as model_rand_for
from data_split import X_test, X_train, y_train, y_test
from matplotlib import pyplot as plt

classifier = model_SVC
#classifier = model_rand_for

features = ['num_of_prev_attempts','weighted_grade','pass_rate','exam_score','date','sum_click']
pdp_goals = pdp.pdp_isolate(model=classifier, dataset=X_train, model_features= features, feature='exam_score')

pdp.pdp_plot(pdp_goals, 'exam_score')
plt.show()