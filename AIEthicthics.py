import pandas as pd
import matplotlib.pyplot as plt
from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. Load the COMPAS dataset
dataset = CompasDataset()

# 2. Define privileged and unprivileged groups (race-based)
privileged_groups = [{'race': 1}]   # Caucasian
unprivileged_groups = [{'race': 0}]  # African-American

# 3. Initial fairness metrics (dataset-level)
metric = BinaryLabelDatasetMetric(
    dataset,
    privileged_groups=privileged_groups,
    unprivileged_groups=unprivileged_groups
)

print("Disparate Impact:", metric.disparate_impact())
print("Statistical Parity Difference:", metric.statistical_parity_difference())

# 4. Split into train and test sets
train, test = dataset.split([0.7], shuffle=True)

# 5. Apply reweighing to mitigate bias
RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
RW.fit(train)
train_transf = RW.transform(train)

# 6. Train a logistic regression model
X_train = train_transf.features
y_train = train_transf.labels.ravel()
X_test = test.features
y_test = test.labels.ravel()

# 7. Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Fit the model
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

# 9. Predict on test set
y_pred = model.predict(X_test)

# 10. Replace labels in test dataset with predictions
test_pred = test.copy()
test_pred.labels = y_pred

# 11. Classification-level fairness metrics
classified_metric = ClassificationMetric(
    test,
    test_pred,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

print("Equal Opportunity Difference:",
      classified_metric.equal_opportunity_difference())
print("Average Odds Difference:", classified_metric.average_odds_difference())
print("False Positive Rate Difference:",
      classified_metric.false_positive_rate_difference())

# 12. Plot False Positive Rates
fpr_priv = classified_metric.false_positive_rate(privileged=True)
fpr_unpriv = classified_metric.false_positive_rate(privileged=False)

races = ['Privileged (Caucasian)', 'Unprivileged (African-American)']
fpr_values = [fpr_priv, fpr_unpriv]

plt.bar(races, fpr_values, color=['green', 'red'])
plt.title("False Positive Rate by Race")
plt.ylabel("FPR")
plt.tight_layout()
plt.savefig("fpr_by_race.png")
plt.show()
