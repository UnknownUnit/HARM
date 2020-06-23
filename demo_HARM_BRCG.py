from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from HARM import HARM_BRCG
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# get our data 
data, labels = load_breast_cancer(return_X_y =True)
train_data, test_data, train_labels, test_labels = train_test_split(data, labels)

# create our models.
random_forest = RandomForestClassifier()
hybrid_model = HARM_BRCG(train_data, train_labels, random_forest)

# train our models.
hybrid_model.train_black_box()
hybrid_model.binarize_data()
hybrid_model.train_r_0()
hybrid_model.train_r_1()

# Evaluate our model.
transparent_calls, total_calls, final_prediction = hybrid_model.predict(test_data)
print("transparency ratio:", transparent_calls/total_calls)
print(classification_report(test_labels, final_prediction))

# print the explanations for r_0
print(hybrid_model.r_0.explain())

# print the explanations for r_1
print(hybrid_model.r_1.explain())