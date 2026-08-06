from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report


def evaluate_model(model,X_test,y_test):

    prediction = model.predict(X_test)

    print("Accuracy :",accuracy_score(y_test,prediction))

    print()

    print(confusion_matrix(y_test,prediction))

    print()

    print(classification_report(y_test,prediction))