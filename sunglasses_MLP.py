from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import os
import numpy as np
import cv2

result = [os.path.join(dp, f) for dp, dn, filenames in os.walk("./faces_4") for f in filenames if os.path.splitext(f)[1] == '.pgm']

X = []; 
y = [];
for imageFile in result:
    #get the pixel values for all the images
    img = cv2.imread(imageFile,0)
    X.append(img.flatten())
    if("sunglasses" in imageFile):
        y.append(1)
    else:
        y.append(0)

#split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=3)

parameters = {'solver':['lbfgs'], 'alpha': [i for i in range(1,3)], 'hidden_layer_sizes': [i for i in range(1,3)], 'random_state': [1,2,3]}
clf = GridSearchCV(estimator=MLPClassifier(),param_grid=parameters,n_jobs=-1,verbose=2,cv=10)

clf.fit(X_train, y_train)
y_pred_test = clf.predict(X_test)
print('Accuracy: %.2f' %accuracy_score(y_test, y_pred_test))
print(confusion_matrix(y_test, y_pred_test, labels=[0,1]))
target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred_test, target_names=target_names))
