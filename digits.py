import pandas as pd
import numpy as np
from sklearn import tree,naive_bayes,neighbors
from sklearn.metrics import accuracy_score,confusion_matrix
import time
import matplotlib.pyplot as plt
import seaborn as sb

train_data=pd.read_csv('mnist_train.txt')
features_train=np.array(train_data.drop(['label'],'columns'))
labels_train=np.array(train_data['label'])

test_data=pd.read_csv('mnist_test.txt')
features_test=np.array(test_data.drop(['label'],'columns'))
labels_test=np.array(test_data['label'])

clf = tree.DecisionTreeClassifier()
#clf=naive_bayes.GaussianNB()

#clf = neighbors.KNeighborsClassifier()

#Training
t1=time.time()
clf.fit(features_train,labels_train)
t2=time.time()
print("training time",t2-t1)
t3=time.time()
pre=clf.predict(features_test)
t4=time.time()
print("testing time",t4-t3)

print("Predicted:",pre)
print("Actual digit:",labels_test)

acc=accuracy_score(pre,labels_test)
print("Accuracy=",acc)

p=29

print("Predicted digit:",pre[p])
print("actual digit:",labels_test[p])
digit=features_test[p]
digit_pixels=digit.reshape(28,28)

plt.imshow(digit_pixels,cmap='rainbow_r')
plt.show()

cm=confusion_matrix(labels_test,pre)
print("confusion matrix")
print(cm)

axis=plt.subplot()
sb.heatmap(cm,ax=axis,annot=True)
axis.set_xlabel("Predicted digits") 
axis.set_ylabel("actual digits")
axis.set_title("DecisionTree")
plt.show()