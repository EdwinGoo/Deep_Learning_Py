# Day_03_03_preprocessing.py
from sklearn import preprocessing
import numpy as np

x = ['bali', 'jeju', 'seoul', 'seoul']

lb = preprocessing.LabelBinarizer()
lb.fit(x)
print(lb.transform(x))

lb = preprocessing.LabelBinarizer().fit(x)
print(lb.transform(x))

print(preprocessing.LabelBinarizer().fit_transform(x))

y = ['bali', 'jeju', 'jeju']
print(lb.transform(y))
print('-' * 50)

print(lb.classes_)      # ['bali' 'jeju' 'seoul']

onehot = lb.transform(y)
onehot_arg = np.argmax(onehot, axis=1)
print(onehot_arg)
print(lb.classes_[onehot_arg])
print('-' * 50)

x = ['bali', 'jeju', 'seoul', 'seoul']

le = preprocessing.LabelEncoder().fit(x)
enc = le.transform(x)
print(enc)
print(le.classes_)
print(le.classes_[enc])

print(np.identity(len(le.classes_)))

eye = np.eye(len(le.classes_), dtype=np.int32)
print(eye)
print(eye[enc])
