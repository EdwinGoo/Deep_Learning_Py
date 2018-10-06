from sklearn import preprocessing
import numpy as np

x = ['bali','jeju', 'seoul', 'seoul', 'jeju']
lb = preprocessing.LabelBinarizer()
lb.fit(x)

print(lb.transform(x))

#요래 하면 편하겠지만
print(preprocessing.LabelBinarizer().fit_transform(x))
#요렇게 다른 값을 폼으로 만들 때는 분리가 편하지
y = ['bali', 'jeju']
print(lb.transform(y))
print(lb.classes_)

onehot = lb.transform(x)
onehot_argmax = np.argmax(onehot, axis=1)
print(onehot_argmax)
print(lb.classes_[onehot_argmax])

print("▨" * 100)

x = ['bali','jeju', 'seoul', 'seoul', 'jeju']
print(preprocessing.LabelEncoder().fit_transform(x)) # onehot_argmax 추출
