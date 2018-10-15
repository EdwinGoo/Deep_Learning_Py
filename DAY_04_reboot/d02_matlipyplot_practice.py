from matplotlib.image import imread
import matplotlib.pyplot as plt

## task.json 지정된 경로를 기준으로 경로를 작성해야함
mozzi = imread('data/mozzi.jpg')
print(type(mozzi))
print(mozzi.shape)
print(mozzi.dtype)

# plt.imshow(mozzi)
plt.imshow(mozzi[:, :, 2])

plt.show()

