import numpy as np

latLower = 23-1
lonLower = 113-1
locBound = (23.0, 38.51, 113, 122.51)
smallLatAxis = np.arange(locBound[0],locBound[1], 0.125)

array = np.stack([smallLatAxis for i in range(20)])
ss = np.arange(0,4)
a,b,c,d = ss
print(array[1:1+17,2:2+17].shape)


