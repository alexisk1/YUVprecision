import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
from scipy.misc import imresize
import numpy as np
def RGB2YCrCb444(img):
   return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)



def YCrCb444to422(img):
   print(img.shape)

   img = imresize(img, img.shape, interp='nearest', mode=None)
   Y= img[:,:,0]
   imgCr= img[:,:,1]
   print(imgCr.shape)
   imgCb= img[:,:,2]
   Cr= imgCr[1::2,:]
   print(Cr.shape)
   Cb= imgCb[1::2,:]

   return img


img = mpimg.imread("./horse.jpg")
fig = plt.figure()
a=fig.add_subplot(1,2,1)
imgplot = plt.imshow(YCrCb444to422(img))
a.set_title('Before')
plt.colorbar(ticks=[0.1,0.3,0.5,0.7], orientation ='horizontal')
a=fig.add_subplot(1,2,2)
imgplot = plt.imshow(img)
imgplot.set_clim(0.0,0.7)
a.set_title('After')
plt.colorbar(ticks=[0.1,0.3,0.5,0.7], orientation='horizontal')
plt.title('Result')
plt.show()



