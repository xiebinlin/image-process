from PIL import Image
from numpy import *
import matplotlib.pyplot as plt
from pylab import *
from scipy.ndimage import filters
import harris

wid = 20
im1 = array(Image.open('1.jpg').convert('L'))
harrisim = harris.compute_harris_response(im1,11)
filtered_coords1 = harris.get_harris_points(harrisim,wid+1)
d1 = harris.get_descriptors(im1,filtered_coords1,wid)

im2 = array(Image.open('2.jpg').convert('L'))
harrisim = harris.compute_harris_response(im2,11)
filtered_coords2 = harris.get_harris_points(harrisim,wid+1)
d2 = harris.get_descriptors(im2,filtered_coords2,wid)


print('starting matching')
matches = harris.match_twosided(d1,d2)
figure()
gray()
harris.plot_matches(im1,im2,filtered_coords1,filtered_coords2,matches[:10])
show()

#plt.imshow()
#plt.imshow(im,origin='image')
#plt.show()