from numpy import *
from PIL import Image
from pylab import *
import sift

imname1 = '1.jpg'
im1 = array(Image.open(imname1).convert('L'))
sift.process_image(imname1,'1.sift')
l1,d1 = sift.read_features_from_file('1.sift')
subplot(121)
sift.plot_features(im1, l1, circle=False)

imname2 = '2.jpg'
im2 = array(Image.open(imname2).convert('L'))
sift.process_image(imname2,'2.sift')
l2,d2 = sift.read_features_from_file('2.sift')
subplot(122)
sift.plot_features(im2, l2, circle=False)


m = sift.match_twosided(d1,d2)
figure()
sift.plot_matches(im1,im2,l1,l2,m,show_below=True)
savefig('xiaoguotu')
gray()
show()
