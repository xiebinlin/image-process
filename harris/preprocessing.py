from PIL import Image
import matplotlib.pyplot as plt
pil_im = Image.open('1.jpg')
box = (100,100,400,400)
region = pil_im.crop(box)

region.save('2.jpg');
plt.show()