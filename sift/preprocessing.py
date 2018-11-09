from PIL import Image
im = Image.open('1.jpg')
box = (100,100,400,400)
region = im.transpose(Image.ROTATE_180)
region = region.crop(box)
region.save('2.jpg');
region.show()