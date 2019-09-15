# coding:utf-8
from PIL import Image
import numpy as np
img = Image.open('D:\\GitHub\\python-star\\img\\lena.jpg')
arrayImg = np.array(img)
height = 100
img_width,img_height = img.size
width = int(1.8 * height * img_width // img_height)
img = img.resize((width,height),Image.ANTIALIAS)
pixels = np.array(img.convert('L'))
print('type(pixeels) = ', type(pixels))
print(pixels.shape)
chars = "MNHQ&OC?7>!:-;. "
N =len(chars)
step = 256 // N
print(N)
result = ''
for i in range(height):
    for j in range(width):
        result += chars[pixels[i][j] // step]
    result += '\n'
with open('lena.txt', mode='w') as f:
    f.write(result)
