import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans


# Define loadDato to solve my image
def loadData(filePath):
    f = open(filePath, 'rb')  # deal with binary
    data = []
    img = image.open(f)  # return to pixel(像素值)
    m, n = img.size  # the size of image
    for i in range(m):
        for j in range(n):
            x, y, z = img.getpixel((i, j))
            # deal with pixel to the range 0-1 and save to data
            data.append([x / 256.0, y / 256.0, z / 256.0])
    f.close()
    return np.mat(data), m, n


imgData, row, col = loadData("lili.jpg")

# setting clusers(聚类中心) is 3
label = KMeans(n_clusters=2).fit_predict(imgData)

# get the label of each pixel
label = label.reshape([row, col])
# create a new image to save the result of K-Means
pic_new = image.new("P", (row, col))
# according to the label to add the pixel
new_pixel = []
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
Sienna4 = (139, 71, 38)
Yellow = (255, 255, 0)
for i in range(row):
    for j in range(col):
        pixel = int(256 / (label[i][j] + 1))
        new_pixel.append(pixel)
        if label[i][j] == 0:
            pic_new.putpixel((i, j), red)
        elif label[i][j] == 1:
            pic_new.putpixel((i, j), green)
        # elif label[i][j] == 2:
        #     pic_new.putpixel((i, j), blue)
        else:
            pic_new.putpixel((i, j), (255, 255, 255))
pic_new.show()
# print(new_pixel)
print(new_pixel.count(256))
print(new_pixel.count(128))
print(len(new_pixel))