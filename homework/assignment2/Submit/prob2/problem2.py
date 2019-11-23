import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift

def loadData(filePath):
    f = open(filePath, 'rb')  # deal with binary
    data = []
    img = image.open(f)  # return to pixel()
    m, n = img.size  # the size of image
    for i in range(m):
        for j in range(n):
            x, y, z = img.getpixel((i, j))
            # deal with pixel to the range 0-1 and save to data
            data.append([x / 256.0, y / 256.0, z / 256.0])
    f.close()
    return np.mat(data), m, n


imgData, row, col = loadData("12003.jpg")

# 1.Kmeans algorithm,set the n_clusters
label = KMeans(n_clusters=2).fit_predict(imgData)

# 2.EM algorithm set clusters
# gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0,)
# gmm.fit(imgData)
# label = gmm.predict(imgData)

# 3.mean-shift algorithm
# ms = MeanShift(bandwidth=2).fit(imgData)
# label = ms.labels_


# reshape the label
label = label.reshape([row, col])
# new image
pic_new = image.new("P", (row, col))
# define different color with RGB
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
sienna4 = (139, 71, 38)
yellow = (255, 255, 0)
lightGreen = (144, 238, 144)
white = (255, 255, 255)
black = (0, 0, 0)
dimGrey = (105, 105, 105)
lightYellow1 = (255, 255, 224)
for i in range(row):
    for j in range(col):
        if label[i][j] == 0:
            pic_new.putpixel((i, j), black)
        elif label[i][j] == 1:
            pic_new.putpixel((i, j), white)
        elif label[i][j] == 2:
            pic_new.putpixel((i, j), dimGrey)
        elif label[i][j] == 3:
            pic_new.putpixel((i, j), lightYellow1)
        elif label[i][j] == 4:
            pic_new.putpixel((i, j), sienna4)
        elif label[i][j] == 5:
            pic_new.putpixel((i, j), lightGreen)
        else:
            pic_new.putpixel((i, j), blue)
pic_new.show()
