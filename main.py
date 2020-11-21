from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.pyplot import draw,show
import numpy as np
import cv2
from collections import Counter
#from skimage.color import rgb2lab, deltaE_cie76
import os
def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))
def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
def func(pct, allvalues):
    absolute = int(pct / 100.*np.int(sum(allvalues)))
    return "{:.1f}%\n({:d} g)".format(pct, absolute)
def get_colors(image, number_of_colors, show_chart):
    modified_image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0] * modified_image.shape[1], 3)

    clf = KMeans(n_clusters=number_of_colors)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)
    # sort to ensure correct color percentage
    counts = dict(sorted(counts.items()))

    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    if (show_chart):
        plt.clf()
        plt.pie(counts.values(),autopct = lambda pct: func(pct, counts.values()), colors=hex_colors)

    return rgb_colors


while True:
    print("Press 1 to use Webcam\nPress 2 to manually add image\nPress 3 to exit")
    a = int(input())
    if a == 1:
        plt.close()
        plt.close()
        cap = cv2.VideoCapture(0)
        ret, image = cap.read()
        cv2.imshow('eg', image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    elif a == 2:
        plt.close()
        plt.close()
        print("Enter filename with extension (File must be present in local directory)")
        name = str(input())
        try:
            image = cv2.imread(name)
            print("The type of this input is {}".format(type(image)))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        except :
            print("Error ! Try again")
            continue
    elif a == 3:
        break
    else:
        continue
    plt.figure(1)
    plt.imshow(image)
    print("Shape: {}".format(image.shape))
    plt.figure(2)
    get_colors(image, 8, True)
    draw()

    plt.pause(0.001)
    cv2.destroyAllWindows()
