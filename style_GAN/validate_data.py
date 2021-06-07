from datagen import dataGenerator

import matplotlib.pyplot as plt

dg = dataGenerator('vfp_512', 512)
imgs = dg.get_batch(10)

for img in imgs:
    plt.imshow(img)
    plt.show()


