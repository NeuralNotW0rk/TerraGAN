import os
import math
import numpy as np
import matplotlib.pyplot as plt

# standard hgt is 1201x1201 where each pixel is 3 arc-seconds (90 m)

dim = 1024
max_elevation = 10000

sample_size = 256
samples_per_img = 64


def process_section(name):
    raw_dir = 'raw_data/vfp/' + name + '/'

    lats = []
    lons = []

    d = int(math.sqrt(os.path.getsize(raw_dir + os.listdir(raw_dir)[0]) / 2))
    for f in os.listdir(raw_dir):
        lats.append(int(f[1:3]))
        lons.append(int(f[4:7]))

    lat0 = min(lats)
    lat1 = max(lats)
    lon0 = min(lons)
    lon1 = max(lons)

    data = np.zeros(((lat1 - lat0 + 1) * d, (lon1 - lon0 + 1) * d))
    print(data.shape)
    for f in os.listdir(raw_dir):
        print(raw_dir + f)
        s = np.fromfile(raw_dir + f, np.dtype('>i2'), d * d).reshape((d, d))
        lat = int(f[1:3])
        lon = int(f[4:7])
        for y in range(len(s)):
            for x in range(len(s[0])):
                data[(lat1 - lat) * d + y][(lon1 - lon) * d + x] = s[y][x]
        # plt.imsave('data/vfp/' + name + '/' + f[:-4] + '.jpg', s, cmap='gray')
        # np.save('data/' + name + '/' + f[:-4] + '.npy', s)
    plt.imsave('data/region/' + name + '.png', data, vmin=-max_elevation, vmax=max_elevation, cmap='gray')


def hgt_to_jpg(file, target, x, y, image=True):
    d = int(math.sqrt(os.path.getsize(file) / 2))
    hgt = np.fromfile(file, np.dtype('>i2'), d * d).reshape((d, d))
    region = str(os.path.split(file)[0].split('/')[-1])
    for s in range(samples_per_img):
        pos = np.random.randint(0, d - sample_size, 2)
        sample = np.copy(hgt[pos[0]:pos[0] + sample_size, pos[1]:pos[1] + sample_size])
        sample = np.rot90(sample, k=np.random.randint(0, 4), axes=[0, 1])
        s_min = np.amin(sample)
        s_max = np.amax(sample)
        sample = (sample - s_min) / (s_max - s_min)
        sample = np.expand_dims(sample, axis=-1)
        x.append(sample)
        y.append([s_min / max_elevation, s_max / max_elevation])

        if image:
            rgb = np.zeros(shape=[sample_size, sample_size, 3])
            rgb[:, :, 0] = sample[:, :, 0]
            rgb[:, :, 1] = sample[:, :, 0]
            rgb[:, :, 2] = sample[:, :, 0]
            plt.imsave(os.path.join(target, region + '_' + os.path.basename(file))[:-4] + '_' + str(s) + '.jpg', rgb,
                       cmap='gray')


def process_all(root, target, x, y, image, depth=0):
    if not os.path.exists(target):
        os.mkdir(target)
    print('  ' * depth + '-Processing ' + root)
    for p in os.listdir(root):
        path = os.path.join(root, p)
        if os.path.isdir(path):
            process_all(path, target, x, y, image, depth + 1)
        elif path[-3:] == 'hgt':
            print('  ' * (depth + 1) + '-Sampling ' + p)
            hgt_to_jpg(path, target, x, y, image=image)


raw_dir = 'raw_data/vfp/'
img_dir = 'data/vfp_256/'
data_archive = 'data/vfp_256_labeled.npz'

x = []
y = []
process_all(raw_dir, img_dir, x, y, False)
x = np.asarray(x)
y = np.asarray(y)
print(x.shape, y.shape)

np.savez_compressed(data_archive, x=x, y=y)
