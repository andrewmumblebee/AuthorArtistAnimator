# ANY UTILITY HELPERS, I.E Reading in files/Generating images.
import os
import numpy as np
from PIL import Image, ImageFilter

def find_images(path):
    paths = np.array(os.listdir(path))
    labels = []
    for path in paths:
        img_label = []
        ids = os.path.splitext(path)[0].split("_")
        for label in ids:
            if ('cl' in label or 'ea' in label):
                img_label.append(label[2:])
            else:
                img_label.append(label[1:])
        labels.append(img_label)
    labels = np.array(labels)
    return paths, labels

def get_image_data(images, path):
    data = []
    for file_path in images:
        im = Image.open(os.path.join(path, file_path))
        data.append(np.reshape(im.getdata(), [64, 64, 4]))
    return np.array(data)

class BatchGenerator:
    def __init__(self, dataset_folder):
        """ Retrieve sprite images and label them based on their filename. """
        self.image, self.label = find_images(dataset_folder)
        self.dataset_folder = dataset_folder

    def getBatch(self, batch_size, color=True):
        idx = np.random.randint(0, len(self.image) - 1, batch_size) # Random index
        x = get_image_data(self.image[idx], self.dataset_folder) # Image and Respective Label
        x = x #/ 255) * 0.9 # Normalize Channel values to 0-1 smoothing of real labels
        t = self.label[idx]
        return x, t

def main():
    img = get_image_data(['Female Dark 2 Blue Big Nose Big Ears.png'], '{}/../CharacterScraper/dump/sprites'.format(os.path.dirname(__file__)))
    paths, labels = find_images((os.path.join(os.path.dirname(__file__), '../CharacterScraper/dump/sprites')))

if __name__ == '__main__':
    main()