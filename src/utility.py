# ANY UTILITY HELPERS, I.E Reading in files/Generating images.
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from PIL import Image, ImageFilter

def find_images(path):
    paths = np.array(os.listdir(path))
    labels = []
    for path in paths:
        img_label = []
        ids = os.path.splitext(path)[0].split("_")
        for label in ids:
            if ('ea' in label or 'cl' in label):
                img_label.append(label[2:])
            else:
                img_label.append(label[1:])
        labels.append(img_label)
    enc = OneHotEncoder()
    labels = np.array(labels, dtype=np.float32)
    enc.fit(labels)
    labels = enc.transform(labels).toarray()
    # labels = labels / max_label_values
    # labels = (labels * 2) - 1 # Normalizing labels to -1 - 1 range. This assumes a minimum of 0 in original data.
    return paths, labels

def get_image_data(images, path):
    data = []
    for file_path in images:
        im = Image.open(os.path.join(path, file_path))
        data.append(np.reshape(im.getdata(), [64, 64, 4]))
    return np.array(data)

def calcImageSize(dh, dw, stride):
    return int(np.ceil(float(dh)/float(stride))), int(np.ceil(float(dw)/float(stride)))

class BatchGenerator:
    """ Class for handling retrieval of images from the dataset.

        Randomly samples from the dataset, until an epoch is completed then resets.
    """
    def __init__(self, dataset_folder):
        """ Retrieve sprite images and label them based on their filename. """
        self.path, self.label = find_images(dataset_folder)
        self.reset_batch()
        self.dataset_folder = dataset_folder

    def getBatch(self, batch_size, color=True):
        idx = np.random.randint(0, len(self.batch_buffer) - 1, batch_size) # Random index
        x = get_image_data(self.batch_buffer[idx], self.dataset_folder) # Image and Respective Label
        x = (x / 255) # Normalize Channel values to 0-1 range.
        x = (x * 2) - 1 # Further Normalize to -1 to 1 range.
        t = self.label[idx]
        self.batch_buffer = np.delete(self.batch_buffer, idx)
        return x, t

    def get_label_size(self):
        return self.label.shape[1]

    def reset_batch(self):
        self.batch_buffer = self.path.clone()


def main():
    img = get_image_data(['Female Dark 2 Blue Big Nose Big Ears.png'], '{}/../CharacterScraper/dump/sprites'.format(os.path.dirname(__file__)))
    paths, labels = find_images((os.path.join(os.path.dirname(__file__), '../CharacterScraper/dump/sprites')))

if __name__ == '__main__':
    main()