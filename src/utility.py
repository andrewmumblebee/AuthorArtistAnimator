# ANY UTILITY HELPERS, I.E Reading in files/Generating images.
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from PIL import Image, ImageFilter

def get_image_data(images, path):
    data = []
    for file_path in images:
        im = Image.open(os.path.join(path, file_path))
        # x_dim = np.reshape(im, [-1, 64, 4])
        # copies = np.reshape(x_dim, [-1, 64, 64, 4])
        data.append(np.reshape(im.getdata(), [64, 64, 4]))
    return np.array(data)

def calcImageSize(dh, dw, stride):
    return int(np.ceil(float(dh)/float(stride))), int(np.ceil(float(dw)/float(stride)))

def tileImage(imgs, size):
    d = int(np.sqrt(imgs.shape[0]-1))+1
    h, w = imgs.shape[1], imgs.shape[2]
    if (imgs.shape[3] in (3,4,1)):
        colour = imgs.shape[3]
        img = np.zeros((h * d, w * d, colour))
        for idx, image in enumerate(imgs):
            i = idx // d
            j = idx-i*d
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return ((img * 255.) + 1) * 2
    else:
        raise ValueError('in merge(images,size) images parameter '
                        'must have dimensions: HxW or HxWx3 or HxWx4')

class BatchGenerator:
    """ Class for handling retrieval of images from the dataset.

        Randomly samples from the dataset, until an epoch is completed then resets.
    """
    def __init__(self, dataset_folder):
        """ Retrieve sprite images and label them based on their filename. """
        self.path, self.label = self.find_images(dataset_folder)
        self.reset_buffer()
        self.dataset_folder = dataset_folder

    def get_batch(self, batch_size, color=True):
        b_idx = np.random.randint(0, self.buffer.shape[0] - 1, batch_size) # Random index
        idx = self.buffer[b_idx]
        x = get_image_data(self.path[idx], self.dataset_folder) # Image and Respective Label
        x = (x / 255) # Normalize Channel values to 0-1 range.
        x = (x * 2) - 1 # Further Normalize to -1 to 1 range.
        t = self.label[idx]
        self.buffer = np.delete(self.buffer, b_idx)
        return x, t

    def get_file_count(self):
        return self.path.shape[0]

    def get_label_size(self):
        return self.label.shape[1]

    def reset_buffer(self):
        self.buffer = np.arange(self.path.shape[0])

    def encode_labels(self, paths):
        labels = []
        for path in paths:
            img_label = []
            ids = os.path.splitext(path)[0].split("_")
            for label in ids:
                if ('ea' in label or 'cl' in label):
                    img_label.append(label[2:])
                elif label.isdigit():
                    pass
                else:
                    img_label.append(label[1:])
            labels.append(img_label)
        enc = OneHotEncoder()
        labels = np.array(labels, dtype=np.float32)
        enc.fit(labels)
        labels = enc.transform(labels).toarray()
        return labels

    def find_images(self, path):
        paths = []
        for file in os.listdir(path):
            if not path.endswith('b.png'):
                paths.append(file)
        labels = self.encode_labels(paths)
        # labels = labels / max_label_values
        # labels = (labels * 2) - 1 # Normalizing labels to -1 - 1 range. This assumes a minimum of 0 in original data.
        return np.array(paths), labels


class ABatchGenerator(BatchGenerator):
    """ Class for handling retrieval of images from the dataset.

        Randomly samples from the dataset, until an epoch is completed then resets.
    """
    def __init__(self, dataset_folder):
        """ Retrieve sprite images and label them based on their filename. """
        self.path, self.label, self.base = self.find_images(dataset_folder)
        self.curr_frame = 0
        self.b_idx = 0
        self.reset_buffer()
        self.dataset_folder = dataset_folder

    # Need a second index in the buffer, corresponding to the frame to retrieve.

    def get_batch(self, batch_size, color=True):
        b_idx = np.random.randint(0, self.buffer.shape[0] - 1, batch_size) # Random index
        idx = self.buffer[b_idx]
        path = self.path[idx]
        x = get_image_data(path, self.dataset_folder) # Image and Respective Label
        x = (x / 255) # Normalize Channel values to 0-1 range.
        x = (x * 2) - 1 # Further Normalize to -1 to 1 range.
        t = self.label[idx]
        b = get_image_data(self.base[idx], self.dataset_folder)
        self.buffer = np.delete(self.buffer, b_idx)
        return x, t, b

    def find_images(self, path):
        paths = []
        for file in os.listdir(path):
            if not file.endswith('b.png'):
                paths.append(file)
        paths = np.array(paths)
        labels, bases = self.encode_labels(paths)
        # labels = labels / max_label_values
        # labels = (labels * 2) - 1 # Normalizing labels to -1 - 1 range. This assumes a minimum of 0 in original data.
        return paths, labels, bases

    def encode_labels(self, paths):
        frames = []
        animations = []
        bases = []
        for path in paths:
            img_label = []
            ids = os.path.splitext(path)[0].split("_")
            animations.append([ids[0][1:]])
            frames.append([ids[1][1:]])
            bases.append(ids[0] + '_' + ids[2] + 'b.png')
        enc = OneHotEncoder()
        #animations = np.array(animations, dtype=np.float32)
        enc.fit(animations)
        animations = enc.transform(animations).toarray()
        labels = np.concatenate((animations, frames), axis=1)
        labels = np.array(labels, dtype=np.float32)
        bases = np.array(bases)
        return labels, bases


    # def get_batch(self, batch_size, color=True):
    #     batch_labels = np.array([])
    #     batch_images = np.array([])
    #     while batch_images.shape[0] < batch_size:

    #         curr_image = get_image_data([self.path[self.b_idx]], self.dataset_folder)
    #         print(curr_image.shape)

    #         if curr_image.shape[0] < self.curr_frame:
    #             batch_images.append(curr_image[self.curr_frame])
    #             batch_labels.append(curr_image[self.curr_frame])
    #             self.curr_frame += 1
    #         else:
    #             self.curr_frame = 0
    #             self.buffer = np.delete(self.buffer, self.b_idx)
    #             self.b_idx = self.buffer[np.random.randint(0, self.buffer.shape[0] - 1, 0)]

    #     idx = self.buffer[b_idx]
    #     x = get_image_data(self.path[idx], self.dataset_folder) # Image and Respective Label
    #     print(x.shape)
    #     x = (x / 255) # Normalize Channel values to 0-1 range.
    #     x = (x * 2) - 1 # Further Normalize to -1 to 1 range.
    #     t = self.label[idx]
    #     t[1] = self.curr_frame
    #     self.buffer = np.delete(self.buffer, b_idx)
    #     return batch_labels, batch_images


def main():
    img = get_image_data(['Female Dark 2 Blue Big Nose Big Ears.png'], '{}/../CharacterScraper/dump/sprites'.format(os.path.dirname(__file__)))
    paths, labels = find_images((os.path.join(os.path.dirname(__file__), '../CharacterScraper/dump/sprites')))

if __name__ == '__main__':
    main()