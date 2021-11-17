import torch
import torchvision

from PIL import Image
from pathlib import Path

PROJECT_ROOT = Path("/home/goda/face_beauty_prediction/")


class SCUT_FBP5500(torch.utils.data.Dataset):
    def __init__(self, is_train=True, is_extension=False, transform=None, directory: str = None):
        self.is_train = is_train
        self.transform = transform
        self.is_extension = is_extension

        if directory is None:
            self.directory = "data/SCUT-FBP5500_v2/train_test_files/split_of_60%training and 40%testing"

        path = PROJECT_ROOT / self.directory
        if self.is_train:
            if self.is_extension:
                path = path / "train_extension.txt"
            else:
                path = path / "train.txt"
        else:
            path = path / "test.txt"

        self.imgs, self.labels = self.read_img(path)

    def read_img(self, list_file_path, source_image_directory=None):
        file = open(list_file_path, 'r')
        image_label_list = file.readlines()
        file.close()

        imgs = []
        labels = []

        if source_image_directory is None:
            source_image_directory = PROJECT_ROOT / "data" / "SCUT-FBP5500_v2" / "Images"

        for image_label in image_label_list:
            image_file, label = image_label.split()

            img = Image.open(source_image_directory / image_file).convert('RGB')
            imgs.append(img), labels.append(label)

        return imgs, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.imgs[index]
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)

        label = torch.Tensor([float(label)])

        return image, label


class SCUT_FBP5500_MAN(torch.utils.data.Dataset):
    def __init__(self, is_train=True, transform=None, directory: str = None):
        self.is_train = is_train
        self.transform = transform

        if directory is None:
            self.directory = "data/SCUT-FBP5500_v2/train_test_files/split_of_60%training and 40%testing"

        path = PROJECT_ROOT / self.directory
        if self.is_train:
            path = path / "train_man_extension.txt"
        else:
            path = path / "test_man.txt"

        self.imgs, self.labels = self.read_img(path)

    def read_img(self, list_file_path, source_image_directory=None):
        file = open(list_file_path, 'r')
        image_label_list = file.readlines()
        file.close()

        imgs = []
        labels = []

        if source_image_directory is None:
            source_image_directory = PROJECT_ROOT / "data" / "SCUT-FBP5500_v2" / "Images"

        for image_label in image_label_list:
            image_file, label = image_label.split()

            img = Image.open(source_image_directory / image_file).convert('RGB')
            imgs.append(img), labels.append(label)

        return imgs, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.imgs[index]
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)

        label = torch.Tensor([float(label)])

        return image, label


class SCUT_FBP5500_ASIAN_MAN(torch.utils.data.Dataset):
    def __init__(self, is_train=True, transform=None, directory: str = None):
        self.is_train = is_train
        self.transform = transform

        if directory is None:
            self.directory = "data/SCUT-FBP5500_v2/train_test_files/split_of_60%training and 40%testing"

        path = PROJECT_ROOT / self.directory
        if self.is_train:
            path = path / "train_asian_man_extension.txt"
        else:
            path = path / "test_asian_man.txt"

        self.imgs, self.labels = self.read_img(path)

    def read_img(self, list_file_path, source_image_directory=None):
        file = open(list_file_path, 'r')
        image_label_list = file.readlines()
        file.close()

        imgs = []
        labels = []

        if source_image_directory is None:
            source_image_directory = PROJECT_ROOT / "data" / "SCUT-FBP5500_v2" / "Images"

        for image_label in image_label_list:
            image_file, label = image_label.split()

            img = Image.open(source_image_directory / image_file).convert('RGB')
            imgs.append(img), labels.append(label)

        return imgs, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.imgs[index]
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)

        label = torch.Tensor([float(label)])

        return image, label


if __name__ == "__main__":
    print("hello")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])

    train_dataset = SCUT_FBP5500(
        is_train=True,
        for_man=True,
        is_extension=True,
        transform=transform
    )
    print(len(train_dataset))
    for i in range(len(train_dataset)):
        print(i, train_dataset[i][1])
