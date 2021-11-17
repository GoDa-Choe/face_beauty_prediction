import os

from project_root import PROJECT_ROOT


def rename(path):
    path = PROJECT_ROOT / path
    for index, file, in enumerate(os.listdir(path), start=1):
        new = f'AM_5_{index}.jpg'
        os.rename(path / file, path / new)
        print(file, "->", new)


def make_train_list(path, base_file, outfile):
    path = PROJECT_ROOT / path

    file_list = os.listdir(path)
    file_list.sort(key=lambda x: int(x[5:-4]))

    for line in base_file:
        if line.startswith("AM") or line.startswith("CM"):
            outfile.write(line)

    for index, file, in enumerate(file_list, start=1):
        print(index, file, 5.0)
        outfile.write(f"{file} 5.0\n")


def make_test_list(base_file, outfile):
    for line in base_file:
        if line.startswith("AM") or line.startswith("CM"):
            outfile.write(line)


if __name__ == "__main__":
    rename("data/SCUT-FBP5500_v2/5_face")

    base_file = open(
        PROJECT_ROOT / 'data/SCUT-FBP5500_v2/train_test_files/split_of_60%training and 40%testing/train.txt', 'r')
    outfile = open(
        PROJECT_ROOT / 'data/SCUT-FBP5500_v2/train_test_files/split_of_60%training and 40%testing/train_man_extension.txt',
        'w')

    make_train_list("data/SCUT-FBP5500_v2/5_face", base_file, outfile)

    base_file = open(
        PROJECT_ROOT / 'data/SCUT-FBP5500_v2/train_test_files/split_of_60%training and 40%testing/test.txt', 'r')
    outfile = open(
        PROJECT_ROOT / 'data/SCUT-FBP5500_v2/train_test_files/split_of_60%training and 40%testing/test_man.txt',
        'w')
    make_test_list(base_file, outfile)
