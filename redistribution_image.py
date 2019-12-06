import os
import shutil


def redistribution_train_and_test():
    data_file = os.listdir('D:/datasets/dogs-vs-cats/train')
    dogs_file = list(filter(lambda x: x[:3] == 'dog', data_file))
    cats_file = list(filter(lambda x: x[:3] == 'cat', data_file))

    train_root = 'D:/datasets/dogs-vs-cats/train/'
    test_root = 'D:/datasets/dogs-vs-cats/test/'

    test_data_file = os.listdir('D:/datasets/dogs-vs-cats/test')
    if len(test_data_file) == (len(data_file) + len(test_data_file)) * 0.1:
        return
    data_root = train_root
    for i in range(len(cats_file)):
        print(i)
        image_path = data_root + cats_file[i]
        if i < len(cats_file) * 0.1:
            new_path = test_root + cats_file[i]
            shutil.move(image_path, new_path)
        elif i == len(cats_file):
            break

    for i in range(len(dogs_file)):
        print(i)
        image_path = data_root + dogs_file[i]
        if i < len(dogs_file) * 0.1:
            new_path = test_root + dogs_file[i]
            shutil.move(image_path, new_path)
        elif i == len(dogs_file):
            break


if __name__ == '__main__':
    redistribution_train_and_test()