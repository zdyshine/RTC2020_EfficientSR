import os
import cv2
import argparse

# Training settings
parser = argparse.ArgumentParser(description="agora_sr_2020")
parser.add_argument("--input_path", default="../../datasets/train_data/source_img/HR/", type=os.path.abspath,
                    help="path to pretrained models")
parser.add_argument("--target_path_train", default="../../datasets/train_data/hr_img", type=os.path.abspath,
                    help="path to pretrained models")
parser.add_argument("--target_path_test", default="../../datasets/test_data/hr", type=os.path.abspath,
                    help="path to pretrained models")
args = parser.parse_args()

input_path = args.input_path
target_path_train = args.target_path_train
target_path_test = args.target_path_test
print(input_path)

test_list = ['2287.JPG', '2288.png', '2289.JPG', '2290.JPG', '2291.png',
             '2293.JPG', '2294.png', '2295.JPG', '2296.JPG', '2297.png',
             '2298.JPG', '2299.JPG', '2300.JPG', '2301.JPG', '2302.JPG']

if not os.path.exists(target_path_train):
    os.makedirs(target_path_train)
if not os.path.exists(target_path_test):
    os.makedirs(target_path_test)

for root, dirs, files in os.walk(input_path):
    for file in sorted(files):

        if file == '1112.JPG' or file == '0925.JPG' or file == '.DS_Store':
            print('Skipping img: ', file)
            continue

        if file in test_list:
            target_path = target_path_test
        else:
            target_path = target_path_train

        print(os.path.join(root, file))
        bgr = cv2.imread(os.path.join(root, file))
        print('Processing img: ', file, bgr.shape)

        if file[-3:] != 'png':
            bgr = cv2.resize(bgr, (bgr.shape[1] // 2, bgr.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(target_path, file), bgr)
        else:
            os.system('cp %s %s' %(os.path.join(root, file), target_path))