# Downscale HR images with factor x2 and skip 1112.JPG.
python img_process.py  --input_path ../../datasets/train_data/source_img/HR/ --target_path_train ../../datasets/train_data/hr_img --target_path_test ../../datasets/test_data/hr

# Downscale LR images with factor x2 and skip 0925.JPG.
python img_process.py  --input_path ../../datasets/train_data/source_img/LR/ --target_path_train ../../datasets/train_data/lr_img/x2 --target_path_test ../../datasets/test_data/lr/x2

# Convert processed HR images to npy.
python png2npy.py --pathFrom ../../datasets/train_data/hr_img/ --pathTo ../../datasets/train_data/hr_npy/

# Convert processed LR images to npy.
python png2npy.py --pathFrom ../../datasets/train_data/lr_img/x2/ --pathTo ../../datasets/train_data/lr_npy/x2/
