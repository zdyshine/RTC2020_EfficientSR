## 1. 训练环境

* torch==1.1.0  
* scikit-image==0.16.2  
* opencv-python==3.4.3.18  

## 2. 训练、测试数据生成

* 将官方提供的训练数据放到指定文件夹:  

&emsp;&emsp;1941张HR图像存放路径:  

&emsp;&emsp;&emsp;&emsp;&emsp;../datasets/train_data/source_img/HR  

&emsp;&emsp;1941张LR图像存放路径:  

&emsp;&emsp;&emsp;&emsp;&emsp;../datasets/train_data/source_img/LR  

* 生成npy格式的训练数据:  
 
```bash
cd data  
chmod a+x gen_data.sh  
./gen_data.sh
```

## 3. 训练模型
 
等待数据全部生成后，开始训练:  

```bash
chmod a+x train.sh  
./train.sh
```

## 4. 测试模型

```bash
chmod a+x test.sh  
./test.sh
```
## 请严格按照操作步骤执行代码