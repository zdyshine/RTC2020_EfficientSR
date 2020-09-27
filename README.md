## RTC2020_EfficientSR FZU-CS510 冠军开源方案
比赛链接: https://www.dcjingsai.com/v2/cmptDetail.html?id=409  
比赛团队(Team)：FZU-CS510  
比赛名次及得分：![image](https://github.com/zdyshine/RTC2020_EfficientSR/blob/master/score.png)  
感谢给力的楠哥。

## 方案分享
1. 数据处理：png图片的不做处理，jpg的GT和LR全部是使用cv2默认的插值，做2倍降采样  
2. 网络设计：![image](https://github.com/zdyshine/RTC2020_EfficientSR/blob/master/net.jpg)  
3. 最终使用MSE 作为loss训练，追求极致的PSNR指标  
 
## 代码说明
文件夹的功能如下：  
--pretrained-model	存放训练好的模型    
--datasets	存放训练数据集和测试集
--results		存在测试的结果图片   
--training	存在训练过程中保存的log信息和模型  
--codes	存放训练和测试的所有代码   
 * data 存放数据预处理和加载数据的相关代码   
 * model 存放模型定义文件  
 * loss 存放损失函数的定义文件    
 * utils 存放工具类的相关文件   
 * train.py 训练模型的文件   
 * test.py 测试模型的文件    
 * README.md 模型训练和测试的使用说明和环境配置说明<br>  
 
 ## 引用
 1. IMDN: https://github.com/Zheng222/IMDN  
 2. AIM2020: https://github.com/cszn/KAIR/blob/master/main_challenge_sr.py  
 3. SR Paper List: https://github.com/ChaofWang/Awesome-Super-Resolution  
 4. GCT: https://github.com/z-x-yang/GCT  
 5. GhostNet: https://github.com/huawei-noah/ghostnet
