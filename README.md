# 项目说明

fork自[HyperPR](https://github.com/zeusees/HyperLPR)，用于理解项目，做注释、修改、实验。感谢原作者的无私分享！

## 对HyperPR的理解

他提供了2种方法做检测，一种是传统的Haar+Casscade，一种是MobileSSD（[https://github.com/zeusees/Mobilenet-SSD-License-Plate-Detection](https://github.com/zeusees/Mobilenet-SSD-License-Plate-Detection)）

识别貌似是一个CNN固定长度的识别网络？？？（待确认）

检测的代码，他说他没有提供，但是有网友给提供了，但是需要去细看一下？？[https://github.com/armaab/hyperlpr-train](https://github.com/armaab/hyperlpr-train)

官方的那个train是训练车牌识别的，而不是检测的：[https://github.com/zeusees/HyperLPR-Training](https://github.com/zeusees/HyperLPR-Training)，这个是ctc字符训练的

## 参考贴

- [一个fork者的代码理解分享](https://blog.csdn.net/qq_37423198/article/details/81266401)
- [HyperLPR车牌识别技术算法之车牌精定位，原作者的，很有参考价值](https://blog.csdn.net/Relocy/article/details/78705566)
- [HyperLPR车牌识别技术算法之车牌粗定位与训练,原作者的](https://blog.csdn.net/Relocy/article/details/78705662)
- [设计一款基于GPU的车牌识别系统,原作者2的思路，比较简短](https://blog.csdn.net/lsy17096535/article/details/102896368)
- [基于mobilenet-ssd训练车牌识别模型,原作者2的思路，比较简短](https://blog.csdn.net/lsy17096535/article/details/78687728)
- [另外一个人的代码解读，参考有限](https://blog.csdn.net/kwame211/article/details/81281923)

## 思路

原作者给出的pipeline思路：

```python
step1. 使用opencv 的 HAAR Cascade 检测车牌大致位置

step2. Extend 检测到的大致位置的矩形区域

step3. 使用类似于MSER的方式的 多级二值化 + RANSAC 拟合车牌的上下边界

step4. 使用CNN Regression回归车牌左右边界

step5. 使用基于纹理场的算法进行车牌校正倾斜

step6. 使用CNN滑动窗切割字符

step7. 使用CNN识别字符
```

## Issue中的内容总结

<https://github.com/zeusees/HyperLPR/issues>

- “我们暂时没有提供开源代码中对应的训练代码，readme中的两个训练代码是群里爱好者提供的，效果也很不错，你可以尝试一下。”2019.8
- 模型用的是ssd，[开源了](https://github.com/zeusees/Mobilenet-SSD-License-Plate-Detection)，说明在[csdn](https://blog.csdn.net/lsy17096535/article/details/78687728)，不过好像是一个另外版本
- 请问一下本项目训练集的样本数量量级？30万，真实样本，我猜是车牌，不是车辆图片
- “端到端的SegmenationFree-Inception模型的训练代码已经提供”，这个是啥？？？
- cascade模型大概用了多少的训练样本？cascade大概用了正样本8000张，负样本15000张.
- 正样本要充分，负样本也要多样，才能降低错误识别。
- 训练用的样本正的，倾斜的都要有，提高新能源准确率需要增加这部分样本。
- 我想问下作者的样本量多大才能有比较好的效果（对全国车牌来说），我们大概使用了5k张在不同环境下的车牌样本训练的gan来生成20w张生成样本。
- MobileNet-300-0.25-ssd 在E5-2650服务器上cpu运行，看一下你的图片分辨率，另外，应该你的opencv 没有开启AVX 加速编译，不然不会这么慢。输入图像分辨率512*512以下能达到7ms。
- 在HyperLPR中使用cascade 而没有使用 SSD，原因是，SSD定位速度慢 适合有GPU的用户，SSD出来的结果在我自己的数据集上　要比 cascade 得到的矩形框 更精准一些；SSD出来的结果 已经带有bounding box区域的feature map，可以 在后续的CNN+RNN recognition部分直接省略CNN compute feature的部分吗？这样一来　是不是可以节约计算量？可以的 端到端的 text spotting 的做法就是这样的。 但是出于 cpu 边缘计算设备考虑 还是选择使用cascade based detector 要 速度快一点。
- 车牌粗定位的具体过程是怎样的，可以采用ssd.mtcnn,yolo,或者传统方式的adaboost cascade等。
- hyperlpr-train is segmentation based code. if you want to train cnn e2e model, you can try [https://github.com/armaab/hyperlpr-train](https://github.com/armaab/hyperlpr-train)
- 采用HyperLPR和easypr 相同的测试集（easypr）提供的general_test 文件夹里面的图片测试，发现HyperLPR字符分割和识别率较低，而easyppr 较高似乎与readme说明的不一致，不知道是不是有什么细节参数之类的需要调整呢？答：python 的效果是最新的模型 应该是最好的，配置python采用端到端模型再跑一下，
- “识别率高,仅仅针对车牌ROI在EasyPR数据集上，0-error达到 95.2%, 1-error识别率达到 97.4% (指在定位成功后的车牌识别率)”
- 车牌检测使用的是浅层模型还是基于cascade的cnn模型？是传统的adaboost cascade模型

ssd效果更好一些，但是适合有gpu的场景。


# 原有内容（保留）


![logo_t](./demo_images/logo.png)

## HyperLPR   高性能开源中文车牌识别框架

#### [![1](https://badge.fury.io/py/hyperlpr.svg "title")](https://pypi.org/project/hyperlpr/)[![1](https://img.shields.io/pypi/pyversions/hyperlpr.svg "title")](https://pypi.org/project/hyperlpr/)

### 一键安装

`python -m pip install hyperlpr`

###### 支持python3,支持Windows  Mac Linux 树莓派等。

###### 720p cpu real-time (st on MBP r15 2.2GHz haswell).

#### 快速上手

```python
#导入包
from hyperlpr import *
#导入OpenCV库
import cv2
#读入图片
image = cv2.imread("demo.jpg")
#识别结果
print(HyperLPR_plate_recognition(image))
```

#### Q&A

Q：Android识别率没有所传demo apk的识别率高？

A：请使用[Prj-Linux](https://github.com/zeusees/HyperLPR/tree/master/Prj-Linux/lpr/model)下的模型，android默认包里的配置是相对较早的模型

Q：车牌的训练数据来源？

A：由于用于训练车牌数据涉及到法律隐私等问题，本项目无法提供。开放较为大的数据集有[CCPD](https://github.com/detectRecog/CCPD)车牌数据集。

Q：训练代码的提供？

A：相关资源中有提供训练代码

Q：关于项目的来源？

A：此项目来源于作者早期的研究和调试代码，代码缺少一定的规范，同时也欢迎PR。


#### 相关资源

- [Android配置教程](https://www.jianshu.com/p/94784c3bf2c1)
- [python配置教程](https://www.jianshu.com/p/7ab673abeaae)
- [Linux下C++配置教程](https://blog.csdn.net/lu_linux/article/details/88707421)
- [带UI界面的工程](https://pan.baidu.com/s/1cNWpK6)(感谢群内小伙伴的工作)。
- [端到端(多标签分类)训练代码](https://github.com/LCorleone/hyperlpr-train_e2e)(感谢群内小伙伴的工作)。
- [端到端(CTC)训练代码](https://github.com/armaab/hyperlpr-train)(感谢群内小伙伴工作)。

### 更新

- 更新了Android实现，增加实时扫描接口 (2019.07.24)
- 更新Windows版本的Visual Studio 2015 工程至端到端模型（2019.07.03）
- 更新基于端到端的IOS车牌识别工程。(2018.11.13)
- 可通过pip一键安装、更新的新的识别模型、倾斜车牌校正算法、定位算法。(2018.08.11)
- 提交新的端到端识别模型，进一步提高识别准确率(2018.08.03)
- [增加PHP车牌识别工程@coleflowers](https://github.com/zeusees/HyperLPR/tree/master/Prj-PHP) (2018.06.20)
- 添加了HyperLPR Lite 仅仅需160 行代码即可实现车牌识别(2018.3.12)
- 感谢 sundyCoder [Android 字符分割版本](https://github.com/sundyCoder/hyperlpr4Android) 
- 增加字符分割[训练代码和字符分割介绍](https://github.com/zeusees/HyperLPR-Training)(2018.1.)


### TODO

- 支持多种车牌以及双层
- 支持大角度车牌
- 轻量级识别模型

### 特性

- 速度快 720p,单核 Intel 2.2G CPU (MaBook Pro 2015)平均识别时间低于100ms
- 基于端到端的车牌识别无需进行字符分割
- 识别率高,卡口场景准确率在95%-97%左右
- 轻量,总代码量不超1k行

### 模型资源说明

- cascade.xml  检测模型 - 目前效果最好的cascade检测模型
- cascade_lbp.xml  召回率效果较好，但其错检太多
- char_chi_sim.h5 Keras模型-可识别34类数字和大写英文字  使用14W样本训练 
- char_rec.h5 Keras模型-可识别34类数字和大写英文字  使用7W样本训练 
- ocr_plate_all_w_rnn_2.h5 基于CNN的序列模型
- ocr_plate_all_gru.h5 基于GRU的序列模型从OCR模型修改，效果目前最好但速度较慢，需要20ms。
- plate_type.h5 用于车牌颜色判断的模型
- model12.h5 左右边界回归模型

### 注意事项:

- Win工程中若需要使用静态库，需单独编译
- 本项目的C++实现和Python实现无任何关联，都为单独实现
- 在编译C++工程的时候必须要使用OpenCV 3.3以上版本 (DNN 库)，否则无法编译 
- 安卓工程编译ndk尽量采用14b版本

### Python 依赖

- Keras (>2.0.0)
- Theano(>0.9) or Tensorflow(>1.1.x)
- Numpy (>1.10)
- Scipy (0.19.1)
- OpenCV(>3.0)
- Scikit-image (0.13.0)
- PIL

### CPP 依赖

- Opencv 3.4 以上版本

### Linux/Mac 编译

- 仅需要的依赖OpenCV 3.4 (需要DNN框架)

```bash
cd Prj-Linux
mkdir build 
cd build
cmake ../
sudo make -j 
```

### CPP demo

```cpp
#include "../include/Pipeline.h"
int main(){
    pr::PipelinePR prc("model/cascade.xml",
                      "model/HorizonalFinemapping.prototxt","model/HorizonalFinemapping.caffemodel",
                      "model/Segmentation.prototxt","model/Segmentation.caffemodel",
                      "model/CharacterRecognization.prototxt","model/CharacterRecognization.caffemodel",
                       "model/SegmentationFree.prototxt","model/SegmentationFree.caffemodel"
                    );
  //定义模型文件

    cv::Mat image = cv::imread("test.png");
    std::vector<pr::PlateInfo> res = prc.RunPiplineAsImage(image,pr::SEGMENTATION_FREE_METHOD);
  //使用端到端模型模型进行识别 识别结果将会保存在res里面
 
    for(auto st:res) {
        if(st.confidence>0.75) {
            std::cout << st.getPlateName() << " " << st.confidence << std::endl;
          //输出识别结果 、识别置信度
            cv::Rect region = st.getPlateRect();
          //获取车牌位置
 cv::rectangle(image,cv::Point(region.x,region.y),cv::Point(region.x+region.width,region.y+region.height),cv::Scalar(255,255,0),2);
          //画出车牌位置
          
        }
    }

    cv::imshow("image",image);
    cv::waitKey(0);
    return 0 ;
}
```

###  

### 可识别和待支持的车牌的类型

- [x] 单行蓝牌
- [x] 单行黄牌
- [x] 新能源车牌
- [x] 白色警用车牌
- [x] 使馆/港澳车牌
- [x] 教练车牌
- [ ] 武警车牌
- [ ] 民航车牌
- [x] 双层黄牌
- [ ] 双层武警
- [ ] 双层军牌
- [ ] 双层农用车牌
- [ ] 双层个性化车牌

###### Note:由于训练的时候样本存在一些不均衡的问题,一些特殊车牌存在一定识别率低下的问题，如(使馆/港澳车牌)，会在后续的版本进行改进。

### 测试样例

![image](./demo_images/demo1.png)

![image](./demo_images/demo2.jpg)

#### Android示例

![android](./demo_images/android.png)

### 识别测试APP

- 体验 Android APP：[https://fir.im/HyperLPR](https://fir.im/HyperLPR) (根据图片尺寸调整程序中的尺度，提高准确率)

#### 获取帮助

- HyperLPR讨论QQ群1: 673071218(已满，邀请可进), 群2: 746123554 ,加前请备注HyperLPR交流。

### 作者和贡献者信息：

##### 作者昵称不分前后

- Jack Yu 作者(jack-yu-business@foxmail.com / https://github.com/szad670401)
- AlanNewImage v2版win工程、python双层完善 (https://github.com/AlanNewImage)
- lsy17096535 整理(https://github.com/lsy17096535)
- xiaojun123456 IOS贡献(https://github.com/xiaojun123456)
- sundyCoder Android第三方贡献(https://github.com/sundyCoder)
- coleflowers php贡献(@coleflowers)
- Free&Easy 资源贡献 
- 海豚嘎嘎 LBP cascade检测器训练
- Windows工程端到端模型 (https://github.com/SalamanderEyes)
- Android实时扫描实现 (https://github.com/lxhAndSmh)
