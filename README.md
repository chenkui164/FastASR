# FastASR
基于PaddleSpeech所使用的conformer模型，使用C++的高效实现模型推理，在树莓派4B等ARM平台运行也可流畅运行。

## 项目简介
本项目仅实现了PaddleSpeech [r1.01版本](https://github.com/PaddlePaddle/PaddleSpeech/releases/tag/r1.0.1)中conformer_wenetspeech-zh-16k预训练模型。
这个预训练模型采用了当下最先进的conformer模型，使用10000+小时的wenetspeech数据集训练得到。
经过测试它识别效果很好,可以媲美许多商用的ASR软件。

PaddleSpeech是基于python实现的，本身的性能已经很不错了，即使在没有GPU的个人电脑上运行，
也能满足实时性的要求（如:时长为10s的语音，推理时间小于10s，即可满足实时性）。


但是要把PaddleSpeech部署在ARM平台，会遇到两个方面的困难。
* 不容易安装，需要自己编译一些组件。
* 执行效率很慢，无法满足实时性的要求。

因此就有这个项目，它由纯C++编写，仅实现了模型的推理过程。

* **语言优势**: 由于C++和Python不同，是编译型语言，编译器会根据编译选项针对不同平台的CPU进行优化，更适合在不同CPU平台上面部署，充分利用CPU的计算资源。
* **独立**: 实现不依赖于现有的深度学习框架如pytorch、paddle、tensorflow等。
* **依赖少**: 项目仅使用了两个第三方库libfftw3和libopenblas，并无其他依赖，所以在各个平台的可移植行很好，通用性很强。
* **效率高**：算法中大量使用指针，减少原有算法中reshape和permute的操作，减少不必要的数据拷贝，从而提升算法性能。

## 快速上手
### 安装依赖

安装依赖库libfftw3
```shell
sudo apt-get install libfftw3-dev libfftw3-single3
```
安装依赖库libopenblas
```shell
sudo apt-get install libopenblas-dev
```
### 编译源码
下载最新版的源码
```shell
git clone https://github.com/chenkui164/FastASR.git
```
编译最新版的源码
```shell
cd FastASR/
```
```shell
make
```

### 下载预训练模型
从PaddleSpeech官网下载预训练模型，如果之前已经在运行过PaddleSpeech，
则可以不用下载，它已经在目录`~/.paddlespeech/models/conformer_wenetspeech-zh-16k`中。
```shell
wget -c https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr1_conformer_wenetspeech_ckpt_0.1.1.model.tar.gz
```

将压缩包解压wenetspeech目录下
```
mkdir wenetspeech
tar -xzvf asr1_conformer_wenetspeech_ckpt_0.1.1.model.tar.gz -C wenetspeech
```
将用于Python的模型转换为C++的，这样更方便通过内存映射的方式直接读取参数，加快模型读取速度。

```shell
./convert.py wenetspeech/exp/conformer/checkpoints/wenetspeech.pdparams
```
查看转换后的参数文件wenet_params.bin的md5码，md5码为9cfcf11ee70cb9423528b1f66a87eafd，表示转换正确。

```
md5sum -b wenet_params.bin
```


同时我也把转换好的wenet_params.bin上传至github，可以直接下载，可能会有些慢。
``` shell
wget -c  https://github.com/chenkui164/FastASR/releases/download/V0.01/wenet_params.bin
```

### 如何使用
下载用于测试的wav文件
```shell
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav 
```

执行程序
```shell
./fastasr zh.wav
```

程序输出
```
Audio time is 4.996812 s.
Model initialization takes 0.163184s
result: "我认为跑步最重要的就是给我带来了身体健康"
Model inference takes 0.462369s.
```

## 树莓派4B上优化部署
由于深度学习推理过程，属于计算密集型算法，所以CPU的指令集对代码的执行效率会有重要影响。
从纯数值计算角度来看，64bit的指令及要比32bit的指令集执行效率要提升1倍。
经过测试同样的算法在64bit系统上，确实是要比32bit系统上，执行效率高很多。

### 为树莓派升级64位系统raspios
到[树莓派官网](https://downloads.raspberrypi.org/)下载最新的raspios 64位系统，
我下载的是没有桌面的精简版[raspios_lite_arm64](https://downloads.raspberrypi.org/raspios_lite_arm64/images/raspios_lite_arm64-2022-04-07/)，
当然也可以下载有桌面的版本[raspios_arm64](https://downloads.raspberrypi.org/raspios_arm64/images/raspios_arm64-2022-04-07/)，
两者没有太大差别，全凭个人喜好。

下载完成镜像，然后烧写SD卡，保证系统新做的系统能正常启动即可。

### 重新编译依赖库

尽管两个依赖库fftw3和openblas都是可以通过`sudo apt install`直接安装的，
但是软件源上的版本是通用版本，是兼容树莓派3B等老版本的型号，
并没有针对树莓派4B的ARM CORTEX A72进行优化，所以执行效率并不高。
因此我们需要针对树莓派4B重新编译，让其发挥最大效率。

**<span style="color:red">
注意：以下编译安装步骤都是在树莓派上完成，不使用交叉编译！！！
</span>**

#### 安装fftw3
下载源码
```shell
wget -c http://www.fftw.org/fftw-3.3.10.tar.gz
```
解压

```shell
tar -xzvf fftw-3.3.10.tar.gz 
cd fftw-3.3.10/
```
配置工程，根据CPU选择适当的编译选项
```shell
./configure --enable-shared --enable-float --prefix=/usr
```
编译和安装
```shell
make -j4
sudo make install
```

#### 安装OpenBLAS

下载源码
```shell
wget -c https://github.com/xianyi/OpenBLAS/releases/download/v0.3.20/OpenBLAS-0.3.20.tar.gz
```

解压
```shell
tar -xzvf OpenBLAS-0.3.20.tar.gz  
cd OpenBLAS-0.3.20
```

编译和安装

```shell
make -j4
sudo make PREFIX=/usr install
```

### 编译和测试
编译和下载预训练模型的过程，请参考上文的<a href="#%E5%BF%AB%E9%80%9F%E4%B8%8A%E6%89%8B"> 快速上手</a>章节。

运行程序
```shell
./fastasr zh.wav
```
结果
```shell
Audio time is 4.996812 s.
Model initialization takes 10.288784s
result: "我认为跑步最重要的就是给我带来了身体健康"
Model inference takes 4.900788s.
```
当第一次运行时，发现模型初始化时间就用了10.2s，
显然不太合理，这是因为预训练模型是在SD卡中，一个450M大小的文件从SD卡读到内存中，主要受限于SD卡的读取速度，所以比较慢。
得利于linux的缓存机制，第二次运行时，模型已经在内存中，不用在从SD卡读取了，所以只有重启后第一次会比较慢。

第二次运行结果
```shell
Audio time is 4.996812 s.
Model initialization takes 0.797091s
result: "我认为跑步最重要的就是给我带来了身体健康"
Model inference takes 4.916471s.
```

从结果中可以看出，当音频文件为4.99s时，推理时间为4.91秒，推理时间小于音频时间，刚刚好能满足实时性的需求。
