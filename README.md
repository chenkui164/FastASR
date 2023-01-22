# FastASR

这是一个用C++实现ASR推理的项目，它依赖很少，安装也很简单，推理速度很快，在树莓派4B等ARM平台也可以流畅的运行。
推理模型是基于目前最先进的conformer模型，使用10000+小时的wenetspeech数据集训练得到， 所以识别效果也很好，可以媲美许多商用的ASR软件。

## 项目简介

目前本项目实现了3个模型，它们是PaddleSpeech [r1.01版本](https://github.com/PaddlePaddle/PaddleSpeech/releases/tag/r1.0.1)中conformer_wenetspeech-zh-16k和conformer_online_wenetspeech-zh-16k
，以及[kaidi2](https://github.com/k2-fsa/icefall/tree/master/egs/wenetspeech/ASR)的rnnt2。

* **非流式模型**：每次识别是以句子为单位，所以实时性会差一些，但准确率会高一些。
* **流式模型**：模型的输入是语音流，并实时返回语音识别的结果，但是准确率会下降些。  

k2_rnnt2和conformer_wenetspeech-zh-16k是属于非流式模型，
conformer_online_wenetspeech-zh-16k属于流式模型。

目前通过使用VAD技术, 非流式模型支持大段的长语音识别。


上面提到的这些模型都是基于深度学习框架（paddlepaddle或pytorch）实现的, 本身的性能已经很不错了，即使在没有GPU的个人电脑上运行，
也能满足实时性的要求（如:时长为10s的语音，推理时间小于10s，即可满足实时性）。


但是要把深度学习模型部署在ARM平台，会遇到两个方面的困难。
* 不容易安装，需要自己编译一些组件。
* 执行效率很慢，无法满足实时性的要求。

因此就有这个项目，它由纯C++编写，仅实现了模型的推理过程。

* **语言优势**: 由于C++和Python不同，是编译型语言，编译器会根据编译选项针对不同平台的CPU进行优化，更适合在不同CPU平台上面部署，充分利用CPU的计算资源。
* **独立**: 实现不依赖于现有的深度学习框架如pytorch、paddle、tensorflow等。
* **依赖少**: 项目仅使用了两个第三方库libfftw3和libopenblas，并无其他依赖，所以在各个平台的可移植行很好，通用性很强。
* **效率高**：算法中大量使用指针，减少原有算法中reshape和permute的操作，减少不必要的数据拷贝，从而提升算法性能。


针对C++用户和python用户，本项目分别生成了静态库libfastasr.a和PyFastASR.XXX模块，调用方法可以参考example目录中的例子。


### 未完成工作
* 量化和压缩模型

## python安装

目前fastasr在个平台的支持情况如下表, 其他未支持的平台可通过源码编译获得对应的whl包。

|   | macOS Intel | Windows 64bit | Windows 32bit | Linux x86 | Linux x64 | Linux aarch64 |
|---------------|----|-----|-----|----|-----|----|
| CPython 3.6   | ✅ | ✅  | ✅  | ✅ | ✅  | ✅ |
| CPython 3.7   | ✅ | ✅  | ✅  | ✅ | ✅  | ✅ |
| CPython 3.8   | ✅ | ✅  | ✅  | ✅ | ✅  | ✅ |
| CPython 3.9   | ✅ | ✅  | ✅  | ✅ | ✅  | ✅ |
| CPython 3.10  | ✅ | ✅  | ✅  | ✅ | ✅  | ✅ |
| CPython 3.11  | ✅ | ✅  | ✅  | ✅ | ✅  | ✅ |

可通过pip直接安装
```
pip install fastasr
```


## 源码编译安装指南
### Ubuntu 安装依赖

安装依赖库libfftw3
```shell
sudo apt-get install libfftw3-dev libfftw3-single3
```
安装依赖库libopenblas
```shell
sudo apt-get install libopenblas-dev
```
安装python环境
```shell
sudo apt-get install python3 python3-dev
```

### MacOS 安装依赖

安装依赖库fftw
```shell
sudo brew install fftw
```
安装依赖库openblas
```shell
sudo brew install openblas
```
### 编译源码

#### Build for Linux
下载最新版的源码
```shell
git clone https://github.com/chenkui164/FastASR.git
```
编译最新版的源码，
```shell
cd FastASR/
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```
编译python的whl安装包

```shell
cd FastASR/
python -m build
```

####  Build for Windows

[Windows编译指南](win/readme.md)

使用VisualStudio 2022打开CMakeLists.txt，选择Release编译。
需要在vs2022安装linux开发组件。

### 下载预训练模型

#### paraformer预训练模型下载

进入FastASR/models/paraformer_cli文件夹，用于存放下载的预训练模型.
```shell
cd ../models/paraformer_cli
```
从modelscope官网下载预训练模型，预训练模型所在的[仓库地址](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/files)
也可通过命令一键下载。

```shell
wget --user-agent="Mozilla/5.0" -c "https://www.modelscope.cn/api/v1/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/repo?Revision=v1.0.4&FilePath=model.pb"

mv repo\?Revision\=v1.0.4\&FilePath\=model.pb model.pb 
```

将用于Python的模型转换为C++的，这样更方便通过内存映射的方式直接读取参数，加快模型读取速度。

```shell
../scripts/paraformer_convert.py model.pb
```
查看转换后的参数文件wenet_params.bin的md5码，md5码为c77bc27e5758ebdc28a9024460e48602，表示转换正确。

```
md5sum -b wenet_params.bin
```

#### k2_rnnt2预训练模型下载

进入FastASR/models/k2_rnnt2_cli文件夹，用于存放下载的预训练模型.
```shell
cd ../models/k2_rnnt2_cli
```
从huggingface官网下载预训练模型，预训练模型所在的[仓库地址](https://huggingface.co/luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless2)
也可通过命令一键下载。

```shell
wget -c https://huggingface.co/luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless2/resolve/main/exp/pretrained_epoch_10_avg_2.pt
```

将用于Python的模型转换为C++的，这样更方便通过内存映射的方式直接读取参数，加快模型读取速度。

```shell
../scripts/k2_rnnt2_convert.py pretrained_epoch_10_avg_2.pt
```
查看转换后的参数文件wenet_params.bin的md5码，md5码为33a941f3c1a20a5adfb6f18006c11513，表示转换正确。

```
md5sum -b wenet_params.bin
```
#### conformer_wenetspeech-zh-16k预训练模型下载
进入FastASR/models/paddlespeech_cli文件夹，用于存放下载的预训练模型.
```shell
cd ../models/paddlespeech_cli
```
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
../scripts/paddlespeech_convert.py wenetspeech/exp/conformer/checkpoints/wenetspeech.pdparams
```
查看转换后的参数文件wenet_params.bin的md5码，md5码为9cfcf11ee70cb9423528b1f66a87eafd，表示转换正确。

```
md5sum -b wenet_params.bin
```

#### 流模式预训练模型下载
进入FastASR/models/paddlespeech_stream文件夹，用于存放下载的预训练模型.
```shell
cd ../models/paddlespeech_stream
```
从PaddleSpeech官网下载预训练模型，如果之前已经在运行过PaddleSpeech，
则可以不用下载，它已经在目录`~/.paddlespeech/models/conformer_online_wenetspeech-zh-16k`中。

```shell
wget -c https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr1/asr1_chunk_conformer_wenetspeech_ckpt_1.0.0a.model.tar.gz
```

将压缩包解压wenetspeech目录下
```
mkdir wenetspeech
tar -xzvf asr1_chunk_conformer_wenetspeech_ckpt_1.0.0a.model.tar.gz -C wenetspeech
```
将用于Python的模型转换为C++的，这样更方便通过内存映射的方式直接读取参数，加快模型读取速度。

```shell
../scripts/paddlespeech_convert.py wenetspeech/exp/chunk_conformer/checkpoints/avg_10.pdparams
```
查看转换后的参数文件wenet_params.bin的md5码，md5码为367a285d43442ecfd9c9e5f5e1145b84，表示转换正确。

```
md5sum -b wenet_params.bin
```


### 测试例子
进入项目的根目录FastASR下载用于测试的wav文件

下载时长为5S的测试音频

```shell
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav 
```

下载时长为30min的测试音频

```shell
wget -c https://github.com/chenkui164/FastASR/releases/download/V0.01/long.wav
```

#### paraformer模型测试

第一个参数为预训练模型存放的目录;  
第二个参数为需要识别的语音文件。

```shell
./build/examples/paraformer_cli models/paraformer_cli/ zh.wav
```

程序输出
```
Audio time is 4.996812 s. len is 79949
Model initialization takes 0.319781s.
Result: "我认为跑步最重要的就是给我带来了身体健康".
Model inference takes 0.695871s.
```

长语音测试

```shell
./build/examples/k2_rnnt2_cli models/k2_rnnt2_cli/ long.wav
```

程序输出
```
Audio time is 1781.655518 s. len is 28506489
Model initialization takes 0.283899s.
Result: "听众朋友您下面将要听到的是世界文学宝库中的珍品海明威最优秀的作品老人与海
................................................................................
................................................................................
................................................................................
那么祝你晚安早上我去叫醒你你是我的闹钟男孩说呵呵年纪是我的闹钟老人说为什么老头醒
醒那么早啊难道是要让白白长一些吗我不知道我只知道少年睡得沉起得晚嗯我记在心上了到
时候会去叫醒你的我不愿让船主人来叫醒我这样似乎我比他差劲儿了自我懂安睡吧老大爷男
孩儿走出屋去"
Model inference takes 238.797095s.
```
#### k2_rnnt2模型测试

第一个参数为预训练模型存放的目录;  
第二个参数为需要识别的语音文件。

```shell
./build/examples/k2_rnnt2_cli models/k2_rnnt2_cli/ zh.wav
```

程序输出
```
Audio time is 5.015000 s. len is 80240
Model initialization takes 0.211781s
result: "我认为跑步最重要的就是给我带来了身体健康"
Model inference takes 0.570641s.
```

长语音测试

```shell
./build/examples/k2_rnnt2_cli models/k2_rnnt2_cli/ long.wav
```

程序输出
```
Audio time is 1781.655518 s. len is 28506489
Model initialization takes 0.172187s.
Result: "听众朋友您下面将要听到的是世界文学宝库中的珍品海明威最优秀的作品老人与海
................................................................................
................................................................................
................................................................................
我也许不像我自以为那样的强壮了可是我懂得不少窍门儿而且有决心啊你该就去睡觉这样明儿
早上才精神饱满我要把这些东西送回露台饭店去啊哦好那么祝你晚安早上我去叫醒你你是我的
闹钟男孩说年纪是我的闹钟啊老人说为什么老头儿醒得那么早啊难道是要让白天长些吗我不知
道我只知道少年睡得沉起得晚啊嗯我记在心上了到时候会去叫醒你的我不愿让船主人来叫醒我
这样似乎我比他差劲儿了哼我懂安睡吧老大爷男孩儿走出屋去"
Model inference takes 186.848961s.
```

python wheel包测试

```shell
python examples/k2_rnnt2_cli.py models/k2_rnnt2_cli/ zh.wav
```

程序输出
```
Audio time is 4.9968125s. len is 79949.
Model initialization takes 0.8s.
Result: "我认为跑步最重要的就是给我带来了身体健康".
Model inference takes 0.57s.
```

#### conformer_wenetspeech-zh-16k模型测试

第一个参数为预训练模型存放的目录;  
第二个参数为需要识别的语音文件。

```shell
./build/examples/paddlespeech_cli models/paddlespeech_cli/ zh.wav
```

程序输出
```
Audio time is 4.996812 s.
Model initialization takes 0.217759s
result: "我认为跑步最重要的就是给我带来了身体健康"
Model inference takes 1.101319s.
```

长语音测试

```shell
./build/examples/paddlespeech_cli models/paddlespeech_cli/ long.wav
```

程序输出
```
Audio time is 1781.655518 s. len is 28506489
Model initialization takes 0.184894s.
Result: "听众朋友您下面将要听到的是世界文学宝库中珍品海明威最优秀的作品老人于海老
................................................................................
................................................................................
................................................................................
好的渔夫是你不我知道还要比我强的哪里好渔夫很多还有些很了不起的不过点呱呱的只有你
谢谢你了你说得叫我高兴我希望不要来一条大鱼打的能证明我们都讲错了这样的鱼是没有的
只要你还是像你说的那样强壮嗯我也许不像我自以为那样的强壮可是我懂得不少窍门而且有
决心你该就去睡觉这样明儿早上才精神饱满我要把这些东西送回露台饭店去好那么祝你晚安
早上我去叫醒你你是我的闹钟男孩说年纪是我的闹钟老人说为什么老头醒得那么早啊难道是
要让白天长些吗我不知道我只知道少年睡得沉起得晚嗯我记得心上啦到时候会去叫醒你的我
不愿让船主人来叫醒我这样似乎我比他差劲儿了我懂安睡吧老大爷男孩走出屋去".
Model inference takes 351.067497s.
```

python wheel包测试

```shell
python examples/paddlespeech_cli.py models/paddlespeech_cli/ zh.wav
```

程序输出
```
Audio time is 4.9968125s. len is 79949.
Model initialization takes 1.1s.
Result: "我认为跑步最重要的就是给我带来身体健康".
Model inference takes 1.1s.
```

#### conformer_online_wenetspeech-zh-16k模型测试

第一个参数为预训练模型存放的目录;
第二个参数为需要识别的语音文件。

```shell
./build/examples/paddlespeech_stream models/paddlespeech_stream/ zh.wav
```

程序输出
```
Model initialization takes 0.222937s
current result: ""
current result: ""
current result: ""
current result: ""
current result: ""
current result: ""
current result: ""
current result: ""
current result: ""
current result: ""
current result: ""
current result: ""
current result: ""
current result: ""
current result: ""
current result: "我认为跑"
current result: "我认为跑"
current result: "我认为跑"
current result: "我认为跑"
current result: "我认为跑"
current result: "我认为跑"
current result: "我认为跑"
current result: "我认为跑"
current result: "我认为跑步最重要的"
current result: "我认为跑步最重要的"
current result: "我认为跑步最重要的"
current result: "我认为跑步最重要的"
current result: "我认为跑步最重要的"
current result: "我认为跑步最重要的"
current result: "我认为跑步最重要的"
current result: "我认为跑步最重要的就是"
current result: "我认为跑步最重要的就是"
current result: "我认为跑步最重要的就是"
current result: "我认为跑步最重要的就是"
current result: "我认为跑步最重要的就是"
current result: "我认为跑步最重要的就是"
current result: "我认为跑步最重要的就是"
current result: "我认为跑步最重要的就是"
current result: "我认为跑步最重要的就是给我"
current result: "我认为跑步最重要的就是给我"
current result: "我认为跑步最重要的就是给我"
current result: "我认为跑步最重要的就是给我"
current result: "我认为跑步最重要的就是给我"
current result: "我认为跑步最重要的就是给我"
current result: "我认为跑步最重要的就是给我"
current result: "我认为跑步最重要的就是给我带来了"
current result: "我认为跑步最重要的就是给我带来了"
current result: "我认为跑步最重要的就是给我带来了"
current result: "我认为跑步最重要的就是给我带来了"
current result: "我认为跑步最重要的就是给我带来了"
current result: "我认为跑步最重要的就是给我带来了"
current result: "我认为跑步最重要的就是给我带来了"
current result: "我认为跑步最重要的就是给我带来了"
current result: "我认为跑步最重要的就是给我带来了身体健康"
current result: "我认为跑步最重要的就是给我带来了身体健康"
current result: "我认为跑步最重要的就是给我带来了身体健康"
current result: "我认为跑步最重要的就是给我带来了身体健康"
current result: "我认为跑步最重要的就是给我带来了身体健康"
current result: "我认为跑步最重要的就是给我带来了身体健康"
final result: "我认为跑步最重要的就是给我带来了身体健康"
Model inference takes 1.657996s.
```


python wheel包测试
```shell
python examples/paddlespeech_stream.py paddlespeech_stream/ zh.wav
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
编译和下载预训练模型的过程，请参考上文的<a href="#%E6%BA%90%E7%A0%81%E7%BC%96%E8%AF%91%E5%AE%89%E8%A3%85%E6%8C%87%E5%8D%97"> 源码编译安装指南</a>章节。

运行程序
```shell
./build/examples/k2_rnnt2_cli models/k2_rnnt2_cli/ zh.wav
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

### 添加标点符号

由于ASR模型并不能处理语音中的停顿，无法直接输出标点符号，需要使用NLP方式添加标点符号，参见 ： https://github.com/yeyupiaoling/PunctuationModel

相关研究方法： https://blog.csdn.net/LJJ_12/article/details/120077119

上面模型的效果比较好，缺点也明显：模型太大，速度比较慢。用于服务器端没有影响，用于客户端则影响性能。


