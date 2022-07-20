# 定点化Roadmap
首先定点化的setting分好几种，主要如下所示 (w代表weight，a代表activation，g代表gradient)

最近两年的目前有13篇直接相关的论文，截止到2016年7月

## float转化为定点版本，不允许fine-tune
- w定点，a浮点
    - Resiliency of Deep Neural Networks under Quantization [Wongyong Sung, Sungho Shin, 2016.01.07, ICLR2016] {5bit在CIFAR10上恢复正确率}
    - Fixed Point Quantization of Deep Convolutional Networks [Darryl D.Lin, Sachin S.Talathi, 2016.06.02] {每层定点化策略不同，解析解求出}
- w+a定点
    - Hardware-oriented approximation of convolutional neural networks [Philipp Gysel, Mohammad Motamedi, ICLR 2016 Workshop] {ImageNet上8bit-8bit掉0.9%，AlexNet}
    - Energy-Efficient ConvNets Through Approximate Computing [Bert Moons, KU leuven, 2016.03.22] {结合硬件的trick可以在ImageNet上4-10bit}
    - Going Deeper with Embedded FPGA Platform for Convolutional Neural Network [Jiantao Qiu, Jie Wang, FPGA2016]{ImageNet上8bit-8bit掉1%，AlexNet}

## float转化为定点版本，允许fine-tune
- fine-tune整个网络
    - w定点，a+g浮点
        - Resiliency of Deep Neural Networks under Quantization [Wongyong Sung, Sungho Shin, 2016.01.07, ICLR2016] {2bit即三值网络在CIFAR10上恢复正确率}
    - w+a定点，g浮点
        - Fixed Point Quantization of Deep Convolutional Networks [Darryl D.Lin, Sachin S.Talathi, 2016.06.02] {每层定点化策略不同，解析解求出，CIFAR10上fine-tune后4bit-4bit掉1.32%}
    - w+a+g定点
        - Overcoming Challenges in Fixed Point Training of Deep Convolutional Networks [Darryl D.Lin, Sachin S. Talathi, Qualcomm Research，2016.07.08] {无随机rounding，ImageNet上4bit-16bit-16bit掉7.2%，a和g再小就不收敛}
        - DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients [Shuchang Zhou, Zekun Ni, 2016.06.20] {1bit-2bit-4bit, 第一层和最后一层没有量化，ImageNet上掉5.2%}
- fine-tune最高几层
    - w+a+g定点
        - Overcoming Challenges in Fixed Point Training of Deep Convolutional Networks [Darryl D.Lin, Sachin S. Talathi, Qualcomm Research，2016.07.08] {无随机rounding，ImageNet上4bit-4bit-4bit掉23.3%}
- 分阶段地从低层到高层fine-tune网络
    - w+a+g定点
        - Overcoming Challenges in Fixed Point Training of Deep Convolutional Networks [Darryl D.Lin, Sachin S. Talathi, Qualcomm Research，2016.07.08] {无随机rounding，ImageNet上4bit-4bit-4bit Top5掉11.5%}

## 直接定点从头开始训练
- w定点，a+g浮点
    - 二值网络
        - BinaryConnect: Training Deep Neural Networks with binary weights during propagations [Matthieu Courbariaux, Yoshua Bengio, 2015.11.02, NIPS] {CIFAR10上8.27%, state-of-art}
        - XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks [Mohammad Rastegari, Washington University, 2016.03.16] {ImageNet上39.2%，掉2.8%, AlexNet}
    - 三值网络
        - Ternary Weight Networks [Fengfu Li, Bin Liu, UCAS, China, 2016.05.16] {ImageNet掉2.3%, ResNet-18B}
        - Trained Ternary Quantization [Chenzhuo Zhu, Song Han, Huizi Mao, William J. Dally, ICLR2017] {ResNet上效果更佳}
- w+a定点，g浮点
    - 二值网络
        - Binarized Neural Networks: Training Neural Networks with Weights and Activations Constrained to +1 or −1 [Matthieu Courbariaux, Yoshua Bengio, 2016.03.17] {CIFAR10上10.15%}
        - XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks [Mohammad Rastegari, Washington University, 2016.03.16] {ImageNet上55.8%， 掉12.4%}
- w+a+g定点
    - Deep Learning with Limited Numerical Precision [ Suyog Gupta, Ankur Agrawal, IBM, 2015.02.09] {随机rounding技巧，CIFAR10上16bit+16bit+16bit复现正确率}
    - Training deep neural networks with low precision multiplications [Matthieu Courbariaux, Yoshua Bengio, ICLR 2015 Workshop] {CIFAR10上10bit+10bit+12bit复现正确率}
    - DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients [Shuchang Zhou, Zekun Ni, 2016.06.20] {1bit-2bit-4bit, 第一层和最后一层没有量化，ImageNet上掉8.8%}
    - Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations [Itay Hubara, Matthieu Courbariaux, 2016.09.22]{1bit-2bit-6bit，ImageNet上超过DoReFa 0.33%}