# 介绍
本仓库将对一些论文中提出的`SR算法`进行总结和复现，使用的推理框架是`TensorFlow`。



# 模型
- [x] [SRCNN 2014](./SRCNN.html)
- [x] [FSRCNN 2016](./FSRCNN.html)
- [x] [ESPCN 2016](./ESPCN.html)
- [x] [VESPCN 2017](./VESPCN.html) （Only Notes）
- [x] [DUF 2018](./DUF.html)
- [x] [FALSR 2019](./FALSR.html) （Only Notes）
- [x] [TGA 2020](./TGA.html)（Only Notes）
- [x] [One-Stage STVSR 2020](./One-Stage-STVSR.html) ）（Partly codes）
- [x] [FSR 2021](./FSR.html)



# 环境配置
1. 激活conda环境
```
conda create -n sr python=3.7
conda activate sr
conda install cudatoolkit=10.1
conda install cudnn
```
2. 安装环境
```
pip install -r requirements.txt
```




# 作者

朱文康

如果你有问题，欢迎联系我。我的邮箱是：[1119741654@qq.com](1119741654@qq.com)，乐意回复。

谢谢。



# 参考文献

- 论文
- https://tensorflow.google.cn/
- [学习笔记之——基于深度学习的图像超分辨率重建](https://blog.csdn.net/gwplovekimi/article/details/83041627?utm_medium=distribute.pc_relevant_download.none-task-blog-baidujs-8.nonecase&depth_1-utm_source=distribute.pc_relevant_download.none-task-blog-baidujs-8.nonecase#ESPCN%EF%BC%88Efficient%20Sub-Pixel%20Convolutional%20Neural%20Network%EF%BC%89)
- [视频超分(VSR)](https://blog.csdn.net/srhyme/category_10487261.html)