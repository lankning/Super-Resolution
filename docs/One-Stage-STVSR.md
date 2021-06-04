# One-Stage STVSR

# 介绍

文章提出了一种网络可以同时超分和插帧，目标是通过低帧率、低分辨率视频生成高分辨率慢动作视频。一般来说要完成这个task要分为两步，分别使用VFI和VSR网络。但是作者认为这样不能充分利用帧间的自然属性（natural property），此外现流行的VFI和VSR网络需要巨大的帧合成和重建模块。

因此文章提出了一种one-stage的网络来克服这些缺点。

首先，提出了一个特征时序插值网络（feature temporal interpolation network），通过获取本地时序信息来给低分辨率帧的特征插值。

然后，提出了一个可变卷积长短期网络来对齐时序信息。

最后，采用了一个深度重建网络来生成慢动作视频帧。

论文链接：[2020-Zooming slow-mo fast and accurate one stage space time video super resolution.pdf](https://arxiv.org/pdf/2002.11616.pdf)

[论文本地链接](./One-Stage-STVSR/2020-Zooming slow-mo fast and accurate one stage space time video super resolution.pdf)

原文的代码地址如下：[https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020)。

## 相关工作

VSR的重要问题是如何对齐前后帧。针对这个问题，一些文章使用了光流法（optical flow），但是获得精确的光流很难，而且光流法是人工产品（artifacts）。

为了规避这个问题，DUF和TDAN分别使用了动态卷积核和可变卷积对齐。EDVR在TDAN基础上对多尺度信息进行采样。然而，它们都是`many to one`的model，需要一堆LR帧来预测HR帧，计算低效。

RNN可以缓解sequence-to-sequence learning，它们可以放进VSR问题中来利用时序信息。可是，没有明确的时序对齐，RNN-based VSR网络也没啥用。

因此，文字提出了一种ConvLSTM架构（`many to many`的），嵌入了一种明确的状态更新单元，用来进行时空视频超分。

![statement-conv-lstm](./One-Stage-STVSR/statement-conv-lstm.png)

## 名词解释

1. HR: high-resolution，高分辨率

2. LFR: low frame rate，低帧率

3. LF: low-resolution，低分辨率

4. VFI: video frame interpolation，视频帧插值

5. VSR: video super resolution，视频超分辨率

6. Video Deblur：视频去模糊

7. Temporal Interpolation：时间插值

8. well-posed problem：解存在且唯一

9. ill-posed problem：解可能不存在也可能不唯一

10. inverse problem：反问题，已知输出求输入



# 网络架构

整个网络的结构如下图所示，先通过残差块提取`Feature Maps`，再通过特征时序插值无中生有出中间帧的`Feature Maps`，经过`LSTM`进一步特征提取，然后每一帧的`Feature Maps`都经过残差块得到标准需要的`Feature Maps`形状，最后通过`Pixel Shuffle`重建出三个HR帧。不寻常的有两个地方，分别是特征插值和ConvLSTM模块。

![net-arch](./One-Stage-STVSR/net-arch.png)

## Frame Feature Temporal Interpolation

![feature interpolation formula](./One-Stage-STVSR/feature interpolation.png)

![frame feature fig](./One-Stage-STVSR/frame feature fig.png)

## Bidirectional Deformable ConvLSTM



# 结果

<iframe frameborder="0" src="https://v.qq.com/txp/iframe/player.html?vid=e3250valjad" height="600" width="100%" allowFullScreen="true"></iframe>





# 作者

朱文康

如果你有问题，欢迎联系我。我的邮箱是：[1119741654@qq.com](1119741654@qq.com)，乐意回复。

谢谢。



# 参考文献

- 
