# One-Stage STVSR

# 介绍

文章提出了一种网络可以同时超分和插帧，目标是通过低帧率、低分辨率视频生成高分辨率慢动作视频。一般来说要完成这个task要分为两步，分别使用VFI和VSR网络。但是作者认为这样不能充分利用帧间的自然属性（natural property），此外现流行的VFI和VSR网络需要巨大的帧合成和重建模块。

因此文章提出了一种one-stage的网络来克服这些缺点。

首先，提出了一个特征时序插值网络（feature temporal interpolation network），通过获取本地时序信息来给低分辨率帧的特征插值。

然后，提出了一个可变卷积长短期网络来对齐时序信息。

最后，采用了一个深度重建网络来生成慢动作视频帧。

论文链接：[2020-Zooming slow-mo fast and accurate one stage space time video super resolution.pdf](https://arxiv.org/pdf/2002.11616.pdf)

[论文本地链接](./STVSR/2020-Zooming slow-mo fast and accurate one stage space time video super resolution.pdf)

原文的代码地址如下：[https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020)。

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

![net-arch](./STVSR/net-arch.png)



# 结果

<iframe frameborder="0" src="https://v.qq.com/txp/iframe/player.html?vid=e3250valjad" height="600" allowFullScreen="true"></iframe>





# 作者

朱文康

如果你有问题，欢迎联系我。我的邮箱是：[1119741654@qq.com](1119741654@qq.com)，乐意回复。

谢谢。



# 参考文献

- 
