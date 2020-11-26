# tensorRt-inference 中文系列教程
```
注释：
    作者：leilei
    这里仅记录学习资料，代码不可执行，可执行代码详见"src"文件夹
    包含 "转onnx", "int8量化", "剪枝", "onnx2trt"
    像deepstream, TensorRT-Inference-Server等这里不介绍，详见官网。
```
## 环境配置要求
```
    darknet 
    mxnet1.6 + 
    pytorch1.5 + (yolov5实际支持pytorch1.4+)
    tensorRt7 + (tensorRt7最高支持python3.7)
    onnx1.5 + (高版本兼容低版本，但是再高tensorRt不一定支持)
    onnxruntime 1.0
    python3.6 +
    docker-cuda10.2 +
    ubuntu16.04 or ubuntu18.04
```

## 安装教程
+ [docker容器安装教程](https://blog.csdn.net/LEILEI18A/article/details/103967652)
+ [tensorRt7安装教程](https://blog.csdn.net/LEILEI18A/article/details/109319983)

## 资料链接
+ (一) 转onnx

    1. darknet2onnx
        + [tensorRt官方样例文档](https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html)
            ```
            建议：不要从网上搜博客，然后按照博客来解决问题！尽量参考deb安装后的sample样例代码，
            每个版本官方都会及时更新，这样不会走弯路！本文参考为trt7版本，因此参考的sample也是7
            版本对应的代码，找对应的版本查阅！
            eg：量化、转onnx均可查阅
            ```
        + [tensorRt7.0支持的onnx的operations](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-700/tensorrt-support-matrix/index.html#supported-ops)
        + [tensorRt不同版本链接](https://docs.nvidia.com/deeplearning/tensorrt/archives/index.html)
        + [yolov3转onnx](https://github.com/gengyanlei/onnx2tensorRt/tree/main/samples/onnx/darknet/yolov3_onnx)
            ```
            采用deb\tar包安装的tensorRt，yolov3转onnx代码在如下路径：
                /usr/src/tensorrt/samples/python/yolov3_onnx trt版本不同转onnx代码略有差异！
            tensorRt所有的python样例都在如下路径：
                /usr/src/tensorrt/samples/python
            ```
        + [**yolov4转onnx的py代码**](https://github.com/gengyanlei/onnx2tensorRt/tree/main/samples/onnx/darknet/yolov4_onnx)
            ```
            darknet yolov4转onnx引自此项目：
            https://github.com/linghu8812/tensorrt_inference/tree/master/Yolov4
            ```
        + 如何成功的转onnx，注意
            ```
            (1)参考官方deb安装包后的sample样例代码，肯定不会错，官方版本代码对应严谨，trt6、7转
               onnx有差异，因此减少了去查api的弯路
            (2)darknet yolov4转onnx，也是参考官方样例代码实现的！pytorch1.1+均可支持任意版本onnx
               mxnet支持onnx最高1.3版本
            (3)tensorRt7支持operator=10，tensorRt7.1+支持operator=12，pytorch1.4默认转onnx
               的operator=10，pytorch1.6默认转onnx的operator=12，本教程就有所瑕疵，
               但转onnx时可以设置op=10，yolov5只会warning，不影响使用！
            (4)yolov4_to_onnx.py目前可以支持YOLOv3，YOLOv3-SPP，YOLOv4等模型。在转换yolov4模型前
               需要将cfg文件里的倒数第一和倒数第三route层的-37改成116，-16改成126，修改后的正值
               可以参考darknet运行时的输出进行设置。yolov3模型不需要修改，batch也对应修改。
            (5)darknet版本yolov4转onnx，需要onnx1.5，op.resize=10，而onnx1.6，op.resize=11，
               tensorRt7仅支持op.resize=10，tensorRt7.1-7.2支持op.resize=11，op.resize版本不同
               输入命名参数不同，可以在make_resize_node中ResizeParams前面添加RoIParams。
            ```
    2. pytorch2onnx(以yolov5为例)
        
    3. mxnet2onnx(以insightface->gender-age为例)
        
+ (二) 剪枝

    1. darknet yolov4 yolov3剪枝
        ```
        (1)先采用darknet训练yolov3，获得权重
        (2)采用u版yolov3-pytorch 设置BN层稀疏操作，fine-tune darknet的yolov3权重
        (3)根据(2)获得需要剪枝的层，并生成新的cfg，以及新的权重
        (4)基于新的cfg，fine-tune剪枝后的yolov3权重，即获得剪枝！
        ```
    
    2. pytorch yolov5剪枝
        ```
        暂不实现
        ```
    
    3. 剪枝步骤
        ```
        (1)BN层稀疏训练，主要用pytorch和mxnet，稀疏训练部分代码:
            def updateBN():
                # 在梯度反向传播前执行
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        # 梯度+惩罚系数args.s*权重，让BN层的gamma变的稀疏
                        m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))
        (2)根据BN层，选择前一层Conv层保留的channel数量
            主要分成3种channel剪枝：res_block层不剪枝，block层中间层剪枝(避免channel数量
            不一致导致add操作错误)，block层全部剪枝(通道的mask进行或运算)
        ```
    
+ (三) int8量化

    1. tensorRt量化
        + [tensorRt int8 量化python样例代码](https://github.com/gengyanlei/onnx2tensorRt/tree/main/samples/int8_calibration/int8_caffe_mnist)
          ```
          采用deb\tar包安装的tensorRt，int8量化的python样例代码在如下路径：
          /usr/src/tensorrt/samples/python/int8_caffe_mnist
          ```
        + [tensorRt int8 量化原理和代码中文翻译](https://zhuanlan.zhihu.com/p/58208691)

## 参考文献
+ [Learning Efficient Convolutional Networks through Network Slimming](https://arxiv.org/abs/1708.06519)
+ [Rethinking the Value of Network Pruning](https://arxiv.org/abs/1810.05270)
