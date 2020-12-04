# yolov4 darknet->onnx->tensorRt

## 1. export onnx
+ export yolov4 onnx
```
    python yolov4_to_onnx.py --cfg_file cfg/yolov4-sg.cfg --weights_file ***/***.weights 
                             --output_file yolov4_608.onnx --strides 8,16,32 --neck PAN
```
+ export yolov3 onnx
```
    python yolov4_to_onnx.py --cfg_file cfg/yolov3.cfg --weights_file ***/***.weights 
                             --output_file yolov3_608.onnx --strides 32，16，8 --neck FPN
```

## 2. run trt inference (onnx2trt-engine)
```
    from onnx_to_trt import Detect
    detect = Detect(yaml_path=r'./config.yaml')
    detect.predict(r'/home/image/Q1helmet20200204_1046.jpg', r'./output')
```

## 3. demo
| ![helmet-detect](https://github.com/gengyanlei/onnx2tensorRt/blob/main/src/yolov4/output/00000.jpg?raw=true) |
| ---- |

## 4. refer
+ [darknet-yolov4](https://github.com/AlexeyAB/darknet)
+ [yolov3-paper](https://arxiv.org/abs/1804.02767)
+ [yolov4-paper](https://arxiv.org/abs/2004.10934)
+ [tensorrt_inference C++ version](https://github.com/linghu8812/tensorrt_inference)