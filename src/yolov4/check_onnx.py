'''
注释：
    (1)测试转onnx是否成功，首先采用onnxruntime测试一下输出
    (2)将onnx转trt后，对测试图像输出结构，进行评估指标，进而验证是否转trt有效
    ----
    若想使用onnxruntime-gpu，需要重新安装，现在使用的是cpu版本！
'''
import onnxruntime
import numpy as np

sess_options = onnxruntime.SessionOptions()
sess = onnxruntime.InferenceSession('./yolov4_608.onnx', sess_options)
data = [np.random.rand(1, 3, 608, 608).astype(np.float32)]
input_names = sess.get_inputs()
feed = zip(sorted(i_.name for i_ in input_names), data)
actual = sess.run(None, dict(feed))
print(len(actual))
print(actual[0].shape)