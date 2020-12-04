'''
注释：
    参考tensorRt对应版本的sample代码，修改成python版本的tensorRt推理
    onnx将yolo的3个输出concat在一起，因此参考官方样例代码，修改！
注：
    记住1点，首先根据你的设计输出，然后才能构建自己的代码，并不能设计成什么都可以兼容的
'''
import os
import cv2
import yaml
import common
import numpy as np
import tensorrt as trt
from PIL import ImageDraw, ImageFont

# 导入yolo的预处理 后处理函数
from data_processing import PreprocessYOLO, load_label_categories, PriorBox, PostprocessYOLO

# 常量
TRT_LOGGER = trt.Logger()

# 创建文件夹
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return

def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color='blue'):
    """Draw the bounding boxes on the original input image and return it.

    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
                更改一下颜色
    """
    np.random.seed(1)
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(all_categories))]
    text_size = 20
    font = ImageFont.truetype("/home/gengyanlei/Datasets/NotoSansCJK-Black.ttc", text_size)

    draw = ImageDraw.Draw(image_raw)
    # print(bboxes, confidences, categories)
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

        bbox_color = tuple(colors[category]) or bbox_color
        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color, width=2)
        draw.text((left, top - 20), '{0} {1:.2f}'.format(all_categories[category], score), fill=bbox_color, font=font)

    return image_raw


# 获取tensorRt的engine
def get_engine(onnx_file_path, engine_file_path, input_shape=[1,3,608,608], int8_calibration=False):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    # 如下警告：是因为 transpose onnx仅支持int64，因此trt强制转成int32，不影响！
    # [TensorRT] WARNING: onnx2trt_utils.cpp:198: Your ONNX model has been generated with INT64 weights,
    # while TensorRT does not natively support INT64. Attempting to cast down to INT32.
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        batch = input_shape[0]
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 28  # 256MiB
            builder.max_batch_size = batch

            if int8_calibration:
                # int8量化 暂不实现
                pass
            else:
                # 设置float16
                builder.fp16_mode = True
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            # 与trt5-6 区别
            network.get_input(0).shape = input_shape
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

'''
注释：
    主要处理3种数据:
    (1)图像
    (2)文件夹中的图像集
    (3)单个视频
    (4)视频流 暂时不支持！！！
注：
    创建class，将engine作为属性存在，这样就可以随时使用了。
'''
class Detect:
    def __init__(self, yaml_path):
        # yaml_path 参数配置文件路径
        with open(yaml_path, 'r', encoding='utf-8') as f:
            self.param_dict = yaml.load(f, Loader=yaml.FullLoader)

        # 获取engine context
        self.engine = get_engine(self.param_dict['onnx_path'], self.param_dict['engine_path'],
                                 self.param_dict['input_shape'], self.param_dict['int8_calibration'])
        # context 执行在engine后面
        self.context = self.engine.create_execution_context()

        # yolo 数据预处理 PreprocessYOLO类
        assert len(self.param_dict['input_shape']) == 4, "input_shape必须是4个维度"
        batch, _, height, width = self.param_dict['input_shape']
        self.preprocessor = PreprocessYOLO((height, width))

        # 生成预先的anchor [x,y,w,h,f_w,f_h]: xy是feature_map的列行坐标，wh是anchor，f_wh是feature_map大小
        self.prior_anchors = PriorBox(cfg=self.param_dict).forward()

        # 一些配置
        # 标签名字
        self.all_categories = load_label_categories(self.param_dict['label_file_path'])
        classes_num = len(self.all_categories)
        # trt输出shape
        stride = self.param_dict['stride']
        num_anchors = self.param_dict['num_anchors']

        grid_num = (height // stride[0]) * (width // stride[0]) * num_anchors[0] + (height // stride[1]) * (
                    width // stride[1]) * num_anchors[1] + (height // stride[2]) * (width // stride[2]) * num_anchors[2]
        self.output_shapes = [(batch, grid_num, (classes_num + 5))]

        self.img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes
        self.vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

        # yolo 后处理， yolov4将3个输出 concat在一起，[N, AHW*3, classes_num+5]，可判断yolov4原始预测 or yolov5新式预测
        self.postprocessor = PostprocessYOLO(self.prior_anchors, self.param_dict)




    def predict(self, input_path='dog.jpg', output_save_root='./output', write_txt=False):
        '''
        :param input_path:  输入：单张图像路径，图像文件夹，单个视频文件路径
        :param output_save_root: 要求全部保存到文件夹内，若是视频统一保存为mp4
        :param write_txt: 将预测的框坐标-类别-置信度以txt保存
        :return:
        '''
        # 开始判断图像，文件夹，视频
        is_video = False
        path = input_path
        if os.path.isdir(path):
            # 图像文件夹
            img_names = os.listdir(path)
            img_names = [name for name in img_names if name.split('.')[-1] in self.img_formats]
        elif os.path.isfile(path):
            # 将 '/hme/ai/111.jpg' -> ('/hme/ai', '111.jpg')
            path, img_name = os.path.split(path)
            # 标记 video
            if img_name.split('.')[-1] in self.vid_formats:
                is_video = True
            else:
                assert img_name.split('.')[-1] in self.img_formats, "必须是单张图像路径"
                img_names = [img_name]
        else:
            print("输入无效！！！" * 3)

        # 创建保存文件夹
        check_path(output_save_root)
        # 判断是否是视频
        if is_video:
            assert img_name.count('.') == 1, "视频名字必须只有1个 . "

            # 读取视频
            cap = cv2.VideoCapture(os.path.join(path, img_name))
            # # 获取视频的fps， width height
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频总帧数
            # 创建视频
            video_save_path = os.path.join(output_save_root, img_name.split('.')[0] + '_pred.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_save_path, fourcc=fourcc, fps=fps, frameSize=(width, height))
        else:
            num = len(img_names)  # 图像数量

        # 推理 默认是0卡
        inputs, outputs, bindings, stream = common.allocate_buffers(self.engine)
        # Do inference
        for i in range(num):
            # 预处理
            if is_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # 读取指定帧
                image = cap.read()
                # 输入的是bgr帧矩阵
                image_raw, image = self.preprocessor.process(image)
            else:
                # 输入的默认是图像路径
                image_raw, image = self.preprocessor.process(os.path.join(path, img_names[i]))

            # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
            inputs[0].host = image
            trt_outputs = common.do_inference_v2(self.context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

            # list中的输出个数，本来要位于外面一层的，但是考虑重新输入图像
            trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, self.output_shapes)]

            # 后处理，按照2种方式判断处理，yolov4原始的预测-参考yolov5变化后的预测
            # 图像原始尺寸 WH，因为时PIL读取
            shape_orig_WH = image_raw.size

            # 后处理是可以处理batch>=1的，但是这里的类写的只能是batch=1
            outputs_pred = self.postprocessor.process(trt_outputs, shape_orig_WH)

            # TODO 将预测的框坐标-类别-置信度 写入txt

            # 画框，由于这里只能是单张图像，因此不必for遍历
            boxes, classes, scores = outputs_pred[0][0]
            obj_detected_img = draw_bboxes(image_raw, boxes, scores, classes, self.all_categories)

            # 视频按照帧数来保存，图像按照名字保存,  注意一般视频不会超过5位数
            # TODO 视频的预测写入视频
            if is_video:
                obj_detected_img.save(os.path.join(output_save_root, str(i).zfill(5)))
            else:
                obj_detected_img.save(os.path.join(output_save_root, img_names[i]))


        # 若是视频，需要 release
        if is_video:
            cap.release()
            cv2.destroyAllWindows()




# # 函数式 推理，但是不合适，一旦中止，就无法继续推理，需要重新加载
# def main(input_path='dog.jpg', output_save_root='./output', yaml_path='./config.yaml'):
#     '''
#     :param input_path: 输入：单张图像路径，图像文件夹，单个视频文件路径
#     :param output_save_root: 要求全部保存到文件夹内，若是视频统一保存为mp4
#     :param yaml_path: 配置文件
#     :return:
#     '''
#     # 获取超参
#     with open(yaml_path, 'r', encoding='utf-8') as f:
#         param_dict = yaml.load(f, Loader=yaml.FullLoader)
#     input_shape = param_dict['input_shape']
#     onnx_path = param_dict['onnx_path']
#     engine_path = param_dict['engine_path']
#     int8_calibration = param_dict['int8_calibration']
#     letter_box = param_dict['letter_box']
#     label_name_txt = param_dict['label_name_txt']
#     stride = param_dict['stride']
#     num_anchors = param_dict['num_anchors']
#
#     assert len(input_shape)==4, "input_shape必须是4个维度"
#     batch, _, height, width = input_shape
#
#     # 读取label_name
#     label_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), label_name_txt)
#     all_categories = load_label_categories(label_file_path)
#     classes_num = len(all_categories)
#
#     # yolo预处理设置参数
#     input_resolution_yolov3_HW = (height, width)
#     # Create a pre-processor object by specifying the required input resolution for YOLOv3
#     preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
#
#     # yolo的3个输出concat在一起
#     # 设置输出形状
#     assert len(stride)==3, "yolo必须是3个stride"
#     grid_num = (height//stride[0])*(width//stride[0])*num_anchors[0] + (height//stride[1])*(width//stride[1])*num_anchors[1] + (height//stride[2])*(width//stride[2])*num_anchors[2]
#     output_shapes = [(batch, grid_num, (classes_num+5))]
#
#     img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes
#     vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
#
#     # 开始判断图像，文件夹，视频
#     path = input_path
#     if os.path.isdir(path):
#         # 图像文件夹
#         img_names = os.listdir(path)
#         img_names = [name for name in img_names if name.split('.')[-1] in img_formats]
#     elif os.path.isfile(path):
#         # 将 '/hme/ai/111.jpg' -> ('/hme/ai', '111.jpg')
#         path, img_name = os.path.split(path)
#         # 标记 video
#         if img_name.split('.')[-1] in vid_formats:
#             is_video = True
#         else:
#             assert img_name.split('.')[-1] in img_formats, "必须是单张图像路径"
#             img_names = [img_name]
#
#     # 创建保存文件夹
#     if not os.path.exists(output_save_root):
#         os.makedirs(output_save_root)
#
#     # 判断是否是视频
#     if is_video:
#         assert img_name.count('.')==1, "视频名字必须只有1个 . "
#
#         # 读取视频
#         cap = cv2.VideoCapture(os.path.join(path, img_name))
#         # # 获取视频的fps， width height
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 帧数
#         # 创建视频
#         video_save_path = os.path.join(output_save_root, img_name.split('.')[0] + '_pred.mp4')
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         video_writer = cv2.VideoWriter(video_save_path, fourcc=fourcc, fps=fps, frameSize=(width, height))
#     else:
#         num = len(img_names)  # 图像数量
#
#     print('='*20+'>>'+'开始推理')
#     # Do inference with TensorRT
#     trt_outputs = []
#     # 在创建engine的范围内进行上下文推理，所以不能中止; 如果改成engine=, context= 就可以一直执行了，因此修改成类，初始化就可以执行了！self.engine, self.context
#     with get_engine(onnx_path, engine_path, input_shape, int8_calibration) as engine, engine.create_execution_context() as context:  # 上下文
#         # 分配空间
#         inputs, outputs, bindings, stream = common.allocate_buffers(engine)
#         # Do inference
#         for i in range(num):
#             # 预处理
#             if is_video:
#                 cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#                 image = cap.read()
#                 # 输入的是帧矩阵
#                 image_raw, image = preprocessor.process(image)
#             else:
#                 # 输入的默认是图像路径
#                 image_raw, image = preprocessor.process(os.path.join(path, img_names[i]))
#
#             # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
#             inputs[0].host = image
#             trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
#
#             # list中的输出个数，本来要位于外面一层的，但是考虑重复输入图像
#             trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
#
#             # 后处理，按照2种方式判断处理，yolov4原始的预测-参考yolov5变化后的预测
#             # 暂时不将坐标return
#
#
# input_resolution_yolov3_HW = (608, 608)
# # Create a pre-processor object by specifying the required input resolution for YOLOv3
# preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
# # with get_engine(r'yolov4_608.onnx', r'yolov4_608_fp16.trt', (1,3,608.608), False) as engine, engine.create_execution_context() as context:  # 上下文
# #     # 分配空间
# #     inputs, outputs, bindings, stream = common.allocate_buffers(engine)
# #     # Do inference
# #     image_raw, image = preprocessor.process(r'/home/gengyanlei/Datasets/image/IMG_20200313_123032.jpg')
# #
# #     # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
# #     inputs[0].host = image
# #     trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
# #     print(trt_outputs[0].shape)
# engine = get_engine(r'yolov4_608.onnx', r'yolov4_608_fp16.trt', (1,3,608.608), False)
# context = engine.create_execution_context()
#
# inputs, outputs, bindings, stream = common.allocate_buffers(engine)
# # 后面可以重复操作
# image_raw, image = preprocessor.process(r'/home/gengyanlei/Datasets/image/IMG_20200313_123032.jpg')
# inputs[0].host = image
# trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
# print(trt_outputs[0].shape)

if __name__ == '__main__':
    detect = Detect(yaml_path=r'./config.yaml')
    detect.predict(r'/home/gengyanlei/Datasets/15_10_17.jpg', r'./output')