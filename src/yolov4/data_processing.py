'''
注释：
    参考对应版本的trt样例代码，修改成属于自己的代码
'''
import cv2
import math
import copy
from math import ceil
import numpy as np
from PIL import Image
from itertools import product

# 读取标签
def load_label_categories(label_file_path):
    categories = [line.rstrip('\n') for line in open(label_file_path)]
    return categories

# yolo预处理
class PreprocessYOLO(object):
    """A simple class for loading images with PIL and reshaping them to the specified
    input resolution for YOLOv3-608.
    """

    def __init__(self, yolo_input_resolution):
        """Initialize with the input resolution for YOLOv3, which will stay fixed in this sample.

        Keyword arguments:
        yolo_input_resolution -- two-dimensional tuple with the target network's (spatial)
        input resolution in HW order
        """
        self.yolo_input_resolution = yolo_input_resolution

    def process(self, input_image_path):
        # 更改，input_image_path默认是图像路径，但是可以是图像ndarray；
        """Load an image from the specified input path,
        and return it together with a pre-processed version required for feeding it into a
        YOLOv3 network.

        Keyword arguments:
        input_image_path -- string path of the image to be loaded
        """
        image_raw, image_resized = self._load_and_resize(input_image_path)
        image_preprocessed = self._shuffle_and_normalize(image_resized)
        return image_raw, image_preprocessed

    def _load_and_resize(self, input_image_path):
        # 更改，input_image_path默认是图像路径，但是可以是图像ndarray；
        """Load an image from the specified path and resize it to the input resolution.
        Return the input image before resizing as a PIL Image (required for visualization),
        and the resized image as a NumPy float array.

        Keyword arguments:
        input_image_path -- string path of the image to be loaded
        """
        if type(input_image_path)==str:
            # bgr->rgb
            image_raw = cv2.imread(input_image_path, -1)[..., ::-1]  # bgr->rgb
            # TODO 判断灰度图，将灰度图转成rgb
            image_raw = Image.fromarray(image_raw)
        else:
            # cv2 读取就是bgr bgr->rgb
            assert type(input_image_path)==np.ndarray, "input_image_path若不是str，必须是array"
            image_raw = Image.fromarray(input_image_path[..., ::-1])
        # Expecting yolo_input_resolution in (height, width) format, adjusting to PIL
        # convention (width, height) in PIL:
        new_resolution = (
            self.yolo_input_resolution[1],
            self.yolo_input_resolution[0])
        image_resized = image_raw.resize(
            new_resolution, resample=Image.BICUBIC)
        image_resized = np.array(image_resized, dtype=np.float32, order='C')
        return image_raw, image_resized

    def _shuffle_and_normalize(self, image):
        """Normalize a NumPy array representing an image to the range [0, 1], and
        convert it from HWC format ("channels last") to NCHW format ("channels first"
        with leading batch dimension).

        Keyword arguments:
        image -- image as three-dimensional NumPy float array, in HWC format
        """
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.array(image, dtype=np.float32, order='C')
        return image

'''
人脸的损失函数和yolo的是不同的，因此anchor的计算使用方式上也是不同的；
人脸的anchor 是 xywh，中心位置以及anchor尺寸一定在feature_map范围内，因此可以归一化至0-1，预测也是根据中心位置进行逻辑回归，进而可以直接使用原图进行预测
而yolo是基于feature_map最近的像素点左上角的偏移，也是0-1，但是不能与人脸的相提并论，而人脸的xywh在每一个feature_map级别上wh是固定的，相对较大，xy是递增的
如果强行借鉴人脸的，那么yolo的'预测'xy偏移是基于像素的[0-1]偏移，anchor的xy(左上角而非人脸的加0.5中心点，这是区别)可以基于整张图，但是'预测'xy偏移是基于feature_map的
那么就无法跳过feature_map了，因此不可相提并论！！！
因此：
yolo的预设anchor
x,y,w,h,f_w,f_h: xy是feature_map的左上角(非中心)列行坐标，wh是anchor，f_wh是feature_map大小
'''
class PriorBox(object):
    def __init__(self, cfg):
        self.anchors = cfg['anchors']  # 'anchors' -wh: [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]
        self.steps = cfg['stride']  # 8-16-32
        self.yolo_masks = cfg['yolo_masks']  # [(0, 1, 2), (3, 4, 5), (6, 7, 8)] 必须与stride一一对应
        self.image_size = cfg['input_shape'][-2:]  # [608,608]；nchw
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]  # hw 特征图大小

    def forward(self):
        anchors = []  # x,y,w,h,f_w,f_h; N AHW 4+1+class, 那么anchor遍历应该在feature_map前面
        for k, f in enumerate(self.feature_maps):  # 3个
            # 按照stride的顺序， f：hw，然后reshape也是按照hw相乘的顺序，那么这里也得按照hw的先行后列
            yolo_mask = self.yolo_masks[k]  # yolo_masks 也是3个，然后每个里面再3个；(0,1,2)
            # N AHW C 那么anchor也需要按照AHW顺序来，yolo_mask在HW前面; 0123-0123-0123，若是HWA，那么就是000-111-222-333
            for index in yolo_mask:
                anchor_w, anchor_h = self.anchors[index]
                for y,x in product(range(f[0]), range(f[1])): #  hw遍历
                    anchors += [x, y, anchor_w, anchor_h, f[1], f[0]]

        output = np.array(anchors).reshape([-1, 6])

        return output


# yolo后处理
class PostprocessYOLO:
    def __init__(self,
                 prior_anchor,
                 param_dict,):
        self.prior_anchor = prior_anchor  # [AHW, 6] 预先生成的anchor
        self.obj_threshold = param_dict['conf_threshold']  # 置信度
        self.nms_threshold = param_dict['nms_threshold']  # iou阈值
        self.yolo_input_hw = param_dict['input_shape'][-2:]  # 网络输入尺寸 HW
        self.letter_box = param_dict['letter_box']  # 是否按照yolov5的预测方式预测，若是则xywh均采用sigmoid操作
        self.onnx_sigmoid = param_dict['onnx_sigmoid']  # 转onnx时是否sigmoid
        self.batch = param_dict['input_shape'][0]
        print(self.yolo_input_hw)

    def process(self, outputs, shape_orig_WH):
        '''
        :param outputs:   list的trt_outputs  shape需要是[N, AHW, 4+1+C]
        :param shape_orig_WH:  图像原始尺寸
        :return:
        '''
        # 先将输出concat在一起，即使之前已经concat输出为1个了，那么后面的就只是对batch=1进行遍历
        outputs_concat = list()  # 一定要有batch的维度！
        for output in outputs:
            outputs_concat.append(self._process_yolo(output, shape_orig_WH))
        outputs_concat = np.concatenate(outputs_concat)

        # 逐batch=1进行遍历，然后再逐类nms
        '''
        ####### 注意：NMS需要逐张图像进行预处理，必须for循环，遍历每张图像的，不能一起操作 #######
        '''
        outputs_pred = list()  # 每张图像1个[]
        for i in range(self.batch):
            output = outputs_concat[i]  # 去掉batch维度
            # 类别nms在_filter函数内
            outputs_pred.append([self._filter(output)])

        return outputs_pred

    def _filter(self, output):
        '''
        :param output:  单张图像的所有输出的concat，[AHW, 4+1+class]
        :return:  单张图像经过nms
        '''
        output = copy.copy(output)
        # 对单张图像进行置信度阈值过滤，nms过滤
        # 1. 先横向求类别中最大的置信度值
        # 置信度最大的类别0-class_num
        index_pred = np.argmax(output[..., 5:], axis=-1)
        # 将对象概率替换为最大的置信度
        output[..., 4] = np.max(output[..., 5:], axis=-1)

        # 2. 再求大于阈值的，参考yolov3_onnx的data_process的_filter_boxes，对象概率*类别概率=置信度
        pos = np.where(output[..., 4] >= self.obj_threshold)  # 保留最后一维整体，过滤的是前面的维度；前面的维度
        # 获取过滤的xywh conf class
        xywh = output[..., :4][pos]
        conf = output[..., 4][pos]
        class_pred = index_pred[pos]

        # 3. nms  参考yolov3_onnx的data_process的_nms_boxes，每个类别都要nms
        # 不管多少维度的输出，这里都是单张图像的输出，都需要转成2维度，然后过滤，形状还是2维度
        boxes = xywh.reshape([-1, 4])  # [N',4]
        confidences = conf.reshape([-1,])  # [N',1]需要改成[N',]
        categories = class_pred.reshape([-1,])  # [N',1]需要改成[N',]
        assert len(boxes)==len(confidences)
        assert len(boxes)==len(categories)

        # 类别操作
        print(set(categories))
        nms_boxes, nms_categories, nscores = list(), list(), list()
        for category in set(categories):
            idxs = np.where(categories == category)
            box = boxes[idxs]
            category = categories[idxs]
            confidence = confidences[idxs]

            keep = self._nms_boxes(box, confidence)

            nms_boxes.append(box[keep])
            nms_categories.append(category[keep])
            nscores.append(confidence[keep])

        if not nms_categories and not nscores:
            return None, None, None

        boxes = np.concatenate(nms_boxes)
        categories = np.concatenate(nms_categories)
        confidences = np.concatenate(nscores)

        return boxes, categories, confidences

    def _process_yolo(self, output, shape_orig_WH):
        '''
        :param output: ndarray  [batch, AHW, 4+1+class] numpy支持broadcast操作
        :param shape_orig_WH:  WH
        :return:处理的是list中的trt输出，可以是1个，也可以是3个
        '''
        def sigmoid(array_value, onnx_sigmoid=False):
            # 若在onnx时已经sigmoid，这时只需要返回即可
            if onnx_sigmoid:
                # onnx时添加sigmoid 一般是yolov4新预测方式用
                return array_value
            return 1.0 / (1.0 + np.exp(-array_value))
        def exponential(array_value):
            return np.exp(array_value)
        def pow(array_value, power=2):
            return np.power(array_value, power)
        '''
        详见 README_Images/6_yolov4_process.jpg
        '''
        output = copy.copy(output)
        if self.letter_box:
            ratio = max(shape_orig_WH[0]/self.yolo_input_hw[1], shape_orig_WH[1]/self.yolo_input_hw[0])
        # 判断采用哪种预测方式，直接原址操作数值
        # x-Width  中心坐标
        output[..., 0] = (sigmoid(output[..., 0], self.onnx_sigmoid) * 2 - 0.5 + self.prior_anchor[..., 0]) / self.prior_anchor[..., 4] * self.yolo_input_hw[1] * ratio if self.letter_box else \
            (sigmoid(output[..., 0], self.onnx_sigmoid) + self.prior_anchor[..., 0]) / self.prior_anchor[..., 4] * shape_orig_WH[0]
        # y-Height
        output[..., 1] = (sigmoid(output[..., 1], self.onnx_sigmoid) * 2 - 0.5 + self.prior_anchor[..., 1]) / self.prior_anchor[..., 5] * self.yolo_input_hw[0] * ratio if self.letter_box else \
            (sigmoid(output[..., 1], self.onnx_sigmoid) + self.prior_anchor[..., 1]) / self.prior_anchor[..., 5] * shape_orig_WH[1]
        # w   AHW 单维度
        output[..., 2] = (pow(sigmoid(output[..., 2], self.onnx_sigmoid) * 2) * self.prior_anchor[..., 2]) * ratio if self.letter_box else \
            (exponential(output[..., 2]) * self.prior_anchor[..., 2] / self.yolo_input_hw[1] * shape_orig_WH[0])
        # h
        output[..., 3] = (pow(sigmoid(output[..., 3], self.onnx_sigmoid) * 2) * self.prior_anchor[..., 3]) * ratio if self.letter_box else \
            (exponential(output[..., 3]) * self.prior_anchor[..., 3] / self.yolo_input_hw[0] * shape_orig_WH[1])
        # yolo的置信度是 对象概率*类别概率，将类别概率替换为置信度，人脸因为是2分类，所以不需要 至少1类，4+1+1， broadcast操作也是有些限制的！单个数or同样的维度(值不一样时为1)，或者单维度
        output[..., 5:] = np.expand_dims(sigmoid(output[..., 4], self.onnx_sigmoid), axis=-1) * sigmoid(output[..., 5:], self.onnx_sigmoid)

        '''
        核心：nms的输入是左上角坐标和wh，而不是中心点坐标！ 幸好有人告知，我以为已经都处理好了！，画框必须是左上角+wh
        中心坐标转左上角坐标，wh不变
        '''
        output[..., 0] = output[..., 0] - (output[..., 2] / 2)
        output[..., 1] = output[..., 1] - (output[..., 3] / 2)

        return output

    def _nms_boxes(self, boxes, box_confidences):
        """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding boxes with their
        confidence scores and return an array with the indexes of the bounding boxes we want to
        keep (and display later).

        Keyword arguments:
        boxes -- a NumPy array containing N bounding-box coordinates that survived filtering,
        with shape (N,4); 4 for x,y,height,width coordinates of the boxes
        box_confidences -- a Numpy array containing the corresponding confidences with shape N
        """
        x_coord = boxes[:, 0]
        y_coord = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]

        areas = width * height
        ordered = box_confidences.argsort()[::-1]

        keep = list()
        while ordered.size > 0:
            # Index of the current element:
            i = ordered[0]
            keep.append(i)
            xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
            yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
            xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
            yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

            width1 = np.maximum(0.0, xx2 - xx1 + 1)
            height1 = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = width1 * height1
            union = (areas[i] + areas[ordered[1:]] - intersection)

            # Compute the Intersection over Union (IoU) score:
            iou = intersection / union

            # The goal of the NMS algorithm is to reduce the number of adjacent bounding-box
            # candidates to a minimum. In this step, we keep only those elements whose overlap
            # with the current bounding box is lower than the threshold:
            indexes = np.where(iou <= self.nms_threshold)[0]
            ordered = ordered[indexes + 1]

        keep = np.array(keep)
        return keep















    # import yaml
# with open(r'/home/gengyanlei/Python_work/onnx2tensorRT/src/yolov4/config.yaml', 'r', encoding='utf-8') as f:
#     param_dict = yaml.load(f, Loader=yaml.FullLoader)
# anchors = PriorBox(param_dict).forward()
# print(anchors)