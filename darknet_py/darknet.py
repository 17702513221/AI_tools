from ctypes import *
import math
import random
import cv2
import numpy
from PIL import Image, ImageDraw, ImageFont

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("../darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res
    
def detect_np(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = array_to_image(image)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_detections(dets, num)
    return res

def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/255.0).flatten()
    data = c_array(c_float, arr)
    im = IMAGE(w,h,c,data)
    return im

def chinese_to_img(font,frame,results):
    names = {'person': '人', 'bicycle': '自行车', 'car': '汽车', 'motorbike': '摩托车', 'aeroplane': '飞机', 'bus': '公共汽车', 'train': '火车', 'truck': '卡车', 'boat': '船', 'traffic light': '交通灯', 'fire hydrant': '消火栓', 'stop sign': '停车标志', 'parking meter': '停车收费表', 'bench': '长凳', 'bird': '鸟', 'cat': '猫', 'dog': '狗', 'horse': '马', 'sheep': '羊', 'cow': '奶牛', 'elephant': '大象', 'bear': '熊', 'zebra': '斑马', 'giraffe': '长颈鹿', 'backpack': '背包', 'umbrella': '雨伞', 'handbag': '手提包', 'tie': '领带', 'suitcase': '手提箱', 'frisbee': '飞盘', 'skis': '滑雪板', 'snowboard': '滑雪板', 'sports ball': '运动球', 'kite': '风筝', 'baseball bat': '棒球棒', 'baseball glove': '棒球手套', 'skateboard': '滑板', 'surfboard': '冲浪板', 'tennis racket': '网球拍', 'bottle': '瓶子', 'wine glass': '酒杯', 'cup': '杯子', 'fork': '叉', 'knife': '刀', 'spoon': '勺子', 'bowl': '碗', 'banana': '香蕉', 'apple': '苹果', 'sandwich': '三明治', 'orange': '橙子', 'broccoli': '西兰花', 'carrot': '胡萝卜', 'hot dog': '热狗', 'pizza': '披萨', 'donut': '甜甜圈', 'cake': '蛋糕', 'chair': '椅子', 'sofa': '沙发', 'pottedplant': '盆栽植物', 'bed': '床', 'diningtable': '餐桌', 'toilet': '厕所', 'tvmonitor': '电视', 'laptop': '笔记本电脑', 'mouse': '鼠标', 'remote': '遥控器', 'keyboard': '键盘', 'cell phone': '手机', 'microwave': '微波炉', 'oven': '烤箱', 'toaster': '烤面包机', 'sink': '水池', 'refrigerator': '冰箱', 'book': '书', 'clock': '时钟', 'vase': '花瓶', 'scissors': '剪刀', 'teddy bear': '泰迪熊', 'hair drier': '吹风机', 'toothbrush': '牙刷'}
    if results:
        for i in range(len(results)+1):
           # result=str(results[i-1][0])
            #cv2.putText(frame, result, (int(results[i-1][2][0]-results[i-1][2][2]/2), int(results[i-1][2][1]-results[i-1][2][3]/2)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 1)             
            cv2.rectangle(frame,(int(results[i-1][2][0]-results[i-1][2][2]/2),int(results[i-1][2][1]-results[i-1][2][3]/2)),(int(results[i-1][2][0]+results[i-1][2][2]/2),int(results[i-1][2][1]+results[i-1][2][3]/2)),(0,255,0),1)
    # 图像从OpenCV格式转换成PIL格式
    img_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # 输出内容
    if results:
        for i in range(len(results)+1):          
            str1 = names[results[i-1][0].decode('ascii')]
            draw = ImageDraw.Draw(img_PIL)
            draw.text((int(results[i-1][2][0]-results[i-1][2][2]/2), int(results[i-1][2][1]-results[i-1][2][3]/2)), str1, font=font, fill=(255,0,0))
    # 转换回OpenCV格式
    frame = cv2.cvtColor(numpy.asarray(img_PIL),cv2.COLOR_RGB2BGR)
    return frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    net = load_net("../darknet/cfg/yolov3-tiny.cfg".encode("utf-8"), "../darknet/yolov3-tiny.weights".encode("utf-8"), 0)
    meta = load_meta("../darknet/cfg/coco.data".encode("utf-8"))
    # 字体  字体*.ttc的存放路径一般是： /usr/share/fonts/opentype/noto/ 查找指令locate *.ttc
    font = ImageFont.truetype('NotoSansCJK-Thin.ttc', 30)
    while True:
        #frame=cv2.imread('../darknet/data/dog.jpg')
        ret,frame = cap.read()      
        r = detect_np(net, meta, frame)
        frame=chinese_to_img(font,frame,r)       
        cv2.imshow('frame',frame)   
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    

