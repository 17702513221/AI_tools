#!/usr/bin/env python
#-*- coding:utf-8 -*- 
import cv2
import subprocess as sp

class ServiceCamera:
    """ 
        Service 
        管理摄像头 识别opencv 判断处理 发送监控提醒socket推送
    """ 
    def __init__(self, serverSocket):
        self.ifRtmpPush = "0"
        self.serverSocket = serverSocket    # 通过此来推送关键消息
        return
    def doMethod(self, method, params):
        # params = params.encode('utf-8')
        # method = method.encode('utf-8')

        # tool.doMethod(self, method, params)
        print("class:  " + self.__class__.__name__)    #className
        print("method: " + method)    #list
        print("params: " + params)    #{arg1: 'a1', arg2: 'a2' }
        #检查成员
        ret = hasattr(self, method) #因为有func方法所以返回True 
        if(ret == True) :
            #获取成员
            method = getattr(self, method)#获取的是个对象
            return method(params) 
        else :
            print("Error ! 该方法不存在")
            return ""


# 开启摄像头监控识别
    def start(self):
        self.rtmpUrl = 'rtmp://127.0.0.1:1935/live/STREAM_NAME'
        # 视频来源
        #filePath='/home/xs/'
        #camera = cv2.VideoCapture(filePath+"head-pose-face-detection-female.mp4") # 从文件读取视频

        camera = cv2.VideoCapture(0) # 参数0表示第一个摄像头 摄像头读取视频
        if (camera.isOpened()):# 判断视频是否打开 
            print ("Open camera")
        else:
            print ("Fail to open camera!")
            return
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 600)  # 2560x1920 2217x2217 2952×1944 1920x1080
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 450)
        camera.set(cv2.CAP_PROP_FPS, 5)

        # 视频属性
        size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # size = (size[0] / 10, size[1] / 10)
        sizeStr = str(size[0]) + 'x' + str(size[1])
        fps = camera.get(cv2.CAP_PROP_FPS)  # 30p/self
        fps = int(fps)
        print ("size:"+ sizeStr + " fps:" + str(fps))  

        # 视频文件保存
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter(filePath+'res_mv.avi',fourcc, fps, size)
        # 管道输出 ffmpeg推送rtmp
        command = ['ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec','rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', sizeStr,
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-f', 'flv', 
            self.rtmpUrl]
        pipe = sp.Popen(command, stdin=sp.PIPE) #,shell=False

        lineWidth = 1 + int((size[1]-400) / 400)# 400 1 800 2 1080 3
        textSize = size[1] / 1000.0# 400 0.45 
        heightDeta = size[1] / 20 + 10# 400 20
        count = 0
        faces = []
        while True:
            count = count + 1
            ret, frame = camera.read() # 逐帧采集视频流
            if not ret:
                break
            detectCount = 0
      
            pipe.stdin.write(frame.tostring())  # 存入管道

            pass
        camera.release()
        print("Over!")
        pass

# 关闭监控识别
    def stop(self):
        pass

# 开启推送视频
    def openPush(self):
        self.ifRtmpPush = "1"
# 关闭推送视频
    def closePush(self):
        self.ifRtmpPush = "0"
 
    def toString(self):
        res = "" 

        return res


if __name__ == '__main__':
    serviceCamera = ServiceCamera(False)
    serviceCamera.start()
