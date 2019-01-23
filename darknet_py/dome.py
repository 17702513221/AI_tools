from OboardCamDisp import Ui_MainWindow
import sys
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog
from PyQt5.QtCore import QTimer,QCoreApplication
from PyQt5.QtGui import QPixmap
import cv2
import qimage2ndarray
import time
import darknet as dn
from PIL import Image, ImageDraw, ImageFont

class CamShow(QMainWindow,Ui_MainWindow):
    def __del__(self):
        try:
            self.camera.release()  # 释放资源
        except:
            return
    def __init__(self,parent=None):
        super(CamShow,self).__init__(parent)
        self.setupUi(self)
        self.PrepWidgets()
        self.PrepParameters()
        self.CallBackFunctions()
        self.Timer=QTimer()
        self.Timer.timeout.connect(self.TimerOutFun)
    def PrepWidgets(self):
        self.PrepCamera()
        self.StopBt.setEnabled(False)
        self.RecordBt.setEnabled(False)
        self.GrayImgCkB.setEnabled(True)
    def PrepCamera(self):
        try:
            self.camera=cv2.VideoCapture(0)
            self.MsgTE.clear()
            self.MsgTE.append('Oboard camera connected.')
            self.MsgTE.setPlainText()
        except Exception as e:
            self.MsgTE.clear()
            self.MsgTE.append(str(e))
    def SetNet(self):
        if self.GrayImgCkB.isChecked():
            self.net = dn.load_net("../darknet/cfg/yolov3-tiny.cfg".encode("utf-8"), "../darknet/yolov3-tiny.weights".encode("utf-8"), 0)
            self.meta = dn.load_meta("../darknet/cfg/coco.data".encode("utf-8"))
        else:
            self.net = dn.load_net("../darknet/cfg/yolov3.cfg".encode("utf-8"), "../darknet/yolov3.weights".encode("utf-8"), 0)
            self.meta = dn.load_meta("../darknet/cfg/coco.data".encode("utf-8"))

    def PrepParameters(self):
        self.RecordFlag=0
        self.RecordPath='/home/xs/softwares/darknet_py/'
        self.FilePathLE.setText(self.RecordPath)
        self.Image_num=0
        self.R=1
        self.G=1
        self.B=1       

        self.MsgTE.clear()
    def CallBackFunctions(self):
        self.FilePathBt.clicked.connect(self.SetFilePath)
        self.ShowBt.clicked.connect(self.StartCamera)
        self.StopBt.clicked.connect(self.StopCamera)
        self.RecordBt.clicked.connect(self.RecordCamera)
        self.ExitBt.clicked.connect(self.ExitApp)


    def StartCamera(self):
        self.ShowBt.setEnabled(False)
        self.StopBt.setEnabled(True)
        self.RecordBt.setEnabled(True)
        self.GrayImgCkB.setEnabled(True)
        self.RecordBt.setText('录像')
        self.SetNet()
        self.font = ImageFont.truetype('NotoSansCJK-Thin.ttc', 30)
        self.Timer.start(1)
        self.timelb=time.clock()
    def SetFilePath(self):
        dirname = QFileDialog.getExistingDirectory(self, "浏览", '.')
        if dirname:
            self.FilePathLE.setText(dirname)
            self.RecordPath=dirname+'/'
    def TimerOutFun(self):
        success,img=self.camera.read()
        if success:
            self.Image = self.ColorAdjust(img)
            self.DispImg()
            self.Image_num+=1
            if self.RecordFlag:
                self.video_writer.write(img)
            if self.Image_num%10==9:
                frame_rate=10/(time.clock()-self.timelb)
                self.FmRateLCD.display(frame_rate)
                self.timelb=time.clock()
                self.ImgWidthLCD.display(self.camera.get(3))
                self.ImgHeightLCD.display(self.camera.get(4))
        else:
            self.MsgTE.clear()
            self.MsgTE.setPlainText('Image obtaining failed.')
    def ColorAdjust(self,img):
        try:
            B=img[:,:,0]
            G=img[:,:,1]
            R=img[:,:,2]
            B=B*self.B
            G=G*self.G
            R=R*self.R

            img1=img
            img1[:,:,0]=B
            img1[:,:,1]=G
            img1[:,:,2]=R
            return img1
        except Exception as e:
            self.MsgTE.setPlainText(str(e))
    def DispImg(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2RGB)
        results = dn.detect_np(self.net, self.meta, img)
        img = dn.chinese_to_img(self.font,img,results) 
        qimg = qimage2ndarray.array2qimage(img)
        self.DispLb.setPixmap(QPixmap(qimg))
        self.DispLb.show()
    def StopCamera(self):
        if self.StopBt.text()=='暂停':
            self.StopBt.setText('继续')
            self.RecordBt.setText('保存')
            self.Timer.stop()
        elif self.StopBt.text()=='继续':
            self.StopBt.setText('暂停')
            self.RecordBt.setText('录像')
            self.Timer.start(1)
    def RecordCamera(self):
        tag=self.RecordBt.text()
        if tag=='保存':
            try:
                image_name=self.RecordPath+'image'+time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))+'.jpg'
                print(image_name)
                cv2.imwrite(image_name, self.Image)
                self.MsgTE.clear()
                self.MsgTE.setPlainText('Image saved.')
            except Exception as e:
                self.MsgTE.clear()
                self.MsgTE.setPlainText(str(e))
        elif tag=='录像':
            self.RecordBt.setText('停止')

            video_name = self.RecordPath + 'video' + time.strftime('%Y%m%d%H%M%S',time.localtime(time.time())) + '.avi'
            fps = self.FmRateLCD.value()
            size = (self.Image.shape[1],self.Image.shape[0])
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            self.video_writer = cv2.VideoWriter(video_name, fourcc,self.camera.get(5), size)
            self.RecordFlag=1
            self.MsgTE.setPlainText('Video recording...')
            self.StopBt.setEnabled(False)
            self.ExitBt.setEnabled(False)
        elif tag == '停止':
            self.RecordBt.setText('录像')
            self.video_writer.release()
            self.RecordFlag = 0
            self.MsgTE.setPlainText('Video saved.')
            self.StopBt.setEnabled(True)
            self.ExitBt.setEnabled(True)
    def ExitApp(self):
        self.camera.release()
        self.MsgTE.setPlainText('Exiting the application..')
        QCoreApplication.quit()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui=CamShow()
    ui.show()
    sys.exit(app.exec_())
