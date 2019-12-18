import cv2, dlib
import numpy as np
import sys, math
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from gesturemode import HandDetect
import time

class ShowVideo(QObject):

    flag = 0

    camera = cv2.VideoCapture(0)

    ret, image = camera.read()
    height, width = image.shape[:2]

    VideoSignal1 = pyqtSignal(QImage)
    VideoSignal2 = pyqtSignal(QImage)

    def __init__(self, parent=None):
        super(ShowVideo, self).__init__(parent)
        self.bridge = ImageViewer()
        self.count = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def overlay_transparent(self,background_img, img_to_overlay_t, x, y, overlay_size=None):
        bg_img = background_img.copy()
        # convert 3 channels to 4 channels
        if bg_img.shape[2] == 3:
          bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)
        slope = (self.shape_2d[0][1] - self.shape_2d[16][1]) / (self.shape_2d[0][0] - self.shape_2d[16][0])
        radian = math.atan(slope)
        degree = math.degrees(radian)
        height, width = img_to_overlay_t.shape[:2]
        M = cv2.getRotationMatrix2D((width/2, height/2), -degree, 1)
        img_rotation = cv2.warpAffine(img_to_overlay_t, M, (width, height))
        img_to_overlay_t = img_rotation

        if overlay_size is not None:
            img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

        b, g, r, a = cv2.split(img_to_overlay_t)

        mask = cv2.medianBlur(a, 5)

        h, w, _ = img_to_overlay_t.shape
        roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
        try:
            img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
            img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)
            bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

        # convert 4 channels to 4 channels
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)
        except:
            print("abraca_dabra")
        return bg_img

    def process(self, face, img):
        dlib_shape = self.bridge.predictor(img, face)

        self.shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

        top_left = np.min(self.shape_2d, axis=0)
        bottom_right = np.max(self.shape_2d, axis=0)

        face_size = int(max(bottom_right-top_left)*self.bridge.controlSize)
        center_x, center_y = np.mean(self.shape_2d, axis=0).astype(np.int)

        result = self.overlay_transparent(img, self.bridge.overlay, center_x, center_y-self.bridge.controlHeight, overlay_size=(face_size,face_size))

        img = cv2.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(), face.bottom()),color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)

        for s in self.shape_2d:
            cv2.circle(img, center=tuple(s), radius=1, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)

        cv2.circle(img, center=tuple(top_left), radius=1, color=(255,0,0), thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(img, center=tuple(bottom_right), radius=1, color=(255,0,0), thickness=2, lineType=cv2.LINE_AA)

        cv2.circle(img, center=tuple((center_x, center_y)), radius=1, color=(0,0,255), thickness=2, lineType=cv2.LINE_AA)
        #print(center_x, center_y)
        self.bridge.getImage(result)

        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        if self.bridge.makeup_num == 0:
            result = cv2.Canny(result, 100, 300)
        elif self.bridge.makeup_num == 1:
            result = cv2.bitwise_not(result)
        elif self.bridge.makeup_num == 2:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            result = cv2.Laplacian(result, -1, ksize = 17, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT)
        elif self.bridge.makeup_num == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            result = cv2.Sobel(result, -1, dx=1, dy=0, ksize=11, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        elif self.bridge.makeup_num == 4:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        else:
            pass

        if self.bridge.savenum == 0:
            cv2.putText(result, "CAPTURED!!", (250, 60), self.font, 1, (0, 0, 255), 2)
        else:
            pass
        qt_image1 = QImage(result.data,
                                result.shape[1],
                                result.shape[0],
                                result.shape[1] * 3,
                                QImage.Format_RGB888)

        self.bridge.setImage(qt_image1)
        self.VideoSignal2.emit(qt_image1)

    @pyqtSlot()
    def startVideo(self):
        global image
        global sticker_list

        run_video = True
        #cv2.putText(이미지, 텍스트, 위치, 폰트, 폰트 스케일, 빨간색, 두께)
        while run_video:
            if self.bridge.checkgesturemode:
                self.count += 1
            ret, image = self.camera.read()
            image = cv2.flip(image, 1)
            color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.count == 15:
                receive_num = self.bridge.sendImageToGestureMode(image)
                print(receive_num)
                self.count = 0


            if self.bridge.num == -1:
                qt_image1 = QImage(color_swapped_image.data,
                                        self.width,
                                        self.height,
                                        color_swapped_image.strides[0],
                                        QImage.Format_RGB888)
                self.VideoSignal1.emit(qt_image1)

            else:
                img = cv2.resize(image, (ShowVideo.width, ShowVideo.height))
                # ori = img.copy()
                #print(self.bridge.sticker_list[self.bridge.num])
                faces = self.bridge.detector(img)

                if len(faces) == 1:
                    face = faces[0]
                    try:
                        self.process(face,img)
                    except UnboundLocalError:
                        print("wait what?")

                elif len(faces) == 0:
                    print("No face")

                else:
                    print(str(len(faces)) + " face detected")
                    pass

            loop = QEventLoop()
            QTimer.singleShot(25, loop.quit) #25 ms
            loop.exec_()

class ImageViewer(QWidget):
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.image = QImage()
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.num = -1
        self.sticker_list = [['samples/beard.png', 0.8, -8], ['samples/ryan_transparent.png', 1.4, 22], ['samples/glasses1.png', 1.0, 35], ['samples/pig_nose.png', 0.4, 7], ['samples/shyshyshy4.png', 0.8, 13], ['samples/untitled.png', 0.1, 0]]
        self.scaler = 0.8

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.controlSize = self.sticker_list[self.num][1]
        self.controlHeight = self.sticker_list[self.num][2]
        self.overlay = cv2.imread(self.sticker_list[self.num][0], cv2.IMREAD_UNCHANGED)
        self.cap = None
        self.gesture = HandDetect()
        self.checkgesturemode = False
        self.savenum = -1
        self.makeup_num = -1

    def getImage(self, image):
        self.cap = image

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QImage()

    def initUI(self):
        self.setWindowTitle('Test')

    @pyqtSlot(QImage)
    def setImage(self, image):
        #print(str(type(image)) + " setImage")
        if image.isNull():
            print("Viewer Dropped frame!")
            return

        self.image = image
        if image.size() != self.size():
            self.setFixedSize(image.size())
        self.update()

    def stickerButtonClicked(self):
        sender = self.sender()
        #print(int(sender.text())-1)
        if (int(sender.text()) - 1) == self.num:
            self.num = 5
            self.overlay = cv2.imread(self.sticker_list[self.num][0], cv2.IMREAD_UNCHANGED)
            self.controlSize = self.sticker_list[self.num][1]
            self.controlHeight = self.sticker_list[self.num][2]
        else:
            self.num = int(sender.text())-1
            print(self.sticker_list[self.num])
            self.overlay = cv2.imread(self.sticker_list[self.num][0], cv2.IMREAD_UNCHANGED)
            self.controlSize = self.sticker_list[self.num][1]
            self.controlHeight = self.sticker_list[self.num][2]
        return int(sender.text())-1

    def makeUpButtonClicked(self):
        sender = self.sender()
        print(int(sender.text()) - 1)
        if self.makeup_num == (int(sender.text())-1):
            self.makeup_num = -1
        else:
            self.makeup_num = int(sender.text()) - 1
        return int(sender.text()) - 1

    def capture(self):
        # 파일명이 중복돼서 이전 사진 파일에 덮어씌워지는 일을 막기 위해 파일명을 사진이 찍힌 시간으로 정함
        cv2.imwrite('{}.png'.format(time.time()), self.cap, params=[cv2.IMWRITE_PNG_COMPRESSION, 0])

    def gesturemode(self):
        if self.checkgesturemode == False:
            self.checkgesturemode = True
        else:
            self.checkgesturemode = False
        """버튼모드로 돌아가는 버튼 추가하고 인터페이스 깔끔하게 하기"""

    def sendImageToGestureMode(self, image):
        num = self.gesture.handGesture(image)
        if num == -1:
            print("pass")
        else:
            if self.savenum != 0:
                if num == 0:
                    try:
                        self.capture()
                    except:
                        print("capture image not defined")
                    self.savenum = num
                else:
                    try:
                        self.savenum = num
                        self.num = num-1
                        self.overlay = cv2.imread(self.sticker_list[self.num][0], cv2.IMREAD_UNCHANGED)
                        self.controlSize = self.sticker_list[self.num][1]
                        self.controlHeight = self.sticker_list[self.num][2]
                    except TypeError:
                        self.num = self.savenum
            else:
                if num != 0:
                    self.savenum = num
                    try:
                        self.num = num-1
                        self.overlay = cv2.imread(self.sticker_list[self.num][0], cv2.IMREAD_UNCHANGED)
                        self.controlSize = self.sticker_list[self.num][1]
                        self.controlHeight = self.sticker_list[self.num][2]
                    except TypeError:
                        self.num = self.savenum
        return num


    def quit(self):
        sys.exit()
if __name__ == '__main__':
    app = QApplication(sys.argv)

    thread = QThread()
    thread.start()
    vid = ShowVideo()
    vid.moveToThread(thread)

    vid.VideoSignal1.connect(vid.bridge.setImage)
    vid.VideoSignal2.connect(vid.bridge.setImage)

    stickerLabel = QLabel("STICKER")
    stickerLabel.setAlignment(Qt.AlignCenter)

    stickerButton = [x for x in range(0, 5)]

    for i in range(5):
        stickerButton[i] = QPushButton()
        stickerButton[i].setText("{}".format(i + 1))
        stickerButton[i].setFixedSize(35, 35)

    stickerLabel.setStyleSheet('font-size:30px')

    # MakeUp Components
    makeUpLabel = QLabel("MAKE UP")
    makeUpLabel.setAlignment(Qt.AlignCenter)

    makeUpButton = [x for x in range(0, 5)]

    for i in range(5):
        makeUpButton[i] = QPushButton()
        makeUpButton[i].setText("{}".format(i + 1))
        makeUpButton[i].setFixedSize(35, 35)

    makeUpLabel.setStyleSheet('font-size:30px')

    # Quit Button
    quitButton = QPushButton()
    quitButton.setText("Quit")
    quitButton.setFixedSize(200, 35)

    # Capture Button
    captureButton = QPushButton()
    captureButton.setText("Capture")
    captureButton.setFixedSize(100, 100)

    # GestureMode Button
    gestureModeButton = QPushButton()
    gestureModeButton.setText("GestureMode")
    gestureModeButton.setFixedSize(200,50)

    # Button Clicked
    for i in range(5):
        stickerButton[i].clicked.connect(vid.bridge.stickerButtonClicked)
        makeUpButton[i].clicked.connect(vid.bridge.makeUpButtonClicked)
    captureButton.clicked.connect(vid.bridge.capture)
    quitButton.clicked.connect(vid.bridge.quit)
    gestureModeButton.clicked.connect(vid.bridge.gesturemode)

    # Layout Setting
    mainLayout = QHBoxLayout()

    vbox = QVBoxLayout()
    hbox = QHBoxLayout()
    tmp = QLabel()
    hbox.addWidget(tmp)

    stickerLayout = QHBoxLayout()
    for i in range(5):
        stickerLayout.addWidget(stickerButton[i])

    makeUpLayout = QHBoxLayout()
    for i in range(5):
        makeUpLayout.addWidget(makeUpButton[i])

    filterLayout = QVBoxLayout()
    filterLayout.addWidget(stickerLabel)
    filterLayout.addLayout(stickerLayout)
    filterLayout.addWidget(makeUpLabel)
    filterLayout.addLayout(makeUpLayout)

    vbox.addLayout(filterLayout)
    vbox.addWidget(captureButton, alignment=Qt.AlignCenter)
    vbox.addLayout(hbox)

    vbox.addWidget(gestureModeButton, alignment=Qt.AlignRight)
    vbox.addWidget(quitButton)

    mainLayout.addWidget(vid.bridge)

    mainLayout.addLayout(vbox)

    layout_widget = QWidget()
    layout_widget.setLayout(mainLayout)

    main_window = QMainWindow()

    main_window.setCentralWidget(layout_widget)
    main_window.show()
    vid.startVideo()
    sys.exit(app.exec_())
