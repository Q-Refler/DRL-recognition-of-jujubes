from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QPoint
from predict import *
from segmentaion import *
import mainwindow
import picture
import os

closeflag = 0  # 用于表示主窗口关闭的变量
stopflag = 0
runflag = 0

# savetext = ''
class Ui_MainWindow_3(mainwindow.Ui_MainWindow, QtWidgets.QWidget):

    def setupUi(self, MymainWindow):
        mainwindow2.Ui_MainWindow.setupUi(self, MymainWindow)

        # self.pushButton.clicked.connect(self.test)
        self.pushButton_2.clicked.connect(self.fun1)
        # self.pushButton_3.clicked.connect(self.fun2)
        self.pushButton_4.clicked.connect(self.alltest)
        self.pushButton_5.clicked.connect(self.stop)
        # self.pushButton_7.clicked.connect(self.txt)
        # self.checkBox.clicked.connect(self.picshow)
        self.reset_btn.clicked.connect(self.reset)
        self.pushButton_6.clicked.connect(self.terminate)

    def alltest(self):  # 用于“开始检测”
        global stopflag, closeflag, runflag
        # global stopflag, closeflag, runflag, savetext
        self.pushButton_4.setEnabled(0)
        self.pushButton_2.setEnabled(0)
        runflag = 1
        stopflag = 0

        # savetext = ''
        content1 = self.lineEdit.text()
        # if not os.path.exists(content1):
        #     savetext = savetext + '\n' + 'ERROR!!!'
        # else:
        for dirpath, dirnames, filenames in os.walk(content1):
            for filename in sorted(filenames, key=lambda x: x[:-4]):
                # pic_result = [0, 0, 0, 0]
                path1 = dirpath + '/' + filename
                pic_result = im_segmentation(path1)

                self.pic_1.setPixmap(QtGui.QPixmap(path2).scaled(self.pic_1.geometry().size()))

                # for pic in pic_list:
                #     result = AI_predict(pic)
                #     if result == 1:
                #         pic_result[0]+=1
                #     elif result == 2:
                #         pic_result[1]+=1
                #     elif result == 3:
                #         pic_result[2]+=1
                #     elif result == 0:
                #         pic_result[3]+=1

                self.num_3.setText(str(pic_result[0]))  # 干瘪
                self.num_2.setText(str(pic_result[1]))  # 合格
                self.num_5.setText(str(pic_result[2]))  # 腐烂
                self.num_4.setText(str(pic_result[3]))  # 变形
                self.num_1.setText(str(int(self.num_1.text()) + 1))  # 已检测
                QtWidgets.QApplication.processEvents()
                if stopflag == 1:
                    while True:
                        QtWidgets.QApplication.processEvents()
                        if (stopflag == 0) | (closeflag == 1):
                            break

                if (closeflag == 1) | (runflag == 0):
                    break
            if (closeflag == 1) | (runflag == 0):
                break
        self.pushButton_4.setEnabled(1)
        self.pushButton_2.setEnabled(1)

    def stop(self):
        global stopflag
        stopflag = not stopflag
        if stopflag == 0:
            self.pushButton_5.setText('暂停')
            self.pushButton_6.setEnabled(1)
        else:
            self.pushButton_5.setText('继续')
            self.pushButton_6.setEnabled(0)

    # def txt(self):  # 导出为txt
    #     global savetext
    #     file = savetext
    #     content = self.fun3()
    #     fw = open(content, 'w')
    #     fw.write(file)
    #     fw.close()

    # def test(self):  # 用于单张检测
    #     global savetext
    #     content2 = self.lineEdit_2.text()
    #     result = AI_predict(content2)
    #     if result == 1:
    #         self.num_3.setText(str(int(self.num_3.text()) + 1))
    #         self.pic_3.setPixmap(QtGui.QPixmap(content2))
    #         text = "Dried"
    #     elif result == 2:
    #         self.num_2.setText(str(int(self.num_2.text()) + 1))
    #         self.pic_4.setPixmap(QtGui.QPixmap(content2))
    #         text = "Qualified"
    #     elif result == 3:
    #         self.num_5.setText(str(int(self.num_5.text()) + 1))
    #         self.pic_1.setPixmap(QtGui.QPixmap(content2))
    #         text = "Rotten"
    #     elif result == 0:
    #         self.num_4.setText(str(int(self.num_4.text()) + 1))
    #         self.pic_2.setPixmap(QtGui.QPixmap(content2))
    #         text = "Deformed"
    #     else:
    #         text = "Error!!!"
    #     if savetext == '':
    #         savetext = os.path.basename(content2) + "  " + text
    #     else:
    #         savetext = savetext + '\n' + os.path.basename(content2) + "  " + text

    def fun1(self):
        # a = tkinter.filedialog.askdirectory()  # 选择目录，返回目录名
        file = QtWidgets.QFileDialog.getExistingDirectory(self, 'Open file', './')
        self.lineEdit.setText(file)

    # def fun2(self):
    #     # a = tkinter.filedialog.askopenfilename()  # 选择打开什么文件，返回文件名
    #     file = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', './')[0]
    #     self.lineEdit_2.setText(file)
    #     # picui.label.setPixmap(QtGui.QPixmap(file))

    # def fun3(self):
    #     file = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', 'record', 'Txt Files(*.txt)')[0]

    # return file

    # def picshow(self):
    #     if self.checkBox.isChecked() == 1:
    #         picture.show()
    #
    #     else:
    #         picture.hide()

    def reset(self):
        # global savetext
        self.num_1.setText('0')
        self.num_2.setText('0')
        self.num_3.setText('0')
        self.num_4.setText('0')
        self.num_5.setText('0')
        self.pic_1.setPixmap(QtGui.QPixmap())
        # self.pic_2.setPixmap(QtGui.QPixmap())
        # self.pic_3.setPixmap(QtGui.QPixmap())
        # self.pic_4.setPixmap(QtGui.QPixmap())
        # savetext = ''

    @staticmethod
    def terminate():
        global runflag
        runflag = 0

    def init(self):
        self.pic_1.setScaledContents(1)
        # self.pic_2.setScaledContents(1)
        # self.pic_3.setScaledContents(1)
        # self.pic_4.setScaledContents(1)


class Mymainwindow(QtWidgets.QMainWindow, QtWidgets.QWidget):
    def closeEvent(self, event):
        global closeflag
        reply = QtWidgets.QMessageBox.question(self, '退出程序', "真的要退出程序吗QAQ?",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            closeflag = 1
            # picture.close()
            event.accept()
        else:
            event.ignore()


class Mypicture(picture.Ui_picture, QtWidgets.QWidget):
    pass


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    MainWindow = Mymainwindow()
    ui = Ui_MainWindow_3()
    ui.setupUi(MainWindow)
    ui.init()
    # picture = QtWidgets.QWidget()
    # picui = Mypicture()
    # picui.setupUi(picture)

    desktop = QtWidgets.QApplication.desktop()
    pos = MainWindow.geometry()
    pos.moveCenter(QPoint(desktop.width() // 2, desktop.height() // 2))
    MainWindow.move(pos.topLeft())
    # picture.move(pos.topRight().x() + 10, pos.topRight().y())

    MainWindow.show()
    # picture.show()

    sys.exit(app.exec_())
