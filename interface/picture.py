# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'picture.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_picture(object):
    def setupUi(self, picture):
        picture.setObjectName("picture")
        picture.resize(700, 600)
        picture.setMinimumSize(QtCore.QSize(700, 600))
        self.gridLayout = QtWidgets.QGridLayout(picture)
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout.setContentsMargins(150, 150, 150, 150)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(picture)
        self.label.setEnabled(True)
        self.label.setText("")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.retranslateUi(picture)
        QtCore.QMetaObject.connectSlotsByName(picture)

    def retranslateUi(self, picture):
        _translate = QtCore.QCoreApplication.translate
        picture.setWindowTitle(_translate("picture", "图片预览"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    picture = QtWidgets.QWidget()
    ui = Ui_picture()
    ui.setupUi(picture)
    picture.show()
    sys.exit(app.exec_())
