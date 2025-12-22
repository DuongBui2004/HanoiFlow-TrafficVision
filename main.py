import sys
from PyQt5 import QtWidgets, QtCore
from hanoidemo.gui.gui import HanoiFlowApp

def main():
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("HanoiFlow")

    window = HanoiFlowApp()
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

