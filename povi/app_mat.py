from app import App

from PyQt5 import uic
from PyQt5.QtWidgets import QToolBox


class MatApp(App):

    def __init__(self, args=[]):
        super(MatApp, self).__init__(args)
        self.dialog = ToolsDialog()

    def run(self):
        self.dialog.show()
        super(MatApp, self).run()



class ToolsDialog(QToolBox):
    def __init__(self, parent=None):
        super(ToolsDialog, self).__init__(parent)
        self.ui = uic.loadUi('povi/tools.ui', self)

        self.ui.slider_tcount.valueChanged.connect(self.slot_tcount)
        # import ipdb; ipdb.set_trace()        

    def slot_tcount(self, value):
        print 'tcount', value