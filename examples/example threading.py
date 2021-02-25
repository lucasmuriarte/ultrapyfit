# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:15:21 2020

@author: 79344
"""

from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QApplication,\
                            QLineEdit, QPushButton, QProgressBar
from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot

import time


#option 1
class GenericWorker(QObject):

    start = pyqtSignal(str)#create signals as the object does not have them
    finished = pyqtSignal()
#    finished = pyqtSignal(int)

    def __init__(self, function, *args, **kwargs):
        super(GenericWorker, self).__init__()
#        logthread('GenericWorker.__init__')
        print('enter')
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.start.connect(self.run)
        self.isRunning=False
    
    start = pyqtSignal(str)
    @pyqtSlot() #I dont get this decorator exactly but is important is controling the nulber of threads created 
    def run(self):
        print('start')
        self.isRunning=True
        self.function(*self.args, **self.kwargs)
        self.finished.emit()#emit the signal 
        #self.finished.emit(value) #emit the signal with a value that can be used later on
        self.isRunning=False

#option 2 run an specific function
#create an specific thread for every fuynction you want to run in athread
#class WorkingThread(QThread):
#    iter_num = pyqtSignal(int)
#    def __init__(self,val, parent=None):
#        super(WorkingThread, self).__init__(parent)
#        self.time=val
#    
#    @pyqtSlot()#maybe is not need here
#    def run(self):
#        longCalculation(self.time//2)
#        self.iter_num.emit(1)#signal we have reach half of calculation
#         longCalculation(self.time//2)
        
        
# same as option 2 
class ThreadMessage(QThread): 
    def __init__(self, parent=None):
        super(ThreadMessage, self).__init__(parent)
        
    def run(self):
        time.sleep(3)

def longCalculation(val):
    time.sleep(val)
    return True
            
class ModelWindow(QMainWindow):
    # Application window
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.centralwidget = QWidget(self)
        self.setGeometry(250, 250, 550, 250) 
        self.setCentralWidget(self.centralwidget)
        self.centralwidget.setLayout(QVBoxLayout(self.centralwidget))
        self.label = QLabel('1- set the calculations to run\n2- Click start to run the calculations',self)
        self.cal = QLineEdit('Number calculation',self)
        self.label.show()
        self.cal.show()
        self.label.setGeometry(85,30,200,25)
        self.cal.setGeometry(85,60,200,25)
        self.progressBar=QProgressBar(self)
        self.progressBar.setMaximum(0)
        self.progressBar.setMinimum(0)
        self.progressBar.setValue(0)
        self.setWindowTitle('We a re dooing really long calculation')
        self.start = QPushButton('Start calculation', self)
        self.start.clicked.connect(self.run)
        self.exit = QPushButton('Stop and closed', self)
        self.exit.clicked.connect(self.salir)
        self.progressBar.setGeometry(85,60+30,433,15)
        self.progressBar.setVisible(False)
        self.thread = QThread() #¸create a thread No need for option 2
        self.thread.start(QThread.TimeCriticalPriority) #start and define priority in the computer a thread
        self.exit.setGeometry(85,60+60,200,25)
        self.start.setGeometry(285,60+60,200,25)
        self.setFixedSize(570, 60+95)
        self.exito=False

    def run(self):
        #call the function it can be inside or outside the class is not a proble
        number=int(self.cal.text())
        self.label.setText(f'running {number} calculations')
        self.progressBar.show()
        my_worker = GenericWorker(longCalculation,number)
        my_worker.moveToThread(self.thread)
        my_worker.start.emit("hello") #calling starting signal will make my_worker.run() but we need to pass the signal
        my_worker.finished.connect(self.finished)

        #option 2
#        this thread is kill after calculation is finish
#        my_worker=WorkingThread(number)
#        my_worker.start() # make my_worker.run No need to pass the signal
#        my_worker.finished.connect(self.finished)
        
    def closeEvent(self, event):
        #remember to kill thread if we use option 1
        self.thread.terminate()
    
    def finished(self):
        print('done')
        self.exito=True
        self.progressBar.setVisible(False)
        self.cal.setVisible(False)
        self.label.setText(f'Calculations Done: {self.exito}\n\t\tGood Job')
        #tipe 2 threading 
        self.mes_t=ThreadMessage()
        self.mes_t.finished.connect(self.salir)
        self.mes_t.start()
        
    def salir(self):
        self.close()


app = QApplication([])
ex = ModelWindow()
ex.show()
app.exec()
  
