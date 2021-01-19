from PyQt5.QtWidgets import QMainWindow,QWidget,QVBoxLayout,QMenuBar,QLabel,QApplication,\
QLineEdit,QPushButton,QComboBox,QTableWidget,QMdiArea,QProgressBar,QCheckBox,QAction,\
QSpinBox,QMdiSubWindow,QGridLayout,QSlider,QHBoxLayout,QFileDialog,QInputDialog,QTextEdit,\
QTableWidgetItem,QMessageBox,QButtonGroup,QRadioButton, QSplashScreen
from PyQt5.QtGui import QPixmap, QIcon, QBrush, QPainter,QColor #QFont
from  PyQt5.QtCore  import QThread, QObject, pyqtSignal, QPoint, QSize, QEvent, pyqtSlot, QCoreApplication, Qt #QTimer
import sys
app = QApplication(sys.argv) 
pixmap = QPixmap('C:/Program Files (x86)/Ultra PyFit/loading.png');
pixmap = pixmap.scaled(750,500)

splash = QSplashScreen(pixmap);
splash.show()


import numpy as np
import pandas as pd
from matplotlib.backends.qt_compat import is_pyqt5
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from scipy.special import erf
import re
import time
import pickle
from matplotlib.offsetbox import AnchoredText
from seaborn import distplot,kdeplot
from lmfit.models import GaussianModel, PolynomialModel
from lmfit import Parameters,minimize,fit_report,conf_interval
from mpl_toolkits.mplot3d import Axes3D
import os
from copy import deepcopy
from Global_Fit_Final_QT_recover2_rangeplot_TargetFit_weights import GlobalFit
from Calibration_Final_QT_2 import Calibration
from targetmodel import Model


experiment=GlobalFit(np.ones(10),np.ones([10,10]))

#experiment.conf_interval={}
#class PixmapContainer(QLabel):
#    def __init__(self, pixmap, parent=None):
#        super(PixmapContainer, self).__init__(parent)
#        self._pixmap = QtGui.QPixmap(pixmap)
#        self.setMinimumSize(1, 1)  # needed to be able to scale down the image
#
#    def resizeEvent(self, event):
#        w = min(self.width(), self._pixmap.width())
#        h = min(self.height(), self._pixmap.height())
#        self.setPixmap(self._pixmap.scaled(w, h, Qt.KeepAspectRatio))

#initialize_global_fit=False
#def errorHandling(func):
#    def wrapped(*args, **kwargs):
#        try:
#            return func(*args, **kwargs)
#        except:
#            if initialize_global_fit==False:
#                error_dialog = QMessageBox(self)
#                error_dialog.setText('Please load data first')
#                error_dialog.exec()
#            else:
#                error_dialog = QMessageBox(self)
#                error_dialog.setText('We could not complete the action')
#                error_dialog.exec()
#    return wrapped

#def logthread(caller):
#    print('%-25s: %s, %s,' % (caller, current_thread().name,
#                              current_thread().ident))

#class MDIArea(QMdiArea):
#    def __init__(self, background_pixmap, parent = None):
#        QMdiArea.__init__(self, parent)
#        self.background_pixmap = background_pixmap
#        self.centered = False
#  
#    def paintEvent(self, event):
#  
#        painter = QPainter()
#        painter.begin(self.viewport())
#          
#        if not self.centered:
#            painter.drawPixmap(0, 0, self.width(), self.height(), self.background_pixmap)
#        else:
#            painter.fillRect(event.rect(), self.palette().color(Qt.darkGray))
#            x = (self.width() - self.display_pixmap.width())/2
#            y = (self.height() - self.display_pixmap.height())/2
#            painter.drawPixmap(x, y, self.display_pixmap)
#         
#        painter.end()
#   
#    def resizeEvent(self, event):
#        self.display_pixmap = self.background_pixmap.scaled(event.size(), Qt.KeepAspectRatio)


class MDIArea(QMdiArea):
    def __init__(self, background_pixmap, parent = None):
        QMdiArea.__init__(self, parent)
        self.background_pixmap = background_pixmap
        self.centered = False
        brush=QBrush(QColor(85,140,100,62))
#        brush=QBrush((QColor(100,125,150,100)))
#        brush=QBrush((QColor(np.random.randint(10,150),np.random.randint(10,150),np.random.randint(10,150),100)))
#        brush=QBrush((QColor(50,150,80,50)))
        self.setBackground(brush)
    
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self.viewport())
        painter.setOpacity(0.30)
        self.paint(painter)
#        painter.end()
        
    def paint(self,painter):
#        painter.setBrush(QBrush(Qt.darkGray))
#        painter.drawRect(self.rect())
        x = (self.width() - self.background_pixmap.width())/2
        y = (self.height() - self.background_pixmap.height())/2
        painter.drawPixmap(x, y, self.background_pixmap)
#   
#    def resizeEvent(self, event):
#        self.display_pixmap = self.background_pixmap.scaled(event.size(), Qt.KeepAspectRatio)

class GenericWorker(QObject):

    start = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, function, *args, **kwargs):
        super(GenericWorker, self).__init__()
#        logthread('GenericWorker.__init__')
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.start.connect(self.run)
        self.isRunning=False
    
    start = pyqtSignal(str)
    @pyqtSlot()
    def run(self):
        print('start')
        self.isRunning=True
        self.function(*self.args, **self.kwargs)
        self.finished.emit()
        self.isRunning=False

class WorkingThread(QThread):
    iter_num = pyqtSignal(int)
    def __init__(self,maxfev,vary_taus,apply_weights, parent=None):
        super(WorkingThread, self).__init__(parent)
        self.maxfev=maxfev
        self.vary_taus=vary_taus
        self.apply_weights=apply_weights
#        self.pool=QThreadPool.globalInstance()
#        self.pool.setMaxThreadCount(7)
#    
    @pyqtSlot()
    def run(self):
        experiment.finalFit(self.vary_taus,self.maxfev,apply_weights=self.apply_weights)
        print('done')
        
class threadItter(QThread):
    iter_num = pyqtSignal(int)
    def __init__(self, parent=None):
        super(threadItter, self).__init__(parent)
        self.final_message=False
    
    @pyqtSlot()
    def run(self):
        experiment.fit_completed=False
        if ex.my_thread_Fit.isRunning: 
            time.sleep(1)
            if experiment.SVD_fit==False:
                while experiment.fit_completed==False:
                    prev_val=experiment.number_it
                    time.sleep(1)
                    val=experiment.number_it
                    if prev_val==val:
                        time.sleep(4)
                        val=experiment.number_it
                        if  prev_val==val and val>1000:
                            if self.final_message==False:
                                self.iter_num.emit(-3)
                                self.final_message=True
                            else:
                                pass
                        else:
                            if self.final_message==False:
                                self.iter_num.emit(-2)
                    else:
                       if self.final_message==False:
                            self.iter_num.emit(val)
                else:
                    val=-1
                    if self.final_message==False:
                          self.iter_num.emit(val)
                
            else:
                val=-1
                if self.final_message==False:
                      self.iter_num.emit(-4)
        else:
           if self.final_message==False:
                self.iter_num.emit(-2)

class ThreadMessage(QThread): 
    def __init__(self, parent=None):
        super(ThreadMessage, self).__init__(parent)
        
    def run(self):
        time.sleep(3)
        
class Grapth(QWidget):
    def __init__(self, figure, parent=None, toolbar=True, **kwargs):
        cursor=dict({'cursor':False,'ax':None,'y':True,'click':True,'number_click':-1,'vertical_draw':True,'draw':'snap','color':False},**kwargs)
        super(Grapth, self).__init__(parent)
        self.figure = figure
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        self.toolbar = NavigationToolbar(self.canvas, self)
        if toolbar:    
            layout.addWidget(self.toolbar)
        self.cursor_dict=cursor
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        if cursor['cursor']:
           if cursor['ax'] is None:
               self.ax=plt.gca()
           else:
               self.ax=cursor['ax']    
           line = self.ax.lines[0]
           x=line.get_xdata()
           if cursor['y']:
                y=line.get_ydata()
           else:
               y=x*0.0
           self.cursore = SnaptoCursor(self.ax,x,y,number_click=cursor['number_click'],vertical_draw=cursor['vertical_draw'],draw=cursor['draw'],color=cursor['color'])
           def onmove(event):
               self.cursore.mouseMove(event)
           def onclick(event):
               if self.toolbar._active is None and cursor['click']:
                   self.cursore.onClick(event)
           def onenter(event):
               self.cursore.onEnterAxes(event)
           def onleave(event):
               self.cursore.onLeaveAxes(event)
           self.canvas.mpl_connect('axes_enter_event', onenter)
           self.canvas.mpl_connect('axes_leave_event', onleave)
           self.canvas.mpl_connect('motion_notify_event', onmove)
           self.canvas.mpl_connect('button_press_event', onclick)
        self.canvas.draw()
    def cursorData(self):
        if self.cursor_dict['cursor'] and self.cursor_dict['y']:
            return list(set(self.cursore.datax)),list(set(self.cursore.datay))
        elif self.cursor_dict['cursor']:
            return list(set(self.cursore.datax))
        else:pass        

class SnaptoCursor(object):
    def __init__(self, ax,x, y,number_click=-1,vertical_draw=True,draw='snap',color=False,single_line=True):
        if number_click==-1:
            self.number_click=np.inf
        else:
            self.number_click=number_click
        self.ax = ax
        self.draw=draw
        self.vertical_draw=vertical_draw
        self.x = x
        self.y = y
        self.similar = y==np.zeros(len(y))
        self.color=color
        self.datax=[]
        self.datay=[]
        self.scat=[]
        self.single_line=single_line
    def mouseMove(self, event):
        if not event.inaxes: return
        self.x_pos, self.y_pos = event.xdata, event.ydata
        if self.single_line:
            indx = np.searchsorted(self.x, [self.x_pos])[0]
            x = self.x[indx]
            y = self.y[indx]
        else:
            x = self.x_pos
            y = self.y_pos
        self.ly.set_xdata(x)
        self.marker.set_data([x],[y])
        if abs(x)>=0.1:
            texto_x=1
        else:
            try:
                texto_x=[True if i=='0' else False for i in str(x).split('.')[1]].index(False)+1
            except:
                texto_x=3
        if abs(y)>=0.1:
            texto_y=1
        else:
            try:
                texto_y=[True if i=='0' else False for i in str(y).split('.')[1]].index(False)+1
            except:
                texto_y=3
        if self.similar.all()==False:
            self.lx.set_ydata(y)
            self.txt.set_text('x='+str(round(x,texto_x))+', y='+str(round(y,texto_y)))
            self.txt.set_position((x,y))
        else:
            self.txt.set_text('x=' +str(round(x,texto_x)))
            self.txt.set_position((x,y+0.0005))
        self.ax.figure.canvas.draw_idle()
    
    def onClick(self,event):
        if not event.inaxes: return
        if event._button_svd_select==1:
            #print(self.number_click)
            if len(self.datax)<self.number_click:
                x,y=event.xdata,event.ydata
                if self.draw=='snap':   
                    indx = np.searchsorted(self.x, [x])[0]
                    x = self.x[indx]
                    y = self.y[indx]
                self.datax.append(x)
                self.datay.append(y)
#                print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
#                      ('double' if event.dblclick else 'single', event.button,
#                       event.x, event.y, event.xdata, event.ydata))
                if self.vertical_draw:
                    self.scat.append(self.ax.axvline(self.datax[-1],alpha=0.5,color='red',zorder=np.inf))
                else:
                    self.scat.append(self.ax.scatter(self.datax,self.datay, color='red',marker='x',zorder=np.inf))
            else:
                pass
            self.ax.figure.canvas.draw_idle()
        elif event._button_svd_select==3:
            if len(self.datax)==0:
                pass
            else:
                del self.datax[-1]
                del self.datay[-1]
                self.scat[-1].remove()
                del self.scat[-1]
                self.ax.figure.canvas.draw_idle()  
    
    def onEnterAxes(self,event):
        if not event.inaxes: return
        try:
            self.onLeaveAxes(event)
        except:
            pass
        if self.similar.all()==False:
            self.lx = self.ax.axhline(color='k',alpha=0.2)  # the horiz line
        if self.single_line:
            try:
                line=self.ax.lines[0]
                self.x=line.get_xdata()
                if self.similar.all()==False:
                    self.y=line.get_ydata()
            except:
                pass
        self.ly = self.ax.axvline(color='k', alpha=0.2)  # the vert line
        self.marker, = self.ax.plot([0],[0], marker="o", color="crimson", zorder=3)
        self.txt = self.ax.text(0.7, 0.9, '')
        if self.color is not False:
            event.inaxes.patch.set_facecolor(self.color)
        event.canvas.draw()
    
    def onLeaveAxes(self,event):
        if not event.inaxes: return
        #print ('leave_axes', event.inaxes)
        self.marker.remove()
        self.ly.remove()
        self.txt.remove()
        if self.similar.all()==False:
            self.lx.remove()
        event.inaxes.patch.set_facecolor('white')
        event.canvas.draw()

class MainWindowFtest(QMainWindow):
    # QMainWindow closed signal
    closed = pyqtSignal(int)
    def __init__(self,fit_number, parent=None):
        QMainWindow.__init__(self, parent)
        self.centralwidget = QWidget(self)
        self.setGeometry(250, 250, 550, 250) 
        self.setCentralWidget(self.centralwidget)
        self.centralwidget.setLayout(QVBoxLayout(self.centralwidget))
        self.setWindowIcon_path='C:/Program Files (x86)/Ultra PyFit/Symbol.png'
        self.setWindowIcon(QIcon(self.setWindowIcon_path))
        self.title='Ultra PyFit Confidence interval by F-Test'
        self.setWindowTitle(self.title)
        self.parent = parent
        self.label = QLabel('Please select the parameter which you want to calculate the Confidence intervals', self)
        self.label.setGeometry(50,10,450,30)
        self.resultados=experiment.all_fit[fit_number][3]
        if experiment.all_fit[fit_number][8][2] == 'Exponential':
            if experiment.all_fit[fit_number][5]:
                params=['t0_1','fwhm_1']+['tau%i_1'%(i+1) for i in range(experiment.all_fit[fit_number][4])]
            else:
                params=['t0_1']+['tau%i_1'%(i+1) for i in range(experiment.all_fit[fit_number][4])]
            self.params=[i for i in params if self.resultados.params[i].vary]
            self.names=[i.split('_')[0] for i in self.params]
        else:
            if experiment.all_fit[fit_number][5]:
                params=['t0_1','fwhm_1']+['k_%i%i' % (i+1,i+1) for i in range(experiment.all_fit[fit_number][4])]

            else:
                params=['t0_1']+['k_%i%i' % (i+1,i+1) for i in range(experiment.all_fit[fit_number][4])]
            self.params=[i for i in params if self.resultados.params[i].vary]
            self.names=[i.split('_')[0] if 'k' not in i else i for i in self.params]
#            self.names=[i.split('_')[0] if 'k' not in i else 'tau'+i.split('_')[1][0] for i in self.params]
        self.group=QButtonGroup(self)
        self.group.setExclusive(False)
        self.tick={}
        pos=20
        for ii,i in enumerate(self.names):
            self.tick[i]=QCheckBox(i,self)
            if ii%2==0:
                pos+=30
                self.tick[i].setGeometry(150,pos,100,30)
            else:
                self.tick[i].setGeometry(250,pos,100,30)
            self.tick[i].show()
            self.group.addButton(self.tick[i])
        self.centralwidget.setContentsMargins(50,10,50,0)
        self.progressBar=QProgressBar(self)
        self.progressBar.setMaximum(0)
        self.progressBar.setMinimum(0)
        self.progressBar.setValue(0)
        self.progressBar.setGeometry(50,pos+30,433,15)
        self.progressBar.setVisible(False)
        self.start = QPushButton('Start calculation', self)
        self.start.clicked.connect(self.run)
        self.exit = QPushButton('Stop and closed', self)
        self.exit.clicked.connect(self.salir)
        self.exit.setGeometry(50,pos+60,200,25)
        self.start.setGeometry(250,pos+60,200,25)
        self.setFixedSize(500, pos+95)
        self.exito=False
        self.running=False
        self.thread = QThread()
        self.thread.start(QThread.TimeCriticalPriority)
        self.fit_number=fit_number
 
    def closeEvent(self, event):
        self.thread.terminate()
        if  self.exito:
            val=1
        else:
            val=-1
        self.closed.emit(val)
    
    def run(self):
        names=[self.params[i] for i,ii in enumerate(self.group.buttons()) if ii.isChecked()]
        if self.running==False and len(names)>=1 :
            self.label.setText('Calculating Confidence interval by F-Test; Please do not close the window!')
            self.running=True
            self.progressBar.show()
            print(names)           
            my_worker = GenericWorker(confinterval,self.fit_number,names)
            my_worker.moveToThread(self.thread)
            my_worker.start.emit("hello")  
            my_worker.finished.connect(self.finished)
        else:
            pass
    
    def finished(self):
        self.exito=True
        self.salir()
        pass
        
    def salir(self):
        self.close()

def confinterval(fit_number,params):
    if experiment.all_fit[fit_number][9]:#check for singular vector fit
        data=experiment.all_fit[fit_number][10]
        wavelength=None
    else:
        data=experiment.all_fit[fit_number][1]
        wavelength=experiment.all_fit[fit_number][2]
    if experiment.all_fit[fit_number][7]==False:#check if the data is derivate
            experiment2=GlobalFit(x=experiment.all_fit[fit_number][0],data=data,\
                                  wavelength=wavelength,exp_no=experiment.all_fit[fit_number][4],\
                                  deconv=experiment.all_fit[fit_number][5])
    else:
        experiment2=GlobalFit(x=experiment.all_fit[fit_number][0],data=data,\
                              wavelength=wavelength,exp_no=experiment.all_fit[fit_number][4],\
                              deconv=experiment.all_fit[fit_number][5],derivate=True,window_length=experiment.all_fit[fit_number][7]['window_length'],\
                              polyorder=experiment.all_fit[fit_number][7]['polyorder'],deriv=experiment.all_fit[fit_number][7]['deriv'],\
                              done=experiment.all_fit[fit_number][7]['done'])
    experiment2.data_before_last_Fit=data
    experiment2.tau_inf=experiment.all_fit[fit_number][6]
    experiment2.type_fit=experiment.all_fit[fit_number][8][2]
    print(experiment2.type_fit)
    if type(params[0]) is int:#check if we calculate bootstrap or f-test
#        print('boot')
        experiment2.all_fit=experiment.all_fit
        experiment2.prefit_done=True
        experiment2.residuesBootStrap(fit_number,params[0],params[1])
        if fit_number in experiment.bootstrap_residues_record.keys():
            first=experiment.bootstrap_residues_record[fit_number][1]
            all_boot=pd.concat([first,experiment2.bootstrap_residues_record[fit_number][1].iloc[1:,:]])
            experiment.bootstrap_residues_record[fit_number][1]=all_boot
            experiment.bootConfInterval(fit_number)
            experiment.general_report['Sequence of actions'].append(f'\t--> More data sets added to Bootstrap for fit {fit_number}')
        else:    
            experiment.bootstrap_residues_record[fit_number]=[params[1],experiment2.residues_boostrap_result]
            experiment.bootConfInterval(fit_number)
            experiment.general_report['Sequence of actions'].append(f'\t--> Confidence interval by Bootstrap for fit {fit_number}')
            experiment.general_report['Fits done'][f'Fit number {fit_number}']=experiment.general_report['Fits done'][f'Fit number {fit_number}']+'\n\t\tConfidence interbal calculated by Bootstrap'
        
    else:
        resultados=experiment.all_fit[fit_number][3]
        if resultados.params[params[0]].stderr is None:
            for p in resultados.params:
                resultados.params[p].stderr = abs(resultados.params[p].value * 0.1)
        print('hola')
        ci,traces=conf_interval(experiment2, resultados,p_names=params, trace=True)
        experiment.conf_interval[fit_number]=(ci,traces)
        experiment.general_report['Sequence of actions'].append(f'\t--> Confidence interval by F-test for fit {fit_number}')
        experiment.general_report['Fits done'][f'Fit number {fit_number}']=experiment.general_report['Fits done'][f'Fit number {fit_number}']+'\n\t\tConfidence interbal calculated by F-test'


class MainWindowBoot(QMainWindow):
    # QMainWindow doesn't have a closed signal, so we'll make one.
    closed = pyqtSignal(int)

    def __init__(self,fit_number, already, parent=None):
        QMainWindow.__init__(self, parent)
        self.parent = parent
        self.already=already
        self.centralwidget = QWidget(self)
        self.setGeometry(250, 250, 750, 250) 
        self.setCentralWidget(self.centralwidget)
        self.centralwidget.setLayout(QVBoxLayout(self.centralwidget))
        self.setWindowIcon_path='C:/Program Files (x86)/Ultra PyFit/Symbol.png'
        self.setWindowIcon(QIcon(self.setWindowIcon_path))
        self.title='Ultra PyFit Confidence interval by Bootstrap'
        self.setWindowTitle(self.title)
        self.parent = parent
        self.label = QLabel('The bootstrap fits a number of data sets generated from the previous fit results and the residues.\n\n\t1--> Select number of generate data sets to be fitted \n\tAdvice: Run a small number, check the results and run more later\n\t2--> select the percentage of random replacement for the data sets ', self)
        self.label.setGeometry(50,10,470,70)
        self.replacement=['10 %','15 %','20 %','25 %','33 %']
        self.boots=['50','100','250','500','1000']
        self.group_boot=QButtonGroup(self)
        self.tick={}
        label1 = QLabel('N° Data sets:',self)
        label2 = QLabel('Percentage:',self)
        label1 .setGeometry(105,150,100,30)
        label2 .setGeometry(310,150,100,30)
        pos=60
        for ii,i in enumerate(self.boots):
            self.tick[i]=QCheckBox(i,self)
            pos+=30
            self.tick[i].setGeometry(185,pos,100,30)
            self.tick[i].show()
            self.group_boot.addButton(self.tick[i])
        if self.already:
            self.replace_used= experiment.bootstrap_residues_record[fit_number][0]
            label=QLabel(f'previously used: {self.replace_used} %',self)
            label.setGeometry(385,150,120,30)
        else:
            self.group_replacement=QButtonGroup(self)
            pos=60    
            for ii,i in enumerate(self.replacement):
                self.tick[i]=QCheckBox(i,self)
                pos+=30
                self.tick[i].setGeometry(385,pos,100,30)
                self.tick[i].show()
                self.group_replacement.addButton(self.tick[i]) 
            self.tick['20 %'].setCheckState(2)
            self.tick['100'].setCheckState(2)
        self.centralwidget.setContentsMargins(50,10,50,0)
        self.progressBar=QProgressBar(self)
        self.progressBar.setMaximum(0)
        self.progressBar.setMinimum(0)
        self.progressBar.setValue(0)
        self.progressBar.setGeometry(85,pos+30,433,15)
        self.progressBar.setVisible(False)
        self.start = QPushButton('Start calculation', self)
        self.start.clicked.connect(self.run)
        self.exit = QPushButton('Stop and closed', self)
        self.exit.clicked.connect(self.salir)
        self.exit.setGeometry(85,pos+60,200,25)
        self.start.setGeometry(285,pos+60,200,25)
        self.setFixedSize(570, pos+95)
        self.exito=False
        self.running=False
        self.thread = QThread()
        self.thread.start(QThread.TimeCriticalPriority)
        self.fit_number=fit_number      

    def closeEvent(self, event):
        self.thread.terminate()
        if  self.exito:
            val=1
        else:
            val=-1
        self.closed.emit(val)
    
    def run(self):
        if self.already:
            replacement=self.replace_used
        else:
            replacement=[int(self.replacement[i].split(' ')[0]) for i,ii in enumerate(self.group_replacement.buttons()) if ii.isChecked()][0]
        boots=[int(self.boots[i].split(' ')[0]) for i,ii in enumerate(self.group_boot.buttons()) if ii.isChecked()][0]
        if self.running==False:
            self.label.setText(f'\tCalculating Confidence interval by bootstrap; number of boots={boots}\n\n\t\t\tPlease do not close the window!')
            self.running=True
            self.progressBar.show()
            print(replacement,boots)           
            my_worker = GenericWorker(confinterval,self.fit_number,[boots,replacement])
            my_worker.moveToThread(self.thread)
            my_worker.start.emit("hello")  
            my_worker.finished.connect(self.finished)
        else:
            pass
    
    def finished(self):
        self.exito=True
        self.salir()
        pass
        
    def salir(self):
        self.close()

class ModelWindow(QMainWindow):

    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.centralwidget = QWidget(self)
        self.setCentralWidget(self.centralwidget)
        self.centralwidget.setLayout(QVBoxLayout(self.centralwidget))
#        self.setWindowIcon_path='C:/Users/lucas/Desktop/LAST ULTRAPYFIT/Symbol.png'
        self.setWindowIcon_path='C:/Program Files (x86)/Ultra PyFit/Symbol.png'
#        self.setWindowIcon_path='//pclasdocnew.univ-lille.fr/Doc/79344/Documents/ULtraPyFit last files/Symbol.png'
        self.pixmapShow = QPixmap(self.setWindowIcon_path).scaled(200,200, aspectRatioMode=Qt.KeepAspectRatio,transformMode=Qt.SmoothTransformation)
        self.setWindowIcon(QIcon(self.setWindowIcon_path))
#        self.mdiArea = QMdiArea(self.centralwidget)
        pix=QPixmap(self.setWindowIcon_path).scaled(375,375, aspectRatioMode=Qt.KeepAspectRatio,transformMode=Qt.SmoothTransformation)
#        brush=QBrush(Qt.darkGray)        
#        brush.setStyle(Qt.SolidPattern)
#        brush=QBrush(QColor(80,80,140,50))#azul claro
#        brush=QBrush(QColor(140,80,80,50))#rojo claro
#        brush=QBrush(QColor(85,140,100,62))#verde claro
#        ex.mdiArea.setBackground(brush) 
        self.mdiArea = MDIArea(pix,self.centralwidget)
        self.centralwidget.layout().addWidget(self.mdiArea)
        
        
        self.window_height=QApplication.desktop().screenGeometry().height()
        self.window_width=QApplication.desktop().screenGeometry().width()
        if self.window_height <700:
            self.a=55
            self.down_shith=0
        elif self.window_height >800:
            self.a=-70
            self.down_shith=75
        else:
            self.down_shith=0
            self.a=0
        self.val = 150
        self.val_width = self.val-10
        self.title = 'Ultra PyFit'
        self.left = 10
        self.top = 35
        self.width = 1000
        self.height = 800

        self.setWindowTitle(self.title) #window header
        #self.setGeometry(self.left, self.top, self.width, self.height) #location of your window on the screen
        self.show()
#        self.mdiArea.setBackground(QImage('C:/Program Files (x86)/Ultra PyFit/Symbol.png'))
        #action menu
        self.bar = QMenuBar(self)
        self.bar.addMenu('&File Menu')
        self.load_data = QAction("Load data",self)
        self.load_data.triggered.connect(self.loadDataFunc)
        self.bar.addAction( self.load_data) 
        self.report = QAction("Display report",self)
        self.report.triggered.connect(self. printGeneralReport)
        self.bar.show()
        
        #self.bar.addAction("Indicate Working directory")
        
        self.load_Experiment = QAction("Load Experiment",self)
        self.load_Experiment.triggered.connect(self.loadExperimentFunc)
        self.bar.addAction(self.load_Experiment)
        self.save_as = QAction("Save as",self)
        self.save_as.triggered.connect(self.saveasFunc)
        self.bar.addAction( self.save_as)
        self.save = QAction("Save",self)
        self.save.triggered.connect(self.saveFunc)
        self.bar.addAction( self.save)  
        self.export = self.bar.addMenu("Export")
        self.save_folder = QAction("Create a saving folder",self)
        self.save_folder.triggered.connect(self.makeSaveDir)
        self.saveF = QAction("Fits",self)
        self.saveD = QAction("Data",self)
        self.saveF.triggered.connect(lambda: self.button61Func(export=True))
        self.saveD.triggered.connect(self.exportWindowD)
        self.export.addAction(self.saveD) 
        self.export.addAction(self.saveF) 
        self.export.addAction(self.save_folder) 
        self.exit=QAction("Exit",self)
        self.exit.setShortcut("Ctrl+Q")
        self.exit.setStatusTip('Leave The App')
        self.exit.triggered.connect(self.exitFunc)
        self.bar.addAction( self.report) 
        self.bar.addAction(self.exit)
        
        #Plotting Menu
        self.bar2 = QMenuBar(self)
        self.bar2.addMenu('&Plot Menu')
        self.bar2.show()
        
        self.explore_Data = QAction("Explore data",self)
        self.explore_Data.triggered.connect(self.exploreDataFunc)
        self.bar2.addAction( self.explore_Data)
        
        self.plot_3D = QAction("Plot 3D surface",self)
        self.plot_3D.triggered.connect(self.plot3DFunc)
        self.bar2.addAction( self.plot_3D)
        
        self.PlotTraces = QAction("Plot traces",self)
        self.PlotTraces.triggered.connect(self.PlotTracesFunc)
        self.bar2.addAction( self.PlotTraces)
        self.Plot10Traces = QAction("Plot 10 separated traces",self)
        self.Plot10Traces.triggered.connect(self.Plot10TracesFuncProcess)
        self.bar2.addAction( self.Plot10Traces)
        self.PlotSVD = QAction("Plot Singular Values",self)
        self.PlotSVD.triggered.connect(self.PlotSVDFunc)
        self.bar2.addAction( self.PlotSVD)
        self.Plot_Full_SVD = QAction("Plot Full SVD",self)
        self.Plot_Full_SVD.triggered.connect(lambda : self.plotFullSVD())
        self.bar2.addAction( self.Plot_Full_SVD)
        self.plots_pectra_menu=self.bar2.addMenu("Plot Spectra")
        self.plotsSpecAll = QAction("All",self)
        self.plotsSpecAll.triggered.connect(self.PlotSpecAllFunc)
        self.plots_pectra_menu.addAction( self.plotsSpecAll)
        self.plotsSpecAuto = QAction("Auto",self)
        self.plotsSpecAuto.triggered.connect(self.PlotSpecAutoFunc)
        self.plots_pectra_menu.addAction( self.plotsSpecAuto)
        self.plotsSpecManual = QAction("Manual",self)
        self.plotsSpecManual.triggered.connect(self.PlotSpecManualFunc)
        self.plots_pectra_menu.addAction( self.plotsSpecManual)
        self.colorPlotChange= QAction("Change color",self)
        self.colorPlotChange.triggered.connect(self.colorPlotChangeFunc)
        self.plots_pectra_menu.addAction( self.colorPlotChange)
        
        self.label04 = QLabel('Average N° points:', self)
        self.text04 = QLineEdit('0', self)
        self.label14 = QLabel('Select times:', self)
        self.text14 = QLineEdit('', self)
        self.label24 = QLabel('Legend columns:', self)
        self.text24 = QLineEdit('False/or a number', self)
        self.button04 = QPushButton('Plot spectra', self)
        self.button04.clicked.connect(self.button04Func)
        self.plotsDAS = QAction("Plot DAS",self)
        self.plotsDAS.triggered.connect(lambda:self.PlotDASFunc())
        self.bar2.addAction( self.plotsDAS)
        self.plotFit = QAction("Plot Fit",self)
        self.plotFit.triggered.connect(lambda:self.button71Func())
        self.bar2.addAction( self.plotFit)
        #self.bar2.addAction("Plot Bootstrap results (still not active)")
        
        self.bar3 = QMenuBar(self)
        self.bar3.addMenu('&Datasets')
        self.o_data = QAction("Original Data",self)
        self.o_data.setShortcut("Ctrl+O")
        self.o_data.triggered.connect(self.originalDataFunc)
        self.bar3.addAction( self.o_data)
        self.b_data = QAction("Data before baseline substraction",self)
        self.b_data.setShortcut("Ctrl+B")
        self.b_data.triggered.connect(self.baselineDataFunc)
        self.bar3.addAction( self.b_data)
        self.g_data = QAction("Data before GVD Correction",self)
        self.g_data.setShortcut("Ctrl+G")
        self.g_data.triggered.connect(self.GVDDataFunc)
        self.bar3.addAction( self.g_data)
        self.d_data = QAction("Data before derivation",self)
        self.d_data.setShortcut("Ctrl+D") 
        self.d_data.triggered.connect(self.derivateDataFunc)
        self.bar3.addAction( self.d_data)
        self.w_data = QAction("Data before Cutting/Selecting wavelenght",self)
        self.w_data.setShortcut("Ctrl+W") 
        self.w_data.triggered.connect(self.cutDataFunc)
        self.bar3.addAction(self.w_data)
        self.t_data = QAction("Data before Cutting/Selecting Time",self)
        self.t_data.setShortcut("Ctrl+T") 
        self.t_data.triggered.connect(self.timeDataFunc)
        self.bar3.addAction( self.t_data)
        self.tp_data = QAction("Data before average time points",self)
        self.tp_data.setShortcut("Ctrl+L")
        self.tp_data.triggered.connect(self.timePointDataFunc)
        self.bar3.addAction( self.tp_data)
        self.dp_data = QAction("Data before deleting points",self)
        self.dp_data.setShortcut("Ctrl+P")
        self.dp_data.triggered.connect(self.delPointDataFunc)
        self.bar3.addAction( self.dp_data)
        self.bar3.show()
        
        self.bar4 = QMenuBar(self)
        self.bar4.addMenu('&Preprocess')
        self.select_traces_submenu=self.bar4.addMenu('Select Traces')
        self.calibration_menu=self.bar4.addMenu("Calibrate")
        self.GVD=self.bar4.addMenu("GVD Correction")
        
        self.IRF=self.bar4.addMenu("IRF fitting")
        self.Raman_peak = QAction("From Raman peak in data",self)
        self.Raman_peak.triggered.connect(self.IRFRamanFunc)
        self.IRF.addAction(self.Raman_peak)
        
        self.solvent_peak = QAction("From Solvent measurement",self)
        self.solvent_peak.triggered.connect(self.IRFsolventFunc)
        self.IRF.addAction(self.solvent_peak)
                
        self.baseline_subtration = QAction("Baseline subtration",self)
        self.baseline_subtration.triggered.connect(self.baselineSubtrationFunc)
        self.bar4.addAction(self.baseline_subtration) 
        
        self.shift_time = QAction("Shift time",self)
        self.shift_time.triggered.connect(self.shiftTimeFunc)
        self.bar4.addAction(self.shift_time)
        
        self.label_t8 = QLabel('Time shift:', self)
        self.text_t8 = QLineEdit('0', self)
        self.button_t8 = QPushButton('Shitf time', self)
        self.button_t8.clicked.connect(self.button_t8Func)
        
        self.GVD_already = QAction("GVD already corrected",self)
        self.GVD_already.triggered.connect(self.GVDAlereadyFunc)
        self.GVD.addAction(self.GVD_already) 
        self.GVD_grapth_sell = QAction("From Grapth using Sellmeiller equation",self)
        self.GVD_grapth_sell.triggered.connect(self.GVDGrapthSellFunc)
        self.GVD.addAction(self.GVD_grapth_sell)
        self.GVD_grapth_pol= QAction("From Grapth fitting a polynom",self)
        self.GVD.addAction(self.GVD_grapth_pol)
        self.GVD_grapth_pol.triggered.connect(self.GVDGrapthPolFunc)
        self.GVD_sell = QAction("Using Sellmeiller equation",self)
        self.GVD_sell.triggered.connect(self.GVDSellFunc)
        self.GVD.addAction(self.GVD_sell)
        #self.GVD.addAction("Auto")
        self.cut_regions = QAction("Wavelenthgs regions",self)
        self.cut_regions.triggered.connect(self.cutRegionsFunc)
        self.cut_time = QAction("Time regions",self)
        self.cut_time.triggered.connect(self.cutTimeFunc)
        self.cut_region=self.bar4.addMenu('Cut/Select data regions')
        self.cut_region.addAction(self.cut_time)
        self.cut_region.addAction(self.cut_regions)
        self.delete_menu=self.bar4.addMenu("Delete points")
        self.average_time = QAction("Average time points",self)
        self.average_time.triggered.connect(self.averageTimeFunc)
        self.bar4.addAction(self.average_time)
        self.fluctuation=self.bar4.addMenu("Baseline fluctuations")
        self.derivate = QAction("Derivate data",self)
        self.derivate.triggered.connect(self.derivateFunc)
        self.fluctuation.addAction(self.derivate) 
        self.flutuation_poly = QAction("Fit  polynom",self)
        self.flutuation_poly.triggered.connect(self.fluctuationPolynom)
        self.fluctuation.addAction(self.flutuation_poly)
        self.bar4.addAction( self.report) 
        self.bar4.show()
        
        self.delete_p_time = QAction("Time dimension",self)
        self.delete_p_time.triggered.connect(self.deletePointTimeFunc)
        self.delete_menu.addAction(self.delete_p_time) 
        
        self.delete_p_wave = QAction("Wavelength dimension",self)
        self.delete_p_wave.triggered.connect(self.deletePointWaveFunc)
        self.delete_menu.addAction(self.delete_p_wave) 
        
        self.init_final_wave = QAction("Set initial and final wavelengths points",self)
        self.init_final_wave.triggered.connect(self.initFianlWAveFunc)
        self.calibration_menu.addAction(self.init_final_wave) 
        
        self.vector_wave = QAction("Introduce vector of wavelengths",self)
        self.vector_wave.triggered.connect(self.vectorWAveFunc)
        self.calibration_menu.addAction(self.vector_wave) 
        
        self.calibrate_table = QAction("Calibrate with table of points",self)
        self.calibrate_table.triggered.connect(self.calibrateTableFunc)
        self.calibration_menu.addAction(self.calibrate_table) 
        
        self.calibrate_reference = QAction("Calibrate from a reference spectra",self)
        self.calibrate_reference.triggered.connect(self.calibrateReferenceFunc)
        self.calibration_menu.addAction(self.calibrate_reference) 
        
        self.bar5 = QMenuBar(self)
        self.bar5.addMenu('&Select Traces')
        self.bar6 = QMenuBar(self)
        self.bar6.addMenu('&Select Traces')
        self.single_fit = QAction("Select One Trace and Fit",self)
        self.single_fit.triggered.connect(lambda:self.oneTraceFit())
        self.bar5.addAction(self.single_fit) 
        self.integral_band = QAction("Integral band fit",self)
        self.integral_band.triggered.connect(lambda:self.integralBandFit())
        self.bar5.addAction(self.integral_band) 
        
        self.Select_singular_vector = QAction('Select Singular Vectors for fit', self)
        self.Select_singular_vector.triggered.connect(lambda: self.plotFullSVD(1,True))
        self.bar5.addAction(self.Select_singular_vector) 
        self.bar6.addAction(self.Select_singular_vector) 
        
        self.Select_all_traces = QAction('Select all traces', self)
        self.Select_all_traces.triggered.connect(self.SelectAllTracesFunc)
        self.bar5.addAction(self.Select_all_traces) 
        self.bar6.addAction(self.Select_all_traces) 
        
        self.Select_Traces_Grapth = QAction("Select Traces From Grapth",self)
        self.Select_Traces_Grapth.triggered.connect(self.SelectTracesGrapthFunc)
        self.bar5.addAction(self.Select_Traces_Grapth)
        self.bar6.addAction(self.Select_Traces_Grapth)
        
        self.Select_Series_Traces = QAction("Select a Series of Traces",self)
        self.Select_Series_Traces.triggered.connect(self.SelectSeriesTracesFunc)
        self.bar5.addAction(self.Select_Series_Traces) 
        self.bar6.addAction(self.Select_Series_Traces) 
        
        self.Select_10_Traces = QAction("Select 10 separated Traces",self)
        self.Select_10_Traces.triggered.connect(self.Select10TracesFunc)
        self.bar5.addAction(self.Select_10_Traces) 
        self.bar6.addAction(self.Select_10_Traces) 
        
        self.Select_Traces_Manually = QAction("Select Traces Manually",self)
        self.Select_Traces_Manually.triggered.connect(self.SelectTracesManuallyFunc)
        self.bar5.addAction(self.Select_Traces_Manually) 
        self.bar6.addAction(self.Select_Traces_Manually) 
        self.select_traces_submenu.addAction(self.Select_all_traces)
        self.select_traces_submenu.addAction(self.Select_Traces_Grapth)
        self.select_traces_submenu.addAction(self.Select_Series_Traces)
        self.select_traces_submenu.addAction(self.Select_10_Traces) 
        self.select_traces_submenu.addAction(self.Select_Traces_Manually)    
        
        ## Create some widgets to be placed inside
        self.button0 = QPushButton('Exp settings', self)
        self.button0.clicked.connect(self.button0Func)
        self.label2 = QLabel('N° components', self)
        #self.text2 = QLineEdit('', self)
        self.button1 = QComboBox(self)
        self.button1.activated.connect(self.button1Func)
        self.button1.addItem("Exponential Fit")
        self.button1.addItem("Target Fit")
        self.select_process = QComboBox(self)
        self.select_process.addItem("Deconvolution")
        self.select_process.addItem("No deconvolution")
        self.select_process.currentIndexChanged.connect(self.deconvComboBox)
        self.button0.show()
        self.label00 = QLabel('Global Fit:', self)
        self.label00.show()
        self.button1.show()

        self.installEventFilter(self) #this is useful to intercept some events, like mouse button press, and execute some conde. if you dont need, you can remove

        self.mousepressed = False #indicates which population is being moved, or False if none
        self.mouse_dx = 0 #markers of relative mouse move from the moment of click
        self.mouse_dy = 0
        self.ref_mouse = QPoint(0,0) #marker of mouse position when population was clicked
        
        #here i mostly configure all the buttons, text fields of the window
#        self.label3 = QLabel('', self)
        self.eps_table = QTableWidget(self)
        self.eps_table.setColumnCount(3)
        self.eps_table.setColumnWidth(0,49);
        self.eps_table.setColumnWidth(1,49);
        self.eps_table.setColumnWidth(2,39);
        self.eps_table.setHorizontalHeaderLabels(('Name','Times', 'Fix'));
        self.eps_table.verticalHeader().setVisible(False);
        self.button2 = QPushButton('Change \n Parameters', self)
        self.button2.clicked.connect(self.button2Func)        
        self.button3 = QPushButton('Initial Parameters!', self)
        self.button3.clicked.connect(self.button3Func)
#        self.button31 = QPushButton('Do pre-Fit', self)
#        self.button31.clicked.connect(self.button31Func) 
        self.button31 = QPushButton('Fit weights', self)
        self.button31.clicked.connect(self.button31WeightsFunc) 
        self.weights_true=QCheckBox('Apply weights',self)
        self.weights_true.stateChanged.connect(self.fitWithWeights)
    
        self.button51 = QPushButton('Final Fit', self)
        self.button51.clicked.connect(self.button51Func)
        
        self.button61 = QPushButton('Previous Fits', self)
        self.button61.clicked.connect(self.button61Func)
        self.button71 = QPushButton('Plot last Fit results', self)
        self.button71.clicked.connect(lambda:self.button71Func())
        self.label51 = QLabel('Number of Iterations:', self)
        self.text51 = QLineEdit('5000', self)

        self.text_proc = QLineEdit('', self) 
        self.button4 = QPushButton('Plot Fit', self)
        self.button4.clicked.connect(self.button4Func)
        self.button41 = QPushButton('Verify Fit', self)
        self.button41.clicked.connect(self.button41Func)
        
        self.centralwidget = QWidget(self)
        self.setCentralWidget(self.centralwidget)
        self.centralwidget.setLayout(QVBoxLayout(self.centralwidget))

#        self.mdiArea = QMdiArea(self.centralwidget)
        self.centralwidget.layout().addWidget(self.mdiArea)
        self.setContentsMargins(self.val, 0, 0, 0) #this forces to move the layout to the right, and create free space on the left
        
        self.label12 = QLabel('Window lenght:', self)
        self.text12 = QLineEdit('15', self)
        self.label22 =  QLabel('polynom order:', self)
        self.text22 = QLineEdit('3', self)
        self.label32 =  QLabel('derivate order:', self)
        self.text32 = QLineEdit('1', self)
        self.button02 = QPushButton('Derivate', self)
        self.button02.clicked.connect(self.button02Func)
        #baseline subtraction buttons
        self.label03 = QLabel('Spectrum number:', self)
        self.text03 = QLineEdit('0', self)
        self.button03 = QPushButton('Selected spectrum', self)
        self.button03.clicked.connect(self.button03Func)
        self.button13 = QPushButton('First N°spectra', self)
        self.button13.clicked.connect(self.button13Func)
        
        self.label05 = QLabel('Time Unit:', self)
        self.text05 = QLineEdit('fs/ps/ns/μs/ms/s/min', self)
        self.label15 =  QLabel('Wavelength Unit:', self)
        self.text15 = QLineEdit('nm/cm-1', self)
        self.button05 = QPushButton('Set Units', self)
        self.button05.clicked.connect(self.button05Func)
        self.label25 =  QLabel('Pump Wavelenght:', self)
        self.text25 = QLineEdit('', self)
        self.button15 = QPushButton('Set Excitation', self)
        self.button15.clicked.connect(self.button15Func)
        
        self.label46 = QLabel('Time range:', self)
        self.text46a = QLineEdit('Min', self) 
        self.text46b = QLineEdit('Max', self)
        self.check56 = QCheckBox('Include max:', self)
        self.check56.setCheckState(2)
        self.label06 = QLabel('Average N° points:', self)
        self.text06 = QLineEdit('0', self)
        self.label16 = QLabel('Legend columns:', self)
        self.text16 = QLineEdit('False/or a number', self)
        self.label26 = QLabel('N° Spectra to plot:', self)
        self.text26 = QLineEdit('8', self)
        self.label36 = QLabel('Equally spaced at:', self)
        self.text36 = QLineEdit('', self)
        self.button06 = QPushButton('Re-Plot spectra', self)
        self.button06.clicked.connect(self.button06Func)
        self.button16 = QPushButton('Plot Manually', self)
        self.button16.clicked.connect(self.button16Func)
        
        self.label012 = QLabel('interval:', self)
        self.text012 = QLineEdit('10', self)
        self.label022 =  QLabel('Average:', self)
        self.text022 = QLineEdit('0', self)
        self.label032 =  QLabel('Avoid excitation:', self)
        self.text032 = QLineEdit('9', self)
        self.label042 =  QLabel('Avoid regions:', self)
        self.text042 = QLineEdit('None', self)
        self.button002 = QPushButton('Select Traces', self)
        self.button002.clicked.connect(self.button002Func)
        
        self.label000 = QLabel('Select time traces:', self)
        self.text000 = QLineEdit('', self)
        self.label001 =  QLabel('Average:', self)
        self.text001 = QLineEdit('0', self)
        self.button001 = QPushButton('Select Traces', self)
        self.button001.clicked.connect(self.button001Func)
        
        self.label07 = QLabel('Wavelength vector:', self)
        self.text07 = QLineEdit('Column/Row', self)
        self.text007 = QLineEdit('0', self)
        self.label17 =  QLabel('Time vector:', self)
        self.text17 = QLineEdit('Column/Row', self)
        self.text117 = QLineEdit('0', self)
        self.label27 =  QLabel('Decimal:', self)
        self.text27 = QLineEdit('.', self)
        self.label37 =  QLabel('separator:', self)
        self.text37 = QLineEdit(',/tab/.', self)
        self.button07 = QPushButton('Load data', self)
        self.button07.clicked.connect(self.button07Func)
        self.button17 = QPushButton('Load by Default', self)
        self.button17.clicked.connect(self.button17Func)
        self.button27 = QPushButton('Save params', self)
        self.button27.clicked.connect(self.button27Func)        
        
        self.label005 = QLabel('Total N° Traces:', self)
        self.label015 = QLabel('', self)
        self.label025 = QLabel('selected N° Traces:', self)
        self.label035 = QLabel('', self)
        #buttons cut region
        self.label023 = QLabel('Wavelength values:', self)
        self.label003 = QLabel('From:', self)
        self.text003 = QLineEdit('', self)
        self.label013 =  QLabel('Until:', self)
        self.text013 = QLineEdit('', self)
        self.button003 = QPushButton('Cut', self)
        self.button003.clicked.connect(self.button003Func)
        self.button013 = QPushButton('Select', self)
        self.button013.clicked.connect(self.button013Func)
        self.label033 = QLabel('Select from grapth:', self)
        self.select_grapth = QComboBox(self)
        self.select_grapth.currentIndexChanged.connect(self.grapthComboBox)
        self.select_grapth.addItem("Clear")
        self.select_grapth.addItem("From")
        self.select_grapth.addItem("Until")
        self.select_grapth.addItem("Both")
        #buttons cut time
        self.label024 = QLabel('Time values:', self)
        self.label004 = QLabel('From:', self)
        self.text004 = QLineEdit('', self)
        self.label014 =  QLabel('Until:', self)
        self.text014 = QLineEdit('', self)
        self.button004 = QPushButton('Cut', self)
        self.button004.clicked.connect(self.button004Func)
        self.button014 = QPushButton('Select', self)
        self.button014.clicked.connect(self.button014Func)
        self.label044 = QLabel('Select from grapth:', self)
        self.select_time = QComboBox(self)
        self.select_time.currentIndexChanged.connect(self.timeComboBox)
        self.select_time.addItem("Clear")
        self.select_time.addItem("From")
        self.select_time.addItem("Until")
        self.select_time.addItem("Both")
        
        self.button37 = QPushButton('Stop Fit', self)
        self.button37.clicked.connect(self.button37Func)
        self.label_progress=QLabel('Fitting progress', self)
        self.progress_fit=QProgressBar(self)
        self.progress_fit.setOrientation(2)
        
        self.label08 = QLabel('BK7:', self)
        self.text08 = QLineEdit('0', self)
        self.label18 =  QLabel('SiO2:', self)
        self.text18 = QLineEdit('0', self)
        self.label28 =  QLabel('CaF2:', self)
        self.text28 = QLineEdit('0', self)
        self.label38 =  QLabel('Offset:', self)
        self.text38 = QLineEdit('0', self)
        self.label48 =  QLabel('Pump excitation:', self)
        self.text48 = QLineEdit('400', self)
        self.check08 = QCheckBox('Verified correction',self)
        self.button08 = QPushButton('Correct GVD', self)
        self.button08.clicked.connect(self.button08Func)
        
        self.label09 = QLabel('Average after:', self)
        self.text09 = QLineEdit('', self)
        self.label19 =  QLabel('Step:', self)
        self.text19 = QLineEdit('0', self)
        self.label29 = QLabel('Tipe of step:', self)
        self.label39 = QLabel('Speed:', self)
        self.select_average = QComboBox(self)
        self.select_average.currentIndexChanged.connect(self.timeAverigeComboBoxFunc)
        self.select_average.addItem("Increase Step")
        self.select_average.addItem("Constant Step")
        self.button09 = QPushButton('Average Points', self)
        self.button09.clicked.connect(self.button09Func)
        self.spin= QSpinBox(self)
        self.spin.setMaximum(10)
        self.spin.setMinimum(1)
        #self.spin.prefix('Speed of increase')
        
        self.button_plot_edit = QPushButton('Plot spectra\nauto', self)
        self.button_plot_edit.clicked.connect(self.PlotSpecAutoFuncProcess)
        self.button_plot_edit2 = QPushButton('Plot spectra\nauto', self)
        self.button_plot_edit2.clicked.connect(self.PlotSpecAutoFuncProcess)
        self.button_plot_edit4 = QPushButton('Plot 10 traces\nauto', self)
        self.button_plot_edit4.clicked.connect(self.Plot10TracesFuncProcess)
        self.button_undo_region= QPushButton('Undo', self)
        self.button_undo_region.clicked.connect(self.undoRegionFunc)
        self.button_undo_time= QPushButton('Undo', self)
        self.button_undo_time.clicked.connect(self.undoTimeFunc)
        
        self.cal=Calibration()
        self.message=QLabel('', self)
        self.message.show()
        
        self.label124 = QLabel('Wavelength values:', self)
        self.label104 = QLabel('Initial:', self)
        self.text104 = QLineEdit('', self)
        self.label114 =  QLabel('Final:', self)
        self.text114 = QLineEdit('', self)
        self.button104 = QPushButton('Calibrate', self)
        self.button104.clicked.connect(self.button104Func)
        
        self.select_grapth = QComboBox(self)
        self.select_grapth.currentIndexChanged.connect(self.grapthComboBox)
        self.select_grapth.addItem("Clear")
        self.select_grapth.addItem("From")
        self.select_grapth.addItem("Until")
        
        self.button_tf_1 = QComboBox(self)
        self.button_tf_1.activated.connect(self.targetModelFunc)
        self.button_tf_1.addItem('Make model')
        self.button_tf_1.addItem('Load model')
        self.button_tf_1.addItem('Change model')
        self.tf_eps_table = QTableWidget(self)
        self.tf_eps_table.verticalHeader().setVisible(False);
        self.tf_eps_table.setColumnCount(3)
        self.tf_eps_table.setColumnWidth(0,49);
        self.tf_eps_table.setColumnWidth(1,49);
        self.tf_eps_table.setColumnWidth(2,39);
        self.tf_eps_table.setHorizontalHeaderLabels(('Name','Times', 'Fix'));
        self.tf_eps_table.verticalHeader().setVisible(False);
        self.button_tf_3 = QPushButton('check K-matrix', self)
        self.button_tf_3.clicked.connect(self.checkKMatrixFunc)
#        self.button_tf_5 = QPushButton('Do pre-Fit', self)
#        self.button_tf_5.clicked.connect(self.modelPreFitFunc)
#        self.button_tf_6 = QPushButton('Final Fit', self)
#        self.button_tf_6.clicked.connect(self.modelFinalFitFunc)
        
#        self.threadPool=QThreadPool()
        self.my_thread_Fit = QThread()
        self.my_thread_Fit.start(QThread.TimeCriticalPriority)
        
        self.my_thread_preFit = QThread()
        self.my_thread_preFit.start(QThread.TimeCriticalPriority)
        
        self.fit_subwindows={}
        self.model_instantiated=False
        self.long_calcul_runing=False
        self.IRF_file=None
        self.manual_load=None
        self.IRF_value=None
        self.total_traces=0
        self.selected_traces='No data'
        self.fit_number=0
        self.undo_wavelength=None
        self.undo_time=None
        self.undo_data=None
        self.undo_wave_already=False
        self.undo_time_already=False
        self.save_fileName=None
        self.fit_in_progress=False
        self.corrected_GVD=False
        self.initialize_global_fit=False
        self.file=None
        self.target_models={}
        self.timeAverigeComboBox_edit=False
        self.edit_general={'load_data_edit':False,'traces_manually_edit':False,'series_traces_edit':False,
                           'pop_edit':False, 'proc_edit':False,'settings_edit':False,'plot_spec_edit':False,
                          'plot_spec_auto_edit':False,'deriv_edit':False, 'baseline_edit':False, 
                          'traces_manually_edit':False,'cut_region_edit':False,'fitting_edit':False,
                          'sellmeier_edit':False,'cut_time_edit':False,'average_time_edit':False,
                          'number_traces_edit':False,'init_fianl_wave_edit':False,'target_fit':False,'time_shift':False}
        self.cmaps={}
        self.cmaps['Perceptually Uniform Sequential'] = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',  
            'hsv','icefire','vlag','mako','rocket','YlOrRd', 'YlGn']
        self.cmaps['Sequential'] = [
                    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                    'YlOrBr', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn']
        self.cmaps['Sequential (2)'] = [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']
        self.cmaps['Diverging'] = [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
        self.cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']
        self.cmaps['Miscellaneous'] = [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']
   
    def makeSaveDir(self):
        path = QFileDialog.getSaveFileName(ex,'',"all data")
        experiment.createNewDir(path[0]+'/')
    
    def exportWindowD(self):
        try:
            def exporting():
                if experiment.working_directory is not None:
                    path = experiment.working_directory
                else:
                    path = QFileDialog.getSaveFileName(ex,'',"all data")
                    path = path[0]
                if txt.isChecked():
                    separator='\t'
                    if coma.isChecked():
                        decimal = ','
                    else:
                        decimal = '.'
                        extra='.txt'
                elif coma.isChecked():
                    separator = ';'
                    decimal = ','
                    extra='.csv'
                else:
                    separator = ','
                    decimal = '.' 
                    extra='.csv'
                x=experiment.x
                if traces.isChecked():
                    if experiment.data_before_first_selection is not None:
                        datos=experiment.data
                        wavelength=experiment.wavelength
                        path=path[0]+'selected traces'+extra
                        self.savePandasDF(x,data,wavelength,path,separator,decimal)
                        subWindow.close()
                    else:
                        data.setCheckState(2)
                if data.isChecked():
                    if experiment.data_before_first_selection is not None:
                        datos=experiment.data_before_first_selection
                        wavelength=experiment.wavelength_before_first_selection
                    else:
                        datos=experiment.data
                        wavelength=experiment.wavelength
                    path=path+' data matrix'+extra
                    self.savePandasDF(x,datos,wavelength,path,separator,decimal)
                    subWindow.close()
                if data.isChecked() == False and traces.isChecked() == False:
                    error_dialog = QMessageBox(self)
                    error_dialog.setText('Select at least data or traces to export')
                    error_dialog.exec() 
            subWindow=QMdiSubWindow(self)    
            subWindow.setWindowTitle("Exporting data")
            mainlayout=QHBoxLayout()
            mainlayout.setContentsMargins(20,20,20,10)
            sublayout1=QVBoxLayout()
            sublayout1.setAlignment(Qt.AlignTop)
            sublayout1.addWidget(QLabel('Select the data:'))
            sublayout1.setContentsMargins(10,0,30,0)
            data=QCheckBox('all data matrix',subWindow)
            traces=QCheckBox('Selected Traces',subWindow)
            sublayout1.addWidget(data)
            sublayout1.addWidget(traces)
            data.setCheckState(2)
            traces.setCheckState(2)
            sublayout2=QVBoxLayout()
            sublayout2.setAlignment(Qt.AlignTop)
            sublayout2.setContentsMargins(10,0,30,0)
            sublayout2.addWidget(QLabel('Select type of file'))
            csv=QCheckBox('.csv',subWindow)
            csv.setCheckState(2)
            txt=QCheckBox('.txt',subWindow)
            sublayout2.addWidget(csv)
            sublayout2.addWidget(txt)
            group1=QButtonGroup(self)
            group1.addButton(csv)
            group1.addButton(txt)
            sublayout3=QVBoxLayout()
            sublayout3.setAlignment(Qt.AlignTop)
            sublayout3.setContentsMargins(10,0,10,0)
            sublayout3.addWidget(QLabel('Select decimal separator'))
            coma=QCheckBox('coma',subWindow)
            punto=QCheckBox('dot',subWindow)
            punto.setCheckState(2)
            group2=QButtonGroup(self)
            group2.addButton(coma)
            group2.addButton(punto)
            sublayout3.addWidget(punto)
            sublayout3.addWidget(coma)
            mainlayout.addLayout(sublayout1)
            mainlayout.addLayout(sublayout2)
            mainlayout.addLayout(sublayout3)
            btn_undo = QPushButton("export", subWindow) 
            btn_undo.clicked.connect(exporting)
            btn = QPushButton("Close", subWindow) 
            btn.clicked.connect(subWindow.close)
            subWindow.layout().addLayout(mainlayout)
            subWindow.layout().addWidget(btn_undo)
            subWindow.layout().addWidget(btn)
            self.mdiArea.addSubWindow(subWindow)
            subWindow.show()
        except:
            self.messageError()

    def savePandasDF(self,x,data,wavelength,path,separator,decimal):
        try:
            data_export=pd.DataFrame(data, index=x, columns=wavelength)
            data_export.to_csv(path,sep=separator, decimal=decimal, index_label='Time')
            self.printMessage('Data exported')
        except:
            error_dialog = QMessageBox(self)
            error_dialog.setText(f'Unable to export {path.split("/")[-1]}')
            error_dialog.exec()
        
    def exportWindowF(self,combo):
        try:
            self.prev_subWindow.close()
            for fit_number,ii in enumerate(combo.buttons()):
                    if ii.isChecked():
                        number=(fit_number+1)
            def exporting():
                if experiment.working_directory is not None:
                    path = experiment.working_directory
                else:
                    path = QFileDialog.getSaveFileName(ex,'',f"Fit {number}")
                    path = path[0]
                if folder.isChecked():
                    if not os.path.exists(path):
                        os.makedirs(path)
                        path=path+'/'
                if txt.isChecked():
                    separator='\t'
                    if coma.isChecked():
                        decimal = ','
                    else:
                        decimal = '.'
                        extra='.txt'
                elif coma.isChecked():
                    separator = ';'
                    decimal = ','
                    extra='.csv'
                else:
                    separator = ','
                    decimal = '.' 
                    extra='.csv'
                for ii,button in enumerate(group0.buttons()):
                    if button.isChecked():
                        datos=sets[ii]
                        sub_path=path+names[ii]+extra
                        print(ii)
                        if len(datos)!= len(x):
                            x_exp=np.array(range(len(datos)))
                            print(ii)
                        else:
                            x_exp=x
                        if len(datos[1])!= len(wavelength):
                            wavelength_exp=[f'SV {i}' for i in range(len(datos[1]))]
                            print(ii)
                        else:
                            wavelength_exp=wavelength
                        self.savePandasDF(x_exp,datos,wavelength_exp,sub_path,separator,decimal)
                        if report.isChecked():
                            texto=fit_report(fit[3])
                            text=open(path+' fit report'+extra,'w')
                            text.write(texto)
                            text.close()
            fit=experiment.all_fit[number]
            x=fit[0]
            trace=fit[1]
            wavelength=fit[2]
            svd=fit[9]
            DAS=experiment.DAS(fit_number=number)
            params=fit[3].params
            if svd:
                paramsSVD=fit[3].params
                params=fit[11]
                dataSVD=fit[10]
                fittesSVD=experiment.results(paramsSVD,fit_number=number)
                fittes=experiment.results(params,fit_number=number,verify_SVD_fit=True)
                names=[' Original Data',' Recompose data',' Residues',' DAS', ' Sigular Vectors', ' Fit Sigular Vectors',' SVD Residues']
            else: 
                params=fit[3].params
                names=[' Original traces',' Fitted Traces',' Residues',' DAS']
                fittes=experiment.results(params,fit_number=number)
            subWindow=QMdiSubWindow(self)    
            subWindow.setWindowTitle(f"Exporting Fit {number}")
            mainlayout=QHBoxLayout()
            mainlayout.setContentsMargins(20,20,20,10)
            sublayout1=QVBoxLayout()
            sublayout1.setAlignment(Qt.AlignTop)
            sublayout1.addWidget(QLabel('Select the data:'))
            sublayout1.setContentsMargins(10,0,30,0)
            data=QCheckBox(names[0],subWindow)
            traces=QCheckBox(names[1],subWindow)
            residues=QCheckBox(names[2],subWindow)
            das=QCheckBox(names[3],subWindow)
            report=QCheckBox('Fit report',subWindow)
            sublayout1.addWidget(data)
            sublayout1.addWidget(traces)
            sublayout1.addWidget(residues)
            sublayout1.addWidget(das)
            residues.setCheckState(2)
            das.setCheckState(2)
            report.setCheckState(2)
            data.setCheckState(2)
            traces.setCheckState(2)
            group0=QButtonGroup(self)
            group0.setExclusive(False)
            group0.addButton(data)
            group0.addButton(traces)
            group0.addButton(residues)
            group0.addButton(das)
            mainlayout.addLayout(sublayout1)
            if svd:
                sublayout3=QVBoxLayout()
                sublayout3.addWidget(QLabel('SVD fit data:'))
                sublayout3.setAlignment(Qt.AlignTop)
                # sublayout3.setContentsMargins(10,0,30,0)
                svd=QCheckBox(names[4],subWindow)
                svdFit=QCheckBox(names[5],subWindow)
                svd_residues=QCheckBox(names[6],subWindow)
                svd.setCheckState(2)
                svdFit.setCheckState(2)
                svd_residues.setCheckState(2)
                group0.addButton(svd)
                group0.addButton(svdFit)
                group0.addButton(svd_residues)
                sublayout3.addWidget(svd)
                sublayout3.addWidget(svdFit)
                sublayout3.addWidget(svd_residues)
                sublayout3.addWidget(report)
                mainlayout.addLayout(sublayout3)
                sets=[trace,fittes,trace-fittes,DAS,dataSVD,fittesSVD,dataSVD-fittesSVD,]
            else:
                sublayout1.addWidget(report)
                sets=[trace,fittes,trace-fittes,DAS]
            sublayout2=QVBoxLayout()
            sublayout2.setAlignment(Qt.AlignTop)
            sublayout2.setContentsMargins(10,0,30,0)
            sublayout2.addWidget(QLabel('Select type of file'))
            csv=QCheckBox('.csv',subWindow)
            csv.setCheckState(2)
            txt=QCheckBox('.txt',subWindow)
            sublayout2.addWidget(csv)
            sublayout2.addWidget(txt)
            group1=QButtonGroup(self)
            group1.addButton(csv)
            group1.addButton(txt)
            sublayout2.addWidget(QLabel('Select decimal separator'))
            coma=QCheckBox('coma',subWindow)
            punto=QCheckBox('dot',subWindow)
            punto.setCheckState(2)
            group2=QButtonGroup(self)
            group2.addButton(coma)
            group2.addButton(punto)
            sublayout2.addWidget(punto)
            sublayout2.addWidget(coma)
            mainlayout.addLayout(sublayout2)
            subWindow.layout().addLayout(mainlayout)
            subWindow.layout().addWidget(QLabel('----------------------------------------------'), alignment=Qt.AlignCenter)
            folder=QCheckBox('make a folder',subWindow)
            folder.setCheckState(2)
            subWindow.layout().addWidget(folder, alignment=Qt.AlignCenter)
            btn_undo = QPushButton("export", subWindow) 
            btn_undo.clicked.connect(exporting)
            btn = QPushButton("Close", subWindow) 
            btn.clicked.connect(subWindow.close)
            subWindow.layout().addWidget(btn_undo)
            subWindow.layout().addWidget(btn)
            self.mdiArea.addSubWindow(subWindow)
            subWindow.show()
        except:
            self.messageError()
    
    def fluctuationPolynom(self):
        if experiment.data_before_first_selection is not None:
            data=experiment.data_before_first_selection*1.00
        else:
            data=experiment.data*1.00
        def fitPol():
            order=spin.value()
            points=sorted(grapth.cursore.datax)
            if len(points) <= order:
                error_dialog = QMessageBox(self)
                error_dialog.setText('The number of point selected should be larger that the polynom order')
                error_dialog.exec() 
            else:
                experiment.correctionFlutuationsPoly(points,order)  
                ax.remove()
                figo,ax2=experiment.plot_spectra(times=points_plot, ncol=2)
                ax2.remove()
                ax2.figure=fig
                fig.add_axes(ax2)
                fig.axes.append(ax2)
                fig.canvas.draw()
                btn_undo.show()
                btn_set.setVisible(False)
        def undo():
            if experiment.data_before_first_selection is not None:
                experiment.data_before_first_selection=data
            else:
                experiment.data=data
            fig.axes[0].remove()
            ax.figure=fig
            fig.add_axes(ax)
            fig.axes.append(ax)
            self.delSelectionFun(grapth,ax)
            btn_set.show()
            btn_undo.setVisible(False)
        if self.mdiArea.size().height()<725:
            size=self.mdiArea.size()*0.75
        else:
            size=QSize(1250,725)
        points_plot=list(experiment.getAutoPoints(15))
        subWindow=QMdiSubWindow(self)    
        fig,ax=experiment.plot_spectra(times=['auto', 16], ncol=2)
        grapth=Grapth(fig,subWindow,toolbar=True,cursor=True,y=False,number_click=-1,)
        subWindow.resize(size)
        spin=QSpinBox(subWindow)
        spin.setMaximum(4)
        spin.setMinimum(2)
        spin.setValue(3)
        spinlayout=QHBoxLayout()
        spinlayout.addWidget(spin)
        spinlayout.addWidget(QLabel('Polynom Order'))
        mainlayout=QHBoxLayout()
#        mainlayout.setRowStretch(10,0)
        mainlayout.addWidget(grapth)
        sublayout=QVBoxLayout()
        sublayout.setAlignment(Qt.AlignTop)
        sublayout.setContentsMargins(10,150,10,50)
        sublayout.addLayout(spinlayout)
        btn_undo = QPushButton("Undo", subWindow) 
        btn_undo.clicked.connect(undo)
        btn_set = QPushButton("Correct data", subWindow) 
        btn_set.clicked.connect(fitPol)
        btn = QPushButton("Close", subWindow) 
        btn.clicked.connect(subWindow.close)
        btn_undo.setVisible(False)
        sublayout.addWidget(btn_set)
        sublayout.addWidget(btn_undo)
        sublayout.addWidget(btn)
        mainlayout.addLayout(sublayout)
        subWindow.layout().addLayout(mainlayout)
        self.mdiArea.addSubWindow(subWindow)
        subWindow.show()
    
    def targetModelFunc(self):
        if self.button_tf_1.currentText()=='Make model':
            self.makeModelFunc()
        elif self.button_tf_1.currentText()=='Load model':
            self.loadModelFunc()
        else:
            if self.model_instantiated:
                self.modifiedModelFunc()
            else:
                GVD_dialog = QMessageBox(self)
                GVD_dialog.setText('Please make or load a model first')
                GVD_dialog.exec()
    
    def makeModelFunc(self):
        '''open the model making window'''
        size=[self.mdiArea.size().width(),self.mdiArea.size().height()]
        self.model= Model()
        widget = self.model.manualModelBuild()
        self.model_instantiated=True
        subWindow = QMdiSubWindow(self)
        btn_set = QPushButton("Set the Model", widget) 
        btn_set.clicked.connect(lambda:self.setModelParamsFunc(subWindow))
        btn = QPushButton("Save Model", widget) 
        btn.clicked.connect(lambda:self.saveModel())
        self.formatSubWindow(subWindow,self.model.manualModelBuild(),'Dessign Model Window',close_button=True,extra_button=[btn_set,btn])
        subWindow.resize(size[0]/2,size[1]/2)
        
    def loadModelFunc(self):
        '''load a model and show it in making model window'''
        try:
            size=[self.mdiArea.size().width(),self.mdiArea.size().height()]
            fname = QFileDialog.getOpenFileName(self, 'Open model file', 
                'c:\\',"files (*.model)")
            self.model= Model.load(fname[0])
            widget = self.model.manualModelBuild()
            self.model_instantiated=True
            subWindow = QMdiSubWindow(self)
            btn_set = QPushButton("Set the Model", widget) 
            btn_set.clicked.connect(lambda:self.setModelParamsFunc(subWindow))
            btn = QPushButton("Save Model", widget) 
            btn.clicked.connect(lambda:self.saveModel())
            self.formatSubWindow(subWindow,widget,'Dessign Model Window',close_button=True,extra_button=[btn_set,btn])
            subWindow.resize(size[0]/2,size[1]/2)
        except:
            error_dialog = QMessageBox(self)
            error_dialog.setText('Unable to load model')
            error_dialog.exec()    
        
    def saveModel(self):
        '''save current model in the window'''
        try:
            save_fileName = QFileDialog.getSaveFileName(self,'Save Model',"Target Model")
            self.model.save(save_fileName)
            self.printMessage('Model saved')
        except:
            error_dialog = QMessageBox(self)
            error_dialog.setText('Unable to save model')
            error_dialog.exec()
        
    def setModelParamsFunc(self,window):
        '''extract params from loaded or constructed model and tranfer to table tf_eps_table'''
        self.target_params=self.model.genParameters().copy()
        experiment.params_initialized=False
        try:
            try:
                 self.paramsWindow.close()
            except:
                pass
            self.corrected_GVD=experiment.GVD_correction
            self.paramsWindow = QMdiSubWindow(self)
            self.paramsWindow.resize(300,400)
            init_val=str(0)
            if self.select_process.currentText() == 'Deconvolution':
                rango=2
                experiment.deconv=True
            else:
                if self.initialize_global_fit==True:
                    init_val=str(experiment.x[0])
                experiment.deconv=False
                rango=1
            param_table = QTableWidget(rango+self.target_params['exp_no'].value,3,self)
            param_table.setHorizontalHeaderLabels(('Parameter', 'Value','Fix/Optimized'))
            param_table.setColumnWidth(0,60)
            param_table.setColumnWidth(1,50)
            param_table.setColumnWidth(3,60)
            if rango==1:
               combo = QComboBox()
               combo.addItem("Fix")
               combo.addItem("Optimize")
               param_table.setCellWidget(0,2,combo)    
            if rango==2:
               for ii in range(rango):
                    combo = QComboBox()
                    if ii==1:
                        combo.addItem("Fix")
                        combo.addItem("Optimize")
                    else:    
                        combo.addItem("Optimize")
                        combo.addItem("Fix")
                    param_table.setCellWidget(ii,2,combo)
            texto='time 0'
            name=QLabel(texto)
            it= QTableWidgetItem(init_val)
            param_table.setCellWidget(0,0,name)
            param_table.setItem(0,1,it)
            if self.select_process.currentText() == 'Deconvolution':    
                experiment.deconv=True
                texto='fwhm'
                name=QLabel(texto)
                if self.IRF_value is None:
                    fwhm_value=0.12
                else:
                    fwhm_value=self.IRF_value
                it= QTableWidgetItem(str(fwhm_value))
                #name=QTableWidgetItem('tau '+str(ii+1))
                param_table.setCellWidget(1,0,name)
                param_table.setItem(1,1,it)
            for i in range(self.target_params['exp_no'].value):
                name=QLabel('tau '+str(i+1))
                tau=-1/self.target_params['k_%i%i' % (i+1,i+1)].value 
                it= QLabel(str(round(tau,1)))
                op='Optimize' if self.target_params['k_%i%i' % (i+1,i+1)].vary else 'Fix'
                param_table.setCellWidget(rango+i,0,name)
                param_table.setCellWidget(rango+i,1,it)    
                param_table.setCellWidget(rango+i,2,QLabel(op))
            self.formatSubWindow(self.paramsWindow,param_table,'Parameters',)
            if rango==2:
#                self.check_inf=QCheckBox('Include a infinite time to evaluate product formation',self.paramsWindow)
#                self.check_inf.setCheckState(2)
#                self.paramsWindow.layout().addWidget(self.check_inf)
                if self.corrected_GVD==False:
                    self.check=QCheckBox('Non of the GVD data correction functions has been Run; \nplease check box if data was corrected previously',self.paramsWindow)
                    self.paramsWindow.layout().addWidget(self.check)
            btn = QPushButton("Set parameters", self.paramsWindow) 
            self.paramsWindow.layout().addWidget(btn)
            btn.clicked.connect(lambda: setParams())
            btn_close = QPushButton("Close", self.paramsWindow) 
            self.paramsWindow.layout().addWidget(btn_close)
            btn_close.clicked.connect(self.paramsWindow.close)
            def setParams():
#                experiment.initialParamsModel(self.target_params.copy(),0,vary_t0=True)
                self.variations=[True]
                valort=QLabel(param_table.item(0,1).text())
                t0=float(param_table.item(0,1).text())
                t0_vary= True if param_table.cellWidget(0, 2).currentText() == 'Optimize' else False
                print(t0_vary)
                try:
                    if self.check.checkState()==2 and self.corrected_GVD==False:
                        self.corrected_GVD=True
                        experiment.general_report['Preprocessing']['GVD correction']=f'\tGVD was previously corrected'
                except:
                    pass
                if rango==2:
                    fwhm=float(param_table.item(1,1).text())
                    fwhm_vary=True if param_table.cellWidget(1, 2).currentText() == 'Optimize' else False
                else:
                    fwhm=0.16
                    fwhm_vary=True
                #set params value in tf_eps_table
                rows=param_table.rowCount()
                self.tf_eps_table.setRowCount(rows) 
                for i in range(rows):
                    namet=QLabel(param_table.cellWidget(i, 0).text())
                    if i<rango:
                        Change=param_table.cellWidget(i, 2).currentText()
                        valort=QLabel(param_table.item(i,1).text())
                    else:
                        Change=param_table.cellWidget(i, 2).text()
                        valort=QLabel(param_table.cellWidget(i, 1).text())
                    if Change=='Fix':
                        vary=QLabel('Yes')
                    else:
                        vary=QLabel('No')
                    self.tf_eps_table.setCellWidget(i,0,namet)
                    self.tf_eps_table.setCellWidget(i,1,valort)
                    self.tf_eps_table.setCellWidget(i,2,vary)
                    experiment.initialParamsModel(self.target_params.copy(),t0=t0,vary_t0=t0_vary,fwhm=fwhm,vary_fwhm=fwhm_vary)
                self.paramsWindow.close()
                window.close()
            self.repaint()
        except:
            self.messageError()
        
        
    def checkKMatrixFunc(self,fit_number):
        '''Check K parameters matrix '''
        if experiment.type_fit=='Exponential':
            error_dialog = QMessageBox(self)
            error_dialog.setText('Last parameters initialized are for expoential fit.\nPlease initialized parameters from a model')
            error_dialog.exec()
        else:
            try:
                if experiment.fit_completed or experiment.prefit_done:
                    params=experiment.params
                else:
                    params=experiment.initial_params
                exp_no=experiment.exp_no
                Window = QMdiSubWindow(self)
                Window.resize(350,300)
                red=QLabel()
                red.setStyleSheet('color:red')
                red.setText('fixed parameter')
                green=QLabel()
                green.setStyleSheet('color:green')
                green.setText('Optimized parameter')
                layout=QHBoxLayout()
                layout.setAlignment(Qt.AlignCenter)
                layout.addWidget(red)
                layout.addWidget(green)
                Window.layout().addLayout(layout)
                k_table = QTableWidget(exp_no,exp_no,self)
                k_table.setHorizontalHeaderLabels(('k_%i'%(i+1) for i in range(exp_no)))
                k_table.setVerticalHeaderLabels(('k_%i'%(i+1) for i in range(exp_no)))
                for i in range(exp_no):
                    k_table.setColumnWidth(i,75)
                for i in range(exp_no):
                    for j in range(exp_no):
                        k_widjet=QLabel()
                        if params['k_%i%i'%(i+1,j+1)].vary:
                            k_widjet.setStyleSheet(('color:green'))
                        else:
                            k_widjet.setStyleSheet(('color:red'))
                        k_widjet.setText('{:.4f}'.format(params['k_%i%i'%(i+1,j+1)].value))
                        k_table.setCellWidget(i,j,k_widjet)
                btn_point = QPushButton("Advance parameters options", Window) 
                Window.layout().addWidget(btn_point)
                btn_point.clicked.connect(self.allparamsTM)
                self.formatSubWindow(Window,k_table,'K matrix',close_button=True,extra_button=btn_point)
                Window.show()
            except:
                self.messageError()
    
    def allparamsTM(self):
        try:
            try:
                self.paramsTMWindowchange.close()
            except:
                pass
            self.paramsTMWindowchange = QMdiSubWindow(self)
            self.paramsTMWindowchange.resize(455,500)
            rows=len(experiment.initial_params)
            if experiment.prefit_done:
                params=experiment.params
            else:
                params=experiment.initial_params
            self.params_table_all_TM = QTableWidget(rows,6,self)
            self.params_table_all_TM.setHorizontalHeaderLabels(('Wavelength','Parameter', 'Value','Fix/Optimized','min','max','Fix/Optimized'))
            self.params_table_all_TM.setColumnWidth(0,100)
            self.params_table_all_TM.setColumnWidth(1,70)
            self.params_table_all_TM.setColumnWidth(2,60)
            self.params_table_all_TM.setColumnWidth(3,100)
            self.params_table_all_TM.setColumnWidth(4,50)
            self.params_table_all_TM.setColumnWidth(5,50)
            number=experiment.exp_no*experiment.exp_no+experiment.exp_no+1
            sli=(rows-number)//len(experiment.wavelength)
            if sli>0:   
                waves=[i for i in range(number,rows)[::sli]]
            else:
                waves=[i for i in range(number,rows)]
            names=[key for key in params.keys()]
            for i in range(rows):
                if i in waves:
                    try:
                        wave= QLabel('{:.1f}'.format(experiment.wavelength[(i-number)//sli]))
                        self.params_table_all_TM.setCellWidget(i,0,wave)
                    except:
                        pass
#                if i < number:
#                    wave= QLabel('recomended \nnot to change')
#                    self.params_table_all_TM.setCellWidget(i,0,wave)
                param= QLabel(names[i])
                self.params_table_all_TM.setCellWidget(i,1,param)
                name=params[names[i]].expr
                if name is None:
                    valor=QTableWidgetItem('{:.4f}'.format(params[names[i]].value))
                    mini=QTableWidgetItem('{:.4f}'.format(params[names[i]].min))
                    maxi=QTableWidgetItem('{:.4f}'.format(params[names[i]].max))
                    self.params_table_all_TM.setItem(i,2,valor)
                    self.params_table_all_TM.setItem(i,4,mini)
                    self.params_table_all_TM.setItem(i,5,maxi)
                    if 'tau' in names[i] or 'exp_no' in names[i]:
                        combo=QLabel(f'Informative param')
                    else:
                        combo = QComboBox()
                        if params[names[i]].vary:
                            combo.addItem("Optimize")
                            combo.addItem("Fix")   
                        else:    
                            combo.addItem("Fix")
                            combo.addItem("Optimize")              
                else:
                    valor=QLabel('{:.4f}'.format(params[names[i]].value))
                    mini=QLabel('{:.4f}'.format(params[names[i]].min))
                    maxi=QLabel('{:.4f}'.format(params[names[i]].max))
                    self.params_table_all_TM.setCellWidget(i,2,valor)
                    self.params_table_all_TM.setCellWidget(i,4,mini)
                    self.params_table_all_TM.setCellWidget(i,5,maxi)
                    combo=QLabel(f'same as: {name}')
                self.params_table_all_TM.setCellWidget(i,3,combo)   
            self.formatSubWindow(self.paramsTMWindowchange,self.params_table_all_TM,'Parameters',)
            btn = QPushButton("Change parameters", self.paramsTMWindowchange) 
            self.paramsTMWindowchange.layout().addWidget(btn)
            btn.clicked.connect(self.modifiedTMParams)
            btn_close = QPushButton("Close", self.paramsTMWindowchange) 
            self.paramsTMWindowchange.layout().addWidget(btn_close)
            btn_close.clicked.connect(self.paramsTMWindowchange.close)
            self.repaint()
        except:
            self.messageError()
    
    def modifiedTMParams(self):
        try:
            if experiment.prefit_done:
                params=experiment.params
            else:
                params=experiment.initial_params
            count=0
            if experiment.deconv: 
                important_names=['t0_1','fwhm_1']+['k_%i%i' %((i+1),(i+1)) for i in range(experiment.exp_no)]
                rango=2
            else:
                important_names=['t0_1']+['k_%i%i' %((i+1),(i+1)) for i in range(experiment.exp_no)]
                rango=1
            critic_names=['C_%i' %(i+1) for i in range(experiment.exp_no)]+['k_%i%i' %((i+1),(j+1)) for i in range(experiment.exp_no) for j in range(experiment.exp_no)]
            critic_change = False
            show_message = True
            names=[key for key in params.keys()]
            def changeParam(valores,change_table=False):
                if experiment.prefit_done:
                       experiment.params[ii].value=valores[0]
                       experiment.params[ii].min=valores[1]
                       experiment.params[ii].max=valores[2]
                       experiment.params[ii].vary=valores[3]
                experiment.initial_params[ii].value=valores[0]
                experiment.initial_params[ii].min=valores[1]
                experiment.initial_params[ii].max=valores[2]
                experiment.initial_params[ii].vary=valores[3]    
            for i,ii in enumerate(names):
                if params[ii].expr is None and 'tau' not in  ii and 'exp_no' not in  ii:
                    print(i)
                    valor=float(self.params_table_all_TM.item(i,2).text())
                    mini=float(self.params_table_all_TM.item(i,4).text())
                    maxi=float(self.params_table_all_TM.item(i,5).text())
                    vary=False if self.params_table_all_TM.cellWidget(i, 3).currentText() == 'Fix' else True
                    if ii in critic_names and show_message:
                        param_change=False
                        if round(params[ii].value,4) != valor: param_change = True
                        if params[ii].min != mini: param_change = True 
                        if params[ii].max != maxi: param_change = True 
                        if params[ii].vary != vary: param_change = True 
                        if param_change:
                                error_dialog = QMessageBox(self)
                                error_dialog.resize(400,200)
                                error_dialog.setText(f'\t\tWe have identified that you have modified : \n\t\t\t{ii} paramter \
                                                     \nInitial concetrations and K matrix parameters are critical. \
                                                     \nWe recomend to change them via the model-window as the consistency of the model is verified.\
                                                     \n\n If you  are shure of the changes click "Change parameters"\n if not click "cancel".')
                                message=QCheckBox('Apply the same to all other critical paramters',error_dialog)
                                cancel=QPushButton("Cancel", error_dialog)
                                accept=QPushButton("Change parameters", error_dialog)
                                error_dialog.addButton(cancel,QMessageBox.AcceptRole)
                                error_dialog.addButton(accept,QMessageBox.RejectRole)
                                error_dialog.setDefaultButton(accept)
                                layout=QVBoxLayout(error_dialog)
                                layout.addWidget(message)
#                                error_dialog.layout().addWidget(message,1,1,alignment=Qt.AlignLeft)
                                tip=QLabel('\n (Changes in non critical parameters will still be done)',error_dialog)
                                layout.addWidget(tip)
                                error_dialog.layout().addLayout(layout,1,1,alignment=Qt.AlignLeft)
                                error_dialog.exec()
                                if error_dialog.clickedButton() == accept:
                                    if message.isChecked():
                                        show_message = False
                                        critic_change = True
                                    changeParam([valor,mini,maxi,vary])
                                else:
                                    if message.isChecked():
                                        show_message=False
                                    pass
                    elif ii in critic_names and critic_change:
                        changeParam([valor,mini,maxi,vary])
                    else:
                        changeParam([valor,mini,maxi,vary])
            for ii in important_names:
                Change=experiment.initial_params[ii].vary
                if important_names.index(ii)<rango:
                    dato=round(experiment.initial_params[ii].value,4)
                else:
                    dato=round(-1/experiment.initial_params[ii].value,4)
                if Change:
                    vary=QLabel('No')
                else:
                    vary=QLabel('Yes')
                self.tf_eps_table.setCellWidget(count,2,vary)
                namet=QLabel(str(dato))
                self.tf_eps_table.setCellWidget(count,1,namet)
                count += 1
            self.paramsTMWindowchange.close()
        except:
            self.messageError()

        
    def printModel(self,fit_number):
        try:
            Window = QMdiSubWindow(self)
            Window.setWindowTitle("Schematic dessign of model fitted")
            lista=experiment.target_models[fit_number]
            pos,H_W,names,taus,max_y,max_x,min_x,init_arrow,final_arrow=lista[0],lista[1],lista[2],lista[3],lista[4],lista[5],lista[6],lista[7],lista[8]  
            legenda=[]
            for i in range(len(taus)):
                tau=taus[i]
                if tau<0.09:
                    tau*=experiment.factor_low
                    legenda.append(rf'$\tau {i+1}$ = '+'{:.2f}'.format(round(tau,2))+' '+experiment.time_unit_low)
                elif tau>999:
                    if tau>1E12:
                        legenda.append(r'$\tau$ = inf')
                    else:    
                        tau/=experiment.factor_high
                        legenda.append(rf'$\tau {i+1}$ = '+'{:.2f}'.format(round(tau,2))+' '+experiment.time_unit_high)
                else:
                    legenda.append(rf'$\tau {i+1}$ = '+'{:.2f}'.format(round(tau,2))+' '+experiment.time_unit)
            figM,ax=plt.subplots(1,1,figsize=(8,8))
            widget=Grapth(figM,Window)
            def pltotModel():
                ax.clear()
                colores=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']         
                for i in range(len(pos)):
                    if ticks_pop.checkState() == 2:
                        rect = Rectangle((pos[i][0],max_y-pos[i][1]),H_W[i][1],H_W[i][0]/2,linewidth=1,edgecolor='k',facecolor=colores[i])
                        ax.add_patch(rect)
                    if ticks_res.checkState() == 2:    
                        plt.annotate(legenda[i],(pos[i][0]+H_W[i][1]*1.05,max_y-pos[i][1]),size=14)
                    if ticks_names.checkState() == 2: 
                        plt.annotate(names[i],(pos[i][0]+H_W[i][1]*0.05,max_y-pos[i][1]+H_W[i][0]*0.1),size=14)
                if ticks_arr.checkState() == 2: 
                    for i in range(len(init_arrow)):
                        if pos[i][0]>pos[i+1][0]:
                            line = plt.arrow(init_arrow[i][0],init_arrow[i][1],-final_arrow[i][0],final_arrow[i][1],width=5,color='darkblue')
                        else:
                            line = plt.arrow(init_arrow[i][0],init_arrow[i][1],final_arrow[i][0],final_arrow[i][1],width=5,color='darkblue')
                        ax.add_patch(line)
                if ticks_exci.checkState() == 2:
                    line = plt.arrow(pos[0][0]+H_W[0][1]*0.33,0,0,(max_y-pos[0][1])-H_W[0][0]*0.95,width=5,color='darkblue',zorder=-1)
                    ax.add_patch(line)
                plt.xlim(min_x-max_x*0.1,max_x+max_x*0.1)
                plt.ylim(-1,max_y)
                ax.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                figM.canvas.draw()
            h_layout=QHBoxLayout(self) 
            h_layout.addWidget(widget)
            v_layout=QVBoxLayout(self)
            v_layout.setContentsMargins(10,50,10,50)
            v_layout.setAlignment(Qt.AlignTop)
            number=QLabel(f'Plot Options',Window)
            v_layout.addWidget(number)
            ticks_names=QCheckBox('names',Window)
            ticks_names.setCheckState(2)
            v_layout.addWidget(ticks_names)
            ticks_res=QCheckBox('results',Window)
            v_layout.addWidget(ticks_res)
            ticks_res.setCheckState(2)
            ticks_pop=QCheckBox('populations',Window)
            v_layout.addWidget(ticks_pop)
            ticks_pop.setCheckState(2)
            ticks_arr=QCheckBox('arrows',Window)
            v_layout.addWidget(ticks_arr)
            ticks_arr.setCheckState(2)
            ticks_exci=QCheckBox('Excitation',Window)
            v_layout.addWidget(ticks_exci)
            ticks_exci.setCheckState(2)
            btn = QPushButton("re-Plot", self) 
            btn.clicked.connect(pltotModel)
            v_layout.addWidget(btn)
            btn_close = QPushButton("close", self) 
            btn_close.clicked.connect(Window.close)
            v_layout.addWidget(btn_close)
            h_layout.addLayout(v_layout)
            Window.layout().addLayout(h_layout)
            pltotModel()
            Window.resize(600,600)
            self.mdiArea.addSubWindow(Window)
            Window.show()
        except:
            self.messageError()
        
    def modifiedModelFunc(self):
        '''modified current model and tranfer to table tf_eps_table'''
        size=[self.mdiArea.size().width(),self.mdiArea.size().height()]
        widget = self.model.manualModelBuild()
        subWindow = QMdiSubWindow(self)
        btn_set = QPushButton("Set the Model", widget) 
        btn_set.clicked.connect(lambda:self.setModelParamsFunc(subWindow))
        btn = QPushButton("Save Model", widget) 
        btn.clicked.connect(lambda:self.saveModel())
        self.formatSubWindow(subWindow,widget,'Dessign Model Window',close_button=True,extra_button=[btn_set,btn])
        subWindow.resize(size[0]/2,size[1]/2)
    

    def colorPlotChangeFunc(self):
        try:
            self.color_base_window.close()
        except:
            pass
        self.color_base_window = QMdiSubWindow(self)
        self.color_base_window.setObjectName('Type of color map selection')
        self.color_base_window.setWindowTitle('Type of color map selection')
        main_layout=QHBoxLayout()
        group=QButtonGroup(self.color_base_window)
        self.color_base_window.layout().addLayout(main_layout)
        widget_dict={}
        label = QLabel('---------------------', self.color_base_window)
        layout=QVBoxLayout()
        layout.addWidget(label)
        layout.setAlignment(Qt.AlignTop)
        for i,ii in enumerate(self.cmaps.keys()):
            widget_dict[ii]=QRadioButton(ii,self.color_base_window)
            widget_dict[ii].setObjectName(ii)
            widget_dict[ii].toggled.connect(lambda:self.displayColorMap(group))
            group.addButton(widget_dict[ii])
            layout.addWidget(widget_dict[ii])
            label = QLabel('---------------------', self.color_base_window)
            layout.addWidget(label)
        main_layout.addLayout(layout)
        main_layout.setContentsMargins(10,0,10,0)
        self.mdiArea.addSubWindow(self.color_base_window)
        btn_close = QPushButton("Close", self.color_base_window) 
        self.color_base_window.layout().addWidget(btn_close)
        btn_close.clicked.connect(self.color_base_window.close)
        self.color_base_window.show()
        self.repaint()
    
    def displayColorMap(self,groups):
        try:
            plt.close(self.fig_color)
            self.color_sub_window.close()
        except:
            pass
        for i,ii in enumerate(groups.buttons()):
            if ii.isChecked():
                name=ii.text()
        self.fig_color=self.plotColorMap(name)
        widget=Grapth(self.fig_color,self,False)
        self.color_sub_window=QMdiSubWindow(self)
        self.color_sub_window.setObjectName('Color map selection')
        self.color_sub_window.setWindowTitle('Color map selection')
        label = QLabel('Color map selection', self.color_sub_window)
        widget_dict={}
        layout=QVBoxLayout()
        layout.addWidget(label)
        layout.setAlignment(Qt.AlignTop)
        group_c=QButtonGroup(self.color_sub_window)
        layout.setContentsMargins(10,30,10,0)
        for i,ii in enumerate(self.cmaps[name]):
            widget_dict[ii]=QRadioButton(ii,self.color_sub_window)
            widget_dict[ii].setObjectName(ii)
            layout.addWidget(widget_dict[ii])
            group_c.addButton(widget_dict[ii])
        label2 = QLabel('', self.color_sub_window)    
        layout.addWidget(label2)
        
        self.color_sub_window.setWidget(widget)
        widget.setLayout(QVBoxLayout()) 
#            widget=self.formatSubWindow(subWindow,widget,'Decay Associted Spectra')
        main=QHBoxLayout()
        main.addWidget(widget)
        main.addLayout(layout)
        
        replot=QPushButton('Select', self.color_sub_window)
        replot.clicked.connect(lambda: setColorMap(group_c))
        layout.addWidget(replot)
        reverse=QCheckBox('reverse map',self.color_sub_window)
        layout.addWidget(reverse)
        self.color_sub_window.layout().addLayout(main)
        self.mdiArea.addSubWindow(self.color_sub_window)
        self.color_sub_window.show()
        def setColorMap(group_c):
            self.color_base_window.close()
            self.color_sub_window.close()
            for ii in group_c.buttons():
                if ii.isChecked():
                    name=ii.text()
            if reverse.isChecked():
                name=name+'_r'
            experiment.color_map=name
    
    def plotColorMap(self,cmap_category):
        cmaps=self.cmaps
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))
        cmap_list=cmaps[cmap_category]
        fig, axes = plt.subplots(nrows=len(cmap_list))
        fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
        axes[0].set_title(cmap_category + ' colormaps', fontsize=14)
    
        for ax, name in zip(axes, cmap_list):
            ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
            pos = list(ax.get_position().bounds)
            x_text = pos[0] - 0.01
            y_text = pos[1] + pos[3]/2.
            fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)
    
        # Turn off *all* ticks & spines, not just the ones with colormaps.
        for ax in axes:
            ax.set_axis_off()
        return fig
        
#        for cmap_category, cmap_list in cmaps.items():
#            plot_color_gradients(cmap_category)
#        
#        plt.show()
    
    def printGeneralReport(self):
        if self.initialize_global_fit:
            try:
                size= self.report_subWindow.size()
                self.report_subWindow.close()
            except:
                size=QSize(550, 400)
            self.report_subWindow = QMdiSubWindow(self)			
            contents=self.formatSubWindow(self.report_subWindow,QTextEdit(),'Report of experiment')
            text=experiment.printGeneralReport()
            contents.setText(text)
            self.report_subWindow.resize(size)
            btn_comments = QPushButton("Add comments", self.report_subWindow) 
            self.report_subWindow.layout().addWidget(btn_comments)
            btn_comments.clicked.connect(self.generalReportComments)
            btn = QPushButton("Update report", self.report_subWindow) 
            self.report_subWindow.layout().addWidget(btn)
            btn.clicked.connect(self.printGeneralReport)
            btn_close = QPushButton("Close", self.report_subWindow) 
            self.report_subWindow.layout().addWidget(btn_close)
            btn_close.clicked.connect(self.report_subWindow.close)
        else:
            self.messageError()
    
    def generalReportComments(self):
        comment_subWindow = QMdiSubWindow(self)	
        def addComment():
            experiment.general_report['User comments'].append('\t'+'\n\t'.join(contents.toPlainText().split('\n'))) 
            self.printGeneralReport()
            comment_subWindow.close()
        btn_comments = QPushButton("Add comments", comment_subWindow) 
        btn_comments.clicked.connect(addComment)	
        contents=self.formatSubWindow(comment_subWindow,QTextEdit(),'Add comments',close_button=True,extra_button=btn_comments)
    
        
    def oneTraceFit(self):
        try:
            if experiment.data_before_first_selection is None:  
                x=experiment.x
                data=experiment.data
                self.wavelength_oneFit=experiment.wavelength
            else:
                x=experiment.x
                data=experiment.data_before_first_selection
                self.wavelength_oneFit=experiment.wavelength_before_first_selection
            texto=self.select_process.currentText()
            if texto=='Deconvolution':
                experiment.deconv=True
            else:
                experiment.deconv=False
            index2ps=np.argmin([abs(i-2) for i in x])
            if index2ps!=0:
                idx=np.unravel_index(np.argmax(abs(data[:index2ps,:]), axis=None), data.shape)
            else:
                idx=[0]
            if experiment.deconv:
                rango=[0,len(x)-1]#define the fitting range
            else:
                rango=[idx[0],len(x)-1]
            self.fig,ax=experiment.plot_spectra(times='auto', ncol=2)
            subWindow = QMdiSubWindow(self)
            self.OneFit_widget=Grapth(self.fig,subWindow,toolbar=False,cursor=True,y=False,click=True,number_click=1)
            subWindow.setWindowTitle("Selecting trace for single fitting")
            def sentToFit():
                if len(self.OneFit_widget.cursorData())==0:
                    error_dialog = QMessageBox(self)
                    error_dialog.setText('Please select a trace by clicking on the figure to fit ')
                    error_dialog.exec()
                else:
                    index=np.argmin([abs(i-self.OneFit_widget.cursorData()[0]) for i in self.wavelength_oneFit])
                    n_exp=int(spin.value())
                    try:
                        average=int(self.textSTAV.text())
                    except:
                        average=0
                    if average>0:
                        self.oneTraceFittingFunc(x,np.mean(data[:,index-average//2:index+average//2],axis=1),n_exp,rango)
                    else:
                        self.oneTraceFittingFunc(x,data[:,index],n_exp,rango)
                    subWindow.close()
            self.textSTAV= QLineEdit('0', self)
            spin= QSpinBox(self)
            spin.setMaximum(10)
            spin.setMinimum(1)
            #rango=[17,149]
            #number_exp=2
            #y=data[:,np.argmin([abs(i-self.OneFit_widget.cursorData()[0]) for i in self.wavelength_oneFit])]
            h_lay=QHBoxLayout(self)
            h_lay.addWidget(self.OneFit_widget)
            v_lay=QVBoxLayout(self)
            v_lay.addWidget(QLabel('Number of\nExponentials',subWindow))
            v_lay.addWidget(spin)
            v_lay.setAlignment(Qt.AlignTop)
            v_lay.setContentsMargins(10,150,10,50)
            v_lay.addWidget(QLabel('Average traces',subWindow))
            v_lay.addWidget(self.textSTAV)
            btn = QPushButton("Select and fit", subWindow) 
            v_lay.addWidget(btn)
            btn.clicked.connect(lambda:sentToFit())
            btn_close = QPushButton("Close", subWindow) 
            v_lay.addWidget(btn_close)
            btn_close.clicked.connect(subWindow.close)
            h_lay.addLayout(v_lay)
            subWindow.layout().addLayout(h_lay)
            subWindow.layout().addWidget(QLabel('Click on the figure to select one trace (left click: select / right click: unselect)',subWindow))
            self.mdiArea.addSubWindow(subWindow)
            subWindow.resize(self.mdiArea.size().width()*0.66,self.mdiArea.size().height()*0.66)
            subWindow.show()
            self.subWindow=QMdiSubWindow(self)
        except:
            self.messageError()
            
    def integralBandFit(self):
        try:
            if experiment.data_before_first_selection is None:  
                    x=experiment.x
                    data=experiment.data
                    self.wavelength_oneFit=experiment.wavelength
            else:
                x=experiment.x
                data=experiment.data_before_first_selection
                self.wavelength_oneFit=experiment.wavelength_before_first_selection
            texto=self.select_process.currentText()
            size=[self.mdiArea.size().width(),self.mdiArea.size().height()]
            if texto=='Deconvolution':
                experiment.deconv=True
            else:
                experiment.deconv=False
            index2ps=np.argmin([abs(i-2) for i in x])
            if index2ps!=0:
                idx=np.unravel_index(np.argmax(abs(data[:index2ps,:]), axis=None), data.shape)
            else:
                idx=[0]
            if experiment.deconv:
                rango=[0,len(x)-1]#define the fitting range
            else:
                rango=[idx[0],len(x)-1]
            self.fig,ax=experiment.plot_spectra(times='auto', ncol=2)
            subWindow = QMdiSubWindow(self)
            self.integrateFig=Grapth(self.fig,subWindow,toolbar=False,cursor=True,y=False,click=True,number_click=2)
            subWindow.setWindowTitle("Selecting area for integral band analysis")
            def inteegrate():
                if len(self.integrateFig.cursorData())==0:
                    error_dialog = QMessageBox(self)
                    error_dialog.setText('Please select two traces by clicking on the figure to define an integration area')
                    error_dialog.exec()
                else:
                    indexes=sorted([np.argmin([abs(i-ii) for ii in self.wavelength_oneFit]) for i in self.integrateFig.cursorData()])
                    areas=np.array([np.trapz(data[i,indexes[0]:indexes[1]], x=self.wavelength_oneFit[indexes[0]:indexes[1]]) for i in range(len(data))])
                    n_exp=int(spin.value())
                    self.oneTraceFittingFunc(x,areas,n_exp,rango,integral_band=True)
                    subWindow.close()
            spin= QSpinBox(self)
            spin.setMaximum(10)
            spin.setMinimum(1)
            h_lay=QHBoxLayout(self)
            h_lay.addWidget(self.integrateFig)
            v_lay=QVBoxLayout(self)
            v_lay.addWidget(QLabel('Number of\nExponentials',subWindow))
            v_lay.addWidget(spin)
            v_lay.setAlignment(Qt.AlignTop)
            v_lay.setContentsMargins(10,150,10,50)
            btn = QPushButton("Integrate and fit", subWindow) 
            v_lay.addWidget(btn)
            btn.clicked.connect(lambda: inteegrate())
            btn_close = QPushButton("Close", subWindow) 
            v_lay.addWidget(btn_close)
            btn_close.clicked.connect(subWindow.close)
            h_lay.addLayout(v_lay)
            subWindow.layout().addLayout(h_lay)
            subWindow.layout().addWidget(QLabel('Click on the figure to select the integration area (left click: select / right click: unselect)',subWindow))
            self.mdiArea.addSubWindow(subWindow)
            subWindow.resize(size[0]*0.66,size[1]*0.66)
            subWindow.show()
            self.subWindow=QMdiSubWindow(self)
        except:
            self.messageError()
    
    def oneTraceFittingFunc(self,x,y,number_exp,rango,params=None,integral_band=False):
        try:
            try:
                self.subWindow_OneTrace_Fit.close()
            except:
                self.check_inf=QCheckBox('Include tau_inf',self)
                self.check_inf.setCheckState(2)
                if experiment.deconv==False:
                    self.check_inf.setVisible(False)
#            self.subWindow.close()
            self.subWindow_OneTrace_Fit = QMdiSubWindow(self)
            self.subWindow_OneTrace_Fit.setWindowTitle("'Fitting One trace with %i exponentials" %number_exp)
            timelabel='Time ('+experiment.time_unit+')'
            deltalabel=r'$\Delta$A'
            if integral_band:
                 w_idx=sorted([np.argmin([abs(i-ii) for ii in self.wavelength_oneFit]) for i in self.integrateFig.cursorData()])
                 wave_label=f'Integral band {round(self.wavelength_oneFit[w_idx][0])}-{round(self.wavelength_oneFit[w_idx][1])} {experiment.wavelength_unit}'
            else:
                w_idx=np.argmin([abs(i-self.OneFit_widget.cursorData()[0]) for i in self.wavelength_oneFit])
                wave_label=f'{round(self.wavelength_oneFit[w_idx])} {experiment.wavelength_unit}'
                val=int(self.textSTAV.text())
                if val>0:
                    wave_label=wave_label+f' Average {val} point' 
            figwave,ax = plt.subplots(2,1,figsize=(4,3),sharex=True,gridspec_kw={'height_ratios': [1, 5]})
            ax[1].plot(x,y,marker='o',alpha=0.6,ms=4,ls='',label=wave_label)
            ax[1].axvline(x[rango[0]],color='b', alpha=0.7,zorder=np.inf)
            ax[1].axvline(x[rango[1]],color='b', alpha=0.7,zorder=np.inf)
            ax[1].set_ylim(np.min(y)-abs(np.min(y)*0.1),np.max(y)+np.max(y)*0.1)
            ax[1].set_xlim(x[0]-x[-1]/50,x[-1]+x[-1]/50)
            self.out_one_fit,fit,res=self.OneTraceFitModel(x[rango[0]:rango[1]+1],y[rango[0]:rango[1]+1],number_exp,params=params)
            ax[0].plot(x[rango[0]:rango[1]+1],res,marker='o',alpha=0.6,ms=4,ls='')
            ax[1].plot(x[rango[0]:rango[1]+1], fit, 'r-', label='best fit')
            ax[1].legend()
            self.formatAxes(ax[1],timelabel,deltalabel)
            ax[0].axhline(linewidth=1,linestyle='--', color='k')
            ax[0].ticklabel_format(style='sci',axis='y')
            ax[0].set_ylabel('Residues',size=12)
            ax[0].minorticks_on()
            ax[0].axes.tick_params(which='both',direction='in',top=True,right=True,labelsize=12)          
            plt.subplots_adjust(bottom=0.1,left=0.18)
            self.One_trace_widget=Grapth(figwave,self.subWindow_OneTrace_Fit,toolbar=True,cursor=True,y=True,number_click=2,)
            layout=QHBoxLayout(self.subWindow_OneTrace_Fit)
            self.One_trace_widget.setMinimumWidth(650)
            self.One_trace_widget.setMinimumHeight(500)
            layout.setStretch(0,5)
            layout.setStretch(1,5)
            sub_layout=QVBoxLayout(self.subWindow_OneTrace_Fit)
            report=QTextEdit(self.subWindow_OneTrace_Fit)
            report.setText(fit_report(self.out_one_fit))
            sub_layout.addWidget(report)
            report.setMinimumWidth(400)
            report.setMinimumHeight(380)
            combo = QComboBox(self)
            if experiment.deconv:
                texto1="Deconvolution"
                texto2="No deconvolution"
                self.names=['t0','fwhm']+['tau_%i' %(i+1) for i in range(number_exp)]
            else:
                texto1="No deconvolution"
                texto2="Deconvolution"
                self.names=['t0']+['tau_%i' %(i+1) for i in range(number_exp)]
            combo.addItem(texto1)
            combo.addItem(texto2)
            self.previous_exp_onefit=number_exp
            self.one_par_table = QTableWidget(self.subWindow_OneTrace_Fit)
            self.one_par_table.setRowCount(2)
            self.one_par_table.setColumnCount(len(self.names))
            self.one_par_table.setHorizontalHeaderLabels(self.names);
            self.one_par_table.verticalHeader().setVisible(False);
            for i,name in enumerate(self.names):
                 it= QTableWidgetItem(str(round(self.out_one_fit.params[name].value,4)))
                 fix=QCheckBox('fix',self)
                 if self.out_one_fit.params[name].vary==False:
                     fix.setCheckState(2)
                 self.one_par_table.setColumnWidth(i,65)
                 self.one_par_table.setItem(0,i,it)
                 self.one_par_table.setCellWidget(1,i,fix)
            self.one_par_table.show()
            sub_layout.addWidget(self.one_par_table)
            label = QLabel('Number of exponential:', self)
            self.exp_spin= QSpinBox(self)
            self.exp_spin.setMaximum(6)
            self.exp_spin.setMinimum(1)
            self.exp_spin.setValue(number_exp)
#                update=GenericWorker(self.OneFitTableUpdate,self.exp_spin.value,number_exp)
            self.exp_spin.valueChanged.connect(lambda:self.OneFitTableUpdate(self.exp_spin.value(),number_exp))
#                update.start(str(exp_spin.valueChanged))
            combo.currentIndexChanged.connect(lambda:self.selectProcessOneFit(combo.currentText(),self.exp_spin.value()))
            layout_reFit=QHBoxLayout(self.subWindow_OneTrace_Fit)
            layout_reFit.addWidget(combo)
            layout_reFit.addWidget(label)
            layout_reFit.addWidget(self.exp_spin)
            layout_reFit.addWidget(self.check_inf)
            btn = QPushButton("Re-Fit with new conditions", self.subWindow_OneTrace_Fit) 
            self.subWindow_OneTrace_Fit.layout().addWidget(btn)
            btn.clicked.connect(lambda:self.OneTraceReFit(x,y,rango,integral_band))
            btn_set_params= QPushButton("Set this set of parameter as Initial conditions", self.subWindow_OneTrace_Fit) 
            self.subWindow_OneTrace_Fit.layout().addWidget(btn_set_params)
            btn_set_params.clicked.connect(self.oenFitSetParams)
            combo_save = QComboBox(self)
            combo_save.addItem('Save')
            if integral_band:
                 combo_save.addItem('Save and Select new integral band areas')
            else:
                combo_save.addItem('Save and Select new wavelength')
            combo_save.addItem('Save and exit')    
            combo_save.activated.connect(lambda: self.saveSingleFit((x,y),self.out_one_fit.params,rango,(fit,res),wave_label,combo_save,number_exp,fit_report(self.out_one_fit)))
            btn_close = QPushButton("Close", self.subWindow_OneTrace_Fit) 
            btn_close.clicked.connect(self.subWindow_OneTrace_Fit.close)
            sub_layout.addLayout(layout_reFit)
            sub_layout.addWidget(btn)
            sub_layout.addWidget(combo_save)
            sub_layout.addWidget(btn_set_params)
            sub_layout.addWidget(btn_close)
            layout.addWidget(self.One_trace_widget)
            layout.addLayout(sub_layout)
            self.subWindow_OneTrace_Fit.layout().addLayout(layout)
            self.mdiArea.addSubWindow(self.subWindow_OneTrace_Fit)
            self.subWindow_OneTrace_Fit.resize(1050,620)
            self.subWindow_OneTrace_Fit.show()
#                
        except:
            self.messageError()
    def saveSingleFit(self,data,params,rango,fit_res,detail,combo,number_exp,report):
        keys= [i for i in experiment.single_fits.keys()]
        if len(keys)==0:
            key=1
        else:
            key=keys[-1]+1
        experiment.single_fits[key]={'data':data,'fit':fit_res[0],'res':fit_res[1],'rango':rango,'params':params,'detail':detail,'n_exp':number_exp,'report':report}
        self.printMessage('Fit saved')
        if 'exit' in combo.currentText():
            self.subWindow_OneTrace_Fit.close()
        elif 'integral' in combo.currentText():
            self.subWindow_OneTrace_Fit.close()
            self.integralBandFit()
        elif 'wavelength' in combo.currentText():
            self.subWindow_OneTrace_Fit.close()
            self.oneTraceFit()
    
    def printMessage(self,mensaje):
        printm=True
        if type(mensaje) != str:
            try:
                mensaje=str(mensaje)
            except:
                printm=False
        if printm:
            self.message.setText(mensaje)
            self.mes_t=ThreadMessage()
            self.mes_t.finished.connect(self.messageReset)
            self.mes_t.start()
            
    def selectProcessOneFit(self,text,number_exp):
        try:
            self.select_process.setCurrentText(text)
            if experiment.deconv:
               self. names=['t0','fwhm']+['tau_%i' %(i+1) for i in range(number_exp)]
               self.check_inf.show()
            else:
                self.names=['t0']+['tau_%i' %(i+1) for i in range(number_exp)]
                self.check_inf.setVisible(False)
            self.one_par_table.setColumnCount(len(self.names))
            self.one_par_table.setHorizontalHeaderLabels(self.names);
            self.one_par_table.verticalHeader().setVisible(False);
            for i,name in enumerate(self.names):
                 fix=QCheckBox('fix',self) 
                 try:
                     it= QTableWidgetItem(str(round(self.out_one_fit.params[name].value,4)))
                     if self.out_one_fit.params[name].vary ==False:
                         fix.setCheckState(2)
                 except:
                     it=QTableWidgetItem('')
                 self.one_par_table.setColumnWidth(i,55)
                 self.one_par_table.setCellWidget(1,i,fix)
                 self.one_par_table.setItem(0,i,it)
        except:
            self.messageError() 
        
    def OneFitTableUpdate(self,val,number_exp):         
        number=self.one_par_table.columnCount()
        print(val)
        print(number_exp)
        if number_exp==val:
            print('nul')
            if experiment.deconv:
                self.names=['t0','fwhm']+['tau_%i' %(i+1) for i in range(number_exp)]
            else:
                self.names=['t0']+['tau_%i' %(i+1) for i in range(number_exp)]
            self.one_par_table.setColumnCount(len(self.names))
            self.previous_exp_onefit=val
        elif val>self.previous_exp_onefit:
            print('plus')
            self.one_par_table.setColumnCount(number+1)
            self.names=self.names+['tau_'+str(val)]
            self.previous_exp_onefit=val
        else:
            print('del')
            self.one_par_table.setColumnCount(number-1)
            del self.names[-1]
            self.previous_exp_onefit=val
        for i,name in enumerate(self.names):
             fix=QCheckBox('fix',self) 
             try:
                 it= QTableWidgetItem(str(round(self.out_one_fit.params[name].value,4)))
                 if self.out_one_fit.params[name].vary ==False:
                         fix.setCheckState(2)
             except:
                 it=QTableWidgetItem('')
             self.one_par_table.setItem(0,i,it)
             self.one_par_table.setCellWidget(1,i,fix)
             self.one_par_table.setColumnWidth(i,55)
        self.one_par_table.setHorizontalHeaderLabels(self.names);
  
            
    def expGaussConv_bis(self,time,tau,sigma):
        return 0.5*np.exp(-tau*time + sigma**2*tau**2/2 )*(1+erf((time-sigma**2*tau)/(sigma*2**0.5)))
    
    def exp1_bis(self,x, tau):
        "basic gaussian"
        return np.exp(-x/tau) 
    
    def expN_bis (self,params,time,exp_no,residue=None):
        """values should be a list of list containing the pre_exps and taus values"""
        y0 = params['y0']
        t0 = params['t0']
        values=[[params['pre_exp_'+ str(i+1)].value,params['tau_'+str(i+1)].value] for i in range(exp_no)] 
        if residue is not None:
            return residue-(y0+sum([pre_exp*self.exp1_bis(time-t0,tau) for pre_exp,tau in values]) )      
        else:
            return y0+sum([pre_exp*self.exp1_bis(time-t0,tau) for pre_exp,tau in values])              
    
    def expNGauss_bis (self,params,time,exp_no,residue=None,tau_inf=True):
        try:
            """values should be a list of list containing the pre_exps and taus values"""
            tau_inf=1E12
            y0 = params['y0']
            t0 = params['t0']
            fwhm=params['fwhm']
            yinf=params['yinf']
            values=[[params['pre_exp_'+ str(i+1)].value,params['tau_' +str(i+1)].value] for i in range(exp_no)] 
            if residue is not None:
                if tau_inf:
#                    print('aaaaa')
                    return residue-(y0+sum([(pre_exp)*self.expGaussConv_bis(time-t0,1/tau,fwhm/2.35482) for pre_exp,tau in values])\
                                    +(yinf)*self.expGaussConv_bis(time-t0,1/tau_inf,fwhm/2.35482))
                else:
#                    print('b')
                    return residue-(y0+sum([(pre_exp)*self.expGaussConv_bis(time-t0,1/tau,fwhm/2.35482) for pre_exp,tau in values]))
            else:
                if tau_inf:
#                    print('c')
                    return y0+sum([(pre_exp)*self.expGaussConv_bis(time-t0,1/tau,fwhm/2.35482) for pre_exp,tau in values])\
                            +(yinf)*self.expGaussConv_bis(time-t0,1/tau_inf,fwhm/2.35482)
                else:
#                    print('d')
                    return y0+sum([(pre_exp)*self.expGaussConv_bis(time-t0,1/tau,fwhm/2.35482) for pre_exp,tau in values])
        except:
            GVD_dialog = QMessageBox(self)
            GVD_dialog.setText('Unable to Fit data')
            GVD_dialog.exec()
        
    def OneTraceFitModel(self,x,y,exp_number,params):
        try:
            initial_params={'tau_'+str (i+1):np.max(x)/(((2*exp_number-i+1)**2*(exp_number-i+1))) for i in range(exp_number)}
            if experiment.deconv:
                initial_params['t0']=0
                initial_params['fwhm']=0.12
            else:
                initial_params['t0']=x[0]
            variation={i:True for i in initial_params.keys()}
            if params is not None:
                for i in params.keys(): 
                    if params[i] is not None:
                        initial_params[i]=params[i][0]
                        variation[i]=params[i][1]
            mini=abs(np.min(y))
            maxi=abs(np.max(y))
            if mini>maxi:
                value=np.min(y)
            else:
                value=np.max(y)
            params=Parameters()
            if experiment.deconv:
                    params.add_many(('t0', initial_params['t0'], variation['t0'],  np.min(x), None, None, None))
                    params.add_many(('y0', y[0],True, None, None, None, None))
                    params.add_many(('fwhm', initial_params['fwhm'], variation['fwhm'], 0.05, 0.25, None, None))         
                    params.add_many(('yinf' , value/1000, True, None, None, None, None))
                    minimo=0.05
            else:
                params.add_many(('t0', initial_params['t0'], variation['t0'],  np.min(x), None, None, None))
                params.add_many(('y0', y[-1],True, None, None, None, None))
                minimo=0.05
            for i in range (exp_number):
                    # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
                    params.add_many(('pre_exp_' +str (i+1), value/exp_number, True, None, None, None, None),
                                    ('tau_' +str (i+1), initial_params['tau_'+str (i+1)], variation['tau_'+str (i+1)], minimo, None, None, None))
                    minimo=params['tau_'+str (i+1)].value
            if experiment.deconv:
                if self.check_inf.checkState()==0:
                    tau_inf=True
                else:
                    tau_inf=False
                resultados=minimize(self.expNGauss_bis,params,args=(x,exp_number,y,tau_inf),nan_policy='propagate')
            else:
                resultados=minimize(self.expN_bis,params,args=(x,exp_number,y),nan_policy='propagate')
            if experiment.deconv:
                fit=self.expNGauss_bis(resultados.params,x,exp_number,tau_inf=tau_inf)
            else:
                fit= self.expN_bis(resultados.params,x,exp_number)
            return resultados,fit,y-fit
        except:
            self.messageError()
    
    def OneTraceReFit(self,x,y,rango,integral_band):
            if len(self.One_trace_widget.cursorData()[0])==2:
                rango=sorted([np.argmin([abs(i-ii) for ii in x]) for i in self.One_trace_widget.cursorData()[0]])
            elif len(self.One_trace_widget.cursorData()[0])==1:
                value=np.argmin([abs(self.One_trace_widget.cursorData()[0]-ii) for ii in x])
                rango=sorted([value,rango[np.argmax((abs(rango[0]-value),abs(rango[1]-value)))]])
            else:
                pass
            number_exp=int(self.exp_spin.value())
            values=[self.one_par_table.item(0,i).text() for i in range(self.one_par_table.columnCount())]
            variations=[False if self.one_par_table.cellWidget(1,i).isChecked() else True for i in range(self.one_par_table.columnCount()) ]
            values=[float(i) if i is not '' else None for i in values]
            params={self.names[i]:[values[i],variations[i]] for i in range(len(values))}
            self.oneTraceFittingFunc(x,y,number_exp,rango,params)
    
    def oenFitSetParams(self):
        try:
             self.corrected_GVD=experiment.GVD_correction
             values=[self.one_par_table.item(0,i).text() for i in range(self.one_par_table.columnCount())]
             values=[float(i) if i is not '' else None for i in values]
             if values[-1]==None:
                 del values[-1]
             changes=[False if self.one_par_table.cellWidget(1,i).isChecked() else True for i in range(self.one_par_table.columnCount()) ]
             if experiment.deconv:
                 changes[1]=False
                 if self.corrected_GVD==False:
                     GVD_dialog = QMessageBox(self)
                     GVD_dialog.setText('Non of the GVD data correction functions has been Run; \nplease check box if data was corrected previously')
                     self.check=QCheckBox('GVD corrected',self)
                     GVD_dialog.setCheckBox(self.check)
                     GVD_dialog.exec()
                 else:
                    self.check=QCheckBox('GVD corrected',self)
                    self.check.setCheckState(2)
             else:
                 changes[0]=False
             if None in values:
                error_dialog = QMessageBox(self)
                error_dialog.setText('There is an empty cell in the parameter table please fill this with a value')
                error_dialog.exec()
             else:
                self.param_table = QTableWidget(len(values),3,self)
                for i in range(len(values)):
                    namet=QLabel(self.names[i])
                    change=changes[i]
                    valor=QTableWidgetItem(str(values[i]))
                    combo = QComboBox()
                    if change==False:
                        combo.addItem("Fix")
                        combo.addItem("Optimize")
                    else:    
                        combo.addItem("Optimize")
                        combo.addItem("Fix")
                    self.param_table.setCellWidget(i,0,namet)
                    self.param_table.setItem(i,1,valor)     
                    self.param_table.setCellWidget(i,2,combo)      
                self.setParams()
        except:
            self.messageError()
    
    def initFianlWAveFunc(self):
        self.noZones()
        self.edit_general['init_fianl_wave_edit']=True
        self.repaint()
    
    def button104Func(self):
        try:
            initial=float(self.text104.text())
            final=float(self.text114.text())
            if experiment.data_before_first_selection is None:  
                lungo=experiment.data.shape[1]
            else:
                lungo=experiment.wavelength_before_first_selection
            cal=np.linspace(initial,final,lungo)
            if experiment.data_before_first_selection is None:  
                experiment.wavelength=cal
            else:
                experiment.wavelength_before_first_selection=cal
            experiment.general_report['Preprocessing']['Calibration']=f'Linear vector from {initial} to {final}'
            self.printMessage('Wavelength calibrated')
        except:
            self.messageError()
    
    def IRFsolventFunc(self):
        try:
            fname = QFileDialog.getOpenFileName(self, 'Open IRF file', 
                'c:\\',"files (*.abt *.asc *.csv *.txt *.dat *.lvm)")
            self.IRF_file=fname[0]
            if self.manual_load==True:
                self.button07Func(True)
            elif self.manual_load==False:
                self.button17Func(True)
            else:
               error_dialog = QMessageBox(self)
               error_dialog.setText('Please Load data before fitting IRF')
               error_dialog.exec() 
            if experiment.data_before_first_selection is None:  
                x=self.IRF_experiment.x
                data=self.IRF_experiment.data
                self.wavelength_irf=self.IRF_experiment.wavelength
            self.subWindow = QMdiSubWindow(self)
            points=[i for i in self.IRF_experiment.x[::2]]
            self.fig,_=self.IRF_experiment.plot_spectra(times=points, ncol=2)
            self.IRF_widget=Grapth(self.fig,self.subWindow,toolbar=False,cursor=True,y=False,click=True,number_click=1)
            self.formatSubWindow(self.subWindow,self.IRF_widget,'Fitting solvent IRF')
            btn = QPushButton("Select wavelength", self.subWindow) 
            self.subWindow.layout().addWidget(btn)
            btn.clicked.connect(lambda:self.IRFFitFunc(x,data[:,np.argmin([abs(i-self.IRF_widget.cursorData()[0]) for i in self.wavelength_irf])],[0,len(x)-1]))
            btn_close = QPushButton("Close", self.subWindow) 
            self.subWindow.layout().addWidget(btn_close)
            btn_close.clicked.connect(self.subWindow.close)
        except:
            self.messageError()
        
    def IRFRamanFunc(self):
        try:
            if experiment.data_before_first_selection is None:  
                x=experiment.x
                data=experiment.data
                self.wavelength_irf=experiment.wavelength
            else:
                x=experiment.x
                data=experiment.data_before_first_selection
                self.wavelength_irf=experiment.wavelength_before_first_selection
            index2ps=np.argmin([abs(i-2) for i in x])
            idx=np.unravel_index(np.argmax(abs(data[:index2ps,:]), axis=None), data.shape)
            point=[np.argmin(abs(data[:idx[0],idx[1]]-i)) for i in\
                   np.linspace(data[0,idx[1]],np.max(data[idx[0],idx[1]]),8)]
            time_points = list(np.sort(np.array(x)[point]))
            self.fig,ax=experiment.plot_spectra(times=time_points, ncol=2)
            self.subWindow = QMdiSubWindow(self)
            self.IRF_widget=Grapth(self.fig,self.subWindow,toolbar=False,cursor=True,y=False,click=True,number_click=1)
            self.formatSubWindow(self.subWindow,self.IRF_widget,'Fitting Raman peak IRF')
            btn = QPushButton("Select wavelength", self.subWindow) 
            self.subWindow.layout().addWidget(btn)
            btn.clicked.connect(lambda:self.IRFFitFunc(x[:index2ps],data[:index2ps,np.argmin([abs(i-self.IRF_widget.cursorData()[0]) for i in self.wavelength_irf])],[0,idx[0]]))
            btn_close = QPushButton("Close", self.subWindow) 
            self.subWindow.layout().addWidget(btn_close)
            btn_close.clicked.connect(self.subWindow.close)
        except:
            self.messageError()
    
    def IRFFitFunc(self,x,y,rango): 
        try:
            if len(self.IRF_widget.cursorData())==0:
                error_dialog = QMessageBox(self)
                error_dialog.setText('Please select the wavelength were the Raman peak is found')
                error_dialog.exec()
            else:
                try:
                    self.subWindow_fitIRF.close()
#                    time.sleep(1)
                except:
                    pass
                self.subWindow.close()
                self.subWindow_fitIRF = QMdiSubWindow(self)
                self.subWindow_fitIRF.setWindowTitle("IRF Fit")
                timelabel='Time ('+experiment.time_unit+')'
                deltalabel=r'$\Delta$A'
                w_idx=np.argmin([abs(i-self.IRF_widget.cursorData()[0]) for i in self.wavelength_irf])
                wave_label=f'{round(self.wavelength_irf[w_idx])} {experiment.wavelength_unit}'
                figwave,ax = plt.subplots(1,figsize=(4,3))
                ax.plot(x,y,label=wave_label)
                ax.axvline(x[rango[0]],color='b', alpha=0.7,zorder=np.inf)
                ax.axvline(x[rango[1]],color='b', alpha=0.7,zorder=np.inf)
                ax.set_ylim(np.min(y)-abs(np.min(y)*0.1),np.max(y)+np.max(y)*0.1)
                ax.set_xlim(x[0]-x[-1]/50,x[-1]+x[-1]/50)
                out=self.fitGaus(x[rango[0]:rango[1]+1],y[rango[0]:rango[1]+1])
                ax.plot(x[rango[0]:rango[1]+1], out.best_fit, 'r-', label='best fit')
                ax.legend()
                self.formatAxes(ax,timelabel,deltalabel)
                self.IRF_grapth=Grapth(figwave,self.subWindow_fitIRF,toolbar=True,cursor=True,y=True,number_click=2,)
                layout=QHBoxLayout(self.subWindow_fitIRF)
                self.IRF_grapth.setMinimumWidth(550)
                self.IRF_grapth.setMinimumHeight(500)
                layout.setStretch(0,5)
                layout.setStretch(1,5)
                sub_layout=QVBoxLayout(self.subWindow_fitIRF)
                report=QTextEdit(self.subWindow_fitIRF)
                report.setText(out.fit_report(min_correl=0.25))
                sub_layout.addWidget(report)
                report.setMinimumWidth(400)
                report.setMinimumHeight(500)
                btn = QPushButton("Re-Fit with new selected regions", self.subWindow_fitIRF) 
                self.subWindow_fitIRF.layout().addWidget(btn)
                btn.clicked.connect(lambda:self.IRFReFitFunc(x,y,rango))
                btn_close = QPushButton("Close", self.subWindow_fitIRF) 
                self.subWindow_fitIRF.layout().addWidget(btn_close)
                btn_close.clicked.connect(self.subWindow_fitIRF.close)
                sub_layout.addWidget(btn)
                sub_layout.addWidget(btn_close)
                layout.addWidget(self.IRF_grapth)
                layout.addLayout(sub_layout)
                self.subWindow_fitIRF.layout().addLayout(layout)
                self.mdiArea.addSubWindow(self.subWindow_fitIRF)
                self.subWindow_fitIRF.show()
                self.IRF_value=out.params['fwhm'].value
                experiment.IRF_value=out.params['fwhm'].value
                if self.IRF_file is None:
                    experiment.general_report['Preprocessing']['IRF Fit']=f'\tFrom Raman peak, value: {round(self.IRF_value,3)}'
                else:
                    experiment.general_report['Preprocessing']['IRF Fit']=f'\tFrom file: {self.IRF_file}\nvalue: {round(self.IRF_value,3)}'
                try:                    
                    index=experiment.general_report['Sequence of actions'].index('\t--> IRF fitted')
                    del experiment.general_report['Sequence of actions'][index]
                    experiment.general_report['Sequence of actions'].append('\t--> IRF fitted')
                except:
                    experiment.general_report['Sequence of actions'].append('\t--> IRF fitted')
        except:
            self.messageError()
    
    def IRFReFitFunc(self,x,y,rango):
        if len(self.IRF_grapth.cursorData()[0])==2:
            self.IRFFitFunc(x,y,rango)
        elif len(self.IRF_grapth.cursorData()[0])==1:
            value=np.argmin([abs(self.IRF_grapth.cursorData()[0]-ii) for ii in x])
            rango=sorted([value,rango[np.argmax((abs(rango[0]-value),abs(rango[1]-value)))]])
            self.IRFFitFunc(x,y,rango)
        else:
            error_dialog = QMessageBox(self)
            error_dialog.setText('Please select at least one points to have a new range for fitting')
            error_dialog.exec()
    
    def fitGaus(self,x,y):
        mod = GaussianModel()
        pars = mod.guess(y, x=x)
        out = mod.fit(y, pars, x=x)
        return out
    
    def messageError(self):
        if self.initialize_global_fit==False:
            error_dialog = QMessageBox(self)
            error_dialog.setText('Please load data first')
            error_dialog.exec()
        else:
            error_dialog = QMessageBox(self)
            error_dialog.setText('We could not complete the action')
            error_dialog.exec()
    
    def shiftTimeFunc(self):
        self.noZones()
        self.edit_general['time_shift']=True
        self.repaint()
    
    def button_t8Func(self):
        try:
            time=self.text_t8.text()
            number=False
            try:
                time=float(time)
                number=True
            except:
                error_dialog = QMessageBox(self)
                error_dialog.setText('please introduce a number')
                error_dialog.exec()
            if number:
                experiment.shitTime(time)
                self.printMessage('Time shifted')
        except:
            self.messageError()
    
    def GVDGrapthPolFunc(self):
        try:
            experiment.GVDFromPolynom(qt=self.setWindowIcon_path)
            self.corrected_GVD=True
        except:
            self.messageError()
       
    def exploreDataFunc(self):
        try:
            try:
                self.subWindow_explore.close()  
            except:
                pass  
            if experiment.data_before_first_selection is None:  
                self.x=experiment.x
                self.data=experiment.data
                self.wavelength=experiment.wavelength
            else:
                self.x=experiment.x
                self.data=experiment.data_before_first_selection
                self.wavelength=experiment.wavelength_before_first_selection     
            self.figSurface=plt.figure(figsize=(5,3))
            ax=self.figSurface.add_subplot(1,1,1)
            timelabel='Time ('+experiment.time_unit+')'
            ax.set_ylabel(timelabel,size=12)
            wavelabel='Wavelength ('+experiment.wavelength_unit+')'
            deltalabel=r'$\Delta$A'
            ax.set_xlabel(wavelabel,size=12)
            init_time=len(self.x)//5
            init_wave=len(self.wavelength)//5
            
            self.lx = ax.axhline(self.x[init_time],color='r',alpha=0.7,zorder=np.inf)  # the horiz line
            self.ly = ax.axvline(self.wavelength[init_wave],color='r', alpha=0.7,zorder=np.inf)  # the vert line
            ax.axis([self.wavelength[0],self.wavelength[-1], self.x[0]-self.x[-1]/50, self.x[-1]])
            ax.pcolormesh(self.wavelength,self.x,self.data,cmap='RdYlBu_r')
            ax.minorticks_on()
            ax.axes.tick_params(which='both',direction='in',top=True,right=True,labelsize=12)
            cnorm = Normalize(vmin=np.min(self.data),vmax=np.max(self.data))
            cpickmap = plt.cm.ScalarMappable(norm=cnorm,cmap='RdYlBu_r')
            cpickmap.set_array([])
            plt.colorbar(cpickmap,format='%.0e')
            plt.tight_layout()
            self.subWindow_explore = QMdiSubWindow(self)
            self.subWindow_explore.setMaximumHeight(self.window_height-80)
            self.subWindow_explore.setWindowTitle("Explore data Window")
            if self.mdiArea.size().width()<1220:
                width=self.mdiArea.size().width()
            else:
                width=1220
            if self.mdiArea.size().height()<700:
                height=self.mdiArea.size().height()
            else:
                height=700
            self.subWindow_explore.resize(width,height)
            surface=Grapth(self.figSurface,self.subWindow_explore,toolbar=False)
    #        surface=self.formatSubWindow(self.subWindow_explore,surface)
            main_layout= QGridLayout(self.subWindow_explore)
            main_layout.addWidget(surface,0,0)
            
            self.explore_wave = QSlider(Qt.Horizontal,self.subWindow_explore)
            self.explore_wave.setFixedHeight(15)
            self.explore_wave.setFixedWidth(400)
            self.explore_wave.setStyleSheet('background-color:darkRed;border: 2px solid #999999;height: 15px; border-radius: 5px')
            self.explore_wave.setTickPosition(self.explore_wave.TicksBelow)
            self.explore_wave.setValue(round(self.wavelength[init_wave]*20))
            self.explore_wave.setMinimum(self.wavelength[0]*20)
            self.explore_wave.setMaximum(self.wavelength[-1]*20)
            self.explore_wave.valueChanged.connect(self.exploreWaveFunc)
            main_layout.addWidget(self.explore_wave,1,0,alignment=Qt.AlignCenter)
            
            self.explore_time = QSlider(Qt.Vertical,self.subWindow_explore)
            self.explore_time.setFixedWidth(15)
            self.explore_time.setFixedHeight(250)
            self.explore_time.setStyleSheet('background-color:darkRed;border: 2px solid #999999;height: 15px; border-radius: 5px')
            self.explore_time.setTickPosition(self.explore_time.TicksBelow)
            self.explore_time.setValue(round(self.x[init_time]*50))
            self.explore_time.setMinimum(self.x[0]*50)
            self.explore_time.setMaximum(self.x[-1]*50)
            self.explore_time.valueChanged.connect(self.exploreTimeFunc)
            space_label = QLabel('', self)
            #main_layout.addWidget(space_label,0,1)
            main_layout.addWidget(self.explore_time,0,1,alignment=Qt.AlignCenter)
            #main_layout.addWidget(space_label,0,1)
            self.subWindow_explore.layout().addLayout(main_layout)
            self.mdiArea.addSubWindow(self.subWindow_explore)
            
            self.figwave,self.axwave = plt.subplots(1,figsize=(5,3))
            wave_label=f'{round(self.wavelength[init_wave])} {experiment.wavelength_unit}'
            self.l_wave=self.axwave.plot(self.x,self.data[:,init_wave],label=wave_label)
            self.axwave.set_ylim(np.min(self.data)-abs(np.min(self.data)*0.1),np.max(self.data)+np.max(self.data)*0.1)
            self.axwave.set_xlim(self.x[0]-self.x[-1]/50,self.x[-1]+self.x[-1]/50)
            self.legend_wave=self.axwave.legend()
            self.formatAxes(self.axwave,timelabel,deltalabel)
            self.time_g=Grapth(self.figwave,self.subWindow_explore,toolbar=False,cursor=True,click=True)
            
            self.figtime,self.axtime = plt.subplots(1,figsize=(5,3))
            
            time_label=f'{self.x[init_time]} {experiment.time_unit}'
            self.l_time=self.axtime.plot(self.wavelength,self.data[init_time,:],label=time_label)
            self.axtime.set_ylim(np.min(self.data)-abs(np.min(self.data)*0.1),np.max(self.data)+np.max(self.data)*0.1)
            self.axtime.set_xlim(self.wavelength[0]-self.wavelength[-1]/50,self.wavelength[-1]+self.wavelength[-1]/50)
            self.legend_time=self.axtime.legend()
            self.formatAxes(self.axtime,wavelabel,deltalabel)
            self.time_w=Grapth(self.figtime,self.subWindow_explore,toolbar=False,cursor=True,click=True)
            main_layout.addWidget(self.time_w,0, 2)
            main_layout.addWidget(self.time_g,2, 0)
            self.subWindow_explore.show()
            self.log_explore_slider=False
            def logScale():
                if log_scale.checkState() != 2:
                    ax.set_yscale('linear')
                    ax.minorticks_on()
                    self.axwave.set_xscale('linear')
                    self.axwave.minorticks_on()
                    self.axwave.set_xlim(self.x[0]-self.x[-1]/50,self.x[-1]+self.x[-1]/50)
                    self.figwave.canvas.draw()
                    self.figSurface.canvas.draw()
                    self.log_explore_slider=False
                    self.explore_time.setMinimum(self.x[0]*50)
                    self.explore_time.setMaximum(self.x[-1]*50)
                else:
                    ax.set_yscale('symlog',subsy=[2,4,6,8])
                    self.axwave.set_xscale('symlog',subsx=[2,4,6,8])
                    ax.minorticks_on()
                    self.axwave.minorticks_on()
                    ax.set_ylim(self.x[0],self.x[-1])
                    self.axwave.set_xlim(self.x[0]*1.2,self.x[-1]*1.5)
                    self.figwave.canvas.draw()
                    self.figSurface.canvas.draw()
                    self.log_explore_slider=True
                    self.explore_time.setMinimum(0)
                    self.explore_time.setMaximum(len(self.x))
            text_lay = QVBoxLayout(self.subWindow_explore)
            log_scale=QCheckBox('Logarithmic time scale',self)
            log_scale.stateChanged.connect(logScale)
            log_lay=QHBoxLayout(self.subWindow_explore)
            log_lay.setContentsMargins(10,0,0,0)
            log_lay.addWidget(log_scale,alignment=Qt.AlignLeft)
            main_layout.addLayout(log_lay,1,2,alignment=Qt.AlignLeft)
            current_wave_label = QLabel('Current wavelength:', self)
            current_wave_label.setFixedSize(QSize(135, 35))
            space_label1 = QLabel('Resume Of data:', self)
            space_label1.setFixedSize(QSize(135, 35))
            space_label2 = QLabel(f'Number of traces: {len(self.wavelength)}', self)
            space_label2.setFixedSize(QSize(135, 35))
            space_label3 = QLabel(f'Points per trace: {len(self.x)}', self)
            space_label3.setFixedSize(QSize(135, 35))
            lay_but1= QHBoxLayout(self.subWindow_explore)
            button_trace_select = QPushButton('Plot traces', self)
            button_trace_select.setFixedSize(QSize(135, 30))
            button_trace_select.clicked.connect(self.PlotTraceFuncExplore)
            button_trace_del = QPushButton('Unselect traces', self)
            button_trace_del.setFixedSize(QSize(135, 30))
            button_trace_del.clicked.connect(lambda:self.delSelectionFun(self.time_w,self.axtime ))
            lay_but1.addWidget(button_trace_select)
            lay_but1.addWidget(button_trace_del)
            lay_but2=QHBoxLayout(self.subWindow_explore)
            button_spec_select = QPushButton('Plot spectra', self)
            button_spec_select.setFixedSize(QSize(135, 30))
            button_spec_select.clicked.connect(self.PlotSpecFuncExplore)
            button_spec_del = QPushButton('Unselect spectra', self)
            button_spec_del.setFixedSize(QSize(135, 30))
            button_spec_del.clicked.connect(lambda:self.delSelectionFun(self.time_g,self.axwave))
            lay_but2.addWidget(button_spec_select)
            lay_but2.addWidget(button_spec_del)
            text_lay.addLayout(lay_but1)
            text_lay.addLayout(lay_but2)
            self.current_wave_text = QLineEdit(wave_label, self)
            self.current_wave_text.setFixedSize(QSize(135, 35))
            current_time_label =  QLabel('Current time:', self)
            current_time_label.setFixedSize(QSize(135, 35))
            self.current_time_text = QLineEdit(time_label, self)
            self.current_time_text.setFixedSize(QSize(135, 35))
            text1=QVBoxLayout(self.subWindow_explore)
            text1.addWidget(current_wave_label)
            text1.addWidget(self.current_wave_text)
            text1.addWidget(current_time_label) 
            text1.addWidget(self.current_time_text)
            text1.addWidget(space_label1)
            text1.addWidget(space_label2)
            text1.addWidget(space_label3)
            text1.addWidget(space_label)
            select_label =  QLabel('select traces for fit:', self)
            select_label.setFixedSize(QSize(135, 25))
            averige_label =  QLabel('Average points:', self)
            averige_label.setFixedSize(QSize(75, 30))
            self.averige_text = QLineEdit('0', self)
            self.averige_text.setFixedSize(QSize(40, 30))
            av_lay=QHBoxLayout(self.subWindow_explore)
            av_lay.setContentsMargins(10,0,10,0)
            av_lay.addWidget(averige_label,alignment=Qt.AlignCenter)
            av_lay.addWidget(self.averige_text,alignment=Qt.AlignCenter)
            traces_fit = QPushButton('Select', self)
            traces_fit.setFixedSize(QSize(135, 35))
            traces_fit.clicked.connect(self.selectTracesExplore)
            text2=QVBoxLayout(self.subWindow_explore)
            text2.addWidget(select_label,alignment=Qt.AlignTop)
            text2.addLayout(av_lay)
            text2.addWidget(traces_fit,alignment=Qt.AlignTop) 
            text2.setAlignment(Qt.AlignTop)
            text2.setContentsMargins(15,60,0,30)
            text_total=QHBoxLayout(self.subWindow_explore)
            text_total.addLayout(text1)
            text_total.addLayout(text2)
            text_total.setContentsMargins(15,10,0,10)
            text_lay.addLayout(text_total)
            self.image = QLabel(self)
#            self.image.resize(self.pixmapShow.size().width()/5,self.pixmapShow.size().width()/5)
#            self.explore_time.setMinimumHeight(self.pixmapShow.size().width()/5)
#            self.image.setFixedSize(self.pixmapShow.size().width()/5,self.pixmapShow.size().width()/5)
#            w, h= self.image.width(),self.image.height()
            self.image.setPixmap(self.pixmapShow)
            lay_tex_fig=QHBoxLayout(self.subWindow_explore)
            lay_tex_fig.addLayout(text_lay)
            lay_tex_fig.addWidget(self.image,alignment=Qt.AlignCenter)
            main_layout.addLayout(lay_tex_fig,2,2,alignment=Qt.AlignCenter)     
            main_layout.setRowMinimumHeight(0,height/2) 
            main_layout.setRowStretch(0,7)
            main_layout.setRowStretch(1,0)
            main_layout.setRowStretch(2,7)
            main_layout.setColumnStretch(0,7)
            main_layout.setColumnStretch(1,0)
            main_layout.setColumnStretch(2,7)
            self.explore_time.setValue(round(self.x[init_time]*50))
            self.explore_wave.setValue(round(self.wavelength[init_wave]*20))
            self.noZones()
            self.edit_general['number_traces_edit']=True     
            self.repaint()
        except:
            self.messageError()
    
    def selectTracesExplore(self):
        points=sorted(self.time_w.cursorData()[0])
        if len(points)==0:
            error_dialog = QMessageBox(self)
            error_dialog.setText('Please select traces by clicking on Spectrum figure')
            error_dialog.exec()
        else:
            average=int(self.averige_text.text())//2
            experiment.selectTraces(space=points,points=average,avoid_excitation=False)
            self.numberTraces()
            self.printMessage('Traces selected')

    
    def exploreDataOn(self):
        try:
            if self.mdiArea.activeSubWindow()==self.subWindow_explore:
                self.subWindow_explore.close()
                self.exploreDataFunc()
            else:
                self.subWindow_explore.close()
        except:
            pass          
         
    def PlotTraceFuncExplore(self):
        points=self.time_w.cursorData()[0]
        if len(points)==0:
            error_dialog = QMessageBox(self)
            error_dialog.setText('Please select traces by clicking on Spectrum figure')
            error_dialog.exec()
        else:
            experiment.selectTraces(space=sorted(points),points=0,avoid_excitation=False)
            self.PlotTracesFunc()
            self.SelectAllTracesFunc()
    
    def PlotSpecFuncExplore(self):
        points=self.time_g.cursorData()[0]
        if len(points)==0:
            error_dialog = QMessageBox(self)
            error_dialog.setText('Please select spectra by clicking on Trace figure')
            error_dialog.exec()
        else:
            fig,_=experiment.plot_spectra(sorted(points))
            subWindow = QMdiSubWindow(self)
            widget=Grapth(fig,subWindow)
            self.formatSubWindow(subWindow,widget,'Selected Spectra')
       
    def delSelectionFun(self,grapth,axes):
        points=grapth.cursorData()[0]
        if type(points) is not list:
            points=grapth.cursorData()
        if len(points)==0:
            error_dialog = QMessageBox(self)
            error_dialog.setText('No points have been selected')
            error_dialog.exec()
        else:
            for i in range(len(points)):
                del grapth.cursore.datax[-1]
                del grapth.cursore.datay[-1]
                grapth.cursore.scat[-1].remove()
                del grapth.cursore.scat[-1]
                axes.figure.canvas.draw_idle()      
    

    
    def exploreTimeFunc(self):
        if self.log_explore_slider:
             value = self.explore_time.value()
        else:
            value = self.explore_time.value()/50
            value=np.argmin(abs(self.x-value))
            value=int(round(value,1))
        self.lx.set_ydata(self.x[value])
        self.l_time[0].set_ydata(self.data[value,:])
        time_label=f'{self.x[value]} {experiment.time_unit}'
        self.current_time_text.setText(time_label)
        self.axtime.legend([time_label,'_'])
        # redraw canvas while idle
        self.figtime.canvas.draw_idle()
        self.figSurface.canvas.draw_idle()
    
    def exploreWaveFunc(self):
        value = self.explore_wave.value()/20
        value=np.argmin(abs(self.wavelength-value))
        self.ly.set_xdata(self.wavelength[value])
        value=int(round(value,1))
        self.l_wave[0].set_ydata(self.data[:,value])
        wave_label=f'{round(self.wavelength[value],1)} {experiment.wavelength_unit}'
        self.current_wave_text.setText(wave_label)
        self.axwave.legend([wave_label,'_'])
        # redraw canvas while idle
        self.figwave.canvas.draw_idle()
        self.figSurface.canvas.draw_idle()
      
    def formatAxes(self,axes,xlabel,ylabel,size=12,tight=True):
        axes.axhline(linewidth=1,linestyle='--', color='k')
        axes.ticklabel_format(style='sci',axis='y')
        axes.set_ylabel(ylabel,size=size)
        axes.set_xlabel(xlabel,size=size)
        axes.minorticks_on()
        axes.axes.tick_params(which='both',direction='in',top=True,right=True,labelsize=size)
        if tight:
            plt.tight_layout()
    
    def deletePointTimeFunc(self):
        try:
            fig,_=experiment.plotRawData(auto=True)
            self.subWindow_del_time = QMdiSubWindow(self)
            self.widget_del_time=Grapth(fig,self.subWindow_del_time,cursor=True,y=False)
            self.formatSubWindow(self.subWindow_del_time,self.widget_del_time,'Delete Time points')
            btn = QPushButton("Delete selected points", self.subWindow_del_time) 
            self.subWindow_del_time.layout().addWidget(btn)
            btn.clicked.connect(self.deletePointTimeActionFunc)
            btn_close = QPushButton("Close", self.subWindow_del_time) 
            self.subWindow_del_time.layout().addWidget(btn_close)
            btn_close.clicked.connect(self.subWindow_del_time.close)
        except:
            self.messageError()
            
    
    def deletePointTimeActionFunc(self):
        try:
            index=self.widget_del_time.cursorData()
            if len(index)==0:
                error_dialog = QMessageBox(self)
                error_dialog.setText('No points have been selected')
                error_dialog.exec()
            else:
                experiment.delPoints(index)
                self.subWindow_del_time.close()
                self.printMessage('Points deleted')
        except:
            self.messageError()
    
    def deletePointWaveFunc(self):
        try:
            fig,_=experiment.plot_spectra(times='auto')
            self.subWindow_del_wave = QMdiSubWindow(self)
            self.widget_del_wave=Grapth(fig,self.subWindow_del_wave,cursor=True,y=False)
            self.formatSubWindow(self.subWindow_del_wave,self.widget_del_wave,'Delete wavelength points')
            btn = QPushButton("Delete selected points", self.subWindow_del_wave) 
            self.subWindow_del_wave.layout().addWidget(btn)
            btn.clicked.connect(self.deletePointWaveActionFunc)
            btn_close = QPushButton("Close", self.subWindow_del_wave) 
            self.subWindow_del_wave.layout().addWidget(btn_close)
            btn_close.clicked.connect(self.subWindow_del_wave.close)
        except:
            self.messageError()
    
    def deletePointWaveActionFunc(self):
        try:
            index=self.widget_del_wave.cursorData()
            if len(index)==0:
                error_dialog = QMessageBox(self)
                error_dialog.setText('No points have been selected')
                error_dialog.exec()
            else:
                experiment.delPoints(index,dimension='wavelength')
                self.subWindow_del_wave.close()
                self.printMessage('Points deleted')
        except:
            self.messageError()
           
    def averageTimeFunc(self):
        self.noZones()
        self.edit_general['average_time_edit']=True
        self.repaint()
    
    def timeAverigeComboBoxFunc(self):
        if self.select_average.currentText()=='Constant Step':
          self.label39.setVisible(False)
          self.spin.setVisible(False)
          self.timeAverigeComboBox_edit=True
          self.repaint()
        else:
          self.timeAverigeComboBox_edit=False
          self.repaint()
    
    def button09Func(self):
        try:
            start=float(self.text09.text())
            step=float(self.text19.text())
            if self.select_average.currentText() is 'Constant Step':
                experiment.averageTimePoints(starting_point=start,step=step,method='constant')
            else:
                speed=int(self.spin.value())
                experiment.averageTimePoints(starting_point=start,step=step,method='log',grid_dense=speed)
            self.exploreDataOn()
            self.printMessage('Times averaged')

        except:
            self.messageError()
        
    def calibrateTableFunc(self):
        self.TableWindow = QMdiSubWindow(self)
        self.TableWindow.resize(200,400)
        self.calib_table = QTableWidget(4,2,self)
        self.calib_table.setHorizontalHeaderLabels(('Pixel', 'Wavelength'))
        self.calib_table.setColumnWidth(0,75)
        self.calib_table.setColumnWidth(1,75)
        self.formatSubWindow(self.TableWindow,self.calib_table,'Calibrate wavelengths')
        btn_point = QPushButton("Add more points", self.TableWindow) 
        self.TableWindow.layout().addWidget(btn_point)
        btn_point.clicked.connect(lambda:self.calib_table.setRowCount(self.calib_table.rowCount()+1))
        btn_del = QPushButton("delete last point", self.TableWindow) 
        self.TableWindow.layout().addWidget(btn_del)
        btn_del.clicked.connect(lambda:self.delParam(self.calib_table,3))
        btn = QPushButton("Calibrate", self.TableWindow) 
        self.TableWindow.layout().addWidget(btn)
        btn.clicked.connect(self.calibrateTable)
        btn_close = QPushButton("Close", self.TableWindow) 
        self.TableWindow.layout().addWidget(btn_close)
        btn_close.clicked.connect(self.TableWindow.close)
    
    def calibrateTable(self):
        try:
            pixel=[]
            wave=[]
            for i in range(self.calib_table.rowCount()):
                text1=self.calib_table.item(i,0).text()
                text2=self.calib_table.item(i,1).text()
                try:
                    pixel.append(float(text1))
                    wave.append(float(text2))
                except:
                    pass
            if len(pixel)==3:
                calib_curve,_=self.cal.doCalibrationFit(pixel,wave,1)
            elif len(pixel)>3:
                calib_curve,_=self.cal.doCalibrationFit(pixel,wave,2)
            else:    
                error_dialog = QMessageBox(self)
                error_dialog.setText('At least three point are needed to calibrate')
                error_dialog.exec()
            experiment.general_report['Preprocessing']['Calibration']=f'From table of points pixel={pixel}, wavelength={wave} '
            experiment.general_report['Sequence of actions'].append('\t--> Wavelength calibration')
            new_wave=calib_curve(experiment.wavelength)
            experiment.wavelength=new_wave
            self.checkNewWAveFunc()
            self.TableWindow.close()
            self.printMessage('Wavelength calibrated')
        except:
            self.messageError()
         
    def calibrateReferenceFunc(self):
        self.subWindow_calibrate = QMdiSubWindow(self)
        self.subWindow_calibrate.setMaximumHeight(self.window_height-80)
        self.subWindow_calibrate.setWindowTitle("Calibration with reference spectra")
        self.mdiArea.addSubWindow(self.subWindow_calibrate)
        if self.mdiArea.size().width()<1220:
            width=self.mdiArea.size().width()
        else:
            width=1220
        if self.mdiArea.size().height()<680:
            height=self.mdiArea.size().height()
        else:
            height=680
        self.subWindow_calibrate.resize(width,height)
        self.subWindow_calibrate.show()
        
        wavelabel='Wavelength ('+experiment.wavelength_unit+')'
        deltalabel=r'$\Delta$A'
        self.ref_fig,self.ref_ax = plt.subplots(1,figsize=(5,3))
        x=np.linspace(1,10,10)
        self.lref=self.ref_ax.plot(x,x*0.001)
        self.ref_ax.set_ylabel(deltalabel,size=12)
        plt.legend(['Thoretical spectrum'])
        self.ref_ax.set_xlabel(wavelabel,size=12)
        self.ref_ax.set_ylabel(deltalabel,size=12)
        plt.tight_layout()
        
        self.sample_fig,self.sample_ax = plt.subplots(1,figsize=(5,3))
        self.lsample=self.sample_ax.plot(x,x*0.001)
        plt.legend(['Measured spectrum'])
        self.sample_ax.set_xlabel('pixel',size=12)
        self.sample_ax.set_ylabel(deltalabel,size=12)
        plt.tight_layout()
        
        self.ref=Grapth(self.ref_fig,self.subWindow_calibrate,toolbar=True,cursor=True,ax=self.ref_ax)
        self.sample=Grapth(self.sample_fig,self.subWindow_calibrate,toolbar=True,cursor=True,ax=self.sample_ax)
        lay_Top=QHBoxLayout(self.subWindow_calibrate)
        lay_Top.addWidget(self.ref)
        lay_Top.addWidget(self.sample)
        lay_Top.setStretch(0,5)
        lay_Top.setStretch(1,5)
        lay_buttons=QVBoxLayout(self.subWindow_calibrate)
        lay_but1=QHBoxLayout(self.subWindow_calibrate)
        button_load_ref = QPushButton('Load theretical reference Spectrum', self)
        button_load_ref.setFixedSize(QSize(250, 40))
        button_load_ref.clicked.connect(lambda:self.loadCalDataFunc('theoretical'))
        button_load_sam = QPushButton('Load meassured reference Spectrum', self)
        button_load_sam.setFixedSize(QSize(250, 40))
        button_load_sam.clicked.connect(lambda:self.loadCalDataFunc('measured'))
        lay_but1.addWidget(button_load_ref)
        lay_but1.addWidget(button_load_sam)
        lay_but2=QHBoxLayout(self.subWindow_calibrate)
        button_ref_del = QPushButton('Unselect thoretical spectrum', self)
        button_ref_del.setFixedSize(QSize(250, 40))
        button_ref_del.clicked.connect(lambda:self.delSelectionFun(self.ref,self.ref_ax))
        button_sam_del = QPushButton('Unselect meassured spectrum', self)
        button_sam_del.setFixedSize(QSize(250, 40))
        button_sam_del.clicked.connect(lambda:self.delSelectionFun(self.sample,self.sample_ax))
        lay_but2.addWidget(button_ref_del)
        lay_but2.addWidget(button_sam_del)
        lay_buttons.addLayout(lay_but1)
        lay_buttons.addLayout(lay_but2)
        button_calibrate = QPushButton('Calibrate using selected points', self)
        button_calibrate.setFixedSize(QSize(250, 40))
        button_calibrate.clicked.connect(self.calReaf)
        lay_but3=QHBoxLayout(self.subWindow_calibrate)
        lay_but3.addWidget(button_calibrate)
        label=QLabel('Polynom order:', self)
        label.setFixedSize(QSize(75, 40))
        lay_but3.setContentsMargins(70,5,70,5)
        lay_but3.addWidget(label)
        self.pol_order = QLineEdit('3', self)
        self.pol_order=QSpinBox(self)
        self.pol_order.setMaximum(3)
        self.pol_order.setMinimum(2)
        self.pol_order.setFixedSize(QSize(40, 40))
        lay_but3.addWidget(self.pol_order)
        button_accept_calibrate = QPushButton('Apply calibration', self)
        button_accept_calibrate.setFixedSize(QSize(250, 40))
        button_accept_calibrate.clicked.connect(self.applyCal)
        close = QPushButton('Close', self)
        close.setFixedSize(QSize(400, 40))
        close.clicked.connect(self.subWindow_calibrate.close)
        lay_buttons.addWidget(QLabel('', self),alignment=Qt.AlignCenter)
        lay_buttons.addLayout(lay_but3)
        #lay_buttons.addWidget(button_calibrate,alignment=Qt.AlignCenter)
        lay_buttons.addWidget(button_accept_calibrate,alignment=Qt.AlignCenter)
        self.res_text=QLabel('', self)
        lay_buttons.addWidget(self.res_text,alignment=Qt.AlignCenter)
        lay_buttons.addWidget(QLabel('', self),alignment=Qt.AlignCenter)
        lay_buttons.addWidget(close,alignment=Qt.AlignCenter)
        
        lay_buttons.setContentsMargins(10,10,10,10)
        
        self.result_fig=plt.figure(figsize=(5,3))
        self.result_ax=self.result_fig.add_subplot(1,1,1)
        #plt.title('Result of calibration', size=16)
        self.result_ax.set_xlabel(wavelabel,size=12)
        self.result_ax.set_ylabel(deltalabel,size=12)
        plt.tight_layout()
        res=Grapth(self.result_fig,self.subWindow_calibrate,toolbar=True)
        lay_Down=QHBoxLayout(self.subWindow_calibrate)
        lay_Down.addLayout(lay_buttons)
        lay_Down.addWidget(res)
        main_layout=QVBoxLayout(self.subWindow_calibrate)
        main_layout.addLayout(lay_Top)
        main_layout.addLayout(lay_Down)
        self.subWindow_calibrate.layout().addLayout(main_layout)
        self.new_x_wave=None
       
    def applyCal(self):
        if self.new_x_wave==None:
            error_dialog = QMessageBox(self)
            error_dialog.setText('Please calibrate first')
            error_dialog.exec()
        else:
            experiment.wavelength=self.new_x_wave  
            self.printMessage('Calibration applied')
    
    def calReaf(self):
        order=int(self.pol_order.value())
        if len (self.ref.cursorData()[0])==len(self.sample.cursorData()[0]):
            if order<len(self.ref.cursorData()[0]):
                calib_curve,r2=self.cal.doCalibrationFit(sorted(self.sample.cursorData()[0]),sorted(self.ref.cursorData()[0]),order)
                x_new_ref=calib_curve(self.x_ref)
                self.new_x_wave=calib_curve(experiment.wavelength)
                self.result_ax.plot(x_new_ref,(self.y_ref-min(self.y_ref))/(max(self.y_ref)-min(self.y_ref)))
                self.result_ax.legend(['theoretical spectrum','measured spectrum'])
                self.result_ax.set_xlim([min(x_new_ref), max(x_new_ref)])
                self.result_fig.canvas.draw_idle()
                lista=[round(calib_curve[i],2) for i in range(order+1)]
                texte=[str(lista[0])]+[str(lista[1])+r'X']+[str(lista[i])+r'X<sup>'+str(i)+'</sup>' for i in range(2,order+1)]
                string='+'.join(texte)
                self.res_text.setText('Polynom: '+string+r'; R<sup>2</sup>: '+'{:.3f}'.format(r2))
                experiment.general_report['Preporcessing']['Calibration']=f'From reference file {self.calib_ref}'
                experiment.general_report['Sequence of actions'].append('\t--> Wavelength calibration')
            else:
                error_dialog = QMessageBox(self)
                error_dialog.setText('For {order} order polynom; minimum {order+1} point are needed')
                error_dialog.exec()
        else:
            error_dialog = QMessageBox(self)
            error_dialog.setText('Select the same number of points in the theoretical as the meassured spectra')
            error_dialog.exec()
    
    def loadCalDataFunc(self,typo):
        try:
            path=QFileDialog.getOpenFileName(self, 'Open calibration file', 
            'c:\\',"files (*.abt *.asc *.csv *.txt *.dat *.lvm)")
            filename=path[0]
            if typo=='theoretical':
                fig,ax=self.cal.importExperimentalData(filename,plot=True)
                x=ax[0].get_xdata()
                y=ax[0].get_ydata()
                self.lref[0].set_data(x,y)
                self.ref_ax.set_xlim([min(x), max(x)])
                self.ref_ax.set_ylim([min(y)-min(y)/10, max(y)+max(y)/10])
                self.ref_fig.canvas.draw_idle()
                self.result_ax.plot(x,(y-min(y))/(max(y)-min(y)))
                self.result_ax.legend(['theoretical spectrum'])
                self.result_fig.canvas.draw_idle()
            else:
                self.calib_ref=filename
                fig,ax=self.cal.importReferenceData(filename,plot=True)
                self.x_ref=ax[0].get_xdata()
                self.y_ref=ax[0].get_ydata()
                self.lsample[0].set_data(self.x_ref,self.y_ref)
                self.sample_ax.set_xlim([min(self.x_ref), max(self.x_ref)])
                self.sample_ax.set_ylim([min(self.y_ref)-min(self.y_ref)/10, max(self.y_ref)+max(self.y_ref)/10])
                self.sample_fig.canvas.draw_idle()
        except:
            error_dialog = QMessageBox(self)
            error_dialog.setText(f'Unable to open and read {typo} reference spectrum files should have to columns separated by "," with no header')
            error_dialog.exec()   
    
    def vectorWAveFunc(self):
        self.subWindow_vector = QMdiSubWindow(self)
        self.subWindow_vector.resize(550,200)
        self.calib_text=self.formatSubWindow(self.subWindow_vector,QTextEdit(),'Introduce wavelengths vector')
        label =  QLabel('\tPlease copy wavelengths in one column or in one line seprated by ",". Then click on Set wavelengths button\t', self.subWindow_vector)
        self.subWindow_vector.layout().addWidget(label)
        btn_calib = QPushButton("Set wavelengths", self.subWindow_vector) 
        self.subWindow_vector.layout().addWidget(btn_calib)
        btn_calib.clicked.connect(self.setwaveFunc)
        btn_close = QPushButton("Close", self.subWindow_vector) 
        self.subWindow_vector.layout().addWidget(btn_close)
        btn_close.clicked.connect(self.subWindow_vector.close)
    
    def setwaveFunc(self):
        try:            
            answer=np.array([float(i) for i in self.calib_text.toPlainText().split(',')]) 
        except:
            try:
                answer=np.array([float(i) for i in self.calib_text.toPlainText().split('\n')]) 
            except:
                error_dialog = QMessageBox(self)
                error_dialog.setText(f'Please copy wavelengths in one column or in one line seprated by ","')
                error_dialog.exec()   
        if len(answer)==len(experiment.data[0,:]):
            experiment.wavelength=answer
            experiment.general_report['Preporcessing']['Calibration']=f'Array copied'
            experiment.general_report['Sequence of actions'].append('\t--> Wavelength calibration')
            self.checkNewWAveFunc()
            self.subWindow_vector.close()
            self.checkNewWAveFunc()
            self.TableWindow.close()
            self.printMessage('Calibration set')
        else:     
            error_dialog = QMessageBox(self)
            error_dialog.setText('The vector introduce has diferent size as the original in data; please introduce same length vector')
            error_dialog.exec()
        
    def checkNewWAveFunc(self):
        sort_wave=np.sort(experiment.wavelength)
        if (sort_wave == experiment.wavelength[::-1]).all():
            experiment.data=experiment.data[:,::-1]
            experiment.wavelength=experiment.wavelength[::-1]
    #correctGVD manully sellmeier    
    def button08Func(self):
        try:
            bk7=float(self.text08.text())
            SiO2=float(self.text18.text())
            CaF2=float(self.text28.text())
            offset=float(self.text38.text())
            experiment.excitation=int(self.text48.text())
            experiment.GVD(CaF2=CaF2,SiO2=SiO2,BK7=bk7,offset=offset)
            status=self.check08.checkState()
            experiment
            if status==0:
                experiment.correctGVD(False)
                self.printMessage('Data corrected')
            else:
                experiment.qt_path=self.setWindowIcon_path
                experiment.correctGVD(True)
        except:
            self.messageError()
    
    def GVDSellFunc(self):
        self.noZones()
        self.edit_general['sellmeier_edit']=True
        self.repaint()
    
    def loadExperimentFunc(self):
        try:
            path=QFileDialog.getOpenFileName(self, 'Open Ultra PyFit experiment file', 
            'c:\\',"files (*.obj )")
            with open(path[0],"rb") as file:
                global experiment
                experiment=pickle.load(file)
            self.manual_load=experiment.experiment_manual_load
            self.IRF_value=experiment.IRF_value
            self.fit_number=len(experiment.all_fit)
            self.undo_wavelength=None
            self.undo_time=None
            self.undo_data=None
            self.undo_wave_already=False
            self.undo_time_already=False
            self.save_fileName=[path[0].split('.')[0],path[1]]
            self.fit_in_progress=False
            self.corrected_GVD=experiment.GVD_correction
            self.initialize_global_fit=True
            self.numberTraces()
            experiment.weights['apply']=False
            self.printMessage('Experiment loaded')
        except:
            error_dialog = QMessageBox(self)
            error_dialog.setText('Unable to load experiment')
            error_dialog.exec()

    def delGVDObjects(self):#pickle cannot dump FigureCanvasQTAgg objectsso this function convert object on empyt string if they exits
        experiment.sofset=''
        experiment.sfreq=''
        experiment.samp=''
        experiment.sbk7=''
        experiment.radio =''
        experiment.radio1=''
        experiment._button_svd_select= ''
        experiment.button2=''
        experiment._fig= ''
        experiment.figGVD=''
        experiment._ax= ''
        experiment.l=''
    
    def saveasFunc(self): 
        try:
            self.save_fileName = QFileDialog.getSaveFileName(self,'Save File',"Ultra PyFit files")
            self.delGVDObjects()
            with open(self.save_fileName[0]+'.obj', 'wb') as f:
                pickle.dump(experiment, f) 
            self.printMessage('Experiment saved')
        except:
            error_dialog = QMessageBox(self)
            error_dialog.setText('Unable to save experiment')
            error_dialog.exec()
        
    def saveFunc(self):
        try:
            if self.save_fileName is None:
                self.save_fileName = QFileDialog.getSaveFileName(self,'Save File',"Ultra PyFit files")
            self.delGVDObjects()
            with open(self.save_fileName[0]+'.obj', 'wb') as f:
                pickle.dump(experiment, f) 
            self.printMessage('Experiment saved')
        except:
            error_dialog = QMessageBox(self)
            error_dialog.setText('Unable to save experiment')
            error_dialog.exec()
        
    def GVDGrapthSellFunc(self):
        if experiment.excitation is None:
            error_dialog = QInputDialog(self)
            x_value, ok = error_dialog.getDouble(self, "Set excitation","The GVD using Sellmeiller equation need the pump excitation wavelength", 0.0,0, 2000)
            if ok and type(x_value) is float:
                print('ok')
                experiment.excitation=x_value
                experiment.GVDFromGrapth(qt=self.setWindowIcon_path)
            else:
                error_dialog = QMessageBox(self)
                error_dialog.setText('Please enter a valid number')
                error_dialog.exec()
        else:
            experiment.GVDFromGrapth(qt=self.setWindowIcon_path)
        
    def GVDAlereadyFunc(self):
        self.general_report['Preprocessing']['GVD correction']=f'\tGVD was previously corrected'
        self.corrected_GVD=True
    
    def deconvComboBox(self):
        texto=self.select_process.currentText()
        if texto=='Deconvolution':
            experiment.deconv=True
        else:
            experiment.deconv=False
            
    def noZones(self):
        for key in self.edit_general.keys():
            self.edit_general[key]=False
    
    def numberTraces(self,SVD=False):
        if experiment.data_before_first_selection is not None:
            times_points,self.total_traces=experiment.data_before_first_selection.shape
            _,self.selected_traces=experiment.data.shape
        else:
            times_points,self.total_traces=experiment.data.shape
            _,self.selected_traces=experiment.data.shape
        if self.total_traces==self.selected_traces:
            self.selected_traces='all'
        if self.total_traces==int(experiment.general_report[ 'Data Shape']['Initial number of traces']):
            experiment.general_report[ 'Data Shape']['Actual number of traces']='all'
        else:
            experiment.general_report['Data Shape']['Actual number of traces']=f'{self.total_traces}'
        if times_points==int(experiment.general_report[ 'Data Shape']['Initial time points']):
            experiment.general_report[ 'Data Shape']['Actual time points']='all'
        else:
            experiment.general_report[ 'Data Shape']['Actual time points']=f'{times_points}'
        if SVD:
            label=str(self.selected_traces) + ' Singular vectors' 
        else:
            label=f'{self.selected_traces}'
        self.label035.setText(label)
        self.label015.setText(str(self.total_traces))

    #cut region 
    def button003Func(self):
        try:
            try:
                mini=float(self.text003.text())
            except:
                mini=None
            try:    
                maxi=float(self.text013.text())
            except:
                maxi=None    
            if experiment.data_before_first_selection is not None:
                self.undo_data=experiment.data_before_first_selection
                self.undo_wavelength=experiment.wavelength_before_first_selection
            else:
                self.undo_wavelength=experiment.wavelength
                self.undo_data=experiment.data
            if maxi is not None and mini is not None:
                if mini<maxi:
                    experiment.cutData(mini,maxi,True)
                else:
                    message="The lower index must be smaller than the higher index"
                    by_dialog = QMessageBox(self)
                    by_dialog.setText(message)
                    by_dialog.exec() 
            else:
                experiment.cutData(mini,maxi,False)
            self.exploreDataOn()
            self.numberTraces()
            self.repaint()
            self.printMessage('Wavelength cutted')
        except:
            self.messageError()
    
    #cut time
    def button004Func(self):
        try:
            try:
                mini=float(self.text004.text())
            except:
                mini=None
            try:    
                maxi=float(self.text014.text())
            except:
                maxi=None    
            print(mini,maxi)
            if maxi is not None and mini is not None:
               error_dialog = QMessageBox(self)
               error_dialog.setText('cut an inside time region is not allowd! \n\
                                    If you want to delete 1 point, Go to delete points\n\
                                    If you want to select an inner region click on Select')
               error_dialog.exec()
                
            else:
                if experiment.data_before_first_selection is not None:
                    self.undo_data=experiment.data_before_first_selection
                else:
                    self.undo_data=experiment.data
                self.undo_time=experiment.x
                experiment.cutTimeData(maxi,mini)
                self.exploreDataOn()
                self.printMessage('Time cutted')
            self.repaint()
        except:
            self.messageError()
    
    def timeComboBox(self):
        try:
            texto=self.select_time.currentText()
            if texto=='Clear':
                self.text004.setText('')
                self.text014.setText('')
                print('a')
            elif texto=='From':
                fig,_=experiment.plotRawData(auto=True)
                self.selectCombo(fig,self.text004)
            elif texto=='Until':
                fig,_=experiment.plotRawData(auto=True)
                self.selectCombo(fig,self.text014)
            else:
                fig,_=experiment.plotRawData(auto=True)
                self.selectCombo(fig,[self.text004,self.text014],number_click=2)
        except:
            self.messageError()
    
    def grapthComboBox(self):
        try:
            texto=self.select_grapth.currentText()
            if texto=='Clear':
                self.text003.setText('')
                self.text013.setText('')
                print('a')
            elif texto=='From':
                fig,_=experiment.plot_spectra()
                self.selectCombo(fig,self.text003)
            elif texto=='Until':
                fig,_=experiment.plot_spectra()
                self.selectCombo(fig,self.text013)
            else:
                fig,_=experiment.plot_spectra()
                self.selectCombo(fig,[self.text003,self.text013],number_click=2)
        except:
            self.messageError()
    
    def selectCombo(self,fig,label,number_click=1):
        try:
            subWindow = QMdiSubWindow(self)
            widget_select=Grapth(fig,subWindow,cursor=True,y=False,number_click=number_click)
            self.formatSubWindow(subWindow,widget_select,"Graphical selection")
            btn_select = QPushButton("Select this Points",subWindow)
            subWindow.layout().addWidget(btn_select)
            btn_select.clicked.connect(lambda:self.comboSelected(fig,label,subWindow,widget_select))
        except:
            self.messageError()
    
    def comboSelected(self,fig,label,subWindow,widget):
        try:
            selection=sorted(widget.cursorData())
            if len(selection)==0:
               error_dialog = QMessageBox(self)
               error_dialog.setText('Please select point by clicking on the figure')
               error_dialog.exec()
            else:
                selection=[round(i,1) for i in selection]
                if type(label)== list:
                    assert len(label)==len(selection)
                    for i in range(len(label)):
                        label[i].setText(str(selection[i]))
                else:
                    if len(selection)==1:
                        label.setText(str(selection[0]))
                    else:
                        label.setText(','.join(selection))
                subWindow.close()
                plt.close(fig)
        except:
            self.messageError()
    
    #select region 
    def button013Func(self):
        try:
            try:
                mini=float(self.text003.text())
            except:
                mini=None
            try:    
                maxi=float(self.text013.text())
            except:
                maxi=None
            print(mini,maxi)    
            if experiment.data_before_first_selection is not None:
                self.undo_data=experiment.data_before_first_selection
                self.undo_wavelength=experiment.wavelength_before_first_selection
            else:
                self.undo_wavelength=experiment.wavelength
                self.undo_data=experiment.data
            if maxi is not None and mini is not None:
                if mini<maxi:
                    experiment.cutData(mini,maxi,innercut='select')
                else:
                    message="The lower index must be smaller than the higher index"
                    by_dialog = QMessageBox(self)
                    by_dialog.setText(message)
                    by_dialog.exec() 
            else:
                experiment.cutData(maxi,mini)
            self.exploreDataOn()
            self.numberTraces()
            self.repaint()
            self.printMessage('Wavelength cutted')
        except:
            self.messageError()
        
    def button014Func(self):
        try:
            try:
                mini=float(self.text004.text())
            except:
                mini=None
            try:    
                maxi=float(self.text014.text())
            except:
                maxi=None
            if experiment.data_before_first_selection is not None:
                    self.undo_data=experiment.data_before_first_selection
            else:
                self.undo_data=experiment.data
            self.undo_time=experiment.x
            if maxi is not None and mini is not None:
                if mini<maxi:
                    experiment.cutTimeData(mini,maxi)
                else:
                    message="The lower index must be smaller than the higher index"
                    by_dialog = QMessageBox(self)
                    by_dialog.setText(message)
                    by_dialog.exec()   
            else:
                experiment.cutTimeData(mini,maxi)
            self.exploreDataOn()
            self.repaint()
            self.printMessage('Time cutted')
        except:
            self.messageError()
        
    def undoTimeFunc(self):
        try:
            if self.undo_time is not None:
                if experiment.data_before_first_selection is not None:
                    experiment.data_before_first_selection=self.undo_data
                else:
                    experiment.data=self.undo_data
                experiment.x=self.undo_time
                self.undo_time=None
                self.undo_data=None
                self.exploreDataOn()
                self.undo_time_already=True
            else:
                if self.undo_time_already:
                    message="Can only undo One Time, please go to data before cut"
                else:
                    message="Please cut data first"
                by_dialog = QMessageBox(self)
                by_dialog.setText(message)
                by_dialog.exec()  
        except:
            self.messageError()
    
    def undoRegionFunc(self):
        try:
            if self.undo_wavelength is not None:
                if experiment.data_before_first_selection is not None:
                    experiment.data_before_first_selection=self.undo_data
                    experiment.wavelength_before_first_selection=self.undo_wavelength
                else:
                    experiment.data=self.undo_data
                    experiment.wavelength=self.undo_wavelength
                self.undo_wavelength=None
                self.undo_data=None
                self.exploreDataOn()
                self.undo_wave_already=True
            else:
                if self.undo_wave_already:
                    message="Can only undo One Time, please go to data before cut"
                else:
                    message="Please cut data first"
                by_dialog = QMessageBox(self)
                by_dialog.setText(message)
                by_dialog.exec()  
        except:
            self.messageError()
    
    def exitFunc(self):
        by_dialog = QMessageBox(self)
        by_dialog.setText("Tanks for using it! \nDo not forget to Cite us PLEASE¡")
        by_dialog.exec()
        sys.exit()
        
    def originalDataFunc(self):
        experiment.data=experiment.original_data
        experiment.wavelength=experiment.original_wavelength
        experiment.x=experiment.original_x
        experiment.data_before_first_selection=None
        experiment.general_report['Sequence of actions'].append('\t--> Original Data restored')
        experiment.general_report['Preprocessing']=experiment.original_preprocessing
        self.numberTraces()
        self.printMessage('Data restored')
        
    def baselineDataFunc(self):
        if experiment.data_before_bg is not None:
            experiment.data=experiment.data_before_bg
            experiment.wavelength=experiment.wavelength_before_bg        
            experiment.x=experiment.x_before_bg
            experiment.data_before_first_selection=None
            experiment.general_report['Preprocessing'] = experiment.preprocess_before_bg
            experiment.general_report['Sequence of actions'].append('\t--> Data before Background correction restored')
            self.printMessage('Data restored')
        else:
            by_dialog = QMessageBox(self)
            by_dialog.setText("Please before try to correct baseline")
            by_dialog.exec()
            
    def GVDDataFunc(self):
        if experiment.data_before_GVD is not None:
            experiment.data=experiment.data_before_GVD
            experiment.wavelength=experiment.wavelength_before_GVD        
            experiment.x=experiment.x_before_GVD
            experiment.data_before_first_selection=None
            experiment.general_report['Preprocessing'] = experiment.preprocess_before_GVD
            experiment.general_report['Sequence of actions'].append('\t--> Data before GVD correction restored')
            self.printMessage('Data restored')
        else:
            by_dialog = QMessageBox(self)
            by_dialog.setText("Please before try to correct GVD")
            by_dialog.exec()
            
    def derivateDataFunc(self):
        if experiment.data_before_deriv is not None:
            experiment.data=experiment.data_before_deriv
            experiment.wavelength=experiment.wavelength_before_deriv       
            experiment.x=experiment.x_before_deriv
            experiment.data_before_first_selection=None
            experiment.general_report['Preprocessing'] = experiment.preprocess_before_deriv
            experiment.general_report['Sequence of actions'].append('\t--> Data before derivation restored')
            experiment.derivative_space=False
            self.message.setText('Data restored')
            self.mes_t=ThreadMessage()
            self.mes_t.finished.connect(self.messageReset)
            self.mes_t.start()
        else:
            by_dialog = QMessageBox(self)
            by_dialog.setText("Data has not been derivated")
            by_dialog.exec()
            
    def cutDataFunc(self):
        if experiment.data_before_cut is not None:
            experiment.data=experiment.data_before_cut
            experiment.wavelength=experiment.wavelength_before_cut        
            experiment.x=experiment.x_before_cut
            experiment.data_before_first_selection=None
            experiment.general_report['Preprocessing']=experiment.preprocess_before_cut
            experiment.general_report['Sequence of actions'].append('\t--> Data before cutting wavelengths restored')
            self.numberTraces()
            self.message.setText('Data restored')
            self.mes_t=ThreadMessage()
            self.mes_t.finished.connect(self.messageReset)
            self.mes_t.start()
        else:
            by_dialog = QMessageBox(self)
            by_dialog.setText("Data has not been cutted")
            by_dialog.exec()
            
    def timeDataFunc(self):
        if experiment.data_before_time_cut is not None:
            experiment.data=experiment.data_before_time_cut
            experiment.wavelength=experiment.wavelength_before_time_cut
            experiment.x=experiment.x_before_time_cut
            experiment.data_before_first_selection=None
            experiment.general_report['Preprocessing']=experiment.preprocess_before_time_cut
            experiment.general_report['Sequence of actions'].append('\t--> Data before cutting time ranges restored')
            self.numberTraces()
            self.message.setText('Data restored')
            self.mes_t=ThreadMessage()
            self.mes_t.finished.connect(self.messageReset)
            self.mes_t.start()
        else:
            by_dialog = QMessageBox(self)
            by_dialog.setText("Data has not been cutted")
            by_dialog.exec()

    def delPointDataFunc(self):    
        if experiment.data_before_del_point is not None:
            experiment.data=experiment.data_before_del_point
            experiment.wavelength=experiment.wavelength_before_del_point
            experiment.x=experiment.x_before_del_point
            experiment.data_before_first_selection=None
            experiment.general_report['Preprocessing']=experiment.preprocess_del_point
            experiment.general_report['Sequence of actions'].append('\t--> Data before deleting points restored')
            self.numberTraces()
            self.message.setText('Data restored')
            self.mes_t=ThreadMessage()
            self.mes_t.finished.connect(self.messageReset)
            self.mes_t.start()
        else:
            by_dialog = QMessageBox(self)
            by_dialog.setText("No points have been deleted")
            by_dialog.exec()
        
    def timePointDataFunc(self):
        if experiment.data_before_average_cut is not None:
            experiment.data=experiment.data_before_average_time
            experiment.wavelength=experiment.wavelength_before_average_time
            experiment.x=experiment.x_before_average_time
            experiment.data_before_first_selection=None
            experiment.general_report['Preprocessing'] = experiment.preprocess_average_time
            experiment.general_report['Sequence of actions'].append('\t--> Data before avererage time points restored')
            self.numberTraces()
            self.message.setText('Data restored')
            self.mes_t=ThreadMessage()
            self.mes_t.finished.connect(self.messageReset)
            self.mes_t.start()
        
    def loadDataFunc(self):
        try:
            fname = QFileDialog.getOpenFileName(self, 'Open data file', 
            'c:\\',"files (*.abt *.asc *.csv *.txt *.dat *.lvm *.out)")
            print(fname)
            f = open(fname[0], 'r')
            self.text_subWindow = QMdiSubWindow(self)
            text=QTextEdit()
            text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            text.setLineWrapMode(QTextEdit.NoWrap)
            self.contents=self.formatSubWindow(self.text_subWindow,text,'Selected File',close_button=True)
            with f:
                data = f.read()
                self.contents.setText(data)
                print('data')
            self.file=fname[0]
            self.noZones()
            self.edit_general['load_data_edit']=True
            self.repaint()
        except:
            self.messageError()

    def cutRegionsFunc(self):
        self.noZones()
        self.edit_general['cut_region_edit']=True
        self.repaint()
    
    def cutTimeFunc(self):
        self.noZones()
        self.edit_general['cut_time_edit']=True
        self.repaint()
    #button after bar load data manually
    
    def button07Func(self,IRF=False):
        try:
            if self.text007.text() == 'None' or self.text007.text() == 'none':
                self.wave_number=None
            else:
                self.wave_number=int(self.text007.text())
            self.wave_type=self.text07.text()
            self.time_number=int(self.text117.text())
            self.time_type=self.text17.text()
            separator=self.text37.text()
            decimal=self.text27.text()
            self.separator=separator
            if IRF:
                file=self.IRF_file
            else:
                file=self.file
            if separator==',/tab/.':
                separator=','
            if separator=='tab':
                separator=r'\t'
            if self.wave_number==None:
                skip=None
                header=None
            else:
                skip=self.wave_number
                header='infer'
            if self.wave_type=='Column' or self.wave_type == 'C' or self.wave_type == 'c' or self.wave_type == 'column':
                data_frame=pd.read_csv(file,sep=separator,index_col=self.wave_number,skiprows=self.time_number,decimal=decimal).dropna(how='all').dropna(how='all',axis=1).sort_index()
            elif self.wave_type=='Row' or self.wave_type == 'R' or self.wave_type == 'r' or self.wave_type=='row':
                data_frame=pd.read_csv(file,sep=separator,index_col=self.time_number,header=header,skiprows=skip,decimal=decimal).dropna(how='all').dropna(how='all',axis=1)
                if self.time_number!=0:
                    data_frame=data_frame.iloc[:,self.time_number:]
                data_frame=data_frame.transpose().sort_index()
            else:
                if self.time_type=='Column' or self.time_type == 'C' or self.time_type == 'c' or self.time_type == 'column':
                    data_frame=pd.read_csv(file,sep=separator,index_col=self.time_number,header=header,skiprows=self.wave_number-1,decimal=decimal).dropna(how='all').dropna(how='all',axis=1)
                    data_frame=data_frame.transpose().sort_index()
                else: 
                    data_frame=pd.read_csv(file,sep=separator,index_col=self.wave_number,skiprows=self.time_number-1,decimal=decimal).dropna(how='all').dropna(how='all',axis=1).sort_index()
            data_frame.fillna(0,inplace=True)
            if IRF:
                if decimal != '.':
                    new_col_name={i:i.replace(decimal,'.') for i in data_frame.columns}
                    data_frame=data_frame.rename(new_col_name, axis =1)
                wavelenght_dimension,time_dimension=self.readPandas(data_frame)
                data_values=data_frame.transpose().values
                self.IRF_experiment=GlobalFit(time_dimension,data_values,wavelength=wavelenght_dimension)
            else:
                if decimal != '.':
                    new_col_name={i:i.replace(decimal,'.') for i in data_frame.columns}
                    data_frame=data_frame.rename(new_col_name, axis =1)
                self.wavelenght_dimension,self.time_dimension=self.readPandas(data_frame)
                self.data_values=data_frame.transpose().values
                #self.experiment=GlobalFit(self.time_dimension,self.data_values,wavelength=self.wavelenght_dimension)
                self.total_traces=len(self.wavelenght_dimension)
                experiment.x=self.time_dimension
                experiment.data=self.data_values
                experiment.wavelength=self.wavelenght_dimension
                experiment.curve_resultados=self.data_values*0.0
                experiment.original_data=self.data_values
                experiment.original_wavelength=self.wavelenght_dimension
                experiment.original_x=self.time_dimension
                self.select_process.setCurrentText('Deconvolution')
                experiment.reinitiate()
                if abs(self.wavelenght_dimension[1]-self.wavelenght_dimension[0])==1:
                     general_cal='\tNone'
                else:
                     general_cal='\tAlready done'
                experiment.general_report['File']=self.file
                experiment.general_report['Sequence of actions'].append('\t--> Data loaded')
                experiment.general_report['Preprocessing']['Calibration']= general_cal
                experiment.general_report['Data Shape']['Initial number of traces']=f'{self.data_values.shape[1]}'
                experiment.general_report['Data Shape']['Initial time points']=f'{self.data_values.shape[0]}'
                self.original_preprocessing=experiment.general_report['Preprocessing']
                self.fit_number=0
                self.printMessage('Data loaded')
                self.dataReloaded()
                self.initialize_global_fit=True
                self.numberTraces()
                self.text_subWindow.close()
                self.noZones()
                experiment.experiment_manual_load=True
                self.manual_load=True
                self.repaint()
        except:
            error_dialog = QMessageBox(self)
            error_dialog.setText('Oh no!! something happend. Checking if you correctly indicate the row and colum number! \n\
                                     We start counting by "0" so for example firt row should be number 0 and second 1')
            error_dialog.exec()

    def dataReloaded(self):
        self.weights_true.setCheckState(0)
        self.eps_table.setRowCount(0)
        self.tf_eps_table.setRowCount(0)
        self.IRF_value=None
        self.undo_wavelength=None
        self.undo_time=None
        self.undo_data=None
        self.undo_wave_already=False
        self.undo_time_already=False
        self.save_fileName=None
        self.fit_in_progress=False
        self.corrected_GVD=False
         
    def readPandas(self,pandas):
        try: 
            try:
                column=np.array([float(i) for i in pandas.columns.values])
            except:
                column=np.array([float((re.findall(r"[-+]?\d*\.\d*[eE]?[-+]?\d*|[-+]?\d+",i))[0]) for i in pandas.columns.values]).flatten()
            if type(pandas.index[0]) == str:
                row=np.array([float((re.findall(r"[-+]?\d*\.\d*[eE]?[-+]?\d*|[-+]?\d+",i))[0]) for i in pandas.index.values]).flatten()
            else:
                row=np.array([float(ii) for ii in pandas.index.values])
            return row, column
        except:
            error_dialog = QMessageBox(self)
            error_dialog.setText('Error reading the data')
            error_dialog.exec()
    #button after bar load data default
    
    def button17Func(self,IRF=False):
        try:
            if IRF:
                file=self.IRF_file
            else:
                file=ex.file
            data_frame=pd.read_csv(file,sep=',',index_col=0).dropna(how='all').dropna(how='all',axis=1).sort_index()
            data_frame.fillna(0,inplace=True)
            if IRF:
                wavelenght_dimension,time_dimension=self.readPandas(data_frame)
                data_values=data_frame.transpose().values
                self.IRF_experiment=GlobalFit(time_dimension,data_values,wavelength=wavelenght_dimension)
            else: 
                self.wavelenght_dimension,self.time_dimension=self.readPandas(data_frame)
                self.data_values=data_frame.transpose().values
                #self.experiment=GlobalFit(self.time_dimension,self.data_values,wavelength=self.wavelenght_dimension)
                self.total_traces=len(self.wavelenght_dimension)
                experiment.x=self.time_dimension
                experiment.data=self.data_values
                experiment.wavelength=self.wavelenght_dimension
                experiment.curve_resultados=self.data_values*0.0
                experiment.original_data=self.data_values
                experiment.original_wavelength=self.wavelenght_dimension
                experiment.original_x=self.time_dimension
                self.select_process.setCurrentText('Deconvolution')
                experiment.reinitiate()
                if abs(self.wavelenght_dimension[1]-self.wavelenght_dimension[0])==1:
                     general_cal='\tNone'
                else:
                     general_cal='\tAlready done'
                experiment.general_report['File']=self.file
                experiment.general_report['Sequence of actions'].append('\t--> Data loaded')
                experiment.general_report['Preprocessing']['Calibration']= general_cal
                experiment.general_report['Data Shape']['Initial number of traces']=f'{self.data_values.shape[1]}'
                experiment.general_report['Data Shape']['Initial time points']=f'{self.data_values.shape[0]}'
    #            subWindow = QMdiSubWindow(self)
    #            self.table_data = QTableWidget(len(self.time_dimension),len(self.wavelenght_dimension),self)
    #            self.formatSubWindow(subWindow,self.table_data,'Loaded File')
    #            print(data_frame.shape)
    #            for i in range(len(self.wavelenght_dimension)):
    #                for ii in range(len(self.time_dimension)):
    #                    it= QTableWidgetItem(str(data_frame.value[ii,i]))
    #                    self.table_data.setIte m(ii,i,it)
                self.numberTraces()
                self.fit_number=0
                self.printMessage('Data loaded')
                self.initialize_global_fit=True
                self.dataReloaded()
                self.eps_table.setRowCount(0)
                self.text_subWindow.close()
                self.noZones()
                self.repaint()
                experiment.experiment_manual_load=False
                self.manual_load=False
        except:
            error_dialog = QMessageBox(self)
            error_dialog.setText('Oh no!! something happend. Try manual loding!')
            error_dialog.exec()
            
    def messageReset(self):
        self.message.setText('')
        
    def button27Func(self):
        self.repaint()
    
    #button for stop fitting
    def button37Func(self):
        self.stopped_thread=True
        self.working_thread_iter.terminate()
        self.my_thread_Fit.terminate()
        self.my_thread_preFit.terminate()
        self.my_thread_Fit = QThread()
        self.my_thread_Fit.start(QThread.TimeCriticalPriority)  
        self.my_thread_preFit = QThread()
        self.my_thread_preFit.start(QThread.TimeCriticalPriority)
        experiment.number_it = 0
        self.button1.show()
        self.label00.show()
        self.button0.show()
        self.noZones()
        self.edit_general['pop_edit']=True
        self.repaint()
    
    #after go to fit menu bar5 action
    def SelectAllTracesFunc(self):
        try:
            if experiment.data_before_first_selection is not None:
        
                experiment.data=experiment.data_before_first_selection
                experiment.curve_resultados=experiment.data*0.0
                experiment.wavelength=experiment.wavelength_before_first_selection
                self.numberTraces()
                experiment.SVD_fit=False
                if experiment.params_initialized==True:
                    experiment.paramsAlready()
            else:
                pass
        except:
            self.messageError()
    
    def plotFullSVD(self,vectors=1,select=False):
        try:
            if experiment.data_before_first_selection is None:
                wavelength=experiment.wavelength
            else:
                wavelength=experiment.wavelength_before_first_selection
            if experiment.S is None:
                self.U,self.S,self.V=experiment.calculateSVD(vectors=15)
            else:
                self.U,self.S,self.V=experiment.U,experiment.S,experiment.V
            assert vectors>0 and vectors<len(self.S), 'vector value should be between 1 and the number of calculated values'
            if vectors== 'all':
                vectors=len(self.S)
            self.fig,self.ax=plt.subplots(1,3, figsize=(14,6))
            self.ax[1].plot(range(1,len(self.S)+1),self.S,marker='o')
            for i in range(vectors): 
                self.ax[0].plot(experiment.x,self.U[:,i])
                self.ax[2].plot(wavelength,self.V[i,:])
            self.ax[0].set_title('Left singular vectors')
            self.ax[1].set_title('Eingen values')
            self.ax[2].set_title('Right singular vectors')
            self.number_of_vectors_plot=vectors
            self.VerticalSVD=self.ax[1].axvline(vectors,alpha=0.5,color='red',zorder=np.inf)  
    #        axspec = self.fig.add_axes([0.20, .02, 0.60, 0.01],facecolor='orange')
    #        self.specSVD = Slider(axspec, 'curve number', 1, len(self.S),valstep=1,valinit=vectors)
    #        self.specSVD.on_changed(self.updatePlotSVD)
            self.subWindowSVD = QMdiSubWindow(self)
            widget=Grapth(self.fig,self.subWindowSVD,cursor=False)
            self.formatSubWindow(self.subWindowSVD,widget,'Singular value decomposition')
            self.siliderSVD = QSlider(Qt.Horizontal,self.subWindowSVD)
            self.siliderSVD.setFixedSize(QSize(850, 15))
            self.siliderSVD.setStyleSheet('background-color:darkRed;border: 2px solid #999999;height: 15px; border-radius: 5px')
            self.siliderSVD.setTickPosition(self.siliderSVD.TicksBelow)
            self.siliderSVD.setValue(1)
            self.siliderSVD.setMinimum(1)
            self.siliderSVD.setMaximum(15)
            self.siliderSVD.valueChanged.connect(self.siliderSVDFunc)
            self.subWindowSVD.layout().addWidget(self.siliderSVD,alignment=Qt.AlignCenter)
    #        if select:
    #            b_ax = plt.axes([0.85, 0.025, 0.1, 0.04])
    #            self.button = QPushButton(b_ax, 'Select', color='tab:red', hovercolor='0.975')
    #            self.button.on_clicked(self.selectSVD)
    
            if select:
                btn_select = QPushButton("Select", self.subWindowSVD) 
                self.subWindowSVD.layout().addWidget(btn_select)
                btn_select.clicked.connect(self.selectSVD)
            btn_close = QPushButton("Close", self.subWindowSVD) 
            self.subWindowSVD.layout().addWidget(btn_close)
            btn_close.clicked.connect(self.subWindowSVD.close)
        except:
            self.messageError()
    
    def siliderSVDFunc(self):
        if experiment.data_before_first_selection is None:
            wavelength=experiment.wavelength
        else:
            wavelength=experiment.wavelength_before_first_selection
        value = int(round(self.siliderSVD.value()))
        colores=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
        if value > self.number_of_vectors_plot:
            valueC=value
            if value>10:
               valueC=value-10*(value//10)  
            self.VerticalSVD.remove()
            self.VerticalSVD=self.ax[1].axvline(value,alpha=0.5,color='red',zorder=np.inf)  
            self.ax[0].plot(experiment.x,self.U[:,value],color=colores[valueC-1])
            self.ax[2].plot(wavelength,self.V[value,:],color=colores[valueC-1])
            self.number_of_vectors_plot=value*1.0
        elif value < self.number_of_vectors_plot:
            self.VerticalSVD.remove()
            self.VerticalSVD=self.ax[1].axvline(value,alpha=0.5,color='red',zorder=np.inf) 
            del self.ax[0].lines[-1]
            del self.ax[2].lines[-1]
            self.number_of_vectors_plot=value*1.0
        else:
            pass
        self.fig.canvas.draw_idle()
    
    def selectSVD(self):
        if experiment.data_before_first_selection is None:
            experiment.data_before_first_selection=experiment.data*1.0
            experiment.wavelength_before_first_selection=experiment.wavelength*1.0
        value = int(round(self.siliderSVD.value()))
        experiment.U=self.U
        experiment.S=self.S
        experiment.V=self.V
        experiment.data=self.U[:,:value]
        experiment.wavelength=np.array(range(1,value+1))
        experiment.curve_resultados=experiment.data*0.0
        experiment.SVD_fit=True
#        self.SVD_wavelength=['left SV %i' %i for i in range(1,value+1)]
        self.numberTraces(SVD=True)
        if experiment.params_initialized==True:
            experiment.paramsAlready()
        self.subWindowSVD.close()
            
    
    #after go to fit menu bar5 action
    def SelectTracesGrapthFunc(self):
        try:
            self.pop_edit_status=self.edit_general['pop_edit']
            fig,self.ax=experiment.plot_spectra()
            self.subWindow = QMdiSubWindow(self)
            self.widget_select=Grapth(fig,self.subWindow,cursor=True,y=False)
            self.formatSubWindow(self.subWindow,self.widget_select)
            btn_select = QPushButton("Select Traces",self.subWindow)
            self.subWindow.layout().addWidget(btn_select)
            btn_select.clicked.connect(self.selectGrapthFunc)
            btn_unselect = QPushButton("Unselect points",self.subWindow)
            self.subWindow.layout().addWidget(btn_unselect)
            btn_unselect.clicked.connect(lambda:self.delSelectionFun(self.widget_select,self.ax))
            btn_close = QPushButton("Close",self.subWindow)
            self.subWindow.layout().addWidget(btn_close)
            btn_close.clicked.connect(self.subWindow.close)
        except:
            self.messageError()

    def selectGrapthFunc(self):
        try:
            a=self.widget_select.cursorData()
            if len(a)==0:
                error_dialog = QMessageBox(self)
                error_dialog.setText('No points have been selected please select by clicking on figure')
                error_dialog.exec()
            else:
                experiment.selectTraces(space=sorted(a),points=0,avoid_excitation=False)
                self.numberTraces()
                self.subWindow.close()
                self.printMessage('Traces selected')
                self.repaint()
        except:
            self.messageError()
    
    #after go to fit menu bar5 action
    def SelectSeriesTracesFunc(self):
        self.target_model_status=self.edit_general['target_fit']
        self.pop_edit_status=self.edit_general['pop_edit']
        self.noZones()
        self.edit_general['series_traces_edit']=True
        self.edit_general['number_traces_edit']=True
        self.repaint()
    
    def Select10TracesFunc(self):
        try:
            if experiment.excitation is not None:
                experiment.selectTraces(space='auto',points=0,avoid_excitation=9)
            else:
                experiment.selectTraces(space='auto',points=0,avoid_excitation=False)
            self.numberTraces()
        except:
            self.messageError()
    #after go to fit menu bar5 button to export series
    def button002Func(self):
        try:
            if self.initialize_global_fit:
                interval=int(self.text012.text())
                average=int(self.text022.text())//2
                exci=int(self.text032.text())
                text_regions=self.text042.text()
                if text_regions == 'None' or text_regions == '' or text_regions == ' ':
                    regions=None
                else:
                    string=re.findall(r'\d*[.]?\d+\-\d*[.]?\d+',text_regions)
                    regions=[[float(ii) for ii in i.split('-')] for i in string]
                    if len(regions)==0:
                        error_dialog = QMessageBox(self)
                        error_dialog.setText('We were unable to understand regions: \n\nplease define region with "-" (i.e 340-345)\nIf you want to define more regions separate them (i.e 340-345 650-655)\nSeparator between 2 regions can be anything except a number\n\nSo far traces have been selected without regions')
                        error_dialog.exec()
                        regions=None    
                if experiment.excitation is not None:
                    experiment.selectTraces(space=interval,points=average,avoid_excitation=exci,avoid_regions=regions)
                else:
                    experiment.selectTraces(space=interval,points=average,avoid_excitation=False,avoid_regions=regions)
                self.numberTraces()
                self.printMessage('Traces selected')
                if self.pop_edit_status==True:
                    self.noZones()
                    self.edit_general['pop_edit']=True
                if self.target_model_status==True:
                    self.noZones()
                    self.edit_general['target_fit']=True
                self.repaint()
            else:
                self.messageError()
        except:
            self.messageError()
    
    #after go to fit menu bar5 action
    def SelectTracesManuallyFunc(self):
        self.target_model_status=self.edit_general['target_fit']
        self.pop_edit_status=self.edit_general['pop_edit']
        self.noZones()
        self.edit_general['traces_manually_edit']=True      
        self.edit_general['number_traces_edit']=True
        self.repaint()
    
    #after go to fit menu bar5 action button to export series manually
    def button001Func(self):
        try:
            if self.initialize_global_fit:
                series=[int(i) for i in self.text000.text().split(',')]
                average=int(self.text001.text())//2
                experiment.selectTraces(space=series,points=average,avoid_excitation=False)
                self.numberTraces()
                self.printMessage('Traces selected')
                if self.pop_edit_status==True:
                    self.noZones()
                    self.edit_general['pop_edit']=True
                if self.target_model_status==True:
                    self.noZones()
                    self.edit_general['target_fit']=True
                self.repaint()
            else:
                self.messageError()
        except:
            self.messageError()
    
    #menu bar2 botom after plot spectra auto to replot auto with new conditions 
    def button06Func(self):
        try:
            mini,maxi = None,None
            if self.text46a.text() not in ('Min',''): mini=float(self.text46a.text()) 
            if self.text46b.text() not in ('Max',''): maxi=float(self.text46b.text())
            if self.check56.checkState() == 2:
                include_max=True
            else:
                include_max=False
            if mini is None and maxi is None: 
                rango=None
            else:
                if mini is not None and maxi is None:
                    rango=[mini,experiment.x[-1]]
                elif maxi is not None and mini is None:
                    rango=[experiment.x[0],maxi]
                else:
                    rango=[mini,maxi]
            average=int(self.text06.text())
            text16_value=self.text16.text()
            number=int(self.text26.text())
            wave=float(self.text36.text())
            if text16_value=='False' or text16_value=='false':
                fig,_=experiment.plot_spectra(times=['auto', number, wave], rango=rango, n_points=average, legend=False, include_rango_max=include_max)
            elif self.text16.text().isdigit():
                fig,_=experiment.plot_spectra(times=['auto', number, wave], rango=rango, n_points=average, ncol=int(self.text16.text()), include_rango_max=include_max)
            else:
                fig,_=experiment.plot_spectra(times=['auto', number, wave], rango=rango, n_points=average, include_rango_max=include_max)
            subWindow = QMdiSubWindow(self)
            widget=Grapth(fig,subWindow)
            self.formatSubWindow(subWindow,widget,'Spectra Auto plotted')
            self.repaint()
        except:
            self.messageError()    
        
     #menu bar2 botom after plot spectra auto to go plot manually
    def button16Func(self):
        self.noZones()
        self.edit_general['plot_spec_edit']=True
        self.repaint()
     
    #bottom after bottom cero for Set Units
    def button05Func(self):
        if self.initialize_global_fit:
            time=self.text05.text()
            wave=self.text15.text()
            if time=='fs/ps/ns/μs/ms/s/min':
                time='ps'
            if wave=='nm/cm-1':
                wave='nm'
            experiment.defineUnits(time,wave)
            self.printMessage('Units set')
            self.repaint()
        else:
            self.messageError()
            
    #bottom after bottom cero Set Excitationfor Set Units
    def button15Func(self):
        if self.initialize_global_fit:
            experiment.excitation=float(self.text25.text())
            experiment.general_report['Excitation']=f'{float(self.text25.text())} nm'
            experiment.general_report['Sequence of actions'].append('\t--> Excitation set')
            self.printMessage('Excitation set')
            self.repaint()
        else:
            self.messageError()
    
    #buton exp settings
    def button0Func(self):
        self.noZones()
        self.edit_general['settings_edit']=True
        self.repaint()
    
    #menu bar2     
    def PlotSpecAutoFunc(self):
        try:
            fig,_=experiment.plot_spectra(times='auto')
            subWindow = QMdiSubWindow(self)
            widget=Grapth(fig,subWindow)
            if experiment.data_before_first_selection is not None:
                data=experiment.data_before_first_selection
                wave=experiment.wavelength_before_first_selection
            else:
                wave=experiment.wavelength
                data=experiment.data
            idx=np.unravel_index(np.argmax(abs(data), axis=None), data.shape)
            self.text36.setText(str(round(wave[idx[1]])))
            self.formatSubWindow(subWindow,widget,'Spectra Auto plotted')
            self.noZones()
            self.edit_general['plot_spec_auto_edit']=True
            self.repaint()
        except:
            self.messageError()
    
    def Plot10TracesFuncProcess(self):
        try:
            fig,_=experiment.plotRawData(auto=True)
            subWindow = QMdiSubWindow(self)
            widget=Grapth(fig,subWindow)
            self.formatSubWindow(subWindow,widget,'Traces Auto plotted')
        except:
            self.messageError()
        
    def PlotSpecAutoFuncProcess(self):
        try:
            fig,_=experiment.plot_spectra(times='auto')
            subWindow = QMdiSubWindow(self)
            widget=Grapth(fig,subWindow)
            self.formatSubWindow(subWindow,widget,'Spectra Auto plotted')
        except:
            self.messageError()
    #menu bar2 action
    def PlotSpecAllFunc(self):
        try:
            fig,_=experiment.plot_spectra()
            subWindow = QMdiSubWindow(self)
            widget=Grapth(fig,subWindow)
            self.formatSubWindow(subWindow,widget,'All Spectra')
            self.repaint()
        except:
            self.messageError()
    
    #menu bar2 action
    def PlotDASFunc(self,number=None):
        try:
            fig,axes=experiment.plotDAS(fit_number=number)
            if number is None:
                exp_no=experiment.exp_no
                tau_inf=experiment.tau_inf
                deconv=experiment.deconv 
            else:
                exp_no=experiment.all_fit[number][4]
                tau_inf=experiment.all_fit[number][6]
                deconv=experiment.all_fit[number][5]
            posible=[i+1 for i in range(exp_no)]
            names=['tau %i' %(i+1) for i in range(exp_no)]
            if tau_inf is not None and deconv:
                names.append('tau inf')
                posible.append(-1)
            subWindow = QMdiSubWindow(self)
            widget=Grapth(fig,subWindow)
#            widget.size()
#            subWindow.size()
            subWindow.setWidget(widget)
            widget.setLayout(QVBoxLayout()) 
            number_name=f'(fit {number})' if number == None  else 'last fit'
            subWindow.setObjectName("Decay Associated Spectra")
            subWindow.setWindowTitle(f"Decay Associated Spectra {number_name}")
            colores=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
#            widget=self.formatSubWindow(subWindow,widget,'Decay Associted Spectra')
            main=QHBoxLayout()
            main.addWidget(widget)
#            subWindow.setLayout(widget)
            v_layout=QVBoxLayout(subWindow)
            v_layout.setAlignment(Qt.AlignTop)
            v_layout.setContentsMargins(0,50,10,10)
            groups=QButtonGroup(self)
            groups.setExclusive(False)
            label=QLabel(f'Select decay',subWindow)
            ticks={}
            v_layout.addWidget(label)
            for ii,name in enumerate(names):
                ticks[name]=QCheckBox(name,subWindow)
                groups.addButton(ticks[name])
                v_layout.addWidget(ticks[name])
                ticks[name].setCheckState(2)
            replot=QPushButton('Plot new', subWindow)
            replot.clicked.connect(lambda: replot(groups,axes))
            v_layout.addWidget(replot)
            reset_index=QCheckBox('re-Index',subWindow)
            v_layout.addWidget(reset_index)
            reset_color=QCheckBox('reset-Color',subWindow)
            v_layout.addWidget(reset_color)
            main.addLayout(v_layout)
            subWindow.layout().addLayout(main)
            self.mdiArea.addSubWindow(subWindow)
            subWindow.show()
            if self.mdiArea.size().width()<1225:
                width=self.mdiArea.size().width()
            else:
                width=1225
            if self.mdiArea.size().height()<700:
                height=self.mdiArea.size().height()
            else:
                height=700
#            widget.resize(width-113,height-95)	
            subWindow.resize(width,height)	
            def replot(groups,axes):
                if experiment.wavelength_unit=='cm-1':
                    xlabel='Wavenumber (cm$^{-1}$)'
                else:
                    xlabel='Wavelength (nm)'
                param=[posible[i] for i,ii in enumerate(groups.buttons()) if ii.isChecked()]
                axes.cla()
                _,ax=experiment.plotDAS(number=param,fit_number=number)
                legend=[]
                for i,l in enumerate(ax.get_lines()[:-1]):
                    lal=l.get_label()
                    original=posible.index(param[i])
                    if param[i]==-1:
                       label=r'$\tau$ = inf'
                    else:
                        if reset_index.isChecked():
                            index=i
                        else:
                            index=original
                        label=r'$\tau '+names[index].split(' ')[1] +'$ ='+lal.split('=')[1]
                    legend.append(label)
                    wavelenght,y=l.get_data()[0], l.get_data()[1]
                    if reset_color.isChecked():
                        axes.plot(wavelenght,y)
                    else:
                        if i > 9:
                            original=original-10
                        axes.plot(wavelenght,y,color=colores[original])
                number_das=i
                axes.set_xlim(wavelenght[0],wavelenght[-1])
                leg=axes.legend(legend,prop={'size': 14})
                leg.set_zorder(np.inf)
                axes.axhline(linewidth=1,linestyle='--', color='k')
                if experiment.excitation is not None and experiment.excitation>np.min(wavelenght) and experiment.excitation<np.max(wavelenght):
                    ymin, ymax = axes.get_ylim()
                    ymin = ymin-ymin*0.05
                    ymax = ymax - ymax*0.05
                    index=np.argmin([abs(experiment.excitation-i) for i in  wavelenght])
                    if experiment.inner_cut_done is not None:
                        mini=np.argmin([abs(experiment.inner_cut_done[0]-i) for i in wavelenght])
                        maxi=np.argmin([abs(experiment.inner_cut_done[1]-i) for i in wavelenght])
                        initial=wavelenght[mini]
                        final=wavelenght[maxi]
                    elif wavelenght[index]>experiment.excitation:
                        initial=wavelenght[index-1]
                        final=wavelenght[index]
                    else:
                        initial=wavelenght[index]
                        final=wavelenght[index+1]
                    rect = Rectangle([initial+1,ymin],width =final-initial-2, 
                                               height=abs(ymax)+abs(ymin),fill=True, color='white',zorder=number_das+2)
                    axes.add_patch(rect)
                axes.minorticks_on()
                axes.axes.tick_params(which='both',direction='in',top=True,right=True,labelsize=14)
                fig.add_subplot(axes)
                axes.set_xlabel(xlabel,size=14)
                axes.set_ylabel('$\Delta$A',size=14)
                fig.canvas.draw()
            self.repaint()
        except:
            self.messageError()
    
    def plotTest(self):
        f, ax= plt.subplots()
        data = [np.random.random() for i in range(10)]
        ax.plot(data, '*-')
        return f,ax
    #menu bar2 action  
    def PlotSVDFunc(self):
        try:
        #fig,ax=self.plotTest()
            fig,ax=experiment.plot_singular_values(log_scale=False)
            subWindow = QMdiSubWindow(self)
            widget=Grapth(fig,subWindow,cursor=True,click=False)
            self.formatSubWindow(subWindow,widget,'Singular Values',close_button=True)
        except:
            self.messageError()
    
    def plot3DFunc(self):
        try:
            try:
                plt.close(self.fig3D)
            except:
                pass
            cmap=experiment.color_map
            X=experiment.x
            if experiment.data_before_first_selection is not None:
                Z=experiment.data_before_first_selection.transpose()
                Y=experiment.wavelength_before_first_selection
            else:
                Z=experiment.data.transpose()
                Y=experiment.wavelength
            if experiment.wavelength_unit=='cm-1':
                xlabel='Wavenumber (cm$^{-1}$)'
            else:
                xlabel=f'Wavelength ({experiment.wavelength_unit})'
            
            X, Y = np.meshgrid(X, Y)
            self.fig3D = plt.figure(figsize=(8,4))
            
            canvas=FigureCanvas(self.fig3D)
            ax = self.fig3D.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, Z, cmap=cmap,
                           linewidth=0, antialiased=False)
    
            # Customize the z axis.
            ax.set_zlim(np.min(Z), np.max(Z))
            ax.set_xlabel(f'Time ({experiment.time_unit})')
            ax.set_ylabel(xlabel)
            ax.set_zlabel('$\Delta$A')
            
            # Add a color bar which maps values to colors.
            self.fig3D.colorbar(surf, shrink=0.5, aspect=5)
            ax.mouse_init()
            subWindow = QMdiSubWindow(self)
#            figure.canvas.draw()
            layout = QVBoxLayout()
            toolbar = NavigationToolbar(canvas, self)  
            layout.addWidget(toolbar)
            layout.addWidget(canvas)
            subWindow.layout().addLayout(layout)    
            btn = QPushButton("close", subWindow) 
            subWindow.layout().addWidget(btn)
            btn.clicked.connect(subWindow.close)
            self.mdiArea.addSubWindow(subWindow)
            subWindow.setWindowTitle("3D surface")
            subWindow.resize(800,500)
            subWindow.show()
        except:
            self.messageError()
    
    def testFunc(self):
        a=self.widget.cursorData()
        print(a)
    #menu bar2 action    
    def PlotTracesFunc(self):
        try:
            fig,_=experiment.plotRawData()
            subWindow = QMdiSubWindow(self)
            widget=Grapth(fig,subWindow)
            self.formatSubWindow(subWindow,widget,'Traces')
            self.repaint()
        except:
            self.messageError()
   
    #menu bar2 after buton plot spectra manually 
    def button04Func(self):
        try:
            text14_str=self.text14.text()
            average=int(self.text04.text())
            times=[float(i) for i in text14_str.split(',')]
            text24_value=self.text24.text()
            if text24_value=='False' or text24_value=='false':
                fig,_=experiment.plot_spectra(times=times, n_points=average, legend=False)
            elif self.text24.text().isdigit():
                fig,_=experiment.plot_spectra(times=times, n_points=average, ncol=int(self.text24.text()))
            else:
                fig,_=experiment.plot_spectra(times=times, n_points=average)
            subWindow = QMdiSubWindow(self)
            widget=Grapth(fig,subWindow)
            self.formatSubWindow(subWindow,widget,'Spectra plotted manually')
            self.repaint()
        except:
            self.messageError()
    
    #menu bar2 
    def PlotSpecManualFunc(self):
        self.pop_edit_status=self.edit_general['pop_edit']
        self.proc_edit_status=self.edit_general['proc_edit']
        self.noZones()
        self.edit_general['plot_spec_edit']=True        
        self.repaint()
    
    #botom after menu bar 4 baseline substraction for baslien with only 1 spectra
    def button03Func(self):
        try:
            if self.initialize_global_fit:
                n_spect=int(self.text03.text())
                experiment.baselineSubstraction(n_spect,only_one=True)
                self.printMessage('Baseline substracted')
                self.repaint() 
            else:
                self.messageError()  
        except:
            self.messageError()
    
    #botom after menu bar 4 baseline substraction for baslien with N spectra average
    def button13Func(self):
        try:
            if self.initialize_global_fit:
                n_spect=int(self.text03.text())
                experiment.baselineSubstraction(n_spect,only_one=False)
                self.printMessage('Baseline substracted')
                self.repaint()
            else:
                self.messageError()
        except:
            self.messageError()
        #short comment: i usually dont use layouts, but apparently qt requires them, so i combined here my code and layout
        #i usually create subobjects in qt (textfields, buttons etc), and spectfy directly their location in the window in pixels using setgeometry function
        #but there is also another way to set arrangement of these subobjects, you can use layouts, which are object in the middle between you window and subobjects
        #so instead of specyfying directly pixels, you just create chosen layout, add items to layout and layout to your window
        #then layout sets the positions of objects and you dont have to do this
        #so in this case i put layout with pyplot, but objects on the left are specified by pixels
        #so you can see two different ways to do the job, but in practice for sure it is better to use only layouts or only directly given coords in pixels

    #menu bar4 derivate 
    def derivateFunc(self) :
        self.pop_edit_status=self.edit_general['pop_edit']
        self.proc_edit_status=self.edit_general['proc_edit']
        self.noZones()
        self.edit_general['deriv_edit']=True        
        self.repaint()
    
    #menu bar4 baselinesubs
    def baselineSubtrationFunc(self):
        self.noZones()
        self.edit_general['baseline_edit']=True        
        self.repaint()
    
    def formatSubWindow(self,subWindow,Widget,name=None,close_button=False,extra_button=None):
        widget = Widget
        subWindow.setWidget(widget)
        widget.setLayout(QVBoxLayout()) 
        if name is None:
            subWindow.setObjectName("New_Window")
            subWindow.setWindowTitle("New SubWindow")
        else:
            subWindow.setObjectName(name)
            subWindow.setWindowTitle(name)
        if extra_button is not None:
            if type(extra_button) == list:
                for i in extra_button:
                    subWindow.layout().addWidget(i)
            else:
                subWindow.layout().addWidget(extra_button)
        if close_button:
            btn = QPushButton("close", widget) 
            subWindow.layout().addWidget(btn)
            btn.clicked.connect(subWindow.close)
        self.mdiArea.addSubWindow(subWindow)
        subWindow.show()
        return widget
#        widget.layout().addWidget(btn)
#        btn.clicked.connect(subWindow.close)
    #button1 go to fit
    def button1Func(self): 
        if self.button1.currentText()=="Exponential Fit":
            self.prev_number_traces_edit=self.edit_general['number_traces_edit']
            self.noZones()
            self.edit_general['pop_edit']=True        
            self.repaint()
        
        elif self.button1.currentText()=="Target Fit":
            self.prev_number_traces_edit=self.edit_general['number_traces_edit']
            self.noZones()
            self.edit_general['target_fit']=True        
            self.repaint()
        
        elif self.edit_general['pop_edit'] or self.edit_general['target_fit']:  
            self.noZones()   
            self.edit_general['number_traces_edit']=self.prev_number_traces_edit
            self.repaint()
        
    #button after go to fit change parameters  
    def button2Func(self): 
        try:
            self.paramsWindowall.close()
        except:
            pass
        self.paramsWindowchange = QMdiSubWindow(self)
        self.paramsWindowchange.resize(300,350)
        rows=self.eps_table.rowCount()
        self.params_table_change = QTableWidget(rows,3,self)
        self.params_table_change.setHorizontalHeaderLabels(('Parameter', 'Value','Fix/Optimized'))
        self.params_table_change.setColumnWidth(0,60)
        self.params_table_change.setColumnWidth(1,50)
        self.params_table_change.setColumnWidth(3,60)
        for i in range(rows):
            namet=QLabel(self.eps_table.cellWidget(i, 0).text())
            change=self.eps_table.cellWidget(i, 2).text()
            valor=QTableWidgetItem(self.eps_table.cellWidget(i,1).text())
            combo = QComboBox()
            if change=='Yes':
                combo.addItem("Fix")
                combo.addItem("Optimize")
            else:    
                combo.addItem("Optimize")
                combo.addItem("Fix")
            self.params_table_change.setCellWidget(i,0,namet)
            self.params_table_change.setItem(i,1,valor)     
            self.params_table_change.setCellWidget(i,2,combo) 
        self.formatSubWindow(self.paramsWindowchange,self.params_table_change,'Parameters')
        if experiment.deconv:
            self.check_inf=QCheckBox('Include a infinite time to evaluate product formation',self.paramsWindowchange)
            if experiment.tau_inf is None:
                self.check_inf.setCheckState(0)
            else:
                self.check_inf.setCheckState(2)
            self.paramsWindowchange.layout().addWidget(self.check_inf)
        btn = QPushButton("Change parameters", self.paramsWindowchange) 
        self.paramsWindowchange.layout().addWidget(btn)
        btn.clicked.connect(self.changeParams)
        btn_all = QPushButton("Advance options", self.paramsWindowchange) 
        self.paramsWindowchange.layout().addWidget(btn_all)
        btn_all.clicked.connect(self.changeAllParams)
        btn_close = QPushButton("Close", self.paramsWindowchange) 
        self.paramsWindowchange.layout().addWidget(btn_close)
        btn_close.clicked.connect(self.paramsWindowchange.close)
        self.repaint()
    
    def changeAllParams(self):
        try:
            self.paramsWindowchange.close()
            self.paramsWindowall = QMdiSubWindow(self)
            self.paramsWindowall.resize(455,500)
            rows=len(experiment.initial_params)
            if experiment.prefit_done:
                params=experiment.params
            else:
                params=experiment.initial_params
            self.params_table_all = QTableWidget(rows,6,self)
            self.params_table_all.setHorizontalHeaderLabels(('Wavelength','Parameter', 'Value','Fix/Optimized','min','max','Fix/Optimized'))
            self.params_table_all.setColumnWidth(0,70)
            self.params_table_all.setColumnWidth(1,70)
            self.params_table_all.setColumnWidth(2,60)
            self.params_table_all.setColumnWidth(3,100)
            self.params_table_all.setColumnWidth(4,50)
            self.params_table_all.setColumnWidth(5,50)
            sli=rows//len(experiment.wavelength)
            if sli>0:   
                waves=[i for i in range(rows)[::sli]]
            else:
                waves=[i for i in range(rows)]
            names=[key for key in params.keys()]
            for i in range(rows):
                if i in waves:
                    wave= QLabel('{:.1f}'.format(experiment.wavelength[i//sli]))
                    self.params_table_all.setCellWidget(i,0,wave)
                param= QLabel(names[i])
                self.params_table_all.setCellWidget(i,1,param)
                name=params[names[i]].expr
                if name is None:
                    valor=QTableWidgetItem('{:.4f}'.format(params[names[i]].value))
                    mini=QTableWidgetItem('{:.4f}'.format(params[names[i]].min))
                    maxi=QTableWidgetItem('{:.4f}'.format(params[names[i]].max))
                    self.params_table_all.setItem(i,2,valor)
                    self.params_table_all.setItem(i,4,mini)
                    self.params_table_all.setItem(i,5,maxi)
                    combo = QComboBox()
                    if params[names[i]].vary:
                        combo.addItem("Optimize")
                        combo.addItem("Fix")   
                    else:    
                        combo.addItem("Fix")
                        combo.addItem("Optimize")              
                else:
                    valor=QLabel('{:.4f}'.format(params[names[i]].value))
                    mini=QLabel('{:.4f}'.format(params[names[i]].min))
                    maxi=QLabel('{:.4f}'.format(params[names[i]].max))
                    self.params_table_all.setCellWidget(i,2,valor)
                    self.params_table_all.setCellWidget(i,4,mini)
                    self.params_table_all.setCellWidget(i,5,maxi)
                    combo=QLabel(f'same as: {name}')
                self.params_table_all.setCellWidget(i,3,combo)   
            self.formatSubWindow(self.paramsWindowall,self.params_table_all,'Parameters',)
            btn = QPushButton("Change parameters", self.paramsWindowall) 
            self.paramsWindowall.layout().addWidget(btn)
            btn.clicked.connect(self.changeParamsAll)
            btn_less = QPushButton("Less Options", self.paramsWindowall) 
            self.paramsWindowall.layout().addWidget(btn_less)
            btn_less.clicked.connect(self.button2Func)
            btn_close = QPushButton("Close", self.paramsWindowall) 
            self.paramsWindowall.layout().addWidget(btn_close)
            btn_close.clicked.connect(self.paramsWindowall.close)
            self.repaint()  
        except:
            self.messageError()
    
    def changeParamsAll(self):
        try:
            if experiment.prefit_done:
                params=experiment.params
            else:
                params=experiment.initial_params
            self.variations=[]
            count=0
            if experiment.deconv: 
                important_names=['t0_1','fwhm_1']+['tau%i_1' %(i+1) for i in range(experiment.exp_no)]
            else:
                important_names=['t0_1']+['tau%i_1' %(i+1) for i in range(experiment.exp_no)]
            names=[key for key in params.keys()]
            for i,ii in enumerate(names):
                if ii in important_names:
                    Change=self.params_table_all.cellWidget(i, 3).currentText()
                    dato=self.params_table_all.item(i, 2).text()
                    if Change=='Fix':
                        vary=QLabel('Yes')
                        self.variations.append(False)
                    else:
                        vary=QLabel('No')
                        self.variations.append(True) 
                    self.eps_table.setCellWidget(count,2,vary)
                    namet=QLabel(str(dato))
                    self.eps_table.setCellWidget(count,1,namet)
                    count += 1
                if params[ii].expr is None:
                    valor=float(self.params_table_all.item(i,2).text())
                    mini=float(self.params_table_all.item(i,4).text())
                    maxi=float(self.params_table_all.item(i,5).text())
                    vary=False if self.params_table_all.cellWidget(i, 3).currentText() == 'Fix' else True
                    if experiment.prefit_done:
                       experiment.params[ii].value=valor
                       experiment.params[ii].min=mini
                       experiment.params[ii].max=maxi
                       experiment.params[ii].vary=vary
                    experiment.initial_params[ii].value=valor
                    experiment.initial_params[ii].min=mini
                    experiment.initial_params[ii].max=maxi
                    experiment.initial_params[ii].vary=vary
                else:
                    pass
            self.paramsWindowall.close()
        except:
            self.messageError()
      
    def changeParams(self):
        try:
            self.variations=[]
            datos=[float(self.params_table_change.item(i,1).text()) for i in range(self.params_table_change.rowCount())]
            if experiment.deconv:
                if self.check_inf.checkState()==0:
                    experiment.tau_inf=None
                else:
                    experiment.tau_inf=1E+12 
            for i in range(len(datos)):
                Change=self.params_table_change.cellWidget(i, 2).currentText()
                if Change=='Fix':
                    vary=QLabel('Yes')
                    self.variations.append(False)
                else:
                    vary=QLabel('No')
                    self.variations.append(True)  
                self.eps_table.setCellWidget(i,2,vary)
                namet=QLabel(str(datos[i]))
                self.eps_table.setCellWidget(i,1,namet)
            extra_count=1
            experiment.initial_params['t0_1'].value=datos[0]
            experiment.initial_params['t0_1'].vary=self.variations[0]
            if experiment.deconv:
                experiment.initial_params['fwhm_1'].value=datos[1]
                experiment.initial_params['fwhm_1'].vary=self.variations[1]
                extra_count=2
            for i in range(experiment.exp_no):
                experiment.initial_params['tau%i_1'%(i+1)].value=datos[extra_count+i]
                experiment.initial_params['tau%i_1'%(i+1)].vary=self.variations[extra_count+i]
            if experiment.prefit_done:
                experiment.params['t0_1'].value=datos[0]
                experiment.params['t0_1'].vary=self.variations[0]
                if experiment.deconv:
                    experiment.params['fwhm_1'].value=datos[1]
                    experiment.params['fwhm_1'].vary=self.variations[1]
                for i in range(experiment.exp_no):
                    experiment.params['tau%i_1'%(i+1)].value=datos[extra_count+i]
                    experiment.params['tau%i_1'%(i+1)].vary=self.variations[extra_count+i]
            self.paramsWindowchange.close()
        except:
            self.messageError()
            
    def newParam(self):
        try:
            new=self.param_table.rowCount()
            self.param_table.insertRow(new)
            if self.select_process.currentText() == 'Deconvolution':
                resta=1
            else:
                resta=0
            combo = QComboBox()
            combo.addItem("Optimize")
            combo.addItem("Fix")
            self.param_table.setCellWidget(new,2,combo)
            texto='tau '+str(new-resta)
            name=QLabel(texto)
            it= QTableWidgetItem(str(0))
            self.param_table.setItem(new,1,it)
            self.param_table.setCellWidget(new,0,name)
        except:
            self.messageError()
    #lambda:self.param_table.insertRow(self.param_table.rowCount())
    #button after go to fit    
    def delParam(self,table,minimum=2):
        dele=table.rowCount()
        if dele>minimum:
            table.setRowCount(dele-1)
      
    def button3Func(self):
        try: 
#            experiment.params_initialized=False
            self.corrected_GVD=experiment.GVD_correction
            self.paramsWindow = QMdiSubWindow(self)
            self.paramsWindow.resize(300,400)
            init_val=str(0)
            if self.select_process.currentText() == 'Deconvolution':
                rango=3
                experiment.deconv=True
            else:
                if self.initialize_global_fit==True:
                    init_val=str(experiment.x[0])
                experiment.deconv=False
                rango=2
            self.param_table = QTableWidget(rango,3,self)
            self.param_table.setHorizontalHeaderLabels(('Parameter', 'Value','Fix/Optimized'))
            self.param_table.setColumnWidth(0,60)
            self.param_table.setColumnWidth(1,50)
            self.param_table.setColumnWidth(3,60)
            if rango==2:
               for ii in range(rango):
                    combo = QComboBox()
                    if ii==0:
                        combo.addItem("Fix")
                        combo.addItem("Optimize")
                    else:    
                        combo.addItem("Optimize")
                        combo.addItem("Fix")
                    self.param_table.setCellWidget(ii,2,combo)    
            if rango==3:
               for ii in range(rango):
                    combo = QComboBox()
                    if ii==1:
                        combo.addItem("Fix")
                        combo.addItem("Optimize")
                    else:    
                        combo.addItem("Optimize")
                        combo.addItem("Fix")
                    self.param_table.setCellWidget(ii,2,combo)
            texto='time 0'
            name=QLabel(texto)
            it= QTableWidgetItem(init_val)
            self.param_table.setCellWidget(0,0,name)
            self.param_table.setItem(0,1,it)
            if self.select_process.currentText() == 'Deconvolution':    
                experiment.deconv=True
                texto='fwhm'
                name=QLabel(texto)
                if self.IRF_value is None:
                    fwhm_value=0.12
                else:
                    fwhm_value=self.IRF_value
                it= QTableWidgetItem(str(fwhm_value))
                #name=QTableWidgetItem('tau '+str(ii+1))
                self.param_table.setCellWidget(1,0,name)
                self.param_table.setItem(1,1,it)
            texto='tau 1'
            name=QLabel(texto)
            it= QTableWidgetItem(str(0.15))
            self.param_table.setCellWidget(rango-1,0,name)
            self.param_table.setItem(rango-1,1,it)    
            self.formatSubWindow(self.paramsWindow,self.param_table,'Parameters',)
            if rango==3:
                self.check_inf=QCheckBox('Include a infinite time to evaluate product formation',self.paramsWindow)
                self.check_inf.setCheckState(2)
                self.paramsWindow.layout().addWidget(self.check_inf)
                if self.corrected_GVD==False:
                    self.check=QCheckBox('Non of the GVD data correction functions has been Run; \nplease check box if data was corrected previously',self.paramsWindow)
                    self.paramsWindow.layout().addWidget(self.check)
            btn_add = QPushButton("Add one extra time", self.paramsWindow) 
            self.paramsWindow.layout().addWidget(btn_add)
            btn_add.clicked.connect(lambda:self.newParam())
            btn_del = QPushButton("Delete one time", self.paramsWindow) 
            self.paramsWindow.layout().addWidget(btn_del)
            btn_del.clicked.connect(lambda:self.delParam(self.param_table,rango))
            btn = QPushButton("Set parameters", self.paramsWindow) 
            self.paramsWindow.layout().addWidget(btn)
            btn.clicked.connect(self.setParams)
            btn_close = QPushButton("Close", self.paramsWindow) 
            self.paramsWindow.layout().addWidget(btn_close)
            btn_close.clicked.connect(self.paramsWindow.close)
            self.repaint()
        except:
            self.messageError()
    
    def setParams(self):
        try:
            if experiment.SVD_fit:
                value=self.param_table.rowCount()
                if experiment.deconv:
                    value -= 2
                else:
                    value -= 1
                if experiment.data.shape[1]==value:
                    pass
                else:
                    error_dialog = QMessageBox(self)
                    error_dialog.setText('The number of selected singular vectors is different than the number of exponential components\
                                         \n\n Althought the parameters will be initialized, we recomend to select the same number of parameters as singular vectors. \
                                         \n\n If you did this in purporse go On, if Not:\n  --> select one singular vector more \
                                         \n  --> Or initialize parameters with one component less')
                    error_dialog.exec()
#            experiment.params_initialized=False
            rows=self.param_table.rowCount()
            self.eps_table.setRowCount(rows)
            self.variations=[]
            if experiment.deconv:
                if self.check_inf.checkState()==0:
                    experiment.tau_inf=None
                else:
                    experiment.tau_inf=1E+12
                try:
                    if self.check.checkState()==2 and self.corrected_GVD==False:
                        self.corrected_GVD=True
                        experiment.general_report['Preprocessing']['GVD correction']=f'\tGVD was previously corrected'
                except:
                    pass
            datos=[]
            for i in range(rows):
                name=self.param_table.cellWidget(i, 0).text()
                namet=valort=QLabel(name)
                Change=self.param_table.cellWidget(i, 2).currentText()
                valort=QLabel(self.param_table.item(i,1).text())
                datos.append(float(self.param_table.item(i,1).text()))
                if Change=='Fix':
                    vary=QLabel('Yes')
                    self.variations.append(False)
                else:
                    vary=QLabel('No')
                    self.variations.append(True)  
                self.eps_table.setCellWidget(i,0,namet)
                self.eps_table.setCellWidget(i,1,valort)
                self.eps_table.setCellWidget(i,2,vary)
            vary_t0=self.variations[0]
            if self.select_process.currentText() == 'Deconvolution':
                opt_fwhm=self.variations[1]
                self.taus=datos[2:]
            else:
                opt_fwhm=False
                self.taus=datos[1:]   
            experiment.initialParams2(datos[0],self.taus,fwhm=datos[1],vary_t0=vary_t0,opt_fwhm=opt_fwhm,GVD_corrected=self.corrected_GVD)
            print(self.variations,datos)
            self.weights_true.setCheckState(0)
            try:
                self.paramsWindow.close()
            except:
                pass
        except:
            self.messageError()
    #button after go to fit do pre fit
    def button31Func(self): 
        try:
            # This causes my_worker.run() to eventually execute in my_thread:
            my_worker = GenericWorker(experiment.preFit)
            my_worker.moveToThread(self.my_thread_preFit)
            my_worker.start.emit("hello")
            # my_worker.finished.connect(self.xxx)
            self.preFit_worker = my_worker
            #experiment.preFit()
            self.repaint()
        except:
            self.messageError()
        
    def button31WeightsFunc(self):
        try:
            subWindow=QMdiSubWindow(self)
            subWindow.setWindowTitle("Setting fit weigths")
            subWindow.layout().setContentsMargins(10,10,10,0)
            sublayout2=QVBoxLayout()
            sublayout2.setAlignment(Qt.AlignTop)
            sublayout2.setContentsMargins(10,10,10,10)
            subWindow.layout().addWidget(QLabel('Time where weigths should be applied:'))
            rango=QLineEdit('', self)
            rango.setMaximumWidth(150)
            subWindow.layout().addWidget(rango)
            label_help=QLabel('Recomendation: start wiht constant mode and a low value.')
            subWindow.layout().addWidget(label_help)
            linear=QCheckBox('constant',subWindow)
            linear.setCheckState(2)
            exp=QCheckBox('exponetial',subWindow)
            r_exp=QCheckBox('reverse exponetial',subWindow)
            combined=QCheckBox('combined exponetial',subWindow)
            sublayout2.addWidget(linear)
            sublayout2.addWidget(exp)
            sublayout2.addWidget(r_exp)
            sublayout2.addWidget(combined)
            def calWeigths():
                try:
                    if linear.isChecked():
                        typo='constant'
                        value=val.value()
                    elif exp.isChecked():
                        typo='exponetial'
                        value=exp_val.value()
                    elif r_exp.isChecked():
                        typo='r_exponetial'
                        value=exp_val.value()
                    else:
                        typo='exp_mix'
                        value=exp_val.value()
                    rang=[float(i) for i in rango.text().split('-')]
                    experiment.define_weights(rang, typo, value)
                    for  i in range(len(experiment.x)):
                        it= QTableWidgetItem(str(experiment.weights['vector'][i]))
                        par_table.setItem(2,i,it)
                except:
                    by_dialog = QMessageBox(self)
                    by_dialog.setText("Weights could not be defined")
                    by_dialog.exec()
            cal_weigths=QPushButton('Define weights', subWindow)
            cal_weigths.clicked.connect(calWeigths)
            sublayout2.addWidget(cal_weigths)
            val_label=QLabel('constant value')
            expval_label=QLabel('exponetila value')
            expval_label.setVisible(False)
            val=QSpinBox(ex) 
            val.setMinimum(2)
            val.setValue(10)
            val.setMaximum(100)
            val.setSingleStep(5) 
            exp_val=QSpinBox(self) 
            exp_val.setMinimum(2)
            exp_val.setMaximum(4)
            exp_val.setVisible(False)
            sublayout1=QVBoxLayout()
#            sublayout1.addWidget(label_help, alignment=Qt.AlignTop)
            sublayout1.setAlignment(Qt.AlignCenter)
            sublayout1.setContentsMargins(10,0,10,10)
            sublayout1.addWidget(val_label,)
            sublayout1.addWidget(expval_label)
            sublayout1.addWidget(val)
            sublayout1.addWidget(exp_val)
            mainlayout=QHBoxLayout()
            mainlayout.setContentsMargins(10,10,10,0)
            mainlayout.addLayout(sublayout2)
            mainlayout.addLayout(sublayout1)
            def showText():
                if linear.isChecked():
                    val_label.show()
                    val.show()
                    expval_label.setVisible(False)
                    exp_val.setVisible(False)
                else:
                    val_label.setVisible(False)
                    val.setVisible(False)
                    expval_label.show()
                    exp_val.show()
            def applyAndClose():
                fit_defined=False
                try:
                    weight=np.array([int(par_table.item(2,i).text()) for i in range(len(experiment.x))])
                    experiment.weights['vector']=weight
                    fit_defined=True
                except:
                    if experiment.weights['vector'] is not None and len(experiment.weights['vector'])==len(experiment.x):
                        by_dialog = QMessageBox(ex)
                        by_dialog.setText("Value introduce in the table manually not understood.\nOriginal weights set")
                        by_dialog.exec()
                        fit_defined=True
                    else:
                        by_dialog = QMessageBox(self)
                        by_dialog.setText("No weights have been defined")
                        by_dialog.exec()
                if fit_defined:
                    ex.weights_true.setCheckState(2)
                    ex.printMessage('weights set')
                subWindow.close()
            linear.stateChanged.connect(showText)
            group1=QButtonGroup(self)
            group1.addButton(linear)
            group1.addButton(exp)
            group1.addButton(r_exp)
            group1.addButton(combined)
            subWindow.layout().addLayout(mainlayout)
            ex.mdiArea.addSubWindow(subWindow)
            par_table = QTableWidget(subWindow)
            par_table.setRowCount(3)
            par_table.setColumnCount(len(experiment.x))
            par_table.setVerticalHeaderLabels(['Index','Time point','weights'])
            for i in range(len(experiment.x)):
                index=QLabel(str(i+1))
                index.setAlignment(Qt.AlignCenter)
                time_p=QLabel('{:.2f}'.format(experiment.x[i]))
                time_p.setAlignment(Qt.AlignCenter)
                par_table.setCellWidget(0,i,index)
                par_table.setCellWidget(1,i,time_p)
                if experiment.weights['vector'] is not None:
                    try:
                        it= QTableWidgetItem(str(experiment.weights['vector'][i]))
                        par_table.setItem(2,i,it)
                    except:
                        experiment.weights['vector']=None
            if experiment.weights['vector'] is not None and len(experiment.weights['vector'])==len(experiment.x):
                rango.setText('-'.join([str(i) for i in experiment.weights['range']]))
                if experiment.weights['type'] == 'mix_exp':
                    combined.setCheckState(2)
                elif experiment.weights['type'] == 'exponential':
                    exp.setCheckState(2)
                elif experiment.weights['type'] == 'r_exponential':
                    r_exp.setCheckState(2)
                else:
                    linear.setCheckState(2)
            label_help2=QLabel('The weights array multiply the error vector for each trace in the fit.')
            par_table.horizontalHeader().setVisible(False)
            subWindow.layout().addWidget(par_table)
            subWindow.layout().addWidget(label_help2)
            apply_weigths=QPushButton('Set and close', subWindow)
            apply_weigths.clicked.connect(applyAndClose)
            subWindow.layout().addWidget(apply_weigths)
            subWindow.resize(400,420)
            subWindow.show()    
        except:
            self.messageError()
            
    def fitWithWeights(self):
        if self.weights_true.isChecked():
            self.weights_true.show()
            self.button31.setVisible(False)
        else:
            experiment.weights['apply']=False
            self.weights_true.setVisible(False)
            self.button31.show()
            self.printMessage('No weights')
    
    def button4Func(self): #finished edition of process
        self.repaint()
        
    #button after go to fit verify fit    
    def button41Func(self): #deletes arrow
        self.repaint()
    #button after go to fit final fit     
    @pyqtSlot()
    def updateParams(self):
        if self.stopped_thread:
            self.button1.show()
            self.label00.show()
            self.button0.show()
            self.noZones()
            self.edit_general['target_fit']=self.target_model_status
            self.edit_general['pop_edit']=self.pop_edit_status
            self.repaint()
        else:
            self.fit_number+=1
            valor=[experiment.params['t0_1'].value]
            if experiment.deconv:
                valor.append(experiment.params['fwhm_1'].value)
            valor=valor+[experiment.params['tau%i_1'% (i+1)].value for i in range(experiment.exp_no)]
#            fit_results=QTextEdit()
#            fit_results.insertPlainText(f'Parameter\tInitial value\tFinal Value\tFixed\n')
            for i in range(len(valor)):
#                line=[self.eps_table.cellWidget(i, 0).text(),
#                      self.eps_table.cellWidget(i, 1).text(),'{:.4f}'.format(valor[i])+'  ',
#                      self.eps_table.cellWidget(i, 2).text(),'\n']
#                fit_results.insertPlainText('\t'.join(line))
                it=QLabel('{:.4f}'.format(valor[i]))
                self.eps_table.setCellWidget(i,1,it)
            
#            if experiment.tau_inf is not None and experiment.deconv:
#                fit_results.insertPlainText(f'\ntau infinite included: Yes')
#            elif experiment.deconv:
#                fit_results.insertPlainText(f'\ntau infinite included: No')
#            else:
#                pass
#            fit_results.insertPlainText(f'\nnumber of Traces fitted: {experiment.data_before_last_Fit.shape[1]}')
#            fit_results.insertPlainText(f'\nnumber of iterations: {experiment.resultados.nfev}')
#            fit_results.insertPlainText(f'\nnumber of parameters optimized: {len(experiment.params)}')
            self.button1.show()
            self.label00.show()
            self.button0.show()
            self.fit_in_progress=False
            self.noZones()
            typo=experiment.all_fit[experiment.fit_number][8][2]
            if typo == 'Exponential':
                self.edit_general['pop_edit']=True 
            else:
                self.edit_general['target_fit']=True
                population = self.model.populations
                arrow = self.model.processes
                names=[i.name for i in population]
                pos=[[i.rect.x(),i.rect.y()] for i in population]
                H_W=[[i.rect_h,i.rect_w] for i in population]
                taus=[experiment.params[i].value for i in experiment.params if 'tau' in i]
                max_y=max([i[1] for i in pos])
                max_x=max([i[0] for i in pos])+H_W[0][1]
                min_x=min([i[0] for i in pos])
                init_arrow=[(i.source.rect.x()+i.source.rect_w/2,max_y-i.source.rect.y()-i.target.rect_h/4) for i in arrow]
                final_arrow=[(abs(i.target.rect.x()-i.source.rect.x()),((max_y-i.target.rect.y())-(max_y-i.source.rect.y())+i.target.rect_h*1.05)) for i in arrow]
                experiment.target_models[experiment.fit_number]=[pos,H_W,names,taus,max_y,max_x,min_x,init_arrow,final_arrow]    
            self.repaint()
            self.windowFit(self.fit_number)
    
    @pyqtSlot(int)     
    def progressFit(self,val):
        QCoreApplication.processEvents()
        if val==-2:
            by_dialog = QMessageBox(self)
            by_dialog.setText(f"Please be paitient")
            by_dialog.exec()
        elif val==-3:
            by_dialog = QMessageBox(self)
            by_dialog.setText(f"Fit completed Transfering Data please wait. We are trying to calculate errors and correlations between parameters, this can take a while.")
            by_dialog.exec()
        elif val==-1:
            pass
        elif val==-4:
            by_dialog = QMessageBox(self)
            by_dialog.setText(f"Fit to singular vectors completed completed. Currently fitting all the traces with time values found fixed")
            by_dialog.exec()
#            num=1
#            while num<5000:
#                num += 0.001 
#                self.progress_fit.setValue(num) 
#            num=1
#            if self.fit_in_progress:
#                while num<5000:
#                    num += 0.001 
#                    self.progress_fit.setValue(num)   
            
        else:
            self.progress_fit.setValue(val) 
        thread_Fit=True
        if val%300==0 and val>290:
            if thread_Fit:
                self.my_thread_Fit.working_thread.yieldCurrentThread()
                thread_Fit=False
            else:
                self.my_thread_prefit.working_thread.yieldCurrentThread()
                thread_Fit=True
    
    def button51Func(self): 
        try:
            number_params=len(experiment.initial_params)
            if number_params<4500:
                if number_params<600:
                    pass
                elif number_params<1200:
                    by_dialog = QMessageBox(self)
                    by_dialog.setText(f"There are {number_params} parameters to optimized. Please do not worry if optimization takes some minutes¡")
                    by_dialog.exec()
                elif number_params<1800:
                    by_dialog = QMessageBox(self)
                    by_dialog.setText(f"There are {number_params} parameters. The operational system might think program is not working¡ \nthe program still works in the background let it finish")
                    by_dialog.exec()
                elif number_params<3250:
                    by_dialog = QMessageBox(self)
                    by_dialog.setText(f"There are {number_params} parameters. This will take long a and the operational will think system  is not working¡ \nthe program still works in the background let it finish")
                    by_dialog.exec()
                else:
                    by_dialog = QMessageBox(self)
                    by_dialog.setText(f"There are {number_params} parameters to optimized we strongly recomend a selection of traces before fit for next time. The fit can be done but will really take long time, the operational system will think program is not working. Please let the optimization finish if you manage to be paitient¡")
                    by_dialog.exec()
                self.stopped_thread=False
                self.fit_in_progress=True
                self.pop_edit_status=self.edit_general['pop_edit']
                self.target_model_status=self.edit_general['target_fit']
                print(self.pop_edit_status)
                self.noZones()
                print(self.pop_edit_status)
                self.progress_fit.setValue(experiment.number_it) 
                self.edit_general['fitting_edit']=True
                self.repaint()
                self.label00.setVisible(False)
                self.button1.setVisible(False)
                self.button0.setVisible(False)
                if experiment.type_fit == 'Exponential':
                    if self.select_process.currentText() == 'Deconvolution':
                        experiment.initial_params['t0_1'].vary=self.variations[0]
                        experiment.initial_params['fwhm_1'].vary=self.variations[1]
                        if experiment.prefit_done:
                             experiment.params['t0_1'].vary=self.variations[0]
                             experiment.params['fwhm_1'].vary=self.variations[1]
                        self.vary_taus=self.variations[2:]
                    else:
                        experiment.initial_params['t0_1'].vary=self.variations[0]
                        if experiment.prefit_done:
                             experiment.params['t0_1'].vary=self.variations[0]
                        self.vary_taus=self.variations[1:]   
                
                self.pop_edit_status=self.edit_general['pop_edit']
                self.proc_edit_status=self.edit_general['proc_edit']
                maxfev=int(self.text51.text())
#                if experiment.prefit_done==False:
#                    experiment.preFit()
                if self.weights_true.isChecked():
                    apply_weights=True
                else:
                    apply_weights=False
                if experiment.SVD_fit:
                    self.progress_fit.setMaximum(0)
                    self.progress_fit.setMinimum(0)
                    self.progress_fit.setValue(0)
                else:
                    self.progress_fit.setMaximum(maxfev)
                if experiment.type_fit == 'Exponential':
                    my_worker = GenericWorker(experiment.finalFit,self.vary_taus,maxfev,apply_weights=apply_weights)
                else:
                    my_worker = GenericWorker(experiment.finalFit,True,maxfev,apply_weights=apply_weights)
                my_worker.moveToThread(self.my_thread_Fit)
                my_worker.start.emit("hello")
                my_worker.finished.connect(self.updateParams)
                self.Fit_working_thread=my_worker
#                self.working_thread=WorkingThread(maxfev,self.vary_taus)
#                self.working_thread.finished.connect(self.updateParams)
                self.working_thread_iter=threadItter()
                self.working_thread_iter.iter_num.connect(self.progressFit)
#                self.working_thread.start(QThread.TimeCriticalPriority)
#                self.priority=self.working_thread.priority()
                self.working_thread_iter.start()
                
            else:
                 by_dialog = QMessageBox(self)
                 by_dialog.setText(f"There are {number_params} parameters to optimized please do a smaller selection of traces. The fit cannot be done as this might block the computer.")
                 by_dialog.exec()
        except:
            self.messageError()

    #button after go to previous fit
    def button61Func(self,export=False):
        if len(experiment.all_fit) == 0 and len(experiment.single_fits) == 0 :
            by_dialog = QMessageBox(self)
            by_dialog.setText("So far no fits have been run")
            by_dialog.exec()
        elif len(experiment.all_fit) == 0 and export:
            by_dialog = QMessageBox(self)
            by_dialog.setText("So far no fits have been run")    
        else:
            self.prev_subWindow = QMdiSubWindow(self)
            self.prev_subWindow.setObjectName('Fit selection')
            self.prev_subWindow.setWindowTitle('Fit selection')
            main_layout=QHBoxLayout()
            label = QLabel('---------------------', self.prev_subWindow)
            layout=QVBoxLayout()
            layout.addWidget(label)
            layout.setAlignment(Qt.AlignTop)
            self.prev_subWindow.layout().addLayout(main_layout)
            if len(experiment.all_fit) != 0:
                
                #♣self.prev_subWindow.resize(150,200)
                #self.prev_subWindow.layout(QVBoxLayout())
                group=QButtonGroup(self.prev_subWindow)
                widget_dict={}
                for i in range(len(experiment.all_fit)):
                    widget_dict['widget  %i'%(i+1)]=QRadioButton("fit number: %i" %(i+1),self.prev_subWindow)
                    widget_dict['widget  %i'%(i+1)].setObjectName(f"fit number: %i" %(i+1))
                    if export:
                        widget_dict['widget  %i'%(i+1)].toggled.connect(lambda:self.exportWindowF(group))
                    else:
                        widget_dict['widget  %i'%(i+1)].toggled.connect(lambda:self.displayFit(group))
                    group.addButton(widget_dict['widget  %i'%(i+1)])
                    layout.addWidget(widget_dict['widget  %i'%(i+1)])
                    label = QLabel('---------------------', self.prev_subWindow)
                    layout.addWidget(label)
                    if i%10==9 and i>1:
                        label2 = QLabel('---------------------', self.prev_subWindow)
                        main_layout.addLayout(layout)
                        layout=QVBoxLayout()
                        layout.addWidget(label2)
                        layout.setAlignment(Qt.AlignTop)
            if len(experiment.single_fits) != 0 and export == False: 
                if len(experiment.all_fit) %10==0 and len(experiment.all_fit) != 0:
                    label2 = QLabel('---------------------', self.prev_subWindow)
                    main_layout.addLayout(layout)
                    layout=QVBoxLayout()
                    layout.addWidget(label2)
                    layout.setAlignment(Qt.AlignTop)
                single_fit=QRadioButton('single fits')
                single_fit.toggled.connect(lambda:self.displaySingleFits())
                layout.addWidget(single_fit)
                label = QLabel('---------------------', self.prev_subWindow)
                layout.addWidget(label)
            main_layout.addLayout(layout)
            main_layout.setContentsMargins(10,0,10,0)
            self.mdiArea.addSubWindow(self.prev_subWindow)
            btn_close = QPushButton("Close", self.prev_subWindow) 
            self.prev_subWindow.layout().addWidget(btn_close)
            btn_close.clicked.connect(self.prev_subWindow.close)
            self.prev_subWindow.setMinimumHeight(120)
            self.prev_subWindow.show()
            self.repaint()
    
    def displayFit(self,radio):
        try:
            self.prev_subWindow.close()
            for fit_number,ii in enumerate(radio.buttons()):
                if ii.isChecked():
                    self.windowFit(fit_number+1)
        except:
            self.messageError()
    
    def displaySingleFits(self):
        try:
            self.prev_subWindow.close()
            subWindow_OneTrace_Fit_Res=QMdiSubWindow(self)
            subWindow_OneTrace_Fit_Res.setWindowTitle("Single and integral band results")
            timelabel='Time ('+experiment.time_unit+')'
            deltalabel=r'$\Delta$A'
            figwave,ax = plt.subplots(2,1,figsize=(4,3),sharex=True,gridspec_kw={'height_ratios': [1, 5]})
            max_exp=np.max([experiment.single_fits[i]['n_exp'] for i in experiment.single_fits.keys()])
            def plotFit():
#                experiment.single_fits[number]={'data':data,'fit':fit_res[0],'res':fit_res[1],'params':params,'detail':detail}
                number=number_spin.value()
                dicto=experiment.single_fits[number]
                wave_label=dicto['detail']
                x=dicto['data'][0]
                y=dicto['data'][1]
                fit=dicto['fit']
                res=dicto['res']
                rango=dicto['rango']
                details=dicto['report']
                ax[1].cla()
                ax[0].cla()
                ax[1].plot(x,y,marker='o',alpha=0.6,ms=4,ls='',label=wave_label)
                ax[1].set_ylim(np.min(y)-abs(np.min(y)*0.1),np.max(y)+np.max(y)*0.1)
                ax[1].set_xlim(x[0]-x[-1]/50,x[-1]+x[-1]/50)
            
                ax[0].plot(x[rango[0]:rango[1]+1],res,marker='o',alpha=0.6,ms=4,ls='')
                ax[1].plot(x[rango[0]:rango[1]+1], fit, 'r-', label='best fit')
                ax[1].legend()
                self.formatAxes(ax[1],timelabel,deltalabel)
                ax[0].axhline(linewidth=1,linestyle='--', color='k')
                ax[0].ticklabel_format(style='sci',axis='y')
                ax[0].set_ylabel('Residues',size=12)
                ax[0].minorticks_on()
                ax[0].axes.tick_params(which='both',direction='in',top=True,right=True,labelsize=12)          
                plt.subplots_adjust(bottom=0.1,left=0.18)
                figwave.canvas.draw()
                report.setText(details)
            number_spin= QSpinBox(self)
            number_spin.setMaximum(len(experiment.single_fits))
            number_spin.setMinimum(1)
            number_spin.setValue(1)
            number_spin.valueChanged.connect(plotFit)
            report=QTextEdit(subWindow_OneTrace_Fit_Res)
            sub_layout=QVBoxLayout(subWindow_OneTrace_Fit_Res)
            report.setMinimumWidth(475)
            report.setMinimumHeight(450)
            sub_layout.addWidget(report)
            plotFit()
            One_trace_widget=Grapth(figwave,subWindow_OneTrace_Fit_Res,toolbar=True,cursor=True,y=True,click=False)
            layout=QHBoxLayout(subWindow_OneTrace_Fit_Res)
            layout.addWidget(One_trace_widget)
            One_trace_widget.setMinimumWidth(600)
            One_trace_widget.setMinimumHeight(650)
            layout.setStretch(0,5)
#            layout.setStretch(1,8)
            h_layout=QHBoxLayout(subWindow_OneTrace_Fit_Res)
            h_layout.addWidget(QLabel('select fit number',self))
            h_layout.addWidget(number_spin)
            h_layout.setAlignment(Qt.AlignCenter)
            sub_layout.addLayout(h_layout)
            btn_close = QPushButton("Close", subWindow_OneTrace_Fit_Res) 
            sub_layout.addWidget(btn_close)
            sub_layout.setContentsMargins(10,10,10,0)
            btn_close.clicked.connect(subWindow_OneTrace_Fit_Res.close)
            layout.addLayout(sub_layout)
            subWindow_OneTrace_Fit_Res.layout().addLayout(layout)
            names =['Nº','Details','t0','fwhm']+['tau_%i' %(i+1) for i in range(max_exp)]
            par_table = QTableWidget(subWindow_OneTrace_Fit_Res)
            par_table.setRowCount(len(experiment.single_fits))
            par_table.setColumnCount(max_exp+4)
            par_table.setColumnWidth(1,300)
            par_table.setColumnWidth(0,40)
            par_table.setHorizontalHeaderLabels(names);
            par_table.verticalHeader().setVisible(False);
            for i in range(len(experiment.single_fits)):
                number=QLabel(str(i+1))
                number.setAlignment(Qt.AlignCenter)
                detail=QLabel(experiment.single_fits[i+1]['detail'])
                detail.setAlignment(Qt.AlignCenter)
                par_table.setCellWidget(i,0,number)
                par_table.setCellWidget(i,1,detail)
                params=experiment.single_fits[i+1]['params']
                for ii,name in enumerate(names[2:]):
                     try:
                         it= QLabel(str(round(params[name].value,2)))
                     except:
                         it= QLabel('-')
                     it.setAlignment(Qt.AlignCenter)
                     par_table.setCellWidget(i,ii+2,it)
            subWindow_OneTrace_Fit_Res.layout().addWidget(par_table)
            subWindow_OneTrace_Fit_Res.layout().setStretch(0,10)
            subWindow_OneTrace_Fit_Res.layout().setStretch(1,2)
            self.mdiArea.addSubWindow(subWindow_OneTrace_Fit_Res)
            subWindow_OneTrace_Fit_Res.show()
        except:
            self.messageError()
    
    def windowFit(self,fit_number):
        try:
            self.fit_subwindows[fit_number].close()
        except:
            pass
        buttonname = f'Fit results %i' %(fit_number)
        params=experiment.all_fit[fit_number][3].params
        exp_no=experiment.all_fit[fit_number][4]
        tau_inf=experiment.all_fit[fit_number][6]
        deconv=experiment.all_fit[fit_number][5]
        type_fit=experiment.all_fit[fit_number][8][2]
        text=['time 0']
        name=['t0_1']
        val=1
        if deconv:
            val=2
            text.append('fwhm')
            name.append('fwhm_1')
        text=text+['tau %i'% (i+1) for i in range(exp_no)]
        name=name+['tau%i_1'% (i+1) for i in range(exp_no)]
        fit_results=QTextEdit()
        fit_results.insertPlainText(f'\t{type_fit} fit\n')
        fit_results.insertPlainText(f'Parameter\tInitial value\tFinal Value\tFixed\n')
        if type_fit == 'Exponential':
            for i in range(len(text)):
                if params[name[i]].vary==False:
                    texto='Yes'
                else:
                    texto='No'
                line=[text[i],'{:.4f}'.format(params[name[i]].init_value),
                      '{:.4f}'.format(params[name[i]].value),texto,'\n']
                fit_results.insertPlainText('\t'.join(line))
        else:
            
            name=name[:val]+['k_%i%i' % (i+1,i+1) for i in range(exp_no)]
            for i in range(len(text)):
                if params[name[i]].vary==False:
                        texto='Yes'
                else:
                    texto='No'
                if i >= val:
                    line=[text[i],'{:.4f}'.format(-1/params[name[i]].init_value),
                          '{:.4f}'.format(-1/params[name[i]].value),texto,'\n']
                else:
                     line=[text[i],'{:.4f}'.format(params[name[i]].init_value),
                      '{:.4f}'.format(params[name[i]].value),texto,'\n']
                fit_results.insertPlainText('\t'.join(line))
        if tau_inf is not None and deconv:
            fit_results.insertPlainText(f'\nDeconvolution, tau infinite included: Yes')
        elif deconv and type_fit == 'Exponential':
            fit_results.insertPlainText(f'\nDeconvolution, tau infinite included: No')
        elif deconv:
            fit_results.insertPlainText(f'\nDeconvolution')
        else:
            fit_results.insertPlainText(f'\nNo deconvolution')
        if experiment.all_fit[fit_number][9]:
            shape=experiment.all_fit[fit_number][10].shape[1]
            fit_results.insertPlainText(f'\nFit done to singular vectors: {shape}')
        else:
            string=experiment.general_report['Fits done'][f'Fit number {fit_number}']
            average=re.findall('average \d+',string)[0].split(' ')[1]
            shape=experiment.all_fit[fit_number][1].shape[1]
            fit_results.insertPlainText(f'\nnumber of Traces fitted: {shape}, average {average}')
        fit_results.insertPlainText(f'\nnumber of iterations: {experiment.all_fit[fit_number][3].nfev}')
        fit_results.insertPlainText(f'\nnumber of parameters optimized: {len(params)}')
        weights=experiment.all_fit[fit_number][8][3]
        if type(weights) is str:
            fit_results.insertPlainText(f'\nWeights: {weights}')
        else:
            fit_results.insertPlainText(f'\nWeights: applied in range {weights["range"][0]}-{weights["range"][1]} {experiment.time_unit},\n\ttype: {weights["type"]}, value {weights["value"]}')
        self.fit_subwindows[fit_number]=QMdiSubWindow(self)
#        text_subWindow = QMdiSubWindow(self)
        self.fit_subwindows[fit_number].resize(350,300)	
        main_layout=QHBoxLayout(self.fit_subwindows[fit_number])	
        sub_layout1=QVBoxLayout(self.fit_subwindows[fit_number])	
        sub_layout2=QVBoxLayout(self.fit_subwindows[fit_number])
        self.formatSubWindow(self.fit_subwindows[fit_number],fit_results,buttonname)
        btn_all_report = QPushButton("Extended fit report", self.fit_subwindows[fit_number]) 
        sub_layout2.addWidget(btn_all_report)
        btn_all_report.clicked.connect(lambda:self.displayCompletellyReport(fit_number)) 
#        btn_test = QPushButton("Verify Fit", self.fit_subwindows[fit_number]) 
        
        if experiment.all_fit[fit_number][9]:
            if fit_number in experiment.conf_interval.keys():
                btn_confidence_interval = QPushButton("F-test conf-intervals results", self.fit_subwindows[fit_number]) 
                btn_confidence_interval.clicked.connect(lambda: self.diplayFtestconf(fit_number))
            else:
                btn_confidence_interval = QPushButton("Calculate confidence intervals", self.fit_subwindows[fit_number]) 
                btn_confidence_interval.clicked.connect(lambda: self.confidenceFTest(fit_number))
            sub_layout2.addWidget(btn_confidence_interval)
            btn_test = QComboBox(self.fit_subwindows[fit_number]) 
            sub_layout1.addWidget(btn_test)
            btn_test.addItem("Verify Traces")
            btn_test.addItem("Verify Spectra")
            btn_test.activated.connect(lambda:btnConfidenceInterval(fit_number))
            def btnConfidenceInterval(fit_number):
                if btn_test.currentText()=="Verify Traces":
                    self.verifiedFit(fit_number)
                elif btn_test.currentText()=="Verify Spectra":
                    self.verifiedSpectra(fit_number)
            fit_select=False
            
        else:    
            combo=False
            if fit_number in experiment.conf_interval.keys() and fit_number in experiment.bootstrap_residues_record.keys():
                btn_confidence_interval = QComboBox(self.fit_subwindows[fit_number])
                btn_confidence_interval.addItem("F-test conf-intervals results")
                btn_confidence_interval.addItem("Bootstrap conf-intervals results")
                combo=True
            elif fit_number in experiment.conf_interval.keys():
                btn_confidence_interval = QComboBox(self.fit_subwindows[fit_number])
                btn_confidence_interval.addItem("F-test conf-intervals results")
                btn_confidence_interval.addItem("Calculate conf-inter by Bootstrap")
                combo=True
            elif fit_number in experiment.bootstrap_residues_record.keys():
                btn_confidence_interval = QComboBox(self.fit_subwindows[fit_number])
                btn_confidence_interval.addItem("Bootstrap conf-intervals results")
                btn_confidence_interval.addItem("Calculate conf-inter by F-test")
                combo=True
            else:
                btn_confidence_interval = QPushButton("Calculate confidence intervals", self.fit_subwindows[fit_number]) 
            sub_layout2.addWidget(btn_confidence_interval)
            if combo:
                btn_confidence_interval.activated.connect(lambda:btnConfidenceInterval(fit_number))
            else:
                btn_confidence_interval.clicked.connect(lambda:self.calculateConfidenceInterval(fit_number)) 
            
            def btnConfidenceInterval(fit_number):
    #            print('enter')
    #            print(btn_confidence_interval.currentText())
                if btn_confidence_interval.currentText()=="F-test conf-intervals results":
                    self.diplayFtestconf(fit_number)
                elif btn_confidence_interval.currentText()=="Calculate conf-inter by F-test":
                    self.confidenceFTest(fit_number)
                elif btn_confidence_interval.currentText()=="Calculate conf-inter by Bootstrap":
                    self.confidenceBoot(fit_number)
                elif btn_confidence_interval.currentText()=="Bootstrap conf-intervals results":
                    self.diplayBootConf(fit_number)
            btn_test = QPushButton("Verify Fit", self.fit_subwindows[fit_number]) 
            sub_layout1.addWidget(btn_test)
            btn_test.clicked.connect(lambda:self.verifiedFit(fit_number)) 
            fit_select=True
                
        if type_fit == 'Exponential':
            btn_hard_model = QPushButton("Export to Tagret-Fit", self.fit_subwindows[fit_number]) 
            sub_layout2.addWidget(btn_hard_model)
            btn_hard_model.clicked.connect(lambda:self.exportToTargetFit(fit_number))
        else:
            btn_hard_model = QPushButton("Plot Fitted model", self.fit_subwindows[fit_number]) 
            sub_layout2.addWidget(btn_hard_model)
            btn_hard_model.clicked.connect(lambda:self.printModel(fit_number))
        btn_fit = QPushButton("Plot Fit", self.fit_subwindows[fit_number]) 
        sub_layout1.addWidget(btn_fit)
        btn_fit.clicked.connect(lambda:self.button71Func(number=fit_number,first_plot=fit_select))
        if type_fit == 'Exponential':
            btn_DAS = QPushButton("Plot DAS", self.fit_subwindows[fit_number]) 
            btn_DAS.clicked.connect(lambda:self.PlotDASFunc(number=fit_number))
        else:
            btn_DAS = QComboBox(self.fit_subwindows[fit_number]) 
            btn_DAS.addItem("Plot EAS")
            btn_DAS.addItem("Plot E-Cons")
            btn_DAS.activated.connect(lambda:self.plotEASFunc(fit_number=fit_number,typo=btn_DAS.currentText()))
        sub_layout1.addWidget(btn_DAS)
        main_layout.addLayout(sub_layout1)
        main_layout.addLayout(sub_layout2)
        self.fit_subwindows[fit_number].layout().addLayout(main_layout)
        btn_close = QPushButton("Close", self.fit_subwindows[fit_number]) 
        self.fit_subwindows[fit_number].layout().addWidget(btn_close)
        btn_close.clicked.connect(self.fit_subwindows[fit_number].close)
   
    def plotEASFunc(self,fit_number,typo):
        '''decide to plot EAS or concentrations'''
        if typo == 'Plot EAS':
            self.PlotDASFunc(number=fit_number)
        else:
            self.plotCons(number=fit_number)
    
    def plotCons(self,number):
        try:
        #fig,ax=self.plotTest()
            names=experiment.target_models[number][2]
            fig,ax=experiment.plotConcentrations(fit_number=number,names=names)
            subWindow = QMdiSubWindow(self)
            widget=Grapth(fig,subWindow,cursor=False)
            self.formatSubWindow(subWindow,widget,'Species Concentration',close_button=True)
        except:
            self.messageError()
    
    def exportToTargetFit(self,fit_number):
        pass
    
    def calculateConfidenceInterval(self,fit_number):
        if self.long_calcul_runing:
            dialog = QMessageBox(self)
            dialog.setWindowTitle('Confidence Interval calculation')
            dialog.setText('There are already some calculation of confidence interval running stopped by closing the window to start another')
            dialog.exec() 
        else:
            dialog = QMessageBox(self)
            dialog.setWindowTitle('Confidence Interval calculation')
            dialog.setText('\t\t\tPlease only run this if you are sure the fit results!!!\n\nThese methods are based on redoing the runned fit several times with different condition so they need much more time than the fit.\n\nTo calculate the Confidence interval, we propose two methods F-test and bootstrap\n')
            cancel=QPushButton("Cancel", dialog)
            accept=QPushButton("Calculate", dialog)
            dialog.addButton(cancel,QMessageBox.AcceptRole)
            dialog.addButton(accept,QMessageBox.RejectRole)
            dialog.setDefaultButton(accept)
            layout=QVBoxLayout()
            F_test=QRadioButton('F-test --> Compares the best fit with an alternate model, where one of the parameters is fixed for each paramter')
            boot=QRadioButton('Bootstrap --> Re-do the fit n-times on data sets recomposed with the best fit and random residue from the fit')
            layout.addWidget(F_test)
            layout.addWidget(boot)
            group=QButtonGroup(dialog)
            group.addButton(F_test)
            group.addButton(boot)
            main_layout=dialog.layout()
            main_layout.addLayout(layout,1,1,alignment=Qt.AlignLeft)
            dialog.exec()       
            if dialog.clickedButton()== accept:
                if F_test.isChecked():
                    self.confidenceFTest(fit_number)
                elif boot.isChecked():
                    self.confidenceBoot(fit_number)
                else:
                   dialog2 = QMessageBox(self)
                   dialog2.setWindowTitle('Confidence Interval calculation')
                   dialog2.setText('Please SELECT One of the available methods, or click cancel') 
                   dialog2.exec() 
                   self.calculateConfidenceInterval(fit_number)    
            else:
                dialog.close()
                
    def diplayBootConf(self,fit_number):
        subWindow = QMdiSubWindow(self)
        bootsTrap=experiment.bootstrap_residues_record[fit_number][1]
        stats=bootsTrap.describe()
        fig_hist, axes = plt.subplots(1, 1)
        widget=Grapth(fig_hist,subWindow)
        widget.resize(500,500)
        names=[i.split(' ')[0] for i in bootsTrap.keys() if 'final' in i]
        stats_values={}
        table=experiment.bootstrap_residues_record[fit_number][2]
        for name in names:    
            stats_values[name+' mean']=round(stats[name+' final']['mean'],4)
            stats_values[name+' std']=round(stats[name+' final']['std'],4)
            stats_values[name+' min']=round(stats[name+' final']['min'],4)
            stats_values[name+' max']=round(stats[name+' final']['max'],4)
        def plot(groupsH,axes,kde):
            param=[names[i] for i,ii in enumerate(groupsH.buttons()) if ii.isChecked()][0]
#            global axes
#            global tick_kde
            if tick_kde.isChecked():
                kde=True
            else:
                kde=False
            axes.cla()
            x_label=f'{experiment.time_unit}' if 'k' not in param else f'1/{experiment.time_unit}'
            distplot(bootsTrap[param+' final'].values,rug=False,norm_hist =False,kde=kde,hist_kws=dict(edgecolor="k", linewidth=2),ax=axes)
            axes.set_xlabel(f'Time ({x_label})')
            maxi=bootsTrap[param+' final'].max()#to select the position of the anchor text
            mini=bootsTrap[param+' final'].min()
            mean=bootsTrap[param+' final'].mean()#[0]
            dif_max=abs(maxi-mean)
            dif_min=abs(mini-mean)
            if kde==False:
                axes.set_ylabel('Counts')
                axes.set_xlim(mini-abs(maxi*0.1),maxi+abs(maxi*0.1))
            else:
                axes.set_ylabel('Density function')
            if dif_max>dif_min:
                pos=1
            else:
                pos=2
            mean=stats_values[param+' mean']
            std=stats_values[param+' std']
            texto=AnchoredText(s=f'$\mu={mean}$ {experiment.time_unit}\n $\sigma={std}$ {experiment.time_unit}',loc=pos)
            axes.add_artist(texto)
            fig_hist.canvas.draw()
        v_layout=QVBoxLayout(subWindow)
        v_layout.setAlignment(Qt.AlignTop)
        v_layout.setContentsMargins(20,120,20,10)
        groupsH=QButtonGroup(self)
        number=QLabel(f'N° Data sets: {bootsTrap.shape[0]}',subWindow)
        v_layout.addWidget(number)
        ticks={}
        label=QLabel(f'Parameters',subWindow)
        v_layout.addWidget(label)
        for ii,name in enumerate(names):
            ticks[name]=QCheckBox(name,subWindow)
            groupsH.addButton(ticks[name])
            v_layout.addWidget(ticks[name])
        label_kde=QLabel(f'\n\tPlot \nKernel density estimation',subWindow)
        v_layout.addWidget(label_kde)
        tick_kde=QCheckBox('KDE',subWindow)
        tick_kde.setCheckState(2)
        v_layout.addWidget(tick_kde)
        ticks[names[0]].setCheckState(2)
        report=QTextEdit(subWindow)
        report.setReadOnly(True)
        texto=['\t\tStatistical description:',f'\t\tparameter\tmaximum\tminimum\tmean\tstandar deviation']
        for name in names:
            line=['\t',name]+['{:.4f}'.format(stats_values[name+' max']),'{:.4f}'.format(stats_values[name+' min']),'{:.4f}'.format(stats_values[name+' mean']), '{:.4f}'.format(stats_values[name+' std'])]
            texto.append('\t'.join(line))
        texto.append('\n\tConfidence interval report:')
        texto.append(f'parameter\t99.73%\t95.45%\t68.27%\t_BEST_\t68.27%\t95.45%\t99.73%')
        for name in names:
            line=[name]+['{:.4f}'.format(i) for i in table.loc[name]]
            texto.append('\t'.join(line))
        text='\n'.join(texto)
        report.setText(text) 
        size_r=15*len(texto)+15
        report.setMinimumHeight(size_r)
        report.setMaximumHeight(size_r+15)
        plot(groupsH,axes,tick_kde)
        replot=QPushButton('Plot histogram', subWindow)
        replot.clicked.connect(lambda: plot(groupsH,axes,tick_kde))
        v_layout.addWidget(replot)
        corr=QPushButton('Check correlations', subWindow)
        corr.clicked.connect(lambda: self.corrPlot(fit_number))
        v_layout.addWidget(corr)
        btn_close = QPushButton("Close", subWindow) 
        v_layout.addWidget(btn_close)
        btn_close.clicked.connect(subWindow.close)
        v_layout.addWidget(QLabel(''))
        btn_more = QPushButton("Do more bootstraps", subWindow) 
        v_layout.addWidget(btn_more,alignment=Qt.AlignBottom)
        btn_more.clicked.connect(lambda:self.confidenceBoot(fit_number,True))
        main_layout=QHBoxLayout(subWindow)
        main_layout.addWidget(widget)
        main_layout.addLayout(v_layout)
        subWindow.layout().addLayout(main_layout)
        subWindow.layout().addWidget(report)
        subWindow.setWindowTitle(f"Confident intervals BootStrap for Fit {fit_number}")
        self.mdiArea.addSubWindow(subWindow)
        subWindow.resize(750,750)
        subWindow.show()
    
    def corrPlot(self,fit_number):
        try:
            subWindow = QMdiSubWindow(self)
            datos=experiment.bootstrap_residues_record[fit_number][1]
            fig2,ax=plt.subplots(2,2,figsize=(8,8),gridspec_kw={'height_ratios': [2, 5],'width_ratios': [5, 2]})
            if len(datos) > 250:
                alpha=0.5
            elif len(datos) > 500:
                alpha=0.25
            else:
                alpha=1
            ax[0,1].axis('off')
            ax[0,0].set_xticklabels([])
            ax[1,1].set_yticklabels([])
            fig2.subplots_adjust(wspace=0.1, hspace=0.1)
            widget=Grapth(fig2,subWindow)
            subWindow.resize(1000,750)
            params_all=[i for i in datos.columns if 'final' in i]
            names=[i.split(' ')[0] if 'k' not in i else i for i in params_all]
            def plot(groups):
                if tick_kde.isChecked():
                    kde=True
                    label='Density'
                else:
                    kde=False   
                    label='Counts'
                first=[params_all[i] for i,ii in enumerate(groups[0].buttons()) if ii.isChecked()][0]
                second=[params_all[i] for i,ii in enumerate(groups[1].buttons()) if ii.isChecked()][0]
                first_label=f'{experiment.time_unit}' if 'k' not in first else f'1/{experiment.time_unit}'
                second_label=f'{experiment.time_unit}' if 'k' not in second else f'1/{experiment.time_unit}'
                ax[1,0].cla()
                ax[0,0].cla()
                ax[1,1].cla()
                ax[0,0].get_shared_x_axes().join(ax[0,0], ax[1,0])
                ax[1,1].get_shared_y_axes().join(ax[1,1], ax[1,0])
                distplot(datos[first].values,rug=False,norm_hist =False,kde=kde,hist_kws=dict(edgecolor="k", linewidth=2),ax=ax[0,0])
                distplot(datos[second].values,rug=False,norm_hist =False,kde=kde,hist_kws=dict(edgecolor="k", linewidth=2),ax=ax[1,1],vertical=True)
                kdeplot(datos[first].values,datos[second].values, ax=ax[1,0],cmap='Spectral_r',shade=True)
                ax[0,0].set_ylabel(label)
                ax[1,1].set_xlabel(label)
                ax[1,0].scatter(datos[first],datos[second],color='r',marker='+', alpha=alpha)
                ax[0,0].set_xticklabels([])
                ax[1,1].set_yticklabels([])
                ax[1,0].set_xlabel(first.split(' ')[0] + f' ({first_label})')
                ax[1,0].set_ylabel(second.split(' ')[0] + f' ({second_label})')
                fig2.canvas.draw()
            v_layout=QVBoxLayout(subWindow)
            v_layout.setAlignment(Qt.AlignTop)
            v_layout.setContentsMargins(20,150,20,10)
            groups=[QButtonGroup(self),QButtonGroup(self)]
            ticks=[{},{}]
            for i in range(2):
                label=QLabel(f'Parameter {i}')
                v_layout.addWidget(label)
                for ii,name in enumerate(names):
                    ticks[i][name]=QCheckBox(name,self)
                    groups[i].addButton(ticks[i][name])
                    v_layout.addWidget(ticks[i][name])
            ticks[0][names[0]].setCheckState(2) 
            ticks[1][names[1]].setCheckState(2)
            replot=QPushButton('Plot', subWindow)
            replot.clicked.connect(lambda: plot(groups))
            v_layout.addWidget(replot)
            tick_kde=QCheckBox('KDE',subWindow)
            tick_kde.setCheckState(2)
            v_layout.addWidget(tick_kde)
            plot(groups)
            btn_close = QPushButton("Close", subWindow) 
            v_layout.addWidget(btn_close)
            btn_close.clicked.connect(subWindow.close)
            main_layout=QHBoxLayout(subWindow)
            main_layout.addWidget(widget)
            main_layout.addLayout(v_layout)
            subWindow.layout().addLayout(main_layout)
            subWindow.setWindowTitle(f"Parameters correlations for Fit {fit_number}")
            self.mdiArea.addSubWindow(subWindow)
            subWindow.show()
        except:
            self.messageError()
        
    
    def confidenceBoot(self,fit_number,already=False):
        if self.long_calcul_runing==False:
            self.last_calculate_fit_number=fit_number
            self.long_calcul_runing=True
            mainwindow2 = MainWindowBoot(1,already,self)
            mainwindow2.closed.connect(self.bootstrapFinished)
            mainwindow2.show()
        else:
            dialog = QMessageBox(self)
            dialog.setWindowTitle('Confidence Interval calculation')
            dialog.setText('There are already some calculation of confidence interval running stopped by closing the window, or wait until is finished before starting another')
            dialog.exec() 
    
    def bootstrapFinished(self,val):
        self.long_calcul_runing=False
        if val==-1:
            print('False')
        else:
            print('True')
            dialog = QMessageBox(self)
            dialog.setWindowTitle('Confidence Interval calculation Finished')
            dialog.setText('The confidence interval calculation BootStrap has finished.\nYou have acces to the result in the fit window.\n\nDo you want to diplay the reults now?')
            cancel=QPushButton("Cancel", dialog)
            accept=QPushButton("Display results", dialog)
            dialog.addButton(cancel,QMessageBox.AcceptRole)
            dialog.addButton(accept,QMessageBox.RejectRole)
            dialog.setDefaultButton(accept)
            dialog.exec()       
            if dialog.clickedButton()==accept:
               try:
                   self.fit_subwindows[self.last_calculate_fit_number].close()
                   self.windowFit(self.last_calculate_fit_number)
               except:
                   pass
               self.diplayBootConf(self.last_calculate_fit_number)
            else:
                try:
                    self.fit_subwindows[self.last_calculate_fit_number].close()
                    self.windowFit(self.last_calculate_fit_number)
                except:
                    pass
                dialog.Close()
    
    def confidenceFTest(self,fit_number):
        if self.long_calcul_runing==False:
            self.last_calculate_fit_number=fit_number
            self.long_calcul_runing=True
            mainwindow2 = MainWindowFtest(fit_number,self)
            mainwindow2.closed.connect(self.FtestFinished)
            mainwindow2.show()
        else:
            dialog = QMessageBox(self)
            dialog.setWindowTitle('Confidence Interval calculation')
            dialog.setText('There are already some calculation of confidence interval running stopped by closing the window, or wait until is finished before starting another')
            dialog.exec() 
    
    def FtestFinished(self,val):
        self.long_calcul_runing=False
        if val==-1:
            print('False')
            pass
        else:
            print('True')
            dialog = QMessageBox(self)
            dialog.setWindowTitle('Confidence Interval calculation Finished')
            dialog.setText('The confidence interval calculation using F-test has finished.\nYou have acces to the result in the fit window.\n\nDo you want to diplay the reults now?')
            cancel=QPushButton("Cancel", dialog)
            accept=QPushButton("Display results", dialog)
            dialog.addButton(cancel,QMessageBox.AcceptRole)
            dialog.addButton(accept,QMessageBox.RejectRole)
            dialog.setDefaultButton(accept)
            dialog.exec()       
            if dialog.clickedButton()==accept:
               try:
                   self.fit_subwindows[self.last_calculate_fit_number].close()
                   self.windowFit(self.last_calculate_fit_number)
               except:
                   pass
               self.diplayFtestconf(self.last_calculate_fit_number)
            else:
                try:
                    self.fit_subwindows[self.last_calculate_fit_number].close()
                    self.windowFit(self.last_calculate_fit_number)
                except:
                    pass
                dialog.Close()
    
    def diplayFtestconf(self,fit_number):
        try:
            subWindow = QMdiSubWindow(self)
            figF, axesF = plt.subplots(1, 2)       
            plt.tight_layout()
            figF.subplots_adjust(bottom=0.2)
            cnorm = Normalize(vmin=0,vmax=1)
            cpickmap = plt.cm.ScalarMappable(norm=cnorm,cmap='viridis_r')
            cpickmap.set_array([])
            plt.colorbar(cpickmap,ax=axesF).set_label(label='Probability',size=15)
            widget=Grapth(figF,subWindow)
            subWindow.resize(1000,500)
            if self.mdiArea.size().width()<1220:
                width=self.mdiArea.size().width()
            else:
                width=1220
            if self.mdiArea.size().height()<650:
                height=self.mdiArea.size().height()
            else:
                height=650
            subWindow.resize(width,height)
            cof_int=experiment.conf_interval[fit_number]
            params_all=[i for i in cof_int[1].keys()]
            names=[i.split('_')[0] if 'k' not in i else i for i in params_all]
            def plot(groups):
                first=[params_all[i] for i,ii in enumerate(groups[0].buttons()) if ii.isChecked()][0]
                second=[params_all[i] for i,ii in enumerate(groups[1].buttons()) if ii.isChecked()][0]
                axesF[0].cla()
                axesF[1].cla()
                name_first=first.split('_')[0] + f' ({experiment.time_unit})' if 'k' not in first else first + f' (1/{experiment.time_unit})'
                name_second=second.split('_')[0] + f' ({experiment.time_unit})' if 'k' not in second else second + f' (1/{experiment.time_unit})'
                cx1, cy1, prob = cof_int[1][first][first], cof_int[1][first][second], cof_int[1][first]['prob']
                cx2, cy2, prob2 = cof_int[1][second][second], cof_int[1][second][first], cof_int[1][second]['prob']
                axesF[0].scatter(cx1, cy1, c=prob, s=30)
                axesF[0].set_xlabel(name_first)
                axesF[0].set_ylabel(name_second)                
                axesF[1].scatter(cx2, cy2, c=prob2, s=30)
                axesF[1].set_xlabel(name_second)
                axesF[1].set_ylabel(name_first)
                figF.canvas.draw()
            v_layout=QVBoxLayout(subWindow)
            v_layout.setAlignment(Qt.AlignTop)
            v_layout.setContentsMargins(20,150,20,10)
            groups=[QButtonGroup(self),QButtonGroup(self)]
            ticks=[{},{}]
            for i in range(2):
                label=QLabel(f'Parameter {i}')
                v_layout.addWidget(label)
                for ii,name in enumerate(names):
                    ticks[i][name]=QCheckBox(name,self)
                    groups[i].addButton(ticks[i][name])
                    v_layout.addWidget(ticks[i][name])
            ticks[0][names[0]].setCheckState(2) 
            ticks[1][names[1]].setCheckState(2)
            report=QTextEdit(subWindow)
            report.setReadOnly(True)
            texto=['\t\tConfidence interval report:\n',f'\t\tparameter\t99.73%\t95.45%\t68.27%\t_BEST_\t68.27%\t95.45%\t99.73%']
            for param in cof_int[0].keys():
                index=len(cof_int[0][param])//2
                line=['\t',param]+['{:.4f}'.format(i[1]-cof_int[0][param][index][1]) if ii!=index  else '{:.4f}'.format(i[1]) for ii,i in enumerate(cof_int[0][param])]
                texto.append('\t'.join(line))
            text='\n'.join(texto)
            report.setText(text) 
            size_r=15*len(texto)+15
    #        if size_r>120:
    #            size_r=120
            report.setMinimumHeight(size_r)
            report.setMaximumHeight(size_r+15)
    #        report.setText(ci_report(cof_int[0]))
            plot(groups)
            replot=QPushButton('Plot probabilities', subWindow)
            replot.clicked.connect(lambda: plot(groups))
            v_layout.addWidget(replot)
            btn_close = QPushButton("Close", subWindow) 
            v_layout.addWidget(btn_close)
            btn_close.clicked.connect(subWindow.close)
            main_layout=QHBoxLayout(subWindow)
            main_layout.addWidget(widget)
            main_layout.addLayout(v_layout)
            final_layout=QVBoxLayout(subWindow)
            final_layout.addLayout(main_layout)
            final_layout.addWidget(report)
            subWindow.layout().addLayout(final_layout)
            subWindow.setWindowTitle(f"Confident intervals by F-Test for Fit {fit_number}")
            self.mdiArea.addSubWindow(subWindow)
            subWindow.show()
        except:
            self.messageError() 
    
    def displayCompletellyReport(self,fit_number):
        try:
            subwindow = QMdiSubWindow(self)
            subwindow.resize(450,300)
            params=deepcopy(experiment.all_fit[fit_number][3])
            contents=self.formatSubWindow(subwindow,QTextEdit(),'Extended fit report',close_button=True)
            text=fit_report(params)
            if 	experiment.all_fit[fit_number][8][2] == 'Target':
                text = [i for i in text.split('\n') if 'tau' not in i]
                text = ('\n').join(text)
            contents.setText(text)
        except:
            self.messageError()
    
    def xLogScales(self,ax,fig,subWindow):
        if ax[0] is '_':
           ax = [ax[1]]
        mini,maxi=ax[0].get_xlim()
        maxi_log=100
        if maxi>maxi_log:
           maxi_log=maxi*1.5 
        sublayout2=QVBoxLayout()
        sublayout2.setAlignment(Qt.AlignTop)
        sublayout2.setContentsMargins(0,70,10,10)
        sublayout2.addWidget(QLabel('Time scale'))
        linear=QCheckBox('linear',subWindow)
        linear.setCheckState(2)
        log=QCheckBox('logarithmic',subWindow)
        combined=QCheckBox('combined',subWindow)
        sublayout2.addWidget(linear)
        sublayout2.addWidget(log)
        sublayout2.addWidget(combined)
        def setScale():
            try:
                if linear.isChecked():
                    for i in ax:
                        i.set_xscale('linear')
                        i.set_xlim(mini,maxi)
                elif log.isChecked():
                    for i in ax:
                        i.set_xscale('log')
                        i.set_xlim(0.02,maxi_log)
                elif combined.isChecked():
                    try:
                        val=abs(float(text.text()))
                    except:
                        val=2
                    for i in ax:
                        i.set_xscale('symlog',linthreshx=val,subsx=[2,4,6,8])
                        i.set_xlim((mini+maxi/50)*2,maxi*1.5)
                fig.canvas.draw() 
            except:
                by_dialog = QMessageBox(self)
                by_dialog.setText("New scale coudl not be set")
                by_dialog.exec()
        replot=QPushButton('set scale', subWindow)
        replot.clicked.connect(setScale)
        label=QLabel('linear limit',subWindow)
        text=QLineEdit('2', self) 
        label.setVisible(False)
        text.setVisible(False)
        sublayout2.addWidget(label)
        sublayout2.addWidget(text)
        sublayout2.addWidget(replot)
        def showText():
            if combined.isChecked():
                label.show()
                text.show()
            else:
                label.setVisible(False)
                text.setVisible(False)
        combined.stateChanged.connect(showText)
        group1=QButtonGroup(self)
        group1.addButton(linear)
        group1.addButton(log)
        group1.addButton(combined)
        return sublayout2
    #button after go to fit plot Fit
    def button71Func(self,number=None,selection=None,residues=True,first_plot=True):
        try:
            fig,ax=experiment.plot_fit(fit_number=number, selection=selection, plot_residues=residues)
            number_name=f'(fit {number})' if number is not None  else 'last fit'
            w,h=fig.get_size_inches()*fig.dpi
            subWindow = QMdiSubWindow(ex)
            widget=Grapth(fig,subWindow)
            log_layout=ex.xLogScales(ax,fig,subWindow)
            subWindow.setWindowTitle(f"Fitted plot {number_name}")
            layout=QHBoxLayout()
            layout.addWidget(widget)
            layout.addLayout(log_layout)
            layout.setStretch(10,0)
            subWindow.layout().addLayout(layout)
            if first_plot:
                h += 30
                replot=QPushButton('Select traces and fits', subWindow)
                replot.clicked.connect(lambda: ex.replotSelectFit(number))
                subWindow.layout().addWidget(replot)
            ex.mdiArea.addSubWindow(subWindow)
            subWindow.resize(w+50,h+100)
            subWindow.show()
        except:
            self.messageError()

    def replotSelectFit(self,number):
        try:
            wavelenghts=experiment.all_fit[number][2]
            fit_subWindow = QMdiSubWindow(self)
            fit_subWindow.setObjectName('Fit selection')
            fit_subWindow.setWindowTitle('Fit selection')
            #♣self.fit_subWindow.resize(150,200)
            #self.fit_subWindow.layout(QVBoxLayout())
            main_layout=QHBoxLayout()
            group=QButtonGroup(fit_subWindow)
            group.setExclusive(False)
            fit_subWindow.layout().addLayout(main_layout)
            widget_dict={}
            label = QLabel('---------------------',fit_subWindow)
            layout=QVBoxLayout()
            layout.addWidget(label)
            layout.setAlignment(Qt.AlignTop)
            for ii,i in enumerate(wavelenghts):
                widget_dict[ii]=QRadioButton(f"{round(i,1)} {experiment.wavelength_unit}",fit_subWindow)
                widget_dict[ii].setObjectName(f"fit number: %i" %(i+1))
                group.addButton(widget_dict[ii])
                layout.addWidget(widget_dict[ii])
#                label = QLabel('---------------------', fit_subWindow)
#                layout.addWidget(label)
                if ii%10==9 and ii>1:
                    label3 = QLabel('---------------------', fit_subWindow)
                    layout.addWidget(label3)
                    label2 = QLabel('---------------------', fit_subWindow)
                    main_layout.addLayout(layout)
                    layout=QVBoxLayout()
                    layout.addWidget(label2)
                    layout.setAlignment(Qt.AlignTop)
            main_layout.addLayout(layout)
            main_layout.setContentsMargins(10,0,10,0)
            plot_residues=QCheckBox('Plot residues',fit_subWindow)
            plot_residues.setCheckState(2) 
            fit_subWindow.layout().addWidget(plot_residues,alignment=Qt.AlignCenter)
            self.mdiArea.addSubWindow(fit_subWindow)
            btn_plot = QPushButton("Plot selected traces", fit_subWindow) 
            fit_subWindow.layout().addWidget(btn_plot)
            btn_plot.clicked.connect(lambda: plotSelection())
            btn_close = QPushButton("Close", fit_subWindow) 
            fit_subWindow.layout().addWidget(btn_close)
            btn_close.clicked.connect(fit_subWindow.close)
            fit_subWindow.show()
            def plotSelection():
                print('ok')
                buton_selection=[i for i,ii in enumerate(widget_dict) if widget_dict[ii].isChecked()]
                selection=wavelenghts[buton_selection]
                if plot_residues.checkState()==2:
                    residues=True
                else:
                    residues=False
                self.button71Func(number,selection=selection,residues=residues,first_plot=False)
                fit_subWindow.close()
            self.repaint()
        except:
            self.messageError()
        
    def button02Func(self):
        try:
            window=int(self.text12.text())
            poly=int(self.text22.text())
            deriv=int(self.text32.text())
            experiment.derivateSpace(window,poly,deriv)
            self.repaint() 
            self.printMessage('Data derivated')
            self.repaint()
        except:
            self.messageError()
    
    def verifiedSpectra(self,fit_number):#only after fitting SVD vectors.
        try:
               self.subWindow_fit.close()
        except:
            pass
        if fit_number is not None:
            self.x_fit=experiment.all_fit[fit_number][0]
            self.data_fit=experiment.all_fit[fit_number][1]
            self.wavelength_fit=experiment.all_fit[fit_number][2]
            if experiment.all_fit[fit_number][9]:
                params=experiment.all_fit[fit_number][11]
            else:
                params=experiment.all_fit[fit_number][3].params
            deconv=experiment.all_fit[fit_number][5]
        else:
            self.data_fit=experiment.data_before_last_Fit
            self.x_fit=experiment.x_before_last_Fit
            self.wavelength_fit=experiment.wavelength_before_last_Fit
            params=experiment.params
            deconv=experiment.deconv
        print(fit_number)
        xlabel='Time ('+experiment.time_unit+')'
        self.fig, ax = plt.subplots(2, 1,sharex=True, figsize=(10,8), gridspec_kw={'height_ratios': [1, 5]})
        self.fittes=experiment.results(params,fit_number=fit_number,verify_SVD_fit=True)
        initial_i=experiment.data.shape[0]//5
        if deconv:
            self.ls,=ax[1].plot(self.wavelength_fit,self.data_fit[initial_i,:],marker='o',ms=3,linestyle=None,label='raw data')
            self.residues= 0.0*self.data_fit[:]
            for i in range(self.fittes.shape[0]):
                self.residues[i,:]=self.data_fit[i,:]-self.fittes[i,:]
            self.llls,=ax[0].plot(self.wavelength_fit,self.residues[initial_i,:],marker='o',ms=3,linestyle=None,label='residues')
            self.lls,=ax[1].plot(self.wavelength_fit,self.fittes[initial_i,:],alpha=0.5,lw=1.5,color='r',label='fit')    
        else:
            t0 = params['t0_1'].value
            index=np.argmin([abs(i-t0) for i in self.x_fit])
            self.data_fit=self.data_fit[index:]
            self.ls,=ax[1].plot(self.wavelength_fit,self.data_fit[initial_i,:],marker='o',ms=3,linestyle=None,label='raw data')
            self.residues= 0.0*self.data_fit[:]
            for i in range(index,self.fittes.shape[0]):
                self.residues[i,:]=self.data_fit[i,:]-self.fittes[i,:]
            self.llls,=ax[0].plot(self.wavelength_fit,self.residues[initial_i,:],marker='o',ms=3,linestyle=None,label='residues')
            self.lls,=ax[1].plot(self.wavelength_fit,self.fittes[initial_i,:],alpha=0.5,lw=1.5,color='r',label='fit')    
        maxi,_=self.data_fit.shape
        ax[0].ticklabel_format(style='sci',axis='y')
        ax[1].ticklabel_format(style='sci',axis='y')
        ax[1].minorticks_on()
        ax[1].axes.tick_params(which='both',direction='in',top=True,right=True,labelsize=14)
        ax[0].minorticks_on()
        ax[0].axes.tick_params(which='both',direction='in',top=True,right=True,labelsize=14)
        ax[0].set_ylim(np.min(self.residues)-abs(np.min(self.residues)*0.1),np.max(self.residues)+np.max(self.residues)*0.1)
        ax[1].set_ylim(np.min(self.data_fit)-abs(np.min(self.data_fit)*0.1),np.max(self.data_fit)+np.max(self.data_fit)*0.1)
        #f.tight_layout()
        ax[1].legend(loc='upper right')
        ax[0].legend(loc='upper right')
        title=round(self.x_fit[initial_i],2)
        plt.title(f'{title} {experiment.time_unit}')
        ax[1].set_xlabel(xlabel, size=14)
        plt.xlim(self.wavelength_fit[0]-self.wavelength_fit[-1]/50,self.wavelength_fit[-1]+self.wavelength_fit[-1]/50)
        ax[1].set_ylabel(r'$\Delta$A',size=14)
        ax[0].set_ylabel('Residues',size=14)
        self.subWindow_fit = QMdiSubWindow(self)
        widget=Grapth(self.fig,self.subWindow_fit,cursor=True,y=True,click=False)
        self.formatSubWindow(self.subWindow_fit,widget,'Verify Fit')
        self.silider_2 = QSlider(Qt.Horizontal,self.subWindow_fit)
        self.silider_2.setFixedSize(QSize(850, 15))
        self.silider_2.setStyleSheet('background-color:darkRed;border: 2px solid #999999;height: 15px; border-radius: 5px')
        self.silider_2.setTickPosition(self.silider_2.TicksBelow)
        self.silider_2.setValue(initial_i)
        if deconv:
            self.silider_2.setMinimum(0)
        else:
            self.silider_2.setMinimum(index)
        self.silider_2.setMaximum(maxi-1)
        self.silider_2.valueChanged.connect(self.silider2Func)
        self.subWindow_fit.layout().addWidget(self.silider_2,alignment=Qt.AlignCenter)
        btn_close = QPushButton("Close", self.subWindow_fit) 
        self.subWindow_fit.layout().addWidget(btn_close)
        btn_close.clicked.connect(self.subWindow_fit.close)

    def silider2Func(self):
        value = self.silider_2.value()
        value=int(round(value))
        # update curve
        title=round(self.x_fit[value],2)
        plt.title(f'{title} {experiment.time_unit}')
        self.ls.set_ydata(self.data_fit[value,:])
        self.lls.set_ydata(self.fittes[value,:])
        self.llls.set_ydata(self.residues[value,:])
        # redraw canvas while idle
        self.fig.canvas.draw_idle()
    
    def verifiedFit(self, fit_number=None):
#        try:
            try:
               self.subWindow_fit.close()
            except:
                pass
            if fit_number is not None:
                self.x_fit=experiment.all_fit[fit_number][0]
                self.data_fit=experiment.all_fit[fit_number][1]
                self.wavelength_fit=experiment.all_fit[fit_number][2]
                if experiment.all_fit[fit_number][9]:
                    params=experiment.all_fit[fit_number][11]
                else:
                    params=experiment.all_fit[fit_number][3].params
                deconv=experiment.all_fit[fit_number][5]
            else:
                self.data_fit=experiment.data_before_last_Fit
                self.x_fit=experiment.x_before_last_Fit
                self.wavelength_fit=experiment.wavelength_before_last_Fit
                params=experiment.params
                deconv=experiment.deconv
            print(fit_number)
            xlabel='Time ('+experiment.time_unit+')'
            self.fig, ax = plt.subplots(2, 1,sharex=True, figsize=(10,8), gridspec_kw={'height_ratios': [1, 5]})
            self.fittes=experiment.results(params,fit_number=fit_number,verify_SVD_fit=True)
            initial_i=self.data_fit.data.shape[1]//5
            self.l,=ax[1].plot(self.x_fit,self.data_fit[:,initial_i],marker='o',ms=3,linestyle=None,label='raw data')
            if deconv:
                self.residues= 0.0*self.data_fit[:]
                for i in range(self.fittes.shape[1]):
                    self.residues[:,i]=self.data_fit[:,i]-self.fittes[:,i]
                self.lll,=ax[0].plot(self.x_fit,self.residues[:,initial_i],marker='o',ms=3,linestyle=None,label='residues')
                self.ll,=ax[1].plot(self.x_fit,self.fittes[:,initial_i],alpha=0.5,lw=1.5,color='r',label='fit')    
            else:
                 t0 = params['t0_1'].value
                 index=np.argmin([abs(i-t0) for i in self.x_fit])
                 self.residues= 0.0*self.data_fit[index:,:]
                 for i in range(self.fittes.shape[1]):
                    self.residues[:,i]=self.data_fit[index:,i]-self.fittes[:,i]
                 self.lll,=ax[0].plot(self.x_fit[index:],self.residues[:,initial_i],marker='o',ms=3,linestyle=None,label='residues')
                 self.ll,=ax[1].plot(self.x_fit[index:],self.fittes[:,initial_i],alpha=0.5,lw=1.5,color='r',label='fit')
            _,maxi=self.data_fit.shape
            ax[0].ticklabel_format(style='sci',axis='y')
            ax[1].ticklabel_format(style='sci',axis='y')
            ax[1].minorticks_on()
            ax[1].axes.tick_params(which='both',direction='in',top=True,right=True,labelsize=14)
            ax[0].minorticks_on()
            ax[0].axes.tick_params(which='both',direction='in',top=True,right=True,labelsize=14)
            ax[0].set_ylim(np.min(self.residues)-abs(np.min(self.residues)*0.1),np.max(self.residues)+np.max(self.residues)*0.1)
            ax[1].set_ylim(np.min(self.data_fit)-abs(np.min(self.data_fit)*0.1),np.max(self.data_fit)+np.max(self.data_fit)*0.1)
            #f.tight_layout()
            ax[1].legend(loc='upper right')
            ax[0].legend(loc='upper right')
            title=round(self.wavelength_fit[initial_i])
            plt.title(f'{title} {experiment.wavelength_unit}')
            ax[1].set_xlabel(xlabel, size=14)
            plt.xlim(self.x_fit[0]-self.x_fit[-1]/50,self.x_fit[-1]+self.x_fit[-1]/50)
            ax[1].set_ylabel(r'$\Delta$A',size=14)
            ax[0].set_ylabel('Residues',size=14)
            self.subWindow_fit = QMdiSubWindow(self)
            widget=Grapth(self.fig,self.subWindow_fit,cursor=True,y=True,click=False)
            self.formatSubWindow(self.subWindow_fit,widget,'Verify Fit')
            self.silider_1 = QSlider(Qt.Horizontal,self.subWindow_fit)
            self.silider_1.setFixedSize(QSize(850, 15))
            self.silider_1.setStyleSheet('background-color:darkRed;border: 2px solid #999999;height: 15px; border-radius: 5px')
            self.silider_1.setTickPosition(self.silider_1.TicksBelow)
            self.silider_1.setValue(initial_i)
            self.silider_1.setMinimum(0)
            self.silider_1.setMaximum(maxi-1)
            self.silider_1.valueChanged.connect(self.silider1Func)
            self.subWindow_fit.layout().addWidget(self.silider_1,alignment=Qt.AlignCenter)
            btn_close = QPushButton("Close", self.subWindow_fit) 
            self.subWindow_fit.layout().addWidget(btn_close)
            btn_close.clicked.connect(self.subWindow_fit.close)
#        except:
#            self.messageError()
        
    def silider1Func(self):
        value = self.silider_1.value()
        value=int(round(value))
        # update curve
        title=round(self.wavelength_fit[value])
        plt.title(f'{title} {experiment.wavelength_unit}')
        self.l.set_ydata(self.data_fit[:,value])
        self.ll.set_ydata(self.fittes[:,value])
        self.lll.set_ydata(self.residues[:,value])
        # redraw canvas while idle
        self.fig.canvas.draw_idle()
        
    #this is essential function, which will be called by window every time something changes. so here you put all the stuff which describes visual
    #behaviour of you window, expecially thing that can be changed, for example graphs, sizes of buttons (because )
    def paintEvent(self,event):
        self.bar.setGeometry(10, 10, self.val_width, 25)
        self.bar4.setGeometry(10, 40, self.val_width, 25)
        self.bar3.setGeometry(10, 70, self.val_width, 25) #standard gui configuration
        self.bar2.setGeometry(10, 100, self.val_width, 25)
        self.button0.setGeometry(10, 130, self.val_width, 30)
        #self.label2.setGeometry(10, 70, 100, 25) #standard gui configuration
        self.label00.setGeometry(15, 157, self.val_width, 25)
        self.button1.setGeometry(10, 177, self.val_width, 30)
        
        #buttons for exp fitting
        self.bar5.setGeometry(10, 260-self.a, self.val_width, 25)
        self.bar6.setGeometry(10, 260-self.a, self.val_width, 25)
        self.select_process.setGeometry(10, 290-self.a, self.val_width, 25)
        #self.label3.setGeometry(10, 190+150, 100, 25)
        self.button3.setGeometry(10, 320-self.a, self.val_width, 30)
        self.eps_table.setGeometry(10, 350-self.a, self.val_width, 100+self.down_shith)
        self.button2.setGeometry(10, 450-self.a+self.down_shith, self.val_width, 32) 
        self.weights_true.setGeometry(11, 485-self.a+self.down_shith, self.val_width, 30)
        self.button31.setGeometry(10, 485-self.a+self.down_shith, self.val_width, 30)
        self.label51.setGeometry(10, 515-self.a+self.down_shith, self.val_width, 30)
        self.text51.setGeometry(10, 545-self.a+self.down_shith, self.val_width, 30)
        self.button51.setGeometry(10, 575-self.a+self.down_shith, self.val_width, 30)
        self.button61.setGeometry(10, 570+50-self.a+self.down_shith, self.val_width, 30)
        self.button71.setGeometry(10, 600+50-self.a+self.down_shith, self.val_width, 30)
        
        #buttons for taget fitting
        self.button_tf_1.setGeometry(10, 320-self.a, self.val_width, 30)
        self.tf_eps_table.setGeometry(10, 350-self.a, self.val_width, 100+self.down_shith)
        self.button_tf_3.setGeometry(10, 455-self.a, self.val_width, 30)
#        self.button_tf_5.setGeometry(10, 485-self.a, 100, 30)
#        self.button_tf_6.setGeometry(10, 575-self.a, 100, 30)

        self.message.setGeometry(10, self.window_height-100-30, self.val_width+10, 30)
        
        self.text_proc.setGeometry(10, 220+150, self.val_width, 25)
        self.button4.setGeometry(10, 300+150, self.val_width, 30)
        self.button41.setGeometry(10, 330+150, self.val_width, 30)
        
        self.label12.setGeometry(10, 100+200, self.val_width, 30)
        self.text12.setGeometry(10, 130+200, self.val_width, 30)
        self.label22.setGeometry(10, 160+200, self.val_width, 30)
        self.text22.setGeometry(10, 190+200, self.val_width, 30)
        self.label32.setGeometry(10, 220+200, self.val_width, 30)
        self.text32.setGeometry(10, 250+200, self.val_width, 30)
        self.button02.setGeometry(10, 280+200, self.val_width, 30)
        
        self.label03.setGeometry(10, 100+300, self.val_width, 30)
        self.text03.setGeometry(10, 130+300, self.val_width, 30)
        self.button03.setGeometry(10, 160+300, self.val_width, 30)
        self.button13.setGeometry(10, 190+300, self.val_width, 30)
        
        self.button_plot_edit.setGeometry(10, 550, self.val_width, 35)
        self.button_plot_edit2.setGeometry(10, 550, self.val_width, 35)
        self.button_plot_edit4.setGeometry(10, 550, self.val_width, 35)
        self.button_undo_region.setGeometry(10, 520, self.val_width, 30)
        self.button_undo_time.setGeometry(10, 520, self.val_width, 30)
        
        self.label04.setGeometry(10, 120+200, self.val_width, 30)
        self.text04.setGeometry(10, 150+200, self.val_width, 30)
        self.label14.setGeometry(10, 180+200, self.val_width, 30)
        self.text14.setGeometry(10, 210+200, self.val_width, 30)
        self.label24.setGeometry(10, 240+200, self.val_width, 30)
        self.text24.setGeometry(10, 270+200, self.val_width, 30)
        self.button04.setGeometry(10, 310+200, self.val_width, 30)
        
        self.label_t8.setGeometry(10, 150+200, self.val_width, 30)
        self.text_t8.setGeometry(10, 180+200, self.val_width, 30)
        self.button_t8.setGeometry(10, 210+200, self.val_width, 30)
        
        self.label05.setGeometry(10, 100+200, self.val_width, 30)
        self.text05.setGeometry(10, 130+200, self.val_width, 30)
        self.label15.setGeometry(10, 160+200, self.val_width, 30)
        self.text15.setGeometry(10, 190+200, self.val_width, 30)
        self.button05.setGeometry(10, 220+200, self.val_width, 30)
        self.label25.setGeometry(10, 270+200, self.val_width, 30)
        self.text25.setGeometry(10, 300+200, self.val_width, 30)
        self.button15.setGeometry(10, 330+200, self.val_width, 30)
        
        self.label46.setGeometry(10, 60+200, self.val_width, 30)
        self.text46a.setGeometry(10, 90+200, self.val_width*0.45, 30)
        self.text46b.setGeometry(self.val_width*0.65, 90+200, self.val_width*0.45, 30)
        self.check56.setGeometry(10, 120+200, self.val_width, 30)
        self.label06.setGeometry(10, 150+200, self.val_width, 30)
        self.text06.setGeometry(10, 180+200, self.val_width, 30)
        self.label16.setGeometry(10, 210+200, self.val_width, 30)
        self.text16.setGeometry(10, 240+200, self.val_width, 30)
        self.label26.setGeometry(10, 270+200, self.val_width, 30)
        self.text26.setGeometry(10, 300+200, self.val_width, 30)
        self.label36.setGeometry(10, 330+200, self.val_width, 30)
        self.text36.setGeometry(10, 360+200, self.val_width, 30)
        self.button06.setGeometry(10, 390+200, self.val_width, 30)
        self.button16.setGeometry(10, 420+200, self.val_width, 30)
       
        self.label012.setGeometry(10, 100+200, self.val_width, 30)
        self.text012.setGeometry(10, 130+200, self.val_width, 30)
        self.label022.setGeometry(10, 160+200, self.val_width, 30)
        self.text022.setGeometry(10, 190+200, self.val_width, 30)
        self.label032.setGeometry(10, 220+200, self.val_width, 30)
        self.text032.setGeometry(10, 250+200, self.val_width, 30)
        self.label042.setGeometry(10, 280+200, self.val_width, 30)
        self.text042.setGeometry(10, 310+200, self.val_width, 30)
        self.button002.setGeometry(10, 340+200, self.val_width, 30)
        
        self.label08.setGeometry(10, 40+200, self.val_width, 30)
        self.text08.setGeometry(10, 70+200, self.val_width, 30)
        self.label18.setGeometry(10, 100+200, self.val_width, 30)
        self.text18.setGeometry(10, 130+200, self.val_width, 30)
        self.label28.setGeometry(10, 160+200, self.val_width, 30)
        self.text28.setGeometry(10, 190+200, self.val_width, 30)
        self.label38.setGeometry(10, 220+200, self.val_width, 30)
        self.text38.setGeometry(10, 250+200, self.val_width, 30)
        self.label48.setGeometry(10, 280+200, self.val_width, 30)
        self.text48.setGeometry(10, 310+200, self.val_width, 30)
        self.check08.setGeometry(10, 340+200, self.val_width*1.05, 30)
        self.button08.setGeometry(10, 365+200, self.val_width, 30)
        
        self.label000.setGeometry(10, 100+200, self.val_width, 30)
        self.text000.setGeometry(10, 130+200, self.val_width, 30)
        self.label001.setGeometry(10, 160+200, self.val_width, 30)
        self.text001.setGeometry(10, 190+200, self.val_width, 30)
        self.button001.setGeometry(10, 220+200, self.val_width, 30)
        self.label07.setGeometry(10, 100+250, self.val_width, 30)
        self.text07.setGeometry(10, 130+250, self.val_width*0.70, 30)
        self.text007.setGeometry(self.val_width*0.85, 130+250, self.val_width*0.25, 30)
        self.label17.setGeometry(10, 160+250, self.val_width, 30)
        self.text17.setGeometry(10, 190+250, self.val_width*0.70, 30)
        self.text117.setGeometry(self.val_width*0.85, 190+250, self.val_width*0.25, 30)
        self.label27.setGeometry(10, 220+250, self.val_width*0.55, 30)
        self.text27.setGeometry(10, 250+250, self.val_width*0.40, 30)
        self.label37.setGeometry(self.val_width*0.60, 220+250, self.val_width*0.55, 30)
        self.text37.setGeometry(self.val_width*0.65, 250+250, self.val_width*0.40, 30)
        self.button07.setGeometry(10, 280+250, self.val_width, 30)
        self.button17.setGeometry(10, 70+250, self.val_width, 30)
        self.button27.setGeometry(10, 330+250, self.val_width, 30)
#        self.button37.setGeometry(10, 20+250, 100, 30)
        #number of traces
        self.label005.setGeometry(15, 10+200, self.val_width, 20)
        self.label015.setGeometry(15, 30+200, self.val_width, 20)
        self.label025.setGeometry(15, 50+200, self.val_width, 20)
        self.label035.setGeometry(15, 70+200, self.val_width, 20)
        #cut wavelength region
        self.label023.setGeometry(10, 80+200, self.val_width, 30)
        self.label003.setGeometry(10, 100+200, self.val_width*0.45, 30)
        self.text003.setGeometry(10, 128+200, self.val_width*0.45, 30)
        self.label013.setGeometry(self.val_width*0.70, 100+200, self.val_width*0.45, 30)
        self.text013.setGeometry(self.val_width*0.60, 128+200, self.val_width*0.45, 30)
        self.button003.setGeometry(10, 160+200, self.val_width, 30)
        self.button013.setGeometry(10, 190+200, self.val_width, 30)
        self.label033.setGeometry(10, 200+250, self.val_width, 30)
        self.select_grapth.setGeometry(10, 227+250, self.val_width, 30)
        #cut time region
        self.label024.setGeometry(10, 80+200, self.val_width, 30)
        self.label004.setGeometry(10, 100+200, self.val_width*0.45, 30)
        self.text004.setGeometry(10, 128+200, self.val_width*0.45, 30)
        self.label014.setGeometry(self.val_width*0.70, 100+200, self.val_width*0.45, 30)
        self.text014.setGeometry(self.val_width*0.60, 128+200, self.val_width*0.45, 30)
        self.button004.setGeometry(10, 160+200, self.val_width, 30)
        self.button014.setGeometry(10, 190+200, self.val_width, 30)
        self.label044.setGeometry(10, 200+250, self.val_width, 30)
        self.select_time.setGeometry(10, 227+250, self.val_width, 30)

        self.label09.setGeometry(10, 100+200, self.val_width, 30)
        self.text09.setGeometry(10, 130+200, self.val_width, 30)
        self.label19.setGeometry(10, 160+200, self.val_width, 30)
        self.text19.setGeometry(10, 190+200, self.val_width, 30)
        self.label29.setGeometry(10, 220+200, self.val_width, 30)
        self.select_average.setGeometry(10, 250+200, self.val_width, 30)
        self.label39.setGeometry(10, 282+200, self.val_width*0.75, 30)
        self.spin.setGeometry(self.val_width*0.75, 282+200, self.val_width*0.32, 30)
        self.button09.setGeometry(10, 315+200, self.val_width, 30)
        
        self.button37.setGeometry(10, 500, self.val_width, 30)
        self.label_progress.setGeometry(10, 180, self.val_width, 30)
        self.progress_fit.setGeometry(10, 210, self.val_width, 275)
        
        self.label124.setGeometry(10, 80+200, self.val_width, 30)
        self.label104.setGeometry(15, 100+200, self.val_width*0.49, 30)
        self.text104.setGeometry(10, 128+200, self.val_width*0.49, 30)
        self.label114.setGeometry(self.val_width*0.70, 100+200, self.val_width*0.45, 30)
        self.text114.setGeometry(self.val_width*0.6, 128+200, self.val_width*0.45, 30)
        self.button104.setGeometry(15, 160+200, self.val_width, 30)
        
        if self.edit_general['time_shift']==False:
            self.label_t8.setVisible(False)
            self.text_t8.setVisible(False)
            self.button_t8.setVisible(False)
            
        else:
            
            self.label_t8.show()
            self.text_t8.show()
            self.button_t8.show()
            
            
        if self.edit_general['load_data_edit']==False:
            self.label07.setVisible(False)
            self.text07.setVisible(False)
            self.text007.setVisible(False)
            self.label17.setVisible(False)
            self.text17.setVisible(False)
            self.text117.setVisible(False)
            self.label27.setVisible(False)
            self.text27.setVisible(False)
            self.label37.setVisible(False)
            self.text37.setVisible(False)
            self.button07.setVisible(False)
            self.button17.setVisible(False)
            self.button27.setVisible(False)
#            self.button37.setVisible(False)
        
        else:   
            self.label07.show()
            self.text07.show()
            self.text007.show()
            self.label17.show()
            self.text17.show()
            self.text117.show()
            self.label27.show()
            self.text27.show()
            self.label37.show()
            self.text37.show()
            self.button07.show()
            self.button17.show()
            self.button27.show() 
#            self.button37.show()  
        
        if self.edit_general['series_traces_edit']==False:
            self.label012.setVisible(False)
            self.text012.setVisible(False)
            self.label022.setVisible(False)
            self.text022.setVisible(False)
            self.label032.setVisible(False)
            self.text032.setVisible(False)
            self.label042.setVisible(False)
            self.text042.setVisible(False)
            self.button002.setVisible(False)
        
        else:   
            self.label012.show()
            self.text012.show()
            self.label022.show()
            self.text022.show()
            self.label032.show()
            self.text032.show()
            self.label042.show()
            self.text042.show()
            self.button002.show()
        
        if self.edit_general['sellmeier_edit']==False:
            self.label08.setVisible(False)
            self.text08.setVisible(False)
            self.label18.setVisible(False)
            self.text18.setVisible(False)
            self.label28.setVisible(False)
            self.text28.setVisible(False)
            self.label38.setVisible(False)
            self.text38.setVisible(False)
            self.label48.setVisible(False)
            self.text48.setVisible(False)
            self.check08.setVisible(False)
            self.button08.setVisible(False)  
        
        else:   
            self.label08.show()
            self.text08.show()
            self.label18.show()
            self.text18.show()
            self.label28.show()
            self.text28.show()
            self.label38.show()
            self.text38.show()
            self.label48.show()
            self.text48.show()
            self.check08.show()
            self.button08.show()
            
        if self.edit_general['traces_manually_edit']==False:
            self.label000.setVisible(False)
            self.text000.setVisible(False)
            self.label001.setVisible(False)
            self.text001.setVisible(False)
            self.button001.setVisible(False)
        
        else:   
            self.label000.show()
            self.text000.show()
            self.label001.show()
            self.text001.show()
            self.button001.show()
        
        if self.edit_general['average_time_edit']==False:
            self.label09.setVisible(False)
            self.text09.setVisible(False)
            self.label19.setVisible(False)
            self.text19.setVisible(False)
            self.label29.setVisible(False)
            self.label39.setVisible(False)
            self.select_average.setVisible(False)
            self.spin.setVisible(False)
            self.button09.setVisible(False)
        
        else:   
            self.label09.show()
            self.text09.show()
            self.label19.show()
            self.text19.show()
            self.label29.show()
            self.select_average.show()
            if self.timeAverigeComboBox_edit==False:
                self.label39.show()
                self.spin.show()
            else:
                self.label39.setVisible(False)
                self.spin.setVisible(False)
            self.button09.show()
        
        
        
        if self.edit_general['pop_edit'] == False: #exp fit menu
            self.message.show()
#            self.label3.setVisible(False)
            self.eps_table.setVisible(False)
            self.button2.setVisible(False)
            self.button3.setVisible(False)
            self.button31.setVisible(False)
            self.weights_true.setVisible(False)
            self.bar5.setVisible(False)
            if self.edit_general['target_fit'] == False:
                self.button61.setVisible(False)
                self.button71.setVisible(False)
                self.label51.setVisible(False)
                self.text51.setVisible(False)
                self.button31.setVisible(False)
                self.button51.setVisible(False)
                self.select_process.setVisible(False)
            
        else:
            self.message.setVisible(False)
#            self.label3.show()
            self.eps_table.show()
            self.button2.show()
            self.button3.show()
            if self.weights_true.isChecked():
                self.weights_true.show()
                self.button31.setVisible(False)
            else:
                self.weights_true.setVisible(False)
                self.button31.show()
            self.button51.show()
            self.button61.show()
            self.button71.show()
            self.label51.show()
            self.text51.show()
            self.select_process.show()
            self.bar5.show()
            if self.a==-70:
                self.edit_general['number_traces_edit']=True
        
        if self.edit_general['target_fit'] == False: #target fit buttons
            self.message.show()
            self.button_tf_1.setVisible(False)
            self.button_tf_3.setVisible(False)
            self.tf_eps_table.setVisible(False)
#            self.button_tf_5.setVisible(False)
#            self.button_tf_6.setVisible(False)
            self.bar6.setVisible(False)
            if self.edit_general['pop_edit'] == False:
                self.button61.setVisible(False)
                self.button71.setVisible(False)
                self.label51.setVisible(False)
                self.text51.setVisible(False)
                self.button51.setVisible(False)
                self.weights_true.setVisible(False)
                self.button31.setVisible(False)
                self.select_process.setVisible(False)
                self.bar5.setVisible(False)
            
        else:
            self.message.setVisible(False)
#            self.label3.show()
            self.button_tf_1.show()
            self.button_tf_3.show()
            self.tf_eps_table.show()
            if self.weights_true.isChecked():
                self.weights_true.show()
                self.button31.setVisible(False)
            else:
                self.weights_true.setVisible(False)
                self.button31.show()
            self.button61.show()
            self.button51.show()
            self.button71.show()
            self.label51.show()
            self.text51.show()
            self.select_process.show()
            self.bar6.show()
            if self.a==-70:
                self.edit_general['number_traces_edit']=True
            
        if self.edit_general['proc_edit'] == False:
            self.button4.setVisible(False)
            self.button41.setVisible(False)
            self.text_proc.setVisible(False)
        else:
            self.button4.show()
            self.button41.show()
            self.text_proc.show()   
#            self.label3.show()
         
        if self.edit_general['deriv_edit']==False:
            self.label12.setVisible(False)
            self.text12.setVisible(False)
            self.label22.setVisible(False)
            self.text22.setVisible(False)
            self.label32.setVisible(False)
            self.text32.setVisible(False)
            self.button02.setVisible(False)
            
        else:   
            self.label12.show()
            self.text12.show()
            self.label22.show()
            self.text22.show()
            self.label32.show()
            self.text32.show()
            self.button02.show()
        
        if self.edit_general['baseline_edit']==False:
            self.label03.setVisible(False)
            self.text03.setVisible(False)
            self.button03.setVisible(False)
            self.button13.setVisible(False)         
            self.button_plot_edit.setVisible(False)        
            
        else:   
            self.label03.show()
            self.text03.show()
            self.button03.show()
            self.button13.show()
            self.button_plot_edit.show()
            
        if self.edit_general['plot_spec_edit']==False:
            self.label04.setVisible(False)
            self.text04.setVisible(False)
            self.label14.setVisible(False)
            self.text14.setVisible(False)
            self.label24.setVisible(False)
            self.text24.setVisible(False)
            self.button04.setVisible(False)  
        
        else:   
            self.label04.show()
            self.text04.show()
            self.label14.show()
            self.text14.show()
            self.label24.show()
            self.text24.show()
            self.button04.show()
        
        if  self.edit_general['settings_edit']==False:
            self.label05.setVisible(False)
            self.text05.setVisible(False)
            self.label15.setVisible(False)
            self.text15.setVisible(False)
            self.button05.setVisible(False)
            self.label25.setVisible(False)
            self.text25.setVisible(False)
            self.button15.setVisible(False)
        
        else:   
            self.label05.show()
            self.text05.show()
            self.label15.show()
            self.text15.show()
            self.button05.show()
            self.label25.show()
            self.text25.show()
            self.button15.show()
        
        if self.edit_general['plot_spec_auto_edit']==False:
            self.label46.setVisible(False)
            self.text46a.setVisible(False)
            self.check56.setVisible(False)
            self.text46b.setVisible(False)
            self.label06.setVisible(False)
            self.text06.setVisible(False)
            self.label16.setVisible(False)
            self.text16.setVisible(False)
            self.label26.setVisible(False)
            self.text26.setVisible(False)
            self.label36.setVisible(False)
            self.text36.setVisible(False)
            self.button06.setVisible(False)
            self.button16.setVisible(False)
        
        else:   
            self.label46.show()
            self.text46a.show()
            self.text46b.show()
            self.check56.show()
            self.label06.show()
            self.text06.show()
            self.label16.show()
            self.text16.show()
            self.label26.show()
            self.text26.show()
            self.label36.show()
            self.text36.show()
            self.button06.show()
            self.button16.show()
        
        if self.edit_general['cut_region_edit']==False:
            self.label023.setVisible(False)
            self.label003.setVisible(False)
            self.text003.setVisible(False)
            self.label013.setVisible(False)
            self.text013.setVisible(False)
            self.button003.setVisible(False)
            self.button013.setVisible(False)
            self.label033.setVisible(False)
            self.select_grapth.setVisible(False)
            self.button_plot_edit2.setVisible(False)
            self.button_undo_region.setVisible(False)
            
        else:   
            self.label023.show()
            self.label003.show()
            self.text003.show()
            self.label013.show()
            self.text013.show()
            self.button003.show()
            self.button013.show()
            self.label033.show()
            self.select_grapth.show()
            self.button_plot_edit2.show()
            self.button_undo_region.show()
        
        if self.edit_general['cut_time_edit']==False:
            self.label024.setVisible(False)
            self.label004.setVisible(False)
            self.text004.setVisible(False)
            self.label014.setVisible(False)
            self.text014.setVisible(False)
            self.button004.setVisible(False)
            self.button014.setVisible(False)
            self.label044.setVisible(False)
            self.select_time.setVisible(False)
            self.button_plot_edit4.setVisible(False)
            self.button_undo_time.setVisible(False)
            
        else:   
            self.label024.show()
            self.label004.show()
            self.text004.show()
            self.label014.show()
            self.text014.show()
            self.button004.show()
            self.button014.show()
            self.label044.show()
            self.select_time.show()
            self.button_plot_edit4.show()
            self.button_undo_time.show()
        
        if self.edit_general['fitting_edit']==False:
            self.label_progress.setVisible(False)
            self.progress_fit.setVisible(False)
            self.button37.setVisible(False)
            
        else:
            self.label_progress.show()
            self.progress_fit.show()
            self.button37.show()
        
        if self.edit_general['number_traces_edit']==False:
             self.label005.setVisible(False)
             self.label015.setVisible(False)
             self.label025.setVisible(False)
             self.label035.setVisible(False)
        else:
            self.label005.show()
            self.label015.show()
            self.label025.show()
            self.label035.show()
            
        if self.edit_general['init_fianl_wave_edit']==False:
            self.label124.setVisible(False)
            self.label104.setVisible(False)
            self.text104.setVisible(False)
            self.label114.setVisible(False)
            self.text114.setVisible(False)
            self.button104.setVisible(False)
            
        else:   
            self.label124.show()
            self.label104.show()
            self.text104.show()
            self.label114.show()
            self.text114.show()
            self.button104.show()
        #painter = QPainter(self) #you can use this tool to draw things/animations
        #painter.begin(self)
        

        #painter.end()
        
    def eventFilter(self, obj, event): #it can be used to catch some events like mouse button press etc.
        
        if event.type() == QEvent.MouseMove:
            mouse_position = event.pos()
            if self.mousepressed != False:
                self.mouse_dx = mouse_position.x() - self.ref_mouse.x()
                self.mouse_dy = mouse_position.y() - self.ref_mouse.y()
                self.repaint()
            
        elif event.type() == QEvent.MouseButtonRelease:
            if self.mousepressed != False:
                self.mousepressed.rect.setX(self.mousepressed.rect.x() + self.mouse_dx)
                self.mousepressed.rect.setY(self.mousepressed.rect.y() + self.mouse_dy)
                self.mousepressed.rect.setWidth(self.mousepressed.rect_w)
                self.mousepressed.rect.setHeight(self.mousepressed.rect_h)
                self.mouse_dx = 0
                self.mouse_dy = 0
                self.mousepressed = False
                self.repaint()
        elif event.type() == QEvent.MouseButtonPress: 
            self.repaint()
        
        elif event.type() == QEvent.MouseButtonDblClick:
            self.repaint()

        return False
#label.setText("Experiment ready!")
#sys.stdout = open('file', 'w')    
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
plt.ioff()
#if hasattr(Qt, 'AA_EnableHighDpiScaling'):
app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
splash.setVisible(False)
#creates app instance
ex = ModelWindow() #creates your window
width = QApplication.desktop().screenGeometry().width()
height = QApplication.desktop().screenGeometry().height()
ex.setGeometry(50,50,width-100,height-100)
ex.show() #shows window
app.exec() #this will execute until you close window

#t = threading.Thread(target=main)
#t.daemon = True
#t.start()

#from PyQt5.QtWidgets import QStyleFactory
#[key for key in QStyleFactory.keys()]
