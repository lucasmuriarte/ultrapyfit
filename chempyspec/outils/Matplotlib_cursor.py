# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 14:51:06 2020

@author: 79344
"""

import numpy as np

class SnaptoCursor(object):
    def __init__(self, ax,x, y,number_click=-1,vertical_draw=True,draw='snap',color=False):
        if number_click==-1:
            self.number_click=np.inf
        else:
            self.number_click=number_click
        self.ax = ax
        self.draw=draw
        self.vertical_draw=vertical_draw
        self.similar=y==np.zeros(len(y))
        self.color=color
        self.x = x
        self.y = y
        self.datax=[]
        self.datay=[]
        self.scat=[]
    def mouseMove(self, event):
        if not event.inaxes: return
        self.x_pos, self.y_pos = event.xdata, event.ydata
        indx = np.searchsorted(self.x, [self.x_pos])[0]
        x = self.x[indx]
        y = self.y[indx]
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
        if event.button==1:
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
        elif event.button==3:
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