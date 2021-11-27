# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 17:59:23 2020

@author: 79344
"""
import numpy as np
from matplotlib.backend_bases import MouseButton

class Cursor:
    def __init__(self, ax, n_clicks=np.inf):
        super().__init__()
        self.ax = ax
        self.n_clicks = n_clicks
        self.fig = self.ax.figure
        self.lines = []
        self.data = []

        self.canvas = self.fig.canvas
        self.canvas.draw()
        self.bbox = self.canvas.copy_from_bbox(self.fig.bbox)

        self.vline = self.ax.axvline(
            np.inf, color='black', alpha=0.4, animated=True)
        self.hline = self.ax.axhline(
            np.inf, color='black', alpha=0.4, animated=True)

        self.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        self.canvas.mpl_connect('resize_event', self.on_resize)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        print(self.n_clicks)

    def on_click(self, event):
        if event.inaxes:
            if event.button is MouseButton.LEFT and len(self.data) < self.n_clicks:
                self.data.append(event.xdata)
                self.lines.append(
                    self.ax.axvline(
                        event.xdata, color='red', alpha=0.6, animated=True)
                )
                self.on_resize(None)

            elif event.button is MouseButton.RIGHT and len(self.data) > 0:
                del self.lines[-1]
                del self.data[-1]
                self.draw_artists()

        print(event.xdata)


    def mouse_move(self, event=None):
        if event.inaxes:
            self.vline.set_xdata(event.xdata)
            self.hline.set_ydata(event.ydata)
            self.draw_artists()
            self.canvas.flush_events()

    def on_resize(self, _):
        self.canvas.draw()
        self.bbox = self.canvas.copy_from_bbox(self.fig.bbox)
        self.draw_artists()

    def draw_artists(self):
        self.canvas.restore_region(self.bbox)
        self.ax.draw_artist(self.vline)
        self.ax.draw_artist(self.hline)

        for line in self.lines:
            self.ax.draw_artist(line)

        self.canvas.blit(self.ax.bbox)

class VerticalCursor(Cursor):
    def __init__(self, ax, n_clicks=np.inf):
        super().__init__(ax, n_clicks)

        self.hline.set_alpha(0.0)

class HorizontalCursor(Cursor):
    def __init__(self, ax, n_clicks=np.inf):
        super().__init__(ax, n_clicks)
        
        self.vline.set_alpha(0.0)

class SnappedCursor(Cursor):
    def __init__(self, ax, x, y, n_clicks=np.inf):
        super().__init__(ax, n_clicks)
        self.x = x
        self.y = y
        self.scatter = self.ax.scatter(
            [np.inf], [np.inf], color='red',
            marker='+', alpha=0.7, animated=True
        )

    def mouse_move(self, event):
        if event.inaxes:
            index = (np.abs(self.x - event.xdata)).argmin()
            x_point, y_point = self.x[index], self.y[index]
            self.vline.set_xdata(x_point)
            self.hline.set_ydata(y_point)
            self.scatter.set_offsets([x_point, y_point])

            self.draw_artists()
            self.canvas.flush_events()

    def draw_artists(self):
        self.canvas.restore_region(self.bbox)

        self.ax.draw_artist(self.vline)
        self.ax.draw_artist(self.hline)
        self.ax.draw_artist(self.scatter)

        for line in self.lines:
            self.ax.draw_artist(line)

        self.canvas.blit(self.ax.bbox)


def main():
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(1, 10, 101) +5
    y = np.sin(x)

    fig, ax = plt.subplots(1, figsize=(18, 6))
    ax.plot(x, y)

    cursor = VerticalCursor(ax)
    plt.show()

class SnaptoCursor(object):
    def __init__(self, ax, x, y, number_click=-1, vertical_draw=True,
                 draw='snap', color=False, single_line=True):
        if number_click == -1:
            self.number_click = np.inf
        else:
            self.number_click = number_click
        self.ax = ax
        self.draw = draw
        self.vertical_draw = vertical_draw
        self.color = color
        self.x = x
        self.y = y
        self.similar = y == np.zeros(len(y))
        self.datax = []
        self.datay = []
        self.scat = []
        self.single_line = single_line
        self.x_pos = None
        self.y_pos = None

    def mouseMove(self, event):
        if not event.inaxes:
            return
        self.x_pos, self.y_pos = event.xdata, event.ydata
        if self.single_line:
            indx = np.searchsorted(self.x, [self.x_pos])[0]
            x = self.x[indx]
            y = self.y[indx]
        else:
            x = self.x_pos
            y = self.y_pos
        self.ly.set_xdata(x)
        self.marker.set_data([x], [y])
        if abs(x) >= 0.1:
            texto_x = 1
        else:
            try:
                texto_x = [True if i == '0' else False for i in str(x).split('.')[1]].index(False)+1
            except:
                texto_x = 3
        if abs(y) >= 0.1:
            texto_y = 1
        else:
            try:
                texto_y = [True if i == '0' else False for i in str(y).split('.')[1]].index(False)+1
            except:
                texto_y = 3
        if self.similar.all() == False:
            self.lx.set_ydata(y)
            self.txt.set_text('x='+str(round(x, texto_x)) +
                              ', y='+str(round(y, texto_y)))
            self.txt.set_position((x, y))
        else:
            self.txt.set_text('x=' + str(round(x, texto_x)))
        self.txt.set_position((x, y))
        self.ax.figure.canvas.draw_idle()
    
    def onClick(self, event):
        if not event.inaxes:
            return
        if event.button == 1:
            # print(self.number_click)
            if len(self.datax) < self.number_click:
                x, y = event.xdata, event.ydata
                if self.draw == 'snap':
                    indx = np.searchsorted(self.x, [x])[0]
                    x = self.x[indx]
                    y = self.y[indx]
                self.datax.append(x)
                self.datay.append(y)
#                print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
#                      ('double' if event.dblclick else 'single', event.button,
#                       event.x, event.y, event.xdata, event.ydata))
                if self.vertical_draw:
                    self.scat.append(self.ax.axvline(self.datax[-1],
                                                     alpha=0.5,
                                                     color='red',
                                                     zorder=np.inf))
                else:
                    self.scat.append(self.ax.scatter(self.datax,
                                                     self.datay,
                                                     color='red',
                                                     marker='x',
                                                     zorder=np.inf))
            else:
                pass
            self.ax.figure.canvas.draw_idle()
        elif event.button == 3:
            if len(self.datax) == 0:
                pass
            else:
                del self.datax[-1]
                del self.datay[-1]
                self.scat[-1].remove()
                del self.scat[-1]
                self.ax.figure.canvas.draw_idle()  
    
    def onEnterAxes(self, event):
        if not event.inaxes:
            return
        try:
            self.onLeaveAxes(event)
        except:
            pass
        if self.similar.all() == False:
            self.lx = self.ax.axhline(color='k', alpha=0.2)  # the horiz line
        if self.single_line:
            try:
                line = self.ax.lines[0]
                self.x = line.get_xdata()
                if self.similar.all()==False:
                    self.y = line.get_ydata()
            except:
                pass
        self.ly = self.ax.axvline(color='k', alpha=0.2)  # the vert line
        self.marker, = self.ax.plot([0], [0], marker="o",
                                    color="crimson", zorder=3)
        self.txt = self.ax.text(0.7, 0.9, '')
        if self.color is not False:
            event.inaxes.patch.set_facecolor(self.color)
        event.canvas.draw()
    
    def onLeaveAxes(self, event):
        if not event.inaxes:
            return
        # print ('leave_axes', event.inaxes)
        self.marker.remove()
        self.ly.remove()
        self.txt.remove()
        if self.similar.all() == False:
            self.lx.remove()
        event.inaxes.patch.set_facecolor('white')
        event.canvas.draw()

    def clear(self):
        for i in range(len(self.scat)):
            self.scat[-1].remove()
            del self.scat[-1]
        self.datax = []
        self.datay = []
        self.ax.figure.canvas.draw_idle()



if __name__ == '__main__':
    main()