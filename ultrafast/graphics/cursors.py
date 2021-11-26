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


if __name__ == '__main__':
    main()