import numpy as np
# from scipy.integrate import odeint
# import matplotlib.pyplot as plt
import lmfit
import random
import sys
import keyword
import copy
import math
import pickle
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5 import QtCore, QtGui, QtWidgets, Qt
# from kineticdata import Experiment

# firstly create populations and put them i model
# then add arrows. Create arrow binded by two populations and add to model list
# arrow should add itself to both end-populations
# remove arrow by calling kill() in this arrow and removing from model list
# arrow should remove itself from both end-populations
# if you kill population, firstly remove all arrows inside


def isIdentifier(identifier):  # check if string is valid python identifier
    if not(isinstance(identifier, str)):
        return False
    if not(identifier.isidentifier()):
        return False
    if keyword.iskeyword(identifier):
        return False
    return True


class ModPopulation:
    def __init__(self, new_name):
        self.arrows = list()  # processes associated with this population
        self.name = new_name  # name of the population
        # self.epsilon = dict() # define epsilons for irr and probe wavelengths
        self.initial = 0.0
        
        self.color = QtCore.Qt.black
        self.rect_w = 120
        self.rect_h = 40
        self.rect = QtCore.QRect(250, 50, self.rect_w, self.rect_h)
        
        self.c = 0
        self.k_all = 1  # both used depending on mode selected
        self.tau = 1
        self.c_fixed = True
        self.k_all_fixed = False
        self.tau_fixed = False
        self.c_active = True
        self.k_all_active = False  # active means editable by user
        self.tau_active = False
        self.c_enabled = True
        self.k_all_enabled = False  # enabled means that user is allowed to turn it active
        self.tau_enabled = False  # if enabled is False, then active also must be False
        
    def remove(self, model):  # before calling ensure that all arrows with this population are removed, in other case there will be exception!
        if len(self.arrows) != 0:
            raise Exception('Attempted to invalid population removal!!')
        n = model.populations.index(self)
        model.populations.pop(n)
        
    def countActive(self):  # count active arrows plus this population
        active_counter = 0
        
        if self.k_all_active:
            active_counter += 1
        if self.tau_active:
            active_counter += 1
        
        for arrow in self.arrows:
            if arrow.source is self:
                if arrow.k_active:
                    active_counter += 1
                if arrow.sf_active:
                    active_counter += 1
                    
        return active_counter
    
    def countFixed(self):
        fixed_counter = 0
        
        if self.k_all_fixed:
            fixed_counter += 1
        elif self.tau_fixed:
            fixed_counter += 1
        
        for arrow in self.arrows:
            if arrow.source is self:
                if arrow.k_fixed:
                    fixed_counter += 1
                if arrow.sf_fixed:
                    fixed_counter += 1
                    
        return fixed_counter    
    
    def countActiveSFs(self):  # count active Scalling Factor fields
        active_counter = 0        
        for arrow in self.arrows:
            if arrow.source is self:
                if arrow.sf_active:
                    active_counter += 1
                    
        return active_counter    
    
    def countActiveSFsValues(self):  # count active Scalling Factor fields
        sfall = 0.0
        for arrow in self.arrows:
            if arrow.source is self:
                if arrow.sf_active:
                    sfall += arrow.sf
                    
        return sfall
    
    def countOutgoingArrows(self):  # count all arrows going out from this population
        arrow_counter = 0
        
        for arrow in self.arrows:
            if arrow.source is self:
                arrow_counter += 1
        
        return arrow_counter
 
    def enableDisableKs(self):  # rewrite enable/disable states in this branch
        active_counter = self.countActive()
        active_SF_counter = self.countActiveSFs()        
        arrow_counter = self.countOutgoingArrows()
        
        if arrow_counter == 0:
            if not self.tau_active:
                self.k_all_enabled = True
            else:
                self.k_all_enabled = False
            if not self.k_all_active:
                self.tau_enabled = True
            else:
                self.tau_enabled = False
            
        elif active_counter < arrow_counter:  # enable everything which is inactive and can be enabled
            double_present = False  # if in some arrow two fields are active, it determines k_all value.
            for arrow in self.arrows:
                if arrow.source is self:
                    if arrow.k_active and arrow.sf_active:
                        double_present = True

            if double_present:
                self.k_all_enabled = False
                self.tau_enabled = False
                for arrow in self.arrows:
                    if arrow.source is self:
                        if not arrow.sf_active:
                            arrow.k_enabled = True
                        else:
                            arrow.k_enabled = False

                        condition = not arrow.k_active and active_SF_counter < arrow_counter - 1
                        if condition or arrow.sf_active:
                            arrow.sf_enabled = True
                        else:
                            arrow.sf_enabled = False
                
            else:
                self.k_all_enabled = True if not self.tau_active else False
                self.tau_enabled = True if not self.k_all_active else False
                for arrow in self.arrows:
                    if arrow.source is self:
                        if not self.k_all_active and not self.tau_active:
                            arrow.k_enabled = True
                            if active_SF_counter < arrow_counter - 1 or arrow.sf_active:
                                arrow.sf_enabled = True
                            else:
                                arrow.sf_enabled = False
                        else:
                            if not arrow.sf_active:
                                arrow.k_enabled = True
                            else:
                                arrow.k_enabled = False

                            condition = not arrow.k_active and active_SF_counter < arrow_counter - 1
                            if condition or arrow.sf_active:
                                arrow.sf_enabled = True
                            else:
                                arrow.sf_enabled = False
                            
        else:  # disable everything which is inactive
            for arrow in self.arrows:
                if arrow.source is self:
                    if not arrow.k_active:
                        arrow.k_enabled = False
                    if not arrow.sf_active:
                        arrow.sf_enabled = False                        
                        
            if not self.k_all_active:
                self.k_all_enabled = False
            if not self.tau_active:
                self.tau_enabled = False              
                
    def updateBranchKs(self):  # calculate not active fields from active ones
        taskdone = False
        arrow_counter = self.countOutgoingArrows()
        lacking = arrow_counter - self.countActive()
        # 0.set markers
        self.k_all_determined = self.k_all_active
        self.tau_determined = self.tau_active
        if self.tau_active == True:
            self.k_all_determined = True
            self.k_all = 1/self.tau if self.tau != 0 else np.inf
        for arr in self.arrows:
            if arr.source is self:
                arr.k_determined = arr.k_active
                arr.sf_determined = arr.sf_active
            
        for i in range(2):
            # 1. search for solutions of sf_x * k_x = k_all (thirds)
            for j in range(2):
                for arr in self.arrows:
                    if arr.source is self:
                        if arr.k_determined and arr.sf_determined and not self.k_all_determined:
                            self.k_all = arr.k / arr.sf if arr.sf != 0 else 0
                            self.k_all_determined = True if arr.sf != 0 else False
                        if arr.k_determined and not arr.sf_determined and self.k_all_determined:
                            arr.sf = arr.k / self.k_all if self.k_all != 0 else 0
                            arr.sf_determined = True if self.k_all != 0 else False
                        if not arr.k_determined and arr.sf_determined and self.k_all_determined:
                            arr.k = arr.sf * self.k_all
                            arr.k_determined = True
                        if not arr.k_determined and arr.sf_determined and arr.sf == 0:
                            arr.k = 0
                            arr.k_determined = True
                        if self.k_all_determined and self.k_all == 0 and not arr.k_determined:
                            arr.k_determined = True
                            arr.k = 0    
            # 2. check if k's and k_al can give all (k_all=k_1+k_2+... equation)
            counter = 0
            if self.k_all_determined:
                counter += 1
            for arr in self.arrows:
                if arr.source is self:
                    if arr.k_determined:
                        counter += 1
            if counter >= arrow_counter:  # means that system is determined
                sumk = 0
                for arr in self.arrows:
                    if arr.source is self and arr.k_determined:
                        sumk += arr.k                
                if not self.k_all_determined:
                    self.k_all = sumk
                    self.k_all_determined = True
                else:
                    for arr in self.arrows:
                        if arr.source is self and not arr.k_determined:
                            arr.k = self.k_all - sumk
                            arr.k_determined = True    
                            break
            # 3. if not all possible parameters are given,
            # add sf's = (1-other known sf's)/no of lacking sf's
            if lacking > 0:
                sf_init = (1-self.countActiveSFsValues())/lacking
                for arr in self.arrows:
                    if lacking > 0 and arr.source is self and not arr.sf_determined:
                        arr.sf = sf_init
                        arr.sf_determined = True
                        lacking -= 1
            # 4. do sf's excluding thing to get k_all or some k's
            # (like in point 2)
            if not self.k_all_determined:
                ksum = 0.0
                sfsum = 0.0
                arrowssum = 0
                for arr in self.arrows:
                    if arr.source is self:
                        if arr.sf_determined:
                            sfsum += arr.sf
                            arrowssum += 1
                        if arr.k_determined:
                            ksum += arr.k
                            arrowssum += 1  
                if arrowssum == arrow_counter:
                    self.k_all = ksum / (1-sfsum) if sfsum != 1 else 0
                    self.k_all_determined = True if sfsum != 1 else False
            # 5. stop if all values are determined
            if self.k_all_determined:
                det_sum = 0
                for arr in self.arrows:
                    if arr.source is self:
                        if arr.k_determined and arr.sf_determined:
                            det_sum += 1
                if det_sum == arrow_counter:
                    taskdone = True
                    break  # everything is determined!! (except tau)

        # take care of tau
        if not self.tau_determined and self.k_all_determined:
            self.tau = 1/self.k_all if self.k_all != 0 else np.inf
            self.tau_determined = True
            
        return taskdone

######################################################################
#        if(not(self.k_all_active) and not(self.tau_active)): #update all inactive elements
#            sum_k = sum([arr.k for arr in self.arrows if not(arr.sf_active) and arr.source == self])
#            sum_sf = sum([arr.sf for arr in self.arrows if arr.sf_active and arr.source == self])
#            if(sum_sf < 1): self.k_all = sum_k/(1-sum_sf) #don't touch if only sf's are set
#            self.tau = 1/self.k_all if self.k_all != 0 else np.inf
#        for arrow in self.arrows:
#            if(arrow.source == self and not(self.k_all_active) and not(self.tau_active)):
#                sum_k = sum([arr.k for arr in self.arrows if not(arr.sf_active) and arr.source == self and not(arr is arrow)])
#                sum_sf = sum([arr.sf for arr in self.arrows if arr.sf_active and arr.source == self and not(arr is arrow)])
#                arrow.k = self.k_all*(1-sum_sf) - sum_k
#                arrow.sf = arrow.k/self.k_all if self.k_all != 0 else 0.0
                
    def updateState(self, c_edit, tau_edit, k_all_edit):
        k_all_active = k_all_edit.isActive()
        k_all_edit.setActive(k_all_active)

        tau_active = tau_edit.isActive()
        tau_edit.setActive(tau_active)
        
        c_active = c_edit.isActive()
        c_edit.setActive(c_active)        
        
        if k_all_active != self.k_all_active:  # change testing
            self.k_all_active = k_all_active
            self.enableDisableKs()
            tau_edit.setEnabled(self.tau_enabled)
        elif tau_active != self.tau_active:  # change testing
            self.tau_active = tau_active
            self.enableDisableKs()
            k_all_edit.setEnabled(self.k_all_enabled)
        if c_active != self.c_active:  # change testing
            self.c_active = c_active

        # update values if given field was active (and fixation state)
        # calculate and update values of not active fields (if some field was changed)
        if k_all_active:
            self.k_all = float(k_all_edit.text.text())
            self.k_all_fixed = k_all_edit.check_fixed.isChecked()
            self.updateBranchKs()
        elif tau_active:
            self.tau = float(tau_edit.text.text())
            self.tau_fixed = tau_edit.check_fixed.isChecked()
            self.updateBranchKs()
        if c_active:
            self.c = float(c_edit.text.text())
            self.c_fixed = c_edit.check_fixed.isChecked()
            
        if not k_all_active:
            k_all_edit.text.setText("%.9f" % self.k_all)
        if not tau_active:
            tau_edit.text.setText("%.9f" % self.tau)
        if not c_active:
            c_edit.text.setText("%.9f" % self.c)
            
            
class ModProcess:
    def __init__(self, pop_source, pop_target):
        self.name = f"{pop_source.name}_to_{pop_target.name}"  # name of the process
        self.source = pop_source  # initialize yourself with both neighbour populations
        self.target = pop_target
        self.source.arrows.append(self)  # initialize neighbour populations with yourself
        self.target.arrows.append(self)
        self.p1 = QtCore.QPoint()
        self.p2 = QtCore.QPoint()
        self.number = 1  # number of arrow between some pair of populations.numbering halps to render arrows separately
        self.displacement = 14  # separation distance between arrows
        self.dist_treshold = 7  # if distance from point to crossing point is below treshold function cotains return true hehehe
        self.color = QtCore.Qt.black
        
        self.k = 1  # both used depending on mode selected
        self.sf = 1
        self.k_fixed = False
        self.sf_fixed = False
        self.k_active = False  # active means editable by user
        self.sf_active = False
        self.k_enabled = False  # enabled means that user is allowed to turn it active
        self.sf_enabled = False  # if enabled is False, then active also must be False

    def updateState(self, k_edit, sf_edit):
        k_active = k_edit.isActive()
        k_edit.setActive(k_active)

        sf_active = sf_edit.isActive()
        sf_edit.setActive(sf_active) 
        
        if k_active != self.k_active:  # change testing
            self.k_active = k_active
            self.source.enableDisableKs()
            sf_edit.setEnabled(self.sf_enabled)
        elif sf_active != self.sf_active:  # change testing
            self.sf_active = sf_active
            self.source.enableDisableKs()     
            k_edit.setEnabled(self.k_enabled)
        
        # update values if given field was active (and fixation state)
        # calculate and update values of not active fields (if some field was changed)
        if k_active:
            self.k = float(k_edit.text.text())
            self.k_fixed = k_edit.check_fixed.isChecked()
            self.source.updateBranchKs()
        if sf_active:
            self.sf = float(sf_edit.text.text())
            self.sf_fixed = sf_edit.check_fixed.isChecked()
            self.source.updateBranchKs()
            
        if not k_active:
            k_edit.text.setText("%.9f" % self.k)
        if not sf_active:
            sf_edit.text.setText("%.9f" % self.sf)

    def remove(self, model):  # removes arrow from neighbouring populations and model
        n1 = self.source.arrows.index(self)
        self.source.arrows.pop(n1)
        n2 = self.target.arrows.index(self)
        self.target.arrows.pop(n2)  # here you have to recount arrows between populations...
        count = 1
        for arr in self.source.arrows:
            if arr.source is self.target or arr.target is self.target:
                arr.number = count
                count += 1
        n3 = model.processes.index(self)
        model.processes.pop(n3)
        
    def getsetLocation(self):
        source_x = self.source.rect.x()+self.source.rect.width()/2
        source_y = self.source.rect.y()+self.source.rect.height()/2
        target_x = self.target.rect.x()+self.target.rect.width()/2
        target_y = self.target.rect.y()+self.target.rect.height()/2
        p1 = QtCore.QPointF(source_x, source_y)
        p2 = QtCore.QPointF(target_x, target_y)
        
        diff = p2 - p1  # just make arrow shorter....
        correction = abs(diff.x() / math.sqrt(diff.x()*diff.x() + diff.y()*diff.y()))  # uzaleznij odjecie od kata...
        to_substr = (40 * correction + 28) * diff / math.sqrt(diff.x()*diff.x() + diff.y()*diff.y())
        p1 = p1 + to_substr
        p2 = p2 - to_substr
        
        diff = p1 - p2
        difflen = math.sqrt(diff.x()**2 + diff.y()**2)
        # next If: move second, third, ... arrow a little bit to avoid overlap
        if self.number > 1 and difflen != 0.0:
            sinkat = diff.y() / difflen
            coskat = diff.x() / difflen
            if sinkat < 0:
                alittle = QtCore.QPointF(sinkat * self.displacement,
                                         -coskat * self.displacement)
            elif sinkat > 0:
                alittle = QtCore.QPointF(-sinkat * self.displacement,
                                         coskat * self.displacement)
            elif coskat > 0:
                alittle = QtCore.QPointF(sinkat * self.displacement,
                                         -coskat * self.displacement)
            else:
                alittle = QtCore.QPointF(-sinkat * self.displacement,
                                         coskat * self.displacement)
            
            p1 += (-1)**self.number * alittle * math.floor(self.number / 2.0)
            p2 += (-1)**self.number * alittle * math.floor(self.number / 2.0)
        
        self.p1 = p1
        self.p2 = p2
        return p1, p2
        
    def contains(self, point):
        try:
            a_p1p2 = float(self.p2.y() - self.p1.y()) / float(self.p2.x() - self.p1.x())  # find linear eq for p1 and p2
            b_p1p2 = float(self.p1.y()) - a_p1p2 * float(self.p1.x())
            
            a_point = -1 / a_p1p2  # find linear eq for point which is perpendicular to p1p2
            b_point = float(point.y()) - a_point * float(point.x())
            
            x_cross = (b_point - b_p1p2) / (a_p1p2 - a_point)  # find crossing point
            y_cross = a_p1p2 * x_cross + b_p1p2
            
            if self.p1.x() >= self.p2.x():  # check if crossing point is between p1 and p2
                if self.p1.x() >= x_cross >= self.p2.x():
                    cond1 = True
                else:
                    cond1 = False
            else:
                if self.p2.x() >= x_cross >= self.p1.x():
                    cond1 = True
                else:
                    cond1 = False  
                    
            if self.p1.y() >= self.p2.y():  # check if crossing point is between p1 and p2
                if self.p1.y() >= y_cross >= self.p2.y():
                    cond2 = True
                else:
                    cond2 = False
            else:
                if self.p2.y() >= y_cross >= self.p1.y():
                    cond2 = True
                else:
                    cond2 = False
            
            dist = math.sqrt(math.pow(float(point.x()) - x_cross, 2) +
                             math.pow(float(point.y()) - y_cross, 2))
            if dist <= self.dist_treshold:
                cond3 = True
            else:
                cond3 = False
            
            return cond1 and cond2 and cond3
        
        except:
            return False

    def paintYourself(self, painter):
        p1, p2 = self.getsetLocation()
        
        # firstly draw sinusiodal shape indicating nonradiative process
        fragm_len = 3.0
        modamp = 5.0  # depth of modulation
        diff = p2 - p1
        full_length = math.sqrt(diff.x()*diff.x() + diff.y()*diff.y())
        iters = math.floor(full_length / fragm_len)
        unit_vect = diff * fragm_len / full_length  # piece of line used to render whole curve
        if diff.x() >= 0:
            angle = math.asin(diff.y() / math.sqrt(diff.x()*diff.x() + diff.y()*diff.y()))
        else:
            angle = math.pi - math.asin(diff.y() / math.sqrt(diff.x()*diff.x() + diff.y()*diff.y()))
        perp_vect = QtCore.QPointF(modamp*math.cos(angle+math.pi/2), modamp*math.sin(angle+math.pi/2))
        
        path = QtGui.QPainterPath(p1)
        
        for i in range(1, iters+1):
            path.lineTo(p1 + unit_vect * i + perp_vect * math.sin(i * math.pi/iters) * math.sin(i * 1))
        
        path.lineTo(p2)
        painter.drawPath(path)
            
        diff = p1 - p2
        if diff.x() >= 0:
            angle = math.asin(diff.y() / math.sqrt(diff.x()*diff.x() + diff.y()*diff.y()))
        else:
            angle = math.pi - math.asin(diff.y() / math.sqrt(diff.x()*diff.x() + diff.y()*diff.y()))
        
        angle_diff = math.pi / 10.0  # determines shape of the arrow
        length = 10.0  # determines shape of the arrow
        p_arr1 = QtCore.QPointF(length*math.cos(angle+angle_diff),
                                length*math.sin(angle+angle_diff))
        p_arr2 = QtCore.QPointF(length*math.cos(angle-angle_diff),
                                length*math.sin(angle-angle_diff))
        
        painter.drawLine(p2, p2 + p_arr1)
        painter.drawLine(p2, p2 + p_arr2)


class ParamControl(QWidget):
    def __init__(self, parent, name, callback):
        super().__init__(parent)
        
        self.name = name
        self.isactive = None  # indicates if you can edit text and fix
        self.isenabled = None  # indicates if you can activate text to be active

        self.label = QtWidgets.QLabel(name, self)
        self.text = QtWidgets.QLineEdit('0.0', self)
        self.check_editable = QtWidgets.QCheckBox("Modify?", self)
        self.check_fixed = QtWidgets.QCheckBox("Fixed?", self)
        self.check_editable.clicked.connect(callback)
        self.check_fixed.clicked.connect(callback)
        self.text.editingFinished.connect(callback)
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.text)
        self.layout.addWidget(self.check_editable)
        self.layout.addWidget(self.check_fixed)
        
        self.setActive(False)
        self.setEnabled(False)
        
    def loadState(self, enabled, active, fixed, value):
        self.text.setText("%.6f" % value)
        self.setActive(active)
        self.setEnabled(enabled)
        self.check_fixed.setChecked(fixed)
        if active and not enabled:
            raise Exception("Algorithm failure, id 1007!")
            
    def isActive(self):  # asks for sate of checkbox, but does not update itself
        return self.check_editable.isChecked()
    
    def setEnabled(self, setenabled):  # does things only if state is really changed
        if setenabled != self.isenabled:
            if setenabled:
                self.check_editable.setEnabled(True)
                # self.setActive(True)
                self.isenabled = True
            else:
                self.check_editable.setChecked(False)
                self.check_editable.setEnabled(False)
                self.setActive(False)
                self.isenabled = False
    
    def setActive(self, setactive):
        if setactive != self.isactive:
            if setactive:
                self.check_editable.setChecked(True)
                self.check_fixed.setEnabled(True)
                self.text.setEnabled(True)
                self.isactive = True
            else:
                self.check_editable.setChecked(False)
                self.check_fixed.setChecked(False)
                self.check_fixed.setEnabled(False)
                self.text.setEnabled(False)  
                self.isactive = False


class VLayout(QVBoxLayout):
    def __init__(self, parent=None):
        super(VLayout, self).__init__(parent)
        self.added_widgets = []

    def addSeveralWidgets(self, lista):
        for i in lista:
            if type(i) is list:
                self.addWidget(i[0], *i[1:])
            else:
                self.addWidget(i)

    def addWidget(self, a0, *args):
        super().addWidget(a0, *args)
        self.added_widgets.append(a0)

    def hide_all(self):
        for i in self.added_widgets:
            i.setVisible(False)

    def show_all(self):
        for i in self.added_widgets:
            i.show()


class ModelWindow(QWidget):

    def __init__(self, model_ref=None, call_back=None, app=None, parent=None):
        super().__init__(parent)
        self.app = app
        if model_ref is None:
            self.model = Model()
        else:
            self.model = model_ref
        self.call_back = call_back
        self.title = 'Model Editor'
        self.left = 10
        self.top = 35
        self.width = 800
        self.height = 700
        self.layout = VLayout(self)
        # self._bottom = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self._middle = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self._middle.splitterMoved.connect(self.splitterMoved)
        self._middle.setStretchFactor(0, 1)
        self._middle.show()
        self.layout.addWidget(self._middle)
        # self.layout.addWidget(self._bottom)
        # self._bottom.addWidget(self._middle)
        # self._bottom.setStretchFactor(0, 1)
        msg ="Generate parameters and close"
        bottom_layout = QHBoxLayout(self)
        bottom_layout.setContentsMargins(500, 0, 0, 0)
        self.layout.addLayout(bottom_layout)
        self.generate_params_button = QtWidgets.QPushButton(msg, self)
        self.generate_params_button.setMaximumHeight(35)
        self.close_button = QtWidgets.QPushButton("Close window", self)
        self.close_button.setMaximumHeight(35)
        self.load_model = QtWidgets.QPushButton("Load model", self)
        self.load_model.setMaximumHeight(35)
        self.Save_model = QtWidgets.QPushButton("Save model", self)
        self.Save_model.setMaximumHeight(35)
        bottom_layout.addWidget(self.Save_model)
        bottom_layout.addWidget(self.load_model)
        bottom_layout.addWidget(self.close_button)
        bottom_layout.addWidget(self.generate_params_button)
        self.close_button.clicked.connect(self.close)
        self.generate_params_button.clicked.connect(self.genParametersAndClose)
        self.load_model.clicked.connect(self.loadModel)
        self.Save_model.clicked.connect(self.saveModel)
        # TODO conect close load/save model and generate parameters

        self.main_widget = QtWidgets.QFrame(self)
        self.main_widget.setMinimumWidth(180)
        self.main_widget.setLayout(VLayout(self))
        self.main_widget.layout().setAlignment(QtCore.Qt.AlignTop)
        self.paint_widget = QtWidgets.QGroupBox("Designed model")
        self.paint_widget.setMinimumWidth(400)
        self.paint_widget.setStyleSheet("border:2px solid %s" % "blue")
        # self.paint_widget.setStyleSheet("background-color: %s" % 'darkGray')
#        p = self.paint_widget.palette()
#        p.setColor(self.paint_widget.backgroundRole(), QtCore.Qt.red)
#        self.paint_widget.show()
#        self.paint_widget.setPalette(p)

        # color order maybe use HSV system instead???
        self.colors = (QtCore.Qt.black, QtCore.Qt.red,
                       QtCore.Qt.green, QtCore.Qt.blue,
                       QtCore.Qt.magenta, QtCore.Qt.darkRed,
                       QtCore.Qt.darkGreen, QtCore.Qt.darkBlue,
                       QtCore.Qt.darkCyan, QtCore.Qt.darkMagenta,
                       QtCore.Qt.darkYellow, QtCore.Qt.darkGray)

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.show()

        self.mouse_position = None  # mouse position from las mouse move event

        # Create some widgets to be placed inside
        self.label1 = QtWidgets.QLabel('Population name:', self.main_widget)
        self.text1 = QtWidgets.QLineEdit('', self.main_widget)
        self.add_population_button = QtWidgets.QPushButton('Add population',
                                                           self.main_widget)
        self.add_population_button.clicked.connect(self.button1Func)
        self.add_process_button = QtWidgets.QPushButton('Add process',
                                                        self.main_widget)
        self.add_process_button.clicked.connect(self.add_process_function)
        self.label2 = QtWidgets.QLabel('', self.main_widget)

        self.main_widget.layout().addWidget(self.label1)
        self.main_widget.layout().addWidget(self.text1)
        self.main_widget.layout().addWidget(self.add_population_button)
        self.main_widget.layout().addWidget(self.add_process_button)
        self.main_widget.layout().addWidget(self.label2)
        self.main_widget.show()
        self._middle.addWidget(self.main_widget)
        self._middle.addWidget(self.paint_widget)
        self._middle.setCollapsible(1, False)

        self.label1.show()
        self.text1.show()
        self.add_population_button.show()
        self.add_process_button.show()

        # install the event filter
        self.installEventFilter(self)
        self.setMouseTracking(True)

        self.mousepressed = False  # indicates which population is being moved, or False if none
        self.mouse_dx = 0  # markers of relative mouse move from the moment of click
        self.mouse_dy = 0
        self.ref_mouse = QtCore.QPoint(0, 0)  # marker of mouse position when population was clicked

        self.pop_edit = False  # to edit populatiions

        #Population widgets
        self.population_widget = QtWidgets.QFrame(self)
        self.population_layout = VLayout(self)
        self.population_widget.setLayout(self.population_layout)
        self.population_widget.layout().setAlignment(QtCore.Qt.AlignTop)

        self.button3 = QtWidgets.QPushButton('Done!', self.population_widget)
        self.button3.clicked.connect(self.population_done)
        self.button31 = QtWidgets.QPushButton('Delete', self.population_widget)
        self.button31.clicked.connect(self.button31Func)

        self.c_edit = ParamControl(self.population_widget, "Initial population",
                                   self.populationEditFunc)
        self.tau_edit = ParamControl(self.population_widget, "Time constant",
                                     self.populationEditFunc)
        self.k_all_edit = ParamControl(self.population_widget, "Rate constant",
                                       self.populationEditFunc)
        self.population_widget.layout().addWidget(self.c_edit)
        self.population_widget.layout().addWidget(self.tau_edit)
        self.population_widget.layout().addWidget(self.k_all_edit)
        self.population_widget.layout().addWidget(self.button3)
        self.population_widget.layout().addWidget(self.button31)
        self.main_widget.layout().addWidget(self.population_widget)

        # process widgets
        self.proc_edit = False  # to edit processes
        self.process_widget = QtWidgets.QFrame(self)
        self.process_layout = VLayout(self)
        self.process_widget.setLayout(self.process_layout)
        self.process_widget.layout().setAlignment(QtCore.Qt.AlignTop)
        self.process_done_button = QtWidgets.QPushButton('Done!',
                                                         self.process_widget)
        self.process_done_button.clicked.connect(self.process_done)
        self.delete_process_button = QtWidgets.QPushButton('Delete!',
                                                           self.process_widget)
        self.delete_process_button.clicked.connect(self.delete_process)

        self.k_edit = ParamControl(self.process_widget, "Rate constant",
                                   self.processEditFunc)
        self.sf_edit = ParamControl(self.process_widget, "Splitting factor",
                                    self.processEditFunc)

        self.process_widget.layout().addWidget(self.k_edit)
        self.process_widget.layout().addWidget(self.sf_edit)
        self.process_widget.layout().addWidget(self.process_done_button)
        self.process_widget.layout().addWidget(self.delete_process_button)
        self.main_widget.layout().addWidget(self.process_widget)

        # self.sf_edit.text.setValidator(QtGui.QDoubleValidator(0, 1, 9,self.sf_edit.text)) #TODO
        # TODO: add validators so sf can be higher than 0 and lower than 1 (or 1 if only one arrow), and k can be only below k_all?

        self.process_adding = False  # indicates that process arrow is being added (select populations)

    def splitterMoved(self):
        for population in self.model.populations:
            minimum_x = self.paint_widget.pos().x()
            if population.rect.x() - 60 < minimum_x:
                population.rect.setX(minimum_x + 60)
                population.rect.setWidth(population.rect_w)
                self.repaint()

    def loadModel(self):
        try:
            path = QtWidgets.QFileDialog.getOpenFileName(self,
                                                         'Load kinetic model',
                                                         'c:\\',
                                                         "files (*.model)")
            self.model = Model.load(path[0])
            self.repaint()
        except Exception as e:
            error_dialog = QtWidgets.QMessageBox(self)
            error_dialog.setText(f'Unable to load the model, the'
                                 f' following error occur: \n{e}')
            error_dialog.exec()

    def saveModel(self):
        try:
            save_file_name = QtWidgets.QFileDialog.getSaveFileName(self,
                                                                   'Save Model',
                                                                   'Model')
            name = save_file_name[0].rsplit(".", 1)[0]+".model"
            self.model.save(name)
            print('Model saved')
        except Exception as e:
            error_dialog = QtWidgets.QMessageBox(self)
            error_dialog.setText(f'Unable to save the model, the'
                                 f' following error occur: \n{e}')
            error_dialog.exec()

    def genParametersAndClose(self):
        try:
            self.model.genParameters()
            print('Model created')
            if self.call_back is not None:
                self.call_back()
            self.close()
        except Exception as e:
            error_dialog = QtWidgets.QMessageBox(self)
            error_dialog.setText(f'Unable to generate parameters, the'
                                 f' following error occur: \n{e}')
            error_dialog.exec()

    def populationEditFunc(self):
        if type(self.pop_edit) != bool:
            self.pop_edit.updateState(self.c_edit,
                                      self.tau_edit,
                                      self.k_all_edit)

    def processEditFunc(self):
        if type(self.proc_edit) != bool:
            self.proc_edit.updateState(self.k_edit, self.sf_edit)

    def button1Func(self):  # creates new population
        found = False  # ensure that new name is unique
        for elem in self.model.populations:
            if elem.name == self.text1.text():
                found = True

        for elem in self.model.processes:
            if elem.name == self.text1.text():
                found = True

        if not(isIdentifier(self.text1.text())):  # it has to be a valid python id
            found = True

        if not found and len(self.text1.text()) > 0:
            tmp_pop = ModPopulation(self.text1.text())
            tmp_pop.color = self.colors[random.randint(0, len(self.colors)-1)]
            if len(self.model.populations) == 0:
                tmp_pop.c = 1
            self.model.addPopulation(tmp_pop)
            tmp_pop.enableDisableKs()
            self.text1.setText('')
            self.repaint()

    def population_done(self):
        self.pop_edit = False
        self.repaint()

    def button31Func(self):  # deletes population if possible
        if len(self.pop_edit.arrows) == 0:
            self.pop_edit.remove(self.model)
            self.pop_edit = False
        self.repaint()

    def process_done(self):  # finished mały miejski ul pszczołyedition of process
        # if self.isStrNumber(self.text_proc.text()):
            # self.proc_edit.k = float(self.text_proc.text())
        self.proc_edit = False
            # self.text_proc.setText('')
        self.repaint()

    def delete_process(self):  # deletes arrow
        self.proc_edit.remove(self.model)
        self.proc_edit = False
        # self.text_proc.setText('')
        self.repaint()

    def add_process_function(self):  # process and starts selection of connected populations
        self.process_adding = True
        self.repaint()

    def countArrows(self, population1, population2):  # gives numer of the existing arrows between populations, and True if some k arrow already exist
        arrows = 0
        for arr in population1.arrows:
            condition1 = arr.source is population1 and arr.target is population2
            condition2 = arr.source is population2 and arr.target is population1
            if condition1 or condition2:
                arrows += 1
        return arrows

    def isStrNumber(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def paintEvent(self, event):
        if self.pop_edit == False:  # population edit menu
            self.population_layout.hide_all()
            self.label2.setVisible(False)

        else:
            self.label2.show()
            self.population_layout.show_all()

        if self.proc_edit == False:
            # check proc_edit is On or Off
            if self.pop_edit == False:
                self.label2.setVisible(False)
            else:
                self.label2.show()
            self.process_layout.hide_all()

        else:
            self.label2.show()
            self.process_layout.show_all()

        if self.process_adding != False:
            self.add_process_button.setText("Select boxes!")
        else:
            self.add_process_button.setText("Add process")

        painter = QtGui.QPainter(self)
        painter.begin(self)

        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        # painter.setWorldMatrixEnabled(False)

        rmpen = QtGui.QPen()
        rmpen.setWidth(3)
        # colorum = QtCore.Qt.red
        # if self.process_adding != False:
        #     colorum = QtCore.Qt.yellow
        marg = 5

        for r in self.model.populations:  # should also call paintYourself function of the population, just like in case of processss
            rmpen.setBrush(r.color)
            rmpen.setWidth(3)
            if self.mousepressed is r:
                tmprect = copy.deepcopy(r.rect)
                tmprect.setX(tmprect.x() + self.mouse_dx)
                tmprect.setY(tmprect.y() + self.mouse_dy)
                tmprect.setWidth(r.rect_w)
                tmprect.setHeight(r.rect_h)
            else:
                tmprect = r.rect
                if self.mouse_position is not None:
                    if r.rect.contains(self.mouse_position):
                        rmpen.setWidth(5)
            painter.setPen(rmpen)
            painter.drawRoundedRect(tmprect, 10, 10)
            painter.drawText(tmprect.x()+marg,
                             tmprect.y()+marg,
                             tmprect.width()-2*marg,
                             tmprect.height()-2*marg,
                             QtCore.Qt.AlignCenter, r.name)

        for r in self.model.processes:
            rmpen.setWidth(3)
            rmpen.setBrush(r.color)
            if self.mouse_position is not None:
                if r.contains(self.mouse_position):
                    rmpen.setWidth(5)
            painter.setPen(rmpen)
            r.paintYourself(painter)

        painter.end()

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.MouseMove:
            self.mouse_position = event.pos()
            if self.mousepressed != False:

                # control of X position dont go out of window
                small = self.mouse_position.x() > self.paint_widget.pos().x()+40
                size = self.paint_widget.width() + self.paint_widget.pos().x()
                big = self.mouse_position.x() < size
                if small and big:
                    self.mouse_dx = self.mouse_position.x() - self.ref_mouse.x()
                else:
                    if big:
                        self.mouse_dx = self.paint_widget.pos().x()+60 - \
                                        self.ref_mouse.x()
                    else:
                        self.mouse_dx = size - self.ref_mouse.x() - 40

                # control of Y position dont go out of window
                small = self.mouse_position.y() > 50
                size = self.paint_widget.height() + self.paint_widget.pos().y()
                big = self.mouse_position.y() < size
                if small and big:
                    self.mouse_dy = self.mouse_position.y() - self.ref_mouse.y()
                else:
                    if big:
                        self.mouse_dy = 60 - self.ref_mouse.y()
                    else:
                        self.mouse_dy = size - self.ref_mouse.y() - 20
            self.repaint()

        elif event.type() == QtCore.QEvent.MouseButtonRelease:
            if self.mousepressed != False:
                self.mousepressed.rect.setX(self.mousepressed.rect.x() +
                                            self.mouse_dx)
                self.mousepressed.rect.setY(self.mousepressed.rect.y() +
                                            self.mouse_dy)
                self.mousepressed.rect.setWidth(self.mousepressed.rect_w)
                self.mousepressed.rect.setHeight(self.mousepressed.rect_h)
                self.mouse_dx = 0
                self.mouse_dy = 0
                self.mousepressed = False
                self.repaint()
        elif event.type() == QtCore.QEvent.MouseButtonPress:
            for r in self.model.populations:
                if r.rect.contains(event.pos()):
                    self.mousepressed = r
                    self.ref_mouse = event.pos()
                    break

            if self.process_adding != True and self.process_adding != False:  # finalize arrow adding process
                for r in self.model.populations:
                    found = False  # if user clicked something useful
                    if r.rect.contains(event.pos()):
                        if not(r is self.process_adding):
                            found = r
                            break
                if found == False:  # Don't modify this to: if not found
                    self.process_adding = False
                else:  # finalize arrow from  self.process_adding to found
                    arrow_count = self.countArrows(self.process_adding, found)
                    # if self.select_process.currentIndex() == 0: # Thermal
                    #     tmp_arrow = ModThermal(self.text5.text(), self.process_adding, found)
                    # elif self.select_process.currentIndex() == 1: # Light activated
                    #     tmp_arrow = ModRadiative(self.text5.text(), self.process_adding, found) #asumes, that there are only 2 kinds of arrows, which can be added
                    already_exists = False  # check if any arrow connecting these populations exists, if yes then dont create new one
                    for arr in self.process_adding.arrows:
                        if arr.source is found:
                            already_exists = True
                        if arr.target is found:
                            already_exists = True
                    if not already_exists:
                        tmp_arrow = ModProcess(self.process_adding,
                                               found)
                        tmp_arrow.number = arrow_count + 1
                        tmp_arrow.color = self.colors[random.randint(0, len(self.colors)-1)]
                        self.model.addProcess(tmp_arrow)
                        tmp_arrow.source.enableDisableKs()
                        tmp_arrow.source.updateBranchKs()
                        # self.text5.setText('')
                    self.process_adding = False
            if self.process_adding == True:  # process arrow adding, first population needs to be selected
                for r in self.model.populations:
                    found = False  # if user clicked something useful
                    if r.rect.contains(event.pos()):
                        found = True
                        self.process_adding = r
                        break
                if found == False:
                    self.process_adding = False

            self.repaint()

        elif event.type() == QtCore.QEvent.MouseButtonDblClick:
            # print(f"double clicked {self.proc_edit}, {self.pop_edit}")
            # the following loop and  if statements allow to open from one
            # object to the next one without clicking in done
            objetc_click = None
            process, population = False, False
            # check if click on an arrow process
            for p in self.model.processes:
                if p.contains(event.pos()):
                    process = True
                    objetc_click = p
                    break
            # check if click on a population box
            for r in self.model.populations:
                if r.rect.contains(event.pos()):
                    population = True
                    objetc_click = r
                    break
            if process or population:
                self.process_done()
                self.population_done()
            # print(f"population: {population}; process: {process}")
            if self.proc_edit == False and self.pop_edit == False:  # should edit process?
                if process:
                    self.proc_edit = objetc_click
                    self.label2.setText('Edit arrow ' + p.name + ' :')
                    self.k_edit.loadState(p.k_enabled, p.k_active,
                                          p.k_fixed, p.k)
                    self.sf_edit.loadState(p.sf_enabled, p.sf_active,
                                           p.sf_fixed, p.sf)
                    # self.text_proc.setText(str(p.k))
                    self.repaint()
            if self.pop_edit == False and self.proc_edit == False:  # should edit population?
                if population:
                    self.pop_edit = objetc_click
                    self.label2.setText('Edit population ' + r.name + ' :')
                    self.c_edit.loadState(r.c_enabled,
                                          r.c_active,
                                          r.c_fixed,
                                          r.c)

                    self.k_all_edit.loadState(r.k_all_enabled,
                                              r.k_all_active,
                                              r.k_all_fixed,
                                              r.k_all)

                    self.tau_edit.loadState(r.tau_enabled,
                                            r.tau_active,
                                            r.tau_fixed,
                                            r.tau)
                    # self.eps_table.setRowCount(len(r.epsilon))
                    # ct_tmp = 0
                    # for k, v in r.epsilon.items():
                    #     tmp1 = QtWidgets.QTableWidgetItem(str(k))
                    #     tmp2 = QtWidgets.QTableWidgetItem(str(v))
                    #     #set flags?
                    #     self.eps_table.setItem(ct_tmp,0,tmp1)
                    #     self.eps_table.setItem(ct_tmp,1,tmp2)
                    #     ct_tmp += 1
                    self.repaint()

        return False


class Model:
    def __init__(self):
        self.populations = list()
        self.processes = list()
        self.psplit = False

    def addPopulation(self, new_population):
        self.populations.append(new_population)

    def addProcess(self, new_process):
        self.processes.append(new_process)

    def manualModelBuild(self, call_back=None, app=None):
        return ModelWindow(self, app=app)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):  # this is static method intentionally
        with open(filename, "rb") as f:
            loaded = pickle.load(f)
        return loaded

#    def setKmatrix(self, paths): #array of (source, destination, rate, varied)
#
#        sources = ["" for i in range(self.exp_no)]
#        for i in paths:
#            source = i[0]
#            destination = i[1]
#            rate = i[2]
#            varied = i[3]
#            if(source != destination):
#                self.params['k_%i%i' % (destination,source)].set(rate, vary=varied)
#                sources[source-1] += '-k_%i%i' % (destination,source)
#            else:
#                self.params['k_%i%i' % (destination,source)].set(-rate, vary=varied) #if destination == source, it means that this is the terminal component...
#
#        for i in range(self.exp_no):
#            if(len(sources[i]) > 0):
#                self.params['k_%i%i' % (i+1,i+1)].set(expr=sources[i])

    def genParameters(self):  # parameters are fixed by default, unfix some of them before fitting procedure
        self.params = lmfit.Parameters()
        # ASSUMES THAT Splitting Factors are not zero or 1!!!
        self.params.add('exp_no', value=len(self.populations), vary=False)
        if len(self.populations) < 1:
            raise Exception("Algorithm failure, because model is empty!")

        for i in range(len(self.populations)):
            self.params.add('c_%i' % (i+1), self.populations[i].c, False,
                            0, None, None, None)  # concentrations cannot be < 0
            for j in range(len(self.populations)):
                self.params.add('k_%i%i' % (i+1, j+1), 0, False,
                                None, None, None, None)

        for i in range(len(self.populations)):
            popul = self.populations[i]
            source = i
            if popul.countOutgoingArrows() == 0:
                vary = not(popul.k_all_fixed or popul.tau_fixed)
                self.params['k_%i%i' % (source+1, source+1)].set(-popul.k_all,
                                                                 vary=vary)
            else:
                # NOW YOU ARE GOING INTO BRANCH
                k_all_fixed = False
                total_sfs = 0.0
                if popul.k_all_fixed or popul.tau_fixed:  # if k_all fixed then do it
                    self.params['k_%i%i' % (source+1, source+1)].set(-popul.k_all,
                                                                     vary=False)
                    k_all_fixed = True  # name should be done not fixed, but nevermind

                for arr in popul.arrows:
                    arr.done = False  # indicator if arrow remains to be done
                    if arr.source is popul:
                        target = self.populations.index(arr.target)
                        if arr.k_fixed and arr.sf_fixed and not k_all_fixed:  # if k_all can be fixed based on arrow then do it
                            self.params['k_%i%i' % (source+1, source+1)].set(expr="-"+'k_%i%i' % (target+1,source+1)+"/"+str(arr.sf))
                            self.params['k_%i%i' % (target+1, source+1)].set(arr.k, vary=False)
                            k_all_fixed = True
                            total_sfs += arr.sf
                            arr.done = True
                        elif arr.k_fixed:  # fix all direclty fixed k's
                            self.params['k_%i%i' % (target+1, source+1)].set(arr.k, vary=False)
                            arr.done = True
                        elif arr.sf_fixed:  # set proper expression for defined sf's
                            expr = "-"+'k_%i%i' % (source+1, source+1)+"*"+str(arr.sf)
                            self.params['k_%i%i' % (target+1, source+1)].set(expr=expr)
                            total_sfs += arr.sf
                            arr.done = True

                freedom = popul.countOutgoingArrows()-popul.countFixed()  # how many free parameters will be
                if not k_all_fixed and freedom > 0:  # start vary from k_all
                    self.params['k_%i%i' % (source+1, source+1)].set(-popul.k_all, vary=True)
                    k_all_fixed = True  # means that now you can put this into eq
                    freedom -= 1

                # go egein over branch and set things to vary
                for arr in popul.arrows:
                    if arr.source is popul and not arr.done:
                        target = self.populations.index(arr.target)
                        if freedom > 0:
                            self.params['k_%i%i' % (target+1, source+1)].set(arr.k, vary=True)
                            arr.done = True
                            freedom -= 1
                if freedom > 0:
                    raise Exception("Algorithm failure, id 1002!")

                expression = "("  # sum all k's which are fixed directly or varied
                count = 0
                for arr in popul.arrows:
                    if arr.source is popul and arr.done and not arr.sf_fixed:
                        target = self.populations.index(arr.target)
                        expression += '-k_%i%i' % (target+1, source+1)
                        count += 1
                expression += ")"
                if total_sfs == 1:
                    raise Exception("Algorithm failure, id 1003!")

                if not k_all_fixed:  # set proper eq for k_all based only on not-sf arrows
                    if count == 0:
                        raise Exception("Algorithm failure, id 1005!")
                    expression += "/(1-" + str(total_sfs) + ")"
                    self.params['k_%i%i' % (source+1, source+1)].set(expr=expression)
                    k_all_fixed = True

                else:  # set eq's for last k, assumes that k_all is fixed
                    expr_begining = '-k_%i%i' % (source+1, source+1) + \
                                    "*(1-"+str(total_sfs) + ")"
                    if count == 0:
                        expression = expr_begining
                    else:
                        expression = expr_begining + "+" + expression
                    count2 = 0
                    for arr in popul.arrows:  # go egein over branch and set remaining expr's (only one can be found)
                        if arr.source is popul and not arr.done and not arr.sf_fixed:
                            target = self.populations.index(arr.target)
                            self.params['k_%i%i' % (target+1, source+1)].set(expr=expression)
                            arr.done = True
                            count2 += 1
                    if count2 > 1:
                        raise Exception("Algorithm failure, id 1004!")

                # just check, probbaly not needed, but i am too tired to ensure that
                for arr in popul.arrows:
                    if arr.source is popul:
                        if arr.done == False:
                            raise Exception("Algorithm failure, id 1006!")
        # add the names to the params
        self.params.model_names = [i.name for i in self.populations]
        return self.params

#    def updateParameters(self, params):#updates values of existing parameters. does not add new values, does not modify model structure
#        p = params.valuesdict()
#        for elem in self.populations:
#
#            if(self.psplit == False):
#                elem.initial = p[elem.name]
#            else:
#                for num in range(len(elem.initial)):
#                    elem.initial[num] = p['_' + str(num) + '_' + elem.name]
#
#            for l, eps in elem.epsilon.items():
#                elem.epsilon[l] = p[elem.name + '__' + str(l).replace('.','_')]
#
#        for elem in self.processes:
#            if elem.type == 'fi':
#                elem.fi = p[elem.name + '__fi']
#            elif elem.type == 'k':
#                elem.k = p[elem.name + '__k']

    def checkParams(self, experiment):  # run tests if params are correctly set. experiment object is to validiate its compatibility with model
        result = True                   # it should be run after updataParameers for both model and experiment, and assume that funcs loaded them correctly

        return result


# app = QApplication(sys.argv)
# model1 = Model()
# ex = model1.manualModelBuild()
# ex.show()
# app.exec()

# model1 = Model()
# model1.manualModelBuild()
