#run to install scipy: conda install -c anaconda scipy
#and this: conda install -c conda-forge matplotlib
#and this: conda install -c conda-forge lmfit

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import lmfit
import random
import sys
import keyword
import copy
import math
import pickle
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import QtCore, QtGui, QtWidgets, Qt
#from kineticdata import Experiment 

#firstly create populations and put them i model
#then add arrows. Create arrow binded by two populations and add to model list
#arrow should add itself to both end-populations
#remove arrow by calling kill() in this arrow and removing from model list
#arrow should remove itself from both end-populations
#if you kill population, firstly remove all arrows inside



def isIdentifier(identifier): #check if string is valid python identifier
    if not(isinstance(identifier, str)):
        return False
    if not(identifier.isidentifier()):
        return False
    if keyword.iskeyword(identifier):
        return False
    return True

class ModPopulation:
    def __init__(self, new_name):
        self.arrows = list() #list of processes associated with this population
        self.name = new_name #name of the population
        #self.epsilon = dict() # define epsilons for irr and probe wavelengths
        self.initial = 0.0
        
        self.color = QtCore.Qt.black
        self.rect_w = 120
        self.rect_h = 40
        self.rect = QtCore.QRect(150, 20, self.rect_w, self.rect_h)
        
        self.c = 0
        self.k_all = 1 #both used depending on mode selected
        self.tau = 1
        self.c_fixed = True
        self.k_all_fixed = False
        self.tau_fixed = False
        self.c_active = True
        self.k_all_active = False #active means editable by user
        self.tau_active = False
        self.c_enabled = True
        self.k_all_enabled = False #enabled means that user is allowed to turn it active
        self.tau_enabled = False #if enabled is False, then active also must be False        
        
    def remove(self, model): #before calling ensure that all arrows with this population are removed, in other case there will be exception!
        if len(self.arrows) != 0:
            raise Exception('Attempted to invalid population removal!!')
        n = model.populations.index(self)
        model.populations.pop(n)
        
    def countActive(self): #count active arrows plus this population
        active_counter = 0
        
        if(self.k_all_active): active_counter += 1
        if(self.tau_active): active_counter += 1            
        
        for arrow in self.arrows:
            if(arrow.source is self):
                if(arrow.k_active): active_counter += 1 
                if(arrow.sf_active): active_counter += 1                     
                    
        return active_counter
    
    def countFixed(self):
        """
        Count fixed parameters
        """
        fixed_counter = 0
        
        if(self.k_all_fixed): fixed_counter += 1
        elif(self.tau_fixed): fixed_counter += 1            
        
        for arrow in self.arrows:
            if(arrow.source is self):
                if(arrow.k_fixed): fixed_counter += 1 
                if(arrow.sf_fixed): fixed_counter += 1                     
                    
        return fixed_counter    
    
    def countActiveSFs(self): #count active Scalling Factor fields
        active_counter = 0        
        for arrow in self.arrows:
            if(arrow.source is self):
                if(arrow.sf_active): active_counter += 1                     
                    
        return active_counter    
    
    def countActiveSFsValues(self): #count active Scalling Factor fields
        sfall = 0.0
        for arrow in self.arrows:
            if(arrow.source is self):
                if(arrow.sf_active): sfall += arrow.sf               
                    
        return sfall
    
    def countOutgoingArrows(self): #count all arrows going out from this population
        arrow_counter = 0
        
        for arrow in self.arrows:
            if(arrow.source is self):
                arrow_counter += 1
        
        return arrow_counter
 
    def enableDisableKs(self): #rewrite enable/disable states in this branch
        active_counter = self.countActive()
        active_SF_counter = self.countActiveSFs()        
        arrow_counter = self.countOutgoingArrows()
        
        if(arrow_counter == 0):
            self.k_all_enabled = True if not(self.tau_active) else False
            self.tau_enabled = True if not(self.k_all_active) else False  
            
        elif active_counter < arrow_counter: #enable everything which is inactive and can be enabled
            double_present = False #if in some arrow two fields are active, it determines k_all value. 
            for arrow in self.arrows:
                if arrow.source is self:
                    if arrow.k_active and arrow.sf_active:
                        double_present = True

            if double_present:
                self.k_all_enabled = False
                self.tau_enabled = False
                for arrow in self.arrows:
                    if arrow.source is self:
                        arrow.k_enabled = True if not(arrow.sf_active) else False
                        arrow.sf_enabled = True if (not(arrow.k_active) and active_SF_counter < arrow_counter-1) or arrow.sf_active else False                
                
            else:
                self.k_all_enabled = True if not(self.tau_active) else False
                self.tau_enabled = True if not(self.k_all_active) else False
                for arrow in self.arrows:
                    if(arrow.source is self):
                        if(not(self.k_all_active) and not(self.tau_active)): 
                            arrow.k_enabled = True
                            arrow.sf_enabled = True if active_SF_counter < arrow_counter-1 or arrow.sf_active else False  
                        else:
                            arrow.k_enabled = True if not(arrow.sf_active) else False
                            arrow.sf_enabled = True if (not(arrow.k_active) and active_SF_counter < arrow_counter-1) or arrow.sf_active else False 
                            
        else: #disable everything which is inactive
            for arrow in self.arrows:
                if(arrow.source is self):
                    if(not(arrow.k_active)):
                        arrow.k_enabled = False
                    if(not(arrow.sf_active)):
                        arrow.sf_enabled = False                        
                        
            if(not(self.k_all_active)):  
                self.k_all_enabled = False
            if(not(self.tau_active)):  
                self.tau_enabled = False              
                
    def updateBranchKs(self): #calculate not active fields from active ones
        taskdone = False
        arrow_counter = self.countOutgoingArrows()
        lacking = arrow_counter - self.countActive()
        #0.set markers
        self.k_all_determined = self.k_all_active
        self.tau_determined = self.tau_active
        if(self.tau_active == True):
            self.k_all_determined = True
            self.k_all = 1/self.tau if self.tau != 0 else np.inf
        for arr in self.arrows:
            if(arr.source is self):
                arr.k_determined = arr.k_active
                arr.sf_determined = arr.sf_active
            
        for i in range(2):
            #1. search for solutions of sf_x * k_x = k_all (thirds)
            for j in range(2):
                for arr in self.arrows:
                    if(arr.source is self):
                        if(arr.k_determined and arr.sf_determined and not(self.k_all_determined)):
                            self.k_all = arr.k / arr.sf if arr.sf != 0 else 0
                            self.k_all_determined = True if arr.sf != 0 else False
                        if(arr.k_determined and not(arr.sf_determined) and self.k_all_determined):
                            arr.sf = arr.k / self.k_all if self.k_all != 0 else 0
                            arr.sf_determined = True if self.k_all != 0 else False
                        if(not(arr.k_determined) and arr.sf_determined and self.k_all_determined):
                            arr.k = arr.sf * self.k_all
                            arr.k_determined = True
                        if(not(arr.k_determined) and arr.sf_determined and arr.sf == 0):
                            arr.k = 0
                            arr.k_determined = True
                        if(self.k_all_determined and self.k_all == 0 and not(arr.k_determined)):
                            arr.k_determined = True
                            arr.k = 0    
            #2. check if k's and k_al can give all (k_all=k_1+k_2+... equation)
            counter = 0
            if(self.k_all_determined): counter += 1
            for arr in self.arrows:
                if(arr.source is self):
                    if(arr.k_determined): counter += 1
            if(counter >= arrow_counter): #means that system is determined
                sumk = 0
                for arr in self.arrows:
                    if(arr.source is self and arr.k_determined):
                        sumk += arr.k                
                if(not(self.k_all_determined)):
                    self.k_all = sumk
                    self.k_all_determined = True
                else:
                    for arr in self.arrows:
                        if(arr.source is self and not(arr.k_determined)):
                            arr.k = self.k_all - sumk
                            arr.k_determined = True    
                            break
            #3. if not all possible parameters are given, add sf's = (1-other known sf's)/no of lacking sf's
            if(lacking > 0):
                sf_init = (1-self.countActiveSFsValues())/lacking
                for arr in self.arrows:
                    if(lacking > 0 and arr.source is self and not(arr.sf_determined)):
                        arr.sf = sf_init
                        arr.sf_determined = True
                        lacking -= 1
            #4. do sf's excluding thing to get k_all or some k's (like in point 2)
            if(not(self.k_all_determined)):
                ksum = 0.0
                sfsum = 0.0
                arrowssum = 0
                for arr in self.arrows:
                    if(arr.source is self):    
                        if(arr.sf_determined):
                            sfsum += arr.sf
                            arrowssum += 1
                        if(arr.k_determined):
                            ksum += arr.k
                            arrowssum += 1  
                if(arrowssum == arrow_counter):
                    self.k_all = ksum / (1-sfsum) if sfsum != 1 else 0
                    self.k_all_determined = True if sfsum != 1 else False
            #5. stop if all values are determined
            if(self.k_all_determined):
                det_sum = 0
                for arr in self.arrows:
                    if(arr.source is self):
                        if(arr.k_determined and arr.sf_determined):
                            det_sum += 1
                if(det_sum == arrow_counter):
                    taskdone = True
                    break #everything is determined!! (except tau)

        #take care of tau 
        if(not(self.tau_determined) and self.k_all_determined):
            self.tau = 1/self.k_all if self.k_all != 0 else np.inf
            self.tau_determined = True
            
        return taskdone
##################################################################################################
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
        
        if(k_all_active != self.k_all_active): #change testing
            self.k_all_active = k_all_active
            self.enableDisableKs()
            tau_edit.setEnabled(self.tau_enabled)
        elif(tau_active != self.tau_active): #change testing  
            self.tau_active = tau_active
            self.enableDisableKs()
            k_all_edit.setEnabled(self.k_all_enabled)
        if(c_active != self.c_active): #change testing
            self.c_active = c_active

        #update values if given field was active (and fixation state)
        #calculate and update values of not active fields (if some field was changed)       
        if(k_all_active):
            self.k_all = float(k_all_edit.text.text())
            self.k_all_fixed = k_all_edit.check_fixed.isChecked()
            self.updateBranchKs()
        elif(tau_active):
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
    def __init__(self, new_name, pop_source, pop_target):
        self.name = new_name #name of the process
        self.source = pop_source #initialize yourself with both neighbour populations
        self.target = pop_target
        self.source.arrows.append(self) #initialize neighbour populations with yourself
        self.target.arrows.append(self)
        self.p1 = QtCore.QPoint()
        self.p2 = QtCore.QPoint()
        self.number = 1 #number of arrow between some pair of populations.numbering halps to render arrows separately
        self.displacement = 14 #separation distance between arrows
        self.dist_treshold = 7 #if distance from point to crossing point is below treshold function cotains return true hehehe
        self.color = QtCore.Qt.black
        
        self.k = 1 #both used depending on mode selected
        self.sf = 1
        self.k_fixed = False
        self.sf_fixed = False
        self.k_active = False #active means editable by user
        self.sf_active = False
        self.k_enabled = False #enabled means that user is allowed to turn it active
        self.sf_enabled = False #if enabled is False, then active also must be False

    def updateState(self, k_edit, sf_edit):
        k_active = k_edit.isActive()
        k_edit.setActive(k_active)

        sf_active = sf_edit.isActive()
        sf_edit.setActive(sf_active) 
        
        if(k_active != self.k_active): #change testing
            self.k_active = k_active
            self.source.enableDisableKs()
            sf_edit.setEnabled(self.sf_enabled)
        elif(sf_active != self.sf_active): #change testing  
            self.sf_active = sf_active
            self.source.enableDisableKs()     
            k_edit.setEnabled(self.k_enabled)
        
        #update values if given field was active (and fixation state)
        #calculate and update values of not active fields (if some field was changed)       
        if(k_active):
            self.k = float(k_edit.text.text())
            self.k_fixed = k_edit.check_fixed.isChecked()
            self.source.updateBranchKs()
        if(sf_active):
            self.sf = float(sf_edit.text.text())
            self.sf_fixed = sf_edit.check_fixed.isChecked()
            self.source.updateBranchKs()
            
        if(not(k_active)):
            k_edit.text.setText("%.9f" %  self.k)
        if(not(sf_active)):    
            sf_edit.text.setText("%.9f" %  self.sf)

    def remove(self, model): #removes arrow from neighbouring populations and model
        n1 = self.source.arrows.index(self)
        self.source.arrows.pop(n1)
        n2 = self.target.arrows.index(self)
        self.target.arrows.pop(n2) #here you have to recount arrows between populations...
        count = 1
        for arr in self.source.arrows:
            if( arr.source is self.target or arr.target is self.target ):
                arr.number = count
                count += 1
        n3 = model.processes.index(self)
        model.processes.pop(n3)
        
    def getsetLocation(self):
        p1 = QtCore.QPointF(self.source.rect.time() + self.source.rect.width() / 2, self.source.rect.y() + self.source.rect.height() / 2)
        p2 = QtCore.QPointF(self.target.rect.time() + self.target.rect.width() / 2, self.target.rect.y() + self.target.rect.height() / 2)
        
        diff = p2 - p1 # just make arrow shorter....
        correction = abs(diff.time() / math.sqrt(diff.time() * diff.time() + diff.y() * diff.y())) #uzaleznij odjecie od kata...
        to_substr = (40 * correction + 28) * diff / math.sqrt(diff.time() * diff.time() + diff.y() * diff.y())
        p1 = p1 + to_substr
        p2 = p2 - to_substr
        
        diff = p1 - p2
        difflen = math.sqrt(diff.x()**2 + diff.y()**2)
        if(self.number > 1 and difflen != 0.0): #move second, third, ... arrow a little bit to avoid overlap
            sinkat = diff.y() / difflen
            coskat = diff.x() / difflen
            if(sinkat < 0):
                alittle = QtCore.QPointF(sinkat * self.displacement, -coskat * self.displacement)
            elif(sinkat > 0):
                alittle = QtCore.QPointF(-sinkat * self.displacement, coskat * self.displacement)
            elif(coskat > 0):
                alittle = QtCore.QPointF(sinkat * self.displacement, -coskat * self.displacement)
            else:
                alittle = QtCore.QPointF(-sinkat * self.displacement, coskat * self.displacement)
            
            p1 += (-1)**self.number * alittle * math.floor(self.number / 2.0)
            p2 += (-1)**self.number * alittle * math.floor(self.number / 2.0)
        
        self.p1 = p1
        self.p2 = p2
        return (p1, p2)
        
    def contains(self, point):
        try:
            a_p1p2 = float(self.p2.y() - self.p1.y()) / float(self.p2.x() - self.p1.x()) #find linear eq for p1 and p2
            b_p1p2 = float(self.p1.y()) - a_p1p2 * float(self.p1.x())
            
            a_point = -1 / a_p1p2 #find linear eq for point which is perpendicular to p1p2
            b_point = float(point.y()) - a_point * float(point.time())
            
            x_cross = (b_point - b_p1p2) / (a_p1p2 - a_point) #find crossing point
            y_cross = a_p1p2 * x_cross + b_p1p2
            
            if self.p1.x() >= self.p2.x(): #check if crossing point is between p1 and p2
                if x_cross <= self.p1.x() and x_cross >= self.p2.x():
                    cond1 = True
                else:
                    cond1 = False
            else:
                if x_cross <= self.p2.x() and x_cross >= self.p1.x():
                    cond1 = True
                else:
                    cond1 = False  
                    
            if self.p1.y() >= self.p2.y(): #check if crossing point is between p1 and p2
                if y_cross <= self.p1.y() and y_cross >= self.p2.y():
                    cond2 = True
                else:
                    cond2 = False
            else:
                if y_cross <= self.p2.y() and y_cross >= self.p1.y():
                    cond2 = True
                else:
                    cond2 = False                 
                
            
            dist = math.sqrt(math.pow(float(point.time()) - x_cross, 2) + math.pow(float(point.y()) - y_cross, 2))
            if dist <= self.dist_treshold:
                cond3 = True
            else:
                cond3 = False
            
            return (cond1 and cond2 and cond3)
        
        except:
            return False

    def paintYourself(self, painter):
        p1, p2 = self.getsetLocation()
        
        #firstly draw sinusiodal shape indicating nonradiative process
        fragm_len = 3.0
        modamp = 5.0 #depth of modulation
        diff = p2 - p1
        full_length = math.sqrt(diff.x()*diff.x() + diff.y()*diff.y())
        iters = math.floor(full_length / fragm_len)
        unit_vect = diff * fragm_len / full_length #piece of line used to render whole curve
        if diff.x() >= 0:
            angle = math.asin(diff.y() / math.sqrt(diff.x()*diff.x() + diff.y()*diff.y()))
        else:
            angle = math.pi - math.asin(diff.y() / math.sqrt(diff.x()*diff.x() + diff.y()*diff.y()))
        perp_vect = QtCore.QPointF(modamp*math.cos(angle+math.pi/2), modamp*math.sin(angle+math.pi/2))
        
        path = QtGui.QPainterPath(p1)
        
        for i in range(1,iters+1):
            path.lineTo(p1 + unit_vect * i + perp_vect * math.sin(i * math.pi / iters) * math.sin(i * 1))
        
        path.lineTo(p2)
        painter.drawPath(path)
            
        diff = p1 - p2 #potrzebne do zrobienia grota strzalki  
        if diff.x() >= 0:
            angle = math.asin(diff.y() / math.sqrt(diff.x()*diff.x() + diff.y()*diff.y()))
        else:
            angle = math.pi - math.asin(diff.y() / math.sqrt(diff.x()*diff.x() + diff.y()*diff.y()))
        
        angle_diff = math.pi / 10.0 #determines shape of the arrow
        length = 10.0 #determines shape of the arrow
        p_arr1 = QtCore.QPointF(length*math.cos(angle+angle_diff), length*math.sin(angle+angle_diff))
        p_arr2 = QtCore.QPointF(length*math.cos(angle-angle_diff), length*math.sin(angle-angle_diff))
        
        painter.drawLine(p2, p2 + p_arr1)
        painter.drawLine(p2, p2 + p_arr2)

class ParamControl(QWidget):
    def __init__(self, parent, name, callback):
        super().__init__(parent)
        
        self.name = name
        self.isactive = None #indicates if you can edit text and fix
        self.isenabled = None #indicates if you can activate text to be active

        self.label = QtWidgets.QLabel(name, self)
        self.text = QtWidgets.QLineEdit('0.0', self)
        self.check_editable = QtWidgets.QCheckBox("Modify?", self)
        self.check_fixed = QtWidgets.QCheckBox("Fixed?", self)
        self.check_editable.clicked.connect(callback)
        self.check_fixed.clicked.connect(callback)
        self.text.editingFinished.connect(callback)
        
        self.setActive(False)
        self.setEnabled(False)
        
    def loadState(self, enabled, active, fixed, value):
        self.text.setText("%.9f" % value)
        self.setActive(active)
        self.setEnabled(enabled)
        self.check_fixed.setChecked(fixed)
        if(active and not(enabled)):
            raise Exception("Algorithm failure, id 1007!")
            
    def isActive(self): #asks for sate of checkbox, but does not update itself
        return self.check_editable.isChecked()
    
    def setEnabled(self, setenabled): #does things only if state is really changed
        if(setenabled != self.isenabled):
            if(setenabled):
                self.check_editable.setEnabled(True)
                #self.setActive(True)
                self.isenabled = True
            else:
                self.check_editable.setChecked(False)
                self.check_editable.setEnabled(False)
                self.setActive(False)
                self.isenabled = False
    
    def setActive(self, setactive):
        if(setactive != self.isactive):
            if(setactive):
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
        
    def paintEvent(self,event): 
        self.label.setGeometry(10, 0, 100, 25)
        self.text.setGeometry(10, 30, 100, 25)
        self.check_editable.setGeometry(10, 60, 100, 25)
        self.check_fixed.setGeometry(10, 85, 100, 25)
        
class ModelWindow(QWidget):

    def __init__(self, model_ref, ):
        super().__init__()
        self.model = model_ref
#        self.title = 'Model Editor'
        self.left = 100
        self.top = 100
        self.width = 800
        self.height = 700
        self.colors = (QtCore.Qt.black, QtCore.Qt.red, QtCore.Qt.green, QtCore.Qt.blue, QtCore.Qt.magenta, \
                       QtCore.Qt.darkRed, QtCore.Qt.darkGreen, QtCore.Qt.darkBlue, QtCore.Qt.darkCyan, QtCore.Qt.darkMagenta, \
                       QtCore.Qt.darkYellow, QtCore.Qt.darkGray)  #color order
            #maybe use HSV system instead???

#        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
#        self.show()
        
        self.mouse_position = None #mouse position from las mouse move event
        
        ## Create some widgets to be placed inside
        self.label1 = QtWidgets.QLabel('Population name:', self)
        self.text1 = QtWidgets.QLineEdit('', self)
        self.button1 = QtWidgets.QPushButton('Add population', self)
        self.button1.clicked.connect(self.button1Func)
        self.label5 = QtWidgets.QLabel('Process name:', self)
        self.text5 = QtWidgets.QLineEdit('', self)
        #self.select_process = QtWidgets.QComboBox(self)
        #self.select_process.addItem("Thermal")
        #self.select_process.addItem("Light activated")
        self.button5 = QtWidgets.QPushButton('Add process', self)
        self.button5.clicked.connect(self.button5Func)
        self.label1.show()
        self.text1.show()
        self.button1.show()
        self.label5.show()
        self.text5.show()
        #self.select_process.show()
        self.button5.show()        

        self.installEventFilter(self)
        self.setMouseTracking(True)

        self.mousepressed = False #indicates which population is being moved, or False if none
        self.mouse_dx = 0 #markers of relative mouse move from the moment of click
        self.mouse_dy = 0
        self.ref_mouse = QtCore.QPoint(0,0) #marker of mouse position when population was clicked
        
        self.pop_edit = False #to edit populatiions
        self.label2 = QtWidgets.QLabel('', self)
        #self.eps_table = QtWidgets.QTableWidget(self)
        #self.eps_table.setColumnCount(2)
        #self.eps_table.setColumnWidth(0,49);
        #self.eps_table.setColumnWidth(1,49);
        #self.eps_table.setHorizontalHeaderLabels(('lambda', 'epsilon'));
        #self.eps_table.verticalHeader().setVisible(False);
        #self.button2 = QtWidgets.QPushButton('Add eps', self)
        #self.button2.clicked.connect(self.button2Func)        
        self.button3 = QtWidgets.QPushButton('Done!', self)
        self.button3.clicked.connect(self.button3Func)
        self.button31 = QtWidgets.QPushButton('Delete', self)
        self.button31.clicked.connect(self.button31Func)    
        
        self.c_edit = ParamControl(self, "Initial population", self.populationEditFunc)
        self.tau_edit = ParamControl(self, "Time constant", self.populationEditFunc)
        self.k_all_edit = ParamControl(self, "Rate constant", self.populationEditFunc)      
        
        self.proc_edit = False #to edit processes
        #self.text_proc = QtWidgets.QLineEdit('', self) 
        self.button4 = QtWidgets.QPushButton('Done!', self)
        self.button4.clicked.connect(self.button4Func)
        self.button41 = QtWidgets.QPushButton('Delete!', self)
        self.button41.clicked.connect(self.button41Func)

        self.k_edit = ParamControl(self, "Rate constant", self.processEditFunc)
        self.sf_edit = ParamControl(self, "Splitting factor", self.processEditFunc)
        #self.sf_edit.text.setValidator(QtGui.QDoubleValidator(0, 1, 9,self.sf_edit.text)) #TODO
        #TODO: add validators so sf can be higher than 0 and lower than 1 (or 1 if only one arrow), and k can be only below k_all? 
        
        self.process_adding = False #indicates that process arrow is being added (select populations)
        
    def populationEditFunc(self):
        if(type(self.pop_edit) != bool):
            self.pop_edit.updateState(self.c_edit, self.tau_edit, self.k_all_edit)
            
        
    def processEditFunc(self):   
        if(type(self.proc_edit) != bool):
            self.proc_edit.updateState(self.k_edit, self.sf_edit)

    def button1Func(self): #creates new population
        found = False #ensure that new name is unique
        for elem in self.model.populations:
            if elem.name == self.text1.text():
                found = True
                
        for elem in self.model.processes:
            if elem.name == self.text1.text():
                found = True
                
        if not(isIdentifier(self.text1.text())): #it has to be a valid python id
            found = True
            
        if found == False and len(self.text1.text()) > 0:
            tmp_pop = ModPopulation(self.text1.text())
            tmp_pop.color = self.colors[random.randint(0, len(self.colors)-1)]
            if(len(self.model.populations) == 0): tmp_pop.c = 1
            self.model.addPopulation(tmp_pop)
            tmp_pop.enableDisableKs()
            self.text1.setText('')
            self.repaint()

#    def button2Func(self): #adds new epsilon entry to population
#        if self.pop_edit != False:
#            added_row = self.eps_table.rowCount()
#            self.eps_table.setRowCount(added_row + 1)
#            tmp1 = QtWidgets.QTableWidgetItem('')
#            tmp2 = QtWidgets.QTableWidgetItem('')
#            #set flags?
#            self.eps_table.setItem(added_row,0,tmp1)
#            self.eps_table.setItem(added_row,1,tmp2)
        
    def button3Func(self): #saves params to population
        #num_rows = self.eps_table.rowCount() ##in future ensure that values are not rounded during this process (dict->txt->dict)
        #new_dict = dict()
        #for row in range(num_rows):
        #    tmp_item1 = self.eps_table.item(row, 0).text()
        #    tmp_item2 = self.eps_table.item(row, 1).text()
        #    if not(self.isStrNumber(tmp_item1) and self.isStrNumber(tmp_item2)):
        #        continue
        #    new_dict[float(tmp_item1)] = float(tmp_item2)
        #self.pop_edit.epsilon = new_dict
        self.pop_edit = False
        self.repaint()
        
    def button31Func(self): #deletes population if possible
        if len(self.pop_edit.arrows) == 0:
            self.pop_edit.remove(self.model)
            self.pop_edit = False
        self.repaint()
    
    def button4Func(self): #finished mały miejski ul pszczołyedition of process
        #if self.isStrNumber(self.text_proc.text()):
            #self.proc_edit.k = float(self.text_proc.text()) 
        self.proc_edit = False
            #self.text_proc.setText('')
        self.repaint()
            
    def button41Func(self): #deletes arrow
        self.proc_edit.remove(self.model)
        self.proc_edit = False
        #self.text_proc.setText('')
        self.repaint()
            
    def button5Func(self): #adds process and starts selection of connected populations
        found = False #ensure that new name is unique
        for elem in self.model.processes:
            if elem.name == self.text5.text():
                found = True
            
        for elem in self.model.populations:
            if elem.name == self.text5.text():
                found = True
                
        if not(isIdentifier(self.text5.text())): #it has to be a valid python id
            found = True
            
        if found == False and len(self.text5.text()) > 0:
            self.process_adding = True
            self.repaint()
            
    def countArrows(self, population1, population2): #gives numer of the existing arrows between populations, and True if some k arrow already exist
        arrows = 0
        for arr in population1.arrows:
            if( ( arr.source is population1 and arr.target is population2 ) or ( arr.source is population2 and arr.target is population1 ) ):
                arrows += 1
        return arrows
        
    def isStrNumber(self,s):
        try:
            float(s)
            return True
        except ValueError:
            return False    
        
    def paintEvent(self,event): 
        self.label1.setGeometry(10, 10, 100, 25) #standard gui configuration
        self.text1.setGeometry(10, 40, 100, 25)
        self.button1.setGeometry(10, 70, 100, 30)
        self.label5.setGeometry(10, 10+100, 100, 25)
        self.text5.setGeometry(10, 40+100, 100, 25)
        #self.select_process.setGeometry(10, 70+100, 100, 25)
        self.button5.setGeometry(10, 70+100, 100, 30)        
        self.label2.setGeometry(10, 100+120, 100, 25)
        #self.eps_table.setGeometry(10, 130+150, 100, 200)
        #self.button2.setGeometry(10, 350+150, 50, 28)
        #self.text_proc.setGeometry(10, 130+150, 100, 25)
        
        self.c_edit.setGeometry(0, 250, 110, 110)
        self.tau_edit.setGeometry(0, 250+110, 110, 110)
        self.k_all_edit.setGeometry(0, 250+110+110, 110, 110)   
        self.button3.setGeometry(10, 250+110+110+110, 100, 30)
        self.button31.setGeometry(10, 250+110+110+110+30, 100, 30)        

        self.k_edit.setGeometry(0, 250, 110, 110)
        self.sf_edit.setGeometry(0, 250+110, 110, 110)   
        self.button4.setGeometry(10, 250+110+110, 100, 30)
        self.button41.setGeometry(10, 250+110+110+30, 100, 30)
        
        if self.pop_edit == False: #population edit menu
            self.label2.setVisible(False)
            #self.eps_table.setVisible(False)
            #self.button2.setVisible(False)
            self.button3.setVisible(False)
            self.button31.setVisible(False)
            self.c_edit.setVisible(False)
            self.tau_edit.setVisible(False)
            self.k_all_edit.setVisible(False) 
        else:
            self.label2.show()
            #self.eps_table.show()
            #self.button2.show()
            self.button3.show()
            self.button31.show()
            self.c_edit.show()
            self.tau_edit.show()
            self.k_all_edit.show()
            
        if self.proc_edit == False:
            self.button4.setVisible(False)
            self.button41.setVisible(False)
            #self.text_proc.setVisible(False)
            self.k_edit.setVisible(False)
            self.sf_edit.setVisible(False)
        else:
            self.button4.show()
            self.button41.show()
            #self.text_proc.show()   
            self.label2.show()
            self.k_edit.show()
            self.sf_edit.show()
            
        if(self.process_adding != False): self.button5.setText("Select boxes!")
        else: self.button5.setText("Add process")
        
        painter = QtGui.QPainter(self)
        painter.begin(self)
        
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        #painter.setWorldMatrixEnabled(False)
        
        rmpen = QtGui.QPen()
        rmpen.setWidth(3)
        #colorum = QtCore.Qt.red
        #if self.process_adding != False:
        #    colorum = QtCore.Qt.yellow
        marg = 5
        
        for r in self.model.populations: #should also call paintYourself function of the population, just like in case of processss
            rmpen.setBrush(r.color)
            rmpen.setWidth(3)
            if self.mousepressed is r:
                tmprect = copy.deepcopy(r.rect)
                tmprect.setX(tmprect.time() + self.mouse_dx)
                tmprect.setY(tmprect.y() + self.mouse_dy)
                tmprect.setWidth(r.rect_w)
                tmprect.setHeight(r.rect_h)  
            else:
                tmprect = r.rect  
                if(self.mouse_position != None):
                    if(r.rect.contains(self.mouse_position)): rmpen.setWidth(5)
            painter.setPen(rmpen)
            painter.drawRoundedRect(tmprect,10,10)
            painter.drawText(tmprect.time() + marg, tmprect.y() + marg, tmprect.width() - 2 * marg, tmprect.height() - 2 * marg, QtCore.Qt.AlignCenter, r.name)
        
        for r in self.model.processes:
            rmpen.setWidth(3)
            rmpen.setBrush(r.color)
            if(self.mouse_position != None):
                if(r.contains(self.mouse_position)): rmpen.setWidth(5)
            painter.setPen(rmpen)
            r.paintYourself(painter)

        painter.end()
        
    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.MouseMove:
            self.mouse_position = event.pos()
            if self.mousepressed != False:
                self.mouse_dx = self.mouse_position.time() - self.ref_mouse.time()
                self.mouse_dy = self.mouse_position.y() - self.ref_mouse.y()
            self.repaint()
            
        elif event.type() == QtCore.QEvent.MouseButtonRelease:
            if self.mousepressed != False:
                self.mousepressed.rect.setX(self.mousepressed.rect.time() + self.mouse_dx)
                self.mousepressed.rect.setY(self.mousepressed.rect.y() + self.mouse_dy)
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
                    
            if self.process_adding != True and self.process_adding != False: #finalize arrow adding process
                for r in self.model.populations:
                    found = False #if user clicked something useful
                    if r.rect.contains(event.pos()):
                        if not(r is self.process_adding):
                            found = r
                            break
                if found == False:
                    self.process_adding = False
                else: #finalize arrow from  self.process_adding to found
                    arrow_count = self.countArrows(self.process_adding, found)
                    #if self.select_process.currentIndex() == 0: # Thermal
                    #    tmp_arrow = ModThermal(self.text5.text(), self.process_adding, found)
                    #elif self.select_process.currentIndex() == 1: # Light activated
                    #    tmp_arrow = ModRadiative(self.text5.text(), self.process_adding, found) #asumes, that there are only 2 kinds of arrows, which can be added
                    already_exists = False #check if any arrow connecting these populations exists, if yes then dont create new one
                    for arr in self.process_adding.arrows:
                        if(arr.source is found): already_exists = True
                        if(arr.target is found): already_exists = True
                    if(already_exists == False):
                        tmp_arrow = ModProcess(self.text5.text(), self.process_adding, found)
                        tmp_arrow.number = arrow_count + 1
                        tmp_arrow.color = self.colors[random.randint(0, len(self.colors)-1)]
                        self.model.addProcess(tmp_arrow)
                        tmp_arrow.source.enableDisableKs()
                        tmp_arrow.source.updateBranchKs()
                        self.text5.setText('')     
                    self.process_adding = False
            if self.process_adding == True: #process arrow adding, first population needs to be selected
                for r in self.model.populations:
                    found = False #if user clicked something useful
                    if r.rect.contains(event.pos()):
                        found = True
                        self.process_adding = r
                        break
                if found == False:
                    self.process_adding = False
                
            self.repaint()
        
        elif event.type() == QtCore.QEvent.MouseButtonDblClick:
            if self.proc_edit == False and self.pop_edit == False: #should edit process?
                for p in self.model.processes:
                    if p.contains(event.pos()):
                        self.proc_edit = p
                        self.label2.setText('Edit arrow ' + p.name + ' :')
                        self.k_edit.loadState(p.k_enabled, p.k_active, p.k_fixed, p.k)
                        self.sf_edit.loadState(p.sf_enabled, p.sf_active, p.sf_fixed, p.sf)
                        #self.text_proc.setText(str(p.k)) 
                        self.repaint()
                        break
            if self.pop_edit == False and self.proc_edit == False: #should edit population?
                for r in self.model.populations:
                    if r.rect.contains(event.pos()):
                        self.pop_edit = r
                        self.label2.setText('Edit ' + r.name + ' :')
                        self.c_edit.loadState(r.c_enabled, r.c_active, r.c_fixed, r.c)
                        self.k_all_edit.loadState(r.k_all_enabled, r.k_all_active, r.k_all_fixed, r.k_all)                        
                        self.tau_edit.loadState(r.tau_enabled, r.tau_active, r.tau_fixed, r.tau)                        
                        #self.eps_table.setRowCount(len(r.epsilon))
                        #ct_tmp = 0
                        #for k, v in r.epsilon.items():
                        #    tmp1 = QtWidgets.QTableWidgetItem(str(k))
                        #    tmp2 = QtWidgets.QTableWidgetItem(str(v))
                        #    #set flags?
                        #    self.eps_table.setItem(ct_tmp,0,tmp1)
                        #    self.eps_table.setItem(ct_tmp,1,tmp2)
                        #    ct_tmp += 1
                        self.repaint()
                        break

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

    def manualModelBuild(self):
        return ModelWindow(self)
     
    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename): #this is static method intentionally
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
 #           else:
 #               self.params['k_%i%i' % (destination,source)].set(-rate, vary=varied) #if destination == source, it means that this is the terminal component...
 #           
 #       for i in range(self.exp_no):
 #           if(len(sources[i]) > 0):
 #               self.params['k_%i%i' % (i+1,i+1)].set(expr=sources[i])

    def genParameters(self): #parameters are fixed by default, unfix some of them before fitting procedure
        self.params = lmfit.Parameters()
        #ASSUMES THAT SFs are not zero or 1!!!
        self.params.add('exp_no', value=len(self.populations), vary = False)
        if(len(self.populations) < 1):
            raise Exception("Algorithm failure, because model is empty!") 
        
        for i in range(len(self.populations)):        
            self.params.add('c_%i' % (i+1), self.populations[i].c, False, None, None, None, None)
            for j in range(len(self.populations)):
                self.params.add('k_%i%i' % (i+1,j+1), 0, False, None, None, None, None)        
        
        for i in range(len(self.populations)):
            popul = self.populations[i]
            source = i
            if(len(popul.arrows) == 0):
                self.params['k_%i%i' % (source+1,source+1)].set(-popul.k_all, vary=not(popul.k_all_fixed or popul.tau_fixed))
            else:
                #NOW YOU ARE GOING INTO BRANCH
                k_all_fixed = False
                total_sfs = 0.0
                if(popul.k_all_fixed or popul.tau_fixed): #if k_all fixed then do it
                    self.params['k_%i%i' % (source+1,source+1)].set(-popul.k_all, vary=False)
                    k_all_fixed = True #name should be done not fixed, but nevermind
                    
                for arr in popul.arrows:
                    arr.done = False #indicator if arrow remains to be done
                    if(arr.source is popul):
                        target = self.populations.index(arr.target)
                        if(arr.k_fixed and arr.sf_fixed and not(k_all_fixed)): #if k_all can be fixed based on arrow then do it
                            self.params['k_%i%i' % (source+1,source+1)].set(expr="-"+'k_%i%i' % (target+1,source+1)+"/"+str(arr.sf))
                            self.params['k_%i%i' % (target+1,source+1)].set(arr.k, vary=False)
                            k_all_fixed = True
                            total_sfs += arr.sf
                            arr.done = True
                        elif(arr.k_fixed): #fix all direclty fixed k's
                            self.params['k_%i%i' % (target+1,source+1)].set(arr.k, vary=False)
                            arr.done = True
                        elif(arr.sf_fixed): #set proper expression for defined sf's
                            self.params['k_%i%i' % (target+1,source+1)].set(expr="-"+'k_%i%i' % (source+1,source+1)+"*"+str(arr.sf))
                            total_sfs += arr.sf
                            arr.done = True
                    
                freedom = popul.countOutgoingArrows()-popul.countFixed() #how many free parameters will be
                if(not(k_all_fixed) and freedom > 0): #start vary from k_all
                    self.params['k_%i%i' % (source+1,source+1)].set(-popul.k_all, vary=True)
                    k_all_fixed = True #means that now you can put this into eq
                    freedom -= 1

                for arr in popul.arrows: #go egein over branch and set things to vary
                    if(arr.source is popul and not(arr.done)):
                        target = self.populations.index(arr.target)                            
                        if(freedom > 0):
                            self.params['k_%i%i' % (target+1,source+1)].set(arr.k, vary=True)
                            arr.done = True
                            freedom -= 1
                if(freedom > 0):
                    raise Exception("Algorithm failure, id 1002!")
                    
                expression="(" #sum all k's which are fixed directly or varied
                count = 0
                for arr in popul.arrows:
                    if(arr.source is popul and arr.done and not(arr.sf_fixed)):
                        target = self.populations.index(arr.target)                            
                        expression += '-k_%i%i' % (target+1,source+1)
                        count += 1
                expression += ")"
                if(total_sfs == 1):
                    raise Exception("Algorithm failure, id 1003!") 

                if(not(k_all_fixed)): #set proper eq for k_all based only on not-sf arrows
                    if(count == 0):
                        raise Exception("Algorithm failure, id 1005!")
                    expression += "/(1-" + str(total_sfs) + ")" 
                    self.params['k_%i%i' % (source+1,source+1)].set(expr=expression)
                    k_all_fixed = True

                else: #set eq's for last k, assumes that k_all is fixed    
                    expr_begining = '-k_%i%i' % (source+1,source+1) + "*(1-"+str(total_sfs) + ")"
                    if(count == 0):
                        expression = expr_begining
                    else:
                        expression = expr_begining + "+" + expression
                    count2 = 0
                    for arr in popul.arrows: #go egein over branch and set remaining expr's (only one can be found)
                        if(arr.source is popul and not(arr.done) and not(arr.sf_fixed)):
                            target = self.populations.index(arr.target)                                                     
                            self.params['k_%i%i' % (target+1,source+1)].set(expr=expression)
                            arr.done = True
                            count2 += 1
                    if(count2 > 1):
                        raise Exception("Algorithm failure, id 1004!")
                        
                for arr in popul.arrows: #just check, probbaly not needed, but i am too tired to ensure that
                    if(arr.source is popul):
                        if(arr.done == False):
                            raise Exception("Algorithm failure, id 1006!")
                            
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
        
    def checkParams(self, experiment): #run tests if params are correctly set. experiment object is to validiate its compatibility with model
        result = True                  #it should be run after updataParameers for both model and experiment, and assume that funcs loaded them correctly

        return result   
       





#model1 = Model()
#model1.manualModelBuild()
