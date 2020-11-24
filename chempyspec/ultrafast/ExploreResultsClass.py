# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 18:35:41 2020

@author: lucas
"""
from FormatFigures import FiguresFormating
import matplotlib.pyplot as plt
import numpy as np
from ModelCreatorClass import  ModelCreator
import scipy.integrate as integral
from matplotlib.widgets import Slider

class ExploreResults(FiguresFormating):
    def __init__(self,fits,**kwargs):
        units=dict({'time_unit':'ps','time_unit_high':'ns','time_unit_low':'fs','wavelength_unit':'nm','factor_high':1000,'factor_low':1000},**kwargs)
        if type(fits) == dict:
            self.all_fit=fits 
        else:
            self.all_fit={1:fits}
        self.units=units
    
    def results(self,result_params,fit_number=None,verify_SVD_fit=False):
        if fit_number == None : fit_number=max(self.all_fit.keys())
        
        x=self.all_fit[fit_number].x
        #verify type of fit is: either fit to Singular vectors or global fit to traces
        if self.all_fit[fit_number].details['svd_fit'] and verify_SVD_fit ==False: #verify type of fit
            data=self.all_fit[fit_number][10]
        else:    
            data=self.all_fit[fit_number].data
        deconv=self.all_fit[fit_number].details['deconv']
        type_fit=self.all_fit[fit_number].details['type']
        exp_no=self.all_fit[fit_number].details['exp_no']
        tau_inf=self.all_fit[fit_number].details['tau_inf']
        model=ModelCreator(x,exp_no,tau_inf)
        ndata, nx = data.shape
        if type_fit == 'Exponential': 
            if deconv:
                curve_resultados=data*0.0
                for i in range(nx):
                    curve_resultados[:,i]=model.expNGaussDataset(result_params, i)
            else:
                t0 = result_params['t0_1'].value
                index=np.argmin([abs(i-t0) for i in x])
                curve_resultados=0.0*data[index:,:]
                for i in range(nx):
                    curve_resultados[:,i]=model.expNDataset(result_params, i)
        else:
            ksize = exp_no #size of the matrix = no of exponenses = no of species
            kmatrix = np.array([[result_params['k_%i%i' % (i+1,j+1)].value for j in range(ksize)] for i in range(ksize)])
            cinitials = [result_params['c_%i' % (i+1)].value for i in range(ksize)]
            eigs, vects = np.linalg.eig(kmatrix)#do the eigenshit
            #eigenmatrix = np.array([[vects[j][i] for j in range(len(eigs))] for i in range(len(eigs))]) 
            eigenmatrix = np.array(vects) 
            coeffs = np.linalg.solve(eigenmatrix, cinitials)
            curve_resultados=data*0.0
            for i in range(nx):
                    curve_resultados[:,i]=model.expNGaussDatasetTM(result_params, i,[coeffs,eigs,eigenmatrix])
        return curve_resultados

    
    def plotFit(self,size=14,fit_number=None,selection=None,plot_residues=True):      
        if fit_number == None : fit_number=max(self.all_fit.keys())
        x=self.all_fit[fit_number].x
        #verify type of fit is: either fit to Singular vectors or global fit to traces
        SVD_fit=self.all_fit[fit_number].details['svd_fit']
        if SVD_fit:
            data=self.all_fit[fit_number][10]
        else:
            data=self.all_fit[fit_number].data
        wavelength=self.all_fit[fit_number].wavelength
        params=self.all_fit[fit_number].params
        deconv=self.all_fit[fit_number].details['deconv']
        
        if wavelength is None:
            wavelength=np.array([i for i in range(len(data[1]))])
        if SVD_fit:
            selection=None
        if selection is None:
            puntos=[i for i in range(data.shape[1])]
        else:
            puntos=[min(range(len(wavelength)), key=lambda i: abs(wavelength[i]-num)) for num in selection]
        if len(puntos)<=10:
            if SVD_fit:
                legenda=['_' for i in range(data.shape[1]+1)]+['left SV %i' %i for i in range(1,data.shape[1]+1)]
            elif wavelength is not None:
                legenda=['_' for i in range(len(puntos)+1)]+[f'{round(wavelength[i])} nm' for i in puntos]
            else:
                legenda=['_' for i in range(len(puntos)+1)]+[f'curve {i}' for i in  range(data.shape[1])]
        xlabel='Time ('+self.units['time_unit']+')'
        if plot_residues==False:
            fig, ax = plt.subplots(figsize=(8,6))
            ax=['_',ax]
        else:
            fig, ax = plt.subplots(2, 1,sharex=True, figsize=(8,6), gridspec_kw={'height_ratios': [1, 5]})
        fittes=self.results(params,fit_number=fit_number)
        if deconv:
            residues= data-fittes
        else:
             t0 = params['t0_1'].value
             index=np.argmin([abs(i-t0) for i in x])
             residues= data[index:,:]-fittes
#        plt.axhline(linewidth=1,linestyle='--', color='k')
        alpha,s=0.80,8
        for i in puntos:
            if plot_residues:
                ax[0].scatter(x,residues[:,i], marker='o',alpha=alpha,s=s)
            ax[1].scatter(x,data[:,i], marker='o',alpha=alpha,s=s)
            ax[1].plot(x, fittes[:,i], '-',color='r',alpha=0.5,lw=1.5)
            if len(puntos)<=10:
                ax[1].legend(legenda,loc='best',ncol=1 if SVD_fit else 2)
        if plot_residues:
            self._formatFigure(ax[0],residues,x,size=size)
            ax[0].set_ylabel('Residues',size=size)
        self._formatFigure(ax[1],data,x,size=size)
        self._axisLabels(ax[1],xlabel,r'$\Delta$A',size=size)
        plt.subplots_adjust(left=0.145,right=0.95)
        return fig, ax
            
    def DAS(self,fit_number=None):
        if fit_number == None : fit_number=max(self.all_fit.keys()) 
        #verify type of fit is: either fit to Singular vectors or global fit to traces
        if self.all_fit[fit_number].details['svd_fit']:#verify type of fitaither SVD or global fit
            result_params=self.all_fit[fit_number][11]
        else:    
            result_params=self.all_fit[fit_number].params
        data=self.all_fit[fit_number].data
        deconv=self.all_fit[fit_number].details['deconv']
        tau_inf=self.all_fit[fit_number].details['tau_inf']
        exp_no=self.all_fit[fit_number].details['exp_no']        
        values=[[result_params['pre_exp%i_' % (ii+1)+str(i+1)].value for i in range(data.shape[1])] for ii in range(exp_no)]
        if deconv and tau_inf is not None:
            values.append([result_params['yinf_'+str(i+1)].value for i in range(data.shape[1])])
        return np.array(values)
    
    def plotDAS(self,number='all',fit_number=None,size=14,precision=2,plot_integrated_DAS=False,cover_range=None):
        '''REtunrs the Decay Asociated Spectra of the FIt
        Parameters: 
            number --> 'all' or a list conataining the decay of the species you want, plus -1 if you want the tau inf if existing
                        eg1.: if you want the decay of the second and third, [2,3] // eg2.:  fist third and inf, [2,3,-1]
                            
        ''' 
        #verify type of fit is: either fit to Singular vectors or global fit to traces
        if fit_number == None : fit_number=max(self.all_fit.keys())
        if self.all_fit[fit_number].details['svd_fit']:
            params=self.all_fit[fit_number][11]
        else:
            params=self.all_fit[fit_number].params
        wavelength=self.all_fit[fit_number].wavelength
        deconv=self.all_fit[fit_number].details['deconv']
        tau_inf=self.all_fit[fit_number].details['tau_inf']
        exp_no=self.all_fit[fit_number].details['exp_no'] 
        derivative_space=self.all_fit[fit_number].details['derivate'] 
        type_fit=self.all_fit[fit_number].details['type']#check for type of fit done target or exponential
        if number is not 'all':
            assert type(number)==list, 'NUmber should be "all" or a list containing the desired species if tau inf include -1 in the list'
            das=self.DAS(fit_number=fit_number)
            posible=[i+1 for i in range(exp_no)]
            if tau_inf is not None:
                posible.append(-1)
            wanted=[ii for ii,i in enumerate(posible) if i in number ]    
            das=das[wanted,:]
        else:
            das=self.DAS(fit_number=fit_number)
        if self.wavelength_unit=='cm-1':
            xlabel='Wavenumber (cm$^{-1}$)'
        else:
            xlabel='Wavelength (nm)'
        legenda=[]
        for i in range(exp_no):
            tau=params['tau%i_1' % (i+1)].value
            if tau<0.09:
                tau*=self.factor_low
                legenda.append(rf'$\tau {i+1}$ = '+'{:.2f}'.format(round(tau,precision))+' '+self.time_unit_low)
            elif tau>999:
                if tau>1E12:
                    legenda.append(r'$\tau$ = inf')
                else:    
                    tau/=self.factor_high
                    legenda.append(rf'$\tau {i+1}$ = '+'{:.2f}'.format(round(tau,precision))+' '+self.time_unit_high)
            else:
                legenda.append(rf'$\tau {i+1}$ = '+'{:.2f}'.format(round(tau,precision))+' '+self.time_unit)
        if deconv and type_fit == 'Exponential':
            if tau_inf is None:
                pass
            elif tau_inf != 1E+12:
                legenda.append(r'$\tau$ = {:.2f}'.format(round(self.tau_inf/self.factor_high,precision))+' '+self.time_unit_high)  
            else:
                legenda.append(r'$\tau$ = inf')
        if number is not 'all':
#            constants=' '.join([str(i.split('=')[1]) for i in legenda[:number]])
#            print(f'WARNING! time constants of value {constants} has been used to fit')
            legenda=[legenda[i] for i in wanted]

        if type(derivative_space)==dict and plot_integrated_DAS:
            das=np.array([integral.cumtrapz(das[i,:],wavelength,initial=0) for i in range(len(das))])
        fig,ax = plt.subplots(1,figsize=(11,6))
        for i in range(das.shape[0]):
            ax.plot(wavelength,das[i,:],label=legenda[i])
            plt.xlim(wavelength[0],wavelength[-1])
        leg=ax.legend(prop={'size': size})
        leg.set_zorder(np.inf)
        self._formatFigure(ax,das,wavelength,x_tight=True,set_ylim=False)
#        plt.axhline(linewidth=1,linestyle='--', color='k')
#        ax.minorticks_on()
#        ax.axes.tick_params(which='both',direction='in',top=True,right=True,labelsize=size)
        plt.xlabel(xlabel,size=size)
        plt.ylabel('$\Delta$A',size=size)
        if cover_range is not None:
            self._coverExcitation(ax,cover_range,wavelength)
        return fig, ax
    
    def verifiedFit(self,fit_number=None):
        if fit_number == None : fit_number=max(self.all_fit.keys()) 
        self.x_verivefit=self.all_fit[fit_number].x
        #verify type of fit is: either fit to Singular vectors or global fit to traces
        SVD_fit=self.all_fit[fit_number].details['svd_fit']
        if SVD_fit:
            self.data_fit=self.all_fit[fit_number][10]
        else:
            self.data_fit=self.all_fit[fit_number].data
        self.wavelength_fit=self.all_fit[fit_number].wavelength
        params=self.all_fit[fit_number].params
        deconv=self.all_fit[fit_number].details['deconv']
        
#        if fit_number is not None:
#            self.x_verivefit=self.all_fit[fit_number][0]
#            self.data_fit=self.all_fit[fit_number][1]
#            self.wavelength_fit=self.all_fit[fit_number][2]
#            if self.all_fit[fit_number][9]:
#                params=self.all_fit[fit_number][11]
#            else:
#                params=self.all_fit[fit_number][3].params
##            self.deconv=self.all_fit[fit_number][5]
       
        xlabel='Time ('+self.time_unit+')'
        self.fig, ax = plt.subplots(2, 1,sharex=True, figsize=(10,8), gridspec_kw={'height_ratios': [1, 5]})
        self.fittes=self.results(params)
        if deconv:
            self.residues= self.data_fit-self.fittes
        else:
             t0 = params['t0_1'].value
             index=np.argmin([abs(i-t0) for i in self.x_fit])
             self.residues=self.data_fit[index:,:]-self.fittes
        initial_i=self.data_fit.shape[1]//5
        self.l,=ax[1].plot(self.x_verivefit,self.data_fit[:,initial_i],marker='o',ms=3,linestyle=None,label='raw data')
        self.lll,=ax[0].plot(self.x_verivefit,self.residues[:,initial_i],marker='o',ms=3,linestyle=None,label='residues')
        self.ll,=ax[1].plot(self.x_verivefit,self.fittes[:,initial_i],alpha=0.5,lw=1.5,color='r',label='fit')
        delta_f=1.0
        _,maxi=self.data_fit.shape
        axcolor='orange'
        axspec = self.fig.add_axes([0.20, .02, 0.60, 0.01],facecolor=axcolor)
        self.sspec = Slider(axspec, 'curve number', 0, maxi-1,valstep=delta_f,valinit=0)
        self.sspec.on_changed(self.updateVerifiedFit)
        self._formatFigure(ax[0],self.residues,self.x_verivefit,size=14)
        ax[0].set_ylabel('Residues',size=14)
        self._formatFigure(ax[1],self.data_fit,self.x_verivefit,size=14)
        ax[1].legend(loc='upper right')
        ax[0].legend(loc='upper right')
        title=round(self.wavelength_fit[initial_i])
        plt.title(f'{title} nm')
        ax[1].set_xlabel(xlabel, size=14)
        ax[1].set_ylabel(r'$\Delta$A',size=14)
        ax[0].set_ylabel('Residues',size=14)
        self.fig.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        return self.fig,ax
        
    def updateVerifiedFit(self,val):
        # amp is the current value of the slider
        value = self.sspec.val
        value=int(round(value))
        # update curve
        title=round(self.wavelength_fit[value])
        plt.title(f'{title} nm')
        self.l.set_ydata(self.data_fit[:,value])
        self.ll.set_ydata(self.fittes[:,value])
        self.lll.set_ydata(self.residues[:,value])
        # redraw canvas while idle
        self.fig.canvas.draw_idle()

    def plotConcentrations(self,fit_number=None,size=14,names=None,plot_total_C=True,legend=True):#tmp function.
        if fit_number == None : fit_number=max(self.all_fit.keys()) 
            #verify type of fit is: either fit to Singular vectors or global fit to traces
        if self.all_fit[fit_number].details['svd_fit']:
            params=self.all_fit[fit_number][11]
            data=self.all_fit[fit_number][10]
        else:
            params=self.all_fit[fit_number].params
            data=self.all_fit[fit_number].data
        type_fit=self.all_fit[fit_number].details['type'] #check for type of fit done     
        x=self.all_fit[fit_number].x
        exp_no=self.all_fit[fit_number].details['exp_no'] 
        
        if type_fit =='Expeonential':
            return 'This fucntion is only available for target fit'
        else:
            xlabel='Time ('+self.time_unit+')'
            ndata, nx = data.shape
    #        resid = 0.0*self.data[:]
            #take kmatrix shit from params and solve eqs
            maxi_tau=-1/params['k_%i%i' % (exp_no-1,exp_no-1)].value
            if maxi_tau > x[-1]:
                maxi_tau = x[-1]
            ksize = exp_no #size of the matrix = no of exponenses = no of species
            kmatrix = np.array([[params['k_%i%i' % (i+1,j+1)].value for j in range(ksize)] for i in range(ksize)])
            cinitials = [params['c_%i' % (i+1)].value for i in range(ksize)]
            eigs, vects = np.linalg.eig(kmatrix)#do the eigenshit
            #eigenmatrix = np.array([[vects[j][i] for j in range(len(eigs))] for i in range(len(eigs))]) 
            eigenmatrix = np.array(vects) 
            coeffs = np.linalg.solve(eigenmatrix, cinitials) #solve the initial conditions sheet            
            t0 = params['t0_1'].value
            fwhm = params['fwhm_1'].value
            expvects = [coeffs[i]*self.expGaussConv(x-t0,-eigs[i],fwhm/2.35482) for i in range(len(eigs))] 
            concentrations = [sum([eigenmatrix[i,j]*expvects[j] for j in range(len(eigs))]) for i in range(len(eigs))]
            if names is None or len(names) != exp_no:
                names=[f'Specie {i}' for i in range(exp_no)]
            fig,ax = plt.subplots(1,figsize=(8,6))
            for i in range(len(eigs)):
                ax.plot(x, concentrations[i],label=names[i])
            allc = sum(concentrations)
            if plot_total_C:
                allc = sum(concentrations)
                ax.plot(x, allc, label='Total concetration') #sum of all for checking => should be unity
            if legend:
                plt.legend(loc='best')
            ax.minorticks_on()
            ax.axes.tick_params(which='both',direction='in',top=True,right=True,labelsize=size)
            plt.xlabel(xlabel,size=size)
            plt.ylabel('Concentration (A.U.)',size=size)
            plt.xlim(-3,round(maxi_tau*7))
            return fig,ax
    