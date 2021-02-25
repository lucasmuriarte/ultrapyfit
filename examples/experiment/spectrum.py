# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:59:31 2019

@author: 79344
"""
import pandas as pd
from scipy.signal import savgol_filter as SF
import matplotlib.pyplot as plt
import numpy as np  
import lmfit
import itertools
from lmfit.models import ExponentialModel, GaussianModel, ExpressionModel

class BasicSpectrum:
    def __init__(self, x, y, sort=True,name=None):
        self.name=name
        self.name=None
        self.zero_correction=False
        self.wavenumber_calculated=False
        self.baseline_correction={'value':False,'low':None,'high':None}
        if sort:
            self.data_table=pd.DataFrame({'wavelengths':x,'absorbances':y}).sort_values(by=['wavelengths']).reset_index(drop=True)  
        else:
            self.data_table=pd.DataFrame({'wavelengths':x,'absorbances':y})
        self.x=self.data_table['wavelengths'].values
        self.y=self.data_table['absorbances'].values
    
    def nmToWavenumber(self,undo=False):
        if undo and self.wavenumber_calculated:
            self.data_table['wavelengths']=self.x_wave
            self.wavenumber_calculated=False
        elif self.wavenumber_calculated==False:
            self.x_wave=self.data_table['wavelengths'].copy().values
            self.data_table['wavelengths']=self.data_table['wavelengths'].apply(lambda x:10**7/x)
            self.wavenumber_calculated=True
    
    def obtainMax(self,Print=False):
        maxi_index=self.data_table['absorbances'].idxmax()
        if Print:
            print('the maximum is at: '+ str(self.data_table['wavelengths'][maxi_index])\
                  +' nm; And the value is: '+str(self.data_table['absorbances'][maxi_index]))
        return {'wavelength':self.data_table['wavelengths'][maxi_index],'absorbance':self.data_table['absorbances'][maxi_index]}
    
    def obtainMin(self,Print=False):
        min_index=self.data_table['absorbances'].idxmin()
        if Print:
            print('the minimum is at: '+ str(self.data_table['wavelengths'][min_index])\
                  +' nm; And the value is: '+str(self.data_table['absorbances'][min_index]))
        return {'wavelength':self.data_table['wavelengths'][min_index],'absorbance':self.data_table['absorbances'][min_index]}
    
    def obtainMaxOfRange(self,low,high,Print=False):
        high_index=(self.data_table['wavelengths']-high).abs().sort_values().index[0]
        low_index=(self.data_table['wavelengths']-low).abs().sort_values().index[0]
        min_index=self.data_table['absorbances'][low_index:high_index].idxmax()
        if Print:
            print('the maximum in the range ('+ str(low)+'-'+str(high)+'nm) is at: '+ str(self.data_table['wavelengths'][min_index])\
                  +' nm; And the value is: '+str(self.data_table['absorbances'][min_index]))
        return {'wavelength':self.data_table[low_index:high_index]['wavelengths'][min_index],'absorbance':self.data_table[low_index:high_index]['absorbances'][min_index]}
    
    def obtainMinOfRange(self,low,high,Print=False):
        high_index=(self.data_table['wavelengths']-high).abs().sort_values().index[0]
        low_index=(self.data_table['wavelengths']-low).abs().sort_values().index[0]
        maxi_index=self.data_table['absorbances'][low_index:high_index].idxmin()
        if Print:
            print('the minimum in the range ('+ str(low)+'-'+str(high)+'nm) is at: '+ str(self.data_table['wavelengths'][maxi_index])\
                  +' nm; And the value is: '+str(self.data_table['absorbances'][maxi_index]))
        return {'wavelength':self.data_table[low_index:high_index]['wavelengths'][maxi_index],'absorbance':self.data_table[low_index:high_index]['absorbances'][maxi_index]}
    
    def obtainValue(self,value,Print=False ):
        index_value=(self.data_table['wavelengths']-value).abs().sort_values().index[0]
        if Print:
            print('the absorbance at: '+ str(self.data_table['wavelengths'][index_value])\
                  +' nm; And the value is: '+str(self.data_table['absorbances'][index_value]))
        return {'wavelength':self.data_table['wavelengths'][index_value],'absorbance':self.data_table['absorbances'][index_value]}
    
    def normByMax(self,itself=False):
        norm_spec=BasicSpectrum(self.data_table['wavelengths'],self.data_table['absorbances'])
        norm_spec.data_table['wavelengths']=self.data_table['wavelengths']
        norm_spec.data_table['absorbances'] = self.data_table['absorbances']/self.data_table['absorbances'].max()
        if itself == True:
            self.data_table['absorbances'] = self.data_table['absorbances']/self.data_table['absorbances'].max()
        else:
            return norm_spec
    
    def minMaxIndex(self,low,high):
        if low == None:
                low=(self.data_table['wavelengths']).min()
        if high == None:
            high=(self.data_table['wavelengths']).max()
        high_index=(self.data_table['wavelengths']-high).abs().sort_values().index[0]
        low_index=(self.data_table['wavelengths']-low).abs().sort_values().index[0]
        if low_index<high_index:
            return low_index,high_index
        elif high_index<low_index:
            return high_index,low_index
        else:
            raise Exception('low and high are same values')
        
    def calculateArea(self,low=None,high=None,Print=False):
        low_index,high_index=self.minMaxIndex(low,high)
        y=self.data_table['absorbances'][low_index:high_index].values
        x=self.data_table['wavelengths'][low_index:high_index].values
        area=np.trapz(y,x)
        if Print:
            print(f'the area unde the range {low}-{high} nm is: {area}')
        return area
    
    def averageOfRange(self,low=None,high=None,Print=False):
        low_index,high_index=self.minMaxIndex(low,high)
        mean=(self.data_table['absorbances'][low_index:high_index]).mean()
        if Print:
            print(f'the mean unde the range {low}-{high} nm is: {mean}')
        return mean
    
    def zeroCorrection(self):
        self.zero_correction=True
        mini=self.data_table['absorbances'].min()
        if mini<0:
            self.data_table['absorbances'] = self.data_table['absorbances']-mini
        
    def baselineCorrection(self,low=None,high=None):
        if low == None:
                low=(self.data_table['wavelengths']).min()
        if high == None:
            high=(self.data_table['wavelengths']).max() 
        self.baseline_correction={'value':True,'low':low,'high':high}
        mean=(self.data_table['absorbances'][(self.data_table['wavelengths'] >= low) \
              & (self.data_table['wavelengths'] <= high)]).mean()
        self.data_table['absorbances'] = self.data_table['absorbances']-mean 
        self.baseline_correction={'value':True,'low':low,'high':high}
        
    def normByNumber(self,number,itself=False):
        norm_spec=BasicSpectrum(self.data_table['wavelengths'],self.data_table['absorbances'])
        norm_spec.data_table['wavelengths']=self.data_table['wavelengths']
        norm_spec.data_table['absorbances'] = self.data_table['absorbances']/number
        if itself == True:
            self.data_table['absorbances'] = self.data_table['absorbances']/number
        else:
            return norm_spec
        
    def normByMaxOfRange(self,low=None,high=None,itself=False):
        number=self.obtainMaxOfRange(low,high)['wavelength']
        index_value=(self.data_table['wavelengths']-number).abs().sort_values().index[0]
        norm_spec=BasicSpectrum(self.data_table['wavelengths'],self.data_table['absorbances'])
        norm_spec.data_table['wavelengths']=self.data_table['wavelengths']
        norm_spec.data_table['absorbances'] = self.data_table['absorbances']/self.data_table['absorbances'][index_value]
        if itself == True:
            self.data_table['absorbances'] = self.data_table['absorbances']/self.data_table['absorbances'][index_value]
        else:
            return norm_spec
    
    def normAtWavelength(self,value,itself=False):
        index_value=(self.data_table['wavelengths']-value).abs().sort_values().index[0]
        norm_spec=BasicSpectrum(self.data_table['wavelengths'],self.data_table['absorbances'])
        norm_spec.data_table['wavelengths']=self.data_table['wavelengths']
        norm_spec.data_table['absorbances'] = self.data_table['absorbances']/self.data_table['absorbances'][index_value]
        if itself == True:
            self.data_table['absorbances'] = self.data_table['absorbances']/self.data_table['absorbances'][index_value]
        else:
            return norm_spec
    
    def smooth(self, window,polynom=3,itself=False):
        smoth_spec=BasicSpectrum(self.data_table['wavelengths'],self.data_table['absorbances'])
        smoth_spec.data_table['absorbances'] = SF(self.data_table['absorbances'].values,window_length=window, polyorder=polynom)
        if itself == True:
            self.data_table['absorbances'] = smoth_spec.data_table['absorbances'] 
        else:
            return smoth_spec
    
    def cut(self, low=None, high=None):
        if low ==None:
            low=self.data_table['wavelengths'].min()
        if high ==None:
            high=self.data_table['wavelengths'].max()
        high_index=(self.data_table['wavelengths']-high).abs().sort_values().index[0]
        low_index=(self.data_table['wavelengths']-low).abs().sort_values().index[0]
        self.data_table= self.data_table[low_index:high_index]
    
    def __sub__(self, obj):
        ret_obj = BasicSpectrum(self.data_table['wavelengths'],self.data_table['absorbances'])
        ret_obj.data_table['wavelengths']= self.data_table['wavelengths']
        ret_obj.data_table['absorbances']= self.data_table['absorbances'] - obj.data_table['absorbances']
        return ret_obj
    
    def __mul__(self, number):
        ret_obj = BasicSpectrum(self.data_table['wavelengths'],self.data_table['absorbances'])
        ret_obj.data_table['wavelengths']= self.data_table['wavelengths']
        ret_obj.data_table['absorbances']= self.data_table['absorbances'] * number
        ret_obj.y=self.y*number
        return ret_obj
    
    def __add__(self, obj):
        ret_obj = BasicSpectrum(self.data_table['wavelengths'],self.data_table['absorbances'])
        ret_obj.data_table['wavelengths']= self.data_table['wavelengths']
        ret_obj.data_table['absorbances']= self.data_table['absorbances'] + obj.data_table['absorbances']
        ret_obj.y= self.data_table['absorbances'] + obj.data_table['absorbances']
        return ret_obj
    
    def __truediv__(self, number):
        ret_obj = BasicSpectrum(self.data_table['wavelengths'],self.data_table['absorbances'])
        ret_obj.data_table['wavelengths']= self.data_table['wavelengths']
        ret_obj.data_table['absorbances']= self.data_table['absorbances']/number
        ret_obj.y=self.y/number
        return ret_obj
    
    def averigeSeveralSpectra(self,listSpec):
        ret_obj = BasicSpectrum(self.data_table['wavelengths'],self.data_table['absorbances'])
        for i in listSpec:
            ret_obj=ret_obj+i
        ret_obj=ret_obj/(len(listSpec)+1)
        return ret_obj
    
    def plotSpectrum(self,X_label='Wavelength (nm)', Y_label='Absorbance',axes='current',\
                     fsize=14,color=None,label=None,linestyle='-',linewidth=1,fill=False,two_Xaxes=True):
        if axes is 'current':
            ax1=plt.gca()
        else:
            ax1=axes
        if self.wavenumber_calculated:
            X_label='Wavenumber (cm$^{-1}$)'
        if label is None:
            ax=ax1.plot(self.data_table['wavelengths'],self.data_table['absorbances'],color=color,label=label,linestyle=linestyle,linewidth=linewidth)
        else:
            if fill: 
                ax=ax1.plot(self.data_table['wavelengths'],self.data_table['absorbances'],color=color,label='_',linestyle=linestyle,linewidth=linewidth)
                plt.fill_between(self.data_table['wavelengths'],self.data_table['absorbances'],color=plt.gca().lines[-1].get_color(),alpha=0.5,label=label)
            else:
                ax=ax1.plot(self.data_table['wavelengths'],self.data_table['absorbances'],color=color,label=label,linestyle=linestyle,linewidth=linewidth)
        ax1.set_xlabel(X_label,size=fsize)
        ax1.set_ylabel( Y_label,size=fsize)
        plt.ticklabel_format(style='sci',scilimits=(-3,4),axis='y')
        if self.name != None:
            plt.legend(self.name,prop={'size': fsize})
        else:
            if label is not None:
                plt.legend(prop={'size': fsize})
        plt.tick_params(axis = 'both', which = 'major', labelsize = fsize)
        plt.xlim(self.data_table['wavelengths'][self.data_table.index[0]],self.data_table['wavelengths'][self.data_table.index[-1]])
        if two_Xaxes and self.wavenumber_calculated:
            self._twinyAxes(fsize)
        return ax
    
    def _twinyAxes(self,fsize):
        """Twin X axes for plotSpectrum function"""
        ax1=plt.gca()
        ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax2=ax1.twiny()
        mini=int(np.min(self.x_wave))
        maxi=int(np.max(self.x_wave))
        labels=[i for i in range(mini,maxi,int(abs(mini-maxi)/5))]
        ax2.set_xlim(ax1.get_xlim())
        new_tick_locations=[10**7/float(i) for i in labels]
        ax2.set_xticks(new_tick_locations)
        ax2.set_xticklabels(labels)
        ax2.set_xlabel('Wavelength (nm)',size=fsize)
        plt.tick_params(axis = 'both', which = 'major', labelsize = fsize)
        
    def saveSpectrum(self,path):    
        plt.savefig(path,dpi=300)

    def derivate():
        return
    
    def gauss(self,x, amp, cen, sigma):
        "basic gaussian"
        return amp*np.exp(-(x-cen)**2/(2.*sigma**2))

    def gaussN (self,params,x='itslef'):
        """values should be a list of list containing the amplitude, the center and sigma values for each gaussian"""
        y0 = params['y0'].value
        if x is not 'itslef':
            x=x
        else:    
            x=self.data_table['wavelengths'].values
        self.parametro=[[params['amp_%i' % (ii+1)].value,params['cen_%i' % (ii+1)].value,params['sig_%i' % (ii+1)].value]\
                    for ii in range(self.N_gaus)] 
        return y0+sum([self.gauss(x,amp, cen, sigma) for amp, cen, sigma in self.parametro])
    
    def objective(self, params):
        """ calculate total residual for fitting data modeled by a sum of Gaussian functions"""
        resid = 0.0*self.data_table['absorbances'].values
        # make residual per data set
        resid = self.data_table['absorbances'].values - self.gaussN(params)
        # now flatten this to a 1D array, as minimize() needs
        return resid.flatten()
        
    def fitSumGaussian(self,centers,low=None,high=None, plot=True,sigma=10,vary_sigmas=True, vary_centers=True,min_max_centers=None,points_for_plot=10000,linewidth=3,two_Xaxes=True):
        """Fit a sum of gausians to the spectrum, where the values given in the centers (list)
            are used to as the initial values of the gusian centers. The fit supose y0 
            (baseline parameter) to be the value at the most red shift part of the spectra
            
            returns: a dictionary containing the fitted gassians as BasicSpectrum Objects, 
                     and the lmFit parameters results 
            
            Parameters
            -------------------------------------------
            centers (list): list containin initial values where gausssina functions should be found
            low (Float or int): Minimum wavelenght value from spectrum to be fit (optional)
            high (Float or int): Maximum wavelenght value from spectrum to be fit (optional)
            plot (bool): If True the results of the fit will be plotted
            sigma (int or list):If Int initial sigma  value for all gaus
                        If list should be same size as centers and containing the initial value for each Gaus
                        (optional)
            vary_sigmas:(bool or list): If bool and True the sigma of the gaussian will be optimized.
                        If bool and False the sigma of the gaussian will not be optimized
                        If list should be same size as centers and containing True or False for 
                                 each of the Gaussians according if the sigmas should be optimized
                        (optional)
            vary_centers (bool or list): If bool and True the center of the gaussian will be optimized.
                        If bool and False the center of the gaussian will not be optimized
                        If list should be same size as centers and containing True or False for 
                                each of the Gaussians according if the centers should be optimized  
                                (optional)
            min_max_centers (None or int or float): add min and max limits for the Gaussian centers
                                    example:  If center=250 and min_max_centers=5 --> center=250, min=245, max=255      
                                    (optional)
            points_for_plot(int or 'itself'): The number of point to plot fitted Gaussians, default: 10000, 
                        if itself, the the number of point will be equal to the point of the spectrum.
                        (optional)
            linewidth: The linewidth of the raw spectra to compare to the Gaussians if plot is True default: 3
                        (optional)             
            """   
            
        assert (type(points_for_plot)== int or  points_for_plot=='itself')
        assert (type(vary_centers)== list or  type(vary_centers)== bool)
        if points_for_plot=='itself':
            points_for_plot=len(self.data_table)
        if self.data_table['wavelengths'].values is 'itself':
           x=self.data_table['wavelengths'].values
        else:
            x=np.linspace(self.data_table['wavelengths'].values[0],self.data_table['wavelengths'].values[-1],points_for_plot)
        data_change=False
        fit_params = lmfit.Parameters()
        #supose the red part of the spectra to be close to base line or be  the baseline for y0
        if self.wavenumber_calculated:
            maxi=self.obtainValue(self.data_table['wavelengths'].min())['absorbance']
        else:
            maxi=self.obtainValue(self.data_table['wavelengths'].max())['absorbance']
        fit_params.add('y0' ,maxi,vary=False)
        if low or high != None:
            data_change=True
            if low == None:
                low=(self.data_table['wavelengths']).min()
            if high == None:
                high=(self.data_table['wavelengths']).max()  
            self.data_table=self.data_table[(self.data_table['wavelengths'] >= low) & (self.data_table['wavelengths'] <= high)]    
        self.N_gaus=len(centers)
        self.gauss_fit={}
        if type(vary_centers)== bool:
            vary_centers=[vary_centers for i in range(len(centers))]
        if type(vary_sigmas)== bool:
            vary_sigmas=[vary_sigmas for i in range(len(centers))]
        if type(sigma)== int:
            sigma=[sigma for i in range(len(centers))]
        for i in range(self.N_gaus):
            fit_params.add('amp_' + str(i+1),value=self.obtainValue(centers[i])['absorbance'], min=0.0,)
            if min_max_centers is None:
                fit_params.add('cen_'+ str(i+1),value=centers[i],vary=vary_centers[i])
            else: 
                fit_params.add('cen_'+ str(i+1),value=centers[i],vary=vary_centers[i],min=centers[i]-min_max_centers,max=centers[i]+min_max_centers)
            fit_params.add('sig_' + str(i+1), value=sigma[i],vary=vary_sigmas[i], min=0.0,  )
        result = lmfit.minimize(self.objective, fit_params, xtol=10**-10,ftol=10**-10) 
        if len(centers)>1:
            self.gauss_fit['Sum of Gaus']=BasicSpectrum(x,self.gaussN(result.params,x=x),sort=False)
        self.result_params=result.params
        if data_change:
            self.data_table=pd.DataFrame({'wavelengths':self.x,'absorbances':self.y}).sort_values(by=['wavelengths']).reset_index(drop=True)
            if self.zero_correction:
                self.zeroCorrection()
            if self.baseline_correction['value']:
               self.baselineCorrection(low=self.baseline_correction['low'],high=self.baseline_correction['high'])
        for i in range(self.N_gaus):
            spec=BasicSpectrum(x,self.gauss(x,result.params['amp_%i' % (i+1)],result.params['cen_%i' % (i+1)],result.params['sig_%i' % (i+1)])+result.params['y0'],sort=False)
            self.gauss_fit[f'gaus {i+1}']=spec
        colors = itertools.cycle(['tab:blue','red','tab:orange','tab:green','tab:red','tab:purple','tab:brown','m','y','c'])
        if plot:
            raw=BasicSpectrum(self.data_table['wavelengths'].values,self.data_table['absorbances'].values,sort=False)
            raw.plotSpectrum(label='raw data',linewidth=linewidth,color=next(colors))
            for key in self.gauss_fit.keys():
                self.gauss_fit[key].plotSpectrum(label=key,color=next(colors))
            if two_Xaxes and self.wavenumber_calculated:
                self._twinyAxes(14)
        return self.gauss_fit , self.result_params
            
    def calculateBaseline(self,peak_max, tipo='poly',poly_complete=True,max_gaus_sigma=None,vary_centers=True, plot=True): 
        x=self.data_table['wavelengths'].values
        y=self.data_table['absorbances'].values
#        rango = x[0]-x[-1]
        if tipo is 'poly':
            def guess(order):
                points=[0,int(len(x)/4),int(len(x)/2),int(len(x)/1.5),-1]
                return sum([y[i]*x[i]**order for i in points])/25 -np.min(y)
            if poly_complete:
                base_mod = ExpressionModel('a/x**4+b/x**3+c/x**2+d/x+e')
                pars = base_mod.make_params(a=guess(4) ,b=guess(3) ,c=guess(2) ,d=guess(1) ,e=np.min(y))
            else:    
                base_mod = ExpressionModel('a/(x**4)+b')
                pars = base_mod.make_params(a=guess(4) ,b=np.min(y))
        else:
            base_mod = ExponentialModel(prefix = 'exp_')
            pars = base_mod.guess(y,x=x)
        gaus_models=[GaussianModel(prefix='gaus%i_' %(i)) for i in range(1,len(peak_max)+1)]
        yy=base_mod.eval(pars,x=x)
        mod = base_mod
        for i,ii in enumerate(peak_max):
            print(i)
            pars.update(gaus_models[i].make_params())
            pars['gaus%i_center' %(i+1)].set(value=ii, min=x[0] , max=x[-1],vary=vary_centers)
            if max_gaus_sigma == None:
                pars['gaus%i_sigma' %(i+1)].set(value=15)
            else:
                pars['gaus%i_sigma' %(i+1)].set(value=15, max=max_gaus_sigma)
            if tipo is 'poly':
                 pars['gaus%i_amplitude' %(i+1)].set(value=np.mean(y)*30,min=0)
            else:
                pars['gaus%i_amplitude' %(i+1)].set(value=pars['exp_amplitude']._val*30,min=0)
            mod=mod+gaus_models[i]
#        init = mod.eval(pars, x=x)
        out = mod.fit(y, pars, x=x)
        comps = out.eval_components(x=x)
        if plot:
            self.plotSpectrum(label='Raw data')
            ax=plt.gca()
#            ax.plot(x, init, 'k-', label='initial_val')
            ax.plot(x, out.best_fit, 'r-', label='fit')
            if tipo is 'poly':
                ax.plot(x, comps['_eval'], label='exponential baseline')
            else:
                ax.plot(x, comps['exp_'], label='exponential baseline')
            plt.legend()
            for i in range(len(peak_max)):
                ax.plot(x, comps['gaus%i_' %(i+1)], label=f'gaus {i}')
        if tipo is 'poly':
            return comps['_eval']
        else:
            return comps['exp_']
    
    def substractBaseline(self,peak_max,tipo='poly', poly_complete=True, max_gaus_sigma=None, vary_centers=True, plot=True):
        ret_obj = BasicSpectrum(self.data_table['wavelengths'].values,self.data_table['absorbances'].values)
        exp_baseline=self.calculateBaseline(peak_max, tipo=tipo, poly_complete=poly_complete, max_gaus_sigma=max_gaus_sigma, vary_centers=vary_centers, plot=plot)
        base_line=BasicSpectrum(self.data_table['wavelengths'].values,exp_baseline)
        resta=ret_obj-base_line
        mini=resta.data_table['absorbances'].min()
        if mini<0:
            base_line=BasicSpectrum(self.data_table['wavelengths'].values,exp_baseline+mini)
        return ret_obj-base_line
    
    def decomposeInSumOf2Spec(self, spec_a,spec_b,low=None,high=None,plot=True):
        if low or high != None:
            if low == None:
                low=(self.data_table['wavelengths']).min()
            if high == None:
                high=(self.data_table['wavelengths']).max()  
        data=self.data_table[(self.data_table['wavelengths'] >= low) & (self.data_table['wavelengths'] <= high)]    
        spec_a_fit=spec_a.data_table[(spec_a.data_table['wavelengths'] >= low) & (spec_a.data_table['wavelengths'] <= high)]    
        spec_b_fit=spec_b.data_table[(spec_b.data_table['wavelengths'] >= low) & (spec_b.data_table['wavelengths'] <= high)]    
        def leastSquares (conentrations,data,spec_a,spec_b):
            residues=data-(conentrations['Ca']*spec_a+(1-conentrations['Ca'])*spec_b)
            return residues
        conentrations=lmfit.Parameters()
        conentrations.add_many(('Ca', 0.5, True, 0, 1, None, None))
        result_fit=lmfit.minimize(leastSquares, conentrations, args=(data['absorbances'].values,spec_a_fit['absorbances'].values,spec_b_fit['absorbances'].values),nan_policy='propagate')
        Ca,Cb=[result_fit.params[key].value for key in result_fit.params.keys()],[1-result_fit.params[key].value for key in result_fit.params.keys()]
        if plot:
            self.plotSpectrum()
            (spec_a*Ca).plotSpectrum()
            (spec_b*Cb).plotSpectrum()
            fit=spec_a_fit['absorbances'].values*Ca+spec_b_fit['absorbances'].values*Cb
            plt.plot(data['wavelengths'].values,fit,c='r')
        return Ca,Cb

                
