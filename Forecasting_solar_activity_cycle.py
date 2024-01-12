"""
Author: Lalitha Sairam
Date: January 2024
"""

# Import necessary libraries
from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import celerite
from celerite import terms
from celerite.modeling import Model
from PyAstronomy.pyTiming import pyPeriod
from numpy.polynomial.polynomial import polyval
from matplotlib.font_manager import FontProperties



# Define a function 'Do_GP' for Gaussian Process analysis
def Do_GP(filelist, single=False, divbysmooth=0, sigclip=1e10, tzero=0, fbeg=0.001, fend=0.9,
          numtimes=10000, ylab="Observed", mediansub=True, bin=0, ndim=0, glsverbose=False,
          nredo=1, xcol=0, ycol=1, yerrcol=2, name=""):
    
    # Define constants and parameters
    empty = " "
    if ndim > 3:
        ndim = 3
        print("!!!!!! no larger polynomial degree than 3 allowed !!!!!!!")
    if ndim < 0:
        ndim = 0
        print("!!!!!! no smaller polynomial degree than 1 allowed !!!!!!!")

    # Define a constant model class for GP
    class MeanModelConst(Model):
        parameter_names = ("A", "B", "C", "D")

        def get_value(self, t):
            const = np.zeros(len(t)) + self.A + self.B * t + self.C * t**2 + self.D * t**3
            return const

    # Define a sine model class for GP
    class MeanModelSine(Model):
        parameter_names = ("Amp", "Per", "Phase", "offset", "slope", "quad", "cube")

        def get_value(self, t):
            sine = self.Amp * np.sin(2 * np.pi / self.Per * (t - self.Phase)) + self.offset + self.slope * t + self.quad * t**2 + self.cube * t**3
            return sine

    # Define negative log likelihood and gradient functions for GP optimization
    def neg_log_like(params, y, gp):
        gp.set_parameter_vector(params)
        nloglike = -gp.log_likelihood(y)
        return nloglike

    def grad_neg_log_like(params, y, gp):
        gp.set_parameter_vector(params)
        gloglike = -gp.grad_log_likelihood(y)
        return gloglike

    # Define negative log likelihood and gradient functions for Sine model optimization
    def neg_log_likeS(params, y, gpS):
        gpS.set_parameter_vector(params)
        nloglike = -gpS.log_likelihood(y)
        return nloglike

    def grad_neg_log_likeS(params, y, gpS):
        gpS.set_parameter_vector(params)
        gloglike = -gpS.grad_log_likelihood(y)
        return gloglike

    # Define negative log likelihood and gradient functions for Const model optimization
    def neg_log_likeC(params, y, gpC):
        gpC.set_parameter_vector(params)
        nloglike = -gpC.log_likelihood(y)
        return nloglike

    def grad_neg_log_likeC(params, y, gpC):
        gpC.set_parameter_vector(params)
        gloglike = -gpC.grad_log_likelihood(y)
        return gloglike
        
    def print_results(glsin, solnC, soln, glsres, data_file):
    	"""
    	Print results for the sine model, models with jitter and offset, GP model,
    	residuals, and overall analysis results.
    	Parameters:
    		glsin: Gls object for sine model
    		solnC: Result of the minimize operation for jitter + offset model
    		soln: Result of the minimize operation for GP model
    		glsres: Gls object for residuals
    		data_file (str): Name of the data file
    	"""
    	print("######################################################################")
    	print(f"Analysing file {data_file}")
    	print("Input data")
    	
    	# Print results for the sine model
    	print(f"Best sine frequency : {glsin.hpstat['fbest']:.6f} +/- {glsin.hpstat['f_err']:.6f}")
    	print(f"Best sine period    : {1. / glsin.hpstat['fbest']:.6f} +/- {glsin.hpstat['Psin_err']:.6f}")
    	print(f"Amplitude           : {glsin.hpstat['amp']:.6f} +/- {glsin.hpstat['amp_err']:.6f}")
    	
    	# Print results for models with jitter and offset
    	print(f"Jitter + offset Result: Jitter      : {np.exp(solnC.x[0]):.6f}")
    	print(f"Jitter + offset Result: offset      : {solnC.x[1]:.6f}")
    	print(f"Final log-likelihood Jitter + offset: {-solnC.fun}")
    	# Print results for the GP model
    	print(f"Jitter + offset + GP SHO model Result: S0          : {np.exp(soln.x[0]):.6f}")
    	print(f"Jitter + offset + GP SHO model Result: Q           : {np.exp(soln.x[1]):.6f}")
    	print(f"Jitter + offset + GP SHO model Result: omega0 [1/d]: {np.exp(soln.x[2]):.6f}")
    	print(f"Jitter + offset + GP SHO model Result: P0  [d]     : {2. * np.pi / np.exp(soln.x[2]):.6f}")
    	print(f"Jitter + offset + GP SHO model Result: P   [d]     : {2. * np.pi / np.sqrt(np.exp(soln.x[2]) ** 2 - 1 / tau / tau):.6f}")
    	print(f"Jitter + offset + GP SHO model Result: tau [d]     : {tau:.6f}")
    	print(f"Jitter + offset + GP SHO model Result: Jitter      : {np.exp(soln.x[3]):.6f}")
    	print(f"Jitter + offset + GP SHO model Result: offset      : {soln.x[4]:.6f}")
    	print(f"Final log-likelihood Jitter + offset + GP SHO model: {-soln.fun}")
    	
    	# Print results for the residuals
    	print("Residuals")
    	print(f"Best sine frequency : {glsres.hpstat['fbest']:.6f} +/- {glsres.hpstat['f_err']:.6f}")
    	print(f"Best sine period    : {1. / glsres.hpstat['fbest']:.6f} +/- {glsres.hpstat['Psin_err']:.6f}")
    	print(f"Amplitude           : {glsres.hpstat['amp']:.6f} +/- {glsres.hpstat['amp_err']:.6f}")
    	
    	# Write data snd forecast results to the file
    	base_name = os.path.basename(data_file)
    	# Construct file paths in the "output" folder
    	output_folder = "output"
    	
    	# Create the "output" folder if it doesn't exist
    	if not os.path.exists(output_folder):
    		os.makedirs(output_folder)
    		
    	data_file = os.path.join(output_folder, f"normalised_data_{base_name}")
    	forecast_file = os.path.join(output_folder, f"forecast_{base_name}")




    	input_data = np.column_stack((x + tnull1, y, yerr))
    	np.savetxt(data_file, input_data, fmt='%s', delimiter='\t')
    	
    	forecast_model = np.column_stack((t + tnull1, mutS, std))
    	np.savetxt(forecast_file, forecast_model, fmt='%s', delimiter='\t')

    	print(f"Results saved to '{data_file}' and '{forecast_file}'")
    	print("End\n")

        
    	
    #### Define plot 
    def plot_sine_model(x, t, tnull, mutS, std, y, yerr, ylab):
    	"""
    	Plot the results of the sine model.
    	Parameters:
    		x (array): The input data x.
    		t (array): The time array for prediction.
    		tnull (float): Time offset.
    		mutS (array): Predicted mean of the sine model.
    		std (array): Standard deviation of the sine model predictions.
    		y (array): Observed y values.
    		yerr (array): Error in the y values.
    		ylab (str): Label for the y-axis.
    	"""
    	font0 = FontProperties()
    	font = font0.copy()
    	font.set_family('times')
    	font.set_weight('bold')
    	font.set_size('medium')
    	tnull1 = tnull - 2400000
    	# Plotting
    	plt.errorbar(x + tnull1, y, yerr, fmt=".k", capsize=0, color='C1', alpha=0.1)
    	plt.plot(x + tnull1, y, '.')
    	plt.plot(t + tnull1, mutS, color='b')
    	plt.fill_between(t + tnull1, mutS + std, mutS - std, color='b', alpha=0.3, edgecolor="none")
    	plt.xticks(fontproperties=font, fontsize=12)
    	plt.yticks(fontproperties=font, fontsize=12)
    	plt.minorticks_on()
    	plt.xlabel("Julian Date [d]", fontproperties=font, fontsize=12)
    	plt.ylabel(ylab, fontproperties=font, fontsize=12)
    	plt.title('Sine model', fontproperties=font, fontsize=12, color='red')
    	plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
    	plt.tight_layout()
    	plt.show()


    # Read input file RV data
    Data = filelist
    nf = 1
    if not single:
        Data = np.genfromtxt(filelist, dtype='string')
        try:
            nf = len(Data)
        except:
            nf = 1

    if nf == 1:
        Data = np.append(Data, " ")

    factor = 0.
    if mediansub:
        factor = 1.

    # Loop over all files
    for k in range(nf):
        print(" %s" % (empty))
        print(" %s" % (empty))
        print(" %s" % (empty))
        print("Analysing file %s" % (Data[k]))
        print(" %s" % (empty))

        try:
            cont = np.genfromtxt(Data[k])
            x = cont[:, xcol]
            y = cont[:, ycol] - np.median(cont[:, ycol]) * factor
            yerr = cont[:, yerrcol]
        except:
            cont = np.genfromtxt(Data[k])
            x = cont[:, xcol]
            y = cont[:, ycol] - np.median(cont[:, ycol]) * factor
            yerr = np.zeros(len(y)) + np.std(y[10:20])

        if name != "":
            Data[k] = name

        # Remove non-finite values
        ind = np.isfinite(y, where=True)
        x = x[ind]
        y = y[ind]
        yerr = yerr[ind]
        ind = np.isfinite(yerr, where=True)
        x = x[ind]
        y = y[ind]
        yerr = yerr[ind]
        
        def remove_outliers(x, y, yerr, sigclip):
        	for i in range(7):
        		ind = np.where(np.abs(y - np.mean(y)) - np.std(y) < sigclip * np.std(y))
        		x = x[ind] * 1.
        		y = y[ind] * 1.
        		yerr = yerr[ind]
        	return x, y, yerr
        	
        x, y, yerr = remove_outliers(x, y, yerr, sigclip)


        # Bin data for rectification
        if divbysmooth > 0:
            sbin = bin * divbysmooth
            binrange = (np.int(np.min(x) - 1), np.int(np.max(x) + 1))
            numbins = math.floor((np.int(np.max(x) + 1) - np.int(np.min(x))) / sbin)
            bin_meansx = np.histogram(x, range=binrange, bins=numbins, weights=x)[0] / np.histogram(x, range=binrange, bins=numbins)[0]
            bin_meansy = np.histogram(x, range=binrange, bins=numbins, weights=y)[0] / np.histogram(x, range=binrange, bins=numbins)[0]
            bin_weights = np.histogram(x, range=binrange, bins=numbins, weights=yerr)[0] / np.histogram(x, range=binrange, bins=numbins)[0]
            bin_nums = np.sqrt(np.histogram(x, range=binrange, bins=numbins)[0])
            ind = np.where(np.isfinite(bin_meansx))
            y = y - np.interp(x, bin_meansx[ind], bin_meansy[ind])

        # Remove outliers
        x, y, yerr = remove_outliers(x, y, yerr, sigclip)

        # Sort data
        tnull = tzero * 1.
        if tzero == 0:
            tnull = np.median(x)
        ind = np.argsort(x)
        x = x[ind] - tnull
        y = y[ind]
        yerr = yerr[ind]

        # Bin data
        if bin != 0:
            binrange = (np.int(np.min(x) - 1), np.int(np.max(x) + 1))
            numbins = np.int((np.int(np.max(x) + 1) - np.int(np.min(x))) / (bin * 1.))
            bin_meansx = np.histogram(x, range=binrange, bins=numbins, weights=x)[0] / np.histogram(x, range=binrange, bins=numbins)[0]
            bin_meansy = np.histogram(x, range=binrange, bins=numbins, weights=y)[0] / np.histogram(x, range=binrange, bins=numbins)[0]
            bin_weights = np.histogram(x, range=binrange, bins=numbins, weights=yerr)[0] / np.histogram(x, range=binrange, bins=numbins)[0]
            bin_nums = np.sqrt(np.histogram(x, range=binrange, bins=numbins)[0])
            ind = np.where(np.isfinite(bin_meansx))
            x = bin_meansx[ind]
            y = bin_meansy[ind]
            yerr = bin_weights[ind] / bin_nums[ind]

                #######################################################################

        # Final part of the code for optimization and plotting

        for redo in range(nredo):
            bounds = dict(A=(np.min(y), np.max(y)), B=(-(np.max(y) - np.min(y)) / (np.max(x) - np.min(x)),
                                                       (np.max(y) - np.min(y)) / (np.max(x) - np.min(x))))
            mean_modelC = MeanModelConst(A=np.median(y), B=0, C=0, D=0, bounds=bounds)
            if ndim < 3:
                mean_modelC.freeze_parameter("D")
            if ndim < 2:
                mean_modelC.freeze_parameter("C")
            if ndim < 1:
                mean_modelC.freeze_parameter("B")

            # Defining faplevels at 10, 5, and 1%
            fapLevels = np.array([0.1, 0.05, 0.01])

            # Calculate Generalized Lomb-Scargle periodogram
            glsin = pyPeriod.Gls((x, y, yerr), verbose=True, fbeg=fbeg, fend=fend, norm='ZK')
            plevels = glsin.powerLevel(fapLevels)

            # Mean model Sine for GP
            p = glsin.hpstat

            # GP model for jitter
            bounds = dict(log_sigma=(-30, 50))
            kernelJitter = celerite.terms.JitterTerm(log_sigma=np.log(np.median(yerr)), bounds=bounds)
            kernelC = kernelJitter
            gpC = celerite.GP(kernelC, mean=mean_modelC, fit_mean=True)
            gpC.compute(x, yerr)

            # Fit for the maximum likelihood parameters Const model (polynomial of degree ndim)
            initial_params = gpC.get_parameter_vector()
            bounds = gpC.get_parameter_bounds()
            solnC = minimize(neg_log_likeC, initial_params, method="L-BFGS-B", bounds=bounds, args=(y, gpC))
            y = y - polyval(x, solnC.x[range(1, len(solnC.x))])
            coeff = solnC.x[range(1, len(solnC.x))]
            if ndim < 3:
                coeff = np.append(coeff, 0.)
            if ndim < 2:
                coeff = np.append(coeff, 0.)
            if ndim < 1:
                coeff = np.append(coeff, 0.)

            bounds = dict(Amp=(0.001 * np.std(y), 1000. * np.std(y)), Per=(1. / fend, 1. / fbeg),
                          Phase=(np.min(x) - 1. / fbeg, np.max(x) + 1. / fbeg),
                          offset=(-1e5, 1e5), slope=(-1e5, 1e5), quad=(-1e5, 1e5), cube=(-1e5, 1e5))

            mean_modelS = MeanModelSine(Amp=p["amp"], Per=1. / p["fbest"],
                                        Phase=(100000. / p["fbest"] + p["T0"]) % (1. / p["fbest"]),
                                        offset=coeff[0], slope=coeff[1], quad=coeff[2], cube=coeff[3], bounds=bounds)

            if ndim < 3:
                mean_modelS.freeze_parameter("quad")
            if ndim < 2:
                mean_modelS.freeze_parameter("cube")
            if ndim < 1:
                mean_modelS.freeze_parameter("slope")

            kernelS = kernelJitter
            gpS = celerite.GP(kernelS, mean=mean_modelS, fit_mean=True)
            gpS.compute(x, yerr)

            # GP model: damped oscillator
            w0 = 2. * np.pi * glsin.hpstat["fbest"]
            tau = 1. / glsin.hpstat["fbest"]
            Q = np.sqrt(np.abs((tau * tau * w0 * w0 - 1.) * 0.25))
            S0 = np.std(y)
            bounds = dict(log_S0=(-15, 15), log_Q=(-3, 5),
                          log_omega0=(np.log(2. * np.pi * fbeg), np.log(2. * np.pi * fend)))
            kernelSHOT = terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q),
                                       log_omega0=np.log(w0), bounds=bounds)

            bounds = dict(A=(np.min(y), np.max(y)), B=(-1e5, 1e5), C=(-1e5, 1e5), D=(-1e5, 1e5))
            mean_modelC = MeanModelConst(A=coeff[0], B=coeff[1], C=coeff[2], D=coeff[3], bounds=bounds)

            kernel = kernelSHOT + kernelJitter
            gp = celerite.GP(kernel, mean=mean_modelC, fit_mean=True)
            gp.compute(x, yerr)

            #######################################################################

            # Fit for the maximum likelihood parameters GP model
            initial_params = gp.get_parameter_vector()
            bounds = gp.get_parameter_bounds()
            soln = minimize(neg_log_like, initial_params, method="L-BFGS-B", bounds=bounds, args=(y, gp))

            # Fit for the maximum likelihood parameters Sine model
            initial_params = gpS.get_parameter_vector()
            bounds = gpS.get_parameter_bounds()
            solnS = minimize(neg_log_likeS, initial_params, method="L-BFGS-B", bounds=bounds, args=(y, gpS))

            #######################################################################

            # Plot results
            t = np.linspace(np.min(x) - 20000, np.max(x) + 20000, numtimes)
            gp.set_parameter_vector(soln.x)
            mut, var = gp.predict(y, t, return_var=True)
            std = np.sqrt(var)
            mux, varx = gp.predict(y, x, return_var=True)

            gpS.set_parameter_vector(solnS.x)
            mutS, var = gpS.predict(y, t, return_var=True)
            stdS = np.sqrt(var)
            muxS, varx = gpS.predict(y, x, return_var=True)
            polyS = polyval(x, solnS.x[range(4, len(solnS.x))])
            polytS = polyval(t, solnS.x[range(4, len(solnS.x))])

            p = glsin.hpstat
            glsres = pyPeriod.Gls((x, y - mux, yerr), verbose=glsverbose, fbeg=fbeg, fend=fend)
            
            
            font0 = FontProperties()
            font = font0.copy()
            font.set_family('times')
            font.set_weight('bold')
            font.set_size('medium')
            
            # Calculate tnull1 based on your logic
            tnull1 = tzero * 1.  # Or calculate it as needed
            # Call print_analysis_results with tnull1
            #print_analysis_results(Data[k], x, t, tnull1, mutS, stdS, y, yerr)
            print_results(glsin, solnC, soln, glsres, Data[k])
            plot_sine_model(x, t, tnull1, mutS, std, y, yerr, ylab)  # Call the new function


    print("End")

data_folder = "data"

# List all files in the data folder
data_files = os.listdir(data_folder)

# List of input files
input_files = ["Solar_S_index_subset1.txt", "Solar_S_index_subset2.txt", "Solar_S_index_subset3.txt", "Solar_S_index_subset4.txt", "Solar_S_index_subset5.txt","Solar_mtwilson_calcium_hk.txt"]

# Iterate over each input file in the data folder
for input_file in data_files:
    # Create the full path to the input file
    full_path = os.path.join(data_folder, input_file)
    # Check if the item in the folder is a file (not a subfolder)
    Do_GP(full_path, single=True, divbysmooth=0, sigclip=3, fbeg=1 / (20 * 365), fend=1 / 2800, bin=0.001, tzero=0, ndim=3, nredo=1)
    print(f"Analysis completed for {input_file}")


        
