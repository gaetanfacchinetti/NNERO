import copy
import scipy 
import random
import warnings

import py21cmcast as p21c

import numpy as np

from os import listdir
from os.path import isfile, join

import scipy
import scipy.integrate


# need to add more stuff to what we save from the database (like the redshift bins predicted by the regressor)
# (better coupling between database and neural network)

# need to add tests of the neural network

# need to finish implementing tau_ion


MPC_TO_M = 3.08567758128e+22
MSUN_TO_KG = 1.989e+30
KM_TO_MPC = 1/MPC_TO_M * 1e+3
MASS_PROTON = 1.67262192e-27 # in kg
MASS_HELIUM = 6.6464731e-27  # in kg
SIGMA_THOMSON = 6.6524587321e-29 # in m^2
C_LIGHT = 299792458 # in m / s
Y_He = 0.24


def tau_ion(z, xHIIdb, ombh2, ommh2, h):
    """ custom function to compute the optical depth to reionization """
    
    rho_b = 2.7754e+11 * ombh2 * MSUN_TO_KG / MPC_TO_M**3 # in kg / m^3
    n_b = rho_b / MASS_PROTON / (1 + Y_He / 4 * (MASS_HELIUM/MASS_PROTON -1)) # in 1/m^3

    small_z = np.linspace(0, z[0], 50)

    Omega_m = ommh2 / h / h
    Omega_L = 1.0 - Omega_m

    res_1 = scipy.integrate.trapezoid((1 + z)**2 / np.sqrt(Omega_L + Omega_m * (1+z)**3) * xHIIdb, z)
    res_2 = scipy.integrate.trapezoid((1 + small_z)**2 / np.sqrt(Omega_L + Omega_m * (1+small_z)**3), small_z)

    pref = C_LIGHT * SIGMA_THOMSON * n_b / (100 * KM_TO_MPC * h)

    return pref * (res_1+res_2)


LABELS_TO_PLOT = {'hlittle' : r'$h$', 'Ln_1010_As' : r'$\ln(10^{10}A_{\rm s})$', 'F_STAR10' : r'$\log_{10}(f_{\star, 10})$',
                  'ALPHA_STAR' : r'$\alpha_\star$', 't_STAR' : r'$t_\star$', 'F_ESC10' : r'$\log_{10}(f_{\rm esc, 10})$', 
                  'ALPHA_ESC' : r'$\alpha_{\rm esc}$', 'M_TURN' : r'$\log_{10}(M_{\rm TURN}/{\rm M_\odot})$', 'Omch2' : r'$\Omega_{\rm c} h^2$', 
                  'Ombh2' : r'$\Omega_{\rm b} h^2$', 'POWER_INDEX' : r'$n_{\rm s}$', 'M_WDM' : r'$m_{\rm WDM}~{\rm [keV]}$'}

def label_to_plot(label) -> None:
    if label in LABELS_TO_PLOT.keys(): 
        return LABELS_TO_PLOT[label]
    else:
        return label


class Database:

    def __init__(self, path:str, *, frac_test:float = 0.2, len_z:int = 32, z_min:float = 4.0, z_max:float = 35) :

        self._path = path

        # load the database files (of the parameters and corresponding )
        self._params  = np.load(self._path + '/Database.npz', allow_pickle=True)
        self._uparams = np.load(self._path + '/Uniform.npz', allow_pickle=True)

        # get the value of the parameters drawn
        self._params_cosmo = self._params.get("params_cosmo")
        self._params_astro = self._params.get("params_astro")
        
        # get the min and max values of the parameters
        self._params_range = self._params.get("params_range", None)

        # get the uniform draws resulting in the drawn values
        self._udraws = self._uparams.get("draws")

        # get all parameters name involved
        self._params_keys = []
        for key, _ in self._udraws[0].items():
            self._params_keys.append(key)

        # construct a dictionnary of all parameters and an array ordered
        # in termps of the params_keys list defined above
        self._params_all_dict = []
        self._params_all_arr  = np.zeros((len(self._params_astro), len(self._params_keys)))
        for i in range(0, len(self._params_astro)):
            self._params_all_dict.append( ( self._params_astro[i] | self._params_cosmo[i] ))
            
            # define 
            for ikey, key in enumerate(self._params_keys):
                self._params_all_arr[i, ikey] = self._params_all_dict[int(i)][key]

    

        self.read_id()
        self.prepare(frac_test=frac_test, len_z=len_z, z_min=z_min, z_max=z_max)
        self.create_train_dataset()
        self.create_test_dataset()

        print("Database initialised :")
        print("--------------------------------")
        print("| n_sample :", self._n_sample)
        print("| n_train :", self._n_train)
        print("| n_test :", self._n_test)
        print("--------------------------------")
        print("| n_valid_train :", self._n_valid_train)
        print("| n_valid_test :", self._n_valid_test)
        print("--------------------------------")
        

    def read_id(self) -> None:

        # get all the files that have succesfully run in the folder
        onlyfiles = [f for f in listdir(self._path + '/cache/') if isfile(join(self._path + '/cache/', f))]
        
        # make a list of the files we need need (only take the tables)
        goodfiles = []
        for file in onlyfiles:
            # only the tables should have a name that start with the letter T
            if file[0] == 'T': 
                goodfiles.append(file)

        # get the id of all the successfull runs
        self._id = np.array([], dtype = np.int64)

        # get the id of the good files
        for file in goodfiles:
            
            try:
                
                # get the new id and the random seed used for the run
                new_id = np.int64(file.split('.')[0].split('_')[-1])
                run_rs = np.int64(file.split('.')[0].split('_')[-2][2:])
                
                # make sure the files are not already counted or corrupted
                if new_id in self._id:
                    print('ERROR:', new_id)
                elif "Param_Lightcone_rs" + str(run_rs) + "_" + str(new_id) + ".h5.pkl" not in onlyfiles:
                    print("ERROR missing info for ", new_id)
                else:
                    self._id = np.append(self._id, new_id)
            
            except Exception as e:
                warnings.warn("Error " + str(e) + " has occured for")

        # sort the id in increasing order
        self._id.sort()



    def prepare(self, *, frac_test:float = 0.2, len_z:int = 32, z_min:float = 4.0, z_max:float = 35) -> None:

        self._n_sample = len(self._id)

        self._n_test  = int(frac_test * self._n_sample)
        self._n_train = self._n_sample - self._n_test

        self._id_test  = np.zeros(self._n_test, dtype = np.int64)
        self._id_train = copy.copy(self._id)

        for i in range(0, self._n_test):
            self._id_test[i] = random.choice(self._id_train)
            self._id_train = np.delete(self._id_train, np.where(self._id_train == self._id_test[i])[0][0])

        # sort the id of the test dataset
        self._id_test.sort()

        # prepare the training / test datasets
        self._len_z = len_z
        self._z_min = z_min
        self._z_max = z_max

        self._uredshift = np.linspace(0, 1, len_z)  # normalised redshift steps
        self._redshift  = (z_max - z_min) * self._uredshift + z_min

        # The training value (normalised to unity)
        self._x_train = np.zeros((self._n_train, len(self._params_keys)))
        self._y_train = np.zeros((self._n_train, len_z))
        self._x_test  = np.zeros((self._n_test, len(self._params_keys)))
        self._y_test  = np.zeros((self._n_test, len_z))

        self._y_train_class = np.zeros((self._n_train, 3))
        self._y_test_class = np.zeros((self._n_test, 3))

        # The parameters that correspond to the training values
        self._params = np.zeros((self._n_sample, len(self._params_keys)))



    def regularised_data(self, run, n_smooth:int = 10):

        n_smooth = 12
        xHIIdb = run.xHIIdb[2:]
        z_glob = run.z_glob[2:]
        z_t = z_glob[np.argmin(xHIIdb)]
        index = np.where(xHIIdb[z_glob < z_t] >= xHIIdb[-1])[0][-1]
        new_array = copy.copy(xHIIdb)
        new_array[index:-1] = xHIIdb[-1]
        new_array = np.convolve(new_array, np.ones(n_smooth)/n_smooth, mode='same')
        new_array[0:n_smooth-1] = xHIIdb[0:n_smooth-1]
        new_array[-(n_smooth-1):] = xHIIdb[-(n_smooth-1):]

        i_max = np.argmax(new_array)
        new_array[0:i_max] = np.max(new_array)

        return z_glob, new_array


    def _create_dataset(self, n, id, x, y, y_class) -> None:

        n_valid = 0
        id_valid = []
    
        for i in range(0, n) :

            run = p21c.Run(self._path, "Lightcone_rs1993_" + str(int(id[i])) + ".h5")
            z_glob, r_xHIIdb = self.regularised_data(run)
            xHIIdb = scipy.interpolate.interp1d(z_glob, r_xHIIdb)(self._redshift)

            val_McGreer = scipy.interpolate.interp1d(z_glob, r_xHIIdb)(5.9)

            # get only the valid runs
            if val_McGreer > 0.69: # McGreer+15 bound at 5 sigma
                n_valid = n_valid + 1
                id_valid.append(i)
            
            # define the labels for the classifier
            if val_McGreer > 0.99:
                y_class[i, :] = [1, 0, 0] 
            else:
                if val_McGreer > 0.69:
                    y_class[i, :] = [0, 1, 0]
                else:
                    y_class[i, :] = [0, 0, 1]

            # define the features
            for ikey, key in enumerate(self._params_keys):
                x[i, ikey] = self._udraws[int(id[i])][key]
            
            # define the labels for the regressor
            y[i, :] = xHIIdb

        x_valid = x[id_valid, :]
        y_valid = y[id_valid, :]

        return (x_valid, y_valid, n_valid, id_valid)


    def create_train_dataset(self) -> None:
        self._x_train_valid, self._y_train_valid, self._n_valid_train, self._id_valid_train = self._create_dataset(self._n_train, self._id_train, self._x_train, self._y_train, self._y_train_class)
        
    def create_test_dataset(self) -> None:
        self._x_test_valid, self._y_test_valid, self._n_valid_test, self._id_valid_test = self._create_dataset(self._n_test, self._id_test, self._x_test, self._y_test, self._y_test_class) 
        

    # create two arrays of the parameters values and the valid parameters values
    def create_params_dataset(self) -> None:
        
        id_valid = []

        for i in range(0, self._n_sample) :

            run = p21c.Run(self._path, "Lightcone_rs1993_" + str(int(self._id[i])) + ".h5")
            z_glob, r_xHIIdb = self.regularised_data(run)

            val_McGreer = scipy.interpolate.interp1d(z_glob, r_xHIIdb)(5.9)

            # get only the valid runs
            if val_McGreer > 0.69: # McGreer+15 bound at 5 sigma
                id_valid.append(i)

            # define the features
            for ikey, key in enumerate(self._params_keys):
                self._params[i, ikey] = self._params_all_dict[int(self._id[i])][key]
            
        self._params_valid = self._params[id_valid, :]
