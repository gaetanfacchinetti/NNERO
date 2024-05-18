import copy
import scipy 
import random
import warnings

import py21cmcast as p21c

import numpy as np

from os import listdir
from os.path import isfile, join, abspath

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


# We need to store somewhere
# - the parameter list
# - the parameter ranges
# - the redshifts on which we train


class MetaData:
    
    """
        class Metadata

    Contain data about the data. ID card of the data.
    Is saved with the neural-network models to use them in a standalone mode once they are trained.

    """

    def __init__(self, path : str, redshifts : np.ndarray = None, param_names : list = None):

        # directory was the data is stored
        self._directory = abspath(path)

        # define a default redshift array on which to make the predictions
        # define the labels of the regressor
        if redshifts is None:
            self._redshifts = np.linspace(4, 35, 32)
        else:
            self._redshifts = redshifts

        # define a default parameter list to run on
        # features of the neural networks
        if param_names is None:
            self._param_names = ['F_STAR10', 'ALPHA_STAR', 't_STAR', 'F_ESC10', 'ALPHA_ESC', 'M_TURN', 'Omch2', 'Ombh2', 'hlittle', 'Ln_1010_As', 'POWER_INDEX', 'M_WDM']
        else : 
            self._param_names = param_names

        # define the length of the lists
        self._nredshifts = len(self.redshifts)
        self._nparams = len(self.param_names)
        
        # The rest will be set later when data is used
        self._param_ranges  = None # range of parameters
        self._nsamples      = None # number of samples
        self._ntrain        = None # number of train point
        self._ntest         = None # number of test point
        self._ntrain_valid  = None # number of valid training point
        self._ntest_valid   = None # number of valid test point
        self._frac_test     = None # fraction of testing point

        # repartition of the data used
        self._id                = None
        self._id_valid          = None
        self._id_train          = None
        self._id_test           = None
        self._id_train_valid    = None
        self._id_test_valid     = None
        self._index_train       = None
        self._index_test        = None
        self._index_valid       = None
        self._index_train_valid = None
        self._index_test_valid  = None

        self._index_valid_reio   = None
        self._index_valid_noreio = None
        self._index_excluded     = None

        self._index_train_valid_reio   = None
        self._index_train_valid_noreio = None
        self._index_test_valid_reio    = None
        self._index_test_valid_noreio  = None


        self._frozen = False

    
    def __getattr__(self, name):
        try:
            return self.__dict__['_' + name]
        except KeyError:
            msg = "'{0}' object has no attribute '{1}'"
            raise AttributeError(msg.format(type(self).__name__, '_' + name))
        
    def __setattr__(self, name: str, value) -> None:
        
        # forbid modification of attributes not starting with '_'
        if not name.startswith('_'):
            msg = "Cannot modify attribute '{1}' of object '{0}'"
            raise AttributeError(msg.format(type(self).__name__, name))
        
        self.__dict__[name] = value

    # saving the metadata
    def save(self, filename):

        with open(filename + '_metadata.npz', 'wb') as file:
            np.savez(file, 
                        directory   = self.directory,
                        redshifts    = self.redshifts,
                        param_names  = self.param_names,
                        nredshifts   = self.nredshifts,
                        nparams      = self.nparams,
                        param_ranges = self.param_ranges,
                        nsamples     = self.nsamples,
                        ntrain       = self.ntrain,
                        ntest        = self.ntest,
                        ntrain_valid = self.ntrain_valid,
                        ntest_valid  = self.ntest_valid,
                        frac_test    = self.frac_test,
                        id                = self.id,
                        id_valid          = self.id_valid,
                        id_train          = self.id_train,
                        id_test           = self.id_test,
                        id_train_valid    = self.id_train_valid,
                        id_test_valid     = self.id_test_valid,
                        index_train       = self.index_train,
                        index_test        = self.index_test,
                        index_valid       = self.index_valid,
                        index_train_valid = self.index_train_valid,
                        index_test_valid  = self.index_test_valid,
                        index_valid_reio   = self.index_valid_reio,
                        index_valid_noreio = self.index_valid_noreio,
                        index_excluded     = self.index_excluded,
                        index_train_valid_reio   = self.index_train_valid_reio,
                        index_train_valid_noreio = self.index_train_valid_noreio,
                        index_test_valid_reio    = self.index_test_valid_reio,
                        index_test_valid_noreio  = self.index_test_valid_noreio,
                        frozen              = self.frozen
    )

    @classmethod    
    def load(cls, filename):

        with open(filename + '_metadata.npz', 'rb') as file:
            data = np.load(file, allow_pickle=True)
            
            # get the user input metadata
            directory    = str(data['directory'])
            redshifts    = data['redshifts']
            param_names  = list(data['param_names'])

            metadata = MetaData(directory, redshifts, param_names)
            
            metadata._nredshifts   = data['nredshifts']
            metadata._nparams      = data['nparams']
            metadata._param_ranges = data['param_ranges'].tolist()
            metadata._nsamples     = data['nsamples']
            metadata._ntrain       = data['ntrain']
            metadata._ntest        = data['ntest']
            metadata._ntrain_valid = data['ntrain_valid']
            metadata._ntest_valid  = data['ntest_valid']
            metadata._frac_test    = data['frac_test']

            metadata._id                  = data['id']
            metadata._id_valid            = data['id_valid']
            metadata._id_train            = data['id_train']
            metadata._id_test             = data['id_test']
            metadata._id_train_valid      = data['id_train_valid']
            metadata._id_test_valid       = data['id_test_valid']
            metadata._index_train         = data['index_train']
            metadata._index_test          = data['index_test']
            metadata._index_valid         = data['index_valid']
            metadata._index_train_valid   = data['index_train_valid']
            metadata._index_test_valid    = data['index_test_valid']
            metadata._index_valid_noreio   = data['index_valid_noreio']
            metadata._index_valid_reio    = data['index_valid_reio']
            metadata._index_excluded      = data['index_excluded']

            metadata._index_train_valid_reio   = data['index_train_valid_reio']
            metadata._index_train_valid_noreio = data['index_train_valid_noreio']
            metadata._index_test_valid_reio    = data['index_test_valid_reio']
            metadata._index_test_valid_noreio  = data['index_test_valid_noreio']

            metadata._frozen = data['frozen']
            
        return metadata


    def __eq__(self, other):
        
        for key, value in self.__dict__.items():
            other_value = other.__dict__[key]

            try: 
                if (key in ["_redshifts", "_param_ranges", "_param_names", "_id", '_id_valid', "_id_train", "_id_test",
                    "_id_train_valid", "_id_test_valid", "_index_train", "_index_test", "_index_valid", "_index_train_valid",
                        "_index_test_valid", '_index_valid_noreio', '_index_valid_reio', '_index_excluded', 
                    '_index_train_valid_reio', '_index_train_valid_noreio', '_index_test_valid_reio', '_index_test_valid_noreio',]):
                
                    if (len(value) != len(other_value)):
                        return False
                    if np.any(value != other_value):
                        return False
                else:
                    if value != other_value:
                        return False
            except ValueError as e:
                print("Value error for : ", key)
                raise e
            
        return True


    @property
    def directory(self):
        return self._directory


def true_to_uniform(x, min, max):
    assert (min <= max), "The minimum value is bigger than the maximum one" 
    return (x - min) / (max - min)

def uniform_to_true(x, min, max):
    assert (min <= max), "The minimum value is bigger than the maximum one" 
    return (max - min) * x + min



class DataBase:

    def __init__(self, metadata, *, frac_test:float = 0.2) :

        self._metadata = metadata     
        self._metadata._frac_test = frac_test

        # load the database files (of the parameters and corresponding )
        _data  = np.load(self.metadata.directory + '/Database.npz', allow_pickle=True)
        #_udata = np.load(self.md.directory + '/Uniform.npz',  allow_pickle=True)

        # get the value of the parameters drawn
        self._data_cosmo = _data.get("params_cosmo")
        self._data_astro = _data.get("params_astro")
        
        # get the min and max values of the parameters
        # store it in the metadata object
        self.metadata._param_ranges = _data.get("params_range", None)
        if self.metadata._param_ranges is not None:
            self.metadata._param_ranges = self.metadata._param_ranges.tolist()

        # get the uniform draws resulting in the drawn values
        #self._udraws = _udata.get("draws")

        # construct a dictionnary of all parameters and an array ordered
        # in termps of the params_keys list defined above
        self._data_dict = []
        self._data_arr  = np.zeros((len(self.data_astro), self.metadata.nparams))
        for i in range(0, len(self.data_astro)):
            self._data_dict.append( ( self.data_astro[i] | self.data_cosmo[i] ))
            
            # define 
            for ikey, key in enumerate(self.metadata.param_names):
                self._data_arr[i, ikey] = self._data_dict[int(i)][key]

    
        if self.metadata.frozen is False:
            self.read_id() 
            self.prepare()
        
        self.create_dataset()
        self.split_dataset()

        # freeze the metadata now than the dataset is initialised
        self.metadata._frozen = True

        print("Database initialised :")
        print("--------------------------------")
        print("| n_sample :", self.metadata.nsamples)
        print("--------------------------------")
        print("| total n_train :", self.metadata.ntrain)
        print("| total n_test :", self.metadata.ntest)
        print("--------------------------------")
        print("| n_valid_train :", self.metadata.ntrain_valid)
        print("|  -> reionized (at z=5.9)     : ", len(self.metadata.index_train_valid_reio))
        print("|  -> not reionized (at z=5.9) : ", len(self.metadata.index_train_valid_noreio))
        print("| n_valid_test :", self.metadata.ntest_valid)
        print("|  -> reionized (at z=5.9)     : ", len(self.metadata.index_test_valid_reio))
        print("|  -> not reionized (at z=5.9) : ", len(self.metadata.index_test_valid_noreio))
        print("--------------------------------")
        

    @property
    def metadata(self):
        return self._metadata

    def read_id(self) -> None:

        # get all the files that have succesfully run in the folder
        onlyfiles = [f for f in listdir(self.metadata.directory + '/cache/') if isfile(join(self.metadata.directory + '/cache/', f))]
        
        # make a list of the files we need need (only take the tables)
        goodfiles = [file for file in onlyfiles if file[0:5] == 'Table']

        # get the id of all the successfull runs
        self.metadata._id = np.array([], dtype = np.int64)

        # get the id of the good files
        for file in goodfiles:
            
            try:
                
                # get the new id and the random seed used for the run
                new_id = np.int64(file.split('.')[0].split('_')[-1])
                run_rs = np.int64(file.split('.')[0].split('_')[-2][2:])
                
                # make sure the files are not already counted or corrupted
                if new_id in self.metadata.id:
                    print('ERROR:', new_id)
                elif "Param_Lightcone_rs" + str(run_rs) + "_" + str(new_id) + ".h5.pkl" not in onlyfiles:
                    print("ERROR missing info for ", new_id)
                else:
                    self.metadata._id = np.append(self.metadata.id, new_id)
            
            except Exception as e:
                warnings.warn("Error " + str(e) + " has occured for")

        # sort the id in increasing order
        self.metadata.id.sort()



    def prepare(self) -> None:
        """ this function sets much of the metadata """

        # define the size of the sample
        self.metadata._nsamples = len(self.metadata.id)
        self.metadata._ntest    = int(self.metadata.frac_test * self.metadata.nsamples)
        self.metadata._ntrain   = self.metadata.nsamples - self.metadata.ntest

        # define the identification number of the test and train subdatasests
        self.metadata._id_test  = np.zeros(self.metadata.ntest, dtype = np.int64)
        self.metadata._id_train = copy.copy(self.metadata.id)

        for i in range(0, self.metadata.ntest):
            self.metadata._id_test[i] = random.choice(self.metadata.id_train)
            self.metadata._id_train = np.delete(self.metadata.id_train, np.where(self.metadata.id_train == self.metadata.id_test[i])[0][0])

        # sort the id of the test dataset
        self.metadata.id_test.sort()
        self.metadata.id_train.sort()

        # define the index (position in the array of sample, different from id)
        # if all runs work then id and indices are the same, in practice they are not
        self.metadata._index_train = np.array([i for i in range(0, self.metadata.nsamples) if self.metadata.id[i] in self.metadata.id_train])
        self.metadata._index_test  = np.array([i for i in range(0, self.metadata.nsamples) if self.metadata.id[i] in self.metadata.id_test])
        


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



    def create_dataset(self):
        """ this function does not modify the metadata but uses what its initialised attributes from prepare() """

        # full datasets containing all the data
        self._x_data = np.zeros((self.metadata.nsamples, self.metadata.nparams))
        self._u_data = np.zeros((self.metadata.nsamples, self.metadata.nparams))
        self._y_data = np.zeros((self.metadata.nsamples, self.metadata.nredshifts))
        self._rtau_data = np.zeros(self.metadata.nsamples) # regularised value of tau
        self._ttau_data = np.zeros(self.metadata.nsamples) # true value of tau
        self._McGreer_data = np.zeros(self.metadata.nsamples)
        
        for i in range(0, self.metadata.nsamples) :

            run = p21c.Run(self.metadata.directory, "Lightcone_rs1993_" + str(int(self.metadata.id[i])) + ".h5")
            z_glob, r_xHIIdb = self.regularised_data(run)
            xHIIdb = scipy.interpolate.interp1d(z_glob, r_xHIIdb)(self.metadata.redshifts)

            ombh2   = self.data_dict[int(self.metadata.id[i])]['Ombh2']
            omch2   = self.data_dict[int(self.metadata.id[i])]['Omch2']
            hlittle = self.data_dict[int(self.metadata.id[i])]['hlittle']

            self._ttau_data[i] = tau_ion(run.z_glob, run.xHIIdb, ombh2, ombh2 + omch2, hlittle)
            self._rtau_data[i] = tau_ion(z_glob, r_xHIIdb, ombh2, ombh2 + omch2, hlittle)

            self._McGreer_data[i] = scipy.interpolate.interp1d(z_glob, r_xHIIdb)(5.9)
    
            # define the features
            for ikey in range(0, self.metadata.nparams):

                self._x_data[i, ikey] =  self.data_arr[int(self.metadata.id[i]), ikey]
                
                min = self.metadata.param_ranges[self.metadata.param_names[ikey]][0]
                max = self.metadata.param_ranges[self.metadata.param_names[ikey]][1]
                
                self._u_data[i, ikey] = true_to_uniform(self._x_data[i, ikey], min, max) 
                #x[i, ikey] = self._udraws[int(id[i])][key]
            
            # define the labels for the regressor
            self._y_data[i, :] = xHIIdb


    def split_dataset(self): 
        
        # we only resplit everything if we need to
        if self.metadata.frozen is False : 

            self.metadata._index_valid    = np.where(self._McGreer_data  > 0.69)[0]
            self.metadata._id_valid       = self.metadata.id[self.metadata.index_valid]
            
            self.metadata._index_train_valid  = np.intersect1d(self.metadata.index_valid, self.metadata.index_train)
            self.metadata._index_test_valid   = np.intersect1d(self.metadata.index_valid, self.metadata.index_test)
            self.metadata._id_train_valid     = np.intersect1d(self.metadata.id_valid, self.metadata.id_train)
            self.metadata._id_test_valid      = np.intersect1d(self.metadata.id_valid, self.metadata.id_test)

            # create the classifier data
            self.metadata._index_valid_reio    = np.where(self._McGreer_data  > 0.99)[0]
            self.metadata._index_valid_noreio  = np.intersect1d(np.where(self._McGreer_data  <= 0.99)[0], np.where(self._McGreer_data  > 0.69)[0])
            self.metadata._index_excluded      = np.where(self._McGreer_data  <= 0.69)[0]
        
            self.metadata._index_train_valid_reio   = np.intersect1d(self.metadata.index_valid_reio, self.metadata.index_train)
            self.metadata._index_train_valid_noreio = np.intersect1d(self.metadata.index_valid_noreio, self.metadata.index_train)
            self.metadata._index_test_valid_reio    = np.intersect1d(self.metadata.index_valid_reio, self.metadata.index_test)
            self.metadata._index_test_valid_noreio  = np.intersect1d(self.metadata.index_valid_noreio, self.metadata.index_test)

            self.metadata._ntrain_valid = len(self.metadata.index_train_valid)
            self.metadata._ntest_valid  = len(self.metadata.index_test_valid)

        self._c_data = np.zeros((self.metadata.nsamples, 3))
        self._c_data[self.metadata.index_valid_reio, :]  = [1, 0, 0]
        self._c_data[self.metadata.index_valid_noreio, :] = [0, 1, 0]
        self._c_data[self.metadata.index_excluded, :] = [0, 0, 1]

        self._x_data_valid = self._x_data[self.metadata.index_valid, :]
        self._y_data_valid = self._y_data[self.metadata.index_valid, :]
        self._u_data_valid = self._u_data[self.metadata.index_valid, :]
        self._ttau_data_valid = self._ttau_data[self.metadata.index_valid]
        self._rtau_data_valid = self._rtau_data[self.metadata.index_valid]

        self._u_data_valid_noreio = self._u_data[self.metadata.index_valid_noreio, :]


        # training datasets

        self._x_train = self._x_data[self.metadata.index_train, :]
        self._u_train = self._u_data[self.metadata.index_train, :]
        self._y_train = self._y_data[self.metadata.index_train, :]
        self._c_train = self._c_data[self.metadata.index_train, :]

        self._x_train_valid = self._x_data[self.metadata.index_train_valid, :]
        self._u_train_valid = self._u_data[self.metadata.index_train_valid, :]
        self._y_train_valid = self._y_data[self.metadata.index_train_valid, :]
        self._rtau_train_valid = self._rtau_data[self.metadata.index_train_valid]
        self._ttau_train_valid = self._ttau_data[self.metadata.index_train_valid]

        self._u_train_valid_noreio = self._u_data[self.metadata.index_train_valid_noreio, :]

        self._McGreer_train_valid = self._McGreer_data[self.metadata.index_train_valid]
        self._McGreer_train_valid_noreio = self._McGreer_data[self.metadata.index_train_valid_noreio]
        
        # test datasets

        self._x_test = self._x_data[self.metadata.index_test, :]
        self._u_test = self._u_data[self.metadata.index_test, :]
        self._y_test = self._y_data[self.metadata.index_test, :]
        self._c_test = self._c_data[self.metadata.index_test, :]

        self._x_test_valid = self._x_data[self.metadata.index_test_valid, :]
        self._u_test_valid = self._u_data[self.metadata.index_test_valid, :]
        self._y_test_valid = self._y_data[self.metadata.index_test_valid, :]
        self._rtau_test_valid = self._rtau_data[self.metadata.index_test_valid]
        self._ttau_test_valid = self._ttau_data[self.metadata.index_test_valid]

        self._u_test_valid_noreio = self._u_data[self.metadata.index_test_valid_noreio, :]

        self._McGreer_test_valid = self._McGreer_data[self.metadata.index_test_valid]
        self._McGreer_test_valid_noreio = self._McGreer_data[self.metadata.index_test_valid_noreio]

        
        
   

    def __getattr__(self, name):
        try:
            return self.__dict__['_' + name]
        except KeyError:
            msg = "'{0}' object has no attribute '{1}'"
            raise AttributeError(msg.format(type(self).__name__, '_' + name))
        

    def __setattr__(self, name: str, value) -> None:
        
        # forbid modification of attributes not starting with '_'
        
        if not name.startswith('_'):
            msg = "Cannot modify attribute '{1}' of object '{0}'"
            raise AttributeError(msg.format(type(self).__name__, name))
        
        self.__dict__[name] = value


    #
    #
    #
    #
    # create two arrays of the parameters values and the valid parameters values
    def create_params_dataset(self) -> None:
        
        id_valid = []
        id_valid_tau_Planck_3sigma = []

        for i in range(0, self._n_sample) :

            run = p21c.Run(self.metadata.directory, "Lightcone_rs1993_" + str(int(self._id[i])) + ".h5")
            z_glob, r_xHIIdb = self.regularised_data(run)

            val_McGreer = scipy.interpolate.interp1d(z_glob, r_xHIIdb)(5.9)

            # define the features
            for ikey, key in enumerate(self._params_keys):
                self._params[i, ikey] = self._data_dict[int(self._id[i])][key]

            # get only the valid runs
            if val_McGreer > 0.69: # McGreer+15 bound at 5 sigma
                id_valid.append(i)

                tau = tau_ion(z_glob, r_xHIIdb, self._data_dict[i]['Ombh2'], self._data_dict[i]['Ombh2'] + self._data_dict[i]['Omch2'], self._data_dict[i]['hlittle'])

                if tau < (0.0561 + 3*0.0071) and tau > (0.0561 - 3*0.0071):
                    id_valid_tau_Planck_3sigma.append(i)
            
        self._params_valid = self._params[id_valid, :]
        self._params_valid_tau_Planck_3sigma = self._params[id_valid_tau_Planck_3sigma, :]
    