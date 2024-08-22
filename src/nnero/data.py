import random
import numpy as np
import torch
from scipy import interpolate
from os.path import abspath, exists

from .cosmology import optical_depth_no_rad_numpy


_LABELS_TO_PLOT_ = {'hlittle' : r'$h$', 'Ln_1010_As' : r'$\ln(10^{10}A_{\rm s})$', 'F_STAR10' : r'$\log_{10}(f_{\star, 10})$',
                  'ALPHA_STAR' : r'$\alpha_\star$', 't_STAR' : r'$t_\star$', 'F_ESC10' : r'$\log_{10}(f_{\rm esc, 10})$', 
                  'ALPHA_ESC' : r'$\alpha_{\rm esc}$', 'M_TURN' : r'$\log_{10}(M_{\rm TURN}/{\rm M_\odot})$', 'Omch2' : r'$\Omega_{\rm c} h^2$', 
                  'Ombh2' : r'$\Omega_{\rm b} h^2$', 'POWER_INDEX' : r'$n_{\rm s}$', 'M_WDM' : r'$m_{\rm WDM}~{\rm [keV]}$'}

def label_to_plot(label) -> None:
    if label in _LABELS_TO_PLOT_.keys(): 
        return _LABELS_TO_PLOT_[label]
    else:
        return label


def preprocess_raw_data(file_path, *, random_seed=1994, frac_test=0.1, frac_valid=0.1):
    """
        preprocessing a raw .npz file
    """

    # start by setting the random seed
    random.seed(random_seed)

    with open(file_path, 'rb') as file:

        # data must have been stored in a numpy archive with the correct format
        data = np.load(file, allow_pickle=True)
        z_glob               = data.get('z_glob', None)
        features_run         = data.get('features_run', None)
        features_super_late  = data.get('features_late', None)
        cosmology_run        = data.get('cosmology_run', None)
        cosmology_super_late = data.get('cosmology_late', None)
        parameters_min_val   = data.get('parameters_min_val', None)
        parameters_max_val   = data.get('parameters_max_val', None)
        xHIIdb               = data.get('xHIIdb', None)
        parameters_name      = data.get('parameters_name', None)

    # --------------------------

    new_redshifts = [5.9]

    # add some values we may need for the data classification
    new_z = np.sort(new_redshifts)
    for _z in new_z: 
        if _z not in z_glob:
            pos = np.searchsorted(z_glob, _z)
            value = interpolate.interp1d(z_glob, xHIIdb, kind='slinear')(_z)
            z_glob = np.sort(np.append(z_glob, _z))
            xHIIdb = np.insert(xHIIdb, pos, value, axis=1)


    features  = np.vstack((features_run, features_super_late))
    cosmology = np.concatenate((cosmology_run, cosmology_super_late)) 

    # total number of features
    n_r  = features_run.shape[0]
    n_sl = features_super_late.shape[0]
    n_tot = n_r + n_sl

    # parameters for xHIIdb
    xHIIdb = np.vstack((xHIIdb, np.zeros((n_sl, xHIIdb.shape[1]))))

    # shuffling all the data between late and run
    r = random.sample(range(n_tot), n_tot)

    features  = features[r]
    cosmology = cosmology[r]
    xHIIdb    = xHIIdb[r]

    # data selection, only considering the "early time" reionizations
    pos = np.searchsorted(z_glob, 5.9) 

    # define the early time reionization quantities
    indices_early = np.where(xHIIdb[:, pos] > 0.69)[0]

    # divide the early data into train, test and validation datasets
    n_early = len(indices_early)
    r_early = random.sample(range(n_early), n_early)
    r_indices_early = indices_early[r_early] # shuffles indices early

    n_early_test  = int(frac_test*n_early)
    n_early_valid = int(frac_valid*n_early)

    indices_early_test  = np.sort(r_indices_early[:n_early_test])
    indices_early_valid = np.sort(r_indices_early[n_early_test:(n_early_test+n_early_valid)])
    indices_early_train = np.sort(r_indices_early[(n_early_test+n_early_valid):])

    # devide now the entire data into train, test and validation datasets
    r_indices_tot = random.sample(range(n_tot), n_tot)

    n_tot_test  = int(frac_test*n_tot)
    n_tot_valid = int(frac_valid*n_tot)

    indices_tot_test  = np.sort(r_indices_tot[:n_tot_test])
    indices_tot_valid = np.sort(r_indices_tot[n_tot_test:(n_tot_test+n_tot_valid)])
    indices_tot_train = np.sort(r_indices_tot[(n_tot_test+n_tot_valid):])

    with open(file_path[:-4] + "_preprocessed.npz", 'wb') as file:

        # data must have been stored in a numpy archive with the correct format
        np.savez(file, 
                 redshifts = z_glob, 
                 features = features, 
                 cosmology = cosmology, 
                 xHIIdb = xHIIdb, 
                 parameters_min_val = parameters_min_val, 
                 parameters_max_val = parameters_max_val, 
                 parameters_name = parameters_name,
                 indices_early_test = indices_early_test,
                 indices_early_valid = indices_early_valid,
                 indices_early_train = indices_early_train,
                 indices_tot_test = indices_tot_test,
                 indices_tot_valid = indices_tot_valid,
                 indices_tot_train = indices_tot_train,
                 random_seed = random_seed,
                 frac_test = frac_test,
                 frac_valid = frac_valid)


def true_to_uniform(x, min, max):
    assert np.all(min <= max), "The minimum value is bigger than the maximum one" 
    return (x - min) / (max - min)

def uniform_to_true(x, min, max):
    assert np.all(min <= max), "The minimum value is bigger than the maximum one" 
    return (max - min) * x + min


class MetaData:

    def __init__(self, z, parameters_name):
        self._z                  = z
        self._parameters_name    = parameters_name

    def __call__(self):
        return {'z' : self._z, 'parameters_name' : self._parameters_name}


class TorchDataset(torch.utils.data.Dataset):
    
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y


class DataSet:

    def __init__(self, 
                 file_path : str, 
                 redshifts = None, 
                 *, 
                 frac_test  = 0.1, 
                 frac_valid = 0.1,
                 seed_split = 1994):

        # --------------------------
        # initialisation from input values 

        # directory was the data is stored
        self._file_path = abspath(file_path)

        # define a default redshift array on which to make the predictions
        # define the labels of the regressor
        if redshifts is None:
            self._z = np.array([4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 
                                        5.9, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 
                                        8, 8.25, 8.5, 8.75, 9, 9.5, 10, 10.5, 11, 
                                        11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 
                                        15.5, 16, 17, 18, 19, 20, 21, 22, 23, 24, 
                                        25, 26, 27, 29, 31, 33, 35])
        else:
            self._z = redshifts


        # --------------------------
        # prepare and read the datafile

        # if not already
        if not exists(file_path[:-4]+ "_preprocessed.npz"):
            preprocess_raw_data(file_path, random_seed=seed_split, frac_test = frac_test, frac_valid = frac_valid)
        else:
            with open(file_path[:-4]+ "_preprocessed.npz", 'rb') as file:
                data = np.load(file, allow_pickle=True)
                
                # if we do not have the same seed or fraction of valid and test samples we preprocess the data again
                if frac_test != data.get('frac_test', None) or frac_valid != data.get('frac_valid', None) or seed_split != data.get('random_seed', None):
                    preprocess_raw_data(file_path, random_seed=seed_split, frac_test = frac_test, frac_valid = frac_valid)


        with open(file_path[:-4]+ "_preprocessed.npz", 'rb') as file:
            data = np.load(file, allow_pickle=True)
            
            self._redshifts = data.get('redshifts', None)
            self._features  = data.get('features', None)
            self._cosmology = data.get('cosmology', None)
            self._xHIIdb    = data.get('xHIIdb', None)
            self._parameters_min_val = data.get('parameters_min_val', None)
            self._parameters_max_val = data.get('parameters_max_val', None) 
            self._parameters_name = data.get('parameters_name', None)
            self._indices_early_test = data.get('indices_early_test', None)
            self._indices_early_valid = data.get('indices_early_valid', None)
            self._indices_early_train = data.get('indices_early_train', None)
            self._indices_tot_test = data.get('indices_tot_test', None)
            self._indices_tot_valid = data.get('indices_tot_valid', None)
            self._indices_tot_train = data.get('indices_tot_train', None)

        n_tot = len(self._features)

        self._pos_h       = np.where(self._parameters_name == 'hlittle')[0][0]
        self._pos_omega_c = np.where(self._parameters_name == 'Omch2')[0][0]
        self._pos_omega_b = np.where(self._parameters_name == 'Ombh2')[0][0]

        self._indices_early = np.sort(np.concatenate((self._indices_early_test, self._indices_early_valid, self._indices_early_train)))
        
        # evaluate the optical depth to reionization for all runs
        # this is done with an optimised function for the evaluation of tau with numpy arrays
        # assume a late time universe with no radiation (very good approximation)
        self._tau = optical_depth_no_rad_numpy(self._redshifts, self._xHIIdb, 
                                                     self._features[:, self._pos_omega_b], 
                                                     self._features[:, self._pos_omega_c],
                                                     self._features[:, self._pos_h])
        
        self._x_array = true_to_uniform(self._features, self._parameters_min_val, self._parameters_max_val)
        
        self._y_classifier = np.zeros(len(self._features))
        self._y_classifier[self._indices_early] = 1.0

        self._y_regressor = np.zeros((n_tot, len(self._z) + 1))
        for i in range(n_tot):
            self._y_regressor[i, -1] = self._tau[i]
            if i in self._indices_early:
                self._y_regressor[i, :-1] = interpolate.interp1d(self._redshifts, self._xHIIdb[i, :])(self._z)
            
        # convert to float32 objects
        self._x_array      = np.array(self._x_array, np.float32)
        self._y_classifier = np.array(self._y_classifier, np.float32)
        self._y_regressor  = np.array(self._y_regressor, np.float32)

        # make torch batch loader objects
        self._train_loader_regressor = torch.utils.data.DataLoader(TorchDataset(self._x_array[self._indices_early_train], self._y_regressor[self._indices_early_train]), batch_size=64, shuffle=True)
        self._valid_loader_regressor = torch.utils.data.DataLoader(TorchDataset(self._x_array[self._indices_early_valid], self._y_regressor[self._indices_early_valid]), batch_size=64, shuffle=True)
        
        # --------------------------

    

    @property
    def z(self):
        return self._z
    
    @property
    def indices_early_train(self):
        return self._indices_early_train
    
    @property
    def indices_early_test(self):
        return self._indices_early_test
    
    @property
    def indices_early_valid(self):
        return self._indices_early_valid

    @property
    def indices_tot_train(self):
        return self._indices_tot_train
    
    @property
    def indices_tot_test(self):
        return self._indices_tot_test
    
    @property
    def indices_tot_valid(self):
        return self._indices_tot_valid

    @property
    def x_array(self):
        return self._x_array
    
    @property
    def y_classifier(self):
        return self._y_classifier
    
    @property
    def y_regressor(self):
        return self._y_regressor
    
    @property
    def train_loader_regressor(self):
        return self._train_loader_regressor
    
    @property
    def test_loader_regressor(self):
        return self._test_loader_regressor
    
    @property
    def valid_loader_regressor(self):
        return self._valid_loader_regressor
    
    @property
    def train_loader_classifier(self):
        return self._train_loader_classifier
    
    @property
    def test_loader_classifier(self):
        return self._test_loader_classifier
    
    @property
    def valid_loader_classifier(self):
        return self._valid_loader_classifier
    
