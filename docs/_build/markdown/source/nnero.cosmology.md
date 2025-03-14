# Cosmology

### *exception* nnero.cosmology.ShortPowerSpectrumRange(scales: ndarray, message: str = 'Matter power spectrum range is too short')

Bases: `Exception`

Exception raised for too short ranges of modes to accurately compute integrals

#### message

Explanation of the error.

* **Type:**
  str

* **Parameters:**
  * **scales** (*np.ndarray*) – Range of modes we consider.
  * **message** (*str*) – Explanation of the error.

### nnero.cosmology.check_ik_R(ik_radius: int, p: int, radius: ndarray) → None

Check that the array of mode is long enough for the radius considered.

* **Parameters:**
  * **ik_radius** (*int*) – Index of the array of mode where equal (or closest) to the desired radius.
  * **p** (*int*) – Length of the array of modes.
  * **radius** (*np.ndarray*) – Array of radiuses corresponding to the modes.
* **Raises:**
  [**ShortPowerSpectrumRange**](#nnero.cosmology.ShortPowerSpectrumRange) – If array not long enough.

### nnero.cosmology.convert_array(arr: float, to_torch: bool = False) → ndarray | Tensor

### nnero.cosmology.dn_dm(z: float | ndarray, mass: float | ndarray, k: ndarray, pk: ndarray, omega_m: float | ndarray, h: float | ndarray, sheth_a: float = 0.322, sheth_q: float = 1.0, sheth_p: float = 0.3, \*, window: str = 'sharpk', c: float = 2.5)

Halo mass function in Msol^{-1} Mpc^{-3}

* **Parameters:**
  * **z** (*float* *|* *numpy.ndarray*) – shape (r,) redshift values
  * **mass** (*float* *|* *numpy.ndarray*) – shape (s,) or (q, r, s,) mass scale in Msol
  * **k** (*numpy.ndarray*) – shape (q, p), modes in Mpc^{-1}
  * **pk** (*numpy.ndarray*) – shape (q, p), power spectrum in Mpc^3
  * **omega_m** (*float* *,* *np.ndarray*) – shape (q,), reduced matter abundance
  * **h** (*float* *,* *numpy.ndarray*) – shape (q1, …, qn), reduced hubble constant
  * **window** (*str* *,* *optional*) – smoothing function
  * **c** (*float* *,* *optional*) – conversion factor for the mass in the sharpk window function
* **Return type:**
  numpy.ndarrray with shape (q, r, s) – in Msol / Mpc^3

### nnero.cosmology.dsigma_m_dm(mass: float | ndarray, k: ndarray, pk: ndarray, omega_m: float | ndarray, \*, window: str = 'sharpk', sigma_mass: ndarray | None = None, c: float = 2.5)

Derivative of the standard deviation of the matter power spectrum
on mass scale. Note that all physical dimensions must be self consistent.

* **Parameters:**
  * **mass** (*float* *|* *numpy.ndarray*) – shape (q1, …, qn, r1, …, rm, s) mass scale in Msol
  * **k** (*numpy.ndarray*) – shape (q1, …, qn, p), modes in Mpc^{-1}
  * **pk** (*numpy.ndarray*) – shape (q1, …, qn, p), power spectrum in Mpc^3
  * **omega_m** (*float* *,* *np.ndarray*) – shape (q1, …, qn)
  * **window** (*str* *,* *optional*) – smoothing function
  * **sigma_mass** (*numpy.ndarray* *,* *optional*) – shape of mass, value of sigma_m at input mass
  * **c** (*float* *,* *optional*) – conversion factor for the mass in the sharpk window function
* **Return type:**
  numpy.ndarray with shape (q1, …, qn, r1, …, rm, s)

### nnero.cosmology.dsigma_r_dr(radius: float | ndarray, k: ndarray, pk: ndarray, \*, window: str = 'sharpk', sigma_radius: ndarray | None = None) → ndarray

Derivative of the standard deviation of the matter power spectrum
inside radius. Note that all physical dimensions must be self consistent.

* **Parameters:**
  * **radius** (*float* *|* *numpy.ndarray*) – shape (q1, …, qn, r1, …, rm, s), smoothing scale r in Mpc
  * **k** (*numpy.ndarray*) – shape (q1, …, qn, p), modes in Mpc^{-1}
  * **pk** (*numpy.ndarray*) – shape (q1, …, qn, p), power spectrum in Mpc^3
  * **window** (*str* *,* *optional*) – smoothing function
  * **ik_radius** (*numpy.ndarray* *,* *optional*) – shape of radius, indices of the k array corresponding to k = 1/radius
* **Return type:**
  numpy.ndarray with shape (q1, …, qn, r1, …, rm, s)

### nnero.cosmology.growth_function(z: float | ndarray, omega_m: float | ndarray, h: float | ndarray)

Growth function of the linear density contrast

Analatical fit from Caroll

* **Parameters:**
  * **z** (*float* *|* *numpy.ndarray*) – shape (r,) if array, redshift range
  * **omega_m** (*float* *,* *numpy.ndarray*) – shape (q1, …, qn), reduced matter abundance
  * **h** (*float* *,* *numpy.ndarray*) – shape (q1, …, qn), reduced hubble constant
* **Return type:**
  numpy.ndarray with shape (q1, …, qn, r)

### nnero.cosmology.h_factor_no_rad(z: float | ndarray | Tensor, omega_b: float | ndarray | Tensor, omega_c: float | ndarray | Tensor, h: float | ndarray | Tensor) → ndarray | Tensor

E(z) = H(z) / H0 without radiation

Efficient evalutation of hubble rate parameters for numpy arrays
or torch tensors (also works with simple float ). In the first
case the shape of these numpy arrays must be compatible (see below).

* **Parameters:**
  * **z** (*float* *|* *numpy.ndarray* *|* *torch.Tensor*) – shape (p, ) if array, redshift range
  * **omega_b** (*float* *|* *numpy.ndarray* *|* *torch.Tensor*) – shape (q1,…, qn) if array, reduced abundance of baryons (i.e. times h^2)
  * **omega_c** (*float* *|* *numpy.ndarray* *|* *torch.Tensor*) – shape (q1,…, qn) if array, reduced abundance of dark matter (i.e. times h^2)
  * **h** (*float* *|* *numpy.ndarray* *|* *torch.Tensor*) – shape (q1,…, qn) if array, Hubble factor
* **Return type:**
  numpy.ndarray or torch.Tensor with shape (q1,…, qn, p)

### nnero.cosmology.h_factor_numpy(z: float | ndarray | Tensor, omega_b: float | ndarray | Tensor, omega_c: float | ndarray | Tensor, h: float | ndarray | Tensor, m_nus: ndarray | Tensor) → ndarray | Tensor

E(z) = H(z) / H0 with radiation

Efficient evalutation of hubble rate parameters for numpy arrays
or torch tensors (also works with simple float ). In the first
case the shape of these numpy arrays must be compatible (see below).

* **Parameters:**
  * **z** (*float* *|* *numpy.ndarray* *|* *torch.Tensor*) – shape (p, ) if array, redshift range
  * **omega_b** (*float* *|* *numpy.ndarray* *|* *torch.Tensor*) – shape (q1,…, qn) if array, reduced abundance of baryons (i.e. times h^2)
  * **omega_c** (*float* *|* *numpy.ndarray* *|* *torch.Tensor*) – shape (q1,…, qn) if array, reduced abundance of dark matter (i.e. times h^2)
  * **h** (*float* *|* *numpy.ndarray* *|* *torch.Tensor*) – shape (q1,…, qn) if array,, Hubble factor
  * **mnus** (*numpy.ndarray* *|* *torch.Tensor*) – shape (q1,…, qn, 3) if array or at least shape (, 3), mass of the neutrinos
* **Return type:**
  numpy.ndarray or torch.Tensor with shape (q1,…, qn, p)

### nnero.cosmology.n_baryons(omega_b: float | ndarray | Tensor) → float | ndarray | Tensor

Baryon number density (in 1 / m^3)

* **Parameters:**
  **omega_b** (*float* *|* *np.ndarray* *|* *torch.Tensor*) – reduced abundance of baryons (i.e. times h^2)
* **Return type:**
  float or np.ndarray or torch.tensor

### nnero.cosmology.n_hydrogen(omega_b: float | ndarray | Tensor) → float | ndarray | Tensor

Hydrogen number density (in 1 / m^3)

* **Parameters:**
  **omega_b** (*float* *|* *np.ndarray* *|* *torch.Tensor*) – reduced abundance of baryons (i.e. times h^2)
* **Return type:**
  float or np.ndarray or torch.tensor

### nnero.cosmology.n_ur(m_nus: ndarray | Tensor) → ndarray | Tensor

Number of ultra-relativistic degrees of freedom

* **Parameters:**
  **m_nus** (*np.ndarray* *|* *torch.Tensor*) – shape (q1, q2, …, qn, 3), mass of the three neutrinos
  in a given model
* **Return type:**
  np.ndarray or torch.Tensor with shape (q1, q2, …, qn)

### nnero.cosmology.omega_nu(z: float | ndarray | Tensor, m_nus: ndarray | Tensor) → ndarray | Tensor

Efficient implementation of reduced neutrino abundance for numpy arrays.
Parameters must be floats, arrays or tensor with compatible dimensions.

* **Parameters:**
  * **m_nus** (*np.ndarray* *|* *torch.Tensor*) – shape (q1, q2, …, qn, 3), mass of the three neutrinos
    in a given model
  * **z** (*float* *|* *numpy.ndarray* *|* *torch.Tensor*) – shape (p, ) if array, redshift range
* **Return type:**
  numpy.ndarray or torch.Tensor with shape (q1, …, qn, p)

### nnero.cosmology.omega_r(m_nus: ndarray | Tensor) → ndarray | Tensor

Reduced abundance of radiation today

* **Parameters:**
  **m_nus** (*np.ndarray* *|* *torch.Tensor*) – shape (q1, q2, …, qn, 3), mass of the three neutrinos
  in a given model
* **Return type:**
  np.ndarray or torch.Tensor with shape (q1, q2, …, qn)

### nnero.cosmology.optical_depth_no_rad(z: float | ndarray | Tensor, xHII: ndarray | Tensor, omega_b: float | ndarray | Tensor, omega_c: float | ndarray | Tensor, h: float | ndarray | Tensor, \*, low_value: float = 1.0, with_helium: bool = True, cut_integral_min: bool = True)

Optical depth to reionization without radiation.

Efficient evaluation of the opetical depth to reionization (dimensionless)
uses fast numpy / torch operations with trapezoid rule
(assume that radiation is neglibible on the range of z). Also neglegts the
influence of double reionization of helium at small redshifts

* **Parameters:**
  * **z** (*float* *|* *numpy.ndarray* *|* *torch.Tensor*) – shape (p,) if array, redshift range
  * **xHII** (*numpy.ndarray* *|* *torch.Tensor*) – shape (n, p), ionization fraction vs the redshift for the n models
  * **omega_b** (*float* *|* *numpy.ndarray* *|* *torch.Tensor*) – shape (n,) if array, reduced abundance of baryons (i.e. times h^2)
  * **omega_c** (*float* *|* *numpy.ndarray* *|* *torch.Tensor*) – shape (n,) if array, reduced abundance of dark matter (i.e. times h^2)
  * **h** (*float* *|* *numpy.ndarray* *|* *torch.Tensor*) – shape (n,) if array, Hubble factor
  * **low_value** (*float*) – value of xHII at redshift smaller than min(z)
* **Return type:**
  numpy.ndarray or torch.Tensor with shape(n, p)

### nnero.cosmology.rho_baryons(omega_b: float | ndarray | Tensor) → float | ndarray | Tensor

Baryon energy density (in eV / m^3)

* **Parameters:**
  **omega_b** (*float* *|* *np.ndarray* *|* *torch.Tensor*) – reduced abundance of baryons (i.e. times h^2)
* **Return type:**
  float or np.ndarray or torch.tensor

### nnero.cosmology.sigma_m(mass: float | ndarray, k: ndarray, pk: ndarray, omega_m: float | ndarray, \*, window: str = 'sharpk', ik_radius: ndarray | None = None, c: float = 2.5)

Standard deviation of the matter power spectrum on mass scale.
Note that all physical dimensions must be self consistent.

* **Parameters:**
  * **mass** (*float* *|* *numpy.ndarray*) – shape (q1, …, qn, r1, …, rm, s) mass scale in Msol
  * **k** (*numpy.ndarray*) – shape (q1, …, qn, p), modes in Mpc^{-1}
  * **pk** (*numpy.ndarray*) – shape (q1, …, qn, p), power spectrum in Mpc^3
  * **omega_m** (*float* *,* *np.ndarray*) – shape (q1, …, qn)
  * **window** (*str* *,* *optional*) – smoothing function
  * **ik_radius** (*numpy.ndarray* *,* *optional*) – shape of radius, indices of the k array corresponding to k = 1/radius
  * **c** (*float* *,* *optional*) – conversion factor for the mass in the sharpk window function
* **Return type:**
  numpy.ndarray with shape (q1, …, qn, r1, …, rm, s)

### nnero.cosmology.sigma_r(radius: float | ndarray, k: ndarray, pk: ndarray, \*, window: str = 'sharpk', ik_radius: ndarray | None = None) → ndarray

Standard deviation of the matter power spectrum inside radius.
Note that all physical dimensions must be self consistent.

* **Parameters:**
  * **radius** (*float* *|* *numpy.ndarray*) – shape (s,) or (q1, …, qn, r1, …, rm, s), smoothing scale r in Mpc
  * **k** (*numpy.ndarray*) – shape (q1, …, qn, p), modes in Mpc^{-1}
  * **pk** (*numpy.ndarray*) – shape (q1, …, qn, p), power spectrum in Mpc^3
  * **window** (*str* *,* *optional*) – smoothing function
  * **ik_radius** (*numpy.ndarray* *,* *optional*) – shape of radius, indices of the k array corresponding to k = 1/radius
* **Return type:**
  numpy.ndarray with shape (q1, …, qn, r1, …, rm, s) or (q1, …, qn, s)
