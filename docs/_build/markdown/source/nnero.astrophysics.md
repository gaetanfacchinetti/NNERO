# Astrophysics

### nnero.astrophysics.dmhalo_dmuv(hz: float | ndarray, m_uv: float | ndarray, alpha_star: float | ndarray, t_star: float | ndarray, f_star10: float | ndarray, omega_b: float | ndarray, omega_m: float | ndarray, \*, mh: ndarray | None = None, mask: ndarray | None = None)

### nnero.astrophysics.f_duty(mh: float | ndarray, m_turn: float | ndarray)

### nnero.astrophysics.m_halo(hz: float | ndarray, m_uv: float | ndarray, alpha_star: float | ndarray, t_star: float | ndarray, f_star10: float | ndarray, omega_b: float | ndarray, omega_m: float | ndarray) → tuple[ndarray, ndarray]

Halo mass in term of the UV magnitude for a given astrophysical model

* **Parameters:**
  * **hz** (*float* *,* *np.ndarray*) – Shape (q, r). Hubble factor given in s^{-1}
  * **m_uv** (*float* *,* *np.ndarry* *(**s* *,* *) or*  *(**r* *,* *s* *)*) – UV magnitude.
  * **omega_b** (*float* *,* *np.ndaray* *(**q* *,* *)*) – Reduced abundance of baryons
* **Return type:**
  numpy.ndarray with shape (q, r, s)

### nnero.astrophysics.phi_uv(z: float | ndarray, hz: float | ndarray, m_uv: float | ndarray, k: ndarray, pk: ndarray, alpha_star: float | ndarray, t_star: float | ndarray, f_star10: float | ndarray, m_turn: float | ndarray, omega_b: float | ndarray, omega_m: float | ndarray, h: float | ndarray, sheth_a: float = 0.322, sheth_q: float = 1.0, sheth_p: float = 0.3, \*, window: str = 'sharpk', c: float = 2.5, mh: ndarray | None = None, mask: ndarray | None = None, dndmh: ndarray | None = None)

UV flux in Mpc^{-3}

* **Parameters:**
  * **m_uv** (*float* *,* *np.ndarry* *(**s* *,* *)*)
  * **omega_b** (*float* *,* *np.ndaray* *(**q* *,* *)*)
* **Return type:**
  result of shape (q, r, s)
