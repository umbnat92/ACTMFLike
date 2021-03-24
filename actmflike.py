import os, sys
import numpy as np
from scipy.io import FortranFile  # need this to read the fortran data format
from scipy import linalg  # need this for cholesky decomposition and inverse
import pkg_resources
from typing import Optional, Sequence
from cobaya.likelihood import Likelihood
from cobaya.conventions import _packages_path
from scipy import constants


class MFLikeACT(Likelihood):
    spectra: dict
    data: dict
    foregrounds: dict

    def initialize(self):
        """
         Prepare any computation, importing any necessary code, files, etc.
        """
        self.freqs = []
        nfreq = len(self.data["frequencies"])
        for f in np.arange(nfreq):
            self.freqs.append(self.data["frequencies"][f])
        
        # amount of low-ell bins to ignore per cross spectrum
        self.b0 = 5
        self.b1 = 0
        self.b2 = 0

        self.regions = self.data["regions"] 
        
        # calibration parameters 
        self.ct = [ 1.0 for _ in self.freqs ]
        #self.yp = [ 1.0 for _ in self.freqs ] #This is sampled, not fixed!

        self.prepare_clidx_dict()
        self.prepare_data()
        

        self.cal_params = ["yp1","yp2"]

        
        self.expected_params = ["a_tSZ", "a_kSZ", "xi", "a_p", "a_c", "beta_p",
                                "a_sw","a_sd", "a_gw", "a_gd", "a_gtew", "a_gted",
                                "a_geew", "a_geed", "a_pste", "a_psee",
                                "T_dd","T_d","n_CIBC"]
        
        self.l_bpws = np.arange(2,self.lmax_win+1)
        self.ellfact = 2.*np.pi/(self.l_bpws*(self.l_bpws+1.))
        self.log.debug("Initialized.")

        self.log.debug("Testing the loglike computation...")
        logp_test = self.actpol_likelihood_test_compute(self.data_folder)
        self.log.debug("Expected Χ² = 1060.6")

    def prepare_data(self):
        """
        Prepare data.
        """
        self.set_bins(self.bintt, self.bintt, self.bintt)

        self.regions = self.data["regions"]
        self.nlike = len(self.regions)
        
        if not self.use_sacc:
            self.log.debug("Reading data from txt ")
            self.load_plaintext(data_dir = self.data_folder)
        else:
            self.log.debug('Read sacc file to be implemented...')
            sys.exit()


        self.inv_cov = {}
        for r in self.regions:
            self.inv_cov[r] = np.linalg.inv(self.covmat[r])


        self.load_leakage(data_dir = self.data_folder)

    
    def set_bins(self, bintt, binte, binee):
        # Set the bin lengths for TT/TE/EE. You can give integers and the class will use the preset array self.freqs to determine the amount of cross-spectra.
        if type(bintt) == list:
            self.nbintt = bintt
        else:
            self.crosstt = []
            self.nbintt = [ int(bintt) for _ in range(len(self.freqs) * (len(self.freqs) + 1) // 2) ]
            
            for i in range(len(self.freqs)):
                for j in range(len(self.freqs)):
                    if j >= i:
                        self.crosstt.append((i,j))
        
        if type(binee) == list:
            self.nbinee = binee
        else:
            self.crossee = []
            self.nbinee = [ int(binee) for _ in range(len(self.freqs) * (len(self.freqs) + 1) // 2) ]
            
            for i in range(len(self.freqs)):
                for j in range(len(self.freqs)):
                    if j >= i:
                        self.crossee.append((i,j))
        
        if type(binte) == list:
            self.nbinte = binte
        else:
            self.crosste = []
            self.nbinte = [ int(binte) for _ in range(len(self.freqs) * len(self.freqs)) ]
            
            for i in range(len(self.freqs)):
                for j in range(len(self.freqs)):
                    self.crossee.append((i,j))
    
    def load_plaintext(self, data_dir = ''):
        if self.nbin == 0:
            raise ValueError('You did not set any spectra bin sizes beforehand!')
        
        self.win_func = {}
        self.b_dat = {}
        self.covmat = {}

        for reg in self.regions:
            self.log.debug("Preparing data for {} region.\n".format(reg))
            self.log.debug(self.regions[reg])

            bbl_filename = self.regions[reg]['bpwfname']
            spec_filename = self.regions[reg]['specname']
            cov_filename = self.regions[reg]['covname']
            
            self.win_func[reg] = np.loadtxt(data_dir + bbl_filename)[:self.nbin,:self.lmax_win]
            self.b_dat[reg] = np.loadtxt(data_dir + spec_filename)[:self.nbin]
            self.covmat[reg] = np.loadtxt(data_dir + cov_filename)[:self.nbin,:self.nbin] #.reshape((self.nbin, self.nbin))
                                
        for reg in self.regions:
            self.covmat[reg] = self.cull_covmat(self.covmat[reg])

    
    def load_leakage(self,data_dir = ''):
        # Note: this assumes all TE bins are the same length (it is hardcoded at a later point).
        self.leakage = {}
        self.a = {}

        for r in self.regions:
            leak_filename = self.regions[r]['leak_TE']
            self.leakage[r] = np.loadtxt(data_dir + leak_filename,unpack=True)
            self.a[r] = [1.,1.]
        
        
    
    def cull_covmat(self,covmat):
        # We have now packed the covariance matrix and the window function matrix.
        # We want to ignore the first b0 data points, we do so by culling the covmat for each observation.
        for i in range(self.b0):
            for j in range(self.nspectt):
                # cull lmin in TT
                covmat[i+sum(self.nbintt[0:j]),:self.nbin] = 0.0
                covmat[:self.nbin,i+sum(self.nbintt[0:j])] = 0.0
                covmat[i+sum(self.nbintt[0:j]),i+sum(self.nbintt[0:j])] = 1e10
        
        for i in range(sum(self.nbintt), sum(self.nbintt) + self.b1):
            for j in range(self.nspecte):
                # cull lmin in TE
                covmat[i+sum(self.nbinte[0:j]),:self.nbin] = 0.0
                covmat[:self.nbin,i+sum(self.nbinte[0:j])] = 0.0
                covmat[i+sum(self.nbinte[0:j]),i+sum(self.nbinte[0:j])] = 1e10
        
        for i in range(sum(self.nbintt) + sum(self.nbinte), sum(self.nbintt) + sum(self.nbinte) + self.b2):
            for j in range(self.nspecee):
                # cull lmin in EE
                covmat[i+sum(self.nbinee[0:j]),:self.nbin] = 0.0
                covmat[:self.nbin,i+sum(self.nbinee[0:j])] = 0.0
                covmat[i+sum(self.nbinee[0:j]),i+sum(self.nbinee[0:j])] = 1e10
        return covmat

    def get_requirements(self):
        """
        Method returning dictionary of requests from a theory code component, if needed.
        See https://cobaya.readthedocs.io/en/latest/likelihoods.html
        """
        return {"Cl": {cl: self.tt_lmax for cl in self.required_cl}}


    def prepare_clidx_dict(self):
        combinations = self.spectra["combinations"]
        binning = self.spectra["bin"]
        self.required_cl = self.spectra["use_spectra"]
        self.required_cl = [c.lower() for c in self.required_cl]

        self.freqspec = {}
        self.lenbin = {}
        self.nbisp = {}
        k = 0
        for s in self.required_cl:
            self.freqspec[s,"freq1"] = [combinations[s][i][0] for i in range(len(combinations[s]))]
            self.freqspec[s,"freq2"] = [combinations[s][i][1] for i in range(len(combinations[s]))]

            self.lenbin[s] = len(self.freqspec[s,"freq1"])
            for j in range(self.lenbin[s]+1):
                self.nbisp[s,'bin'+np.str(j)] = (j + k) * binning[s]
            k += self.lenbin[s]

    def _get_power_spectra(self,cl,fg_model):
        x_model = {r: np.zeros(self.nbin) for r in self.regions}

        for r in self.regions:
            for s in self.required_cl:
                cltmp = np.zeros(self.lmax_win+1)
                cltmp[2:self.tt_lmax+1] = cl[s][2:self.tt_lmax+1]
                for j in range(self.lenbin[s]):
                    sidx1 = self.freqspec[s,"freq1"][j]
                    sidx2 = self.freqspec[s,"freq2"][j]
                    bidx1 = self.nbisp[s,'bin'+np.str(j)]
                    bidx2 = self.nbisp[s,'bin'+np.str(j+1)]
                    x_model[r][bidx1:bidx2] = self.win_func[r][bidx1:bidx2,:self.lmax_win] \
                                @ ((cltmp[2:self.lmax_win+1] + fg_model[s,'all',sidx1,sidx2,r])*self.ellfact)

        return x_model

    def _doLeakage(self,vecmodel):
        for r in self.regions:
            a1 = self.a[r][0]
            a2 = self.a[r][1]
            vecmodel[r][3*self.bintt:3*self.bintt+self.binte] +=  vecmodel[r][:self.bintt]*a1*self.leakage[r][1]
            vecmodel[r][self.binte+3*self.bintt:3*self.bintt+2*self.binte] += vecmodel[r][self.bintt:2*self.bintt]*a2*self.leakage[r][2]
            vecmodel[r][2*self.binte+3*self.bintt:3*self.bintt+3*self.binte] += vecmodel[r][self.bintt:2*self.bintt]*a1*self.leakage[r][1]
            vecmodel[r][3*self.binte+3*self.bintt:3*self.bintt+4*self.binte] += vecmodel[r][2*self.bintt:3*self.bintt]*a2*self.leakage[r][2]
            vecmodel[r][4*self.binte+3*self.bintt:3*self.bintt+4*self.binte+self.binee] += 2*vecmodel[r][3*self.bintt:3*self.bintt+self.binte]*a1*self.leakage[r][1]+vecmodel[r][:self.bintt]*(a1*self.leakage[r][1])**2.
            vecmodel[r][self.binee+4*self.binte+3*self.bintt:3*self.bintt+4*self.binte+2*self.binee] += vecmodel[r][self.binte+3*self.bintt:3*self.bintt+2*self.binte]*a1*self.leakage[r][1]+vecmodel[r][2*self.binte+3*self.bintt:3*self.bintt+3*self.binte]*a2*self.leakage[r][2]+vecmodel[r][self.bintt:2*self.bintt]*a1*self.leakage[r][1]*a2*self.leakage[r][2]
            vecmodel[r][2*self.binee+3*self.bintt+4*self.binte:3*self.bintt+4*self.binte+3*self.binee] += 2*vecmodel[r][3*self.binte+3*self.bintt:3*self.bintt+4*self.binte]*a2*self.leakage[r][2]+vecmodel[r][2*self.bintt:3*self.bintt]*(a2*self.leakage[r][2])**2
        return vecmodel

    def calibrate(self,vecmodel):
        self.idxcal = {np.str(self.freqs[i]):i for i in range(len(self.freqs))}
        for r in self.regions:
            ct1 = self.ct[0]
            ct2 = self.ct[1]
            yp1 = self.yp[0]
            yp2 = self.yp[1]
            self.log.debug("Calibrating region {}".format(r))
            vecmodel[r][:self.bintt] = vecmodel[r][:self.bintt]*ct1*ct1
            vecmodel[r][self.bintt:2*self.bintt] = vecmodel[r][self.bintt:2*self.bintt]*ct1*ct2
            vecmodel[r][2*self.bintt:3*self.bintt] = vecmodel[r][2*self.bintt:3*self.bintt]*ct2*ct2
            vecmodel[r][3*self.bintt:3*self.bintt+self.binte] = vecmodel[r][3*self.bintt:3*self.bintt+self.binte]*ct1*ct1*yp1
            vecmodel[r][self.binte+3*self.bintt:3*self.bintt+2*self.binte] = vecmodel[r][self.binte+3*self.bintt:3*self.bintt+2*self.binte]*ct1*ct2*yp2
            vecmodel[r][2*self.binte+3*self.bintt:3*self.bintt+3*self.binte] = vecmodel[r][2*self.binte+3*self.bintt:3*self.bintt+3*self.binte]*ct1*ct2*yp1
            vecmodel[r][3*self.binte+3*self.bintt:3*self.bintt+4*self.binte] = vecmodel[r][3*self.binte+3*self.bintt:3*self.bintt+4*self.binte]*ct1*ct2*yp2
            vecmodel[r][4*self.binte+3*self.bintt:3*self.bintt+4*self.binte+self.binee] = vecmodel[r][4*self.binte+3*self.bintt:3*self.bintt+4*self.binte+self.binee]*ct1*ct1*yp1*yp1
            vecmodel[r][self.binee+4*self.binte+3*self.bintt:3*self.bintt+4*self.binte+2*self.binee] = vecmodel[r][self.binee+4*self.binte+3*self.bintt:3*self.bintt+4*self.binte+2*self.binee]*ct1*ct2*yp1*yp2
            vecmodel[r][2*self.binee+3*self.bintt+4*self.binte:3*self.bintt+4*self.binte+3*self.binee] = vecmodel[r][2*self.binee+3*self.bintt+4*self.binte:3*self.bintt+4*self.binte+3*self.binee]*ct2*ct2*yp2*yp2

        return vecmodel

    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter, values params_values
        and return a log-likelihood.
        """
        cl = self.theory.get_Cl(ell_factor=True)
        fg_model = self._get_foreground_model(
            {k: params_values[k] for k in self.expected_params})
        x_model = self._get_power_spectra(cl,fg_model)

        self.yp = [params_values[y] for y in self.cal_params]
        logp = self.loglike(x_model)
        return logp
    
    def loglike(self, x_model):
        logp = 0.
        x_model = self._doLeakage(x_model)

        x_model = self.calibrate(x_model)

        for r in self.regions:
            diff_vec = self.b_dat[r] - x_model[r]

            if self.enable_tt and not self.enable_te and not self.enable_ee:
                bin_no = sum(self.nbintt)
                diff_vec = diff_vec[:bin_no]
                subcov = self.covmat[r][:bin_no,:bin_no]
                inv_covtt = np.linalg.inv(subcov)
                logptmp = - 0.5 * diff_vec @ inv_covtt @ diff_vec
                logp += logptmp
                self.log.debug('Using only TT.')
            elif not self.enable_tt and self.enable_te and not self.enable_ee:
                n0 = sum(self.nbintt)
                bin_no = sum(self.nbinte)
                diff_vec = diff_vec[n0:n0 + bin_no]
                subcov = self.covmat[r][n0:n0 + bin_no, n0:n0 + bin_no]
                inv_covte = np.linalg.inv(subcov)
                logptmp = - 0.5 * diff_vec @ inv_covte @ diff_vec
                logp += logptmp
                self.log.debug('Using only TE.')
            elif not self.enable_tt and not self.enable_te and self.enable_ee:
                n0 = sum(self.nbintt) + sum(self.nbinte)
                bin_no = sum(self.nbinee)
                diff_vec = diff_vec[n0:n0 + bin_no]
                subcov = self.covmat[r][n0:n0 + bin_no, n0:n0 + bin_no]
                inv_covee = np.linalg.inv(subcov)
                logptmp = - 0.5 * diff_vec @ inv_covee @ diff_vec
                logp += logptmp
                self.log.debug('Using only EE.')
            elif self.enable_tt and self.enable_te and self.enable_ee:
                self.log.debug('Using TT+TE+EE.')
                logptmp = - 0.5 * diff_vec @ self.inv_cov[r] @ diff_vec
                logp += logptmp
            else:
                raise Exception('Improper combination of TT/TE/EE spectra selected.')
        
            self.log.debug(
            "Χ² value computed for region {} "
            "Χ² = {}".format(r,-2 * logptmp))
        self.log.debug("Total Χ² = {}".format(-2 * logp))
        return logp

    def actpol_likelihood_test_compute(self,cellpath):
        cell = np.loadtxt(cellpath+'bf_ACTPol_lcdm.minimum.theory_cl',usecols=(1,2,3),unpack=True)

        cl = {s: np.zeros(self.tt_lmax+1) for s in self.required_cl}
        cl['tt'][2:] = cell[0]
        cl['ee'][2:] = cell[2]
        cl['te'][2:] = cell[1]

        nuisance_params = {
                            "yp1": 0.9860632E+00,
                            "yp2": 0.9714017E+00,
                            "a_tSZ": 0.5806115E+01,
                            "a_kSZ": 0.1024734E-03,
                            "xi": 0.1998220E+00,
                            "a_p": 0.6872966E+01,
                            "a_c": 0.3648102E+01,
                            "beta_p": 0.2447908E+01,
                            "a_sd": 0.3682240E+01,
                            "a_gd": 0.2811710E+01,
                            "a_gted": 0.1049940E+00,
                            "a_geed": 0.3436806E-01,
                            "a_pste": 0.4443353E-01,
                            "a_psee": 0.5342375E-04,
                            "a_sw": 0.2249012E+02,
                            "a_gw": 0.8717251E+01,
                            "a_gtew": 0.3559930E+00,
                            "a_geew": 0.1293905E+00,
                            "T_dd": 19.6,
                            "T_d": 9.7,
                            "n_CIBC": 1.2,
                            }

        fg_model = self._get_foreground_model(
            {k: nuisance_params[k] for k in self.expected_params})

        x_model = self._get_power_spectra(cl,fg_model)
        self.yp = [nuisance_params[y] for y in self.cal_params]

        logp = self.loglike(x_model)

        return logp

    
    
    @property
    def nspectt(self):
        return len(self.nbintt)
    
    @property
    def nspecte(self):
        return len(self.nbinte)
    
    @property
    def nspecee(self):
        return len(self.nbinee)
    
    @property
    def nbin(self):
        # total number of bins
        return sum(self.nbintt) + sum(self.nbinte) + sum(self.nbinee)
    
    @property
    def nspec(self):
        # total number of spectra
        return self.nspectt + self.nspecte + self.nspecee
    
    @property
    def shape(self):
        return self.lmax_win-1
    
    @property
    def input_shape(self):
        return self.tt_lmax-1

    def _get_foreground_model(self, fg_params):
        return get_foreground_model(fg_params=fg_params,
                                    fg_model=self.foregrounds,
                                    frequencies=self.freqs,
                                    ell=self.l_bpws,
                                    requested_cls=self.required_cl)


# Standalone function to return the foreground model
# given the nuisance parameters
def get_foreground_model(fg_params, fg_model,
                         frequencies, ell,
                         requested_cls=["tt", "te", "ee"]):
    normalisation = fg_model["normalisation"]
    nu_0 = normalisation["nu_0"]
    ell_0 = normalisation["ell_0"]
    partitions = fg_model["partitions"]

    freq_eff = fg_model["frequencies"]
    
    fdust = freq_eff["fdust"]
    fsz = freq_eff["fsz"]
    fsyn = freq_eff["fsyn"]

    external_cl = fg_model["external_cl"]
    cibc_file = external_cl["cibc"]
    tSZxcib_file = external_cl["tszxcib"]

    clszcib = np.loadtxt(tSZxcib_file,usecols=(1),unpack=True)[:ell.max()-1]  

    ell_clp = ell*(ell+1.)
    ell_0clp = 3000.*3001.

    T_CMB = 2.72548

    from fgspectra import cross as fgc
    from fgspectra import frequency as fgf
    from fgspectra import power as fgp

    cibc = fgc.FactorizedCrossSpectrum(fgf.CIB(), fgp.PowerSpectrumFromFile(cibc_file))
    cibp = fgc.FactorizedCrossSpectrum(fgf.ModifiedBlackBody(), fgp.PowerLaw())
    radio = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.PowerLaw())
    dust = fgc.FactorizedCrossSpectrum(fgf.ModifiedBlackBody(), fgp.PowerLaw())
    tsz = fgc.FactorizedCrossSpectrum(fgf.ThermalSZ(), fgp.tSZ_150_bat())
    ksz = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.kSZ_bat())

    # Make sure to pass a numpy array to fgspectra
    if not isinstance(frequencies, np.ndarray):
        frequencies = np.array(frequencies)
        
        fdustd = np.array(fdust['deep'])
        fdustw = np.array(fdust['wide'])
        
        fszd = np.array(fsz['deep'])
        fszw = np.array(fsz['wide'])
        
        fsynd = np.array(fsyn['deep'])
        fsynw = np.array(fsyn['wide'])

    fszcorrd,f0d = sz_func(fszd,nu_0)
    fszcorrw,f0w = sz_func(fszw,nu_0)

    planckratiod = plankfunctionratio(fdustd,nu_0,fg_params["T_d"])
    fluxtempd = flux2tempratiod(fdustd,nu_0)
     
    planckratiow = plankfunctionratio(fdustw,nu_0,fg_params["T_d"])
    fluxtempw = flux2tempratiod(fdustw,nu_0)

    model = {}
    

    if "wide" in partitions:
        model["tt", "cibc","wide"] = fg_params["a_c"] * cibc(
            {"nu": fdustw, "nu_0": nu_0,
            "temp": fg_params["T_d"], "beta": fg_params["beta_p"]},
            {'ell':ell, 'ell_0':ell_0})

        model["tt", "cibp","wide"] = fg_params["a_p"] * cibp(
            {"nu": fdustw, "nu_0": nu_0,
            "temp": fg_params["T_d"], "beta": fg_params["beta_p"]},
            {"ell": ell_clp, "ell_0": ell_0clp, "alpha": 1})

        model["tt", "tSZ","wide"] = fg_params["a_tSZ"] * tsz(
            {"nu": fszw, "nu_0": nu_0},
            {"ell": ell, "ell_0": ell_0})

        model["tt", "kSZ","wide"] = fg_params["a_kSZ"] * ksz(
            {"nu": (fszw/fszw)},
            {"ell": ell, "ell_0": ell_0})

        model["tt", "dust", "wide"] = fg_params["a_gw"] * dust(
            {"nu": fdustw, "nu_0": nu_0,
            "temp": fg_params["T_dd"], "beta": 1.5},
            {"ell": ell, "ell_0": 500., "alpha": -0.6})

        model["tt", "radio", "wide"] = fg_params["a_sw"] * radio(
            {"nu": fsynw, "nu_0": nu_0, "beta": -0.5 - 2.},
            {"ell": ell_clp, "ell_0": ell_0clp,"alpha":1}) 

        model["ee", "radio", "wide"] = fg_params["a_psee"] * radio(
            {"nu": fsynw, "nu_0": nu_0, "beta": -0.5 - 2.},
            {"ell": ell_clp, "ell_0": ell_0clp,"alpha":1})
        
        model["ee", "dust", "wide"] = fg_params["a_geew"] * dust(
            {"nu": fdustw, "nu_0": nu_0,
            "temp": fg_params["T_dd"], "beta": 1.5},
            {"ell": ell, "ell_0": 500., "alpha": -0.4})

        model["te", "radio", "wide"] = fg_params["a_pste"] * radio(
            {"nu": fsynw, "nu_0": nu_0, "beta": -0.5 - 2.},
            {"ell": ell_clp, "ell_0": ell_0clp,"alpha":1})     
        
        model["te", "dust", "wide"] = fg_params["a_gtew"] * dust(
            {"nu": fdustw, "nu_0": nu_0,
            "temp": fg_params["T_dd"], "beta": 1.5},
            {"ell": ell, "ell_0": 500., "alpha": -0.4})
        
    if "deep" in partitions:
        model["tt", "cibc","deep"] = fg_params["a_c"] * cibc(
            {"nu": fdustd, "nu_0": nu_0,
            "temp": fg_params["T_d"], "beta": fg_params["beta_p"]},
            {'ell':ell, 'ell_0':ell_0})

        model["tt", "cibp","deep"] = fg_params["a_p"] * cibp(
            {"nu": fdustd, "nu_0": nu_0,
             "temp": fg_params["T_d"], "beta": fg_params["beta_p"]},
            {"ell": ell_clp, "ell_0": ell_0clp, "alpha": 1})

        model["tt", "tSZ","deep"] = fg_params["a_tSZ"] * tsz(
            {"nu": fszd, "nu_0": nu_0},
            {"ell": ell, "ell_0": ell_0})

        model["tt", "kSZ","deep"] = fg_params["a_kSZ"] * ksz(
            {"nu": (fszd/fszd)},
            {"ell": ell, "ell_0": ell_0})

        model["tt", "dust", "deep"] = fg_params["a_gd"] * dust(
            {"nu": fdustd, "nu_0": nu_0,
            "temp": fg_params["T_dd"], "beta": 1.5},
            {"ell": ell, "ell_0": 500., "alpha": -0.6})
        
        model["tt", "radio", "deep"] = fg_params["a_sd"] * radio(
            {"nu": fsynd, "nu_0": nu_0, "beta": -0.5 - 2.},
            {"ell": ell_clp, "ell_0": ell_0clp,"alpha":1})

        model["ee", "radio", "deep"] = fg_params["a_psee"] * radio(
            {"nu": fsynd, "nu_0": nu_0, "beta": -0.5 - 2.},
            {"ell": ell_clp, "ell_0": ell_0clp,"alpha":1})
        
        model["ee", "dust", "deep"] = fg_params["a_geed"] * dust(
            {"nu": fdustd, "nu_0": nu_0,
            "temp": fg_params["T_dd"], "beta": 1.5},
            {"ell": ell, "ell_0": 500., "alpha": -0.4})

        model["te", "radio", "deep"] = fg_params["a_pste"] * radio(
            {"nu": fsynd, "nu_0": nu_0, "beta": -0.5 - 2.},
            {"ell": ell_clp, "ell_0": ell_0clp,"alpha":1})
        
        model["te", "dust", "deep"] = fg_params["a_gted"] * dust(
            {"nu": fdustd, "nu_0": nu_0,
            "temp": fg_params["T_dd"], "beta": 1.5},
            {"ell": ell, "ell_0": 500., "alpha": -0.4})


    components = fg_model["components"]
    component_list = {s: components[s] for s in requested_cls}
    fg_dict = {}
    for c1, f1 in enumerate(frequencies):
        for c2, f2 in enumerate(frequencies):
            for s in requested_cls:
                for p in np.arange(len(partitions)):
                    fg_dict[s, "all", f1, f2, partitions[p]] = np.zeros(len(ell))
                    for comp in component_list[s]:
                        if comp == "tSZxcib" and partitions[p] == "deep":
                            fg_dict[s, comp, f1, f2, partitions[p]] = -2. * np.sqrt(fg_params['a_c'] * fg_params['a_tSZ']) * fg_params['xi']\
                                  * clszcib * ((fdustd[c1]**fg_params['beta_p'] * fszcorrd[c2]\
                                  * planckratiod[c1] * fluxtempd[c1]+fdustd[c2]**fg_params['beta_p'] \
                                  * fszcorrd[c1] * planckratiod[c2] * fluxtempd[c2])/(2 * nu_0**fg_params['beta_p'] * f0d))
                            fg_dict[s, "all", f1, f2 , partitions[p]] += fg_dict[s, comp, f1, f2, partitions[p]]
                        elif comp == "tSZxcib" and partitions[p] == "wide":
                            fg_dict[s, comp, f1, f2, partitions[p]] = -2. * np.sqrt(fg_params['a_c'] * fg_params['a_tSZ']) * fg_params['xi']\
                                  * clszcib * ((fdustw[c1]**fg_params['beta_p'] * fszcorrw[c2]\
                                  * planckratiow[c1] * fluxtempw[c1]+fdustw[c2]**fg_params['beta_p'] \
                                  * fszcorrw[c1] * planckratiow[c2] * fluxtempw[c2])/(2 * nu_0**fg_params['beta_p'] * f0w))
                            fg_dict[s, "all", f1, f2 , partitions[p]] += fg_dict[s, comp, f1, f2, partitions[p]]
                        else:
                            fg_dict[s, comp, f1, f2, partitions[p]] = model[s, comp, partitions[p]][c1, c2]
                            fg_dict[s, "all", f1, f2 , partitions[p]] += fg_dict[s, comp, f1, f2, partitions[p]]

    return fg_dict

def sz_func(fsz,feff):
  T_cmb = 2.72548
  nu = fsz * 1e9
  nu0 = feff * 1e9
  x = constants.h * nu/(constants.k * T_cmb)
  x0 = constants.h * nu0/(constants.k * T_cmb)
  
  fszcorr = (x*(1/np.tanh(x/2.))-4)
  f0 = (x0*(1/np.tanh(x0/2.))-4)
  return fszcorr,f0
  

def plankfunctionratio(fdust,feff,T_eff):
  nu = fdust * 1e9 
  nu0 = feff * 1e9
  x = constants.h * nu/(constants.k * T_eff)
  x0 = constants.h * nu0/(constants.k * T_eff)
  
  return (nu/nu0)**3 * (np.exp(x0)-1.)/(np.exp(x)-1.)


def flux2tempratiod(fdust,feff):
  T_cmb = 2.72548
  nu =  fdust * 1e9
  nu0 = feff * 1e9
  x = constants.h * nu/(constants.k * T_cmb)
  x0 = constants.h * nu0/(constants.k * T_cmb)
  
  return (nu0/nu)**4 * (np.exp(x0)/np.exp(x)) * ((np.exp(x)-1.)/(np.exp(x0)-1.))**2

  
     
