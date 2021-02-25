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
        self.cells = None
        self.win_ells = None
        self.b_ell = None
        self.ells = None
        
        # Total number of bins per cross spectrum
        self.nbintt = []
        self.nbinte = []
        self.nbinee = []
        
        # Which frequencies are crossed per spectrum
        self.crosstt = []
        self.crosste = []
        self.crossee = []
        
        self.freqs = []
        nfreq = len(self.data["frequencies"])
        for f in np.arange(nfreq):
            self.freqs.append(self.data["frequencies"][f])
        
        # amount of low-ell bins to ignore per cross spectrum
        self.b0 = 5
        self.b1 = 0
        self.b2 = 0
        
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

    def prepare_data(self):
        """
        Prepare data.
        """
        self.set_bins(self.bintt, self.bintt, self.bintt)

        self.regions = self.data["regions"]
        self.nlike = len(self.regions)
        self.log.debug("Preparing data for {} regions.".format(self.regions))
        

        if not self.use_sacc:
            self.log.debug("Reading data from txt ")
            self.load_plaintext(self.specname, self.covname, self.bpwfname, data_dir = self.data_folder)
        else:
            self.log.debug('Read sacc file to be implemented...')
            sys.exit()


        self.inv_cov = {self.regions[r]:np.linalg.inv(self.covmat[self.regions[r]])
                                for r in range(self.nlike)}


        if self.doLeakage:
            self.load_leakage(self.leak_TE,data_dir = self.data_folder)

    
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
    
    def load_plaintext(self, spec_filename, cov_filename, bbl_filename, data_dir = ''):
        if self.nbin == 0:
            raise ValueError('You did not set any spectra bin sizes beforehand!')
        if len(spec_filename)!=self.nlike:
            raise ValueError('You need {} spectra!'.format(self.nlike))
            sys.exit()
        
        self.win_func = {self.regions[s]: np.loadtxt(data_dir + bbl_filename[s])[:self.nbin,:self.lmax_win]
                                for s in range(self.nlike)}

        self.b_dat = {self.regions[s]: np.loadtxt(data_dir + spec_filename[s],unpack=True)[:self.nbin]
                                for s in range(self.nlike)}
        
        self.covmat = {self.regions[s]: np.loadtxt(data_dir + cov_filename[s], dtype = float)[:self.nbin,:self.nbin] #.reshape((self.nbin, self.nbin))
                                for s in range(self.nlike)}
        for r in range(self.nlike):
            self.covmat[self.regions[r]] = self.cull_covmat(self.covmat[self.regions[r]])

    
    def load_leakage(self, leak_filename, data_dir = ''):
        # Note: this assumes all TE bins are the same length (it is hardcoded at a later point).
        self.leakage = {self.regions[s]: np.loadtxt(data_dir + leak_filename[s],unpack=True)[:self.nbinte[0]] for s in range(self.nlike)}
        self.a = {self.regions[s]:[1.,1.] for s in range(self.nlike)}
        self.idxleak = {np.str(self.freqs[i]):i for i in range(len(self.freqs))}
        
    
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
        #self.l_max = 9000

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
        x_model = {self.regions[s]: np.zeros(self.nbin) for s in range(self.nlike)}

        for r in range(self.nlike):
            for s in self.required_cl:
                cltmp = np.zeros(self.lmax_win+1)
                cltmp[2:self.tt_lmax+1] = cl[s][2:self.tt_lmax+1]
                for j in range(self.lenbin[s]):
                    sidx1 = self.freqspec[s,"freq1"][j]
                    sidx2 = self.freqspec[s,"freq2"][j]
                    bidx1 = self.nbisp[s,'bin'+np.str(j)]
                    bidx2 = self.nbisp[s,'bin'+np.str(j+1)]
                    x_model[self.regions[r]][bidx1:bidx2] = self.win_func[self.regions[r]][bidx1:bidx2,:self.lmax_win] \
                                @ ((cltmp[2:self.lmax_win+1] + fg_model[s,'all',sidx1,sidx2,"both"]\
                                    + fg_model[s,'all',sidx1,sidx2,self.regions[r]])*self.ellfact)
                    self.log.debug('Spectrum: '+s+
                                    ', freq1 = '+np.str(sidx1)+
                                    ', freq2 = '+np.str(sidx2)+
                                    ', bin range ['+np.str(bidx1)+','+np.str(bidx2)+']')

        return x_model

    def _doLeakage(self,vecmodel):
        for r in range(self.nlike):          
            self.log.debug("Leakage model {} rerion for TE".format(self.regions[r]))
            for j in range(self.lenbin["te"]):
                bidx1 = self.nbisp["te",'bin'+np.str(j)]
                bidx2 = self.nbisp["te",'bin'+np.str(j+1)]

                k = j - int(np.round(j/3,0)) # Trick to have the right indexing for TT
                ttidx1 = self.nbisp["tt",'bin'+np.str(k)]
                ttidx2 = self.nbisp["tt",'bin'+np.str(k+1)]

                freq = self.freqspec["te","freq2"][j]
                lidx = self.idxleak[np.str(freq)]

                self.log.debug("bin range [{}:{}], leakage range [{}:{}], leakage idx {} ".format(bidx1,
                                                    bidx2,ttidx1,ttidx2,lidx))

                vecmodel[self.regions[r]][bidx1:bidx2] += vecmodel[self.regions[r]][ttidx1:ttidx2]*self.a[self.regions[r]][lidx]*self.leakage[self.regions[r]][lidx+1]
            
            self.log.debug("Leakage model {} rerion for EE".format(self.regions[r]))
            for j in range(self.lenbin["ee"]):
                bidx1 = self.nbisp["ee",'bin'+np.str(j)]
                bidx2 = self.nbisp["ee",'bin'+np.str(j+1)]

                ttidx1 = self.nbisp["tt",'bin'+np.str(j)]
                ttidx2 = self.nbisp["tt",'bin'+np.str(j+1)]

                k = j + int(np.round(j/3,0)) # Trick to have the right indexing for TE
                teidx11 = self.nbisp["te",'bin'+np.str(k)]
                teidx12 = self.nbisp["te",'bin'+np.str(k+1)]

                l = j + int(np.round(j/1.5,0)) # Trick to have the right indexing for TE
                teidx21 = self.nbisp["te",'bin'+np.str(l)]
                teidx22 = self.nbisp["te",'bin'+np.str(l+1)]

                freqj = self.freqspec["ee","freq2"][j]
                lidxj = self.idxleak[np.str(freqj)]

                freqi = self.freqspec["ee","freq1"][j]
                lidxi = self.idxleak[np.str(freqi)]

                vecmodel[self.regions[r]][bidx1:bidx2] += vecmodel[self.regions[r]][teidx11:teidx12]*self.a[self.regions[r]][lidxi]*self.leakage[self.regions[r]][lidxi+1] \
                                                        + vecmodel[self.regions[r]][teidx21:teidx22]*self.a[self.regions[r]][lidxj]*self.leakage[self.regions[r]][lidxj+1] \
                                                        + vecmodel[self.regions[r]][ttidx1:ttidx2]*self.a[self.regions[r]][lidxi]*self.leakage[self.regions[r]][lidxi+1] \
                                                        * self.a[self.regions[r]][lidxj]*self.leakage[self.regions[r]][lidxj+1]

                self.log.debug("bin range [{}:{}], leakage range TiTj [{}:{}], ".format(bidx1,bidx2,ttidx1,ttidx2))
                self.log.debug("leakage range TiTj [{}:{}], ".format(teidx11,teidx12))
                self.log.debug("leakage range TjEi [{}:{}], ".format(teidx21,teidx22))
                self.log.debug("leakage idx i {}, leakage idx j {}.".format(lidxi,lidxj))
            
        return vecmodel

    def calibrate(self,vecmodel):
        for r in range(self.nlike):
            self.log.debug("Calibrate region {}".format(self.regions[r]))
            # calibrate TT
            self.log.debug("Calibrate TT")
            for j in range(self.lenbin["tt"]):
                bidx1 = self.nbisp["tt",'bin'+np.str(j)]
                bidx2 = self.nbisp["tt",'bin'+np.str(j+1)]
                freqi = self.freqspec["tt","freq1"][j]
                cidxi = self.idxleak[np.str(freqi)]
                freqj = self.freqspec["tt","freq2"][j]
                cidxj = self.idxleak[np.str(freqj)]
                vecmodel[self.regions[r]][bidx1:bidx2] = vecmodel[self.regions[r]][bidx1:bidx2] * self.ct[cidxi] * self.ct[cidxj]
                self.log.debug("Bin range [{}:{}], idxi = {}, idxj = {}".format(bidx1,bidx2,cidxi,cidxj))
            # calibrate TE
            self.log.debug("Calibrate TE")
            for j in range(self.lenbin["te"]):
                bidx1 = self.nbisp["te",'bin'+np.str(j)]
                bidx2 = self.nbisp["te",'bin'+np.str(j+1)]
                freqi = self.freqspec["te","freq1"][j]
                cidxi = self.idxleak[np.str(freqi)]
                freqj = self.freqspec["te","freq2"][j]
                cidxj = self.idxleak[np.str(freqj)]
                vecmodel[self.regions[r]][bidx1:bidx2] = vecmodel[self.regions[r]][bidx1:bidx2] * self.ct[cidxi] * self.ct[cidxj] * self.yp[cidxj]
                self.log.debug("Bin range [{}:{}], idxi = {}, idxj = {}".format(bidx1,bidx2,cidxi,cidxj))
            # calibrate EE
            self.log.debug("Calibrate EE")
            for j in range(self.lenbin["ee"]):
                bidx1 = self.nbisp["ee",'bin'+np.str(j)]
                bidx2 = self.nbisp["ee",'bin'+np.str(j+1)]
                freqi = self.freqspec["ee","freq1"][j]
                cidxi = self.idxleak[np.str(freqi)]
                freqj = self.freqspec["ee","freq2"][j]
                cidxj = self.idxleak[np.str(freqj)]
                vecmodel[self.regions[r]][bidx1:bidx2] = vecmodel[self.regions[r]][bidx1:bidx2] * self.ct[cidxi] * self.ct[cidxj] * self.yp[cidxi] * self.yp[cidxj]
                self.log.debug("Bin range [{}:{}], idxi = {}, idxj = {}".format(bidx1,bidx2,cidxi,cidxj))

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
        if self.doLeakage:
            x_model = self._doLeakage(x_model)

        x_model = self.calibrate(x_model)

        for r in range(self.nlike):
            diff_vec = self.b_dat[self.regions[r]] - x_model[self.regions[r]]
            logptmp = - 0.5 * diff_vec @ self.inv_cov[self.regions[r]] @ diff_vec
            logp += logptmp
        
            self.log.debug(
            "Χ² value computed for region {}"
            "Χ² = {}".format(self.regions[r],-2 * logptmp))
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

    if not isinstance(fdust, np.ndarray):
        fdust = np.array(fdust)


    if not isinstance(fsz, np.ndarray):
        fsz = np.array(fsz)


    if not isinstance(fsyn, np.ndarray):
        fsyn = np.array(fsyn)

    fszcorr,f0 = sz_func(fsz,nu_0)
    planckratiod = plankfunctionratio(fdust,nu_0,fg_params["T_d"])
    fluxtempd = flux2tempratiod(fdust,nu_0)


    model = {}
    
    #TT Foreground
    model["tt", "cibc","both"] = fg_params["a_c"] * cibc(
        {"nu": fdust, "nu_0": nu_0,
         "temp": fg_params["T_d"], "beta": fg_params["beta_p"]},
         {'ell':ell, 'ell_0':ell_0})

    model["tt", "cibp","both"] = fg_params["a_p"] * cibp(
        {"nu": fdust, "nu_0": nu_0,
         "temp": fg_params["T_d"], "beta": fg_params["beta_p"]},
        {"ell": ell_clp, "ell_0": ell_0clp, "alpha": 1})

    model["tt", "tSZ","both"] = fg_params["a_tSZ"] * tsz(
        {"nu": fsz, "nu_0": nu_0},
        {"ell": ell, "ell_0": ell_0})

    model["tt", "kSZ","both"] = fg_params["a_kSZ"] * ksz(
        {"nu": (fsz/fsz)},
        {"ell": ell, "ell_0": ell_0})
    
    model["ee", "radio", "both"] = fg_params["a_psee"] * radio(
        {"nu": fsyn, "nu_0": nu_0, "beta": -0.5 - 2.},
        {"ell": ell, "ell_0": ell_0,"alpha":1})
    
    model["te", "radio", "both"] = fg_params["a_pste"] * radio(
        {"nu": fsyn, "nu_0": nu_0, "beta": -0.5 - 2.},
        {"ell": ell, "ell_0": ell_0,"alpha":1})

    if "wide" in partitions:
        model["tt", "dust", "wide"] = fg_params["a_gw"] * dust(
            {"nu": fdust, "nu_0": nu_0,
            "temp": fg_params["T_dd"], "beta": 1.5},
            {"ell": ell, "ell_0": 500., "alpha": -0.6})

        model["tt", "radio", "wide"] = fg_params["a_sw"] * radio(
            {"nu": fsyn, "nu_0": nu_0, "beta": -0.5 - 2.},
            {"ell": ell_clp, "ell_0": ell_0clp,"alpha":1})      
        
        model["ee", "dust", "wide"] = fg_params["a_geew"] * dust(
            {"nu": fdust, "nu_0": nu_0,
            "temp": fg_params["T_dd"], "beta": 1.5},
            {"ell": ell, "ell_0": 500., "alpha": -0.4})
        
        model["te", "dust", "wide"] = fg_params["a_gtew"] * dust(
            {"nu": fdust, "nu_0": nu_0,
            "temp": fg_params["T_dd"], "beta": 1.5},
            {"ell": ell, "ell_0": 500., "alpha": -0.4})
        
    if "deep" in partitions:
        model["tt", "dust", "deep"] = fg_params["a_gd"] * dust(
            {"nu": fdust, "nu_0": nu_0,
            "temp": fg_params["T_dd"], "beta": 1.5},
            {"ell": ell, "ell_0": 500., "alpha": -0.6})
        
        model["tt", "radio", "deep"] = fg_params["a_sd"] * radio(
            {"nu": fsyn, "nu_0": nu_0, "beta": -0.5 - 2.},
            {"ell": ell_clp, "ell_0": ell_0clp,"alpha":1})
        
        model["ee", "dust", "deep"] = fg_params["a_geed"] * dust(
            {"nu": fdust, "nu_0": nu_0,
            "temp": fg_params["T_dd"], "beta": 1.5},
            {"ell": ell, "ell_0": 500., "alpha": -0.4})
        
        model["te", "dust", "deep"] = fg_params["a_gted"] * dust(
            {"nu": fdust, "nu_0": nu_0,
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
                        if comp == "tSZxcib" and partitions[p] == "both":
                            fg_dict[s, comp, f1, f2, partitions[p]] = -2. * np.sqrt(fg_params['a_c'] * fg_params['a_tSZ']) * fg_params['xi']\
                                  * clszcib * ((fdust[c1]**fg_params['beta_p'] * fszcorr[c2]\
                                  * planckratiod[c1] * fluxtempd[c1]+fdust[c2]**fg_params['beta_p'] \
                                  * fszcorr[c1] * planckratiod[c2] * fluxtempd[c2])/(2 * nu_0**fg_params['beta_p'] * f0))
                        else:
                            try:
                                fg_dict[s, comp, f1, f2, partitions[p]] = model[s, comp, partitions[p]][c1, c2]
                                fg_dict[s, "all", f1, f2 , partitions[p]] += fg_dict[s, comp, f1, f2, partitions[p]]
                            except: continue

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

  
     
