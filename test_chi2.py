import os
import tempfile

import camb
import cobaya
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from cobaya.model import get_model

from actmflike import MFLikeACT

plotFG = False

nuisance_params = {
    "yp1": 0.985218,
    "yp2": 0.970455,
    "a_tSZ": 5.28206,
    "a_kSZ": 0.636907,
    "xi": 0.143992,
    "a_p": 6.57088,
    "a_c": 3.15598,
    "beta_p": 2.89395,
    "a_sd": 3.74655,
    "a_gd": 2.78366,
    "a_gted": 0.129318,
    "a_geed": 0.0723663,
    "a_pste": 0.0427959,
    "a_psee": 0.0223792,
    "a_sw": 22.5619,
    "a_gw": 8.70528,
    "a_gtew": 0.355872,
    "a_geew": 0.130194,
    "T_d": 9.7,
    "T_dd": 19.6,
    "n_CIBC": 1.2
}

cosmo_params = {
    "theta_MC_100": {"value": 1.04222, "drop": True},
    "cosmomc_theta": {"value": "lambda theta_MC_100: 1.e-2*theta_MC_100"},
    "logA": {"value": 3.04702, "drop": True},
    "As": {"value": "lambda logA: 1e-10*np.exp(logA)"},
    "ombh2": 0.0214494,
    "omch2": 0.118542,
    "ns": 1.00622,
    "tau": 0.0633689,
}


mflike_config = {
    "actmflike.MFLikeACT": {
        "data_folder": "/data/ACT/",
        "enable_tt": True,
        "enable_te": True,
        "enable_ee": True,

        "data": {
            "frequencies": [98, 150],
            "regions": {
                "deep": {"specname": "coadd_cl_15mJy_data_200124.txt",
                        "covname": "coadd_cov_15mJy_200519.txt",
                        "bpwfname": "coadd_bpwf_15mJy_191127_lmin2.txt",
                        "leak_TE": "leak_TE_deep_200519.txt"},
                "wide": {"specname": "coadd_cl_100mJy_data_200124.txt",
                        "covname": "coadd_cov_100mJy_200519.txt",
                        "bpwfname": "coadd_bpwf_100mJy_191127_lmin2.txt",
                        "leak_TE": "leak_TE_wide_200519.txt"}
                        }
                },

        "foregrounds": {
            "normalisation": {
                "nu_0": 150.0,
                "ell_0": 3000,
                "T_CMB": 2.72548
                            },
            "partitions": ["deep","wide"],

            "frequencies": {
                "nominal": [98, 150],
                "fdust": {"deep": [98.8, 151.2], "wide": [98.8, 150.9]},
                "fsz": {"deep": [98.4, 150.1], "wide": [98.4, 149.9]},
                "fsyn": {"deep": [95.8, 147.2], "wide": [95.8, 147.1]},
                            },

            "external_cl": {
                "cibc": "/data/ACT/Foregrounds/cib_extra.dat",
                "tszxcib": "/data/ACT/Foregrounds/sz_x_cib_template.dat"
                            },

            "components": {
                "tt": {"kSZ", "cibp", "radio", "tSZ", "cibc", "tSZxcib", "dust"},
                "te": {"radio", "dust"},
                "ee": {"radio", "dust"}
                        },
                        }

                            }
                }        


packages_path = '/Packages/MFLikeACT/'

info = {
    "debug": True,
    "params": {**cosmo_params,**nuisance_params},
    "likelihood": mflike_config,
    "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
    "packages_path": packages_path,
}

model = get_model(info)
mflike = model.likelihood["actmflike.MFLikeACT"]

loglikes = model.loglikes({})[0]
print("Χ² value = {}".format(-2 * loglikes[0]))

if plotFG:
    from actmflike import get_foreground_model

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
                                "a_psee": 0.0223792,
                                "a_sw": 0.2249012E+02,
                                "a_gw": 0.8717251E+01,
                                "a_gtew": 0.3559930E+00,
                                "a_geew": 0.1293905E+00,
                                "T_dd": 19.6,
                                "T_d": 9.7,
                                "n_CIBC": 1.2,
                                }


    ell = np.arange(2, 7925)
    fg_models = get_foreground_model(nuisance_params, mflike.foregrounds, mflike.freqs, ell=ell)

    modes = ["tt","ee","te"]
    for mode in modes:
        components = mflike.foregrounds["components"][mode]
        freqs = sorted(mflike.freqs)
        nfreqs = len(freqs)
        for r in mflike.foregrounds["partitions"]:
            fig, axes = plt.subplots(nfreqs, nfreqs, sharex=True, sharey=True, figsize=(10, 10))
            from itertools import product

            for i, cross in enumerate(product(freqs, freqs)):
                idx = (i % nfreqs, i // nfreqs)
                ax = axes[idx]
                if idx in zip(*np.triu_indices(nfreqs, k=1)):
                    fig.delaxes(ax)
                    continue
                ax.plot(ell, fg_models[mode, "all", cross[0], cross[1], r], color="k")
                for compo in components:
                    ax.plot(ell, fg_models[mode, compo, cross[0], cross[1], r])
                #ax.plot(ell, dls[mode], color="tab:gray")
                ax.legend([], title="{}x{} GHz".format(*cross))
                if mode == "tt":
                    ax.set_yscale("log")
                    ax.set_ylim(10 ** -1, 10 ** 4)
                else:
                    ax.set_yscale("log")
                    ax.set_ylim(10 ** -3, 10 ** 2)

            for i in range(nfreqs):
                axes[-1, i].set_xlabel("$\ell$")
                axes[i, 0].set_ylabel("$D_\ell$")
            fig.legend(["all"] + list(components), title=mode.upper(), bbox_to_anchor=(0.8, 1))
            plt.title('Region - '+r)
            plt.tight_layout()
            plt.savefig(r+'_fgmodel_'+mode+'.pdf')


