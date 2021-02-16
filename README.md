# ACTMFLike

This is a [Cobaya](https://cobaya.readthedocs.io/en/latest/) likelihood based on the original [Fortran code](https://lambda.gsfc.nasa.gov/product/act/act_dr4_likelihood_multi_info.cfm) presented in [Choi et al. 2020](https://arxiv.org/abs/2007.07289) and [Aiola et al. 2020](https://arxiv.org/abs/2007.07288). 

## Installing the code

The first step is to clone this repository to some location

```
$ git clone https://github.com/umbnat92/ACTMFLike.git /where/to/clone
```

Then we can install this likelihood and its dependencies via

```
$ pip install -e /where/to/clone
```

## Installing the Likelihood

To install the likelihood and all required code, such as [CAMB](https://github.com/cmbant/CAMB), you can use the next command

```
$ cobaya-install /where/to/clone/examples/MFLikeACT_example.yaml -p /where/to/put/packages
```

At this stage you can modify the path of dataset (including foregrounds) inside the `MFLikeACT_example.yaml` file. The data are available at [this](https://lambda.gsfc.nasa.gov/product/act/act_dr4_likelihood_multi_info.cfm) page. 

## Running the code

You can run the `actmflike` likelihood by doing

```
$ cobaya-run /where/to/clone/examples/MFLikeACT_example.yaml -p /where/to/put/packages
```

## License

This code was made by Umberto Natale and use the python transcription of the original Fortran code made by Hidde Jense. The code that calculate the log-likelihood for an ACT foreground model is available [here](https://github.com/HTJense/loglike). 
