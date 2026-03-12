DATA_PATH = "/Users/kaihung/optimalEnsemble/OTFactorDiscovery/var-reduction/data"
OUTPUT_PATH = "/Users/kaihung/optimalEnsemble/OTFactorDiscovery/var-reduction/outputs"
AMR_UTI_PATH = "/Users/kaihung/optimalEnsemble/OTFactorDiscovery/var-reduction/data/amr-uti-antimicrobial-resistance-in-urinary-tract-infections-1.0.0"

hyperparams = {
    "seed": 125, 
    "sigma_y": 1,
    "sigma_z": 1,
    "lr": 0.005,
    "epsilon": 0.001,
    "max_iter": 4000,
    "growing_lambda": True,
    "init_lam": 0,
    "warm_stop": 500,
    "max_lam": 250,
    "mock_prob": False,
    "eta": 0.01,
    "verbose": True,
    "monitoring_skip": 1,
    "monitoring_skip_2": 5
}

gauss_params = {
    "mu_A": -2,
    "sigma_A": 1,
    "mu_B": 2,
    "sigma_B": 1,
    "num_samples": 200
}