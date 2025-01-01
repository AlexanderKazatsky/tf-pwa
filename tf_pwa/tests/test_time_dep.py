import os

import numpy as np

from tf_pwa.amp.time_dep import fix_cp_params
from tf_pwa.config_loader import ConfigLoader

this_dir = os.path.dirname(os.path.abspath(__file__))


def test_time_dep():
    config = ConfigLoader(f"{this_dir}/config_time_dep.yml")
    config.set_params({"A_life_time": 1.5, "A_delta_m": 0.5, "A_poqi": -0.77})
    amp = config.get_amplitude()

    config2 = ConfigLoader(f"{this_dir}/config_time_dep2.yml")
    config2.set_params(config.get_params())
    amp2 = config2.get_amplitude()

    phsp = config.generate_phsp(10)

    assert np.allclose(amp(phsp).numpy(), amp2(phsp).numpy())


def test_time_dep_cp():
    config = ConfigLoader(f"{this_dir}/config_time_dep.yml")
    fix_cp_params(config, ["R_BD"], ["R_CD"])
    config.set_params({"A_life_time": 1.5, "A_delta_m": 0.5, "A_poqi": -0.77})
    amp = config.get_amplitude()

    config2 = ConfigLoader(f"{this_dir}/config_time_dep2.yml")
    fix_cp_params(config2, ["R_BD"], ["R_CD"])
    config2.set_params(config.get_params())
    amp2 = config2.get_amplitude()

    config_cp = ConfigLoader(f"{this_dir}/config_time_dep3.yml")
    config_cp.set_params(
        {k: v for k, v in config.get_params().items() if "lsb" not in k}
    )
    amp_cp = config_cp.get_amplitude()

    phsp = config_cp.generate_phsp(10)

    a = amp_cp(phsp).numpy()
    phsp2 = phsp.copy()
    del phsp2["cp_swap"]
    b = amp(phsp2).numpy()
    c = amp2(phsp2).numpy()

    assert np.allclose(a, b, c)
