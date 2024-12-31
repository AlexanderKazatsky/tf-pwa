import os

import numpy as np

from tf_pwa.amp import time_dep
from tf_pwa.config_loader import ConfigLoader

this_dir = os.path.dirname(os.path.abspath(__file__))


def test_time_dep():
    config = ConfigLoader(f"{this_dir}/config_time_dep.yml")
    amp = config.get_amplitude()

    config2 = ConfigLoader(f"{this_dir}/config_time_dep2.yml")
    config2.set_params(config.get_params())
    amp2 = config2.get_amplitude()

    phsp = config.generate_phsp(10)

    assert np.allclose(amp(phsp).numpy(), amp2(phsp).numpy())
