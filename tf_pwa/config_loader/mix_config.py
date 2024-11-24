import functools

from tf_pwa.amp import AmplitudeModel
from tf_pwa.data import EvalLazy
from tf_pwa.model import FCN

from .multi_config import MultiConfig


class MixAmplitude(AmplitudeModel):
    def __init__(self, amps):
        self.amps = amps
        self.vm = amps[0].vm

    def __getattr__(self, name):
        return getattr(self.amps[-1], name)

    def pdf(self, data):
        ret = 0
        scale = 0
        for amp, data_i in zip(self.amps, data["datas"]):
            w = data_i.get("weight", 1)
            ret = ret + amp(data_i) * w
            scale = scale + w
        return ret / scale


class MixConfig(MultiConfig):
    def get_data(self, name):
        all_data = [i.get_data(name) for i in self.configs]
        if all_data[0] is None:
            return all_data[0]
        ret = []
        for i in range(len(all_data[0])):
            tmp = {"datas": [j[i] for j in all_data]}
            tmp["weight"] = sum(j[i]["weight"] for j in all_data)
            ret.append(tmp)
        return ret

    def get_all_data(self):
        datafile = ["data", "phsp", "bg", "inmc"]
        data, phsp, bg, inmc = [self.get_data(i) for i in datafile]
        self._Ngroup = len(data)
        assert len(phsp) == self._Ngroup
        if bg is None:
            bg = [None] * self._Ngroup
        if inmc is None:
            inmc = [None] * self._Ngroup
        assert len(bg) == self._Ngroup
        assert len(inmc) == self._Ngroup
        return data, phsp, bg, inmc

    def get_amplitude(self, vm=None, name=""):
        amps = self.get_amplitudes(vm=vm)
        return MixAmplitude(amps)

    @functools.lru_cache()
    def _get_model(self, vm=None, name=""):
        amp = self.get_amplitude(vm=vm, name=name)
        models = self.configs[0]._get_model(vm=vm, name=name, amp=amp)
        return models

    def get_fcn(self, datas=None, vm=None, batch=65000):
        all_data = datas
        if all_data is None:
            all_data = self.get_all_data()
        model = self._get_model(vm=vm)
        return self.configs[-1].get_fcn(
            all_data=all_data, vm=model[0].vm, batch=batch, model=model
        )
