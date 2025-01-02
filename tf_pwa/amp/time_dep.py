import tensorflow as tf

from tf_pwa.amp.amp import BaseAmplitudeModel, register_amp_model
from tf_pwa.amp.core import HelicityDecay, register_decay, to_complex
from tf_pwa.config import get_config
from tf_pwa.data import data_shape


def cal_gp_gm(t, gamma, delta_m, delta_gamma):
    r"""
    .. math::
        g_{+}(t) = \left[\cosh\frac{\Delta\Gamma t}{4}\cos\frac{\Delta mt}{2}-i\sinh\frac{\Delta\Gamma t}{4}\sin\frac{\Delta mt}{2}\right]\exp\left[-\frac{\Gamma t}{2}\right],

    .. math::
        g_{-}(t) = \left[-\sinh\frac{\Delta\Gamma t}{4}\cos\frac{\Delta mt}{2}+i\cosh\frac{\Delta\Gamma t}{4}\sin\frac{\Delta mt}{2}\right]\exp\left[-\frac{\Gamma t}{2}\right].

    :math:`\exp\left[-i mt\right]` is not included, since its has :math:`|A|=1`.

    """

    decay = tf.exp(-t / 2 * gamma)
    dmt = delta_m * t / 2
    cmt = tf.math.cos(dmt)
    smt = tf.math.sin(dmt)
    dgt = delta_gamma * t / 4
    cgt = tf.math.cosh(dgt)
    sgt = tf.math.sinh(dgt)
    gp = tf.complex(decay * cgt * cmt, -decay * sgt * smt)
    gm = tf.complex(-decay * sgt * cmt, decay * cgt * smt)
    return gp, gm


@register_decay("time_dep_params")
class TimeDepParamsHelicityDecay(HelicityDecay):
    """
    model with `g_ls` and `g_lsb` which dependent on the `tag` in data.

    """

    def init_params(self):
        super().init_params()
        ls = self.get_ls_list()
        self.g_ls_bar = self.add_var(
            "g_lsb", is_complex=True, polar=self.params_polar, shape=(len(ls),)
        )

    def get_mix_g_ls(self, data, data_p, **kwargs):
        g_ls = self.get_g_ls()
        g_ls_bar = self.get_g_ls_bar()
        all_data = kwargs["all_data"]
        ones = tf.ones_like(data_p[self.core]["m"])
        tag = all_data.get("tag", ones)
        return tf.where(tag[..., None] > 0, g_ls, g_ls_bar)

    def get_ls_amp(self, data, data_p, **kwargs):
        g_ls = self.get_mix_g_ls(data, data_p, **kwargs)
        q0 = self.get_relative_momentum2(data_p, False)
        data["|q0|2"] = q0
        q = self.cache_relative_p2(data, data_p)
        if self.has_barrier_factor:
            bf = self.get_barrier_factor2(
                data_p[self.core]["m"], q, q0, self.d
            )
            mag = g_ls
            bf = to_complex(bf)
            m_dep = mag * tf.cast(bf, mag.dtype)
        else:
            m_dep = g_ls  # tf.reshape(g_ls, (1, -1))
        return m_dep

    def get_g_ls_bar(self):
        gls = self.g_ls_bar()
        if self.ls_index is None:
            ret = tf.stack(gls)
        else:
            ret = tf.stack([gls[k] for k in self.ls_index])
        if self.mask_factor:
            return tf.ones_like(ret)
        return ret


@register_decay("time_dep_gls")
class TimeDepHelicityDecay(TimeDepParamsHelicityDecay):
    r"""
    Implement time effect  `arxiv:0904.1869 <https://arxiv.org/abs/0904.1869>`_

    .. math::
        |M(t)\rangle=g_{+}(t)|M\rangle + \frac{q}{p} g_{-}(t)|\bar{M}\rangle, |\bar{M}(t)\rangle=\frac{p}{q}g_{-}(t)|M\rangle + g_{+}(t)|\bar{M}\rangle,

    :math:`|M\rangle` will use `g_ls` and :math:`|\bar{M}\rangle` will use `g_lsb`.


    """

    def init_params(self, *args, **kwargs):
        super().init_params(*args, **kwargs)
        self.core.delta_m = self.core.add_var("delta_m", value=0.0)
        self.core.delta_gamma = self.core.add_var("delta_gamma", value=0.0)
        self.core.gamma = self.core.add_var("gamma", value=1.0)
        self.core.poq = self.core.add_var(
            "poq", is_complex=True, fix=True, fix_vals=(1.0, 0.0)
        )

    def get_mix_g_ls(self, data, data_p, **kwargs):
        g_ls = self.get_g_ls()
        g_ls_bar = self.get_g_ls_bar()
        all_data = kwargs["all_data"]
        ones = tf.ones_like(data_p[self.core]["m"])

        t = all_data.get("time", 0.0 * ones)
        gp, gm = cal_gp_gm(
            t,
            self.core.gamma(),
            self.core.delta_m(),
            self.core.delta_gamma(),
        )
        phase = self.core.poq()
        ret1 = gp[..., None] * g_ls + phase * gm[..., None] * g_ls_bar
        ret2 = 1 / phase * gm[..., None] * g_ls + gp[..., None] * g_ls_bar
        tag = all_data.get("tag", ones)
        ret = tf.where(tag[..., None] > 0, ret1, ret2)
        return ret


@register_amp_model("time_dep_params")
class TimeDepParamsAmplitudeModel(BaseAmplitudeModel):
    """
    Implement time effect `arxiv:0904.1869 <https://arxiv.org/abs/0904.1869>`_
    Require to use decay model `time_dep_params`.

    """

    def init_params(self, *args, **kwargs):
        super().init_params()
        top = self.decay_group.top
        top.delta_m = top.add_var("delta_m", value=0.0)
        top.delta_gamma = top.add_var("delta_gamma", value=0.0)
        top.gamma = top.add_var("gamma", value=1.0)
        top.poq = top.add_var(
            "poq", is_complex=True, fix=True, fix_vals=(1.0, 0.0), polar=True
        )

    def eval_A_Abar(self, data):
        top = self.decay_group.top
        ones = tf.ones((1,), dtype=get_config("dtype"))
        A = self.decay_group.get_amp({**data, "tag": ones})
        Abar = self.decay_group.get_amp({**data, "tag": -ones})
        return A, Abar

    def eval_A_Abar_time(self, data):
        A, Abar = self.eval_A_Abar(data)

        top = self.decay_group.top
        ones = tf.ones((1,), dtype=get_config("dtype"))
        t = data.get("time", 0.0 * ones)
        gp, gm = cal_gp_gm(
            t, 1 / top.life_time(), top.delta_m(), top.delta_gamma()
        )
        n_pad = len(A.shape) - len(t.shape)
        phase = top.poq()
        for i in range(n_pad):
            gp = tf.expand_dims(gp, -1)
            gm = tf.expand_dims(gm, -1)
        ret1 = gp * A + phase * gm * Abar
        ret2 = 1 / phase * gm * A + gp * Abar
        return ret1, ret2

    def pdf(self, data):
        A, Abar = self.eval_A_Abar_time(data)
        ret1 = self.decay_group.sum_with_polarization(A)
        ret2 = self.decay_group.sum_with_polarization(Abar)
        ones = tf.ones((1,), dtype=get_config("dtype"))
        tag = data.get("tag", ones)
        return tf.where(tag > 0, ret1, ret2)


@register_amp_model("time_dep_cp")
class TimeDepCpAmplitudeModel(TimeDepParamsAmplitudeModel):
    """

    Time dependent amplitude with self-CP related process, the Abar will calculate through :math:`\\bar{A}(p_{+}, p_{-}, p_{0}) = A(-p_{-}, -p_{+}, -p_{0})`

    """

    def init_params(self, *args, **kwargs):
        super().init_params(*args, **kwargs)
        top = self.decay_group.top
        top.poq.freed()

    def eval_A_Abar(self, data):
        A = self.decay_group.get_amp(data)
        Abar = self.decay_group.get_amp(data["cp_swap"])
        return A, Abar


def fix_cp_params(config, r1, r2):
    """
    using the same paramters for A and Abar

    only work for dalitz plot (all J=0 for initial and final particles)

    """
    config.get_params()  # asume parameters exist
    decay_group = config.get_decay()
    fix_params = {}
    same_var = []
    free_var = []
    for a, b in zip(r1, r2):
        chain_a = decay_group.get_decay_chain(a)
        chain_b = decay_group.get_decay_chain(b)
        dec1 = [i for i in chain_a if i.core == chain_a.top][0]
        dec2 = [i for i in chain_b if i.core == chain_b.top][0]
        dec1.g_ls.sameas(dec2.g_ls_bar)
        dec2.g_ls.sameas(dec1.g_ls_bar)
        if not (chain_a.total.is_fixed()):
            dec1.g_ls.set_fix_idx(free_idx=0)
        if not (chain_b.total.is_fixed()):
            dec2.g_ls.set_fix_idx(free_idx=0)
        chain_a.total.sameas(chain_b.total)
        chain_a.total.set_fix_idx(0, fix_vals=1.0)
    for i in decay_group.resonances:
        if str(i) not in r1 and str(i) not in r2:
            chain = decay_group.get_decay_chain(i)
            dec = [i for i in chain if i.core == chain.top][0]
            if i.J % 2 == 1:  # (-1)^l
                dec.g_ls_bar.set_fix_idx(0, fix_vals=-1.0)
            else:
                dec.g_ls_bar.set_fix_idx(0, fix_vals=1.0)
