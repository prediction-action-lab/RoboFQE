"""Runtime monkey-patches for upstream LIBERO.

`apply_no_proprio_patch()` must be called before `LIBEROPolicyWrapper` builds
the policy when the BC ckpt was trained without proprio (e.g. `img128_noprop`).
It is a no-op for ckpts that use proprio.
"""


def apply_no_proprio_patch():
    """Make `ExtraModalityTokens` accept all `use_*=False` (VLA-style).

    Upstream LIBERO asserts at least one proprio modality is enabled in
    `ExtraModalityTokens.__init__` (bc_transformer_policy.py line ~50:
    ``assert extra_low_level_feature_dim > 0, "[error] no extra information"``).
    For VLA-style image+language-only ablations this assert blocks startup.

    Doing the fix as a runtime monkey-patch (rather than forking LIBERO)
    means no-proprio BC ckpts load on **any** LIBERO checkout
    — clean upstream, CHTC slot, fresh clone — without requiring a patched
    fork. Safe to apply on top of an already-patched LIBERO (re-installs
    equivalent functions).

    Logically identical to patching
    libero/lifelong/models/bc_transformer_policy.py:50-119 in place.
    """
    import torch
    import torch.nn as nn
    import libero.lifelong.models.bc_transformer_policy as bct

    def patched_init(self, use_joint=False, use_gripper=False, use_ee=False,
                     extra_num_layers=0, extra_hidden_size=64,
                     extra_embedding_size=32):
        nn.Module.__init__(self)
        self.use_joint = use_joint
        self.use_gripper = use_gripper
        self.use_ee = use_ee
        self.extra_embedding_size = extra_embedding_size

        joint_states_dim = 7
        gripper_states_dim = 2
        ee_dim = 3
        self.num_extra = int(use_joint) + int(use_gripper) + int(use_ee)
        self.extra_encoders = {}

        def _build_mlp(modality_name, dim):
            if extra_num_layers > 0:
                layers = [nn.Linear(dim, extra_hidden_size)]
                for _ in range(1, extra_num_layers):
                    layers += [nn.Linear(extra_hidden_size, extra_hidden_size),
                               nn.ReLU(inplace=True)]
                layers += [nn.Linear(extra_hidden_size, extra_embedding_size)]
            else:
                layers = [nn.Linear(dim, extra_embedding_size)]
            self.proprio_mlp = nn.Sequential(*layers)
            self.extra_encoders[modality_name] = {"encoder": self.proprio_mlp}

        for (dim, use_mod, name) in [
            (joint_states_dim, use_joint, "joint_states"),
            (gripper_states_dim, use_gripper, "gripper_states"),
            (ee_dim, use_ee, "ee_states"),
        ]:
            if use_mod:
                _build_mlp(name, dim)

        self.encoders = nn.ModuleList(
            [x["encoder"] for x in self.extra_encoders.values()]
        )

    def patched_forward(self, obs_dict):
        tensor_list = []
        for (use_mod, name) in [
            (self.use_joint, "joint_states"),
            (self.use_gripper, "gripper_states"),
            (self.use_ee, "ee_states"),
        ]:
            if use_mod:
                tensor_list.append(
                    self.extra_encoders[name]["encoder"](obs_dict[name])
                )
        if len(tensor_list) == 0:
            # No-proprio path: return (B, T, 0, E) so downstream token concat
            # sees zero proprio tokens. Recover (B, T) from any 5D image obs.
            for v in obs_dict.values():
                if v.ndim == 5:
                    B, T = v.shape[0], v.shape[1]
                    return v.new_zeros(B, T, 0, self.extra_embedding_size)
            raise ValueError(
                "ExtraModalityTokens no-proprio path: no 5D image obs in "
                "obs_dict to infer (B, T)."
            )
        return torch.stack(tensor_list, dim=-2)

    bct.ExtraModalityTokens.__init__ = patched_init
    bct.ExtraModalityTokens.forward = patched_forward
