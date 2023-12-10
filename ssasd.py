from dataclasses import dataclass, field
import math
import copy
from typing import Type, Literal, Callable, Any
import inspect
import torch
import torchvision.transforms.functional as F
import einops

@dataclass
class SSaInfo:
    hooks: list = field(default_factory=list)
    original_call: Callable|None = None
    original_compute_merge: Callable|None = None
    max_downsample: int = 2
    scale: int = 2
    steps: range|float = 1.0
    interpolation: F.InterpolationMode|None = F.InterpolationMode.BILINEAR
    total_steps: int = -1
    current_steps: int = -1
    latent_width: int = -1
    latent_height: int = -1
    use_tome: bool = False


def shrink(hidden_states: torch.Tensor, ssa_info: SSaInfo) -> tuple[torch.Tensor, tuple[int, int, int]|None, bool]:
    """(h, w) -> (h/scale, w/scale)"""
    steps = ssa_info.current_steps
    if steps not in ssa_info.steps:
        return hidden_states, None, False
    
    b, hw, d = hidden_states.shape

    base_hw = ssa_info.latent_width * ssa_info.latent_height
    assert base_hw % hw == 0, f'base={base_hw}, input={hw}'
    r = (base_hw // hw) ** 0.5
    assert r.is_integer(), f'r={r}, hidden_states.size(1)={hw}, latent_width={ssa_info.latent_width}, latent_height={ssa_info.latent_height}'
    r = int(r)
    #assert r in (1, 2, 4, 8), f'r = {r}'

    if ssa_info.max_downsample < r:
        return hidden_states, None, False

    h, w = ssa_info.latent_height//r, ssa_info.latent_width//r
    assert ssa_info.latent_height % r == 0, f'r={r}, latent_height={ssa_info.latent_height}'
    assert ssa_info.latent_width % r == 0,  f'r={r}, latent_width={ssa_info.latent_width}'
    assert h % ssa_info.scale == 0, f'h={h}, scale={ssa_info.scale}'
    assert w % ssa_info.scale == 0, f'h={w}, scale={ssa_info.scale}'
    
    scale = ssa_info.scale

    # (h, w) -> (h/scale, w/scale)
    
    if ssa_info.interpolation is None:
        # slice
        hidden_states = einops.rearrange(hidden_states, 'b (h w) d -> b h w d', h=h, w=w)
        hidden_states = hidden_states[:, ::scale, ::scale, :]
        hidden_states = einops.rearrange(hidden_states, 'b h w d -> b (h w) d')
    else:
        hidden_states = einops.rearrange(hidden_states, 'b (h w) d -> b d h w', h=h, w=w)
        hidden_states = F.resize(
            hidden_states, (h//scale, w//scale),
            interpolation=ssa_info.interpolation,
            antialias=False,
        )
        hidden_states = einops.rearrange(hidden_states, 'b d h w -> b (h w) d')
    assert hidden_states.shape == (b, (h*w)//(scale*scale), d)
    
    return hidden_states, (h, w, scale), True


def expand(hidden_states: torch.Tensor, ssa_info: SSaInfo, h: int, w: int, scale: int) -> tuple[torch.Tensor, bool]:
    """(h/scale, w/scale) -> (h, w)"""
    if ssa_info.interpolation is None:
        hidden_states = einops.rearrange(hidden_states, 'b (h w) d -> b h w d', h=h//scale, w=w//scale)
        hidden_states = hidden_states.repeat_interleave(scale, dim=1, output_size=h)
        hidden_states = hidden_states.repeat_interleave(scale, dim=2, output_size=w)
        hidden_states = einops.rearrange(hidden_states, 'b h w d -> b (h w) d')
    else:
        hidden_states = einops.rearrange(hidden_states, 'b (h w) d -> b d h w', h=h//scale, w=w//scale)
        hidden_states = F.resize(
            hidden_states, (h, w),
            interpolation=ssa_info.interpolation,
            antialias=False,
        )
        hidden_states = einops.rearrange(hidden_states, 'b d h w -> b (h w) d')
    
    return hidden_states, True


def make_diffusers_sa(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    class SmallAttention(block_class):
        _parent = block_class

        def forward(
                self,
                hidden_states: torch.Tensor,
                *args,
                **kwargs
        ):
            ssa_info = self._ssa_info

            if ssa_info.use_tome:
                # shrink/expand will be processed in m_a and u_a
                return super().forward(hidden_states, *args, **kwargs)

            # (h, w) -> (h/scale, w/scale)
            hidden_states, hws, success = shrink(hidden_states, ssa_info)

            if not success:
                return super().forward(hidden_states, *args, **kwargs)
                
            # self-attention
            hidden_states = super().forward(hidden_states, *args, **kwargs)

            # (h/scale, w/scale) -> (h, w)
            hidden_states, success = expand(hidden_states, ssa_info, *hws)

            if not success:
                raise RuntimeError('must not happen')

            return hidden_states
    
    return SmallAttention


def make_sa(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    return make_diffusers_sa(block_class)


Interpolation = Literal['slice', 'nearest', 'nearest_exact', 'bilinear', 'bicubic']

def apply_patch(
        model_or_pipeline: torch.nn.Module,
        max_downsample: int = 2,
        scale: int = 2,
        steps: range|float = 1.0,
        interpolation: Interpolation = 'bilinear',
        **kwargs):
    """
    Patches a stable diffusion model with SSa.
    
    Args:
    - model_or_pipe: A top level Stable Diffusion module to patch in place.
                     Should have a ".model.diffusion_model"
    - max_downsample [1, 2, 4, or 8]: Apply SSa to layers with at most this amount of downsampling.
                                       E.g., 1 only applies to layers with no downsampling (5/16) while
                                       8 applies to all layers (16/16). Recomended value is 1 or 2.
    - scale: The downscaling factor for the latent. Recomended value is 2.
    - steps: The steps SSa will be applied. Steps will start from 0 (first step) to n-1 where n is the specified number of steps.
             On the steps `n`, SSa will be skipped if `n not in steps` is True.
             If float value is given, the range is computed by `range(0, total_steps * steps)`
    - interpolation ['slice', 'nearest', 'nearest_exact', 'bilinear', 'bicubic']: The interpolation method. The default value is 'bilinear'
    - kwargs: If the caller is not diffusers such as A1111 WebUI, you must pass `total_steps`.
    """

    remove_patch(model_or_pipeline)

    diffusers = is_diffusers(model_or_pipeline)

    ssa_info = SSaInfo(
        max_downsample=max_downsample,
        scale=scale,
        steps=steps,
        interpolation=interpolation_mode(interpolation),
        total_steps=-1,
        current_steps=-1,
        latent_width=-1,
        latent_height=-1,
        use_tome=False,
    )

    if diffusers:
        setup_diffusers(model_or_pipeline, ssa_info)
        hook_diffusers(model_or_pipeline, ssa_info)
    else:
        setup_else(model_or_pipeline, ssa_info, **kwargs)
        hook_else(model_or_pipeline, ssa_info)

    return model_or_pipeline, ssa_info


def setup_diffusers(pipe, ssa_info: SSaInfo):
    """
    initialize SSaInfo
    """
    original_call = pipe.__class__.__call__
    default_params = { k: (v.default, v.empty) for k, v in inspect.signature(pipe.__call__).parameters.items() }

    def call(self, *args, **kwargs):
        def params(param: str):
            x = kwargs.get(param, None)
            if x is None:
                p = default_params[param]
                x, empty = p.default
                if x == empty:
                    raise ValueError(f'{param} is not specified')
            return x
        total_steps = params('num_inference_steps')
        #width = params('width')
        #height = params('height')
        steps = (
            range(0, int(math.ceil(total_steps*ssa_info.steps))) if type(ssa_info.steps) == float
            else ssa_info.steps
        )
        
        ssa_info.steps = steps
        ssa_info.total_steps = total_steps
        ssa_info.current_steps = 0
        
        setup_tome(pipe.unet, ssa_info)
        
        return original_call(self, *args, **kwargs)
    
    pipe.__class__.__call__ = call
    ssa_info.original_call = original_call

    def hook_latent_size(mod, inputs):
        latent = inputs[0]
        b, c, h, w = latent.shape
        ssa_info.latent_width = w
        ssa_info.latent_height = h
    
    def hook_step(mod, inputs, output):
        ssa_info.current_steps += 1
    
    ssa_info.hooks.extend([
        pipe.unet.register_forward_pre_hook(hook_latent_size),
        pipe.unet.register_forward_hook(hook_step)
    ])
    
    pipe._ssa_info = ssa_info


def hook_diffusers(pipe, ssa_info: SSaInfo):
    """
    hook self-attention modules
    """
    for (name, mod) in pipe.unet.named_modules():
        if isinstance_str(mod, 'BasicTransformerBlock'):
            module = mod.attn1
            make_ssa_block_fn = make_diffusers_sa
            module.__class__ = make_ssa_block_fn(module.__class__)
            module._ssa_info = ssa_info


def setup_else(
        model,
        ssa_info: SSaInfo,
        total_steps,
):
    """
    initialize SSaInfo
    """
    steps = (
        range(0, int(math.ceil(total_steps*ssa_info.steps))) if type(ssa_info.steps) == float
        else ssa_info.steps
    )
    
    ssa_info.steps = steps
    ssa_info.total_steps = total_steps
    ssa_info.current_steps = 0

    def hook_latent_size(mod, inputs):
        latent = inputs[0]
        b, c, h, w = latent.shape
        ssa_info.latent_width = w
        ssa_info.latent_height = h
    
    def hook_step(mod, inputs, output):
        ssa_info.current_steps += 1
    
    model = model.model.diffusion_model
    ssa_info.hooks.extend([
        model.register_forward_pre_hook(hook_latent_size),
        model.register_forward_hook(hook_step)
    ])
    
    setup_tome(model, ssa_info)

    model._ssa_info = ssa_info


def hook_else(model, ssa_info: SSaInfo):
    """
    hook self-attention modules
    """
    for (name, mod) in model.model.diffusion_model.named_modules():
        if isinstance_str(mod, 'BasicTransformerBlock'):
            #print('SSaSD hook:', name)
            module = mod.attn1
            make_ssa_block_fn = make_sa
            module.__class__ = make_ssa_block_fn(module.__class__)
            module._ssa_info = ssa_info


def remove_patch(model_or_pipeline):
    """
    Removes a SSa patch from the module if it was already patched.
    """

    diffusers = is_diffusers(model_or_pipeline)

    if diffusers:
        pipe = model_or_pipeline
        ssa_info: SSaInfo|None = getattr(pipe, '_ssa_info', None)
        if ssa_info:
            if ssa_info.original_call is not None:
                pipe.__class__.__call__ = ssa_info.original_call
            remove_tome(ssa_info)
            del pipe._ssa_info
        model = pipe.unet
    else:
        model = model_or_pipeline
        ssa_info: SSaInfo|None = getattr(model, '_ssa_info', None)
        if ssa_info:
            remove_tome(ssa_info)
            del model._ssa_info
        model = model.model.diffusion_model

    for _, mod in model.named_modules():
        if isinstance_str(mod, 'BasicTransformerBlock'):
            module = mod.attn1
            if hasattr(module, '_ssa_info'):
                for hook in module._ssa_info.hooks:
                    hook.remove()
                module._ssa_info.hooks.clear()
                del module._ssa_info
                
                if module.__class__.__name__ == 'SmallAttention':
                    module.__class__ = module._parent

    return model_or_pipeline


def setup_tome(unet: torch.nn.Module, ssa_info: SSaInfo):
    """
    - unet: pipe.unet or model.diffusion_model
    """
    use_tome = False
    if hasattr(unet, '_tome_info'):
        tome_info = unet._tome_info
        p = tome_info['args']['ratio']
        if 0 < p:
            use_tome = True
    ssa_info.use_tome = use_tome
    
    if use_tome:
        import tomesd
        old = tomesd.patch.compute_merge
        do_nothing = tomesd.merge.do_nothing
        def compute_merge(x: torch.Tensor, tome_info: dict[str, Any]):
            x, _, ok = shrink(x, ssa_info)
            if ok:
                tome_info = copy.copy(tome_info)
                h, w = tome_info['size']
                scale = ssa_info.scale
                tome_info['size'] = (h/scale, w/scale)
            m_a, m_c, m_m, u_a, u_c, u_m = old(x, tome_info)
            success = True
            hws = None
            if not (m_a is do_nothing):
                def m_a_new(x, *args, **kwargs):
                    nonlocal success, hws
                    x, hws, success = shrink(x, ssa_info)
                    x = m_a(x)
                    return x
                def u_a_new(x, *args, **kwargs):
                    x = u_a(x)
                    if success:
                        x, _ = expand(x, ssa_info, *hws)
                    return x
                return m_a_new, m_c, m_m, u_a_new, u_c, u_m
            else:
                return m_a, m_c, m_m, u_a, u_c, u_m
        tomesd.patch.compute_merge = compute_merge
        ssa_info.original_compute_merge = old


def remove_tome(ssa_info: SSaInfo):
    if ssa_info.use_tome and ssa_info.original_compute_merge is not None:
        import tomesd
        tomesd.patch.compute_merge = ssa_info.original_compute_merge


#
# utilitys
#

def isinstance_str(x, names: str|list[str]|tuple[str]):
    if not isinstance(names, (list, tuple)): names = (names,)
    return any(map(lambda cls: cls.__name__ in names, x.__class__.__mro__))


def interpolation_mode(x: str):
    intp = None
    match x.lower():
        case 'slice':
            intp = None
        case 'nearest':
            intp = F.InterpolationMode.NEAREST
        case 'nearest_exact':
            intp = F.InterpolationMode.NEAREST_EXACT
        case 'bilinear':
            intp = F.InterpolationMode.BILINEAR
        case 'bicubic':
            intp = F.InterpolationMode.BICUBIC
        case _:
            raise ValueError(f'unknown interpolation mode: {x}')
    return intp


def is_diffusers(model_or_pipeline):
    return isinstance_str(model_or_pipeline, ('DiffusionPipeline', 'ModelMixin'))
