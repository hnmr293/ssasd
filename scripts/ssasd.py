import pprint

import gradio as gr
import modules.scripts as scripts
import modules.shared as shared
import modules.sd_models as sd_models
from modules.processing import Processed, StableDiffusionProcessing

import extensions.ssasd as ssasd

NAME = 'SSaSD'

class Script(scripts.Script):

    def title(self):
        return 'SSa'
    
    def show(self, is_img2img: bool):
        return scripts.AlwaysVisible
    
    def ui(self, is_img2img: bool):
        with gr.Accordion(NAME, open=False, elem_classes=[NAME]):
            enabled = gr.Checkbox(value=False, label='Enable SSa')
            max_downsample = gr.Slider(minimum=1, maximum=8, value=2, label='Max downsample', info=(
'''
value: target layers
0: (none)
1: down_blocks.0.attentions.*
   up_blocks.3.attentions.*
2: 1
   + down_blocks.1.attentions.*
   + up_blocks.2.attentions.*
4: 2
   + down_blocks.2.attentions.*
   + up_blocks.1.attentions.*
8: 4
   + mid_block.attentions.0.*
'''.strip()), elem_classes=[f'{NAME}_max_downsample'])
            scale = gr.Slider(minimum=2, maximum=16, value=2, step=1, label='Scale')
            end_step_ratio = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label='End step (0.0: disable, 1.0: whole steps)')
            intp = gr.Radio(choices=['slice', 'nearest', 'nearest_exact', 'bilinear', 'bicubic'], value='bilinear', label='Interpolation mode')
            #
            #start_step = gr.Slider(minimum=0, maximum=100, value=0, label='start step')
            #end_step = gr.Slider(minimum=-1, maximum=100, value=-1, label='end step (exclusive)')
        
        return [
            enabled,
            max_downsample,
            scale,
            end_step_ratio,
            intp,
        ]
    
    def process(
            self,
            p: StableDiffusionProcessing,
            enabled: bool,
            max_downsample: float|int,
            scale: float|int,
            end_step_ratio: float|int,
            intp: str,
    ):
        model = p.sd_model
        ssasd.remove_patch(model)

        if not enabled:
            return
        
        max_downsample = int(max_downsample)
        scale = int(scale)
        end_step_ratio = float(end_step_ratio)
        
        tome_ratio = p.get_token_merging_ratio()
        
        _, ssa_info = ssasd.apply_patch(model, max_downsample, scale, end_step_ratio, intp,
                                        total_steps=p.steps,
        )
        
        args = {
            f'{NAME} enabled': enabled,
            f'{NAME} max_downsample': ssa_info.max_downsample,
            f'{NAME} scale': ssa_info.scale,
            f'{NAME} steps': str(ssa_info.steps),
            f'{NAME} interpolation': intp,
        }
        p.extra_generation_params.update(args)
        pprint.pprint(args)

        ssa_info.use_tome = (
            model.model.diffusion_model._ssa_info.use_tome
            and 0 < tome_ratio
        )
        print(f'[{NAME}] ToMe {["disabled", "enabled"][ssa_info.use_tome]}')
    
    def postprocess(self, p, processed, enabled: bool, *args):
        if enabled:
            ssasd.remove_patch(p.sd_model)


from scripts.ssasdlib.xyz import init_xyz
init_xyz(Script, NAME)
