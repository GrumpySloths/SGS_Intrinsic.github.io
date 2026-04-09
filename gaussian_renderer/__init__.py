from gaussian_renderer.render_r3dg import render
from gaussian_renderer.render_neilf import render_neilf
from gaussian_renderer.render_sgs import render_sgs

render_fn_dict = {"render": render, "neilf": render_neilf, "sgs": render_sgs}
