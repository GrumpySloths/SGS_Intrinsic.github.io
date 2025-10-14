# from gaussian_renderer.render_origin import render
from gaussian_renderer.render_modify import render_view 
from gaussian_renderer.render_r3dg import render 
from gaussian_renderer.render_neilf import render_neilf
# from gaussian_renderer.render_gsid import render_gsid
from gaussian_renderer.render_gsid import render_gsid
render_fn_dict = {
    "render": render,
    "render_modify": render_view,
    "neilf": render_neilf,
    "gsid": render_gsid
}