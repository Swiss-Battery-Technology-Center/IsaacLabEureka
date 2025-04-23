# isaaclab_eureka/success_metric/__init__.py

import importlib

def load_success_metric(rl_task_type: str):
    module_path = f"isaaclab_eureka.success_metric.{rl_task_type}"
    mod = importlib.import_module(module_path)

    if not hasattr(mod, "compute_success_metric"):
        raise AttributeError(f"Module '{module_path}' does not define compute_success_metric()")
    
    return mod.compute_success_metric

# write your task_specific success metric in this folder
# filename should be equivalent to sbtc_tasks folder name
# sbtc_lift -> lift.py, sbtc_unscrew -> unscrew.py, sbtc_plug -> plug.py, etc
# it will be dynamically attached to 'env', so 'self' in the function will be 'env'
# return a dict with the key 'success_metric' and a tensor of shape (num_envs, 1)
# additionally, you can return other keys for debuggig purposes(to see if your definition of success metric is valid), which will be logged in tensorboard but not given to llm
#    return {
#         "success_metric": success.float(),
#         "is_close": is_close.float(),
#         "is_lifted": lifted.float(),
#     }