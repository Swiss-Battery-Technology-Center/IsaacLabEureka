def main():    
    from isaaclab.app import AppLauncher
    from isaaclab_eureka.utils import get_freest_gpu
    import copy

    _device = 'cuda'
    _env_type = 'manager_based'
    _task = "SBTC-Unscrew-Franka-OSC-v0"
    _num_envs = 256
    _env_seed = 42

    if _device == "cuda":
        device_id = get_freest_gpu()
        _device = f"cuda:{device_id}"
    app_launcher = AppLauncher(headless=True, device=_device)
    _simulation_app = app_launcher.app

    import isaaclab_tasks  # noqa: F401
    from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
    from isaaclab_tasks.utils import parse_env_cfg
    import gymnasium as gym

    # Load fresh env config from registry (avoids curriculum side effects)
    if _env_type == "manager_based":
        env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(_task)
    else:
        env_cfg: DirectRLEnvCfg = parse_env_cfg(_task)

    env_cfg.sim.device = _device
    env_cfg.seed = _env_seed
    env_cfg.scene.num_envs = _num_envs  # ensure consistency
    _original_env_cfg = copy.deepcopy(env_cfg)
    # Create new env
    _env = gym.make(_task, cfg=env_cfg)
    print("ENV CREATED")

    ## my code for testing...
    try:
        from extraction_functions import extract_environment_learning_logic
        learning_info = extract_environment_learning_logic(_env.unwrapped)

        for mgr, terms in learning_info.items():
            print(f"=== {mgr} ===")
            for name, meta in terms.items():
                print(f"\nTerm: {name}")
                print(f"  Function: {meta['function']}")
                print(f"  Module: {meta['module']}")
                print(f"  Params: {meta['params']}")
                print(f"  Weight: {meta.get('weight', 'N/A')}")
                print(f"  Source preview:\n{meta['source'][:300]}\n")
    except Exception as e:
        print(f"Error extracting learning logic: {e}")

    ## close env and sim app
    _env.close()
    _simulation_app.close()


if __name__ == "__main__":
    main()
