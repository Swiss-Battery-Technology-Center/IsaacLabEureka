def main():
    import os
    import traceback
    from extraction_functions import extract_func_sources_from_cfg_source_static

    # === Paths ===
    source_code_path = "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sbtc/manager_based/sbtc_unscrew/sbtc_unscrew_env_cfg.py"
    mdp_dirs = [
        "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sbtc/manager_based/mdp",
        "/workspace/isaaclab/source/isaaclab/isaaclab/envs/mdp",
    ]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "texts", "full_context.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # === Load env cfg source code ===
        with open(source_code_path, "r", encoding="utf-8") as f:
            cfg_source = f.read()

        # === Extract grouped function definitions ===
        func_sources = extract_func_sources_from_cfg_source_static(
            cfg_path=source_code_path,
            mdp_dirs=mdp_dirs,
        )

        # === Compose full context string ===
        all_text = "##### === SBTC ENV CONFIG === #####\n\n"
        all_text += cfg_source + "\n"

        for group, funcs in func_sources.items():
            all_text += f"\n\n##### === {group.upper()} FUNCTIONS === #####\n"
            for name, src in funcs.items():
                all_text += f"\n=== {name} ===\n{src}\n"

        # === Save to file ===
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(all_text)
        print(f"✅ Saved full context to: {output_path}")

    except Exception:
        print("❌ Failed during extraction or writing.")
        traceback.print_exc()


if __name__ == "__main__":
    main()
