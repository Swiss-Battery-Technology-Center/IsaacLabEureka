import ast
import os

def load_functions_from_file(filepath):
    """Load all top-level functions from a Python file."""
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)
    functions = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            name = node.name
            code = ast.get_source_segment(source, node)
            functions[name] = code
    return functions

def find_called_functions(source_code: str):
    """Extract calls like sbtc_utils.xyz() from the function source."""
    tree = ast.parse(source_code)
    called_funcs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                module_name = func.value.id
                func_name = func.attr
                if module_name not in ("env", "robot", "object", "self", "contact_sensor"):
                    called_funcs.append((module_name, func_name))
    return called_funcs

def extract_function_recursively_static(func_name, all_modules_map, visited=None):
    """Static version: recursively extract source from static maps."""
    if visited is None:
        visited = set()
    if func_name in visited:
        return ""
    visited.add(func_name)

    output = ""
    for module_name, func_map in all_modules_map.items():
        if func_name in func_map:
            code = func_map[func_name]
            output += f"# === {module_name}.{func_name} ===\n{code}\n"
            for mod, fname in find_called_functions(code):
                if fname not in visited:
                    output += extract_function_recursively_static(fname, all_modules_map, visited)
            break
    return output

def extract_func_sources_from_cfg_source_static(cfg_path: str, mdp_dirs: list[str]) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        source_code = f.read()
    tree = ast.parse(source_code)

    grouped_funcs = {
        "rewards": set(),
        "events": set(),
        "curriculum": set(),
        "terminations": set(),
        "observations": set(),
    }

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_name = node.name.lower()
            if "reward" in class_name:
                group = "rewards"
            elif "event" in class_name:
                group = "events"
            elif "curriculum" in class_name:
                group = "curriculum"
            elif "observation" in class_name:
                group = "observations"
            elif "termination" in class_name or "done" in class_name:
                group = "terminations"
            else:
                continue

            for class_node in ast.walk(node):
                if isinstance(class_node, ast.keyword) and class_node.arg == "func":
                    value = class_node.value
                    if (
                        isinstance(value, ast.Attribute)
                        and isinstance(value.value, ast.Name)
                        and value.value.id == "mdp"
                    ):
                        grouped_funcs[group].add(value.attr)

    # Load all mdp function maps
    all_modules_map = {}
    for mdp_dir in mdp_dirs:
        for filename in os.listdir(mdp_dir):
            if filename.endswith(".py"):
                filepath = os.path.join(mdp_dir, filename)
                module_key = filename.replace(".py", "")
                all_modules_map[module_key] = load_functions_from_file(filepath)

    # Resolve all relevant functions
    visited = set()
    grouped_sources = {k: {} for k in grouped_funcs}
    for group, func_names in grouped_funcs.items():
        for func_name in sorted(func_names):
            code = extract_function_recursively_static(func_name, all_modules_map, visited)
            grouped_sources[group][func_name] = code if code else f"# Failed to statically extract {func_name}\n"
    return grouped_sources
