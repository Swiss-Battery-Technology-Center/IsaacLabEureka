# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from collections import defaultdict

import GPUtil
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_logs(path: str):
    """Load tensorboard logs from a given path.

    Args:
        path: The path to the tensorboard logs.

    Returns:
        A dictionary with the tags and their respective values.
    """
    data = defaultdict(list)
    event_acc = EventAccumulator(path)
    event_acc.Reload()  # Load all data written so far

    for tag in event_acc.Tags()["scalars"]:
        events = event_acc.Scalars(tag)
        for event in events:
            data[tag].append(event.value)

    return data


def get_freest_gpu():
    """Get the GPU with the most free memory."""
    gpus = GPUtil.getGPUs()
    if not gpus:
        return None
    # Sort GPUs by memory usage
    gpus.sort(key=lambda gpu: gpu.memoryUsed)
    return gpus[0].id


class MuteOutput:
    """Context manager to mute stdout and stderr."""

    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115
        sys.stderr = open(os.devnull, "w")  # noqa: SIM115
        return self

    def __exit__(self, *args):
        sys.stdout = self._stdout
        sys.stderr = self._stderr

import ast
import inspect
import importlib


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


def load_functions_from_file(filepath):
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


def is_builtin_or_external(func_obj):
    """Skip torch/builtins/external C-extension functions."""
    try:
        mod = inspect.getmodule(func_obj)
        return mod is None or mod.__name__.startswith("torch") or mod.__name__ == "builtins"
    except Exception:
        return True


def extract_function_recursively(func_obj, visited=None, sbtc_utils_map=None):
    if visited is None:
        visited = set()
    if sbtc_utils_map is None:
        sbtc_utils_map = {}

    try:
        func_name = func_obj.__name__
        func_module = func_obj.__module__
    except Exception:
        return ""

    full_name = f"{func_module}.{func_name}"
    if full_name in visited or is_builtin_or_external(func_obj):
        return ""
    visited.add(full_name)

    try:
        source = inspect.getsource(func_obj)
    except Exception:
        return ""

    output = f"# === {full_name} ===\n{source}\n"

    for mod, fname in find_called_functions(source):
        if mod == "sbtc_utils" and fname in sbtc_utils_map:
            output += f"# === sbtc_utils.{fname} ===\n{sbtc_utils_map[fname]}\n"
        else:
            try:
                submod = importlib.import_module(mod)
                subfunc = getattr(submod, fname)
                output += extract_function_recursively(subfunc, visited, sbtc_utils_map)
            except Exception:
                continue

    return output


def extract_func_sources_from_cfg_source(cfg_path: str, mdp_module_path: str) -> dict:
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

    cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
    mdp_dir = os.path.abspath(os.path.join(cfg_dir, "..", "mdp"))
    sbtc_utils_path = os.path.join(mdp_dir, "utils.py")
    sbtc_utils_map = load_functions_from_file(sbtc_utils_path)

    try:
        sys.path.append("/workspace/isaaclab/source")
        mdp_module = importlib.import_module(mdp_module_path)
    except ImportError as e:
        raise ImportError(f"Failed to import mdp module '{mdp_module_path}': {e}")

    visited = set()
    grouped_sources = {k: {} for k in grouped_funcs}

    for group, func_names in grouped_funcs.items():
        for func_name in sorted(func_names):
            try:
                func_obj = getattr(mdp_module, func_name)
                src = extract_function_recursively(func_obj, visited, sbtc_utils_map)
                grouped_sources[group][func_name] = src
            except Exception as e:
                grouped_sources[group][func_name] = f"# Failed to extract {func_name}: {e}\n"

    return grouped_sources

class WrongStringFormatException(Exception):
    """Raised when the LLM-generated string is malformed or missing required keys."""
    pass

from enum import Enum

class TrainingStatus(Enum):
    SUCCESS = "success"
    CRASH = "crash"
    FORMAT_ERROR = "format_error"
    SKIPPED = "skipped"

def get_curriculum_term_cfg(manager, term_name: str):
    if term_name not in manager._term_names:
        raise ValueError(f"Curriculum term '{term_name}' not found.")
    return manager._term_cfgs[manager._term_names.index(term_name)]

def set_curriculum_term_cfg(manager, term_name: str, cfg):
    if term_name not in manager._term_names:
        raise ValueError(f"Curriculum term '{term_name}' not found.")
    manager._term_cfgs[manager._term_names.index(term_name)] = cfg
