# A file to load the config from the config.yaml file

import yaml
import os

class NoneConfig:
    """Returned for missing keys; any attribute access returns self (falsy, so 'x or default' works)."""

    def __getattr__(self, name: str):
        return self

    def __bool__(self):
        return False

class Config:
    """Recursive config: nested dicts become Config, missing attributes return None."""
    def __init__(self, data: dict):
        for key, value in data.items():
            setattr(self, key, Config(value) if isinstance(value, dict) else value)

    def __getattr__(self, name: str):
        return NoneConfig()

data = {}
for path in ["config.yaml", "../config.yaml"]:
    if os.path.exists(path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        # Normalise CWD to the directory that contains config.yaml so that all
        # relative paths in the config (e.g. "models/…") resolve correctly
        # regardless of which sub-directory the process was launched from.
        os.chdir(os.path.dirname(os.path.abspath(path)))
        break

config = Config(data)
