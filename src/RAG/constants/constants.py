from pathlib import Path

_PACKAGE_NAME = "RAG"
_ROOT_DIR = Path(__file__).parent.parent.parent.parent
_CONFIG_DIR = _ROOT_DIR / "config"
_CONFIG_FILE_NAME = "config.yaml"
_PARAMS_FILE_NAME = "params.yaml"
CONFIG_FILE_PATH = _CONFIG_DIR / _CONFIG_FILE_NAME
PARAMS_FILE_PATH = _CONFIG_DIR / _PARAMS_FILE_NAME