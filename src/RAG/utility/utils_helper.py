import re

def resolve_config(content: dict):
    '''
    This function resolves the placeholders in the configuration dictionary.
    Placeholders are in the format ${key1.key2} and will be replaced with the
    corresponding value from the config dictionary.
    Args:
        config (dict): The configuration dictionary with potential placeholders.
    Errors:
        Exception: Raises an exception if a placeholder cannot be resolved.
    Returns:
        dict: The configuration dictionary with placeholders resolved.
        
    '''
    pattern = re.compile(r"\$\{([^}^{]+)\}")

    def replacer(match):
        try:
            keys = match.group(1).split(".")
            value = content
            for k in keys:
                value = value[k]
            return str(value)
        except Exception as e:
            raise e

    def walk(d):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v)
            elif isinstance(v, str):
                d[k] = pattern.sub(replacer, v)

    walk(content)
    return content