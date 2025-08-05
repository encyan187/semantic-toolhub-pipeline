import ast
from collections.abc import Iterable


def parse_list(x):

    if isinstance(x, str):
        s = x.strip()
        if s.startswith('[') and s.endswith(']'):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [p.strip() for p in parsed if isinstance(p, str)]
            except Exception:
                inner = s[1:-1]
                return [p.strip() for p in inner.split(',') if p.strip()]
        return [s.strip()]
        # if it's already a list of labels
    if isinstance(x, list):
        return [p.strip() for p in x if isinstance(p, str)]
    return [str(x)]