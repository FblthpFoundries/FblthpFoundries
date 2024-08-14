proxyshop_path = "C:\\Users\\Sam\\Documents\\Proxyshop"

import importlib.util
import sys
spec = importlib.util.spec_from_file_location("proxyshop", proxyshop_path)
foo = importlib.util.module_from_spec(spec)
sys.modules["proxyshop"] = foo
spec.loader.exec_module(foo)

