from packaging.version import Version
from .__version__ import __version__

import mace
assert Version(mace.__version__) >= Version("0.3.14"), "requires mace-torch >=0.3.14 (validated on 0.3.14 and 0.3.16)."

import graph_longrange
assert Version(graph_longrange.__version__) >= Version("0.3.0"), "please update your graph_longrange version to >= 0.3.0"