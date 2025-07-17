from dynaconf import Dynaconf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize dynaconf settings
settings = Dynaconf(
    settings_files=[os.path.join(BASE_DIR, "parameters.yaml")],
)
