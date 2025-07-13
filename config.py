from dynaconf import Dynaconf

# Initialize dynaconf settings
settings = Dynaconf(
    settings_files=['./parameters.yaml'],  # Configuration files
)
