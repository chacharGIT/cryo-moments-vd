from dynaconf import Dynaconf

# Initialize dynaconf settings
settings = Dynaconf(
    envvar_prefix="VNN",  # Environment variables prefix
    settings_files=['parameters.yaml'],  # Configuration files
    environments=True,  # Enable layered environments
    load_dotenv=True,  # Load .env file if present
)
