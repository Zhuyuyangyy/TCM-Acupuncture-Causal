from dataclasses import dataclass

@dataclass
class Settings:
    app_name: str = "TCM-Acupuncture-Causal"
    version: str = "0.1.0"
    port: int = 8021

settings = Settings()
