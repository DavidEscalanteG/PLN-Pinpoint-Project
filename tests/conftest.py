"""Configuración común de pytest: asegura que el paquete sea importable desde la raíz."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
