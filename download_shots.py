#!/usr/bin/env python3
"""
⬇️ WRAPPER PARA DESCARGA DE SHOTS
=================================

Script wrapper que ejecuta el descargador de shots desde la raíz del proyecto,
asegurando que las rutas sean correctas.
"""

import sys
import os

# Asegurar que estamos en la raíz del proyecto
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir.endswith('/src'):
    os.chdir(os.path.dirname(script_dir))
elif not os.path.exists('src'):
    print("❌ Error: Ejecutar desde la raíz del proyecto mlNBA")
    sys.exit(1)

# Agregar src al path
sys.path.insert(0, 'src')

# Importar y ejecutar
try:
    from shots_downloader import main
    main()
except ImportError as e:
    print(f"❌ Error importando módulo: {e}")
    print("💡 Asegúrate de ejecutar desde la raíz del proyecto")
    sys.exit(1)
