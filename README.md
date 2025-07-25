# mlNBA - Análisis de Datos de la NBA

Este proyecto utiliza la NBA API para obtener y analizar datos de cualquier equipo de la NBA, mostrando los resultados de los últimos 20 partidos de la temporada más reciente disponible.

## Características

- 🏀 Obtiene automáticamente la última temporada disponible en la base de datos de la NBA
- 📊 Muestra los resultados de los últimos 20 partidos de cualquier equipo
- 📈 Proporciona estadísticas resumidas (victorias, derrotas, promedio de puntos)
- 🔄 Detección dinámica de temporadas sin necesidad de actualizar código
- 🎯 Análisis configurable para todos los equipos de la NBA

## Requisitos del Sistema

- Python 3.10 o superior
- [UV](https://docs.astral.sh/uv/) (gestor de paquetes y entornos virtuales)

## Instalación

### 1. Instalar UV (si no lo tienes)

```bash
# En Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# O usando pip
pip install uv
```

### 2. Clonar y configurar el proyecto

```bash
# Clonar el repositorio (o navegar al directorio del proyecto)
cd mlNBA

# UV automáticamente detectará el pyproject.toml y configurará el entorno
```

## Ejecución

### Opción 1: Ejecutar directamente con UV (Recomendado)

```bash
# Ejecuta el análisis del equipo configurado por defecto (Dallas Mavericks)
uv run main.py
```

### Opción 2: Personalizar el equipo analizado

Para analizar un equipo diferente, modifica la línea 123 en `main.py`:

```python
# Cambiar 'Dallas Mavericks' por el nombre completo del equipo deseado
main(ultima_temporada, 'Los Angeles Lakers')  # Lakers
main(ultima_temporada, 'Boston Celtics')     # Celtics  
main(ultima_temporada, 'Golden State Warriors') # Warriors
# etc.
```

### Opción 3: Usar el entorno virtual de UV

```bash
# Crear/activar el entorno virtual e instalar dependencias
uv sync

# Ejecutar el script
uv run python main.py
```

### Opción 4: Activar manualmente el entorno

```bash
# Sincronizar dependencias
uv sync

# Activar el entorno virtual
source .venv/bin/activate  # En Linux/macOS
# .venv\Scripts\activate   # En Windows

# Ejecutar el script
python main.py
```

## Equipos Disponibles

El script puede analizar cualquier equipo de la NBA. Algunos ejemplos de nombres válidos:

- `'Los Angeles Lakers'`
- `'Boston Celtics'`
- `'Golden State Warriors'`
- `'Miami Heat'`
- `'Chicago Bulls'`
- `'Dallas Mavericks'`
- `'Brooklyn Nets'`
- `'Phoenix Suns'`
- `'Milwaukee Bucks'`
- `'Philadelphia 76ers'`

**Nota:** Usa el nombre completo oficial del equipo tal como aparece en la NBA.

## Resultado Esperado

El script mostrará información similar a esta (ejemplo con Dallas Mavericks):

```
Verificando última temporada disponible...
==================================================
Última temporada encontrada en la base de datos: 2024-25
Número de juegos encontrados: 3514
Temporada desde: October 22, 2024
Temporada hasta: April 13, 2025

==================================================
Usando temporada: 2024-25

RESULTADOS DE LOS ÚLTIMOS 20 PARTIDOS DE DALLAS MAVERICKS (TEMPORADA 2024-25)
=====================================================================================
FECHA        | RIVAL           | RESULTADO | PUNTOS | RECORD  
-------------------------------------------------------------------------------------
APR 13, 2025 | @ MEM          | DERROTA  | 97     | 39-43   
APR 11, 2025 | vs TOR          | VICTORIA | 124    | 38-42   
...

=====================================================================================
RESUMEN DE LOS ÚLTIMOS 20 JUEGOS:
Victorias: 7
Derrotas: 13
Porcentaje de victorias: 35.0%
Promedio de puntos por juego: 112.7
Máximos puntos anotados: 133
Mínimos puntos anotados: 91
Record final de temporada: 39-43
```

## Dependencias

El proyecto utiliza las siguientes librerías principales:

- `nba-api>=1.10.0` - API oficial de la NBA para obtener datos
- `numpy>=2.2.6` - Computación numérica
- `pandas` - Manipulación y análisis de datos

## Estructura del Proyecto

```
mlNBA/
├── main.py              # Script principal
├── pyproject.toml       # Configuración del proyecto y dependencias
├── README.md           # Este archivo
├── uv.lock            # Archivo de bloqueo de dependencias
└── .venv/             # Entorno virtual (creado por UV)
```

## Funciones Principales

### `obtener_ultima_temporada()`
Detecta automáticamente la última temporada disponible en la base de datos de la NBA.

### `main(temporada=None, team_name='Los Angeles Lakers')`
Función principal que:
- Acepta cualquier nombre de equipo de la NBA como parámetro
- Obtiene los datos del equipo especificado para la temporada dada (o la más reciente)
- Muestra los resultados de los últimos 20 partidos
- Proporciona estadísticas resumidas del rendimiento del equipo

## Personalización

### Cambiar el equipo por defecto

Para cambiar el equipo que se analiza por defecto, modifica las líneas en el bloque `if __name__ == "__main__":`:

```python
# Línea aproximadamente 123-130
main(ultima_temporada, 'Tu Equipo Favorito')  # Cambiar aquí
```

### Análisis de múltiples equipos

Puedes modificar el script para analizar varios equipos en una sola ejecución:

```python
equipos = ['Los Angeles Lakers', 'Boston Celtics', 'Golden State Warriors']
for equipo in equipos:
    print(f"\n{'='*100}")
    main(ultima_temporada, equipo)
```

## Solución de Problemas

### Equipo no encontrado
Si obtienes un error de "equipo no encontrado", verifica que estés usando el nombre oficial completo del equipo. Algunos ejemplos correctos:
- ✅ `'Los Angeles Lakers'` 
- ❌ `'Lakers'`
- ✅ `'Golden State Warriors'`
- ❌ `'Warriors'`

### Error de conexión a la API
Si obtienes errores de conexión, verifica tu conexión a internet y reintenta después de unos minutos.

### Dependencias faltantes
Si UV no instala las dependencias automáticamente:
```bash
uv sync --force
```

### Python no encontrado
Asegúrate de tener Python 3.10+ instalado:
```bash
python --version
```

## Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit tus cambios (`git commit -am 'Agrega nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para detalles.