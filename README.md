# 🏀 NBA 3-Point Analytics System

Sistema completo de análisis de tiros de 3 puntos de la NBA con capacidades online y offline.

## 🎯 Características Principales

- **🌐 Modo Híbrido**: Funciona online (NBA API) y offline (datos descargados)
- **📊 Análisis Completo**: Estadísticas por zonas, mapas de tiros, mapas de calor
- **📍 Coordenadas Reales**: Conversión automática a pies reales de cancha NBA
- **✅ Geometría Correcta**: Línea de 3 puntos dibujada según datos reales
- **🎮 Datos Individuales**: Cada tiro con fecha, coordenadas, resultado, etc.

## 📁 Archivos del Sistema

### Scripts Principales
- **`main.py`** - Script principal de análisis (funciona online/offline)
- **`download_all_shots.py`** - Descarga completa de datos (~3-4 horas)
- **`download_test.py`** - Prueba rápida con 10 jugadores (~3 minutos)

### Archivos Generados
- **`nba_offline_loader.py`** - Módulo de carga offline (generado automáticamente)
- **`all_shots_3pt_2024_25_COMPLETO.csv`** - Base de datos offline completa

## 🚀 Cómo Usar

### Opción 1: Uso Rápido (Online)
```bash
uv run python main.py
```
Requiere conexión a internet. Analiza directamente desde la NBA API.

### Opción 2: Descarga Completa (Offline)
```bash
# 1. Descargar todos los datos (3-4 horas, una sola vez)
uv run python download_all_shots.py

# 2. Usar sin internet
uv run python main.py
```

### Opción 3: Prueba Rápida
```bash
# Probar con solo 10 jugadores (3 minutos)
uv run python download_test.py
```

## 📊 Análisis Incluidos

1. **📈 Estadísticas por Zonas**
   - Corner 3s (izquierda/derecha)
   - Above the Break 3s
   - Porcentajes de acierto por zona

2. **📍 Mapa de Tiros**
   - Ubicación exacta de cada tiro
   - Tiros anotados vs fallados
   - Línea de 3 puntos geométricamente correcta

3. **🔥 Mapa de Calor**
   - Frecuencia de tiros por zona
   - Identificación de zonas preferidas

## 🎮 Datos Incluidos

- **~92,000+ tiros individuales** de 3 puntos
- **~480 jugadores activos** (temporada 2024-25)
- **Coordenadas convertidas** a pies reales
- **Clasificación por zonas** automática
- **Información completa**: fecha, equipo, resultado, distancia

## 🔧 Cambiar Jugador Analizado

Para analizar otro jugador, edita `main.py` líneas 610-620:

```python
# Cambiar estas líneas:
player3PTS(2544, ultima_temporada, "LeBron James")
shot_chart_map(2544, ultima_temporada, "LeBron James") 
shot_heatmap(2544, ultima_temporada, "LeBron James")

# Por ejemplo, para Stephen Curry:
player3PTS(201939, ultima_temporada, "Stephen Curry")
shot_chart_map(201939, ultima_temporada, "Stephen Curry")
shot_heatmap(201939, ultima_temporada, "Stephen Curry")
```

Los IDs de jugadores están en `all_shots_3pt_2024_25_COMPLETO.csv`.

## 📋 Requisitos

- Python 3.8+
- uv (gestor de paquetes)
- Dependencias: pandas, matplotlib, nba_api

## 🏆 Jugadores Disponibles

El sistema incluye datos de todos los jugadores activos con >100 minutos jugados en la temporada 2024-25, incluyendo:

- LeBron James, Stephen Curry, Luka Dončić
- Jayson Tatum, Damian Lillard, Anthony Edwards
- Y muchos más...

## 📈 Características Técnicas

- **Rate Limiting**: Respeta límites de la NBA API
- **Reintentos Automáticos**: Manejo robusto de errores de red
- **Conversión de Coordenadas**: Matemáticamente precisa
- **Línea de 3 Puntos**: Basada en análisis de datos reales
- **Guardado Automático**: Imágenes en alta resolución

---

🏀 **¡Disfruta analizando los tiros de 3 puntos de la NBA!** 🏀
