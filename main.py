#!/usr/bin/env python3
"""
🏀 NBA ANALYTICS SYSTEM - MAIN SCRIPT
=====================================

Este es el script principal del sistema de análisis de tiros de 3 puntos de la NBA.

CARACTERÍSTICAS:
• 🌐 Funciona ONLINE y OFFLINE automáticamente
• 📊 Análisis completo de tiros de 3 puntos por jugador
• 📍 Mapas de tiros con coordenadas convertidas a pies reales
• 🔥 Mapas de calor de frecuencia de tiros
• 📈 Estadísticas por zonas (Corner 3s, Above the Break, etc.)
• ✅ Línea de 3 puntos geométricamente correcta
• 🏆 Análisis de últimos 20 juegos de equipos

MODOS DE FUNCIONAMIENTO:
1. MODO ONLINE: Conecta directamente a la NBA API (requiere internet)
2. MODO OFFLINE: Usa datos descargados previamente (sin internet)

PARA USAR DATOS OFFLINE:
1. Ejecuta primero: download_all_shots.py (descarga completa de ~3-4 horas)
2. O ejecuta: test_download.py (prueba rápida con 10 jugadores)
3. Luego ejecuta este script: main.py

El sistema detecta automáticamente si hay datos offline disponibles.

EJECUCIÓN:
uv run python main.py
"""

from nba_api.stats.endpoints import teamgamelog, shotchartdetail, leaguegamefinder
from nba_api.stats.static import teams
import pandas as pd
from datetime import datetime
import matplotlib
import os

# Importar cargador de datos offline si está disponible
try:
    from nba_offline_loader import nba_offline
    OFFLINE_MODE = nba_offline is not None
    if OFFLINE_MODE:
        print("🔌 MODO OFFLINE ACTIVADO - Usando datos descargados")
except ImportError:
    OFFLINE_MODE = False
    nba_offline = None
    print("🌐 MODO ONLINE - Conectando a NBA API")

def configure_matplotlib():
    """Configura matplotlib para usar el backend más apropiado según el entorno."""
    interactive_backends = ['TkAgg', 'Qt5Agg', 'GTK3Agg', 'webagg']
    
    if os.environ.get('DISPLAY') is not None or os.name == 'nt':
        for backend in interactive_backends:
            try:
                matplotlib.use(backend)
                import matplotlib.pyplot as plt_test
                plt_test.figure()
                plt_test.close()
                print(f"✅ GUI disponible usando backend: {backend}")
                return True
            except Exception:
                continue
        
        print("⚠️  GUI no disponible - backends interactivos fallan")
        matplotlib.use('Agg')
        return False
    else:
        print("⚠️  GUI no disponible - entorno sin display")
        matplotlib.use('Agg')
        return False

def get_player_shot_data(player_id, season, player_name=None):
    """Obtiene datos de tiros de un jugador - funciona online y offline"""
    if OFFLINE_MODE and nba_offline:
        print(f"📁 Cargando datos offline para {player_name or player_id}...")
        try:
            if player_name:
                shot_data = nba_offline.get_player_shots(player_name=player_name)
            else:
                shot_data = nba_offline.get_player_shots(player_id=player_id)
            
            if not shot_data.empty:
                print(f"✅ Datos offline cargados: {len(shot_data)} tiros")
                return shot_data
            else:
                print(f"⚠️ No hay datos offline para este jugador")
                return pd.DataFrame()
        except Exception as e:
            print(f"❌ Error cargando datos offline: {e}")
    
    # Modo online
    print(f"🌐 Descargando datos online para {player_name or player_id}...")
    try:
        shot_chart = shotchartdetail.ShotChartDetail(
            team_id=0,
            player_id=player_id,
            season_nullable=season,
            context_measure_simple='FGA',
            season_type_all_star='Regular Season'
        )
        return shot_chart.get_data_frames()[0]
    except Exception as e:
        print(f"❌ Error descargando datos online: {e}")
        return pd.DataFrame()

def convertir_coordenadas_api_a_pies(loc_x, loc_y):
    """Convierte las coordenadas de la NBA API a coordenadas reales en pies."""
    factor_x = 50.0 / 500.0  # 0.1 pies por unidad API
    factor_y = 47.0 / 420.0  # ~0.112 pies por unidad API
    
    x_pies = loc_x * factor_x
    y_pies = loc_y * factor_y
    
    return x_pies, y_pies

def explicar_coordenadas_nba():
    """Explica el sistema de coordenadas de la NBA API y la conversión a distancias reales."""
    print("\n" + "="*70)
    print("📐 SISTEMA DE COORDENADAS CONVERTIDO A DISTANCIAS REALES")
    print("="*70)
    print("🏀 CONVERSIÓN IMPLEMENTADA:")
    print("• Coordenadas API → Distancias reales en pies")
    print("• X=0, Y=0: Centro de la línea de fondo")
    print("• Eje X: Distancia del centro (pies reales)")
    print("• Eje Y: Distancia desde la línea de fondo (pies reales)")
    print("• SHOT_DISTANCE: Distancia real desde el aro")
    print()
    print("🏀 REFERENCIAS DE LA CANCHA (en pies reales):")
    print("• Línea de 3 puntos (arco): 23.75 pies desde el aro")
    print("• Línea de 3 puntos (esquinas): 22.0 pies desde el aro")
    print("• Esquinas: ±22 pies del centro")
    print("• Ancho total: 50 pies")
    print("• Largo total: 94 pies")
    print("• Distancia línea de fondo al aro: ~25 pies")
    print("="*70)

def mostrar_imagen_en_vscode(ruta_imagen):
    """Intenta mostrar la imagen usando diferentes métodos disponibles en VS Code."""
    try:
        import subprocess
        import platform
        
        print("📊 GUI no disponible en matplotlib.")
        print("🔍 Buscando formas de mostrar la imagen...")
        
        if platform.system() != "Windows":
            result = subprocess.run(['code', ruta_imagen], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"🎯 Imagen abierta en VS Code: {ruta_imagen}")
                return True
        
        print(f"💾 Imagen guardada en: {ruta_imagen}")
        return False
        
    except Exception:
        print(f"💾 Imagen guardada en: {ruta_imagen}")
        return False

def obtener_ultima_temporada():
    """Determina automáticamente cuál es la última temporada disponible en la base de datos."""
    print("Verificando última temporada disponible...")
    
    current_year = datetime.now().year
    
    # Generar lista de temporadas posibles (últimos 3 años)
    temporadas_posibles = []
    for year in range(current_year, current_year - 3, -1):
        temporadas_posibles.extend([
            f"{year}-{str(year+1)[2:]}",  # Ej: 2024-25
            f"{year-1}-{str(year)[2:]}"   # Ej: 2023-24
        ])
    
    # Remover duplicados manteniendo orden
    temporadas_posibles = list(dict.fromkeys(temporadas_posibles))
    
    for temporada in temporadas_posibles:
        try:
            print(f"Probando temporada: {temporada}")
            
            gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=temporada)
            df = gamefinder.get_data_frames()[0]
            
            if len(df) > 100:  # Si hay suficientes juegos
                print("="*50)
                print(f"Última temporada encontrada en la base de datos: {temporada}")
                print(f"Número de juegos encontrados: {len(df)}")
                
                fechas = pd.to_datetime(df['GAME_DATE'])
                fecha_inicio = fechas.min().strftime('%B %d, %Y')
                fecha_fin = fechas.max().strftime('%B %d, %Y')
                
                print(f"Temporada desde: {fecha_inicio}")
                print(f"Temporada hasta: {fecha_fin}")
                
                return temporada
                
        except Exception:
            continue
    
    print("No se pudo determinar la última temporada disponible")
    return None

def player3PTS(player_id, season, player_name=None):
    """Obtiene los tiros de 3 puntos de un jugador específico en una temporada."""
    df = get_player_shot_data(player_id, season, player_name)
    
    if df.empty:
        print("❌ No se pudieron obtener datos de tiros")
        return pd.DataFrame()
    
    # Filtrar solo triples (distancia >= 22 pies)
    shot_data_3pt = df[df['SHOT_DISTANCE'] >= 22].copy()
    
    if shot_data_3pt.empty:
        print("❌ No hay datos de tiros de 3 puntos")
        return pd.DataFrame()
    
    # Convertir coordenadas si no están ya convertidas
    if 'LOC_X_PIES' not in shot_data_3pt.columns:
        coords_converted = shot_data_3pt.apply(
            lambda row: convertir_coordenadas_api_a_pies(row['LOC_X'], row['LOC_Y']), 
            axis=1, result_type='expand'
        )
        shot_data_3pt['LOC_X_PIES'] = coords_converted[0]
        shot_data_3pt['LOC_Y_PIES'] = coords_converted[1]
    
    # Clasificar por zonas de tiro
    if 'Zone' not in shot_data_3pt.columns:
        shot_data_3pt['Zone'] = shot_data_3pt.apply(
            lambda x: 'Left Corner 3' if (x['SHOT_DISTANCE'] <= 22.5 and x['LOC_X'] < -200) else
                    'Right Corner 3' if (x['SHOT_DISTANCE'] <= 22.5 and x['LOC_X'] > 200) else
                    'Above the Break 3' if (x['SHOT_DISTANCE'] > 22.5 and x['LOC_Y'] > 100) else 'Other',
            axis=1
        )
    
    # Agrupar por zona
    shot_summary = shot_data_3pt.groupby('Zone').agg({
        'SHOT_MADE_FLAG': 'sum',
        'SHOT_ATTEMPTED_FLAG': 'sum',
    }).reset_index()

    shot_summary['3P%'] = (shot_summary['SHOT_MADE_FLAG'] / shot_summary['SHOT_ATTEMPTED_FLAG'] * 100).round(1)
    shot_summary.columns = ['Zone', '3PM', '3PA', '3P%']

    print(shot_summary)

    # Visualización
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico 1: 3PM vs 3PA por zona
    x = np.arange(len(shot_summary))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, shot_summary['3PM'], width, label='3PM (Anotados)', color='green', alpha=0.7)
    bars2 = ax1.bar(x + width/2, shot_summary['3PA'], width, label='3PA (Intentados)', color='orange', alpha=0.7)
    
    ax1.set_xlabel('Zona de Tiro')
    ax1.set_ylabel('Cantidad de Tiros')
    ax1.set_title('Tiros de 3 Puntos por Zona')
    ax1.set_xticks(x)
    ax1.set_xticklabels(shot_summary['Zone'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Agregar valores en las barras
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Gráfico 2: Porcentaje de acierto por zona
    bars3 = ax2.bar(shot_summary['Zone'], shot_summary['3P%'], 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(shot_summary)])
    
    ax2.set_xlabel('Zona de Tiro')
    ax2.set_ylabel('Porcentaje de Acierto (%)')
    ax2.set_title('Eficiencia de Tiros de 3 Puntos por Zona')
    ax2.set_xticklabels(shot_summary['Zone'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(shot_summary['3P%']) * 1.1)
    
    # Agregar valores en las barras
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('player_3pt_zones_chart.png', dpi=300, bbox_inches='tight')
    print("Gráfico guardado como 'player_3pt_zones_chart.png'")
    
    mostrar_imagen_en_vscode('player_3pt_zones_chart.png')
    plt.close()
    
    shot_summary.to_csv('player_3pt_zones.csv', index=False)
    print("Datos guardados en 'player_3pt_zones.csv'")
    
    return shot_data_3pt

def shot_chart_map(player_id, season, player_name):
    """Crea un mapa completo de los tiros de 3 puntos de un jugador."""
    print(f"🏀 Generando mapa de tiros de 3 puntos para {player_name}...")
    
    df = get_player_shot_data(player_id, season, player_name)
    
    if df.empty:
        print("❌ No se pudieron obtener datos de tiros")
        return pd.DataFrame()
    
    shot_data_3pt = df[df['SHOT_DISTANCE'] >= 22].copy()
    
    if shot_data_3pt.empty:
        print("❌ No hay datos de tiros de 3 puntos")
        return pd.DataFrame()
    
    # Convertir coordenadas a pies reales
    if 'LOC_X_PIES' not in shot_data_3pt.columns:
        coordenadas_convertidas = shot_data_3pt.apply(
            lambda row: convertir_coordenadas_api_a_pies(row['LOC_X'], row['LOC_Y']), 
            axis=1
        )
        shot_data_3pt['LOC_X_PIES'] = [coord[0] for coord in coordenadas_convertidas]
        shot_data_3pt['LOC_Y_PIES'] = [coord[1] for coord in coordenadas_convertidas]
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Separar tiros anotados y fallados
    made_shots = shot_data_3pt[shot_data_3pt['SHOT_MADE_FLAG'] == 1]
    missed_shots = shot_data_3pt[shot_data_3pt['SHOT_MADE_FLAG'] == 0]
    
    # Plotear tiros
    ax.scatter(missed_shots['LOC_X_PIES'], missed_shots['LOC_Y_PIES'], 
              c='red', alpha=0.6, s=30, label=f'Fallados ({len(missed_shots)})', 
              marker='x')
    
    ax.scatter(made_shots['LOC_X_PIES'], made_shots['LOC_Y_PIES'], 
              c='green', alpha=0.8, s=40, label=f'Anotados ({len(made_shots)})', 
              marker='o')
    
    # Dibujo NBA: línea de 3 puntos reglamentaria con esquinas
    radio_3pt = 23.75  # Radio oficial NBA: 23.75 pies desde el centro del aro
    corner_x = 22.0    # Posición de las esquinas en el eje X: 22 pies
    aro_y = 0.0        # Centro del aro en la línea de fondo
    
    # Calcular el punto de intersección donde el arco se une con las líneas de esquina
    y_interseccion = np.sqrt(radio_3pt**2 - corner_x**2)
    angulo_interseccion = np.arccos(corner_x/radio_3pt)  # en radianes
    
    # Dibujar el arco principal (desde la intersección izquierda hasta la derecha)
    theta = np.linspace(np.pi - angulo_interseccion, angulo_interseccion, 200)
    x_arc = radio_3pt * np.cos(theta)
    y_arc = radio_3pt * np.sin(theta) + aro_y
    ax.plot(x_arc, y_arc, 'black', linewidth=2.5, alpha=0.9, label='Línea de 3 puntos')
    
    # Dibujar las líneas rectas de las esquinas
    ax.plot([-corner_x, -corner_x], [0, y_interseccion], 'black', linewidth=2.5, alpha=0.9)
    ax.plot([corner_x, corner_x], [0, y_interseccion], 'black', linewidth=2.5, alpha=0.9)
    
    # Configurar gráfico
    ax.set_xlabel('Posición X (pies reales desde el centro)', fontsize=12)
    ax.set_ylabel('Posición Y (pies reales desde línea de fondo)', fontsize=12)
    ax.set_title(f'Mapa de Tiros de 3 Puntos - {player_name} ({season})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-25, 25)
    ax.set_ylim(0, 35)
    
    plt.tight_layout()
    
    filename = f'shot_chart_map_{player_name.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📍 Mapa guardado como '{filename}'")
    
    # Estadísticas
    total_shots = len(shot_data_3pt)
    made_shots_count = len(made_shots)
    missed_shots_count = len(missed_shots)
    percentage = (made_shots_count / total_shots * 100) if total_shots > 0 else 0
    
    print(f"\n📊 ESTADÍSTICAS DEL MAPA:")
    print(f"Total de tiros de 3: {total_shots}")
    print(f"Tiros anotados: {made_shots_count}")
    print(f"Tiros fallados: {missed_shots_count}")
    print(f"Porcentaje de acierto: {percentage:.1f}%")
    
    mostrar_imagen_en_vscode(filename)
    plt.close()
    
    return shot_data_3pt

def shot_heatmap(player_id, season, player_name):
    """Crea un mapa de calor de los tiros de 3 puntos de un jugador."""
    print(f"🔥 Generando mapa de calor para {player_name}...")
    
    df = get_player_shot_data(player_id, season, player_name)
    
    if df.empty:
        print("❌ No se pudieron obtener datos de tiros")
        return pd.DataFrame()
    
    shot_data_3pt = df[df['SHOT_DISTANCE'] >= 22].copy()
    
    if shot_data_3pt.empty:
        print("❌ No hay datos de tiros de 3 puntos")
        return pd.DataFrame()
    
    # Convertir coordenadas a pies reales
    if 'LOC_X_PIES' not in shot_data_3pt.columns:
        coordenadas_convertidas = shot_data_3pt.apply(
            lambda row: convertir_coordenadas_api_a_pies(row['LOC_X'], row['LOC_Y']), 
            axis=1
        )
        shot_data_3pt['LOC_X_PIES'] = [coord[0] for coord in coordenadas_convertidas]
        shot_data_3pt['LOC_Y_PIES'] = [coord[1] for coord in coordenadas_convertidas]
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Crear bins para el mapa de calor
    x_bins = np.linspace(-25, 25, 25)
    y_bins = np.linspace(0, 35, 20)
    
    # Crear histograma 2D
    hist, x_edges, y_edges = np.histogram2d(
        shot_data_3pt['LOC_X_PIES'], shot_data_3pt['LOC_Y_PIES'], 
        bins=[x_bins, y_bins]
    )
    
    # Crear mapa de calor
    im = ax.imshow(hist.T, origin='lower', aspect='auto', 
                   extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
                   cmap='hot', alpha=0.8)
    
    # Dibujo NBA: línea de 3 puntos reglamentaria con esquinas
    radio_3pt = 23.75  # Radio oficial NBA: 23.75 pies desde el centro del aro
    corner_x = 22.0    # Posición de las esquinas en el eje X: 22 pies
    aro_y = 0.0        # Centro del aro en la línea de fondo
    
    # Calcular el punto de intersección donde el arco se une con las líneas de esquina
    y_interseccion = np.sqrt(radio_3pt**2 - corner_x**2)
    angulo_interseccion = np.arccos(corner_x/radio_3pt)  # en radianes
    
    # Dibujar el arco principal (desde la intersección izquierda hasta la derecha)
    theta = np.linspace(np.pi - angulo_interseccion, angulo_interseccion, 200)
    x_arc = radio_3pt * np.cos(theta)
    y_arc = radio_3pt * np.sin(theta) + aro_y
    ax.plot(x_arc, y_arc, 'white', linewidth=3.5, alpha=1, label='Línea de 3 puntos')
    
    # Dibujar las líneas rectas de las esquinas
    ax.plot([-corner_x, -corner_x], [0, y_interseccion], 'white', linewidth=3.5, alpha=1)
    ax.plot([corner_x, corner_x], [0, y_interseccion], 'white', linewidth=3.5, alpha=1)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Frecuencia de Tiros', rotation=270, labelpad=20)
    
    ax.set_xlabel('Posición X (pies reales desde el centro)', fontsize=12)
    ax.set_ylabel('Posición Y (pies reales desde línea de fondo)', fontsize=12)
    ax.set_title(f'Mapa de Calor - Tiros de 3 Puntos - {player_name} ({season})', 
                fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    
    filename = f'shot_heatmap_{player_name.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"🔥 Mapa de calor guardado como '{filename}'")
    
    mostrar_imagen_en_vscode(filename)
    plt.close()
    
    return shot_data_3pt

def analizar_equipo_mavericks(ultima_temporada):
    """Analiza los últimos 20 juegos de Dallas Mavericks."""
    print("\n" + "="*50)
    print(f"🏀 ANÁLISIS DALLAS MAVERICKS - TEMPORADA {ultima_temporada}")
    print("="*50)
    
    # Obtener información del equipo
    teams_info = teams.get_teams()
    mavericks = next((team for team in teams_info if team['full_name'] == 'Dallas Mavericks'), None)
    
    if not mavericks:
        print("No se encontró información de Dallas Mavericks")
        return
    
    team_id = mavericks['id']
    
    try:
        # Obtener los últimos 20 juegos
        gamelog = teamgamelog.TeamGameLog(team_id=team_id, season=ultima_temporada)
        games_df = gamelog.get_data_frames()[0]
        
        if games_df.empty:
            print("No se encontraron juegos para esta temporada")
            return
        
        last_20_games = games_df.head(20)
        
        print(f"\nRESULTADOS DE LOS ÚLTIMOS 20 PARTIDOS DE DALLAS MAVERICKS")
        print("=" * 85)
        print("FECHA        | RIVAL           | RESULTADO | PUNTOS | RECORD  ")
        print("-" * 85)
        
        victorias = 0
        derrotas = 0
        total_puntos = 0
        puntos_maximos = 0
        puntos_minimos = float('inf')
        
        for _, game in last_20_games.iterrows():
            if game['WL'] == 'W':
                resultado = "VICTORIA"
                victorias += 1
            else:
                resultado = "DERROTA"
                derrotas += 1
            
            rival = game['MATCHUP'].split()[-1]
            if '@' in game['MATCHUP']:
                rival = f"@ {rival}"
            else:
                rival = f"vs {rival}"
            
            puntos = game['PTS']
            total_puntos += puntos
            puntos_maximos = max(puntos_maximos, puntos)
            puntos_minimos = min(puntos_minimos, puntos)
            
            fecha = pd.to_datetime(game['GAME_DATE']).strftime('%b %d, %Y').upper()
            
            print(f"{fecha} | {rival:<15} | {resultado:<8} | {puntos:<6} | {game['W']}-{game['L']:<6}")
        
        # Mostrar resumen
        print("\n" + "=" * 85)
        print("📊 RESUMEN DE LOS ÚLTIMOS 20 JUEGOS:")
        print(f"Victorias: {victorias}")
        print(f"Derrotas: {derrotas}")
        print(f"Porcentaje de victorias: {(victorias/20)*100:.1f}%")
        print(f"Promedio de puntos por juego: {total_puntos/20:.1f}")
        print(f"Máximos puntos anotados: {puntos_maximos}")
        print(f"Mínimos puntos anotados: {puntos_minimos}")
        print(f"Record final de temporada: {games_df.iloc[0]['W']}-{games_df.iloc[0]['L']}")
        
    except Exception as e:
        print(f"❌ Error analizando Mavericks: {e}")

def main():
    """Función principal que ejecuta el análisis completo."""
    print("🏀 NBA ANALYTICS SYSTEM")
    print("=" * 50)
    
    # Configurar matplotlib
    configure_matplotlib()
    
    # Obtener la última temporada disponible
    ultima_temporada = obtener_ultima_temporada()
    
    if not ultima_temporada:
        print("No se pudo obtener información de temporadas. Saliendo...")
        return
    
    # Analizar Dallas Mavericks
    analizar_equipo_mavericks(ultima_temporada)
    
    # Análisis de tiros de 3 puntos
    print("\n" + "="*60)
    print("🏀 ANÁLISIS DE TIROS DE 3 PUNTOS - LeBron James")
    print("="*60)
    
    explicar_coordenadas_nba()
    
    # 1. Estadísticas por zonas
    print("\n1️⃣ Estadísticas por zonas:")
    player3PTS(2544, ultima_temporada, "LeBron James")
    
    # 2. Mapa de tiros
    print("\n2️⃣ Mapa de ubicaciones de tiros:")
    shot_chart_map(2544, ultima_temporada, "LeBron James")
    
    # 3. Mapa de calor
    print("\n3️⃣ Mapa de calor de frecuencia:")
    shot_heatmap(2544, ultima_temporada, "LeBron James")
    
    # Información adicional del modo offline
    if OFFLINE_MODE and nba_offline:
        print("\n" + "="*60)
        print("📁 INFORMACIÓN DEL MODO OFFLINE")
        print("="*60)
        try:
            available_players = nba_offline.list_available_players()
            print(f"👥 Jugadores disponibles offline: {len(available_players)}")
            print("\n🏆 Algunos jugadores disponibles:")
            for _, player in available_players.head(10).iterrows():
                print(f"  • {player['PLAYER_NAME']} ({player['TEAM']})")
            print("  ...")
            print(f"\n💡 Para analizar otro jugador, modifica el ID y nombre en main()")
        except Exception as e:
            print(f"❌ Error listando jugadores offline: {e}")
    
    print("\n" + "="*60)
    print("✅ ANÁLISIS COMPLETADO")
    print("="*60)

if __name__ == "__main__":
    main()
