#!/usr/bin/env python3
"""
🧪 NBA DATA DOWNLOADER - TEST DE 10 JUGADORES
=============================================

Este script prueba la descarga de datos de tiros de 3 puntos de 10 jugadores
para verificar que el sistema funciona correctamente antes de hacer la descarga completa.

CARACTERÍSTICAS:
• 👥 Descarga datos de 10 jugadores populares
• ⚡ Rápido (5-10 minutos)
• 🧪 Perfecto para pruebas
• 📊 Crea archivo test para verificar funcionamiento
• 🔄 Usa rate limiting para respetar la API

EJECUCIÓN:
uv run python test_download.py

RESULTADO: all_shots_3pt_2024_25_TEST.csv + nba_offline_loader.py
"""

import pandas as pd
import time
import os
import random
from datetime import datetime
from nba_api.stats.endpoints import shotchartdetail, commonallplayers

def convertir_coordenadas_api_a_pies(loc_x, loc_y):
    """
    Convierte las coordenadas de la NBA API a coordenadas reales en pies.
    """
    factor_x = 50.0 / 500.0  # 0.1 pies por unidad API
    factor_y = 47.0 / 420.0  # ~0.112 pies por unidad API
    
    x_pies = loc_x * factor_x
    y_pies = loc_y * factor_y
    
    return x_pies, y_pies

def clasificar_zona_tiro(shot_distance, loc_x_pies, loc_y_pies):
    """
    Clasifica el tiro en una zona específica basado en coordenadas reales.
    """
    # Corner 3s: generalmente en las esquinas con distancia <= 22.5 pies
    if shot_distance <= 22.5:
        if loc_x_pies < -18:  # Esquina izquierda
            return 'Left Corner 3'
        elif loc_x_pies > 18:  # Esquina derecha
            return 'Right Corner 3'
    
    # Above the Break 3: tiros desde arriba del arco
    if shot_distance > 22.5 and loc_y_pies > 5:
        return 'Above the Break 3'
    
    return 'Other 3PT'

def obtener_temporada_actual():
    """
    Determina automáticamente la temporada actual disponible.
    """
    current_year = datetime.now().year
    
    # La temporada NBA va de octubre a abril del siguiente año
    # Si estamos antes de octubre, usar la temporada anterior
    if datetime.now().month < 10:
        season_start = current_year - 1
    else:
        season_start = current_year
    
    season = f"{season_start}-{str(season_start + 1)[2:]}"
    print(f"🏀 Usando temporada: {season}")
    return season

def descargar_tiros_jugador(player_id, player_name, season):
    """
    Descarga y procesa todos los tiros de 3 puntos de un jugador.
    """
    print(f"📊 Descargando tiros de {player_name}...")
    
    try:
        # Obtener datos de tiros
        shot_chart = shotchartdetail.ShotChartDetail(
            team_id=0,
            player_id=player_id,
            season_nullable=season,
            context_measure_simple='FGA',
            season_type_all_star='Regular Season'
        )
        
        shot_data = shot_chart.get_data_frames()[0]
        
        if shot_data.empty:
            print(f"⚠️  No hay datos para {player_name}")
            return pd.DataFrame()
        
        # Filtrar solo tiros de 3 puntos (distancia >= 22 pies)
        shots_3pt = shot_data[shot_data['SHOT_DISTANCE'] >= 22].copy()
        
        if shots_3pt.empty:
            print(f"⚠️  No hay tiros de 3 puntos para {player_name}")
            return pd.DataFrame()
        
        # Convertir coordenadas a pies reales
        coords_converted = shots_3pt.apply(
            lambda row: convertir_coordenadas_api_a_pies(row['LOC_X'], row['LOC_Y']), 
            axis=1, result_type='expand'
        )
        shots_3pt['LOC_X_PIES'] = coords_converted[0]
        shots_3pt['LOC_Y_PIES'] = coords_converted[1]
        
        # Clasificar por zonas
        shots_3pt['Zone'] = shots_3pt.apply(
            lambda row: clasificar_zona_tiro(
                row['SHOT_DISTANCE'], 
                row['LOC_X_PIES'], 
                row['LOC_Y_PIES']
            ), axis=1
        )
        
        # Agregar información del jugador
        shots_3pt['PLAYER_NAME'] = player_name
        
        print(f"✅ {player_name}: {len(shots_3pt)} tiros de 3 puntos descargados")
        return shots_3pt
        
    except Exception as e:
        print(f"❌ Error descargando {player_name}: {e}")
        return pd.DataFrame()

def main():
    """
    Función principal para descargar datos de test.
    """
    print("🧪 NBA DATA DOWNLOADER - TEST DE 10 JUGADORES")
    print("=" * 50)
    
    # Obtener temporada actual
    season = obtener_temporada_actual()
    
    # Jugadores populares para test (ID, Nombre)
    test_players = [
        (2544, "LeBron James"),
        (77, "Luka Dončić"),
        (201939, "Stephen Curry"),
        (201935, "James Harden"),
        (203081, "Damian Lillard"),
        (203999, "Nikola Jokić"),
        (1628369, "Jayson Tatum"),
        (1629029, "Jaylen Brown"),
        (203507, "Giannis Antetokounmpo"),
        (201566, "Russell Westbrook")
    ]
    
    all_shots_data = []
    total_jugadores = len(test_players)
    
    print(f"📊 Descargando datos de {total_jugadores} jugadores...")
    print(f"⏱️  Tiempo estimado: 5-10 minutos")
    print()
    
    for i, (player_id, player_name) in enumerate(test_players, 1):
        print(f"[{i}/{total_jugadores}] Procesando {player_name}...")
        
        # Descargar tiros del jugador
        player_shots = descargar_tiros_jugador(player_id, player_name, season)
        
        if not player_shots.empty:
            all_shots_data.append(player_shots)
        
        # Rate limiting - esperar entre 3-7 segundos entre llamadas
        if i < total_jugadores:  # No esperar después del último
            delay = random.uniform(3, 7)
            print(f"⏳ Esperando {delay:.1f}s para la siguiente descarga...")
            time.sleep(delay)
    
    print("\n" + "=" * 50)
    print("📊 CONSOLIDANDO DATOS...")
    
    if not all_shots_data:
        print("❌ No se descargaron datos. Saliendo...")
        return
    
    # Combinar todos los datos
    combined_data = pd.concat(all_shots_data, ignore_index=True)
    
    # Crear archivo CSV
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
    filename = f"all_shots_3pt_{season}_TEST.csv"
    combined_data.to_csv(filename, index=False)
    
    # Estadísticas finales
    total_shots = len(combined_data)
    unique_players = len(combined_data['PLAYER_NAME'].unique())
    
    print(f"✅ DESCARGA COMPLETADA")
    print(f"📁 Archivo creado: {filename}")
    print(f"📊 Total de tiros: {total_shots:,}")
    print(f"👥 Jugadores procesados: {unique_players}")
    print()
    
    # Mostrar estadísticas por jugador
    print("📈 ESTADÍSTICAS POR JUGADOR:")
    player_stats = combined_data.groupby('PLAYER_NAME').agg({
        'SHOT_MADE_FLAG': 'sum',
        'SHOT_ATTEMPTED_FLAG': 'sum'
    }).reset_index()
    player_stats['3P%'] = (player_stats['SHOT_MADE_FLAG'] / player_stats['SHOT_ATTEMPTED_FLAG'] * 100).round(1)
    player_stats.columns = ['Jugador', '3PM', '3PA', '3P%']
    
    for _, row in player_stats.iterrows():
        print(f"  • {row['Jugador']}: {row['3PM']}/{row['3PA']} ({row['3P%']}%)")
    
    print("\n" + "=" * 50)
    print("🎯 SIGUIENTE PASO:")
    print("Para usar estos datos offline, ejecuta:")
    print("uv run python main.py")
    print()
    print("Para descargar TODOS los jugadores (3-4 horas):")
    print("uv run python download_all_shots.py")

if __name__ == "__main__":
    main()
