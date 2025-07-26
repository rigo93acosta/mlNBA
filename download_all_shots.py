#!/usr/bin/env python3
"""
🚀 NBA DATA DOWNLOADER - DESCARGA COMPLETA
==========================================

Este script descarga TODOS los tiros de 3 puntos individuales de TODOS los jugadores 
activos de la NBA para crear una base de datos offline completa.

CARACTERÍSTICAS:
• 📊 Descarga ~90,000+ tiros individuales de 3 puntos
• 👥 Procesa ~480-500 jugadores activos
• 📍 Incluye coordenadas convertidas a pies reales
• 🎯 Clasifica tiros por zonas (Corner 3s, Above the Break, etc.)
• 💾 Crea archivo CSV para uso offline con main.py
• 🔄 Sistema de reintentos automático para errores de red
• ⏱️ Rate limiting para evitar sobrecargar la NBA API

TIEMPO ESTIMADO: 3-4 horas
RESULTADO: all_shots_3pt_2024_25_COMPLETO.csv + nba_offline_loader.py

FILTROS APLICADOS:
• Solo jugadores con >100 minutos jugados en la temporada
• Solo tiros de 3 puntos (distancia >= 22 pies)
• Solo temporada regular (no playoffs)

EJECUCIÓN:
uv run python download_all_shots.py

ADVERTENCIA: Este script hace ~500 llamadas a la NBA API con delays de 8-15 segundos
entre cada llamada para respetar los límites de rate limiting.
"""

import pandas as pd
import time
import os
import random
from datetime import datetime
from nba_api.stats.endpoints import commonallplayers, playerdashboardbygeneralsplits, shotchartdetail

def convertir_coordenadas_api_a_pies(loc_x, loc_y):
    """
    Convierte las coordenadas de la NBA API a coordenadas reales en pies.
    """
    factor_x = 50.0 / 500.0  # 0.1 pies por unidad API
    factor_y = 47.0 / 420.0  # ~0.112 pies por unidad API
    
    x_pies = loc_x * factor_x
    y_pies = loc_y * factor_y
    
    return x_pies, y_pies

def download_player_all_shots(player_info, season):
    """
    Descarga TODOS los tiros individuales de 3 puntos de un jugador
    """
    player_id = player_info['PLAYER_ID']
    player_name = player_info['PLAYER_NAME']
    team = player_info['TEAM']
    
    # Múltiples reintentos para manejar timeouts
    for retry in range(5):
        try:
            shot_chart = shotchartdetail.ShotChartDetail(
                team_id=0,
                player_id=player_id,
                season_nullable=season,
                season_type_all_star='Regular Season',
                context_measure_simple='FGA'  # Todos los tiros, no solo 3FGA
            )
            shot_data = shot_chart.get_data_frames()[0]
            break
            
        except Exception as e:
            if retry < 4:
                wait_time = (retry + 1) * 5
                print(f"   ⚠️ Intento {retry+1}/5 falló: {str(e)[:100]}...")
                print(f"   🔄 Reintentando en {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"   ❌ Falló después de 5 intentos: {str(e)[:100]}...")
                return None
    
    if shot_data.empty:
        return None
    
    # Filtrar solo triples (distancia >= 22 pies)
    shot_data_3pt = shot_data[shot_data['SHOT_DISTANCE'] >= 22].copy()
    
    if len(shot_data_3pt) == 0:
        return None
    
    # Convertir coordenadas a pies reales
    coords_converted = shot_data_3pt.apply(
        lambda row: convertir_coordenadas_api_a_pies(row['LOC_X'], row['LOC_Y']), 
        axis=1, result_type='expand'
    )
    shot_data_3pt['LOC_X_PIES'] = coords_converted[0]
    shot_data_3pt['LOC_Y_PIES'] = coords_converted[1]
    
    # Agregar información del jugador a cada tiro
    shot_data_3pt['PLAYER_NAME'] = player_name
    shot_data_3pt['TEAM'] = team
    shot_data_3pt['MINUTES_PLAYED'] = player_info['MINUTES_PLAYED']
    
    # Clasificar por zonas
    shot_data_3pt['Zone'] = shot_data_3pt.apply(
        lambda x: 'Left Corner 3' if (x['SHOT_DISTANCE'] <= 22.5 and x['LOC_X'] < -200) else
                'Right Corner 3' if (x['SHOT_DISTANCE'] <= 22.5 and x['LOC_X'] > 200) else
                'Above the Break 3' if (x['SHOT_DISTANCE'] > 22.5 and x['LOC_Y'] > 100) else 'Other',
        axis=1
    )
    
    return shot_data_3pt

def download_all_individual_shots(season='2024-25', min_minutes=100):
    """
    Descarga TODOS los tiros individuales de 3 puntos de todos los jugadores calificados
    """
    print(f"🏀 DESCARGA COMPLETA DE TIROS INDIVIDUALES - TEMPORADA {season}")
    print("="*70)
    print(f"💾 Objetivo: Crear base de datos offline para main_fixed.py")
    print(f"📊 Incluye: TODOS los tiros individuales de 3 puntos con coordenadas")
    print(f"⏱️ Mínimo de minutos: {min_minutes}")
    print("="*70)
    
    # Archivos
    qualified_file = f'qualified_players_{season.replace("-", "_")}.csv'
    output_file = f'all_shots_3pt_{season.replace("-", "_")}_COMPLETO.csv'
    temp_file = f'temp_all_shots_{season.replace("-", "_")}.csv'
    
    # Cargar jugadores calificados
    if not os.path.exists(qualified_file):
        print("❌ No se encontró la lista de jugadores calificados")
        print("   Ejecuta primero download_safe.py")
        return
    
    qualified_df = pd.read_csv(qualified_file)
    qualified_players = qualified_df.to_dict('records')
    
    # Filtrar por minutos mínimos para reducir la carga
    qualified_players = [p for p in qualified_players if p['MINUTES_PLAYED'] >= min_minutes]
    
    print(f"📋 {len(qualified_players)} jugadores calificados (>{min_minutes} min)")
    
    # Cargar progreso existente
    all_shots = []
    processed_players = set()
    start_from = 0
    
    if os.path.exists(temp_file):
        print("📁 Cargando progreso existente...")
        existing_data = pd.read_csv(temp_file)
        all_shots.append(existing_data)
        processed_players = set(existing_data['PLAYER_NAME'].unique())
        start_from = len(processed_players)
        print(f"✅ {len(processed_players)} jugadores ya procesados")
    
    print(f"🚀 Comenzando desde jugador {start_from}...")
    
    # Procesar jugadores
    for i, player_info in enumerate(qualified_players[start_from:], start_from):
        player_name = player_info['PLAYER_NAME']
        team = player_info['TEAM']
        
        # Saltar si ya fue procesado
        if player_name in processed_players:
            print(f"⏭️ [{i+1}/{len(qualified_players)}] {player_name} ya procesado")
            continue
        
        print(f"🏀 [{i+1}/{len(qualified_players)}] Descargando tiros: {player_name} ({team})")
        
        try:
            # Rate limiting agresivo
            wait_time = random.uniform(8.0, 15.0)
            print(f"   ⏳ Esperando {wait_time:.1f}s...")
            time.sleep(wait_time)
            
            # Descargar todos los tiros del jugador
            player_shots = download_player_all_shots(player_info, season)
            
            if player_shots is not None:
                all_shots.append(player_shots)
                processed_players.add(player_name)
                
                total_shots = len(player_shots)
                made_shots = player_shots['SHOT_MADE_FLAG'].sum()
                percentage = (made_shots / total_shots * 100) if total_shots > 0 else 0
                
                print(f"   ✅ {total_shots} tiros descargados ({made_shots}/{total_shots} = {percentage:.1f}%)")
                
                # Guardar progreso cada 5 jugadores
                if (i - start_from + 1) % 5 == 0:
                    combined_df = pd.concat(all_shots, ignore_index=True)
                    combined_df.to_csv(temp_file, index=False)
                    total_shots_so_far = len(combined_df)
                    unique_players = len(combined_df['PLAYER_NAME'].unique())
                    print(f"   💾 Progreso guardado: {unique_players} jugadores, {total_shots_so_far:,} tiros total")
            else:
                print(f"   ⚠️ Sin datos de tiros para {player_name}")
                
        except KeyboardInterrupt:
            print(f"\n⚠️ Descarga interrumpida por el usuario")
            print(f"📍 Para continuar: start_from={i}")
            break
        except Exception as e:
            print(f"   ❌ Error procesando {player_name}: {str(e)[:100]}")
            time.sleep(15)
            continue
    
    # Crear archivo final
    if all_shots:
        print("\n🎉 Finalizando descarga...")
        final_df = pd.concat(all_shots, ignore_index=True)
        
        # Ordenar por jugador y fecha
        final_df = final_df.sort_values(['PLAYER_NAME', 'GAME_DATE'])
        
        # Guardar archivo final
        final_df.to_csv(output_file, index=False)
        
        # Estadísticas finales
        total_shots = len(final_df)
        unique_players = len(final_df['PLAYER_NAME'].unique())
        unique_games = len(final_df['GAME_ID'].unique())
        total_made = final_df['SHOT_MADE_FLAG'].sum()
        overall_percentage = (total_made / total_shots * 100) if total_shots > 0 else 0
        
        print(f"\n📊 DESCARGA COMPLETA FINALIZADA!")
        print(f"📁 Archivo final: {output_file}")
        print(f"👥 Jugadores procesados: {unique_players}")
        print(f"🎮 Juegos únicos: {unique_games}")
        print(f"🏀 Total tiros de 3 puntos: {total_shots:,}")
        print(f"✅ Tiros anotados: {total_made:,}")
        print(f"📈 Porcentaje global: {overall_percentage:.1f}%")
        
        # Top 10 jugadores por volumen de tiros
        print(f"\n🏆 TOP 10 JUGADORES POR VOLUMEN DE TIROS:")
        top_shooters = final_df.groupby('PLAYER_NAME').agg({
            'SHOT_ATTEMPTED_FLAG': 'sum',
            'SHOT_MADE_FLAG': 'sum',
            'TEAM': 'first'
        }).reset_index()
        top_shooters['3P%'] = (top_shooters['SHOT_MADE_FLAG'] / top_shooters['SHOT_ATTEMPTED_FLAG'] * 100).round(1)
        top_shooters = top_shooters.sort_values('SHOT_ATTEMPTED_FLAG', ascending=False).head(10)
        
        for _, player in top_shooters.iterrows():
            print(f"{player['PLAYER_NAME']:<20} ({player['TEAM']:<3}): {player['SHOT_ATTEMPTED_FLAG']:>3} tiros, {player['3P%']:>5.1f}%")
        
        # Limpiar archivo temporal
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"🗑️ Archivo temporal eliminado")
        
        print(f"\n✅ Base de datos offline lista para main_fixed.py!")
        return output_file
    
    return None

def create_offline_data_loader():
    """
    Crea un módulo para cargar datos offline en main_fixed.py
    """
    offline_loader_code = '''#!/usr/bin/env python3
"""
Cargador de datos offline para main_fixed.py
"""

import pandas as pd
import os

class OfflineNBAData:
    def __init__(self, shots_file=None):
        """
        Inicializa el cargador de datos offline
        """
        if shots_file is None:
            # Buscar automáticamente el archivo más reciente
            files = [f for f in os.listdir('.') if f.startswith('all_shots_3pt_') and f.endswith('_COMPLETO.csv')]
            if not files:
                raise FileNotFoundError("No se encontró archivo de tiros offline")
            shots_file = max(files)  # Tomar el más reciente
        
        print(f"📁 Cargando datos offline: {shots_file}")
        self.all_shots = pd.read_csv(shots_file)
        
        total_shots = len(self.all_shots)
        unique_players = len(self.all_shots['PLAYER_NAME'].unique())
        print(f"✅ Datos cargados: {unique_players} jugadores, {total_shots:,} tiros")
    
    def get_player_shots(self, player_id=None, player_name=None):
        """
        Obtiene todos los tiros de un jugador específico
        Compatible con shotchartdetail.ShotChartDetail().get_data_frames()[0]
        """
        if player_name:
            shots = self.all_shots[self.all_shots['PLAYER_NAME'] == player_name].copy()
        elif player_id:
            # Buscar por ID si está disponible
            shots = self.all_shots[self.all_shots['PLAYER_ID'] == player_id].copy()
        else:
            raise ValueError("Debe especificar player_name o player_id")
        
        return shots
    
    def list_available_players(self):
        """
        Lista todos los jugadores disponibles
        """
        players = self.all_shots[['PLAYER_NAME', 'TEAM']].drop_duplicates()
        return players.sort_values('PLAYER_NAME')
    
    def get_player_by_name_similarity(self, partial_name):
        """
        Busca jugadores por similitud de nombre
        """
        players = self.list_available_players()
        matches = players[players['PLAYER_NAME'].str.contains(partial_name, case=False, na=False)]
        return matches

# Crear instancia global
try:
    nba_offline = OfflineNBAData()
    print("🔌 Modo offline activado - datos cargados exitosamente")
except Exception as e:
    print(f"❌ Error cargando datos offline: {e}")
    nba_offline = None
'''
    
    with open('nba_offline_loader.py', 'w') as f:
        f.write(offline_loader_code)
    
    print("📦 Módulo de carga offline creado: nba_offline_loader.py")

if __name__ == "__main__":
    # Configuración para descarga completa
    season = '2024-25'
    min_minutes = 100  # Solo jugadores con más de 100 minutos para reducir volumen
    
    print("🚀 INICIANDO DESCARGA DE TODOS LOS TIROS INDIVIDUALES")
    print("="*70)
    print(f"🎯 Objetivo: Crear base de datos offline completa")
    print(f"📊 Incluye: Cada tiro individual con coordenadas, fechas, etc.")
    print(f"💡 Para usar en main_fixed.py sin conexión a internet")
    print("="*70)
    
    # Ejecutar descarga
    output_file = download_all_individual_shots(
        season=season,
        min_minutes=min_minutes
    )
    
    if output_file:
        print(f"\n✅ Descarga exitosa!")
        print(f"📁 Archivo: {output_file}")
        
        # Crear cargador offline
        create_offline_data_loader()
        
        print(f"\n🔧 PRÓXIMOS PASOS:")
        print(f"1. ✅ Base de datos offline creada")
        print(f"2. 🔄 Modificar main_fixed.py para usar datos offline")
        print(f"3. 🏀 Ejecutar análisis sin conexión a internet")
    else:
        print(f"\n❌ La descarga falló")
