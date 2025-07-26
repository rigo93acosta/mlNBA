#!/usr/bin/env python3
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
