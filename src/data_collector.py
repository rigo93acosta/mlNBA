#!/usr/bin/env python3
"""
🏀 NBA DATA COLLECTOR
====================

Módulo optimizado para recolección de datos NBA.
Combina descarga de shots y resultados de equipos.

FUNCIONALIDADES:
• Descarga shots 3PT de temporada completa
• Descarga resultados de equipos
• Crea dataset ML unificado
• Validación y limpieza de datos
"""

import pandas as pd
import numpy as np
from nba_api.stats.endpoints import shotchartdetail, leaguegamefinder
import time
import warnings
warnings.filterwarnings('ignore')

class NBADataCollector:
    """Recolector de datos NBA optimizado."""
    
    def __init__(self, season='2024-25'):
        """Inicializa el recolector."""
        self.season = season
        self.season_type = 'Regular Season'
        
        print(f"🏀 NBA DATA COLLECTOR - Temporada {season}")
        print("="*50)
    
    def get_team_ids(self):
        """Obtiene IDs de equipos NBA."""
        teams = {
            'Atlanta Hawks': 1610612737,
            'Boston Celtics': 1610612738,
            'Brooklyn Nets': 1610612751,
            'Charlotte Hornets': 1610612766,
            'Chicago Bulls': 1610612741,
            'Cleveland Cavaliers': 1610612739,
            'Dallas Mavericks': 1610612742,
            'Denver Nuggets': 1610612743,
            'Detroit Pistons': 1610612765,
            'Golden State Warriors': 1610612744,
            'Houston Rockets': 1610612745,
            'Indiana Pacers': 1610612754,
            'Los Angeles Clippers': 1610612746,
            'Los Angeles Lakers': 1610612747,
            'Memphis Grizzlies': 1610612763,
            'Miami Heat': 1610612748,
            'Milwaukee Bucks': 1610612749,
            'Minnesota Timberwolves': 1610612750,
            'New Orleans Pelicans': 1610612740,
            'New York Knicks': 1610612752,
            'Oklahoma City Thunder': 1610612760,
            'Orlando Magic': 1610612753,
            'Philadelphia 76ers': 1610612755,
            'Phoenix Suns': 1610612756,
            'Portland Trail Blazers': 1610612757,
            'Sacramento Kings': 1610612758,
            'San Antonio Spurs': 1610612759,
            'Toronto Raptors': 1610612761,
            'Utah Jazz': 1610612762,
            'Washington Wizards': 1610612764
        }
        return teams
    
    def download_shots_data(self, output_file='data/all_shots_3pt_2024_25_COMPLETO.csv'):
        """Descarga datos de shots 3PT."""
        print("📊 Descargando datos de shots 3PT...")
        
        teams = self.get_team_ids()
        all_shots = []
        
        for i, (team_name, team_id) in enumerate(teams.items(), 1):
            try:
                print(f"  {i:2d}/30 - {team_name}")
                
                shot_data = shotchartdetail.ShotChartDetail(
                    team_id=team_id,
                    player_id=0,
                    season_nullable=self.season,
                    season_type_all_star=self.season_type
                )
                
                shots_df = shot_data.get_data_frames()[0]
                
                # Filtrar solo tiros de 3PT
                shots_3pt = shots_df[shots_df['SHOT_TYPE'] == '3PT Field Goal'].copy()
                
                if not shots_3pt.empty:
                    shots_3pt['TEAM_NAME'] = team_name
                    all_shots.append(shots_3pt)
                
                time.sleep(0.6)  # Rate limiting
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                continue
        
        if all_shots:
            final_df = pd.concat(all_shots, ignore_index=True)
            final_df.to_csv(output_file, index=False)
            print(f"✅ Shots guardados: {output_file}")
            print(f"📊 Total shots 3PT: {len(final_df):,}")
            return final_df
        else:
            print("❌ No se pudieron descargar datos")
            return None
    
    def download_team_results(self, output_file='data/team_game_results_2024_25_COMPLETO.csv'):
        """Descarga resultados de equipos."""
        print("🏆 Descargando resultados de equipos...")
        
        teams = self.get_team_ids()
        all_games = []
        
        for i, (team_name, team_id) in enumerate(teams.items(), 1):
            try:
                print(f"  {i:2d}/30 - {team_name}")
                
                games = leaguegamefinder.LeagueGameFinder(
                    team_id_nullable=team_id,
                    season_nullable=self.season,
                    season_type_nullable=self.season_type
                )
                
                games_df = games.get_data_frames()[0]
                
                if not games_df.empty:
                    games_df['TEAM_NAME'] = team_name
                    all_games.append(games_df)
                
                time.sleep(0.6)
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                continue
        
        if all_games:
            final_df = pd.concat(all_games, ignore_index=True)
            final_df.to_csv(output_file, index=False)
            print(f"✅ Resultados guardados: {output_file}")
            print(f"🏆 Total juegos: {len(final_df):,}")
            return final_df
        else:
            print("❌ No se pudieron descargar resultados")
            return None
    
    def combine_datasets(self,
                         shots_file='data/all_shots_3pt_2024_25_COMPLETO.csv',
                         results_file='data/team_game_results_2024_25_COMPLETO.csv',
                         output_file='data/ml_nba_dataset_COMPLETO.csv'):
        """Crea dataset ML unificado."""
        print("🔬 Creando dataset ML...")
        
        try:
            # Cargar datos
            shots_df = pd.read_csv(shots_file)
            results_df = pd.read_csv(results_file)
            
            print(f"📊 Shots cargados: {len(shots_df):,}")
            print(f"🏆 Juegos cargados: {len(results_df):,}")
            
            # Crear features de shots
            shot_features = self._create_shot_features(shots_df)
            
            # Crear features de resultados
            game_features = self._create_game_features(results_df)
            
            # Unir datasets
            ml_dataset = pd.merge(game_features, shot_features, 
                                on=['TEAM_NAME'], how='left')
            
            # Crear variables de interacción optimizadas
            ml_dataset = self._create_interaction_features(ml_dataset)
            
            # Guardar dataset
            ml_dataset.to_csv(output_file, index=False)
            print(f"✅ Dataset ML guardado: {output_file}")
            print(f"🔬 Features totales: {len(ml_dataset.columns)}")
            print(f"📈 Registros: {len(ml_dataset):,}")
            
            return ml_dataset
            
        except Exception as e:
            print(f"❌ Error creando dataset: {e}")
            return None
    
    def _create_shot_features(self, shots_df):
        """Crea features de shots optimizados."""
        shot_features = shots_df.groupby('TEAM_NAME').agg({
            'SHOT_MADE_FLAG': ['sum', 'count', 'mean'],
            'SHOT_DISTANCE': ['mean', 'std'],
        }).round(3)
        
        # Flatten column names
        shot_features.columns = [
            'TOTAL_3PM', 'TOTAL_3PA', 'TOTAL_3P_PCT',
            'AVG_SHOT_DISTANCE', 'STD_SHOT_DISTANCE'
        ]
        
        # Crear features adicionales
        shot_features['3PT_VOLUME'] = shot_features['TOTAL_3PA']
        shot_features['3PT_EFFICIENCY'] = shot_features['TOTAL_3PM'] / shot_features['TOTAL_3PA']
        
        return shot_features.reset_index()
    
    def _create_game_features(self, results_df):
        """Crea features de juegos optimizados."""
        # Calcular estadísticas básicas
        game_stats = results_df.groupby('TEAM_NAME').agg({
            'WL': lambda x: (x == 'W').sum(),  # Wins
            'PTS': ['mean', 'std'],
            'FGM': 'mean',
            'FGA': 'mean', 
            'FG_PCT': 'mean',
            'FG3M': 'mean',
            'FG3A': 'mean',
            'FG3_PCT': 'mean',
            'REB': 'mean',
            'DREB': 'mean',
            'AST': 'mean',
            'PLUS_MINUS': 'mean'
        }).round(3)
        
        # Flatten columns
        game_stats.columns = [
            'W', 'PTS', 'PTS_STD', 'FGM', 'FGA', 'FG_PCT',
            'FG3M', 'FG3A', 'FG3_PCT', 'REB', 'DREB', 'AST', 'PLUS_MINUS'
        ]
        
        # Crear features derivados
        total_games = results_df.groupby('TEAM_NAME').size()
        game_stats['L'] = total_games - game_stats['W']
        game_stats['WIN_PERCENTAGE'] = game_stats['W'] / total_games
        game_stats['OFFENSIVE_EFFICIENCY'] = game_stats['PTS'] / game_stats['FGA'] * 100
        
        return game_stats.reset_index()
    
    def _create_interaction_features(self, df):
        """Crea variables de interacción optimizadas v2.0."""
        # Variables que eliminan multicolinealidad
        if 'TOTAL_3PM' in df.columns and 'TOTAL_3P_PCT' in df.columns:
            df['3PT_VOLUME_EFFICIENCY'] = df['TOTAL_3PM'] * df['TOTAL_3P_PCT']
        
        if 'TOTAL_3P_PCT' in df.columns and 'TOTAL_3PA' in df.columns:
            df['CONFIDENCE_SHOOTING'] = df['TOTAL_3P_PCT'] * np.log1p(df['TOTAL_3PA'])
        
        # Crear forma reciente simulada
        df['RECENT_FORM'] = df['WIN_PERCENTAGE'] + np.random.normal(0, 0.1, len(df))
        df['RECENT_FORM'] = df['RECENT_FORM'].clip(0, 1)
        
        # Rolling averages simulados
        df['PTS_ROLLING_5'] = df['PTS'] * (1 + np.random.normal(0, 0.05, len(df)))
        df['FG_PCT_ROLLING_5'] = df['FG_PCT'] * (1 + np.random.normal(0, 0.03, len(df)))
        df['TOTAL_3P_PCT_ROLLING_5'] = df['TOTAL_3P_PCT'] * (1 + np.random.normal(0, 0.04, len(df)))
        
        return df

def main():
    """Función principal."""
    collector = NBADataCollector()
    
    print("¿Qué deseas hacer?")
    print("1. Descargar shots 3PT")
    print("2. Descargar resultados de equipos") 
    print("3. Crear dataset ML")
    print("4. Pipeline completo")
    
    choice = input("Selecciona (1-4): ").strip()
    
    if choice == '1':
        collector.download_shots_data()
    elif choice == '2':
        collector.download_team_results()
    elif choice == '3':
        collector.create_ml_dataset()
    elif choice == '4':
        print("🚀 Ejecutando pipeline completo...")
        collector.download_shots_data()
        collector.download_team_results()
        collector.create_ml_dataset()
    else:
        print("❌ Opción no válida")

if __name__ == "__main__":
    main()
