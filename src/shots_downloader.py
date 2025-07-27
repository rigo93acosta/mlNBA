#!/usr/bin/env python3
"""
⬇️ NBA SHOTS DOWNLOADER
=======================

Script específico para descargar datos individuales de shots de jugadores.
Complementa el data_collector principal con descarga detallada.

FUNCIONALIDADES:
• Descarga shots individuales por jugador
• Descarga shots por equipo específico
• Descarga shots de toda la liga
• Filtros por temporada y tipo de juego
• Datos detallados de posición y zona
"""

import pandas as pd
import numpy as np
from nba_api.stats.endpoints import shotchartdetail, playercareerstats
from nba_api.stats.static import teams, players
import time
import warnings
warnings.filterwarnings('ignore')

class NBAShotsDownloader:
    """Descargador específico de shots NBA."""
    
    def __init__(self, season='2024-25'):
        """Inicializa el descargador."""
        self.season = season
        self.season_type = 'Regular Season'
        
        print(f"⬇️ NBA SHOTS DOWNLOADER - Temporada {season}")
        print("="*55)
        print("📊 Descarga detallada de shots individuales")
    
    def get_all_players(self):
        """Obtiene lista de todos los jugadores activos."""
        try:
            all_players = players.get_active_players()
            return {player['full_name']: player['id'] for player in all_players}
        except Exception as e:
            print(f"❌ Error obteniendo jugadores: {e}")
            return {}
    
    def get_team_roster(self, team_name):
        """Obtiene roster de un equipo específico."""
        try:
            # Obtener ID del equipo
            team_info = teams.find_teams_by_full_name(team_name)
            if not team_info:
                print(f"❌ Equipo no encontrado: {team_name}")
                return {}
            
            team_id = team_info[0]['id']
            
            # Obtener shots del equipo para encontrar jugadores
            team_shots = shotchartdetail.ShotChartDetail(
                team_id=team_id,
                player_id=0,
                season_nullable=self.season,
                season_type_all_star=self.season_type
            )
            
            shots_df = team_shots.get_data_frames()[0]
            
            if not shots_df.empty:
                roster = shots_df[['PLAYER_ID', 'PLAYER_NAME']].drop_duplicates()
                return dict(zip(roster['PLAYER_NAME'], roster['PLAYER_ID']))
            
            return {}
            
        except Exception as e:
            print(f"❌ Error obteniendo roster: {e}")
            return {}
    
    def download_player_shots(self, player_name, player_id=None):
        """Descarga shots de un jugador específico."""
        try:
            if player_id is None:
                # Buscar ID del jugador
                all_players = self.get_all_players()
                if player_name not in all_players:
                    print(f"❌ Jugador no encontrado: {player_name}")
                    return None
                player_id = all_players[player_name]
            
            print(f"📊 Descargando shots de {player_name}...")
            
            # Obtener shots del jugador
            player_shots = shotchartdetail.ShotChartDetail(
                team_id=0,
                player_id=player_id,
                season_nullable=self.season,
                season_type_all_star=self.season_type
            )
            
            shots_df = player_shots.get_data_frames()[0]
            
            if not shots_df.empty:
                # Agregar información del jugador
                shots_df['SEASON'] = self.season
                shots_df['DOWNLOAD_DATE'] = pd.Timestamp.now()
                
                # Guardar
                filename = f"data/shots_{player_name.replace(' ', '_')}_{self.season.replace('-', '_')}.csv"
                shots_df.to_csv(filename, index=False)
                
                print(f"✅ Shots guardados: {len(shots_df):,} tiros")
                print(f"📁 Archivo: {filename}")
                
                # Estadísticas rápidas
                total_3pt = shots_df[shots_df['SHOT_TYPE'] == '3PT Field Goal']
                if not total_3pt.empty:
                    made_3pt = total_3pt['SHOT_MADE_FLAG'].sum()
                    pct_3pt = made_3pt / len(total_3pt)
                    print(f"🎯 Tiros 3PT: {len(total_3pt):,} ({made_3pt:,} anotados, {pct_3pt:.1%})")
                
                return shots_df
            else:
                print(f"❌ No se encontraron shots para {player_name}")
                return None
                
        except Exception as e:
            print(f"❌ Error descargando shots de {player_name}: {e}")
            return None
    
    def download_team_shots(self, team_name):
        """Descarga shots de todo un equipo."""
        try:
            # Obtener ID del equipo
            team_info = teams.find_teams_by_full_name(team_name)
            if not team_info:
                print(f"❌ Equipo no encontrado: {team_name}")
                return None
            
            team_id = team_info[0]['id']
            
            print(f"🏀 Descargando shots de {team_name}...")
            
            # Obtener shots del equipo
            team_shots = shotchartdetail.ShotChartDetail(
                team_id=team_id,
                player_id=0,
                season_nullable=self.season,
                season_type_all_star=self.season_type
            )
            
            shots_df = team_shots.get_data_frames()[0]
            
            if not shots_df.empty:
                # Agregar información
                shots_df['TEAM_NAME'] = team_name
                shots_df['SEASON'] = self.season
                shots_df['DOWNLOAD_DATE'] = pd.Timestamp.now()
                
                # Guardar
                filename = f"data/shots_{team_name.replace(' ', '_')}_{self.season.replace('-', '_')}.csv"
                shots_df.to_csv(filename, index=False)
                
                print(f"✅ Shots guardados: {len(shots_df):,} tiros")
                print(f"👥 Jugadores: {shots_df['PLAYER_NAME'].nunique()}")
                print(f"📁 Archivo: {filename}")
                
                # Top shooters del equipo
                top_shooters = shots_df.groupby('PLAYER_NAME').size().nlargest(5)
                print(f"🏆 Top 5 tiradores:")
                for player, shots in top_shooters.items():
                    print(f"   {player}: {shots:,} tiros")
                
                return shots_df
            else:
                print(f"❌ No se encontraron shots para {team_name}")
                return None
                
        except Exception as e:
            print(f"❌ Error descargando shots de {team_name}: {e}")
            return None
    
    def download_all_3pt_shots(self, output_file='data/all_shots_3pt_detailed.csv'):
        """Descarga todos los tiros de 3PT de la liga."""
        print("🌟 Descargando TODOS los tiros 3PT de la liga...")
        print("⚠️ Este proceso puede tomar mucho tiempo...")
        
        # Obtener todos los equipos
        nba_teams = teams.get_teams()
        all_shots = []
        
        for i, team in enumerate(nba_teams, 1):
            try:
                team_name = team['full_name']
                team_id = team['id']
                
                print(f"  {i:2d}/30 - {team_name}")
                
                # Obtener shots del equipo
                team_shots = shotchartdetail.ShotChartDetail(
                    team_id=team_id,
                    player_id=0,
                    season_nullable=self.season,
                    season_type_all_star=self.season_type
                )
                
                shots_df = team_shots.get_data_frames()[0]
                
                # Filtrar solo 3PT
                if not shots_df.empty:
                    shots_3pt = shots_df[shots_df['SHOT_TYPE'] == '3PT Field Goal'].copy()
                    
                    if not shots_3pt.empty:
                        shots_3pt['TEAM_NAME'] = team_name
                        shots_3pt['SEASON'] = self.season
                        all_shots.append(shots_3pt)
                        print(f"    ✅ {len(shots_3pt):,} tiros 3PT")
                    else:
                        print(f"    ⚠️ Sin tiros 3PT")
                
                time.sleep(0.6)  # Rate limiting
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                continue
        
        if all_shots:
            # Combinar todos los datos
            final_df = pd.concat(all_shots, ignore_index=True)
            final_df['DOWNLOAD_DATE'] = pd.Timestamp.now()
            
            # Guardar
            final_df.to_csv(output_file, index=False)
            
            print(f"\n🎉 DESCARGA COMPLETADA")
            print("="*50)
            print(f"✅ Archivo guardado: {output_file}")
            print(f"📊 Total tiros 3PT: {len(final_df):,}")
            print(f"👥 Jugadores únicos: {final_df['PLAYER_NAME'].nunique()}")
            print(f"🏀 Equipos: {final_df['TEAM_NAME'].nunique()}")
            print(f"🎯 Tiros convertidos: {final_df['SHOT_MADE_FLAG'].sum():,}")
            print(f"📈 Porcentaje 3PT liga: {final_df['SHOT_MADE_FLAG'].mean():.1%}")
            
            return final_df
        else:
            print("❌ No se pudieron descargar datos")
            return None
    
    def download_top_players_shots(self, num_players=20):
        """Descarga shots de los mejores tiradores de 3PT."""
        print(f"🌟 Descargando shots de top {num_players} tiradores 3PT...")
        
        # Primero necesitamos obtener datos generales para identificar top players
        try:
            # Usar el archivo principal si existe
            shots_df = pd.read_csv('data/all_shots_3pt_2024_25_COMPLETO.csv')
            
            # Calcular top players
            player_stats = shots_df.groupby(['PLAYER_NAME', 'PLAYER_ID']).agg({
                'SHOT_MADE_FLAG': ['count', 'sum']
            }).round(3)
            
            player_stats.columns = ['Total_Attempts', 'Made']
            player_stats['Percentage'] = player_stats['Made'] / player_stats['Total_Attempts']
            
            # Filtrar players con suficientes intentos y mejor porcentaje
            min_attempts = 50
            qualified_players = player_stats[player_stats['Total_Attempts'] >= min_attempts]
            top_players = qualified_players.nlargest(num_players, 'Percentage')
            
            print(f"📊 Top {num_players} tiradores 3PT (mín. {min_attempts} intentos):")
            for i, (player_info, stats) in enumerate(top_players.iterrows(), 1):
                player_name = player_info[0]
                print(f"  {i:2d}. {player_name}: {stats['Percentage']:.1%} ({stats['Made']:.0f}/{stats['Total_Attempts']:.0f})")
            
            # Descargar shots detallados de cada uno
            detailed_shots = []
            for player_info, stats in top_players.iterrows():
                player_name, player_id = player_info
                player_shots = self.download_player_shots(player_name, player_id)
                if player_shots is not None:
                    detailed_shots.append(player_shots)
                time.sleep(1)  # Rate limiting
            
            if detailed_shots:
                # Combinar todos
                all_detailed = pd.concat(detailed_shots, ignore_index=True)
                filename = f"data/top_{num_players}_shooters_detailed_{self.season.replace('-', '_')}.csv"
                all_detailed.to_csv(filename, index=False)
                print(f"✅ Shots detallados guardados: {filename}")
                return all_detailed
            
        except FileNotFoundError:
            print("❌ No se encontró archivo principal. Ejecuta primero download_all_3pt_shots()")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

def interactive_downloader():
    """Interfaz interactiva del descargador."""
    downloader = NBAShotsDownloader()
    
    while True:
        print("\n⬇️ NBA SHOTS DOWNLOADER - MENÚ")
        print("="*45)
        print("1. 👤 Descargar shots de jugador específico")
        print("2. 🏀 Descargar shots de equipo específico")
        print("3. 🌟 Descargar TODOS los tiros 3PT")
        print("4. 🏆 Descargar top tiradores 3PT")
        print("5. ℹ️ Información de equipos")
        print("6. 🚪 Salir")
        
        try:
            choice = input("\n👉 Selecciona (1-6): ").strip()
            
            if choice == '1':
                player = input("🏀 Nombre del jugador: ").strip()
                downloader.download_player_shots(player)
            
            elif choice == '2':
                team = input("🏀 Nombre del equipo: ").strip()
                downloader.download_team_shots(team)
            
            elif choice == '3':
                confirm = input("⚠️ Descarga completa (puede tomar 20+ min). ¿Continuar? (s/n): ")
                if confirm.lower().startswith('s'):
                    downloader.download_all_3pt_shots()
            
            elif choice == '4':
                try:
                    num = int(input("🏆 Número de top players (default 20): ") or "20")
                    downloader.download_top_players_shots(num)
                except ValueError:
                    print("❌ Número no válido")
            
            elif choice == '5':
                nba_teams = teams.get_teams()
                print("\n🏀 EQUIPOS NBA:")
                for i, team in enumerate(nba_teams, 1):
                    print(f"  {i:2d}. {team['full_name']}")
            
            elif choice == '6':
                print("👋 ¡Hasta luego!")
                break
            
            else:
                print("❌ Opción no válida")
                
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break

def main():
    """Función principal."""
    print("⬇️ NBA SHOTS DOWNLOADER")
    print("="*50)
    print("📊 Descarga detallada de shots individuales")
    print()
    
    interactive_downloader()

if __name__ == "__main__":
    main()
