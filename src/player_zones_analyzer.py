#!/usr/bin/env python3
"""
🎯 NBA PLAYER SHOT ZONES ANALYZER
=================================

Analiza las zonas desde donde los jugadores realizan tiros de 3 puntos.
Genera visualizaciones y estadísticas detalladas por jugador y zona.

FUNCIONALIDADES:
• Análisis de zonas de tiro por jugador
• Mapas de calor de tiros
• Estadísticas de efectividad por zona
• Visualizaciones interactivas
• Comparación entre jugadores
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class PlayerShotZonesAnalyzer:
    """Analizador de zonas de tiro por jugador."""
    
    def __init__(self, shots_file='data/all_shots_3pt_2024_25_COMPLETO.csv'):
        """Inicializa el analizador."""
        self.shots_file = shots_file
        self.shots_df = None
        
        print("🎯 NBA PLAYER SHOT ZONES ANALYZER")
        print("="*50)
        print("📊 Análisis detallado de zonas de tiro 3PT por jugador")
        
    def load_shots_data(self):
        """Carga datos de tiros."""
        try:
            self.shots_df = pd.read_csv(self.shots_file)
            print(f"✅ Datos cargados: {len(self.shots_df):,} tiros de 3PT")
            print(f"👥 Jugadores únicos: {self.shots_df['PLAYER_NAME'].nunique()}")
            print(f"🏀 Equipos: {self.shots_df['TEAM_NAME'].nunique()}")
            return True
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return False
    
    def analyze_player_zones(self, player_name):
        """Analiza zonas de tiro de un jugador específico."""
        if self.shots_df is None:
            print("❌ Primero carga los datos")
            return None
        
        # Filtrar por jugador
        player_shots = self.shots_df[self.shots_df['PLAYER_NAME'] == player_name].copy()
        
        if player_shots.empty:
            print(f"❌ No se encontraron datos para {player_name}")
            return None
        
        print(f"\n🎯 ANÁLISIS DE ZONAS - {player_name}")
        print("="*60)
        print(f"📊 Total tiros 3PT: {len(player_shots):,}")
        print(f"🎯 Tiros convertidos: {player_shots['SHOT_MADE_FLAG'].sum():,}")
        print(f"📈 Porcentaje 3PT: {player_shots['SHOT_MADE_FLAG'].mean():.1%}")
        
        # Análisis por zona
        zone_stats = player_shots.groupby('SHOT_ZONE_BASIC').agg({
            'SHOT_MADE_FLAG': ['count', 'sum', 'mean'],
            'SHOT_DISTANCE': 'mean'
        }).round(3)
        
        # Flatten columns
        zone_stats.columns = ['Attempts', 'Made', 'Percentage', 'Avg_Distance']
        zone_stats = zone_stats.sort_values('Attempts', ascending=False)
        
        print(f"\n📍 ESTADÍSTICAS POR ZONA:")
        print(zone_stats.to_string())
        
        # Zona más efectiva
        if not zone_stats.empty:
            best_zone = zone_stats.loc[zone_stats['Percentage'].idxmax()]
            most_attempts_zone = zone_stats.iloc[0]
            
            print(f"\n🏆 ZONA MÁS EFECTIVA:")
            print(f"   {zone_stats['Percentage'].idxmax()}: {best_zone['Percentage']:.1%}")
            print(f"   ({best_zone['Made']:.0f}/{best_zone['Attempts']:.0f} tiros)")
            
            print(f"\n🎯 ZONA CON MÁS INTENTOS:")
            print(f"   {zone_stats.index[0]}: {most_attempts_zone['Attempts']:.0f} tiros")
            print(f"   Efectividad: {most_attempts_zone['Percentage']:.1%}")
        
        return zone_stats
    
    def create_shot_chart(self, player_name):
        """Crea mapa de tiros del jugador."""
        if self.shots_df is None:
            print("❌ Primero carga los datos")
            return
        
        player_shots = self.shots_df[self.shots_df['PLAYER_NAME'] == player_name].copy()
        
        if player_shots.empty:
            print(f"❌ No se encontraron datos para {player_name}")
            return
        
        # Crear figura
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'🎯 ANÁLISIS DE TIROS 3PT - {player_name}', fontsize=16, fontweight='bold')
        
        # 1. Mapa de dispersión de tiros
        made_shots = player_shots[player_shots['SHOT_MADE_FLAG'] == 1]
        missed_shots = player_shots[player_shots['SHOT_MADE_FLAG'] == 0]
        
        ax1.scatter(missed_shots['LOC_X'], missed_shots['LOC_Y'], 
                   c='red', alpha=0.6, s=30, label=f'Fallados ({len(missed_shots)})')
        ax1.scatter(made_shots['LOC_X'], made_shots['LOC_Y'], 
                   c='green', alpha=0.8, s=30, label=f'Anotados ({len(made_shots)})')
        
        ax1.set_title('📍 Mapa de Tiros', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Posición X (court)')
        ax1.set_ylabel('Posición Y (court)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Heatmap por zona
        zone_stats = player_shots.groupby('SHOT_ZONE_BASIC').agg({
            'SHOT_MADE_FLAG': ['count', 'mean']
        }).round(3)
        
        zone_stats.columns = ['Attempts', 'Percentage']
        zone_stats = zone_stats.sort_values('Attempts', ascending=True)
        
        # Crear heatmap horizontal
        colors = ['lightcoral' if pct < 0.35 else 'lightblue' if pct < 0.40 else 'lightgreen' 
                 for pct in zone_stats['Percentage']]
        
        bars = ax2.barh(range(len(zone_stats)), zone_stats['Percentage'], color=colors)
        ax2.set_yticks(range(len(zone_stats)))
        ax2.set_yticklabels(zone_stats.index, fontsize=10)
        ax2.set_xlabel('Porcentaje de Acierto')
        ax2.set_title('📊 Efectividad por Zona', fontsize=14, fontweight='bold')
        
        # Agregar valores en barras
        for i, (bar, pct, att) in enumerate(zip(bars, zone_stats['Percentage'], zone_stats['Attempts'])):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{pct:.1%}\n({att:.0f})', ha='left', va='center', fontsize=9)
        
        # Línea de referencia
        ax2.axvline(x=0.36, color='red', linestyle='--', alpha=0.7, label='Liga (~36%)')
        ax2.legend()
        
        plt.tight_layout()
        
        # Guardar
        filename = f"reports/shot_chart_map_{player_name.replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Mapa guardado: {filename}")
        
        plt.show()
    
    def create_shot_heatmap(self, player_name):
        """Crea heatmap de densidad de tiros."""
        if self.shots_df is None:
            print("❌ Primero carga los datos")
            return
        
        player_shots = self.shots_df[self.shots_df['PLAYER_NAME'] == player_name].copy()
        
        if player_shots.empty:
            print(f"❌ No se encontraron datos para {player_name}")
            return
        
        # Crear heatmap
        plt.figure(figsize=(12, 8))
        
        # Heatmap de posiciones
        plt.hexbin(player_shots['LOC_X'], player_shots['LOC_Y'], 
                  C=player_shots['SHOT_MADE_FLAG'], 
                  gridsize=20, cmap='RdYlGn', alpha=0.8)
        
        plt.colorbar(label='Porcentaje de Acierto Promedio')
        plt.title(f'🔥 HEATMAP DE EFECTIVIDAD - {player_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Posición X (court)')
        plt.ylabel('Posición Y (court)')
        
        # Agregar estadísticas
        total_shots = len(player_shots)
        made_shots = player_shots['SHOT_MADE_FLAG'].sum()
        percentage = made_shots / total_shots
        
        stats_text = f"""
        📊 ESTADÍSTICAS:
        • Total tiros: {total_shots:,}
        • Anotados: {made_shots:,}
        • Porcentaje: {percentage:.1%}
        • Distancia promedio: {player_shots['SHOT_DISTANCE'].mean():.1f} ft
        """
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
        
        plt.grid(True, alpha=0.3)
        
        # Guardar
        filename = f"reports/shot_heatmap_{player_name.replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Heatmap guardado: {filename}")
        
        plt.show()
    
    def compare_players(self, player_names):
        """Compara zonas de tiro entre jugadores."""
        if self.shots_df is None:
            print("❌ Primero carga los datos")
            return
        
        print(f"\n⚔️ COMPARACIÓN DE JUGADORES")
        print("="*50)
        
        comparison_data = []
        
        for player in player_names:
            player_shots = self.shots_df[self.shots_df['PLAYER_NAME'] == player]
            
            if not player_shots.empty:
                stats = {
                    'Jugador': player,
                    'Total_Tiros': len(player_shots),
                    'Anotados': player_shots['SHOT_MADE_FLAG'].sum(),
                    'Porcentaje': player_shots['SHOT_MADE_FLAG'].mean(),
                    'Distancia_Promedio': player_shots['SHOT_DISTANCE'].mean()
                }
                comparison_data.append(stats)
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('Porcentaje', ascending=False)
            
            print(comparison_df.to_string(index=False, float_format='%.3f'))
            
            # Visualización comparativa
            plt.figure(figsize=(12, 6))
            
            # Subplot 1: Porcentajes
            plt.subplot(1, 2, 1)
            bars = plt.bar(range(len(comparison_df)), comparison_df['Porcentaje'], 
                          color=['gold', 'silver', 'chocolate'] + ['lightblue'] * (len(comparison_df) - 3))
            plt.xticks(range(len(comparison_df)), 
                      [name.split()[-1] for name in comparison_df['Jugador']], rotation=45)
            plt.ylabel('Porcentaje 3PT')
            plt.title('📊 Comparación Porcentajes')
            
            # Agregar valores
            for bar, pct in zip(bars, comparison_df['Porcentaje']):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                        f'{pct:.1%}', ha='center', va='bottom')
            
            # Subplot 2: Volumen vs Efectividad
            plt.subplot(1, 2, 2)
            plt.scatter(comparison_df['Total_Tiros'], comparison_df['Porcentaje'], 
                       s=100, alpha=0.7, c=range(len(comparison_df)), cmap='viridis')
            
            for i, player in enumerate(comparison_df['Jugador']):
                plt.annotate(player.split()[-1], 
                           (comparison_df.iloc[i]['Total_Tiros'], comparison_df.iloc[i]['Porcentaje']),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            plt.xlabel('Total Tiros 3PT')
            plt.ylabel('Porcentaje 3PT')
            plt.title('🎯 Volumen vs Efectividad')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('reports/player_comparison.png', dpi=300, bbox_inches='tight')
            print("✅ Comparación guardada: player_comparison.png")
            plt.show()
    
    def generate_zones_report(self):
        """Genera reporte completo de zonas."""
        if self.shots_df is None:
            print("❌ Primero carga los datos")
            return
        
        # Top 10 jugadores por volumen
        top_players = self.shots_df.groupby('PLAYER_NAME').agg({
            'SHOT_MADE_FLAG': ['count', 'sum', 'mean']
        }).round(3)
        
        top_players.columns = ['Total_Tiros', 'Anotados', 'Porcentaje']
        top_players = top_players.sort_values('Total_Tiros', ascending=False).head(10)
        
        print(f"\n🏆 TOP 10 JUGADORES POR VOLUMEN DE TIROS 3PT")
        print("="*60)
        print(top_players.to_string())
        
        # Análisis de zonas general
        zone_analysis = self.shots_df.groupby('SHOT_ZONE_BASIC').agg({
            'SHOT_MADE_FLAG': ['count', 'sum', 'mean']
        }).round(3)
        
        zone_analysis.columns = ['Total_Tiros', 'Anotados', 'Porcentaje']
        zone_analysis = zone_analysis.sort_values('Total_Tiros', ascending=False)
        
        print(f"\n📍 ANÁLISIS GENERAL POR ZONAS")
        print("="*40)
        print(zone_analysis.to_string())
        
        # Guardar datos de zonas
        zones_data = []
        for player in top_players.index[:5]:  # Top 5 jugadores
            player_shots = self.shots_df[self.shots_df['PLAYER_NAME'] == player]
            zone_stats = player_shots.groupby('SHOT_ZONE_BASIC')['SHOT_MADE_FLAG'].agg(['count', 'mean']).round(3)
            zone_stats.columns = ['Attempts', 'Percentage']
            
            for zone, stats in zone_stats.iterrows():
                zones_data.append({
                    'Player': player,
                    'Zone': zone,
                    'Attempts': stats['Attempts'],
                    'Percentage': stats['Percentage']
                })
        
        zones_df = pd.DataFrame(zones_data)
        zones_df.to_csv('data/player_3pt_zones.csv', index=False)
        print("✅ Datos de zonas guardados: player_3pt_zones.csv")

def interactive_analyzer():
    """Interfaz interactiva del analizador."""
    analyzer = PlayerShotZonesAnalyzer()
    
    if not analyzer.load_shots_data():
        return
    
    while True:
        print("\n🎯 PLAYER SHOT ZONES ANALYZER - MENÚ")
        print("="*50)
        print("1. 📊 Analizar jugador específico")
        print("2. 🗺️ Crear mapa de tiros")
        print("3. 🔥 Crear heatmap")
        print("4. ⚔️ Comparar jugadores")
        print("5. 📋 Reporte general de zonas")
        print("6. 🚪 Salir")
        
        try:
            choice = input("\n👉 Selecciona (1-6): ").strip()
            
            if choice == '1':
                player = input("🏀 Nombre del jugador: ").strip()
                analyzer.analyze_player_zones(player)
            
            elif choice == '2':
                player = input("🏀 Nombre del jugador: ").strip()
                analyzer.create_shot_chart(player)
            
            elif choice == '3':
                player = input("🏀 Nombre del jugador: ").strip()
                analyzer.create_shot_heatmap(player)
            
            elif choice == '4':
                players_input = input("🏀 Jugadores (separados por comas): ").strip()
                players = [p.strip() for p in players_input.split(',')]
                analyzer.compare_players(players)
            
            elif choice == '5':
                analyzer.generate_zones_report()
            
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
    print("🎯 NBA PLAYER SHOT ZONES ANALYZER")
    print("="*50)
    print("📊 Análisis detallado de zonas de tiro por jugador")
    print()
    
    interactive_analyzer()

if __name__ == "__main__":
    main()
