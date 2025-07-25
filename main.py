from nba_api.stats.endpoints import teamgamelog, scoreboardv2
from nba_api.stats.static import teams
import pandas as pd
from datetime import datetime

def main(temporada=None, team_name='Los Angeles Lakers'):
    # Obtener el ID del equipo
    team_info = teams.find_teams_by_full_name(team_name)[0]
    team_id = team_info['id']
    
    # Si no se proporciona temporada, usar la última disponible
    if temporada is None:
        temporada = obtener_ultima_temporada()
        if temporada is None:
            temporada = "2023-24"  # fallback
    
    # Obtener los juegos de la temporada
    game_log = teamgamelog.TeamGameLog(team_id=team_id, season=temporada)
    games_df = game_log.get_data_frames()[0]
    
    # Obtener los últimos 20 juegos
    last_20_games = games_df.head(20)
    
    # Mostrar información relevante de cada juego
    print(f"RESULTADOS DE LOS ÚLTIMOS 20 PARTIDOS DE {team_name.upper()} (TEMPORADA {temporada})")
    print("=" * 85)
    print(f"{'FECHA':<12} | {'RIVAL':<15} | {'RESULTADO':<8} | {'PUNTOS':<6} | {'RECORD':<8}")
    print("-" * 85)
    
    for index, game in last_20_games.iterrows():
        fecha = game['GAME_DATE']
        # Extraer el equipo rival del MATCHUP
        if 'vs.' in game['MATCHUP']:
            rival = game['MATCHUP'].split('vs. ')[1]
            local = "vs"
        else:
            rival = game['MATCHUP'].split('@ ')[1]
            local = "@"
        
        resultado = "VICTORIA" if game['WL'] == 'W' else "DERROTA"
        puntos_equipo = game['PTS']
        record = f"{game['W']}-{game['L']}"
        
        print(f"{fecha:<12} | {local} {rival:<12} | {resultado:<8} | {puntos_equipo:<6} | {record:<8}")
    
    # Resumen estadístico
    victorias = len(last_20_games[last_20_games['WL'] == 'W'])
    derrotas = len(last_20_games[last_20_games['WL'] == 'L'])
    puntos_promedio = last_20_games['PTS'].mean()
    puntos_max = last_20_games['PTS'].max()
    puntos_min = last_20_games['PTS'].min()
    
    print("\n" + "=" * 85)
    print(f"RESUMEN DE LOS ÚLTIMOS 20 JUEGOS:")
    print(f"Victorias: {victorias}")
    print(f"Derrotas: {derrotas}")
    print(f"Porcentaje de victorias: {(victorias/20)*100:.1f}%")
    print(f"Promedio de puntos por juego: {puntos_promedio:.1f}")
    print(f"Máximos puntos anotados: {puntos_max}")
    print(f"Mínimos puntos anotados: {puntos_min}")
    print(f"Record final de temporada: {last_20_games.iloc[0]['W']}-{last_20_games.iloc[0]['L']}")


def obtener_ultima_temporada():
    """
    Obtiene la última temporada disponible en la base de datos de la NBA.
    Retorna el año de la temporada más reciente.
    """
    from nba_api.stats.endpoints import leaguegamefinder
    from datetime import datetime
    
    # Obtener el año actual
    año_actual = datetime.now().year
    
    # La temporada NBA típicamente comienza en octubre del año anterior
    # y termina en abril del año siguiente
    # Por ejemplo: temporada 2023-24 va de octubre 2023 a abril 2024
    
    # Intentamos encontrar la última temporada disponible
    # empezando desde el año actual hacia atrás
    for año in range(año_actual, año_actual - 5, -1):
        try:
            temporada = f"{año-1}-{str(año)[2:]}"  # formato: "2023-24"
            
            # Intentamos obtener algunos juegos de esta temporada
            games = leaguegamefinder.LeagueGameFinder(
                season_nullable=temporada,
                season_type_nullable='Regular Season'
            )
            
            df = games.get_data_frames()[0]
            
            if len(df) > 0:
                print(f"Última temporada encontrada en la base de datos: {temporada}")
                print(f"Número de juegos encontrados: {len(df)}")
                
                # Obtener fechas del primer y último juego
                fechas = pd.to_datetime(df['GAME_DATE'])
                fecha_inicio = fechas.min().strftime('%B %d, %Y')
                fecha_fin = fechas.max().strftime('%B %d, %Y')
                
                print(f"Temporada desde: {fecha_inicio}")
                print(f"Temporada hasta: {fecha_fin}")
                
                return temporada
                
        except Exception as e:
            continue
    
    print("No se pudo determinar la última temporada disponible")
    return None


if __name__ == "__main__":
    print("Verificando última temporada disponible...")
    print("=" * 50)
    ultima_temporada = obtener_ultima_temporada()
    print("\n" + "=" * 50)
    
    if ultima_temporada:
        print(f"Usando temporada: {ultima_temporada}")
        print()
        main(ultima_temporada, 'Dallas Mavericks')
    else:
        print("Error: No se pudo obtener información de temporadas")
        print("Usando temporada por defecto: 2023-24")
        main("2023-24", 'Dallas Mavericks')
