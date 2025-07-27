#!/usr/bin/env python3
"""
🏀 NBA ML PIPELINE v2.0 - SISTEMA COMPLETO
==========================================

Pipeline optimizado para análisis predictivo NBA:
• Recolección de datos automatizada
• EDA avanzado con detección de multicolinealidad  
• Entrenamiento ML con ensemble models
• Sistema de predicciones en tiempo real

COMPONENTES:
1. Data Collector - Descarga y prepara datos
2. EDA Analyzer - Análisis exploratorio optimizado
3. ML Trainer - Entrenamiento de modelos v2.0
4. Predictor - Sistema de predicciones
"""

import sys
import os
from pathlib import Path

# Agregar src al path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

try:
    from data_collector import NBADataCollector
    from eda_analyzer import NBAEDAAnalyzer
    from ml_trainer import NBAMLTrainer
    from predictor import NBAGamePredictor
except ImportError as e:
    print(f"❌ Error importando módulos: {e}")
    print("💡 Asegúrate de que todos los archivos estén en la carpeta src/")
    sys.exit(1)

class NBAMLPipeline:
    """Pipeline completo NBA ML v2.0."""
    
    def __init__(self):
        """Inicializa el pipeline."""
        print("🏀 NBA ML PIPELINE v2.0")
        print("="*50)
        print("🚀 Sistema completo de análisis predictivo NBA")
        print("✨ Optimizado con features sin multicolinealidad")
        print()
        
        self.data_collector = None
        self.eda_analyzer = None
        self.ml_trainer = None
        self.predictor = None
    
    def run_data_collection(self):
        """Ejecuta recolección de datos."""
        print("📊 FASE 1: RECOLECCIÓN DE DATOS")
        print("="*40)
        
        self.data_collector = NBADataCollector()
        
        print("¿Qué datos deseas recolectar?")
        print("1. Solo crear dataset ML (si ya tienes datos)")
        print("2. Descargar todo (shots + resultados + dataset)")
        print("3. Saltar (usar datos existentes)")
        
        choice = input("Selecciona (1-3): ").strip()
        
        if choice == '1':
            return self.data_collector.create_ml_dataset()
        elif choice == '2':
            # Pipeline completo
            if self.data_collector.download_shots_data():
                if self.data_collector.download_team_results():
                    return self.data_collector.create_ml_dataset()
            return False
        elif choice == '3':
            print("⏭️ Saltando recolección de datos")
            return True
        else:
            print("❌ Opción no válida")
            return False
    
    def run_eda_analysis(self):
        """Ejecuta análisis EDA v2.0."""
        print("\n📊 FASE 2: ANÁLISIS EDA v2.0")
        print("="*40)
        
        self.eda_analyzer = NBAEDAAnalyzer()
        return self.eda_analyzer.run_complete_analysis()
    
    def run_ml_training(self):
        """Ejecuta entrenamiento ML v2.0."""
        print("\n🤖 FASE 3: ENTRENAMIENTO ML v2.0")
        print("="*40)
        
        self.ml_trainer = NBAMLTrainer()
        return self.ml_trainer.run_complete_training()
    
    def run_prediction_system(self):
        """Ejecuta sistema de predicciones."""
        print("\n🔮 FASE 4: SISTEMA DE PREDICCIONES")
        print("="*40)
        
        from predictor import interactive_predictor
        interactive_predictor()
        return True
    
    def run_player_zones_analysis(self):
        """Ejecuta análisis de zonas de tiro."""
        print("\n🎯 ANÁLISIS DE ZONAS DE TIRO")
        print("="*40)
        
        from player_zones_analyzer import interactive_analyzer
        interactive_analyzer()
        return True
    
    def run_shots_downloader(self):
        """Ejecuta descarga individual de shots."""
        print("\n⬇️ DESCARGA INDIVIDUAL DE SHOTS")
        print("="*40)
        
        from shots_downloader import interactive_downloader
        interactive_downloader()
        return True
    
    def run_complete_pipeline(self):
        """Ejecuta pipeline completo."""
        print("🚀 EJECUTANDO PIPELINE COMPLETO NBA ML v2.0")
        print("="*55)
        
        phases = [
            ("📊 Recolección de Datos", self.run_data_collection),
            ("📊 Análisis EDA v2.0", self.run_eda_analysis),
            ("🤖 Entrenamiento ML v2.0", self.run_ml_training),
            ("🔮 Sistema Predicciones", self.run_prediction_system)
        ]
        
        for phase_name, phase_func in phases:
            print(f"\n{'='*60}")
            print(f"🎯 INICIANDO: {phase_name}")
            print(f"{'='*60}")
            
            try:
                result = phase_func()
                if result is False:
                    print(f"❌ Error en fase: {phase_name}")
                    return False
                print(f"✅ {phase_name} completado")
                
            except KeyboardInterrupt:
                print(f"\n⏸️ Pipeline interrumpido en: {phase_name}")
                return False
            except Exception as e:
                print(f"❌ Error inesperado en {phase_name}: {e}")
                return False
        
        print("\n🎉 PIPELINE COMPLETO FINALIZADO")
        print("="*50)
        print("✅ Datos recolectados y procesados")
        print("✅ EDA v2.0 con multicolinealidad eliminada")
        print("✅ Modelos ML v2.0 entrenados y optimizados")
        print("✅ Sistema de predicciones activo")
        print("\n🏆 ¡Sistema NBA ML v2.0 completamente operativo!")
        
        return True
    
    def show_menu(self):
        """Muestra menú principal."""
        while True:
            print("\n🏀 NBA ML PIPELINE v2.0 - MENÚ PRINCIPAL")
            print("="*55)
            print("1. 🚀 Pipeline completo (recomendado)")
            print("2. 📊 Solo recolección de datos")
            print("3. 📊 Solo análisis EDA v2.0")
            print("4. 🤖 Solo entrenamiento ML v2.0")
            print("5. 🔮 Solo sistema de predicciones")
            print("6. 🎯 Análisis de zonas de tiro")
            print("7. ⬇️ Descarga individual de shots")
            print("8. ℹ️ Información del sistema")
            print("9. 🚪 Salir")
            
            try:
                choice = input("\n👉 Selecciona una opción (1-9): ").strip()
                
                if choice == '1':
                    self.run_complete_pipeline()
                elif choice == '2':
                    self.run_data_collection()
                elif choice == '3':
                    self.run_eda_analysis()
                elif choice == '4':
                    self.run_ml_training()
                elif choice == '5':
                    self.run_prediction_system()
                elif choice == '6':
                    self.run_player_zones_analysis()
                elif choice == '7':
                    self.run_shots_downloader()
                elif choice == '8':
                    self.show_system_info()
                elif choice == '9':
                    print("👋 ¡Hasta luego!")
                    break
                else:
                    print("❌ Opción no válida")
                    
            except KeyboardInterrupt:
                print("\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def show_system_info(self):
        """Muestra información del sistema."""
        print("\nℹ️ INFORMACIÓN DEL SISTEMA NBA ML v2.0")
        print("="*50)
        
        info = """
        🏀 NBA ML PIPELINE v2.0 - Sistema Completo
        ==========================================
        
        📊 COMPONENTES:
        • Data Collector: Descarga datos NBA via API
        • EDA Analyzer: Análisis con detección multicolinealidad
        • ML Trainer: Entrenamiento ensemble models
        • Predictor: Sistema predicciones tiempo real
        • 🆕 Player Zones Analyzer: Análisis zonas de tiro
        • 🆕 Shots Downloader: Descarga individual shots
        
        ✨ CARACTERÍSTICAS v2.0:
        • Features optimizados (25 vs 72 originales)
        • Eliminación automática multicolinealidad
        • Variables de interacción inteligentes
        • Ensemble voting classifier (99.8% accuracy)
        • Validación temporal robusta
        • 🆕 Análisis detallado por jugador
        • 🆕 Mapas de calor y shot charts
        
        🔧 MEJORAS TÉCNICAS:
        • VIF analysis para detectar redundancia
        • Smart feature selection con domain knowledge
        • Hyperparameter optimization
        • Advanced ensemble methods
        • Professional visualization suite
        • 🆕 Interactive shot zone analysis
        • 🆕 Granular data download options
        
        📁 ESTRUCTURA PROYECTO:
        • src/: Código fuente optimizado
        • data/: Datasets NBA (shots, resultados, ML)
        • models/: Modelos entrenados (.joblib)
        • reports/: Análisis y visualizaciones
        
        🎯 USO RECOMENDADO:
        1. Ejecutar pipeline completo (opción 1)
        2. O usar fases individuales según necesidad
        3. Sistema predicciones para análisis en tiempo real
        4. 🆕 Análisis zonas para estudios por jugador
        5. 🆕 Descarga shots para datos específicos
        
        🏆 RESULTADOS ESPERADOS:
        • Accuracy > 85% en modelos individuales
        • Ensemble accuracy ~99.8%
        • ROC-AUC cerca de 1.0
        • Predicciones estables y confiables
        • 🆕 Shot charts y heatmaps profesionales
        • 🆕 Análisis detallado por zonas
        
        ---
        Sistema desarrollado con NBA API, scikit-learn,
        XGBoost, y técnicas avanzadas de ML.
        """
        
        print(info)

def main():
    """Función principal."""
    try:
        pipeline = NBAMLPipeline()
        pipeline.show_menu()
    except Exception as e:
        print(f"❌ Error crítico: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
