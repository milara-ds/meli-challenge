import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any, Optional
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from math import ceil
import os
import json
from datetime import datetime

class DataAnalyzer:
    """
    Clase para analizar el dataset de productos de Mercado Libre.
    """
    
    def __init__(self, file_path: str):
        """
        Inicializa el DataAnalyzer.
        
        Args:
            file_path (str): Ruta al archivo CSV del dataset.
        """
        self.file_path = file_path
        self.data = None
        self._setup_logging()
    
    def _setup_logging(self):
        """Configura el sistema de logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def read_dataset(self) -> pd.DataFrame:
        """
        Lee el dataset desde el archivo CSV.
        
        Returns:
            pd.DataFrame: DataFrame con los datos cargados.
        """
        try:
            self.logger.info(f"Leyendo dataset desde {self.file_path}")
            self.data = pd.read_csv(self.file_path)
            self.logger.info(f"Dataset cargado exitosamente. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            self.logger.error(f"Error al leer el dataset: {str(e)}")
            raise
    
    def get_statistical_summary(self) -> Dict:
        """
        Genera un resumen estadístico de precios y cantidades vendidas.
        
        Returns:
            Dict: Diccionario con estadísticas descriptivas.
        """
        if self.data is None:
            self.logger.warning("Dataset no cargado. Llamando a read_dataset()")
            self.read_dataset()
            
        stats = {
            'base_price': {
                'summary': self.data['base_price'].describe().to_dict(),
                'missing_values': self.data['base_price'].isnull().sum()
            },
            'price': {
                'summary': self.data['price'].describe().to_dict(),
                'missing_values': self.data['price'].isnull().sum()
            },
            'sold_quantity': {
                'summary': self.data['sold_quantity'].describe().to_dict(),
                'missing_values': self.data['sold_quantity'].isnull().sum()
            }
        }
        
        self.logger.info("Resumen estadístico generado")
        return stats
    
    def handle_missing_values(self) -> pd.DataFrame:
        """
        Identifica y gestiona valores faltantes o inconsistentes.
        
        Returns:
            pd.DataFrame: DataFrame con los valores faltantes gestionados.
        """
        if self.data is None:
            self.logger.warning("Dataset no cargado. Llamando a read_dataset()")
            self.read_dataset()
            
        # Obtener información sobre valores faltantes
        missing_info = self.data.isnull().sum()
        self.logger.info(f"Valores faltantes por columna:\n{missing_info[missing_info > 0]}")
        
        # Gestionar valores faltantes según el tipo de columna
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        categorical_columns = self.data.select_dtypes(exclude=[np.number]).columns
        
        # Para columnas numéricas, rellenar con la mediana
        self.data[numeric_columns] = self.data[numeric_columns].fillna(self.data[numeric_columns].median())
        
        # Para columnas categóricas, rellenar con 'unknown'
        self.data[categorical_columns] = self.data[categorical_columns].fillna('unknown')
        
        self.logger.info("Valores faltantes gestionados")
        return self.data
    
    def detect_outliers(self, column: str = 'price', method: str = 'iqr') -> Tuple[pd.Series, pd.DataFrame]:
        """
        Detecta outliers en una columna específica usando el método IQR o Z-score.
        
        Args:
            column (str): Nombre de la columna a analizar.
            method (str): Método para detectar outliers ('iqr' o 'zscore').
            
        Returns:
            Tuple[pd.Series, pd.DataFrame]: (máscara de outliers, datos de outliers)
        """
        if self.data is None:
            self.logger.warning("Dataset no cargado. Llamando a read_dataset()")
            self.read_dataset()
            
        if method == 'iqr':
            Q1 = self.data[column].quantile(0.25)
            Q3 = self.data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_mask = (self.data[column] < lower_bound) | (self.data[column] > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((self.data[column] - self.data[column].mean()) / self.data[column].std())
            outliers_mask = z_scores > 3
            
        else:
            raise ValueError("Método debe ser 'iqr' o 'zscore'")
        
        outliers = self.data[outliers_mask]
        self.logger.info(f"Detectados {len(outliers)} outliers usando método {method}")
        
        return outliers_mask, outliers

    def plot_numeric_histograms(self, figsize: Tuple[int, int] = (20, 15)) -> None:
        """
        Crea histogramas para todas las columnas numéricas del dataset usando seaborn.
        Los histogramas se organizan en una matriz de 5 columnas y n filas según sea necesario.
        
        Args:
            figsize (Tuple[int, int]): Tamaño de la figura en pulgadas (ancho, alto)
        """
        if self.data is None:
            self.logger.warning("Dataset no cargado. Llamando a read_dataset()")
            self.read_dataset()
        
        # Obtener columnas numéricas
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        n_cols = len(numeric_columns)
        
        if n_cols == 0:
            self.logger.warning("No se encontraron columnas numéricas en el dataset")
            return
        
        # Configurar el layout de los subplots
        n_cols_per_row = 5
        n_rows = ceil(n_cols / n_cols_per_row)
        
        # Crear la figura y los subplots
        fig, axes = plt.subplots(n_rows, n_cols_per_row, figsize=figsize)
        fig.suptitle('Histogramas de Variables Numéricas', fontsize=16, y=1.02)
        
        # Aplanar el array de axes si es necesario
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plotear los histogramas
        for idx, column in enumerate(numeric_columns):
            row = idx // n_cols_per_row
            col = idx % n_cols_per_row
            
            # Crear el histograma con seaborn
            sns.histplot(
                data=self.data,
                x=column,
                ax=axes[row, col],
                kde=True  # Agregar la curva de densidad
            )
            
            axes[row, col].set_title(f'Distribución de {column}')
            axes[row, col].tick_params(labelrotation=45)
        
        # Ocultar los subplots vacíos
        for idx in range(n_cols, n_rows * n_cols_per_row):
            row = idx // n_cols_per_row
            col = idx % n_cols_per_row
            axes[row, col].set_visible(False)
        
        # Ajustar el layout
        plt.tight_layout()
        self.logger.info(f"Generados {n_cols} histogramas para las columnas numéricas")

    def plot_log_transformed_histograms(self, figsize: Tuple[int, int] = (20, 15)) -> None:
        """
        Crea histogramas para todas las columnas numéricas del dataset aplicando una transformación logarítmica.
        Los histogramas se organizan en una matriz de 5 columnas y n filas según sea necesario.
        
        Args:
            figsize (Tuple[int, int]): Tamaño de la figura en pulgadas (ancho, alto)
        """
        if self.data is None:
            self.logger.warning("Dataset no cargado. Llamando a read_dataset()")
            self.read_dataset()
        
        # Obtener columnas numéricas
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        n_cols = len(numeric_columns)
        
        if n_cols == 0:
            self.logger.warning("No se encontraron columnas numéricas en el dataset")
            return
        
        # Configurar el layout de los subplots
        n_cols_per_row = 5
        n_rows = ceil(n_cols / n_cols_per_row)
        
        # Crear la figura y los subplots
        fig, axes = plt.subplots(n_rows, n_cols_per_row, figsize=figsize)
        fig.suptitle('Histogramas de Variables Numéricas (Transformación Logarítmica)', fontsize=16, y=1.02)
        
        # Aplanar el array de axes si es necesario
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plotear los histogramas
        for idx, column in enumerate(numeric_columns):
            row = idx // n_cols_per_row
            col = idx % n_cols_per_row
            
            # Obtener los datos de la columna
            data = self.data[column]
            
            # Aplicar transformación logarítmica
            # Para manejar valores negativos o cero, añadimos un pequeño valor y tomamos el valor absoluto
            min_positive = data[data > 0].min() if any(data > 0) else 1
            log_data = np.log1p(np.abs(data) + min_positive * 0.01)
            
            # Crear el histograma con seaborn
            sns.histplot(
                data=log_data,
                ax=axes[row, col],
                kde=True  # Agregar la curva de densidad
            )
            
            axes[row, col].set_title(f'Distribución Log de {column}')
            axes[row, col].set_xlabel(f'log({column})')
            axes[row, col].tick_params(labelrotation=45)
            
            # Agregar estadísticas básicas
            stats_text = f'Media: {log_data.mean():.2f}\nStd: {log_data.std():.2f}'
            axes[row, col].text(0.95, 0.95, stats_text,
                              transform=axes[row, col].transAxes,
                              verticalalignment='top',
                              horizontalalignment='right',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Ocultar los subplots vacíos
        for idx in range(n_cols, n_rows * n_cols_per_row):
            row = idx // n_cols_per_row
            col = idx % n_cols_per_row
            axes[row, col].set_visible(False)
        
        # Ajustar el layout
        plt.tight_layout()
        self.logger.info(f"Generados {n_cols} histogramas con transformación logarítmica")

    def plot_boxplots(self, figsize: Tuple[int, int] = (20, 15)) -> None:
        """
        Crea boxplots para todas las columnas numéricas del dataset.
        Los boxplots se organizan en una matriz de 5 columnas y n filas según sea necesario.
        
        Args:
            figsize (Tuple[int, int]): Tamaño de la figura en pulgadas (ancho, alto)
        """
        if self.data is None:
            self.logger.warning("Dataset no cargado. Llamando a read_dataset()")
            self.read_dataset()
        
        # Obtener columnas numéricas
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        n_cols = len(numeric_columns)
        
        if n_cols == 0:
            self.logger.warning("No se encontraron columnas numéricas en el dataset")
            return
        
        # Configurar el layout de los subplots
        n_cols_per_row = 5
        n_rows = ceil(n_cols / n_cols_per_row)
        
        # Crear la figura y los subplots
        fig, axes = plt.subplots(n_rows, n_cols_per_row, figsize=figsize)
        fig.suptitle('Boxplots de Variables Numéricas', fontsize=16, y=1.02)
        
        # Aplanar el array de axes si es necesario
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plotear los boxplots
        for idx, column in enumerate(numeric_columns):
            row = idx // n_cols_per_row
            col = idx % n_cols_per_row
            
            # Crear el boxplot con seaborn
            sns.boxplot(
                data=self.data,
                y=column,
                ax=axes[row, col],
                color='skyblue',
                flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 4}
            )
            """
            # Añadir estadísticas básicas
            stats = self.data[column].describe()
            stats_text = (f'Q1: {stats["25%"]:.2f}\n'
                         f'Q2: {stats["50%"]:.2f}\n'
                         f'Q3: {stats["75%"]:.2f}\n'
                         f'IQR: {stats["75%"] - stats["25%"]:.2f}')
            
            axes[row, col].text(0.95, 0.95, stats_text,
                              transform=axes[row, col].transAxes,
                              verticalalignment='top',
                              horizontalalignment='right',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            axes[row, col].set_title(f'Distribución de {column}')
            axes[row, col].tick_params(labelrotation=45)
            
            # Ajustar los límites del eje y para mejor visualización
            q1 = stats['25%']
            q3 = stats['75%']
            iqr = q3 - q1
            axes[row, col].set_ylim(
                max(stats['min'], q1 - 1.5 * iqr),
                min(stats['max'], q3 + 1.5 * iqr)
            )
            """
        
        # Ocultar los subplots vacíos
        for idx in range(n_cols, n_rows * n_cols_per_row):
            row = idx // n_cols_per_row
            col = idx % n_cols_per_row
            axes[row, col].set_visible(False)
        
        # Ajustar el layout
        plt.tight_layout()
        self.logger.info(f"Generados {n_cols} boxplots para las variables numéricas")

    def plot_boxplots_without_outliers(self, figsize: Tuple[int, int] = (20, 15)) -> None:
        """
        Crea boxplots para todas las columnas numéricas del dataset, removiendo outliers mediante el método IQR.
        Los boxplots se organizan en una matriz de 5 columnas y n filas según sea necesario.
        
        Args:
            figsize (Tuple[int, int]): Tamaño de la figura en pulgadas (ancho, alto)
        """
        if self.data is None:
            self.logger.warning("Dataset no cargado. Llamando a read_dataset()")
            self.read_dataset()
        
        # Obtener columnas numéricas
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        n_cols = len(numeric_columns)
        
        if n_cols == 0:
            self.logger.warning("No se encontraron columnas numéricas en el dataset")
            return
        
        # Configurar el layout de los subplots
        n_cols_per_row = 5
        n_rows = ceil(n_cols / n_cols_per_row)
        
        # Crear la figura y los subplots
        fig, axes = plt.subplots(n_rows, n_cols_per_row, figsize=figsize)
        fig.suptitle('Boxplots de Variables Numéricas (Sin Outliers)', fontsize=16, y=1.02)
        
        # Aplanar el array de axes si es necesario
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Crear una copia del DataFrame para no modificar los datos originales
        data_no_outliers = self.data.copy()
        
        # Plotear los boxplots
        for idx, column in enumerate(numeric_columns):
            row = idx // n_cols_per_row
            col = idx % n_cols_per_row
            
            # Calcular los límites IQR para la columna actual
            Q1 = data_no_outliers[column].quantile(0.25)
            Q3 = data_no_outliers[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Filtrar outliers
            mask = (data_no_outliers[column] >= lower_bound) & (data_no_outliers[column] <= upper_bound)
            clean_data = data_no_outliers[mask]
            
            # Calcular el porcentaje de datos removidos
            total_points = len(data_no_outliers[column])
            points_removed = total_points - len(clean_data[column])
            percent_removed = (points_removed / total_points) * 100
            
            # Crear el boxplot con seaborn
            sns.boxplot(
                data=clean_data,
                y=column,
                ax=axes[row, col],
                color='lightgreen',
                width=0.5
            )
            
            """
            # Añadir estadísticas
            stats = clean_data[column].describe()
            stats_text = (f'Datos removidos: {percent_removed:.1f}%\n'
                         f'Q1: {stats["25%"]:.2f}\n'
                         f'Q2: {stats["50%"]:.2f}\n'
                         f'Q3: {stats["75%"]:.2f}\n'
                         f'IQR: {stats["75%"] - stats["25%"]:.2f}')
            
            axes[row, col].text(0.95, 0.95, stats_text,
                              transform=axes[row, col].transAxes,
                              verticalalignment='top',
                              horizontalalignment='right',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            axes[row, col].set_title(f'Distribución de {column}\n(Sin Outliers)')
            axes[row, col].tick_params(labelrotation=45)
            """
        
        # Ocultar los subplots vacíos
        for idx in range(n_cols, n_rows * n_cols_per_row):
            row = idx // n_cols_per_row
            col = idx % n_cols_per_row
            axes[row, col].set_visible(False)
        
        # Ajustar el layout
        plt.tight_layout()
        self.logger.info(f"Generados {n_cols} boxplots sin outliers para las variables numéricas")

    def plot_temporal_distribution(self, threshold: int = 28, figsize: Tuple[int, int] = (15, 7)) -> None:
        """
        Crea un gráfico de la distribución temporal de registros por fecha.
        
        Args:
            threshold (int): Número mínimo de registros por fecha para ser incluido en el gráfico
            figsize (Tuple[int, int]): Tamaño de la figura en pulgadas (ancho, alto)
        """
        if self.data is None:
            self.logger.warning("Dataset no cargado. Llamando a read_dataset()")
            self.read_dataset()
            
        try:
            # Convertir la columna date_created a datetime usando coerce para manejar valores inválidos
            dates = pd.to_datetime(self.data['date_created'], errors='coerce')
            
            # Contar registros por fecha
            daily_counts = dates.dt.date.value_counts().reset_index()
            daily_counts.columns = ['fecha', 'conteo']
            
            # Convertir la columna fecha a datetime para poder extraer el nombre del día
            daily_counts['fecha'] = pd.to_datetime(daily_counts['fecha'])
            
            # Añadir el nombre del día
            dias = {
                0: 'Lunes', 1: 'Martes', 2: 'Miércoles',
                3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'
            }
            daily_counts['dia'] = daily_counts['fecha'].dt.dayofweek.map(dias)
            
            # Ordenar por fecha
            daily_counts = daily_counts.sort_values('fecha')
            
            # Filtrar por threshold
            daily_counts_filtered = daily_counts[daily_counts['conteo'] >= threshold]
            
            # Crear la figura
            plt.figure(figsize=figsize)
            
            # Crear el gráfico con seaborn
            sns.lineplot(
                data=daily_counts_filtered,
                x='fecha',
                y='conteo',
                color='blue',
                linewidth=2
            )
            
            # Añadir puntos para mejor visualización
            sns.scatterplot(
                data=daily_counts_filtered,
                x='fecha',
                y='conteo',
                color='red',
                s=50,
                alpha=0.5
            )
            
            # Añadir el nombre del día sobre cada punto
            for idx, row in daily_counts_filtered.iterrows():
                plt.annotate(
                    row['dia'],
                    (row['fecha'], row['conteo']),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    alpha=0.7
                )
            
            # Configurar el gráfico
            plt.title('Distribución Temporal de Registros\n(Fechas con más de {} registros)'.format(threshold))
            plt.xlabel('Fecha')
            plt.ylabel('Número de Registros')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Añadir estadísticas en el gráfico
            stats_text = (
                f'Total días: {len(daily_counts_filtered)}\n'
                f'Promedio diario: {daily_counts_filtered["conteo"].mean():.1f}\n'
                f'Máximo diario: {daily_counts_filtered["conteo"].max()}\n'
                f'Mínimo diario: {daily_counts_filtered["conteo"].min()}'
            )
            
            plt.text(0.02, 0.98, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Ajustar el layout
            plt.tight_layout()
            
            # Logging
            self.logger.info(
                f"Gráfico temporal generado. "
                f"Registros totales: {daily_counts['conteo'].sum()}, "
                f"Días mostrados: {len(daily_counts_filtered)}"
            )
            
        except Exception as e:
            self.logger.error(f"Error al generar el gráfico temporal: {str(e)}")
            raise 

    def get_unique_categorical_values(self) -> pd.DataFrame:
        """
        Crea una tabla que muestra los valores únicos presentes en cada columna no numérica.
        
        Returns:
            pd.DataFrame: DataFrame con dos columnas:
                - 'columna': Nombre de la columna no numérica
                - 'valores_unicos': Lista de valores únicos en esa columna
        """
        if self.data is None:
            self.logger.warning("Dataset no cargado. Llamando a read_dataset()")
            self.read_dataset()
            
        try:
            # Obtener columnas no numéricas
            non_numeric_cols = self.data.select_dtypes(exclude=['number']).columns
            
            # Crear lista para almacenar los resultados
            unique_values = []
            
            # Para cada columna no numérica, obtener sus valores únicos
            for col in non_numeric_cols:
                # Obtener valores únicos y eliminar valores nulos
                unique_vals = self.data[col].dropna().unique()
                
                # Convertir todos los valores a string para poder ordenarlos
                unique_vals = [str(val) for val in unique_vals]
                
                # Ordenar los valores convertidos a string
                unique_vals.sort()
                
                unique_values.append({
                    'columna': col,
                    'valores_unicos': unique_vals,
                    'cantidad_valores': len(unique_vals)  # Añadimos contador de valores únicos
                })
            
            # Crear DataFrame con los resultados
            result_df = pd.DataFrame(unique_values)
            
            # Ordenar el DataFrame por cantidad de valores únicos para mejor visualización
            result_df = result_df.sort_values('cantidad_valores', ascending=True)
            
            # Logging
            self.logger.info(
                f"Análisis de valores únicos completado para {len(non_numeric_cols)} columnas no numéricas"
            )
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error al obtener valores únicos de columnas categóricas: {str(e)}")
            raise 

    def remove_rows_by_values(self, column_name: str, values_to_remove: List[Any]) -> None:
        """
        Elimina las filas del dataset donde la columna especificada contiene alguno de los valores de la lista.
        
        Args:
            column_name (str): Nombre de la columna donde buscar los valores
            values_to_remove (List[Any]): Lista de valores a buscar para eliminar las filas
            
        Raises:
            ValueError: Si la columna no existe en el dataset
            ValueError: Si el dataset no está cargado
        """
        if self.data is None:
            raise ValueError("Dataset no cargado. Debe llamar a read_dataset() primero.")
            
        if column_name not in self.data.columns:
            raise ValueError(f"La columna '{column_name}' no existe en el dataset.")
            
        try:
            # Guardar el número de filas original para el logging
            original_rows = len(self.data)
            
            # Crear una máscara booleana que identifica las filas a mantener
            mask = ~self.data[column_name].isin(values_to_remove)
            
            # Aplicar la máscara para filtrar el dataset
            self.data = self.data[mask].reset_index(drop=True)
            
            # Calcular filas eliminadas
            removed_rows = original_rows - len(self.data)
            
            # Logging
            self.logger.info(
                f"Filas eliminadas: {removed_rows} "
                f"({(removed_rows/original_rows)*100:.2f}% del total) "
                f"basado en {len(values_to_remove)} valores en la columna '{column_name}'"
            )
            
            # Si se eliminaron todas las filas, advertir
            if len(self.data) == 0:
                self.logger.warning(
                    "¡ADVERTENCIA! Se han eliminado todas las filas del dataset."
                )
                
        except Exception as e:
            self.logger.error(f"Error al eliminar filas por valores: {str(e)}")
            raise 

    def remove_rows_with_high_missing(self, threshold: float = 0.5) -> None:
        """
        Elimina las filas que tienen más del porcentaje especificado de valores nulos o strings vacíos.
        
        Args:
            threshold (float): Porcentaje máximo permitido de valores faltantes (entre 0 y 1).
                             Por defecto es 0.5 (50%)
                             
        Raises:
            ValueError: Si el threshold no está entre 0 y 1
            ValueError: Si el dataset no está cargado
        """
        if self.data is None:
            raise ValueError("Dataset no cargado. Debe llamar a read_dataset() primero.")
            
        if not 0 <= threshold <= 1:
            raise ValueError("El threshold debe estar entre 0 y 1")
            
        try:
            # Guardar el número de filas original para el logging
            original_rows = len(self.data)
            
            # Crear una copia del DataFrame para no modificar los datos durante el cálculo
            df_check = self.data.copy()
            
            # Reemplazar strings vacíos por NaN para contarlos como valores faltantes
            df_check = df_check.replace(r'^\s*$', np.nan, regex=True)
            
            # Calcular el porcentaje de valores faltantes por fila
            missing_percentage = df_check.isnull().sum(axis=1) / len(df_check.columns)
            
            # Crear máscara para filas a mantener (menos del threshold de valores faltantes)
            mask = missing_percentage <= threshold
            
            # Aplicar la máscara al dataset original
            self.data = self.data[mask].reset_index(drop=True)
            
            # Calcular filas eliminadas
            removed_rows = original_rows - len(self.data)
            
            # Logging
            self.logger.info(
                f"Filas eliminadas por alto porcentaje de valores faltantes: {removed_rows} "
                f"({(removed_rows/original_rows)*100:.2f}% del total). "
                f"Threshold usado: {threshold*100}%"
            )
            
            # Mostrar estadísticas de valores faltantes después de la limpieza
            remaining_missing = self.data.isnull().sum().sum()
            remaining_empty = self.data.eq('').sum().sum()
            
            self.logger.info(
                f"Valores faltantes restantes: {remaining_missing} NaN, "
                f"{remaining_empty} strings vacíos"
            )
            
            # Si se eliminaron todas las filas, advertir
            if len(self.data) == 0:
                self.logger.warning(
                    "¡ADVERTENCIA! Se han eliminado todas las filas del dataset."
                )
                
        except Exception as e:
            self.logger.error(f"Error al eliminar filas con alto porcentaje de valores faltantes: {str(e)}")
            raise 

    def remove_outliers_by_iqr(self, columns: List[str] = None, threshold: float = 1.5) -> None:
        """
        Elimina las filas que contienen outliers basándose en el método IQR.
        Un valor se considera outlier si está fuera del rango:
        [Q1 - threshold*IQR, Q3 + threshold*IQR]
        
        Args:
            columns (List[str], optional): Lista de columnas numéricas a analizar.
                                         Si es None, se usan todas las columnas numéricas.
            threshold (float): Factor multiplicador del IQR para determinar outliers.
                             Por defecto es 1.5 (el valor estándar en boxplots)
                             
        Raises:
            ValueError: Si el dataset no está cargado
            ValueError: Si alguna columna especificada no existe o no es numérica
        """
        if self.data is None:
            raise ValueError("Dataset no cargado. Debe llamar a read_dataset() primero.")
            
        try:
            # Si no se especifican columnas, usar todas las numéricas
            if columns is None:
                numeric_cols = self.data.select_dtypes(include=['number']).columns
                columns = list(numeric_cols)
            else:
                # Verificar que todas las columnas existan y sean numéricas
                for col in columns:
                    if col not in self.data.columns:
                        raise ValueError(f"La columna '{col}' no existe en el dataset")
                    if not np.issubdtype(self.data[col].dtype, np.number):
                        raise ValueError(f"La columna '{col}' no es numérica")
            
            # Guardar el número de filas original para el logging
            original_rows = len(self.data)
            
            # Crear máscara inicial (todos True)
            mask = pd.Series([True] * len(self.data))
            
            # Para cada columna, actualizar la máscara
            outliers_per_column = {}
            for col in columns:
                # Calcular Q1, Q3 e IQR
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Calcular límites
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Actualizar máscara para esta columna
                col_mask = (self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)
                
                # Contar outliers en esta columna
                outliers_count = (~col_mask).sum()
                outliers_per_column[col] = outliers_count
                
                # Actualizar máscara general
                mask &= col_mask
            
            # Aplicar la máscara al dataset
            self.data = self.data[mask].reset_index(drop=True)
            
            # Calcular filas eliminadas
            removed_rows = original_rows - len(self.data)
            
            # Logging
            self.logger.info(
                f"Filas eliminadas por outliers: {removed_rows} "
                f"({(removed_rows/original_rows)*100:.2f}% del total)"
            )
            
            # Logging detallado por columna
            for col, count in outliers_per_column.items():
                self.logger.info(
                    f"Columna '{col}': {count} outliers detectados "
                    f"({(count/original_rows)*100:.2f}% del total)"
                )
            
            # Si se eliminaron todas las filas, advertir
            if len(self.data) == 0:
                self.logger.warning(
                    "¡ADVERTENCIA! Se han eliminado todas las filas del dataset."
                )
                
        except Exception as e:
            self.logger.error(f"Error al eliminar outliers por método IQR: {str(e)}")
            raise 

    def remove_columns(self, columns_to_remove: List[str], validate: bool = True) -> None:
        """
        Elimina las columnas especificadas del dataset.
        
        Args:
            columns_to_remove (List[str]): Lista de nombres de columnas a eliminar
            validate (bool): Si es True, verifica que todas las columnas existan antes de eliminar.
                           Si es False, ignora silenciosamente las columnas que no existen.
                           
        Raises:
            ValueError: Si el dataset no está cargado
            ValueError: Si validate=True y alguna columna no existe
        """
        if self.data is None:
            raise ValueError("Dataset no cargado. Debe llamar a read_dataset() primero.")
            
        try:
            # Verificar columnas existentes si validate=True
            if validate:
                non_existent = [col for col in columns_to_remove if col not in self.data.columns]
                if non_existent:
                    raise ValueError(
                        f"Las siguientes columnas no existen en el dataset: {non_existent}"
                    )
            
            # Obtener solo las columnas que existen en el dataset
            columns_to_remove = [col for col in columns_to_remove if col in self.data.columns]
            
            # Guardar información para logging
            original_columns = len(self.data.columns)
            columns_before = set(self.data.columns)
            
            # Eliminar columnas
            self.data = self.data.drop(columns=columns_to_remove)
            
            # Calcular columnas eliminadas efectivamente
            columns_after = set(self.data.columns)
            removed_columns = columns_before - columns_after
            
            # Logging
            self.logger.info(
                f"Columnas eliminadas: {len(removed_columns)} de {original_columns} "
                f"({(len(removed_columns)/original_columns)*100:.2f}% del total)"
            )
            
            # Logging detallado de las columnas eliminadas
            if removed_columns:
                self.logger.info(f"Columnas eliminadas: {sorted(list(removed_columns))}")
            
            # Advertir si se eliminaron todas las columnas
            if len(self.data.columns) == 0:
                self.logger.warning(
                    "¡ADVERTENCIA! Se han eliminado todas las columnas del dataset."
                )
                
        except Exception as e:
            self.logger.error(f"Error al eliminar columnas: {str(e)}")
            raise 

    def clean_dates_by_frequency(self, date_column: str = 'date_created', min_records: int = 50) -> None:
        """
        Elimina las filas donde la fecha no es válida y aquellas cuya fecha tiene menos registros que el mínimo especificado.
        
        Args:
            date_column (str): Nombre de la columna de fecha a analizar. Por defecto 'date_created'
            min_records (int): Número mínimo de registros que debe tener una fecha para mantenerla.
                             Por defecto 50 registros.
                             
        Raises:
            ValueError: Si el dataset no está cargado
            ValueError: Si la columna especificada no existe
        """
        if self.data is None:
            raise ValueError("Dataset no cargado. Debe llamar a read_dataset() primero.")
            
        if date_column not in self.data.columns:
            raise ValueError(f"La columna '{date_column}' no existe en el dataset")
            
        try:
            # Guardar el número de filas original para el logging
            original_rows = len(self.data)
            
            # Convertir la columna a datetime usando coerce para manejar valores inválidos
            dates = pd.to_datetime(self.data[date_column], errors='coerce')
            
            # Contar cuántas fechas inválidas hay
            invalid_dates = dates.isnull().sum()
            
            # Eliminar filas con fechas inválidas
            self.data = self.data[~dates.isnull()].copy()
            self.data[date_column] = dates[~dates.isnull()]
            
            # Contar registros por fecha
            daily_counts = self.data[date_column].dt.date.value_counts()
            
            # Identificar fechas que cumplen con el mínimo de registros
            valid_dates = daily_counts[daily_counts >= min_records].index
            
            # Crear máscara para filtrar por fechas válidas
            mask = self.data[date_column].dt.date.isin(valid_dates)
            
            # Aplicar el filtro
            self.data = self.data[mask].reset_index(drop=True)
            
            # Calcular estadísticas para el logging
            removed_invalid = invalid_dates
            removed_by_frequency = original_rows - invalid_dates - len(self.data)
            total_removed = original_rows - len(self.data)
            
            # Logging detallado
            self.logger.info(
                f"Limpieza de fechas completada:\n"
                f"- Filas con fechas inválidas eliminadas: {removed_invalid} "
                f"({(removed_invalid/original_rows)*100:.2f}% del total)\n"
                f"- Filas eliminadas por baja frecuencia: {removed_by_frequency} "
                f"({(removed_by_frequency/original_rows)*100:.2f}% del total)\n"
                f"- Total de filas eliminadas: {total_removed} "
                f"({(total_removed/original_rows)*100:.2f}% del total)"
            )
            
            # Información sobre las fechas restantes
            remaining_dates = len(self.data[date_column].dt.date.unique())
            self.logger.info(
                f"Fechas únicas restantes: {remaining_dates}\n"
                f"Promedio de registros por fecha: "
                f"{len(self.data)/remaining_dates:.1f}"
            )
            
            # Si se eliminaron todas las filas, advertir
            if len(self.data) == 0:
                self.logger.warning(
                    "¡ADVERTENCIA! Se han eliminado todas las filas del dataset."
                )
                
        except Exception as e:
            self.logger.error(f"Error al limpiar fechas: {str(e)}")
            raise

    def clean_string_whitespace(self, columns: List[str] = None) -> None:
        """
        Limpia los espacios extra al inicio y final de los valores en columnas de tipo string.
        
        Args:
            columns (List[str], optional): Lista de columnas a limpiar.
                                         Si es None, se limpian todas las columnas de tipo objeto/string.
                             
        Raises:
            ValueError: Si el dataset no está cargado
            ValueError: Si alguna columna especificada no existe
        """
        if self.data is None:
            raise ValueError("Dataset no cargado. Debe llamar a read_dataset() primero.")
            
        try:
            # Si no se especifican columnas, usar todas las de tipo objeto
            if columns is None:
                columns = self.data.select_dtypes(include=['object']).columns
            else:
                # Verificar que todas las columnas existan
                non_existent = [col for col in columns if col not in self.data.columns]
                if non_existent:
                    raise ValueError(f"Las siguientes columnas no existen: {non_existent}")
                
                # Filtrar solo las columnas que son de tipo objeto/string
                columns = [col for col in columns if self.data[col].dtype == 'object']
                if not columns:
                    self.logger.warning("Ninguna de las columnas especificadas es de tipo string/objeto")
                    return
            
            # Diccionario para almacenar estadísticas por columna
            stats = {}
            
            # Procesar cada columna
            for col in columns:
                # Contar valores antes de la limpieza
                original_values = self.data[col].value_counts().to_dict()
                
                # Aplicar strip() a la columna
                self.data[col] = self.data[col].astype(str).str.strip()
                
                # Contar valores después de la limpieza
                cleaned_values = self.data[col].value_counts().to_dict()
                
                # Calcular estadísticas
                stats[col] = {
                    'valores_originales': len(original_values),
                    'valores_unicos_despues': len(cleaned_values),
                    'diferencia': len(original_values) - len(cleaned_values)
                }
            
            # Logging detallado
            self.logger.info(f"Limpieza de espacios completada en {len(columns)} columnas:")
            
            for col, stat in stats.items():
                if stat['diferencia'] > 0:
                    self.logger.info(
                        f"Columna '{col}':\n"
                        f"- Valores únicos originales: {stat['valores_originales']}\n"
                        f"- Valores únicos después de limpieza: {stat['valores_unicos_despues']}\n"
                        f"- Valores unificados por espacios: {stat['diferencia']}"
                    )
                else:
                    self.logger.info(
                        f"Columna '{col}': No se encontraron diferencias después de la limpieza"
                    )
                
        except Exception as e:
            self.logger.error(f"Error al limpiar espacios en blanco: {str(e)}")
            raise

    def save_processed_dataset(self, filename: str = None, include_timestamp: bool = True) -> str:
        """
        Guarda el dataset procesado en la carpeta data/processed.
        
        Args:
            filename (str, optional): Nombre del archivo a guardar. 
                                    Si no se especifica, se usa el nombre del archivo original con sufijo '_processed'
            include_timestamp (bool): Si es True, añade timestamp al nombre del archivo.
                                    Por defecto es True.
                                    
        Returns:
            str: Ruta completa donde se guardó el archivo
            
        Raises:
            ValueError: Si el dataset no está cargado
            ValueError: Si hay error al crear el directorio o guardar el archivo
        """
        if self.data is None:
            raise ValueError("Dataset no cargado. Debe llamar a read_dataset() primero.")
            
        try:
            # Obtener la ruta al directorio raíz del proyecto (2 niveles arriba desde este archivo)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            
            # Crear directorio processed si no existe
            processed_dir = os.path.join(project_root, 'data', 'processed')
            os.makedirs(processed_dir, exist_ok=True)
            
            # Si no se especifica nombre, usar el del archivo original
            if filename is None and hasattr(self, 'filename'):
                # Obtener el nombre base sin extensión
                base_name = os.path.splitext(os.path.basename(self.filename))[0]
                filename = f"{base_name}_processed"
            elif filename is None:
                filename = "processed_dataset"
            
            # Añadir timestamp si se solicita
            if include_timestamp:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{filename}_{timestamp}"
            
            # Asegurar que tenga extensión .csv
            if not filename.endswith('.csv'):
                filename += '.csv'
            
            # Ruta completa del archivo
            output_path = os.path.join(processed_dir, filename)
            
            # Crear metadatos del procesamiento
            metadata = {
                'filas_originales': getattr(self, 'original_rows', 'N/A'),
                'filas_finales': len(self.data),
                'columnas': list(self.data.columns),
                'fecha_procesamiento': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'memoria_usado_mb': self.data.memory_usage(deep=True).sum() / 1024 / 1024
            }
            
            # Guardar metadatos
            metadata_file = os.path.splitext(output_path)[0] + '_metadata.json'
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Guardar el dataset
            self.data.to_csv(output_path, index=False)
            
            # Logging
            self.logger.info(
                f"Dataset guardado exitosamente:\n"
                f"- Archivo: {output_path}\n"
                f"- Filas: {len(self.data)}\n"
                f"- Columnas: {len(self.data.columns)}\n"
                f"- Tamaño en memoria: {metadata['memoria_usado_mb']:.2f} MB\n"
                f"- Metadatos guardados en: {metadata_file}"
            )
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error al guardar el dataset procesado: {str(e)}")
            raise

    def plot_seller_distribution(self, figsize: Tuple[int, int] = (15, 8), top_n: int = 20) -> None:
        """
        Crea visualizaciones de la distribución de registros por seller_id.
        
        Args:
            figsize (Tuple[int, int]): Tamaño de la figura en pulgadas (ancho, alto)
            top_n (int): Cantidad de vendedores top a mostrar en el gráfico de barras
            
        Raises:
            ValueError: Si el dataset no está cargado
            ValueError: Si la columna seller_id no existe en el dataset
        """
        if self.data is None:
            raise ValueError("Dataset no cargado. Debe llamar a read_dataset() primero.")
            
        if 'seller_id' not in self.data.columns:
            raise ValueError("La columna 'seller_id' no existe en el dataset")
            
        try:
            # Calcular la distribución de registros por vendedor
            seller_counts = self.data['seller_id'].value_counts()
            
            # Crear una figura con dos subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # 1. Gráfico de barras para los top N vendedores
            seller_counts.head(top_n).plot(kind='bar', ax=ax1)
            ax1.set_title(f'Top {top_n} Vendedores por Cantidad de Productos')
            ax1.set_xlabel('Seller ID')
            ax1.set_ylabel('Cantidad de Productos')
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. Histograma de la distribución
            seller_counts.plot(kind='hist', bins=50, ax=ax2)
            ax2.set_title('Distribución de Productos por Vendedor')
            ax2.set_xlabel('Cantidad de Productos')
            ax2.set_ylabel('Cantidad de Vendedores')
            
            # Añadir estadísticas en el gráfico
            stats_text = (
                f'Estadísticas:\n'
                f'Total Vendedores: {len(seller_counts):,}\n'
                f'Promedio: {seller_counts.mean():.1f}\n'
                f'Mediana: {seller_counts.median():.1f}\n'
                f'Máximo: {seller_counts.max():,}\n'
                f'Mínimo: {seller_counts.min():,}'
            )
            
            # Añadir texto con estadísticas en el segundo subplot
            ax2.text(0.95, 0.95, stats_text,
                    transform=ax2.transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Ajustar el layout
            plt.tight_layout()
            
            # Logging
            self.logger.info(
                f"Análisis de distribución de vendedores completado:\n"
                f"- Total de vendedores únicos: {len(seller_counts):,}\n"
                f"- Vendedor con más productos: {seller_counts.index[0]} ({seller_counts.max():,} productos)\n"
                f"- Promedio de productos por vendedor: {seller_counts.mean():.1f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error al generar la distribución de vendedores: {str(e)}")
            raise

    def analyze_sales_volume(self, top_n: int = 20, figsize: Tuple[int, int] = (15, 8)) -> None:
        """
        Analiza y visualiza el volumen de ventas por publicación (precio * cantidad vendida).
        
        Args:
            top_n (int): Cantidad de publicaciones top a mostrar en el gráfico
            figsize (Tuple[int, int]): Tamaño de la figura en pulgadas (ancho, alto)
            
        Raises:
            ValueError: Si el dataset no está cargado
            ValueError: Si faltan las columnas necesarias
        """
        if self.data is None:
            raise ValueError("Dataset no cargado. Debe llamar a read_dataset() primero.")
            
        required_columns = ['id', 'price', 'sold_quantity']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Faltan las siguientes columnas requeridas: {missing_columns}")
            
        try:
            # Calcular el volumen de ventas por publicación
            sales_volume = self.data.copy()
            sales_volume['sales_volume'] = sales_volume['price'] * sales_volume['sold_quantity']
            
            # Agrupar por ID y calcular métricas
            sales_analysis = sales_volume.groupby('id').agg({
                'sales_volume': 'sum',
                'sold_quantity': 'sum',
                'price': 'mean'  # Promedio en caso de que haya variaciones de precio
            }).reset_index()
            
            # Ordenar por volumen de ventas y tomar los top N
            sales_analysis = sales_analysis.sort_values('sales_volume', ascending=False)
            top_sales = sales_analysis.head(top_n)
            
            # Crear figura
            plt.figure(figsize=figsize)
            
            # Crear el gráfico de barras
            bars = plt.bar(range(len(top_sales)), top_sales['sales_volume'], color='skyblue')
            
            # Personalizar el gráfico
            plt.title(f'Top {top_n} Publicaciones por Volumen de Ventas', pad=20)
            plt.xlabel('ID de Publicación')
            plt.ylabel('Volumen de Ventas ($)')
            
            # Configurar eje X con los IDs
            plt.xticks(range(len(top_sales)), top_sales['id'], rotation=45, ha='right')
            
            # Añadir etiquetas de valor sobre las barras
            for idx, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'${height:,.0f}',
                        ha='center', va='bottom', rotation=0,
                        fontsize=8)
            
            # Añadir grid para mejor lectura
            plt.grid(True, axis='y', alpha=0.3)
            
            # Añadir estadísticas en el gráfico
            stats_text = (
                f'Estadísticas:\n'
                f'Total: ${sales_analysis["sales_volume"].sum():,.0f}\n'
                f'Promedio: ${sales_analysis["sales_volume"].mean():,.0f}\n'
                f'Mediana: ${sales_analysis["sales_volume"].median():,.0f}'
            )
            
            # Posicionar estadísticas en la esquina superior derecha
            plt.text(0.98, 0.98, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9)
            
            # Ajustar layout para evitar cortes en las etiquetas
            plt.tight_layout()
            
            # Logging
            self.logger.info(
                f"Análisis de volumen de ventas completado:\n"
                f"- ID con mayor volumen: {top_sales.iloc[0]['id']}\n"
                f"- Volumen máximo: ${top_sales.iloc[0]['sales_volume']:,.2f}\n"
                f"- Cantidad vendida: {top_sales.iloc[0]['sold_quantity']:,}\n"
                f"- Precio promedio: ${top_sales.iloc[0]['price']:,.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error al analizar el volumen de ventas: {str(e)}")
            raise

    def plot_price_vs_quantity(self, top_n: int = 30, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Crea un scatter plot de precio vs cantidad vendida para los productos con mayor volumen de ventas.
        
        Args:
            top_n (int): Cantidad de productos top a mostrar en el gráfico
            figsize (Tuple[int, int]): Tamaño de la figura en pulgadas (ancho, alto)
            
        Raises:
            ValueError: Si el dataset no está cargado
            ValueError: Si faltan las columnas necesarias
        """
        if self.data is None:
            raise ValueError("Dataset no cargado. Debe llamar a read_dataset() primero.")
            
        required_columns = ['id', 'price', 'sold_quantity']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Faltan las siguientes columnas requeridas: {missing_columns}")
            
        try:
            # Calcular el volumen de ventas y obtener los top N
            sales_volume = self.data.copy()
            sales_volume['sales_volume'] = sales_volume['price'] * sales_volume['sold_quantity']
            
            # Agrupar por ID y calcular métricas
            sales_analysis = sales_volume.groupby('id').agg({
                'sales_volume': 'sum',
                'sold_quantity': 'sum',
                'price': 'mean'
            }).reset_index()
            
            # Ordenar por volumen de ventas y obtener top N
            top_sales = sales_analysis.nlargest(top_n, 'sales_volume')
            
            # Crear figura
            plt.figure(figsize=figsize)
            
            # Crear scatter plot
            scatter = plt.scatter(top_sales['sold_quantity'], 
                                top_sales['price'],
                                c=top_sales['sales_volume'],  # Color basado en volumen de ventas
                                s=100,  # Tamaño de los puntos
                                alpha=0.6,  # Transparencia
                                cmap='viridis')  # Esquema de color
            
            # Añadir etiquetas para algunos puntos
            for idx, row in top_sales.head(10).iterrows():  # Etiquetar solo los top 10 para evitar sobrecarga
                plt.annotate(f"ID: {row['id']}", 
                           (row['sold_quantity'], row['price']),
                           xytext=(5, 5),
                           textcoords='offset points',
                           fontsize=6.4,  # Reducido en 20%
                           bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
            
            # Personalizar el gráfico
            plt.title(f'Precio vs Cantidad Vendida\n(Top {top_n} productos por volumen de ventas)', pad=20)
            plt.xlabel('Cantidad Vendida')
            plt.ylabel('Precio ($)')
            
            # Añadir colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Volumen de Ventas ($)')
            
            # Añadir grid
            plt.grid(True, alpha=0.3)
            
            # Añadir estadísticas en el gráfico
            stats_text = (
                f'Estadísticas de los top {top_n}:\n'
                f'Precio promedio: ${top_sales["price"].mean():,.2f}\n'
                f'Cantidad promedio: {top_sales["sold_quantity"].mean():,.0f}\n'
                f'Volumen promedio: ${top_sales["sales_volume"].mean():,.2f}'
            )
            
            # Posicionar estadísticas en la esquina superior derecha
            plt.text(0.98, 0.98, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',  # Alinear a la derecha
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9)
            
            # Ajustar layout
            plt.tight_layout()
            
            # Logging
            self.logger.info(
                f"Análisis de precio vs cantidad completado:\n"
                f"- Rango de precios: ${top_sales['price'].min():,.2f} - ${top_sales['price'].max():,.2f}\n"
                f"- Rango de cantidades: {top_sales['sold_quantity'].min():,} - {top_sales['sold_quantity'].max():,}\n"
                f"- ID más vendido: {top_sales.iloc[0]['id']} "
                f"(Volumen: ${top_sales.iloc[0]['sales_volume']:,.2f})"
            )
            
        except Exception as e:
            self.logger.error(f"Error al crear scatter plot de precio vs cantidad: {str(e)}")
            raise

    def plot_sales_volume_distribution(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Crea un boxplot para visualizar la distribución del volumen de ventas por ID.
        
        Args:
            figsize (Tuple[int, int]): Tamaño de la figura en pulgadas (ancho, alto)
            
        Raises:
            ValueError: Si el dataset no está cargado
            ValueError: Si faltan las columnas necesarias
        """
        if self.data is None:
            raise ValueError("Dataset no cargado. Debe llamar a read_dataset() primero.")
            
        required_columns = ['id', 'price', 'sold_quantity']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Faltan las siguientes columnas requeridas: {missing_columns}")
            
        try:
            # Calcular el volumen de ventas
            sales_volume = self.data.copy()
            sales_volume['sales_volume'] = sales_volume['price'] * sales_volume['sold_quantity']
            
            # Agrupar por ID y sumar el volumen de ventas
            sales_by_id = sales_volume.groupby('id')['sales_volume'].sum().reset_index()
            
            # Crear figura
            plt.figure(figsize=figsize)
            
            # Crear boxplot
            box = plt.boxplot(sales_by_id['sales_volume'],
                            patch_artist=True,  # Rellenar el boxplot
                            medianprops=dict(color="black", linewidth=1.5),  # Personalizar la línea de la mediana
                            flierprops=dict(marker='o',  # Personalizar los outliers
                                          markerfacecolor='red',
                                          markersize=4,
                                          alpha=0.5),
                            boxprops=dict(facecolor='lightblue',  # Color de relleno
                                        alpha=0.7))
            
            # Personalizar el gráfico
            plt.title('Distribución del Volumen de Ventas por ID', pad=20)
            plt.ylabel('Volumen de Ventas ($)')
            
            # Rotar eje x para mejor visualización
            plt.xticks([1], ['Volumen de Ventas'])
            
            # Añadir grid
            plt.grid(True, alpha=0.3, axis='y')
            
            # Calcular estadísticas
            stats = sales_by_id['sales_volume'].describe()
            
            # Añadir estadísticas en el gráfico
            stats_text = (
                f'Estadísticas:\n'
                f'Mediana: ${stats["50%"]:,.2f}\n'
                f'Media: ${stats["mean"]:,.2f}\n'
                f'Q1: ${stats["25%"]:,.2f}\n'
                f'Q3: ${stats["75%"]:,.2f}\n'
                f'Máx: ${stats["max"]:,.2f}\n'
                f'Mín: ${stats["min"]:,.2f}'
            )
            
            # Posicionar estadísticas
            plt.text(1.4, stats["75%"], stats_text,
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9)
            
            # Ajustar layout
            plt.tight_layout()
            
            # Logging
            self.logger.info(
                f"Análisis de distribución de volumen de ventas completado:\n"
                f"- Mediana: ${stats['50%']:,.2f}\n"
                f"- Media: ${stats['mean']:,.2f}\n"
                f"- Cantidad de outliers: {len(sales_by_id[sales_by_id['sales_volume'] > stats['75%'] + 1.5 * (stats['75%'] - stats['25%'])])}"
            )
            
        except Exception as e:
            self.logger.error(f"Error al crear boxplot de volumen de ventas: {str(e)}")
            raise

    def plot_log_sales_volume_histogram(self, figsize: Tuple[int, int] = (12, 6), bins: int = 50) -> None:
        """
        Crea un histograma del volumen de ventas con transformación logarítmica.
        
        Args:
            figsize (Tuple[int, int]): Tamaño de la figura en pulgadas (ancho, alto)
            bins (int): Número de bins para el histograma
            
        Raises:
            ValueError: Si el dataset no está cargado
            ValueError: Si faltan las columnas necesarias
        """
        if self.data is None:
            raise ValueError("Dataset no cargado. Debe llamar a read_dataset() primero.")
            
        required_columns = ['id', 'price', 'sold_quantity']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Faltan las siguientes columnas requeridas: {missing_columns}")
            
        try:
            # Calcular el volumen de ventas
            sales_volume = self.data.copy()
            sales_volume['sales_volume'] = sales_volume['price'] * sales_volume['sold_quantity']
            
            # Agrupar por ID y sumar el volumen de ventas
            sales_by_id = sales_volume.groupby('id')['sales_volume'].sum().reset_index()
            
            # Aplicar transformación logarítmica (agregamos 1 para evitar log(0))
            log_sales = np.log1p(sales_by_id['sales_volume'])
            
            # Crear figura
            plt.figure(figsize=figsize)
            
            # Crear histograma
            n, bins_values, patches = plt.hist(log_sales, 
                                             bins=bins,
                                             color='skyblue',
                                             alpha=0.7,
                                             edgecolor='black')
            
            # Personalizar el gráfico
            plt.title('Distribución del Volumen de Ventas (Escala Logarítmica)', pad=20)
            plt.xlabel('Log(Volumen de Ventas + 1)')
            plt.ylabel('Frecuencia')
            
            # Añadir grid
            plt.grid(True, alpha=0.3)
            
            # Calcular estadísticas de los datos transformados
            log_stats = log_sales.describe()
            
            # Calcular estadísticas de los datos originales
            orig_stats = sales_by_id['sales_volume'].describe()
            
            # Añadir estadísticas en el gráfico
            stats_text = (
                f'Estadísticas (Log):\n'
                f'Media: {log_stats["mean"]:.2f}\n'
                f'Mediana: {log_stats["50%"]:.2f}\n'
                f'Desv. Est.: {log_stats["std"]:.2f}\n\n'
                f'Valores Originales:\n'
                f'Media: ${orig_stats["mean"]:,.2f}\n'
                f'Mediana: ${orig_stats["50%"]:,.2f}\n'
                f'Máx: ${orig_stats["max"]:,.2f}'
            )
            
            # Posicionar estadísticas
            plt.text(0.98, 0.98, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9)
            
            # Ajustar layout
            plt.tight_layout()
            
            # Logging
            self.logger.info(
                f"Análisis de histograma log-transformado completado:\n"
                f"- Media (log): {log_stats['mean']:.2f}\n"
                f"- Mediana (log): {log_stats['50%']:.2f}\n"
                f"- Asimetría: {log_sales.skew():.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error al crear histograma de volumen de ventas log-transformado: {str(e)}")
            raise

    def plot_sales_volume_histogram(self, figsize: Tuple[int, int] = (12, 6), bins: int = 50) -> None:
        """
        Crea un histograma del volumen de ventas sin transformación.
        
        Args:
            figsize (Tuple[int, int]): Tamaño de la figura en pulgadas (ancho, alto)
            bins (int): Número de bins para el histograma
            
        Raises:
            ValueError: Si el dataset no está cargado
            ValueError: Si faltan las columnas necesarias
        """
        if self.data is None:
            raise ValueError("Dataset no cargado. Debe llamar a read_dataset() primero.")
            
        required_columns = ['id', 'price', 'sold_quantity']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Faltan las siguientes columnas requeridas: {missing_columns}")
            
        try:
            # Calcular el volumen de ventas
            sales_volume = self.data.copy()
            sales_volume['sales_volume'] = sales_volume['price'] * sales_volume['sold_quantity']
            
            # Agrupar por ID y sumar el volumen de ventas
            sales_by_id = sales_volume.groupby('id')['sales_volume'].sum().reset_index()
            
            # Crear figura
            plt.figure(figsize=figsize)
            
            # Crear histograma
            n, bins_values, patches = plt.hist(sales_by_id['sales_volume'], 
                                             bins=bins,
                                             color='lightgreen',
                                             alpha=0.7,
                                             edgecolor='black')
            
            # Personalizar el gráfico
            plt.title('Distribución del Volumen de Ventas', pad=20)
            plt.xlabel('Volumen de Ventas ($)')
            plt.ylabel('Frecuencia')
            
            # Añadir grid
            plt.grid(True, alpha=0.3)
            
            # Calcular estadísticas
            stats = sales_by_id['sales_volume'].describe()
            
            # Calcular percentiles adicionales para mejor comprensión
            p90 = np.percentile(sales_by_id['sales_volume'], 90)
            p95 = np.percentile(sales_by_id['sales_volume'], 95)
            p99 = np.percentile(sales_by_id['sales_volume'], 99)
            
            # Añadir estadísticas en el gráfico
            stats_text = (
                f'Estadísticas:\n'
                f'Media: ${stats["mean"]:,.2f}\n'
                f'Mediana: ${stats["50%"]:,.2f}\n'
                f'Desv. Est.: ${stats["std"]:,.2f}\n'
                f'Máx: ${stats["max"]:,.2f}\n\n'
                f'Percentiles:\n'
                f'P90: ${p90:,.2f}\n'
                f'P95: ${p95:,.2f}\n'
                f'P99: ${p99:,.2f}'
            )
            
            # Posicionar estadísticas
            plt.text(0.98, 0.98, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9)
            
            # Ajustar layout
            plt.tight_layout()
            
            # Logging
            self.logger.info(
                f"Análisis de histograma de volumen de ventas completado:\n"
                f"- Media: ${stats['mean']:,.2f}\n"
                f"- Mediana: ${stats['50%']:,.2f}\n"
                f"- Asimetría: {sales_by_id['sales_volume'].skew():.2f}\n"
                f"- % de datos sobre P95: {(sales_by_id['sales_volume'] > p95).mean() * 100:.1f}%"
            )
            
        except Exception as e:
            self.logger.error(f"Error al crear histograma de volumen de ventas: {str(e)}")
            raise

    def plot_sales_volume_cdf(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Crea un gráfico de la función de distribución acumulada (CDF) del volumen de ventas.
        
        Args:
            figsize (Tuple[int, int]): Tamaño de la figura en pulgadas (ancho, alto)
            
        Raises:
            ValueError: Si el dataset no está cargado
            ValueError: Si faltan las columnas necesarias
        """
        if self.data is None:
            raise ValueError("Dataset no cargado. Debe llamar a read_dataset() primero.")
            
        required_columns = ['id', 'price', 'sold_quantity']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Faltan las siguientes columnas requeridas: {missing_columns}")
            
        try:
            # Calcular el volumen de ventas
            sales_volume = self.data.copy()
            sales_volume['sales_volume'] = sales_volume['price'] * sales_volume['sold_quantity']
            
            # Agrupar por ID y sumar el volumen de ventas
            sales_by_id = sales_volume.groupby('id')['sales_volume'].sum().reset_index()
            
            # Ordenar los valores para calcular el CDF
            sorted_sales = np.sort(sales_by_id['sales_volume'])
            
            # Calcular los valores del CDF (desde 0 a 1)
            cdf_values = np.arange(1, len(sorted_sales) + 1) / len(sorted_sales)
            
            # Crear figura
            plt.figure(figsize=figsize)
            
            # Crear gráfico CDF
            plt.plot(sorted_sales, cdf_values * 100,
                    color='darkblue',
                    linewidth=2,
                    label='CDF')
            
            # Personalizar el gráfico
            plt.title('Distribución Acumulada del Volumen de Ventas', pad=20)
            plt.xlabel('Volumen de Ventas ($)')
            plt.ylabel('Porcentaje de IDs (%)')
            
            # Añadir grid
            plt.grid(True, alpha=0.3)
            
            # Calcular percentiles importantes
            percentiles = [25, 50, 75, 90, 95, 99]
            percentile_values = np.percentile(sorted_sales, percentiles)
            
            # Añadir líneas verticales para percentiles importantes
            colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
            for percentile, value, color in zip(percentiles, percentile_values, colors):
                plt.axvline(x=value, 
                          color=color,
                          linestyle='--',
                          alpha=0.5,
                          label=f'P{percentile}')
            
            # Añadir leyenda
            plt.legend(loc='lower right')
            
            # Añadir estadísticas en el gráfico
            stats_text = (
                f'Percentiles:\n'
                f'P25: ${percentile_values[0]:,.2f}\n'
                f'P50: ${percentile_values[1]:,.2f}\n'
                f'P75: ${percentile_values[2]:,.2f}\n'
                f'P90: ${percentile_values[3]:,.2f}\n'
                f'P95: ${percentile_values[4]:,.2f}\n'
                f'P99: ${percentile_values[5]:,.2f}'
            )
            
            # Posicionar estadísticas
            plt.text(0.02, 0.98, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9)
            
            # Ajustar layout
            plt.tight_layout()
            
            # Logging
            self.logger.info(
                f"Análisis CDF de volumen de ventas completado:\n"
                f"- Mediana (P50): ${percentile_values[1]:,.2f}\n"
                f"- 90% de los IDs tienen ventas menores a: ${percentile_values[3]:,.2f}\n"
                f"- 99% de los IDs tienen ventas menores a: ${percentile_values[5]:,.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error al crear gráfico CDF de volumen de ventas: {str(e)}")
            raise

    def plot_sold_quantity_cdf(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Crea un gráfico de la función de distribución acumulada (CDF) de la cantidad vendida.
        
        Args:
            figsize (Tuple[int, int]): Tamaño de la figura en pulgadas (ancho, alto)
            
        Raises:
            ValueError: Si el dataset no está cargado
            ValueError: Si faltan las columnas necesarias
        """
        if self.data is None:
            raise ValueError("Dataset no cargado. Debe llamar a read_dataset() primero.")
            
        required_columns = ['id', 'sold_quantity']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Faltan las siguientes columnas requeridas: {missing_columns}")
            
        try:
            # Agrupar por ID y sumar la cantidad vendida
            sales_by_id = self.data.groupby('id')['sold_quantity'].sum().reset_index()
            
            # Ordenar los valores para calcular el CDF
            sorted_quantities = np.sort(sales_by_id['sold_quantity'])
            
            # Calcular los valores del CDF (desde 0 a 1)
            cdf_values = np.arange(1, len(sorted_quantities) + 1) / len(sorted_quantities)
            
            # Crear figura
            plt.figure(figsize=figsize)
            
            # Crear gráfico CDF
            plt.plot(sorted_quantities, cdf_values * 100,
                    color='darkred',  # Cambio de color para distinguirlo del CDF de volumen
                    linewidth=2,
                    label='CDF')
            
            # Personalizar el gráfico
            plt.title('Distribución Acumulada de la Cantidad Vendida', pad=20)
            plt.xlabel('Cantidad Vendida (unidades)')
            plt.ylabel('Porcentaje de IDs (%)')
            
            # Añadir grid
            plt.grid(True, alpha=0.3)
            
            # Calcular percentiles importantes
            percentiles = [25, 50, 75, 90, 95, 99]
            percentile_values = np.percentile(sorted_quantities, percentiles)
            
            # Añadir líneas verticales para percentiles importantes
            colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
            for percentile, value, color in zip(percentiles, percentile_values, colors):
                plt.axvline(x=value, 
                          color=color,
                          linestyle='--',
                          alpha=0.5,
                          label=f'P{percentile}')
            
            # Añadir leyenda
            plt.legend(loc='lower right')
            
            # Añadir estadísticas en el gráfico
            stats_text = (
                f'Percentiles (unidades):\n'
                f'P25: {percentile_values[0]:,.0f}\n'
                f'P50: {percentile_values[1]:,.0f}\n'
                f'P75: {percentile_values[2]:,.0f}\n'
                f'P90: {percentile_values[3]:,.0f}\n'
                f'P95: {percentile_values[4]:,.0f}\n'
                f'P99: {percentile_values[5]:,.0f}'
            )
            
            # Posicionar estadísticas
            plt.text(0.02, 0.98, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9)
            
            # Calcular algunas estadísticas adicionales
            mean_quantity = sales_by_id['sold_quantity'].mean()
            total_quantity = sales_by_id['sold_quantity'].sum()
            zero_sales = (sales_by_id['sold_quantity'] == 0).sum()
            zero_sales_pct = (zero_sales / len(sales_by_id)) * 100
            
            # Añadir estadísticas adicionales
            additional_stats = (
                f'Estadísticas Adicionales:\n'
                f'Media: {mean_quantity:,.1f} unidades\n'
                f'Total vendido: {total_quantity:,.0f} unidades\n'
                f'IDs sin ventas: {zero_sales_pct:.1f}%'
            )
            
            # Posicionar estadísticas adicionales
            plt.text(0.98, 0.98, additional_stats,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9)
            
            # Ajustar layout
            plt.tight_layout()
            
            # Logging
            self.logger.info(
                f"Análisis CDF de cantidad vendida completado:\n"
                f"- Mediana (P50): {percentile_values[1]:,.0f} unidades\n"
                f"- 90% de los IDs tienen ventas menores a: {percentile_values[3]:,.0f} unidades\n"
                f"- Media de unidades vendidas: {mean_quantity:,.1f}\n"
                f"- Porcentaje de IDs sin ventas: {zero_sales_pct:.1f}%"
            )
            
        except Exception as e:
            self.logger.error(f"Error al crear gráfico CDF de cantidad vendida: {str(e)}")
            raise

    def plot_column_cdf(self, column_name: str, figsize: Tuple[int, int] = (12, 6), 
                       title: Optional[str] = None, x_label: Optional[str] = None,
                       is_currency: bool = False) -> None:
        """
        Crea un gráfico de la función de distribución acumulada (CDF) de cualquier columna numérica.
        
        Args:
            column_name (str): Nombre de la columna a analizar
            figsize (Tuple[int, int]): Tamaño de la figura en pulgadas (ancho, alto)
            title (Optional[str]): Título personalizado para el gráfico. Si es None, se genera automáticamente
            x_label (Optional[str]): Etiqueta personalizada para el eje X. Si es None, se usa el nombre de la columna
            is_currency (bool): Indica si la columna representa valores monetarios para formateo
            
        Raises:
            ValueError: Si el dataset no está cargado
            ValueError: Si la columna no existe
            ValueError: Si la columna no es numérica
        """
        if self.data is None:
            raise ValueError("Dataset no cargado. Debe llamar a read_dataset() primero.")
            
        if column_name not in self.data.columns:
            raise ValueError(f"La columna '{column_name}' no existe en el dataset")
            
        if not np.issubdtype(self.data[column_name].dtype, np.number):
            raise ValueError(f"La columna '{column_name}' debe ser numérica")
            
        try:
            # Agrupar por ID y sumar los valores
            data_by_id = self.data.groupby('id')[column_name].sum().reset_index()
            
            # Ordenar los valores para calcular el CDF
            sorted_values = np.sort(data_by_id[column_name])
            
            # Calcular los valores del CDF (desde 0 a 1)
            cdf_values = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            
            # Crear figura
            plt.figure(figsize=figsize)
            
            # Crear gráfico CDF
            plt.plot(sorted_values, cdf_values * 100,
                    color='darkred',
                    linewidth=2,
                    label='CDF')
            
            # Personalizar el gráfico
            plt.title(title or f'Distribución Acumulada de {column_name}', pad=20)
            plt.xlabel(x_label or column_name)
            plt.ylabel('Porcentaje de IDs (%)')
            
            # Añadir grid
            plt.grid(True, alpha=0.3)
            
            # Calcular percentiles importantes
            percentiles = [25, 50, 75, 90, 95, 99]
            percentile_values = np.percentile(sorted_values, percentiles)
            
            # Añadir líneas verticales para percentiles importantes
            colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
            for percentile, value, color in zip(percentiles, percentile_values, colors):
                plt.axvline(x=value, 
                          color=color,
                          linestyle='--',
                          alpha=0.5,
                          label=f'P{percentile}')
            
            # Añadir leyenda
            plt.legend(loc='lower right')
            
            # Función auxiliar para formatear valores
            def format_value(val):
                if is_currency:
                    return f"${val:,.2f}"
                return f"{val:,.0f}" if val.is_integer() else f"{val:,.2f}"
            
            # Añadir estadísticas en el gráfico
            stats_text = (
                f'Percentiles{" ($)" if is_currency else ""}:\n' +
                '\n'.join(f"P{p}: {format_value(v)}" for p, v in zip(percentiles, percentile_values))
            )
            
            # Posicionar estadísticas de percentiles
            plt.text(0.02, 0.98, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9)
            
            # Calcular estadísticas adicionales
            mean_val = data_by_id[column_name].mean()
            total_val = data_by_id[column_name].sum()
            zero_vals = (data_by_id[column_name] == 0).sum()
            zero_vals_pct = (zero_vals / len(data_by_id)) * 100
            
            # Añadir estadísticas adicionales
            additional_stats = (
                f'Estadísticas Adicionales:\n'
                f'Media: {format_value(mean_val)}\n'
                f'Total: {format_value(total_val)}\n'
                f'IDs con valor 0: {zero_vals_pct:.1f}%'
            )
            
            # Posicionar estadísticas adicionales
            plt.text(0.98, 0.98, additional_stats,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9)
            
            # Ajustar layout
            plt.tight_layout()
            
            # Logging
            self.logger.info(
                f"Análisis CDF de {column_name} completado:\n"
                f"- Mediana (P50): {format_value(percentile_values[1])}\n"
                f"- 90% de los IDs tienen valores menores a: {format_value(percentile_values[3])}\n"
                f"- Media: {format_value(mean_val)}\n"
                f"- Porcentaje de IDs con valor 0: {zero_vals_pct:.1f}%"
            )
            
        except Exception as e:
            self.logger.error(f"Error al crear gráfico CDF de {column_name}: {str(e)}")
            raise

    def plot_column_cdf_with_threshold(self, column_name: str, threshold: float,
                                     figsize: Tuple[int, int] = (12, 6),
                                     title: Optional[str] = None, 
                                     x_label: Optional[str] = None,
                                     is_currency: bool = False) -> None:
        """
        Crea un gráfico de la función de distribución acumulada (CDF) de cualquier columna numérica,
        filtrando por un umbral mínimo.
        
        Args:
            column_name (str): Nombre de la columna a analizar
            threshold (float): Valor mínimo para filtrar los IDs
            figsize (Tuple[int, int]): Tamaño de la figura en pulgadas (ancho, alto)
            title (Optional[str]): Título personalizado para el gráfico. Si es None, se genera automáticamente
            x_label (Optional[str]): Etiqueta personalizada para el eje X. Si es None, se usa el nombre de la columna
            is_currency (bool): Indica si la columna representa valores monetarios para formateo
            
        Raises:
            ValueError: Si el dataset no está cargado
            ValueError: Si la columna no existe
            ValueError: Si la columna no es numérica
        """
        if self.data is None:
            raise ValueError("Dataset no cargado. Debe llamar a read_dataset() primero.")
            
        if column_name not in self.data.columns:
            raise ValueError(f"La columna '{column_name}' no existe en el dataset")
            
        if not np.issubdtype(self.data[column_name].dtype, np.number):
            raise ValueError(f"La columna '{column_name}' debe ser numérica")
            
        try:
            # Agrupar por ID y sumar los valores
            data_by_id = self.data.groupby('id')[column_name].sum().reset_index()
            
            # Aplicar el filtro de umbral
            total_ids = len(data_by_id)
            data_filtered = data_by_id[data_by_id[column_name] >= threshold]
            filtered_ids = len(data_filtered)
            
            # Ordenar los valores para calcular el CDF
            sorted_values = np.sort(data_filtered[column_name])
            
            # Calcular los valores del CDF (desde 0 a 1)
            cdf_values = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            
            # Crear figura
            plt.figure(figsize=figsize)
            
            # Crear gráfico CDF
            plt.plot(sorted_values, cdf_values * 100,
                    color='darkred',
                    linewidth=2,
                    label='CDF')
            
            # Personalizar el gráfico
            threshold_text = f" (Umbral >= {threshold:,.0f})" if threshold.is_integer() else f" (Umbral >= {threshold:,.2f})"
            plt.title((title or f'Distribución Acumulada de {column_name}') + threshold_text, pad=20)
            plt.xlabel(x_label or column_name)
            plt.ylabel('Porcentaje de IDs (%)')
            
            # Añadir grid
            plt.grid(True, alpha=0.3)
            
            # Calcular percentiles importantes
            percentiles = [25, 50, 75, 90, 95, 99]
            percentile_values = np.percentile(sorted_values, percentiles)
            
            # Añadir líneas verticales para percentiles importantes
            colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
            for percentile, value, color in zip(percentiles, percentile_values, colors):
                plt.axvline(x=value, 
                          color=color,
                          linestyle='--',
                          alpha=0.5,
                          label=f'P{percentile}')
            
            # Añadir leyenda
            plt.legend(loc='lower right')
            
            # Función auxiliar para formatear valores
            def format_value(val):
                if is_currency:
                    return f"${val:,.2f}"
                return f"{val:,.0f}" if val.is_integer() else f"{val:,.2f}"
            
            # Añadir estadísticas en el gráfico
            stats_text = (
                f'Percentiles{" ($)" if is_currency else ""}:\n' +
                '\n'.join(f"P{p}: {format_value(v)}" for p, v in zip(percentiles, percentile_values))
            )
            
            # Posicionar estadísticas de percentiles
            plt.text(0.02, 0.98, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9)
            
            # Calcular estadísticas adicionales
            mean_val = data_filtered[column_name].mean()
            total_val = data_filtered[column_name].sum()
            filtered_pct = (filtered_ids / total_ids) * 100
            
            # Añadir estadísticas adicionales
            additional_stats = (
                f'Estadísticas Adicionales:\n'
                f'Media: {format_value(mean_val)}\n'
                f'Total: {format_value(total_val)}\n'
                f'IDs sobre umbral: {filtered_pct:.1f}%\n'
                f'({filtered_ids:,} de {total_ids:,} IDs)'
            )
            
            # Posicionar estadísticas adicionales
            plt.text(0.98, 0.98, additional_stats,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9)
            
            # Ajustar layout
            plt.tight_layout()
            
            # Logging
            self.logger.info(
                f"Análisis CDF de {column_name} (umbral >= {format_value(threshold)}) completado:\n"
                f"- Total IDs originales: {total_ids:,}\n"
                f"- IDs sobre umbral: {filtered_ids:,} ({filtered_pct:.1f}%)\n"
                f"- Media sobre umbral: {format_value(mean_val)}\n"
                f"- Total acumulado: {format_value(total_val)}"
            )
            
        except Exception as e:
            self.logger.error(f"Error al crear gráfico CDF de {column_name} con umbral: {str(e)}")
            raise

    def plot_column_cdf_with_range(self, column_name: str, 
                                 lower_threshold: float,
                                 upper_threshold: float,
                                 figsize: Tuple[int, int] = (12, 6),
                                 title: Optional[str] = None, 
                                 x_label: Optional[str] = None,
                                 is_currency: bool = False) -> None:
        """
        Crea un gráfico de la función de distribución acumulada (CDF) de cualquier columna numérica,
        filtrando por un rango de valores.
        
        Args:
            column_name (str): Nombre de la columna a analizar
            lower_threshold (float): Valor mínimo para filtrar los IDs
            upper_threshold (float): Valor máximo para filtrar los IDs
            figsize (Tuple[int, int]): Tamaño de la figura en pulgadas (ancho, alto)
            title (Optional[str]): Título personalizado para el gráfico. Si es None, se genera automáticamente
            x_label (Optional[str]): Etiqueta personalizada para el eje X. Si es None, se usa el nombre de la columna
            is_currency (bool): Indica si la columna representa valores monetarios para formateo
            
        Raises:
            ValueError: Si el dataset no está cargado
            ValueError: Si la columna no existe
            ValueError: Si la columna no es numérica
            ValueError: Si el umbral inferior es mayor que el superior
        """
        if self.data is None:
            raise ValueError("Dataset no cargado. Debe llamar a read_dataset() primero.")
            
        if column_name not in self.data.columns:
            raise ValueError(f"La columna '{column_name}' no existe en el dataset")
            
        if not np.issubdtype(self.data[column_name].dtype, np.number):
            raise ValueError(f"La columna '{column_name}' debe ser numérica")
            
        if lower_threshold > upper_threshold:
            raise ValueError(f"El umbral inferior ({lower_threshold}) no puede ser mayor que el superior ({upper_threshold})")
            
        try:
            # Agrupar por ID y sumar los valores
            data_by_id = self.data.groupby('id')[column_name].sum().reset_index()
            
            # Guardar el total de IDs original
            total_ids = len(data_by_id)
            
            # Aplicar los filtros de umbral
            data_filtered = data_by_id[
                (data_by_id[column_name] >= lower_threshold) & 
                (data_by_id[column_name] <= upper_threshold)
            ]
            filtered_ids = len(data_filtered)
            
            # Calcular IDs por debajo y por encima de los umbrales
            below_threshold = len(data_by_id[data_by_id[column_name] < lower_threshold])
            above_threshold = len(data_by_id[data_by_id[column_name] > upper_threshold])
            
            # Ordenar los valores para calcular el CDF
            sorted_values = np.sort(data_filtered[column_name])
            
            # Calcular los valores del CDF (desde 0 a 1)
            cdf_values = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            
            # Crear figura
            plt.figure(figsize=figsize)
            
            # Crear gráfico CDF
            plt.plot(sorted_values, cdf_values * 100,
                    color='darkred',
                    linewidth=2,
                    label='CDF')
            
            # Función auxiliar para formatear valores
            def format_value(val):
                if is_currency:
                    return f"${val:,.2f}"
                return f"{val:,.0f}" if val.is_integer() else f"{val:,.2f}"
            
            # Personalizar el gráfico
            range_text = f" (Rango: {format_value(lower_threshold)} - {format_value(upper_threshold)})"
            plt.title((title or f'Distribución Acumulada de {column_name}') + range_text, pad=20)
            plt.xlabel(x_label or column_name)
            plt.ylabel('Porcentaje de IDs (%)')
            
            # Añadir grid
            plt.grid(True, alpha=0.3)
            
            # Calcular percentiles importantes
            percentiles = [25, 50, 75, 90, 95, 99]
            percentile_values = np.percentile(sorted_values, percentiles)
            
            # Añadir líneas verticales para percentiles importantes
            colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
            for percentile, value, color in zip(percentiles, percentile_values, colors):
                plt.axvline(x=value, 
                          color=color,
                          linestyle='--',
                          alpha=0.5,
                          label=f'P{percentile}')
            
            # Añadir líneas verticales para los umbrales
            plt.axvline(x=lower_threshold, color='red', linestyle='-', alpha=0.3, label='Umbral Inferior')
            plt.axvline(x=upper_threshold, color='red', linestyle='-', alpha=0.3, label='Umbral Superior')
            
            # Añadir leyenda
            plt.legend(loc='lower right')
            
            # Añadir estadísticas en el gráfico
            stats_text = (
                f'Percentiles{" ($)" if is_currency else ""}:\n' +
                '\n'.join(f"P{p}: {format_value(v)}" for p, v in zip(percentiles, percentile_values))
            )
            
            # Posicionar estadísticas de percentiles
            plt.text(0.02, 0.98, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9)
            
            # Calcular estadísticas adicionales
            mean_val = data_filtered[column_name].mean()
            total_val = data_filtered[column_name].sum()
            filtered_pct = (filtered_ids / total_ids) * 100
            below_pct = (below_threshold / total_ids) * 100
            above_pct = (above_threshold / total_ids) * 100
            
            # Añadir estadísticas adicionales
            additional_stats = (
                f'Estadísticas del Rango:\n'
                f'Media: {format_value(mean_val)}\n'
                f'Total: {format_value(total_val)}\n'
                f'IDs en rango: {filtered_pct:.1f}%\n'
                f'({filtered_ids:,} de {total_ids:,} IDs)\n\n'
                f'IDs por umbral:\n'
                f'< {format_value(lower_threshold)}: {below_pct:.1f}%\n'
                f'> {format_value(upper_threshold)}: {above_pct:.1f}%'
            )
            
            # Posicionar estadísticas adicionales
            plt.text(0.98, 0.98, additional_stats,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9)
            
            # Ajustar layout
            plt.tight_layout()
            
            # Logging
            self.logger.info(
                f"Análisis CDF de {column_name} (rango: {format_value(lower_threshold)} - {format_value(upper_threshold)}) completado:\n"
                f"- Total IDs: {total_ids:,}\n"
                f"- IDs en rango: {filtered_ids:,} ({filtered_pct:.1f}%)\n"
                f"- IDs bajo umbral inferior: {below_threshold:,} ({below_pct:.1f}%)\n"
                f"- IDs sobre umbral superior: {above_threshold:,} ({above_pct:.1f}%)\n"
                f"- Media en rango: {format_value(mean_val)}\n"
                f"- Total acumulado en rango: {format_value(total_val)}"
            )
            
        except Exception as e:
            self.logger.error(f"Error al crear gráfico CDF de {column_name} con rango: {str(e)}")
            raise