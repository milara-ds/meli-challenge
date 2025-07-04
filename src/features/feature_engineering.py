import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import json

class FeatureEngineering:
    """
    Clase para la creación y transformación de características (feature engineering).
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Inicializa la clase FeatureEngineering.
        
        Args:
            data (Optional[pd.DataFrame]): DataFrame con los datos a procesar
        """
        self.data = data.copy() if data is not None else None
        self.logger = logging.getLogger(__name__)
        
    def set_data(self, data: pd.DataFrame) -> None:
        """
        Establece el DataFrame a procesar.
        
        Args:
            data (pd.DataFrame): DataFrame con los datos a procesar
        """
        self.data = data.copy()
        
    def create_binary_sold_feature(self) -> pd.DataFrame:
        """
        Crea una columna binaria 'sold' que indica si el producto tuvo ventas (1) o no (0).
        
        Returns:
            pd.DataFrame: DataFrame con la nueva columna 'sold'
            
        Raises:
            ValueError: Si el DataFrame no está cargado
            ValueError: Si la columna 'sold_quantity' no existe
        """
        if self.data is None:
            raise ValueError("DataFrame no cargado. Debe llamar a set_data() primero.")
            
        if 'sold_quantity' not in self.data.columns:
            raise ValueError("La columna 'sold_quantity' no existe en el DataFrame")
            
        try:
            # Crear la columna binaria
            self.data['sold'] = (self.data['sold_quantity'] > 0).astype(int)
            
            # Logging de información
            total_products = len(self.data)
            sold_products = self.data['sold'].sum()
            sold_percentage = (sold_products / total_products) * 100
            
            self.logger.info(
                f"Columna 'sold' creada exitosamente:\n"
                f"- Total de productos: {total_products:,}\n"
                f"- Productos vendidos: {sold_products:,} ({sold_percentage:.1f}%)\n"
                f"- Productos sin ventas: {total_products - sold_products:,} ({100 - sold_percentage:.1f}%)"
            )
            
            return self.data
            
        except Exception as e:
            self.logger.error(f"Error al crear la columna 'sold': {str(e)}")
            raise
            
    def get_processed_data(self) -> pd.DataFrame:
        """
        Retorna el DataFrame procesado.
        
        Returns:
            pd.DataFrame: DataFrame con las características procesadas
            
        Raises:
            ValueError: Si el DataFrame no está cargado
        """
        if self.data is None:
            raise ValueError("DataFrame no cargado. Debe llamar a set_data() primero.")
            
        return self.data.copy()

    def remove_outliers_iqr(self, data: np.ndarray) -> np.ndarray:
        """
        Elimina outliers usando el método IQR (Rango Intercuartil).
        
        Args:
            data (np.ndarray): Array con los datos a procesar
            
        Returns:
            np.ndarray: Array con los datos sin outliers
        """
        if len(data) == 0:
            return data
            
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        # Si IQR es 0, significa que los datos son muy similares
        if IQR == 0:
            self.logger.warning(f"IQR es 0, los datos son muy similares. No se eliminarán outliers.")
            return data
            
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data[(data >= lower_bound) & (data <= upper_bound)]

    def analyze_distribution_by_sold(self, numeric_column: str, sold_column: str = 'sold',
                                   figsize: Tuple[int, int] = (15, 6),
                                   remove_outliers: bool = True) -> None:
        """
        Analiza la distribución de una variable numérica entre dos grupos definidos por una variable binaria.
        Realiza pruebas de normalidad, paramétricas y no paramétricas, y calcula el tamaño del efecto.
        
        Args:
            numeric_column (str): Nombre de la columna numérica a analizar
            sold_column (str): Nombre de la columna binaria que define los grupos (default: 'sold')
            figsize (Tuple[int, int]): Tamaño de la figura para las visualizaciones
            remove_outliers (bool): Si True, elimina outliers usando el método IQR
        """
        if self.data is None:
            raise ValueError("DataFrame no cargado. Debe llamar a set_data() primero.")
            
        # Verificar columnas requeridas
        required_columns = [sold_column, numeric_column]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Faltan las siguientes columnas requeridas: {missing_columns}")
            
        # Verificar que la columna sea numérica
        if not pd.api.types.is_numeric_dtype(self.data[numeric_column]):
            raise ValueError(f"La columna '{numeric_column}' debe ser numérica")
            
        try:
            # Preparar los datos
            data_copy = self.data.copy()
            data_copy[numeric_column] = pd.to_numeric(data_copy[numeric_column], errors='coerce')
            
            # Separar los grupos
            sold_group_raw = data_copy[data_copy[sold_column] == 1][numeric_column].dropna().values
            not_sold_group_raw = data_copy[data_copy[sold_column] == 0][numeric_column].dropna().values
            
            # Verificar si hay suficientes datos
            if len(sold_group_raw) < 2 or len(not_sold_group_raw) < 2:
                raise ValueError(f"No hay suficientes datos para analizar. Grupo vendidos: {len(sold_group_raw)}, Grupo no vendidos: {len(not_sold_group_raw)}")
            
            # Eliminar outliers si se solicita
            if remove_outliers:
                sold_group = self.remove_outliers_iqr(sold_group_raw)
                not_sold_group = self.remove_outliers_iqr(not_sold_group_raw)
                
                # Reportar outliers removidos
                pct_removed_sold = ((len(sold_group_raw) - len(sold_group)) / len(sold_group_raw)) * 100
                pct_removed_not_sold = ((len(not_sold_group_raw) - len(not_sold_group)) / len(not_sold_group_raw)) * 100
                
                self.logger.info(
                    f"\nOutliers removidos usando método IQR:\n"
                    f"- Grupo vendidos: {len(sold_group_raw) - len(sold_group):,} "
                    f"registros ({pct_removed_sold:.1f}%)\n"
                    f"- Grupo no vendidos: {len(not_sold_group_raw) - len(not_sold_group):,} "
                    f"registros ({pct_removed_not_sold:.1f}%)"
                )
            else:
                sold_group = sold_group_raw
                not_sold_group = not_sold_group_raw
            
            # Crear visualizaciones
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # Boxplot
            data_plot = pd.DataFrame({
                'Grupo': ['No Vendido'] * len(not_sold_group) + ['Vendido'] * len(sold_group),
                numeric_column: np.concatenate([not_sold_group, sold_group])
            })
            data_plot['Grupo'] = pd.Categorical(data_plot['Grupo'], 
                                              categories=['No Vendido', 'Vendido'],
                                              ordered=True)
            
            sns.boxplot(data=data_plot, x='Grupo', y=numeric_column, ax=ax1)
            ax1.set_title(f'Boxplot de {numeric_column}' + (' (sin outliers)' if remove_outliers else ''))
            
            # Density plot
            if np.var(sold_group) > 0:
                sns.kdeplot(data=pd.Series(sold_group), label='Vendidos', ax=ax2, warn_singular=False)
            if np.var(not_sold_group) > 0:
                sns.kdeplot(data=pd.Series(not_sold_group), label='No Vendidos', ax=ax2, warn_singular=False)
            ax2.set_title(f'Distribución de {numeric_column}' + (' (sin outliers)' if remove_outliers else ''))
            ax2.legend()
            
            plt.tight_layout()
            
            # Estadísticas descriptivas
            stats_sold = pd.Series(sold_group).describe()
            stats_not_sold = pd.Series(not_sold_group).describe()
            
            # Prueba de normalidad (Shapiro-Wilk)
            # Tomar muestra si hay más de 5000 datos
            sample_size = min(5000, len(sold_group))
            sold_sample = np.random.choice(sold_group, size=sample_size, replace=False) if len(sold_group) > sample_size else sold_group
            not_sold_sample = np.random.choice(not_sold_group, size=sample_size, replace=False) if len(not_sold_group) > sample_size else not_sold_group
            
            shapiro_sold = stats.shapiro(sold_sample)
            shapiro_not_sold = stats.shapiro(not_sold_sample)
            
            # T-test
            t_stat, t_pvalue = stats.ttest_ind(sold_group, not_sold_group)
            
            # Mann-Whitney U test
            u_stat, u_pvalue = stats.mannwhitneyu(sold_group, not_sold_group, alternative='two-sided')
            
            # Tamaño del efecto (Cohen's d)
            d = float((np.mean(sold_group) - np.mean(not_sold_group)) / np.sqrt(
                ((len(sold_group) - 1) * np.var(sold_group) + 
                 (len(not_sold_group) - 1) * np.var(not_sold_group)) / 
                (len(sold_group) + len(not_sold_group) - 2)
            ))
            
            # Interpretación de resultados
            interpretations = []
            
            # Interpretar normalidad
            sold_normal = shapiro_sold[1] > 0.05
            not_sold_normal = shapiro_not_sold[1] > 0.05
            
            interpretations.append(
                f"Test de Normalidad (Shapiro-Wilk):\n"
                f"- Grupo vendidos: {'Normal' if sold_normal else 'No normal'} "
                f"(p={shapiro_sold[1]:.4f})\n"
                f"- Grupo no vendidos: {'Normal' if not_sold_normal else 'No normal'} "
                f"(p={shapiro_not_sold[1]:.4f})"
            )
            
            # Interpretar t-test
            t_significant = t_pvalue < 0.05
            interpretations.append(
                f"T-test (prueba paramétrica):\n"
                f"- {'Hay' if t_significant else 'No hay'} diferencia significativa "
                f"(p={t_pvalue:.4f})"
            )
            
            # Interpretar Mann-Whitney
            u_significant = u_pvalue < 0.05
            interpretations.append(
                f"Mann-Whitney U test (prueba no paramétrica):\n"
                f"- {'Hay' if u_significant else 'No hay'} diferencia significativa "
                f"(p={u_pvalue:.4f})"
            )
            
            # Interpretar tamaño del efecto
            effect_size_interpretation = ""
            if abs(d) < 0.2:
                effect_size_interpretation = "efecto insignificante"
            elif abs(d) < 0.5:
                effect_size_interpretation = "efecto pequeño"
            elif abs(d) < 0.8:
                effect_size_interpretation = "efecto mediano"
            else:
                effect_size_interpretation = "efecto grande"
            
            interpretations.append(
                f"Tamaño del efecto (Cohen's d):\n"
                f"- d={d:.3f} ({effect_size_interpretation})"
            )
            
            # Logging de resultados
            self.logger.info(
                f"\nAnálisis de distribución de {numeric_column} por grupo completado:\n"
                f"Estadísticas descriptivas{' (sin outliers)' if remove_outliers else ''}:\n"
                f"- Grupo vendidos (n={len(sold_group):,}):\n"
                f"  Media: {stats_sold['mean']:.2f}\n"
                f"  Mediana: {stats_sold['50%']:.2f}\n"
                f"  Desv. Est.: {stats_sold['std']:.2f}\n"
                f"- Grupo no vendidos (n={len(not_sold_group):,}):\n"
                f"  Media: {stats_not_sold['mean']:.2f}\n"
                f"  Mediana: {stats_not_sold['50%']:.2f}\n"
                f"  Desv. Est.: {stats_not_sold['std']:.2f}\n"
                f"\nResultados de las pruebas estadísticas:\n"
                f"{chr(10).join(interpretations)}"
            )
            
        except Exception as e:
            self.logger.error(f"Error al analizar distribución de {numeric_column}: {str(e)}")
            raise

    def add_post_days(self) -> None:
        """
        Crea una nueva columna 'post_days' que representa los días transcurridos
        desde la creación del post hasta la fecha más reciente en el dataset.
        
        La columna se calcula como la diferencia entre la fecha máxima del dataset
        y la fecha de creación de cada publicación (date_created).
        """
        if self.data is None:
            raise ValueError("DataFrame no cargado. Debe llamar a set_data() primero.")
            
        if 'date_created' not in self.data.columns:
            raise ValueError("La columna 'date_created' no existe en el DataFrame")
            
        try:
            # Convertir date_created a datetime si no lo es
            self.data['date_created'] = pd.to_datetime(self.data['date_created'])
            
            # Obtener la fecha máxima del dataset
            max_date = self.data['date_created'].max()
            
            # Calcular los días transcurridos
            self.data['post_days'] = (max_date - self.data['date_created']).dt.days
            
            self.logger.info(
                f"Columna 'post_days' creada exitosamente.\n"
                f"Rango de días: {self.data['post_days'].min():.0f} a {self.data['post_days'].max():.0f} días\n"
                f"Promedio de días: {self.data['post_days'].mean():.1f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error al crear la columna 'post_days': {str(e)}")
            raise

    def add_weekday(self) -> None:
        """
        Crea una nueva columna 'weekday' que representa el día de la semana
        en formato numérico (1=Lunes, 2=Martes, ..., 7=Domingo) basado en
        la columna date_created.
        """
        if self.data is None:
            raise ValueError("DataFrame no cargado. Debe llamar a set_data() primero.")
            
        if 'date_created' not in self.data.columns:
            raise ValueError("La columna 'date_created' no existe en el DataFrame")
            
        try:
            # Convertir date_created a datetime si no lo es
            self.data['date_created'] = pd.to_datetime(self.data['date_created'])
            
            # Obtener el día de la semana (por defecto 0=Lunes, 6=Domingo)
            # Sumamos 1 para tener 1=Lunes, 7=Domingo
            self.data['weekday'] = self.data['date_created'].dt.weekday + 1
            
            # Calcular distribución de días
            day_distribution = self.data['weekday'].value_counts().sort_index()
            
            # Mapeo de números a nombres de días
            days = {1: 'Lunes', 2: 'Martes', 3: 'Miércoles', 
                   4: 'Jueves', 5: 'Viernes', 6: 'Sábado', 7: 'Domingo'}
            
            # Logging de la distribución
            distribution_text = "\nDistribución por día de la semana:"
            for day_num, count in day_distribution.items():
                percentage = (count / len(self.data)) * 100
                distribution_text += f"\n- {days[day_num]}: {count:,} ({percentage:.1f}%)"
            
            self.logger.info(
                f"Columna 'weekday' creada exitosamente."
                f"{distribution_text}"
            )
            
        except Exception as e:
            self.logger.error(f"Error al crear la columna 'weekday': {str(e)}")
            raise

    def add_is_discount(self) -> None:
        """
        Crea una nueva columna binaria 'is_discount' que es 1 cuando el precio
        actual (price) es menor que el precio base (base_price), y 0 en caso contrario.
        """
        if self.data is None:
            raise ValueError("DataFrame no cargado. Debe llamar a set_data() primero.")
            
        required_columns = ['price', 'base_price']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Faltan las siguientes columnas requeridas: {missing_columns}")
            
        try:
            # Asegurar que las columnas sean numéricas
            self.data['price'] = pd.to_numeric(self.data['price'])
            self.data['base_price'] = pd.to_numeric(self.data['base_price'])
            
            # Crear columna is_discount
            self.data['is_discount'] = (self.data['price'] < self.data['base_price']).astype(int)
            
            # Calcular estadísticas
            total_items = len(self.data)
            items_with_discount = self.data['is_discount'].sum()
            discount_percentage = (items_with_discount / total_items) * 100
            
            # Calcular estadísticas de descuento para items con descuento
            discount_mask = self.data['is_discount'] == 1
            if items_with_discount > 0:
                discount_amounts = self.data.loc[discount_mask, 'base_price'] - self.data.loc[discount_mask, 'price']
                discount_percentages = (discount_amounts / self.data.loc[discount_mask, 'base_price']) * 100
                
                avg_discount = discount_amounts.mean()
                avg_discount_percentage = discount_percentages.mean()
                max_discount = discount_amounts.max()
                max_discount_percentage = discount_percentages.max()
                
                self.logger.info(
                    f"Columna 'is_discount' creada exitosamente.\n"
                    f"Items con descuento: {items_with_discount:,} de {total_items:,} ({discount_percentage:.1f}%)\n"
                    f"Estadísticas de descuentos:\n"
                    f"- Descuento promedio: ${avg_discount:.2f} ({avg_discount_percentage:.1f}%)\n"
                    f"- Descuento máximo: ${max_discount:.2f} ({max_discount_percentage:.1f}%)"
                )
            else:
                self.logger.info(
                    f"Columna 'is_discount' creada exitosamente.\n"
                    f"No se encontraron items con descuento en el dataset"
                )
            
        except Exception as e:
            self.logger.error(f"Error al crear la columna 'is_discount': {str(e)}")
            raise

    def add_title_length(self) -> None:
        """
        Crea una nueva columna 'len_title' que contiene la longitud (número de caracteres)
        del título de cada item.
        """
        if self.data is None:
            raise ValueError("DataFrame no cargado. Debe llamar a set_data() primero.")
            
        if 'title' not in self.data.columns:
            raise ValueError("La columna 'title' no existe en el DataFrame")
            
        try:
            # Asegurar que la columna title sea de tipo string
            self.data['title'] = self.data['title'].astype(str)
            
            # Crear columna len_title
            self.data['len_title'] = self.data['title'].str.len()
            
            # Calcular estadísticas
            stats = self.data['len_title'].describe()
            
            # Calcular distribución por rangos
            bins = [0, 25, 50, 75, 100, float('inf')]
            labels = ['1-25', '26-50', '51-75', '76-100', '>100']
            length_distribution = pd.cut(self.data['len_title'], bins=bins, labels=labels, right=False)
            distribution_counts = length_distribution.value_counts().sort_index()
            
            # Preparar texto de distribución
            distribution_text = "\nDistribución por longitud de título:"
            for length_range, count in distribution_counts.items():
                percentage = (count / len(self.data)) * 100
                distribution_text += f"\n- {length_range} caracteres: {count:,} ({percentage:.1f}%)"
            
            self.logger.info(
                f"Columna 'len_title' creada exitosamente.\n"
                f"Estadísticas de longitud de títulos:\n"
                f"- Mínimo: {stats['min']:.0f} caracteres\n"
                f"- Promedio: {stats['mean']:.1f} caracteres\n"
                f"- Mediana: {stats['50%']:.0f} caracteres\n"
                f"- Máximo: {stats['max']:.0f} caracteres"
                f"{distribution_text}"
            )
            
        except Exception as e:
            self.logger.error(f"Error al crear la columna 'len_title': {str(e)}")
            raise

    def add_num_tags(self) -> None:
        """
        Crea una nueva columna 'num_tags' que contiene el número de tags
        asociados a cada item.
        """
        if self.data is None:
            raise ValueError("DataFrame no cargado. Debe llamar a set_data() primero.")
            
        if 'tags' not in self.data.columns:
            raise ValueError("La columna 'tags' no existe en el DataFrame")
            
        try:
            # Crear columna num_tags
            # Asumimos que tags es una lista. Si es string, intentamos evaluarlo como lista
            def count_tags(tags):
                if isinstance(tags, str):
                    try:
                        # Intentar convertir string a lista
                        tags_list = eval(tags)
                        if isinstance(tags_list, list):
                            return len(tags_list)
                    except:
                        return 0
                elif isinstance(tags, list):
                    return len(tags)
                return 0
            
            self.data['num_tags'] = self.data['tags'].apply(count_tags)
            
            # Calcular estadísticas
            stats = self.data['num_tags'].describe()
            
            # Calcular distribución
            value_counts = self.data['num_tags'].value_counts().sort_index()
            
            # Preparar texto de distribución
            distribution_text = "\nDistribución de número de tags:"
            for num_tags, count in value_counts.items():
                percentage = (count / len(self.data)) * 100
                distribution_text += f"\n- {num_tags} tags: {count:,} ({percentage:.1f}%)"
            
            self.logger.info(
                f"Columna 'num_tags' creada exitosamente.\n"
                f"Estadísticas de número de tags:\n"
                f"- Mínimo: {stats['min']:.0f}\n"
                f"- Promedio: {stats['mean']:.1f}\n"
                f"- Mediana: {stats['50%']:.0f}\n"
                f"- Máximo: {stats['max']:.0f}"
                f"{distribution_text}"
            )
            
        except Exception as e:
            self.logger.error(f"Error al crear la columna 'num_tags': {str(e)}")
            raise

    def add_num_variations(self) -> None:
        """
        Crea una nueva columna 'num_variations' que contiene el número de variaciones
        disponibles para cada item.
        """
        if self.data is None:
            raise ValueError("DataFrame no cargado. Debe llamar a set_data() primero.")
            
        if 'variations' not in self.data.columns:
            raise ValueError("La columna 'variations' no existe en el DataFrame")
            
        try:
            # Crear columna num_variations
            # Asumimos que variations es una lista. Si es string, intentamos evaluarlo como lista
            def count_variations(variations):
                if isinstance(variations, str):
                    try:
                        # Intentar convertir string a lista
                        variations_list = eval(variations)
                        if isinstance(variations_list, list):
                            return len(variations_list)
                    except:
                        return 0
                elif isinstance(variations, list):
                    return len(variations)
                return 0
            
            self.data['num_variations'] = self.data['variations'].apply(count_variations)
            
            # Calcular estadísticas
            stats = self.data['num_variations'].describe()
            
            # Calcular distribución
            value_counts = self.data['num_variations'].value_counts().sort_index()
            
            # Preparar texto de distribución
            distribution_text = "\nDistribución de número de variaciones:"
            for num_variations, count in value_counts.items():
                percentage = (count / len(self.data)) * 100
                distribution_text += f"\n- {num_variations} variaciones: {count:,} ({percentage:.1f}%)"
            
            # Calcular porcentaje de items con y sin variaciones
            items_with_variations = (self.data['num_variations'] > 0).sum()
            total_items = len(self.data)
            percentage_with_variations = (items_with_variations / total_items) * 100
            
            self.logger.info(
                f"Columna 'num_variations' creada exitosamente.\n"
                f"Items con variaciones: {items_with_variations:,} de {total_items:,} ({percentage_with_variations:.1f}%)\n"
                f"\nEstadísticas de número de variaciones:\n"
                f"- Mínimo: {stats['min']:.0f}\n"
                f"- Promedio: {stats['mean']:.1f}\n"
                f"- Mediana: {stats['50%']:.0f}\n"
                f"- Máximo: {stats['max']:.0f}"
                f"{distribution_text}"
            )
            
        except Exception as e:
            self.logger.error(f"Error al crear la columna 'num_variations': {str(e)}")
            raise

    def add_num_pictures(self) -> None:
        """
        Crea una nueva columna 'num_pictures' que contiene el número de imágenes
        disponibles para cada item.
        """
        if self.data is None:
            raise ValueError("DataFrame no cargado. Debe llamar a set_data() primero.")
            
        if 'pictures' not in self.data.columns:
            raise ValueError("La columna 'pictures' no existe en el DataFrame")
            
        try:
            # Crear columna num_pictures
            # Asumimos que pictures es una lista. Si es string, intentamos evaluarlo como lista
            def count_pictures(pictures):
                if isinstance(pictures, str):
                    try:
                        # Intentar convertir string a lista
                        pictures_list = eval(pictures)
                        if isinstance(pictures_list, list):
                            return len(pictures_list)
                    except:
                        return 0
                elif isinstance(pictures, list):
                    return len(pictures)
                return 0
            
            self.data['num_pictures'] = self.data['pictures'].apply(count_pictures)
            
            # Calcular estadísticas
            stats = self.data['num_pictures'].describe()
            
            # Calcular distribución
            value_counts = self.data['num_pictures'].value_counts().sort_index()
            
            # Preparar texto de distribución
            distribution_text = "\nDistribución de número de imágenes:"
            for num_pictures, count in value_counts.items():
                percentage = (count / len(self.data)) * 100
                distribution_text += f"\n- {num_pictures} imágenes: {count:,} ({percentage:.1f}%)"
            
            # Calcular porcentaje de items con y sin imágenes
            items_with_pictures = (self.data['num_pictures'] > 0).sum()
            total_items = len(self.data)
            percentage_with_pictures = (items_with_pictures / total_items) * 100
            
            # Calcular rangos de imágenes
            bins = [0, 1, 3, 5, 10, float('inf')]
            labels = ['0', '1-3', '4-5', '6-10', '>10']
            range_distribution = pd.cut(self.data['num_pictures'], bins=bins, labels=labels, right=False)
            range_counts = range_distribution.value_counts().sort_index()
            
            # Preparar texto de distribución por rangos
            range_text = "\nDistribución por rangos de imágenes:"
            for range_label, count in range_counts.items():
                percentage = (count / total_items) * 100
                range_text += f"\n- {range_label} imágenes: {count:,} ({percentage:.1f}%)"
            
            self.logger.info(
                f"Columna 'num_pictures' creada exitosamente.\n"
                f"Items con imágenes: {items_with_pictures:,} de {total_items:,} ({percentage_with_pictures:.1f}%)\n"
                f"\nEstadísticas de número de imágenes:\n"
                f"- Mínimo: {stats['min']:.0f}\n"
                f"- Promedio: {stats['mean']:.1f}\n"
                f"- Mediana: {stats['50%']:.0f}\n"
                f"- Máximo: {stats['max']:.0f}"
                f"{distribution_text}"
                f"{range_text}"
            )
            
        except Exception as e:
            self.logger.error(f"Error al crear la columna 'num_pictures': {str(e)}")
            raise

    def add_categorical_encoding(self, column: str) -> None:
        """
        Crea una nueva columna con el sufijo '_cat' que contiene una codificación
        numérica secuencial de los valores únicos en la columna original.
        
        Args:
            column (str): Nombre de la columna a codificar
        
        Example:
            Si la columna 'color' tiene los valores ['rojo', 'azul', 'verde'],
            la nueva columna 'color_cat' tendrá [1, 2, 3] respectivamente.
        """
        if self.data is None:
            raise ValueError("DataFrame no cargado. Debe llamar a set_data() primero.")
            
        if column not in self.data.columns:
            raise ValueError(f"La columna '{column}' no existe en el DataFrame")
            
        try:
            # Crear el nombre de la nueva columna
            new_column = f"{column}_cat"
            
            # Obtener valores únicos ordenados y crear el mapeo
            unique_values = self.data[column].unique()
            unique_values.sort()  # Ordenar para asegurar consistencia
            value_to_number = {value: i+1 for i, value in enumerate(unique_values)}
            
            # Crear la nueva columna
            self.data[new_column] = self.data[column].map(value_to_number)
            
            # Preparar el mapeo para el logging
            mapping_text = "\nMapeo de valores:"
            for value, number in value_to_number.items():
                count = (self.data[column] == value).sum()
                percentage = (count / len(self.data)) * 100
                mapping_text += f"\n- {value} → {number}: {count:,} ocurrencias ({percentage:.1f}%)"
            
            self.logger.info(
                f"Columna '{new_column}' creada exitosamente.\n"
                f"Valores únicos encontrados: {len(value_to_number)}"
                f"{mapping_text}"
            )
            
        except Exception as e:
            self.logger.error(f"Error al crear la columna '{column}_cat': {str(e)}")
            raise

    def add_tag_indicator(self, tag_value: str) -> None:
        """
        Crea una nueva columna binaria 'tags_{valor}' que indica si el valor específico
        existe en la lista de tags de cada item.
        
        Args:
            tag_value (str): Valor a buscar en la lista de tags
            
        Example:
            Si tag_value es 'best_seller', creará una columna 'tags_best_seller'
            con 1 si el item tiene ese tag y 0 si no lo tiene.
        """
        if self.data is None:
            raise ValueError("DataFrame no cargado. Debe llamar a set_data() primero.")
            
        if 'tags' not in self.data.columns:
            raise ValueError("La columna 'tags' no existe en el DataFrame")
            
        try:
            # Crear el nombre de la nueva columna (reemplazar espacios por _)
            new_column = f"tags_{tag_value.lower().replace(' ', '_')}"
            
            # Función para verificar si el tag existe en la lista
            def has_tag(tags):
                if isinstance(tags, str):
                    try:
                        # Intentar convertir string a lista
                        tags_list = eval(tags)
                        if isinstance(tags_list, list):
                            return int(tag_value in tags_list)
                    except:
                        return 0
                elif isinstance(tags, list):
                    return int(tag_value in tags)
                return 0
            
            # Crear la nueva columna
            self.data[new_column] = self.data['tags'].apply(has_tag)
            
            # Calcular estadísticas
            total_items = len(self.data)
            items_with_tag = self.data[new_column].sum()
            percentage_with_tag = (items_with_tag / total_items) * 100
            
            self.logger.info(
                f"Columna '{new_column}' creada exitosamente.\n"
                f"Items con el tag '{tag_value}': {items_with_tag:,} de {total_items:,} ({percentage_with_tag:.1f}%)"
            )
            
        except Exception as e:
            self.logger.error(f"Error al crear la columna '{new_column}': {str(e)}")
            raise

    def remove_columns(self, columns_to_remove: List[str]) -> None:
        """
        Elimina las columnas especificadas del DataFrame.
        
        Args:
            columns_to_remove (List[str]): Lista con los nombres de las columnas a eliminar
            
        Example:
            remove_columns(['columna1', 'columna2', 'columna3'])
        """
        if self.data is None:
            raise ValueError("DataFrame no cargado. Debe llamar a set_data() primero.")
            
        if not isinstance(columns_to_remove, list):
            raise ValueError("El parámetro columns_to_remove debe ser una lista")
            
        try:
            # Verificar qué columnas existen en el DataFrame
            existing_columns = [col for col in columns_to_remove if col in self.data.columns]
            non_existing_columns = [col for col in columns_to_remove if col not in self.data.columns]
            
            if not existing_columns:
                self.logger.warning("Ninguna de las columnas especificadas existe en el DataFrame")
                return
            
            # Eliminar las columnas existentes
            self.data.drop(columns=existing_columns, inplace=True)
            
            # Preparar mensaje de log
            removed_text = "\nColumnas eliminadas:"
            for col in existing_columns:
                removed_text += f"\n- {col}"
                
            if non_existing_columns:
                not_found_text = "\nColumnas no encontradas:"
                for col in non_existing_columns:
                    not_found_text += f"\n- {col}"
            else:
                not_found_text = ""
            
            self.logger.info(
                f"Se eliminaron {len(existing_columns)} columnas exitosamente."
                f"{removed_text}"
                f"{not_found_text}\n"
                f"Columnas restantes en el DataFrame: {len(self.data.columns)}"
            )
            
        except Exception as e:
            self.logger.error(f"Error al eliminar columnas: {str(e)}")
            raise

    def plot_correlation_matrix(self, figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Genera y muestra una matriz de correlación de Pearson para todas las variables
        numéricas del DataFrame. Muestra un heatmap triangular con los valores de
        correlación y una escala de colores.
        
        Args:
            figsize (Tuple[int, int]): Tamaño de la figura (ancho, alto)
        """
        if self.data is None:
            raise ValueError("DataFrame no cargado. Debe llamar a set_data() primero.")
            
        try:
            # Seleccionar solo columnas numéricas
            numeric_columns = self.data.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_columns) < 2:
                raise ValueError("Se necesitan al menos 2 columnas numéricas para calcular correlaciones")
                
            # Calcular matriz de correlación
            correlation_matrix = self.data[numeric_columns].corr(method='pearson')
            
            # Crear máscara para el triángulo superior
            mask = np.triu(np.ones_like(correlation_matrix), k=1)
            
            # Configurar el estilo del gráfico
            plt.figure(figsize=figsize)
            
            # Crear heatmap
            sns.heatmap(correlation_matrix, 
                       mask=mask,
                       annot=True,  # Mostrar valores
                       fmt='.2f',   # Formato de 2 decimales
                       cmap='RdBu_r',  # Mapa de colores (rojo-blanco-azul)
                       center=0,     # Centrar en 0
                       square=True,  # Celdas cuadradas
                       linewidths=0.5,  # Líneas entre celdas
                       cbar_kws={'label': 'Correlación de Pearson'})
            
            # Ajustar título y etiquetas
            plt.title('Matriz de Correlación de Variables Numéricas', 
                     pad=20, 
                     fontsize=14)
            
            # Rotar etiquetas para mejor legibilidad
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Ajustar layout
            plt.tight_layout()
            
            # Logging de correlaciones más fuertes
            strong_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i):
                    corr = correlation_matrix.iloc[i, j]
                    if abs(corr) >= 0.5:  # Correlaciones con |r| >= 0.5
                        strong_correlations.append({
                            'variables': (correlation_matrix.columns[i], correlation_matrix.columns[j]),
                            'correlation': corr
                        })
            
            # Ordenar por valor absoluto de correlación
            strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            if strong_correlations:
                correlation_text = "\nCorrelaciones más fuertes encontradas (|r| >= 0.5):"
                for corr in strong_correlations:
                    var1, var2 = corr['variables']
                    correlation_text += f"\n- {var1} vs {var2}: {corr['correlation']:.3f}"
            else:
                correlation_text = "\nNo se encontraron correlaciones fuertes (|r| >= 0.5)"
            
            self.logger.info(
                f"Matriz de correlación generada exitosamente.\n"
                f"Variables numéricas analizadas: {len(numeric_columns)}"
                f"{correlation_text}"
            )
            
        except Exception as e:
            self.logger.error(f"Error al generar matriz de correlación: {str(e)}")
            raise

    def save_processed_dataset(self, filename: str, include_metadata: bool = True) -> None:
        """
        Guarda el dataset procesado en la carpeta data/processed.
        
        Args:
            filename (str): Nombre base del archivo (sin extensión)
            include_metadata (bool): Si True, guarda un archivo JSON con metadatos
                                   del procesamiento
        
        El archivo se guardará con la siguiente estructura:
        - data/processed/{filename}_{timestamp}.csv
        - data/processed/{filename}_{timestamp}_metadata.json (si include_metadata=True)
        """
        if self.data is None:
            raise ValueError("DataFrame no cargado. Debe llamar a set_data() primero.")
            
        try:
            # Crear timestamp para el nombre del archivo
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            
            # Obtener la ruta absoluta al directorio raíz del proyecto
            # Asumiendo que este archivo está en src/features/feature_engineering.py
            current_file = os.path.abspath(__file__)  # Ruta absoluta al archivo actual
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            
            # Crear directorio si no existe
            output_dir = os.path.join(project_root, 'data', 'processed')
            os.makedirs(output_dir, exist_ok=True)
            
            # Construir nombres de archivo
            csv_filename = f"{filename}_{timestamp}.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            
            # Guardar CSV
            self.data.to_csv(csv_path, index=False)
            
            # Preparar y guardar metadatos si se solicita
            if include_metadata:
                metadata = {
                    'filename': csv_filename,
                    'timestamp': timestamp,
                    'rows': len(self.data),
                    'columns': len(self.data.columns),
                    'column_info': {
                        col: {
                            'dtype': str(self.data[col].dtype),
                            'null_count': int(self.data[col].isnull().sum()),
                            'null_percentage': float(self.data[col].isnull().mean() * 100)
                        }
                        for col in self.data.columns
                    },
                    'numeric_columns': {
                        col: {
                            'min': float(self.data[col].min()) if pd.api.types.is_numeric_dtype(self.data[col]) else None,
                            'max': float(self.data[col].max()) if pd.api.types.is_numeric_dtype(self.data[col]) else None,
                            'mean': float(self.data[col].mean()) if pd.api.types.is_numeric_dtype(self.data[col]) else None,
                            'std': float(self.data[col].std()) if pd.api.types.is_numeric_dtype(self.data[col]) else None
                        }
                        for col in self.data.columns
                        if pd.api.types.is_numeric_dtype(self.data[col])
                    },
                    'categorical_columns': {
                        col: {
                            'unique_values': int(self.data[col].nunique()),
                            'most_common': self.data[col].value_counts().nlargest(5).to_dict()
                        }
                        for col in self.data.columns
                        if pd.api.types.is_object_dtype(self.data[col])
                    }
                }
                
                # Guardar metadatos
                metadata_filename = f"{filename}_{timestamp}_metadata.json"
                metadata_path = os.path.join(output_dir, metadata_filename)
                
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                self.logger.info(
                    f"Dataset guardado exitosamente:\n"
                    f"- CSV: {csv_path}\n"
                    f"- Metadatos: {metadata_path}\n"
                    f"\nInformación del dataset:\n"
                    f"- Filas: {len(self.data):,}\n"
                    f"- Columnas: {len(self.data.columns):,}\n"
                    f"- Tamaño del archivo: {os.path.getsize(csv_path) / (1024*1024):.1f} MB"
                )
            else:
                self.logger.info(
                    f"Dataset guardado exitosamente:\n"
                    f"- CSV: {csv_path}\n"
                    f"\nInformación del dataset:\n"
                    f"- Filas: {len(self.data):,}\n"
                    f"- Columnas: {len(self.data.columns):,}\n"
                    f"- Tamaño del archivo: {os.path.getsize(csv_path) / (1024*1024):.1f} MB"
                )
            
        except Exception as e:
            self.logger.error(f"Error al guardar el dataset: {str(e)}")
            raise