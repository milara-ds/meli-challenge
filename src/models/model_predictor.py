# Importación de librerías necesarias
# Manipulación y análisis de datos
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

# Modelos y métricas de scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

# XGBoost y visualización
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pandas import DataFrame, Series

class ModelPredictor:
    """
    Clase para el entrenamiento, evaluación y explicación de modelos de clasificación.
    
    Esta clase implementa una pipeline completa de machine learning que incluye:
    - Carga y preprocesamiento de datos
    - Entrenamiento de múltiples modelos (Random Forest, XGBoost, Regresión Logística)
    - Optimización de hiperparámetros
    - Evaluación de modelos con múltiples métricas
    - Visualización de resultados y explicabilidad con SHAP
    
    Atributos:
        X_train (DataFrame): Features de entrenamiento
        X_test (DataFrame): Features de prueba
        y_train (Series): Etiquetas de entrenamiento
        y_test (Series): Etiquetas de prueba
        feature_names (List[str]): Nombres de las características
        rf_model (RandomForestClassifier): Modelo Random Forest entrenado
        xgb_model (XGBClassifier): Modelo XGBoost entrenado
        best_model: Mejor modelo seleccionado
        best_model_name (str): Nombre del mejor modelo
    """
    
    def __init__(self):
        """
        Inicializa el ModelPredictor configurando el sistema de logging
        y los atributos necesarios para almacenar modelos y datos.
        """
        # Configuración del sistema de logging para seguimiento del proceso
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Inicialización de atributos para almacenar datos
        self.X_train: Optional[Union[DataFrame, np.ndarray]] = None  # Features de entrenamiento
        self.X_test = None  # Features de prueba
        self.y_train: Optional[Union[Series, np.ndarray]] = None  # Etiquetas de entrenamiento
        self.y_test = None  # Etiquetas de prueba
        self.feature_names: Optional[List[str]] = None  # Nombres de las características
        self.data = None  # Dataset completo
        
        # Diccionarios para almacenar encoders y mapeos de variables categóricas
        self.label_encoders = {}
        self.label_mappings = {}
        
        # Modelos de clasificación
        self.rf_model: Optional[RandomForestClassifier] = None  # Modelo Random Forest
        self.xgb_model: Optional[xgb.XGBClassifier] = None  # Modelo XGBoost
        self.best_model = None  # Mejor modelo según evaluación
        self.best_model_name = None  # Nombre del mejor modelo
        
    def load_data(self, file_path: str, target_column: str = 'condition') -> None:
        """
        Carga y valida los datos desde un archivo CSV.
        
        Esta función realiza las siguientes tareas:
        1. Verifica la existencia del archivo
        2. Carga los datos usando pandas
        3. Valida la presencia de la columna objetivo
        4. Registra información sobre el dataset cargado
        
        Args:
            file_path (str): Ruta al archivo CSV con los datos
            target_column (str): Nombre de la columna objetivo/etiquetas
                               (default: 'condition')
        
        Raises:
            FileNotFoundError: Si no se encuentra el archivo
            ValueError: Si la columna objetivo no existe en el dataset
        """
        try:
            # Verificar existencia del archivo
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
            
            self.logger.info(f"Cargando datos desde: {file_path}")
            self.data = pd.read_csv(file_path)
            
            # Validar presencia de columna objetivo
            if target_column not in self.data.columns:
                raise ValueError(f"La columna objetivo '{target_column}' no existe en el dataset")
            
            # Registrar información del dataset
            self.logger.info(
                f"Dataset cargado exitosamente:\n"
                f"- Número de muestras: {len(self.data):,}\n"
                f"- Número de columnas: {len(self.data.columns):,}\n"
                f"- Columnas disponibles: {', '.join(self.data.columns.tolist())}"
            )
            
        except Exception as e:
            self.logger.error(f"Error al cargar el dataset: {str(e)}")
            raise

    def remove_columns(self, columns_to_remove: List[str]) -> None:
        """
        Elimina columnas específicas del dataset y actualiza los metadatos.
        
        Esta función es útil para:
        - Eliminar columnas irrelevantes para el modelo
        - Remover identificadores o timestamps
        - Limpiar features con alta cardinalidad
        
        Args:
            columns_to_remove (List[str]): Lista de nombres de columnas a eliminar
            
        Raises:
            ValueError: Si no hay datos cargados o si alguna columna no existe
        """
        try:
            # Validar que haya datos cargados
            if self.data is None:
                raise ValueError("No hay datos cargados. Ejecute load_data primero.")
            
            # Verificar existencia de todas las columnas
            non_existent = [col for col in columns_to_remove if col not in self.data.columns]
            if non_existent:
                raise ValueError(f"Las siguientes columnas no existen: {', '.join(non_existent)}")
            
            # Guardar columnas originales para logging
            original_columns = self.data.columns.tolist()
            
            # Eliminar columnas y actualizar feature_names
            self.data = self.data.drop(columns=columns_to_remove)
            if self.feature_names is not None:
                self.feature_names = [col for col in self.feature_names if col not in columns_to_remove]
            
            # Registrar cambios realizados
            self.logger.info(
                f"Columnas eliminadas exitosamente:\n"
                f"- Columnas eliminadas: {', '.join(columns_to_remove)}\n"
                f"- Columnas restantes: {len(self.data.columns):,}\n"
                f"- Nombres de columnas: {', '.join(self.data.columns.tolist())}"
            )
            
        except Exception as e:
            self.logger.error(f"Error al eliminar columnas: {str(e)}")
            raise
            
    def split_dataset(self, target_column: str = 'sold', test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Divide el dataset en conjuntos de entrenamiento y prueba de manera estratificada.
        
        Esta función:
        1. Separa features (X) y variable objetivo (y)
        2. Realiza división estratificada para mantener distribución de clases
        3. Almacena los nombres de las features
        4. Registra estadísticas de la división
        
        Args:
            target_column (str): Nombre de la columna objetivo (default: 'sold')
            test_size (float): Proporción de datos para el conjunto de prueba (default: 0.2)
            random_state (int): Semilla para reproducibilidad (default: 42)
            
        Raises:
            ValueError: Si no hay datos cargados o si la columna objetivo no existe
        """
        try:
            # Validaciones iniciales
            if self.data is None:
                raise ValueError("No hay datos cargados. Ejecute load_data primero.")
            
            if target_column not in self.data.columns:
                raise ValueError(f"La columna objetivo '{target_column}' no existe en el dataset")
            
            # Separar features y target
            X = self.data.drop(columns=[target_column])
            y = self.data[target_column]
            
            # Realizar división estratificada
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Almacenar nombres de features
            self.feature_names = X.columns.tolist()
            
            # Registrar información de la división
            self.logger.info(
                f"Dataset dividido exitosamente:\n"
                f"- Conjunto de entrenamiento: {len(self.X_train):,} muestras ({(1-test_size)*100:.0f}%)\n"
                f"- Conjunto de prueba: {len(self.X_test):,} muestras ({test_size*100:.0f}%)\n"
                f"- Número de features: {len(self.feature_names)}\n"
                f"- Features utilizadas: {', '.join(self.feature_names)}\n"
                f"- Distribución de clases en entrenamiento:\n{self.y_train.value_counts(normalize=True)}"
            )
            
        except Exception as e:
            self.logger.error(f"Error al dividir el dataset: {str(e)}")
            raise
            
    def train_random_forest(self, n_iter: int = 10) -> None:
        """
        Entrena un modelo Random Forest con optimización de hiperparámetros.
        
        El proceso incluye:
        1. Búsqueda aleatoria de hiperparámetros con validación cruzada
        2. Entrenamiento del modelo final con los mejores parámetros
        3. Evaluación completa del modelo con múltiples métricas
        4. Balance de clases automático con class_weight='balanced'
        
        Args:
            n_iter (int): Número de combinaciones de hiperparámetros a probar (default: 10)
            
        Raises:
            ValueError: Si los datos de entrenamiento no están preparados
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Datos de entrenamiento no cargados. Ejecute split_dataset primero.")
            
        try:
            # Definir espacio de búsqueda de hiperparámetros
            param_distributions = {
                'n_estimators': [100, 200, 300],  # Número de árboles
                'max_depth': [10, 20, 30],        # Profundidad máxima de los árboles
                'min_samples_leaf': [1, 2, 4]     # Mínimo de muestras por hoja
            }
            
            # Crear modelo base con balance de clases
            base_model = RandomForestClassifier(
                class_weight='balanced',  # Manejo automático de desbalance
                random_state=42          # Reproducibilidad
            )
            
            # Configurar búsqueda aleatoria de hiperparámetros
            random_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_distributions,
                n_iter=n_iter,           # Número de combinaciones a probar
                cv=3,                    # Validación cruzada con 3 folds
                scoring='f1',            # Optimizar F1-score
                n_jobs=-1,              # Usar todos los núcleos disponibles
                random_state=42,
                verbose=1
            )
            
            # Realizar búsqueda de hiperparámetros
            self.logger.info("Iniciando búsqueda de hiperparámetros para Random Forest...")
            random_search.fit(self.X_train, self.y_train)
            
            # Registrar mejores parámetros encontrados
            best_params = random_search.best_params_
            self.logger.info(f"Mejores hiperparámetros encontrados:\n{best_params}")
            
            # Entrenar modelo final con los mejores parámetros
            self.rf_model = RandomForestClassifier(
                **best_params,
                class_weight='balanced',
                random_state=42
            )
            self.rf_model.fit(self.X_train, self.y_train)
            
            # Evaluar rendimiento del modelo
            train_score = self.rf_model.score(self.X_train, self.y_train)
            test_score = self.rf_model.score(self.X_test, self.y_test)
            
            # Calcular métricas en conjunto de prueba
            y_pred = self.rf_model.predict(self.X_test)
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred)
            }
            
            # Registrar resultados detallados
            self.logger.info(
                f"Modelo Random Forest entrenado exitosamente:\n"
                f"- Hiperparámetros óptimos: {best_params}\n"
                f"- Accuracy en entrenamiento: {train_score:.4f}\n"
                f"- Accuracy en prueba: {test_score:.4f}\n"
                f"- Precisión en prueba: {metrics['precision']:.4f}\n"
                f"- Recall en prueba: {metrics['recall']:.4f}\n"
                f"- F1-score en prueba: {metrics['f1']:.4f}\n"
                f"\nMatriz de confusión:\n{confusion_matrix(self.y_test, y_pred)}\n"
                f"\nReporte de clasificación detallado:\n{classification_report(self.y_test, y_pred)}"
            )
            
        except Exception as e:
            self.logger.error(f"Error durante el entrenamiento del Random Forest: {str(e)}")
            raise
            
    def train_xgboost(self, n_iter: int = 10) -> None:
        """
        Entrena un modelo XGBoost con optimización de hiperparámetros.
        
        El proceso incluye:
        1. Búsqueda aleatoria de hiperparámetros con validación cruzada
        2. Entrenamiento del modelo final con los mejores parámetros
        3. Evaluación completa del modelo con múltiples métricas
        4. Manejo automático del desbalance de clases
        
        Args:
            n_iter (int): Número de combinaciones de hiperparámetros a probar (default: 10)
            
        Raises:
            ValueError: Si los datos de entrenamiento no están preparados
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Datos de entrenamiento no cargados. Ejecute split_dataset primero.")
            
        try:
            # Calcular peso para balanceo de clases
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(self.y_train),
                y=self.y_train
            )
            sample_weights = np.where(self.y_train == 1, 
                                    class_weights[1], 
                                    class_weights[0])
            
            # Definir espacio de búsqueda de hiperparámetros
            param_distributions = {
                'max_depth': [3, 5, 7],          # Profundidad máxima del árbol
                'learning_rate': [0.01, 0.1],    # Tasa de aprendizaje
                'n_estimators': [100, 200],      # Número de árboles
                'min_child_weight': [1, 3, 5],   # Peso mínimo necesario en un nodo hijo
                'gamma': [0, 0.1, 0.2],          # Gamma mínimo para realizar una partición
                'subsample': [0.8, 0.9, 1.0],    # Fracción de muestras para entrenar cada árbol
                'colsample_bytree': [0.8, 0.9, 1.0]  # Fracción de features para cada árbol
            }
            
            # Crear modelo base
            base_model = xgb.XGBClassifier(
                objective='binary:logistic',  # Clasificación binaria
                random_state=42,             # Reproducibilidad
                eval_metric='logloss'        # Métrica de evaluación durante el entrenamiento
            )
            
            # Configurar búsqueda aleatoria de hiperparámetros
            random_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_distributions,
                n_iter=n_iter,           # Número de combinaciones a probar
                cv=3,                    # Validación cruzada con 3 folds
                scoring='f1',            # Optimizar F1-score
                n_jobs=-1,              # Usar todos los núcleos disponibles
                random_state=42,
                verbose=1
            )
            
            # Realizar búsqueda de hiperparámetros
            self.logger.info("Iniciando búsqueda de hiperparámetros para XGBoost...")
            random_search.fit(
                self.X_train, 
                self.y_train,
                sample_weight=sample_weights  # Aplicar pesos para balance de clases
            )
            
            # Registrar mejores parámetros encontrados
            best_params = random_search.best_params_
            self.logger.info(f"Mejores hiperparámetros encontrados:\n{best_params}")
            
            # Entrenar modelo final con los mejores parámetros
            self.xgb_model = xgb.XGBClassifier(
                **best_params,
                objective='binary:logistic',
                random_state=42,
                eval_metric='logloss'
            )
            
            # Entrenar modelo final con pesos de clase
            self.xgb_model.fit(
                self.X_train,
                self.y_train,
                sample_weight=sample_weights
            )
            
            # Evaluar rendimiento del modelo
            train_score = self.xgb_model.score(self.X_train, self.y_train)
            test_score = self.xgb_model.score(self.X_test, self.y_test)
            
            # Calcular métricas en conjunto de prueba
            y_pred = self.xgb_model.predict(self.X_test)
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred)
            }
            
            # Registrar resultados detallados
            self.logger.info(
                f"Modelo XGBoost entrenado exitosamente:\n"
                f"- Hiperparámetros óptimos: {best_params}\n"
                f"- Accuracy en entrenamiento: {train_score:.4f}\n"
                f"- Accuracy en prueba: {test_score:.4f}\n"
                f"- Precisión en prueba: {metrics['precision']:.4f}\n"
                f"- Recall en prueba: {metrics['recall']:.4f}\n"
                f"- F1-score en prueba: {metrics['f1']:.4f}\n"
                f"\nMatriz de confusión:\n{confusion_matrix(self.y_test, y_pred)}\n"
                f"\nReporte de clasificación detallado:\n{classification_report(self.y_test, y_pred)}"
            )
            
        except Exception as e:
            self.logger.error(f"Error durante el entrenamiento del XGBoost: {str(e)}")
            raise
            
    def perform_cross_validation(self, model_type: str = 'both', cv: int = 5) -> Dict[str, List[float]]:
        """
        Realiza validación cruzada para evaluar el rendimiento de los modelos.
        
        Esta función permite:
        1. Evaluar Random Forest y/o XGBoost
        2. Usar múltiples folds de validación cruzada
        3. Obtener métricas promedio y su variabilidad
        
        Args:
            model_type (str): Tipo de modelo a evaluar: 'rf', 'xgb' o 'both' (default: 'both')
            cv (int): Número de folds para validación cruzada (default: 5)
            
        Returns:
            Dict[str, List[float]]: Diccionario con scores de validación cruzada por modelo
            
        Raises:
            ValueError: Si el tipo de modelo no es válido o los modelos no están entrenados
        """
        results = {}
        
        try:
            if model_type not in ['rf', 'xgb', 'both']:
                raise ValueError("model_type debe ser 'rf', 'xgb' o 'both'")
            
            # Validar que los modelos necesarios estén entrenados
            if model_type in ['rf', 'both'] and self.rf_model is None:
                raise ValueError("Random Forest no está entrenado. Ejecute train_random_forest primero.")
            if model_type in ['xgb', 'both'] and self.xgb_model is None:
                raise ValueError("XGBoost no está entrenado. Ejecute train_xgboost primero.")
            
            # Realizar validación cruzada para Random Forest
            if model_type in ['rf', 'both']:
                self.logger.info("Realizando validación cruzada para Random Forest...")
                rf_scores = cross_val_score(
                    self.rf_model,
                    self.X_train,
                    self.y_train,
                    cv=cv,
                    scoring='f1',
                    n_jobs=-1
                )
                results['random_forest'] = rf_scores.tolist()
                self.logger.info(
                    f"Resultados Random Forest:\n"
                    f"- F1-score promedio: {rf_scores.mean():.4f} (+/- {rf_scores.std() * 2:.4f})"
                )
            
            # Realizar validación cruzada para XGBoost
            if model_type in ['xgb', 'both']:
                self.logger.info("Realizando validación cruzada para XGBoost...")
                xgb_scores = cross_val_score(
                    self.xgb_model,
                    self.X_train,
                    self.y_train,
                    cv=cv,
                    scoring='f1',
                    n_jobs=-1
                )
                results['xgboost'] = xgb_scores.tolist()
                self.logger.info(
                    f"Resultados XGBoost:\n"
                    f"- F1-score promedio: {xgb_scores.mean():.4f} (+/- {xgb_scores.std() * 2:.4f})"
                )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error durante la validación cruzada: {str(e)}")
            raise
            
    def select_best_model(self) -> None:
        """
        Selecciona el mejor modelo basado en el F1-score en el conjunto de prueba.
        
        Esta función:
        1. Compara el rendimiento de Random Forest y XGBoost
        2. Selecciona el modelo con mejor F1-score
        3. Actualiza los atributos best_model y best_model_name
        
        Raises:
            ValueError: Si los modelos no están entrenados
        """
        try:
            if self.rf_model is None or self.xgb_model is None:
                raise ValueError("Ambos modelos deben estar entrenados antes de seleccionar el mejor.")
            
            # Calcular predicciones y métricas para Random Forest
            rf_pred = self.rf_model.predict(self.X_test)
            rf_f1 = f1_score(self.y_test, rf_pred)
            
            # Calcular predicciones y métricas para XGBoost
            xgb_pred = self.xgb_model.predict(self.X_test)
            xgb_f1 = f1_score(self.y_test, xgb_pred)
            
            # Seleccionar el mejor modelo
            if rf_f1 >= xgb_f1:
                self.best_model = self.rf_model
                self.best_model_name = 'random_forest'
                best_f1 = rf_f1
            else:
                self.best_model = self.xgb_model
                self.best_model_name = 'xgboost'
                best_f1 = xgb_f1
            
            self.logger.info(
                f"Selección de mejor modelo completada:\n"
                f"- F1-score Random Forest: {rf_f1:.4f}\n"
                f"- F1-score XGBoost: {xgb_f1:.4f}\n"
                f"- Mejor modelo: {self.best_model_name} (F1-score: {best_f1:.4f})"
            )
            
        except Exception as e:
            self.logger.error(f"Error al seleccionar el mejor modelo: {str(e)}")
            raise
            
    def encode_categorical(self, column_name: str) -> Dict[str, int]:
        """
        Codifica una columna categórica usando LabelEncoder.
        
        Esta función:
        1. Aplica LabelEncoder a la columna especificada
        2. Guarda el encoder para uso futuro
        3. Almacena el mapeo de valores para interpretabilidad
        
        Args:
            column_name (str): Nombre de la columna a codificar
            
        Returns:
            Dict[str, int]: Diccionario con el mapeo de valores originales a códigos
            
        Raises:
            ValueError: Si la columna no existe en el dataset
        """
        try:
            if column_name not in self.data.columns:
                raise ValueError(f"La columna '{column_name}' no existe en el dataset")
            
            # Crear y ajustar el encoder
            le = LabelEncoder()
            self.data[column_name] = le.fit_transform(self.data[column_name])
            
            # Guardar el encoder y crear mapeo
            self.label_encoders[column_name] = le
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            self.label_mappings[column_name] = mapping
            
            self.logger.info(
                f"Columna '{column_name}' codificada exitosamente:\n"
                f"- Valores únicos: {len(mapping)}\n"
                f"- Mapeo: {mapping}"
            )
            
            return mapping
            
        except Exception as e:
            self.logger.error(f"Error al codificar la columna '{column_name}': {str(e)}")
            raise
            
    def get_encoding_mapping(self, column_name: str) -> Optional[Dict[str, int]]:
        """
        Obtiene el mapeo de codificación para una columna categórica.
        
        Args:
            column_name (str): Nombre de la columna
            
        Returns:
            Optional[Dict[str, int]]: Diccionario con el mapeo o None si no existe
        """
        return self.label_mappings.get(column_name)
            
    def explain_model(self, model_type: str = 'rf', n_samples: int = 100, random_state: int = 42) -> None:
        """
        Genera explicaciones detalladas del modelo usando SHAP values sobre una muestra de datos.
        Incluye múltiples visualizaciones para una mejor interpretabilidad.
        
        Args:
            model_type (str): Tipo de modelo a explicar ('rf' o 'xgb')
            n_samples (int): Número de muestras a usar para el análisis SHAP
            random_state (int): Semilla para reproducibilidad del muestreo
        """
        try:
            if model_type not in ['rf', 'xgb']:
                raise ValueError("model_type debe ser 'rf' o 'xgb'")
            
            model = self.rf_model if model_type == 'rf' else self.xgb_model
            if model is None:
                raise ValueError(f"Modelo {model_type} no entrenado")
            
            # Asegurarse de que X_train sea un DataFrame
            if not isinstance(self.X_train, pd.DataFrame):
                self.X_train = pd.DataFrame(self.X_train, columns=self.feature_names)
            
            # Tomar una muestra aleatoria de los datos
            if n_samples >= len(self.X_train):
                self.logger.warning(f"n_samples ({n_samples}) es mayor que el número de datos disponibles "
                                f"({len(self.X_train)}). Usando todos los datos.")
                X_sample = self.X_train
            else:
                X_sample = self.X_train.sample(n=n_samples, random_state=random_state)
            
            self.logger.info(f"Generando explicaciones SHAP para modelo {model_type} "
                         f"usando {len(X_sample)} muestras...")
            
            # Crear el explicador
            explainer = shap.TreeExplainer(model)
            
            # Calcular SHAP values
            if model_type == 'rf':
                shap_values = explainer.shap_values(X_sample)[:,:,1]  # Clase positiva para RF
            else:
                shap_values = explainer.shap_values(X_sample)
            
            # 1. Summary Plot - Visión general de la importancia de características
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.title(f'SHAP Summary Plot - {model_type.upper()} (n={len(X_sample)})')
            plt.tight_layout()
            plt.show()
            
            # 2. Bar Plot - Magnitud absoluta de importancia de características
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
            plt.title(f'SHAP Feature Importance - {model_type.upper()}')
            plt.tight_layout()
            plt.show()
            
            # 3. Force Plots - Ejemplos de predicción para cada clase
            # Hacer predicciones para encontrar ejemplos de cada clase
            y_pred = model.predict(X_sample)
            
            # Encontrar índices para cada clase
            idx_class_0 = np.where(y_pred == 0)[0][0] if 0 in y_pred else None
            idx_class_1 = np.where(y_pred == 1)[0][0] if 1 in y_pred else None
            
            if idx_class_0 is not None:
                plt.figure(figsize=(20, 3))
                shap.force_plot(
                    explainer.expected_value[1] if model_type == 'rf' else explainer.expected_value,
                    shap_values[idx_class_0:idx_class_0+1,:],
                    X_sample.iloc[idx_class_0:idx_class_0+1,:],
                    show=False,
                    matplotlib=True
                )
                plt.title(f'SHAP Force Plot - Predicción Clase 0 (Ejemplo)')
                plt.tight_layout()
                plt.show()
            
            if idx_class_1 is not None:
                plt.figure(figsize=(20, 3))
                shap.force_plot(
                    explainer.expected_value[1] if model_type == 'rf' else explainer.expected_value,
                    shap_values[idx_class_1:idx_class_1+1,:],
                    X_sample.iloc[idx_class_1:idx_class_1+1,:],
                    show=False,
                    matplotlib=True
                )
                plt.title(f'SHAP Force Plot - Predicción Clase 1 (Ejemplo)')
                plt.tight_layout()
                plt.show()
            
            # Guardar información de importancia de características
            feature_importance = np.abs(shap_values).mean(0)
            feature_importance_dict = dict(zip(X_sample.columns, feature_importance))
            sorted_features = sorted(
                feature_importance_dict.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            self.logger.info("\nImportancia de características (ordenadas):")
            for feature, importance in sorted_features:
                self.logger.info(f"{feature}: {importance:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error al generar explicaciones del modelo: {str(e)}")
            self.logger.error(f"Forma de X_sample: {X_sample.shape}")
            if isinstance(shap_values, list):
                self.logger.error(f"Forma de shap_values[1]: {shap_values[1].shape}")
            else:
                self.logger.error(f"Forma de shap_values: {shap_values.shape}")
            raise

    def train_logistic_regression(self, random_state: int = 42) -> Dict[str, float]:
        """
        Entrena un modelo de Regresión Logística con estandarización de features.
        
        Esta función:
        1. Estandariza las features usando StandardScaler
        2. Maneja el desbalance de clases automáticamente
        3. Entrena el modelo con los mejores hiperparámetros
        4. Evalúa el rendimiento con múltiples métricas
        
        Args:
            random_state (int): Semilla para reproducibilidad (default: 42)
            
        Returns:
            Dict[str, float]: Diccionario con las métricas de rendimiento
            
        Raises:
            ValueError: Si los datos de entrenamiento no están preparados
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Datos de entrenamiento no cargados. Ejecute split_dataset primero.")
            
        try:
            # Estandarizar features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(self.X_train)
            X_test_scaled = scaler.transform(self.X_test)
            
            # Calcular pesos para balance de clases
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(self.y_train),
                y=self.y_train
            )
            class_weight_dict = dict(zip(np.unique(self.y_train), class_weights))
            
            # Crear y entrenar modelo
            self.lr_model = LogisticRegression(
                class_weight=class_weight_dict,  # Manejo de desbalance
                random_state=random_state,
                max_iter=1000,                   # Aumentar iteraciones máximas
                n_jobs=-1                        # Usar todos los núcleos
            )
            
            # Entrenar modelo
            self.lr_model.fit(X_train_scaled, self.y_train)
            
            # Evaluar rendimiento
            train_score = self.lr_model.score(X_train_scaled, self.y_train)
            test_score = self.lr_model.score(X_test_scaled, self.y_test)
            
            # Calcular métricas en conjunto de prueba
            y_pred = self.lr_model.predict(X_test_scaled)
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred)
            }
            
            # Registrar resultados
            self.logger.info(
                f"Modelo de Regresión Logística entrenado exitosamente:\n"
                f"- Accuracy en entrenamiento: {train_score:.4f}\n"
                f"- Accuracy en prueba: {test_score:.4f}\n"
                f"- Precisión en prueba: {metrics['precision']:.4f}\n"
                f"- Recall en prueba: {metrics['recall']:.4f}\n"
                f"- F1-score en prueba: {metrics['f1']:.4f}\n"
                f"\nMatriz de confusión:\n{confusion_matrix(self.y_test, y_pred)}\n"
                f"\nReporte de clasificación detallado:\n{classification_report(self.y_test, y_pred)}"
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error durante el entrenamiento de la Regresión Logística: {str(e)}")
            raise
            
    def plot_roc_curves(self) -> Dict[str, float]:
        """
        Genera y muestra curvas ROC comparativas para todos los modelos entrenados.
        
        Esta función:
        1. Calcula y grafica curvas ROC para cada modelo
        2. Compara el rendimiento usando AUC-ROC
        3. Muestra la visualización en la celda del notebook
        
        Returns:
            Dict[str, float]: Diccionario con los valores AUC-ROC de cada modelo
            
        Raises:
            ValueError: Si ningún modelo está entrenado
        """
        try:
            # Verificar que al menos un modelo esté entrenado
            models = {
                'Random Forest': self.rf_model,
                'XGBoost': self.xgb_model,
                'Logistic Regression': getattr(self, 'lr_model', None)
            }
            
            trained_models = {name: model for name, model in models.items() if model is not None}
            
            if not trained_models:
                raise ValueError("No hay modelos entrenados para generar curvas ROC")
            
            # Configurar el gráfico
            plt.figure(figsize=(10, 8))
            auc_scores = {}
            
            # Graficar curva ROC para cada modelo
            for name, model in trained_models.items():
                if name == 'Logistic Regression':
                    # Usar datos escalados para regresión logística
                    scaler = StandardScaler()
                    X_test_scaled = scaler.fit_transform(self.X_test)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                
                # Calcular curva ROC
                fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
                auc_score = auc(fpr, tpr)
                auc_scores[name] = auc_score
                
                # Graficar curva
                plt.plot(
                    fpr, tpr,
                    label=f'{name} (AUC = {auc_score:.3f})',
                    linewidth=2
                )
            
            # Agregar línea de referencia y detalles del gráfico
            plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
            plt.xlabel('Tasa de Falsos Positivos')
            plt.ylabel('Tasa de Verdaderos Positivos')
            plt.title('Comparación de Curvas ROC')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
            
            # Mostrar el gráfico
            plt.show()
            
            # Registrar resultados
            self.logger.info(
                f"Curvas ROC generadas exitosamente:\n"
                f"- Scores AUC-ROC:\n" +
                "\n".join(f"  • {name}: {score:.4f}" for name, score in auc_scores.items())
            )
            
            return auc_scores
            
        except Exception as e:
            self.logger.error(f"Error al generar curvas ROC: {str(e)}")
            raise

    def predict_random_sample(self, model_type: str = 'rf', random_state: int = None) -> Dict[str, Any]:
        """
        Realiza una predicción sobre una muestra aleatoria del conjunto de prueba.
        
        Esta función:
        1. Selecciona un registro aleatorio del conjunto de prueba
        2. Realiza la predicción usando el modelo especificado
        3. Devuelve la probabilidad de la clase positiva y detalles del registro
        
        Args:
            model_type (str): Tipo de modelo a usar: 'rf', 'xgb' o 'lr' (default: 'rf')
            random_state (int): Semilla para reproducibilidad (default: None)
            
        Returns:
            Dict[str, Any]: Diccionario con:
                - 'sample': Registro seleccionado
                - 'true_label': Etiqueta verdadera
                - 'predicted_proba': Probabilidad predicha para clase 1
                - 'features': Valores de las features más importantes
                
        Raises:
            ValueError: Si el modelo especificado no existe o no está entrenado
        """
        try:
            # Validar que existan datos de prueba
            if self.X_test is None or self.y_test is None:
                raise ValueError("No hay datos de prueba disponibles")
            
            # Seleccionar el modelo
            models = {
                'rf': self.rf_model,
                'xgb': self.xgb_model,
                'lr': getattr(self, 'lr_model', None)
            }
            
            if model_type not in models:
                raise ValueError(f"Tipo de modelo '{model_type}' no válido. Usar 'rf', 'xgb' o 'lr'")
            
            model = models[model_type]
            if model is None:
                raise ValueError(f"El modelo {model_type} no está entrenado")
            
            # Seleccionar muestra aleatoria
            np.random.seed(random_state)
            sample_idx = np.random.randint(0, len(self.X_test))
            X_sample = self.X_test.iloc[[sample_idx]]
            y_true = self.y_test.iloc[sample_idx]
            
            # Realizar predicción
            if model_type == 'lr':
                # Estandarizar para regresión logística
                scaler = StandardScaler()
                X_sample_scaled = scaler.fit_transform(X_sample)
                pred_proba = model.predict_proba(X_sample_scaled)[0, 1]
            else:
                pred_proba = model.predict_proba(X_sample)[0, 1]
            
            # Obtener features más importantes si hay SHAP values disponibles
            important_features = {}
            if hasattr(self, 'feature_importance_dict'):
                # Obtener las 5 features más importantes
                top_features = sorted(
                    self.feature_importance_dict.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:5]
                
                for feature, _ in top_features:
                    important_features[feature] = X_sample[feature].iloc[0]
            
            result = {
                'sample': X_sample.to_dict('records')[0],
                'true_label': int(y_true),
                'predicted_proba': float(pred_proba),
                'features': important_features
            }
            
            # Logging de resultados
            self.logger.info(
                f"\nPredicción realizada con {model_type.upper()}:\n"
                f"- Probabilidad de clase 1: {pred_proba:.4f}\n"
                f"- Etiqueta verdadera: {y_true}\n"
                f"- Features importantes:\n" +
                "\n".join(f"  • {k}: {v}" for k, v in important_features.items())
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error al realizar predicción: {str(e)}")
            raise