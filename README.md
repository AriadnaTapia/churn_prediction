# Predicción de Cancelación de Clientes – Interconnect

Este proyecto desarrolla un modelo de Machine Learning para predecir si un cliente de la empresa de telecomunicaciones *Interconnect* cancelará su contrato. El objetivo es apoyar las estrategias de retención mediante promociones o ajustes de plan antes de que ocurra la cancelación.

---

## 📊 Objetivo

Construir un modelo predictivo, evaluado principalmente con la métrica AUC-ROC, capaz de clasificar de forma precisa a los clientes que tienen mayor probabilidad de cancelar su contrato.

---

## 🧠 Herramientas y tecnologías

- Python  
- pandas, NumPy  
- Seaborn (visualización)  
- scikit-learn  
- LightGBM (modelo final)

---

## ⚙️ Proceso general

1. **Análisis exploratorio** de los datos para entender patrones y variables importantes.
2. **Limpieza y preprocesamiento**: tratamiento de valores faltantes, codificación de variables, escalado.
3. **Entrenamiento de múltiples modelos** de clasificación (árboles, regresión logística, random forest, LightGBM).
4. **Evaluación comparativa** usando AUC-ROC, Accuracy y F1-score.
5. **Selección del modelo final**: LightGBM con las 7 variables más relevantes.
6. **Validación** con matriz de confusión para verificar su capacidad de generalización.

---

## ✅ Resultados

El modelo final (LightGBM) mostró un rendimiento sólido y balanceado, con una excelente capacidad de distinguir entre clientes que cancelan y los que permanecen. Es apto para futuras etapas de despliegue o integración en sistemas de retención.
