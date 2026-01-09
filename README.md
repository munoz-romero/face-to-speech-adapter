# Face-to-Speech: Adaptación de identidad facial para la síntesis de voz sobre StyleTTS2

Este repositorio contiene el código desarrollado para el **Trabajo Final de Máster (TFM) en la UOC**, titulado **"Síntesis de voz zero-shot condicionada por atributos faciales mediante adaptación de espacios latentes en modelos de difusión"**.

El objetivo principal es realizar una transferencia de estilo cross-modal (de cara a voz), permitiendo que un sistema de Text-to-Speech (TTS) genere audio cuyas características (identidad, género, edad) sean coherentes con una imagen facial de entrada.

## Estructura del proyecto

El repositorio está organizado en las siguientes carpetas:

- **`analysis/`**: Contiene notebooks (Jupyter) para el análisis del espacio latente y la evolución del modelo durante el entrenamiento, facilitando la visualización de la alineación entre los dominios visual y acústico.
- **`dataset_preprocessing/`**: Scripts para la preparación del dataset **LRS3** (Lip Reading Sentences 3). Incluye el procesamiento de metadatos, extracción de atributos faciales y validación de muestras.
- **`training/`**: Scripts de entrenamiento para el *Face Adapter*. Incluye dos versiones:
  - `train_face_adapter_flexible.py`: Implementación con regularización de varianza y pérdidas contrastivas.
  - `train_face_adapter_aux_balanced.py`: Versión con tareas auxiliares balanceadas (predicción de género y edad) para mejorar la representatividad de la voz.
- **`inference/`**: Código para la generación de inferencia. Incluye una interfaz interactiva con **Gradio** (`inference_gradio_flexible.py`) que permite cargar una imagen facial, introducir un texto y generar la voz correspondiente utilizando el modelo entrenado integrado con StyleTTS2.

## Características principales

- **Codificador visual**: Uso de InceptionResnetV1 (FaceNet) preentrenado en VGGFace2 para la extracción de descriptores faciales robustos.
- **Mapeo latente**: Arquitectura de tipo *Adapter* (MLP) diseñada para proyectar embeddings visuales al espacio de estilo de StyleTTS2.
- **Entrenamiento contrastivo**: Implementación de pérdida NCE (Noise Contrastive Estimation) para maximizar la similitud entre la identidad visual y acústica.
- **Supervisión auxiliar**: Regularización mediante la predicción de género y edad para asegurar que los atributos demográficos se preserven en la síntesis de voz.

## Requisitos

El proyecto depende de una instalación funcional de **StyleTTS2**. Los scripts de inferencia y entrenamiento esperan encontrar el repositorio de StyleTTS2 en el path configurado o dentro del entorno de ejecución.

## Autor

**Carlos Muñoz Romero** - Estudiante de Máster en Ciencia de Datos en la UOC.

## Licencia

Esta obra está sujeta a una licencia de **Reconocimiento-NoComercial-SinObraDerivada 3.0 España de Creative Commons**.

---
*Este proyecto ha sido desarrollado como parte del Trabajo Final de Máster en la Universitat Oberta de Catalunya (UOC).*
