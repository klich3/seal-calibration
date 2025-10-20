# SEAL Calibration Examples

Este directorio contiene ejemplos de uso de la librería SEAL Calibration.

## 📝 Ejemplos disponibles

### 1. `basic_calibration.py`
Calibración básica de una cámara individual.

**Uso:**
```bash
python basic_calibration.py
```

**Características:**
- Captura manual (presiona SPACE)
- Patrón chessboard 6x9
- Tamaño de cuadrado: 25mm
- Exporta parámetros en formato `.npz`

---

### 2. `stereo_calibration.py`
Calibración estéreo con dos cámaras.

**Uso desde cámaras en vivo:**
```bash
python stereo_calibration.py --left 0 --right 1 --images 15
```

**Uso desde imágenes existentes:**
```bash
python stereo_calibration.py --from-images calib_imgs/
```

**Parámetros:**
- `--left`: Índice de cámara izquierda (default: 0) proyector laseer
- `--right`: Índice de cámara derecha (default: 1) camera con UV
- `--rows`: Filas del patrón (default: 6)
- `--cols`: Columnas del patrón (default: 9)
- `--square-size`: Tamaño del cuadrado en mm (default: 25.0)
- `--images`: Número de imágenes a capturar (default: 15)
- `--from-images`: Directorio con imágenes existentes

**Formato de imágenes:**
- Izquierda: `*_left.png` o `*_left.jpg`
- Derecha: `*_right.png` o `*_right.jpg`

**Salida:**
- `stereo_calibration.npz`: Parámetros de calibración

---

### 3. `stereo_calibration_complete.py`
Calibración estéreo completa con exportación SEAL.

**Uso desde cámaras:**
```bash
python stereo_calibration_complete.py \
  --left 1 --right 0 \
  --template calibJMS1006207.txt \
  --dev-id JMS1006207
```

**Uso desde imágenes:**
```bash
python stereo_calibration_complete.py \
  --from-images calib_imgs/ \
  --template calibJMS1006207.txt \
  --dev-id JMS1006207
```

**Parámetros adicionales:**
- `--pattern-type`: Tipo de patrón (chessboard, charuco, circles)
- `--no-auto-capture`: Deshabilitar captura automática
- `--output-dir`: Directorio para imágenes capturadas (default: calib_imgs)
- `--template`: Archivo plantilla SEAL para parámetros de fábrica
- `--dev-id`: ID del dispositivo (default: JMS1006207)

**Características:**
- Soporte para múltiples patrones de calibración
- Reporte detallado de calibración
- Exportación a formato SEAL con preservación de parámetros de fábrica
- Captura automática o manual

**Salidas:**
- `stereo_calibration.npz`: Datos técnicos de calibración
- `stereo_calibration_seal.txt`: Archivo SEAL compatible (si se proporciona template)

---

### 4. `seal_format_export.py`
Exportación de calibración a formato SEAL.

**Uso:**
```bash
python seal_format_export.py
```

**Requisitos previos:**
- Archivo `stereo_calibration.npz` (generado por calibración estéreo)
- Archivo plantilla `calibJMS1006207.txt` (opcional, pero recomendado)

**Características:**
- Carga parámetros de calibración desde `.npz`
- Preserva parámetros de fábrica (líneas 2-4) de la plantilla
- Preserva tabla LUT y Gray Code
- Actualiza solo parámetros intrínsecos de cámaras
- Genera advertencias si no se usa plantilla

**Salida:**
- `stereo_calibration_seal.txt`: Archivo SEAL compatible

⚠️ **IMPORTANTE:** Sin plantilla, usa valores predeterminados estimados que NO son aptos para producción.

---

## 🎯 Flujo de trabajo recomendado

### Para calibración completa con SEAL:

1. **Capturar imágenes:**
```bash
python stereo_calibration_complete.py \
  --left 1 --right 0 \
  --images 20 \
  --output-dir calib_imgs
```

o

```bash
python stereo_calibration_complete.py --template calibJMS1006207.txt  --dev-id JMS1006207 --pattern-type chessboard --rows 6 --cols 9  --square-size 3.0 --left 0 --right 1
```

2. **Procesar y exportar:**
```bash
python stereo_calibration_complete.py \
  --from-images calib_imgs/ \
  --template calibJMS1006207.txt \
  --dev-id JMS1006207
```

### Para recalibración desde imágenes existentes:

```bash
python stereo_calibration_complete.py \
  --from-images /ruta/a/imagenes/ \
  --template calibJMS1006207.txt
```

Y mas completo

```bash
python stereo_calibration_complete.py --from-images calib_imgs/ --template calibJMS1006207.txt  --dev-id JMS1006207 --pattern-type chessboard --rows 6 --cols 9  --square-size 3.0
```



---

## 📊 Interpretación de resultados

### Calidad de calibración (RMS Error):
- **< 0.5 píxeles**: Excelente ✓
- **0.5 - 1.0 píxeles**: Buena ✓
- **> 1.0 píxeles**: Necesita mejora ⚠️

### Parámetros clave:
- **Baseline**: Distancia entre cámaras (mm)
- **fx, fy**: Distancia focal en píxeles
- **cx, cy**: Centro óptico
- **k1, k2, k3**: Distorsión radial
- **p1, p2**: Distorsión tangencial

---

## ⚠️ Notas importantes

1. **Resolución**: La calibración estéreo debe realizarse a **1280x720**
2. **Parámetros de fábrica**: Las líneas 2-4 del archivo SEAL son críticas y NO deben modificarse sin equipamiento especializado
3. **Número mínimo de imágenes**: Se requieren al menos 4 pares válidos, pero se recomiendan 15-20 para mejor precisión
4. **Patrón**: Asegúrate de que el patrón sea visible en ambas cámaras simultáneamente

---

## 🔧 Solución de problemas

**Error: "Cannot open cameras"**
- Verifica permisos de cámara en Configuración del Sistema
- Prueba con diferentes índices de cámara (0, 1, 2, ...)

**Error: "Pattern not found"**
- Mejora la iluminación
- Asegúrate de que el patrón esté completamente visible
- Verifica dimensiones del patrón (rows, cols)

**RMS Error alto**
- Captura más imágenes
- Asegúrate de cubrir diferentes ángulos y posiciones
- Verifica que el patrón esté plano y sin distorsiones

---

## Charuco

```shell
python stereo_calibration_complete.py \
  --template calibJMS1006207.txt \
  --dev-id JMS1006207 \
  --pattern-type charuco \
  --square-size 40.0 \
  --marker-size 7.85 \
  --rows 4 \
  --cols 11 \
  --left 0 \
  --right 1
```

## 📚 Referencias

- [README principal](../README.md)
