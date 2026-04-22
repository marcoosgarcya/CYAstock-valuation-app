# Motor de Valoración de Empresas 📈

DCF, ratios de calidad y múltiplos comparables con datos reales de Yahoo Finance.

## Stack

| Componente       | Tecnología          | Propósito                          |
|------------------|---------------------|------------------------------------|
| Frontend/UI      | Streamlit           | Dashboard interactivo (Python puro)|
| Datos            | yfinance            | API directa de Yahoo Finance       |
| Cálculos         | numpy / pandas      | DCF, WACC, ratios — sin LLM       |
| Gráficos         | Plotly               | Barras, proyecciones               |

## Instalación

```bash
# 1. Clonar o descargar los archivos
# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar
streamlit run valuation_app.py
```

La app se abrirá en `http://localhost:8501`.

## Qué incluye

- **Ingesta de datos**: Yahoo Finance API — datos estructurados en JSON, no extraídos por LLM.
- **Motor DCF**: FCF proyectado a 5 años, valor terminal (Gordon Growth), descuento por WACC dinámico.
- **Deuda Neta**: Puente EV → Equity Value explícito (Deuda Total - Caja).
- **Ratios de calidad**: ROIC, Margen Operativo, Ratio de Solvencia.
- **Múltiplos relativos**: P/E, EV/EBITDA, P/S vs. media del sector.
- **Auditoría**: Validación Activos = Pasivos + Patrimonio por año.
- **Detección de outliers**: Variaciones >50% en márgenes o FCF.
- **Sliders interactivos**: Ajustar crecimiento, WACC y crecimiento terminal en tiempo real.
- **Escenarios Bull/Base/Bear**: Un clic para ver valoración optimista/pesimista.
- **Terminal de logs**: Trazabilidad completa de cada dato.

## Diferencias clave vs. la versión Base44

| Aspecto              | Base44                        | Python (esta app)              |
|----------------------|-------------------------------|--------------------------------|
| Fuente de datos      | LLM + internet (riesgo)      | API estructurada (fiable)      |
| Cálculos             | JavaScript                    | Python (numpy/pandas)          |
| Divisas              | Conversión opaca              | Divisa original visible        |
| Trazabilidad         | Logs básicos                  | Logs detallados + fuente API   |
| Despliegue           | Base44 hosting                | Local / Streamlit Cloud / VPS  |

## Despliegue en la nube

### Streamlit Cloud (gratis)
1. Sube los archivos a un repo de GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta el repo → la app se despliega automáticamente

### VPS / servidor propio
```bash
streamlit run valuation_app.py --server.port 8501 --server.address 0.0.0.0
```

## Limitaciones

- yfinance depende del scraping de Yahoo Finance; puede fallar si Yahoo cambia su estructura.
- Los múltiplos sectoriales son aproximaciones estáticas; para datos dinámicos se necesitaría una API de pago (Financial Modeling Prep, SimFin, etc.).
- La beta proviene de Yahoo Finance y puede diferir de otras fuentes.
- No incluye conversión de divisas automática (los datos se muestran en la divisa de reporte de la empresa).
