# Proyecto: Laboratorio 1 - Software Seguro (Unidad 2)

## Descripción General
Este proyecto corresponde al **Laboratorio 1** de la asignatura de Software Seguro, Unidad 2. El objetivo principal es analizar y corregir vulnerabilidades de seguridad en código fuente, aplicando buenas prácticas y herramientas de análisis estático. El laboratorio se centra en la identificación, documentación y remediación de fallos de seguridad comunes en aplicaciones de software.

## Objetivos
- Identificar vulnerabilidades de seguridad en código fuente.
- Aplicar herramientas de análisis estático para detectar fallos.
- Documentar los hallazgos y proponer soluciones seguras.
- Implementar correcciones en el código para mitigar los riesgos detectados.

## Actividades Principales
1. **Análisis de Código Fuente:**
   - Revisión manual del código proporcionado para identificar posibles vulnerabilidades.
   - Enfoque en fallos comunes como inyecciones, manejo inseguro de datos, errores de validación, etc.

2. **Uso de Herramientas de Análisis Estático:**
   - Aplicación de herramientas como SonarQube, Bandit, o similares para el análisis automatizado.
   - Registro de los hallazgos relevantes y comparación con el análisis manual.

3. **Documentación de Vulnerabilidades:**
   - Elaboración de un reporte detallado de cada vulnerabilidad encontrada.
   - Descripción del tipo de vulnerabilidad, su impacto y ejemplos de explotación.
   - Propuesta de soluciones o mitigaciones para cada caso.

4. **Corrección de Vulnerabilidades:**
   - Modificación del código fuente para eliminar o mitigar los riesgos detectados.
   - Justificación de los cambios realizados y evidencia de la mejora en la seguridad.

## Estructura Recomendada del Proyecto
- `modelo.py`: Código fuente principal a analizar y corregir.
- `CVEFixes.csv`: Registro de vulnerabilidades (CVE) y sus respectivas correcciones.
- `README.md`: Este documento, con la guía y explicación del laboratorio.

## Recomendaciones de Seguridad
- Validar y sanear todas las entradas de usuario.
- Utilizar funciones y librerías seguras para el manejo de datos sensibles.
- Evitar el uso de contraseñas o datos sensibles en texto plano.
- Aplicar el principio de menor privilegio en el acceso a recursos.
- Documentar todos los cambios y justificar las decisiones de seguridad.

## Ejemplo de Documentación de Vulnerabilidad
| ID CVE         | Descripción Breve                | Impacto         | Solución Propuesta         |
|----------------|----------------------------------|-----------------|---------------------------|
| CVE-XXXX-YYYY  | Inyección SQL en función login() | Alto            | Uso de consultas preparadas|

## Herramientas Sugeridas
- [SonarQube](https://www.sonarqube.org/)
- [Bandit](https://bandit.readthedocs.io/en/latest/)
- [PyLint](https://pylint.org/)

## Entregables
- Código fuente corregido (`modelo.py`).
- Archivo `CVEFixes.csv` con el registro de vulnerabilidades y correcciones.
- Reporte/documentación detallada de los hallazgos y soluciones.

## Créditos
Elaborado para la materia de **Software Seguro**.

---

> **Nota:** Este laboratorio NO incluye la parte de integración continua ni despliegue automatizado.
