# ambientacionuns
Un conjunto de ejemplos y scripts para ambientar lotes agricolas. Clase practica de Genesis, Clasificación y Cartografía de Suelos. Dpto. Agronomia. UNS. 

## Pasos necesarios: 
1- Instalar QGIS y R. Enlace a Youtube con explicación

https://youtu.be/BiyYDAUVHlg

2- Descargar del sitio https://github.com/francofrolla/ambientacionuns toda la información necesaria para realizar la actividad, desde el boton verde "Code"

![figura1](https://github.com/francofrolla/ambientacionuns/blob/main/imagenes/figura%201.png?raw=true)

3- En su pc descomprimir el archivo **"QGIS_1.1 estable.zip"**, y **"Ambientación.zip"** no descomprimir **"processing_r-2.0.0.zip"**

4- IMPORTANTE!!! Inicar QGIS en modo **Administrador** para ello sobre el icono de acceso a QGIS> clic derecho > **Ejecutar como Administrador**. En algunas pc Qgis puede generar una carpeta con varios accesos directos, utilizar la que indique **"QGIS Desktop (id version)"**

4-En Qgis, instalar el complemento **"processing_r-2.0.0.zip"**. En Qgis buscar en el menu superior **Complementos > Administrar e instalar complementos > ... > (buscar archivo en su pc) "processing_r-2.0.0.zip" > Instalar Complemento**. Existe una version mas actual de este complemento (2.2.0) pero  no funciona correctamente, desestimar su uso si QGIS sugiere su instalación. 

![figura1](https://github.com/francofrolla/ambientacionuns/blob/main/imagenes/figura%202.png?raw=true)

5-Se deben pegar los scripts en una carpeta especifica para ser usados en QGIS. Para ello ir a:

**Procesos> Caja de Herramientas > R**

Bajo el icono de **R** hay un menu desplegable, ampliar el menu despegable y elegir cualquier script que exista por defecto. **Copiar ruta a script**. 

![figura1](https://github.com/francofrolla/ambientacionuns/blob/main/imagenes/figura%203.png?raw=true)

Debemos llegar hasta esa carpeta y pegar los scripts que descargamos, se puede ir manualmente o por el siguiente metodo:

En Windows se puede llegar a la carpeta desde **Inicio>Ejecutar** o presionando (Win+R). Pegar ruta de acceso hasta la ultima barra separadora (/). 

Ej: Si mi ruta es *C:/lospipis/.local/share/QGIS/QGIS3/profiles/default/python/plugins/processing_r/builtin_scripts/Histograma.rsx*

(Win+R) > Pegar Ruta > *C:/lospipis/.local/share/QGIS/QGIS3/profiles/default/python/plugins/processing_r/builtin_scripts/ > Ejecutar 

En la carpeta que se abre pegar todos los archivos presentes en la carpeta **QGIS_1.1 estable.zip**

En QGIS en **Caja de Herramientas** ir a **Opciones** (simbolo llave francesa) > **Aceptar**, esto cargara los nuevos scripts ingresados. Al desplegar nuevamente el menu bajo el icono de **R** se deben ver los scripts cargados.

6- La primera vez que se ejecuta el programa se instalaran una serie de paquetes necesarios para ejecutar el programa. Solo ocurrira una vez.

7- Instalado todo se puede probar su funcionamiento con las capas:

1- Campo Javier Seewald/mapa de convexidad.tif

1- Campo Javier Seewald/lote 5346.shp

![figura4](https://github.com/francofrolla/ambientacionuns/blob/main/imagenes/figura%204.png?raw=true)

## Links a Youtube.

**Clase Viernes 17/09** : https://youtu.be/gP3URhBvNzY

**Ambientación de lotes agricolas (con explicación instalacion programas)**, en este caso usar el presente repositorio (github/ambientacionuns) no el que figura en el video (github/mapaderindes) :  https://youtu.be/epv6MgmqGHI 

# Busqueda de Imagenes Sateliteles mediante el explorador:

Link programa Explorador: https://francofrolla.users.earthengine.app/view/explorador-sentinel-2---landsat-8---landsat-5

Link conversor kmz a geojson: http://lmingenieria.com.ar/kmzageojson/

# Ambientar sin R
Metodologia que no require R, simplemente los complementos que trae QGIS por defecto. Simplemente iniciar QGIS desde el icono "QGIS with Grass"

https://www.youtube.com/watch?v=eLT_izEQ4_Q

## Ante cualquier error tomar una captura de pantalla e informar del error via Moodle. Se corregira y documentara lo mas rapido posible.  






