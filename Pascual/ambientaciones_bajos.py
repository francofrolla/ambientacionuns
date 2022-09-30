

def funcion_sentinel2():

    """
    Descarga de serie Sentinel 2

    """

    import ee
    import pandas
    import config

    #Map the function over one year of data.
    coleccionfiltrada = ee.ImageCollection('COPERNICUS/S2').filterBounds(config.lote.bounds())
    #filtro de duplicadas
    lista = coleccionfiltrada.toList(coleccionfiltrada.size())
    imagen = ee.Image(lista.get(0))
    lista = lista.add(imagen)

    def detectar_duplicador(imagen):
            esduplicado = ee.String("")
            numero = lista.indexOf(imagen)
            imagen1 = ee.Image(lista.get(numero.add(1)))
            #Compare the image(0) in the ImageCollection with the image(1) in the List
            fecha1 = imagen.date().format("Y-M-d")
            fecha2 = imagen1.date().format("Y-M-d")
            estado = ee.Algorithms.IsEqual(fecha1,fecha2)
            esduplicado = ee.String(ee.Algorithms.If(estado,"duplicado","no duplicado"));
            imagen = imagen.set("duplicado", esduplicado)
            return imagen

    coleccionfiltrada = coleccionfiltrada.map(lambda image: detectar_duplicador(image))
    coleccionfiltrada = coleccionfiltrada.filter(ee.Filter.eq('duplicado', "no duplicado"))


    #Recorte de lote
    coleccionfiltrada = coleccionfiltrada.map(lambda img: img.clip(config.lote.dissolve()))

    def agregar_nubes(image): 
          meanDict = image.reduceRegion(
          reducer= ee.Reducer.anyNonZero(),
          geometry= config.lote,
          scale= 10,
          )
          image = image.set("mascara",meanDict.get("QA60"))
          return image
      

    print("Pasando por filtro de nubes")
    coleccionfiltrada = coleccionfiltrada.map(lambda image: agregar_nubes(image));
    coleccionfiltrada = coleccionfiltrada.filterMetadata('mascara', 'equals', 0);
      
    def ndvi(img):
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        img = img.addBands(ndvi)
        return img

    coleccionfiltrada = coleccionfiltrada.map(lambda img: ndvi(img))
        
    years = ee.List.sequence(2016,2021,1)
    #years = list(range(1984,2013,1))

    """
        Enmascaro todos los pixeles con NDVI menor a 0.1
        """

    def enmascarar_pixeles(image):
        ndvi = image.select('NDVI');
        #THRESHOLD
        # if NDVI less or equal to 0 => 0 else 1
        mask = ndvi.gte(0.15).rename('mask')
        ndvi_mask = ndvi.updateMask(mask).rename(['NDVI2'])
        image = image.addBands(ndvi_mask);
        return image

    coleccionfiltrada = coleccionfiltrada.map(lambda image: enmascarar_pixeles(image))


       
    def fechas(img):
          fecha = img.date().format("YYYY-MM")
          año = img.date().format("YYYY")
          mes = img.date().format("M")
          img = img.set('month', mes)  
          img = img.set("system:time_piola", fecha) 
          img = img.set("year", año) 
          return img
      
    coleccionfiltrada = coleccionfiltrada.map(lambda imagen:fechas(imagen))

    def ndvi(img):
        ndvi = img.normalizedDifference(['B5', 'B4']).rename('NDVI')
        img = img.addBands(ndvi)
        return img

    #coleccionfiltrada = coleccionfiltrada.map(lambda img: ndvi(img))

      #fechas = fechas.map(lambda valor:convertir_string(valor))
    fechas = coleccionfiltrada.aggregate_array("system:time_piola")
    meses = coleccionfiltrada.aggregate_array("month").getInfo()
    years = coleccionfiltrada.aggregate_array("year").getInfo()

    años_cliente = map(float, years)
    meses_cliente = map(float, meses)

      #Genero dataframe con los datos
    import pandas as pd
    import numpy

    df = pd.DataFrame(list(zip(años_cliente,meses_cliente)),columns =["años","meses"])
    print(df)    




    años = df['años'].unique().tolist()
    meses = df['meses'].tolist()

    print(años)
    años = [2017,2018,2019,2020,2021]

    #paso a integer
    años = [int(año) for año in años]
    meses = [int(mes) for mes in meses]
   
    coleccion = coleccionfiltrada
    years = años
    
    laimagen = ee.Image()

    for a in years:
     subset = df[df["años"] == a]
     meses = subset['meses'].tolist()
     meses = [int(mes) for mes in meses]
     meses = list(set(meses))

     print(meses)
     for i in meses:
       
       coleccion1 = coleccion.filterMetadata('year', 'equals', str(a))
       coleccion1 = coleccion1.filterMetadata('month', 'equals', str(i))
       imagen = ee.Image(coleccion1.median().select(["NDVI"])).rename(str(a)+"-"+str(i))
       laimagen = laimagen.addBands(imagen)
    
    #print(laimagen.getInfo())
    
    print("Descargando...")
    path = laimagen.getDownloadUrl({
      'name': "Datos mensuales",
      'scale' : 10,
      'region' : config.lote})
    print(path)
    
    return 





#Script para cargar KML a Geemao y GEE
def imprimir(output):
    with output:
     print("hola")
    
def ingresar_poligono(nombrelote,ruta):

  ###Para ingresar un KML
  import geopandas as gpd
  import fiona
  # Enable fiona driver
  gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
  # Read file
  df = gpd.read_file(ruta, driver='KML')

  import os
  #os.chdir("/content/lotes")
  direccion = os.getcwd()
  rutafinal = direccion+"/salida1.geojson"
  df.to_file(rutafinal, driver='GeoJSON')

  f=open(rutafinal, "r")
  contents =f.read()
  geojson1 = contents.replace(", 0.0", "")
  
  
   
  #df.plot()
  df1 = df.to_crs("EPSG:3857")
  df1['dissolvefield'] = 1
  df1 = df1.dissolve(by='dissolvefield')
  df1 = df1["geometry"]
  return geojson1,df1,nombrelote

#script para hacer composiciones mensuales
def busqueda_imagenes(output,lote,año_inicio,año_fin,mes_inicio,mes_fin):
  #los valores pord defecto son 0.3,0.75 y 100
  import config
  import ee
  import json
  from pandas.io.json import json_normalize
  import pandas as pd
  import numpy as np
  import fiona
  import seaborn as sns
  import geopandas as gpd
  import matplotlib.pyplot as plt
  from IPython.display import HTML, display
  import time
  from pyproj import CRS

  
  #d = json.loads(geojson)
  #geometria = (d['features'][0]["geometry"]["coordinates"])
  #lote = ee.Geometry.Polygon(geometria[0])

    
  loteentero = lote
  lote = lote.buffer(-35)

  coleccionfiltrada = ee.ImageCollection('COPERNICUS/S2').filterBounds(lote.bounds())                  .filter(ee.Filter.calendarRange(año_inicio,año_fin,'year'))
  
  lista = coleccionfiltrada.toList(coleccionfiltrada.size())
  imagen = ee.Image(lista.get(0))
  lista = lista.add(imagen)

  def detectar_duplicador(imagen):
        esduplicado = ee.String("")
        numero = lista.indexOf(imagen)
        imagen1 = ee.Image(lista.get(numero.add(1)))
        #Compare the image(0) in the ImageCollection with the image(1) in the List
        fecha1 = imagen.date().format("Y-M-d")
        fecha2 = imagen1.date().format("Y-M-d")
        estado = ee.Algorithms.IsEqual(fecha1,fecha2)
        esduplicado = ee.String(ee.Algorithms.If(estado,"duplicado","no duplicado"));
        imagen = imagen.set("duplicado", esduplicado)
        return imagen

  coleccionfiltrada = coleccionfiltrada.map(lambda image: detectar_duplicador(image))
  coleccionfiltrada = coleccionfiltrada.filter(ee.Filter.eq('duplicado', "no duplicado"))
  with output:
   print("Filtrado de imagenes con la misma fecha")



  coleccionfiltrada = coleccionfiltrada.map(lambda img: img.clip(loteentero.dissolve()))

  def agregar_nubes(image): 
      meanDict = image.reduceRegion(
      reducer= ee.Reducer.anyNonZero(),
      geometry= lote,
      scale= 10,
      )
      image = image.set("mascara",meanDict.get("QA60"))
      return image
  
  with output:
   print("Pasando por filtro de nubes")
  coleccionfiltrada = coleccionfiltrada.map(lambda image: agregar_nubes(image));
  coleccionfiltrada = coleccionfiltrada.filterMetadata('mascara', 'equals', 0);
  
  with output:
   print("NDVI para coleccion")

  def ndvi(img):
    ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
    img = img.addBands(ndvi)
    return img

  coleccionfiltrada = coleccionfiltrada.map(lambda img: ndvi(img))
    
  with output:
   print("NDWI para coleccion")

  def ndwi(img):
    ndwi = img.normalizedDifference(['B3', 'B11']).rename('NDWI')
    img = img.addBands(ndwi)
    return img

  coleccionfiltrada = coleccionfiltrada.map(lambda img: ndwi(img))
    
  with output:
   print("Genero un mosaico mensual")
 
  #Genero un mosaico mensual
  #Mosaico mensual para NDVI. 
  # Modificado de https://gis.stackexchange.com/questions/258344/reduce-image-collection-to-get-annual-monthly-sum-precipitation
  #Se modificca para coincidir con los meses
 
    

  def sacar_meses(imagen):
    fecha = imagen.date().format("MM")
    imagen = imagen.set("meses",fecha)
    return imagen
   
  coleccion2017 = coleccionfiltrada.filter(ee.Filter.calendarRange(2017, 2017, 'year'))
  coleccion2018 = coleccionfiltrada.filter(ee.Filter.calendarRange(2018, 2018, 'year'))
  coleccion2019 = coleccionfiltrada.filter(ee.Filter.calendarRange(2019, 2019, 'year'))
  coleccion2020 = coleccionfiltrada.filter(ee.Filter.calendarRange(2020, 2020, 'year'))

  coleccion2017 = coleccion2017.map(lambda imagen:sacar_meses(imagen))
  coleccion2018 = coleccion2018.map(lambda imagen:sacar_meses(imagen))
  coleccion2019 = coleccion2019.map(lambda imagen:sacar_meses(imagen))
  coleccion2020 = coleccion2020.map(lambda imagen:sacar_meses(imagen))

  meses2017 = coleccion2017.aggregate_array("meses").distinct()   
  meses2018 = coleccion2018.aggregate_array("meses").distinct()   
  meses2019 = coleccion2019.aggregate_array("meses").distinct()   
  meses2020 = coleccion2020.aggregate_array("meses").distinct()   
    
  def convertir_string(valor):
    return ee.Number.parse(valor)

  meses2017 = meses2017.map(lambda valor:convertir_string(valor))
  meses2018 = meses2018.map(lambda valor:convertir_string(valor))
  meses2019 = meses2019.map(lambda valor:convertir_string(valor))
  meses2020 = meses2020.map(lambda valor:convertir_string(valor))

    
    
    

  def mosaico_mensual_2017(m):
      coleccion = coleccion2017.filter(ee.Filter.calendarRange(m, m, 'month')).select(['NDVI','NDWI','B4', 'B3', 'B2'])
      fecha = coleccion.first().date().format("YYYY-MM")
      año = coleccion.first().date().format("YYYY")
      imagen = coleccion.mean()
      imagen = imagen.set('month', m)  
      #fecha = imagen.select("NDVI").date().format("YYYY-MM")
      imagen = imagen.set("system:time_start", fecha) 
      imagen = imagen.set("year", año) 
      return imagen
    
  def mosaico_mensual_2018(m):
      coleccion = coleccion2018.filter(ee.Filter.calendarRange(m, m, 'month')).select(['NDVI','NDWI','B4', 'B3', 'B2'])
      fecha = coleccion.first().date().format("YYYY-MM")
      año = coleccion.first().date().format("YYYY")
      imagen = coleccion.mean()
      imagen = imagen.set('month', m)  
      #fecha = imagen.select("NDVI").date().format("YYYY-MM")
      imagen = imagen.set("system:time_start", fecha) 
      imagen = imagen.set("year", año) 
      return imagen

  def mosaico_mensual_2019(m):
      coleccion = coleccion2019.filter(ee.Filter.calendarRange(m, m, 'month')).select(['NDVI','NDWI','B4', 'B3', 'B2'])
      fecha = coleccion.first().date().format("YYYY-MM")
      año = coleccion.first().date().format("YYYY")
      imagen = coleccion.mean()
      imagen = imagen.set('month', m)  
      #fecha = imagen.select("NDVI").date().format("YYYY-MM")
      imagen = imagen.set("system:time_start", fecha) 
      imagen = imagen.set("year", año) 
      return imagen
    
  def mosaico_mensual_2020(m):
      coleccion = coleccion2020.filter(ee.Filter.calendarRange(m, m, 'month')).select(['NDVI','NDWI','B4', 'B3', 'B2'])
      fecha = coleccion.first().date().format("YYYY-MM")
      año = coleccion.first().date().format("YYYY")
      imagen = coleccion.mean()
      imagen = imagen.set('month', m)  
      #fecha = imagen.select("NDVI").date().format("YYYY-MM")
      imagen = imagen.set("system:time_start", fecha) 
      imagen = imagen.set("year", año) 
      return imagen

  
  coleccionfiltrada2017 = ee.ImageCollection.fromImages(meses2017.map(lambda m: mosaico_mensual_2017(m)))
  coleccionfiltrada2018 = ee.ImageCollection.fromImages(meses2018.map(lambda m: mosaico_mensual_2018(m)))
  coleccionfiltrada2019 = ee.ImageCollection.fromImages(meses2019.map(lambda m: mosaico_mensual_2019(m)))
  coleccionfiltrada2020 = ee.ImageCollection.fromImages(meses2020.map(lambda m: mosaico_mensual_2020(m)))
                                                     
  coleccionfiltrada = coleccionfiltrada2017.merge(coleccionfiltrada2018)         
  coleccionfiltrada = coleccionfiltrada.merge(coleccionfiltrada2019)
  coleccionfiltrada = coleccionfiltrada.merge(coleccionfiltrada2020)
  
  with output:
   print("Calculo de estadisticas")

  def ndvi_medio(image):
      image1 = image.select("NDVI").rename("NDVI_medio")
      reduced = image1.reduceRegion(geometry=lote, reducer=ee.Reducer.mean(), scale=10)
      image = image.set(reduced)
      return image
  def ndvi_sd(image):
      image1 = image.select("NDVI").rename("NDVI_sd")
      reduced = image1.reduceRegion(geometry=lote, reducer=ee.Reducer.stdDev(), scale=10)
      image = image.set(reduced)
      return image
  def normalidad(image):
      image1 = image.select("NDVI").rename("normalidad")
      reduced = ee.Number(ee.Number(image1.get("NDVI_medio")).divide(ee.Number(image1.get("NDVI_mediana"))))
      image = image.set("normalidad",reduced)
      return image
  def ndvi_cv(image):
      sd = ee.Number(image.get("NDVI_sd"))
      medio = ee.Number(image.get("NDVI_medio"))
      cv = sd.divide(medio).multiply(100)
      image = image.set("cv",cv)
      return image
  

  coleccionfiltrada = coleccionfiltrada.map(lambda imagen: ndvi_medio(imagen))
  coleccionfiltrada = coleccionfiltrada.map(lambda imagen: ndvi_sd(imagen))
  coleccionfiltrada = coleccionfiltrada.map(lambda imagen: ndvi_cv(imagen))
  
  with output:
   print("Armando diccionario y descargando datos del servidor")

  fechas = coleccionfiltrada.aggregate_array("system:time_start")
  años = coleccionfiltrada.aggregate_array("year")

  NDVI_medio = coleccionfiltrada.aggregate_array("NDVI_medio")
  NDVI_sd = coleccionfiltrada.aggregate_array("NDVI_sd")
  meses = coleccionfiltrada.aggregate_array("month")
  
     
  test_dict = ee.Dictionary.fromLists(['system:time_start', 'NDVI_medio','NDVI_sd','años','mes'], [fechas, NDVI_medio,NDVI_sd,años,meses])
  featureCollection = ee.FeatureCollection([ee.Feature(None, test_dict)])
  
    
 
    
    
  link = featureCollection.getDownloadURL(filetype="CSV", selectors=None, filename=None)
  #Generamos el diccionario con los valores.
  import csv, urllib.request
  response = urllib.request.urlopen(link)
  lines = [l.decode('utf-8') for l in response.readlines()]
  reader = csv.DictReader(lines)

  with output:
   print("Covirtiendo a dataframe")

  data = list(reader)
  # Converting string to list
  fechas_cliente = data[0]["system:time_start"].strip('][').split(', ')
  ndvi_cliente = data[0]["NDVI_medio"].strip('][').split(', ')
  sd_ndvi_cliente = data[0]["NDVI_sd"].strip('][').split(', ')
  años_cliente = data[0]["años"].strip('][').split(', ')
  meses_cliente = data[0]["mes"].strip('][').split(', ')
  
  fechas_cliente = map(str, fechas_cliente)
  fechas = list(fechas_cliente)
  
  ndvi_cliente = map(float, ndvi_cliente)
  sd_ndvi_cliente = map(float, sd_ndvi_cliente)
  años_cliente = map(float, años_cliente)
  meses_cliente = map(float, meses_cliente)

  #Genero dataframe con los datos
  df = pd.DataFrame(list(zip(fechas, ndvi_cliente,sd_ndvi_cliente,años_cliente,meses_cliente)),columns =['Fecha', 'NDVI_medio',"sd_ndvi","años","meses"])
    
    
  config.coleccion = coleccionfiltrada
  config.lote = loteentero
  config.df = df

  with output:
   print("Exito!")
  return df, coleccionfiltrada,loteentero


#Graficamos lo que vemos
def graficar_series(output1,datos_lote):

    from matplotlib import pyplot as plt
    import numpy as np
    import pandas as pd
    

    #Si algun mes no tiene datos lo reemplazo or NULL
    for a in range(2017,2021):
     datos = datos_lote[datos_lote['años'] == a]
     for i in range(1,12):
      condicion = i in list(datos["meses"])
      if condicion is False:
       fecha =  (str(a)+"-"+str(i))
       df = pd.DataFrame([[fecha, None,None,a,i]],columns =['Fecha', 'NDVI_medio',"sd_ndvi","años","meses"])
       datos_lote = datos_lote.append(df)


    datos2017 = datos_lote[datos_lote['años'] == 2017].sort_values(by=['meses'])
    datos2018 = datos_lote[datos_lote['años'] == 2018].sort_values(by=['meses'])
    datos2019 = datos_lote[datos_lote['años'] == 2019].sort_values(by=['meses'])
    datos2020 = datos_lote[datos_lote['años'] == 2020].sort_values(by=['meses'])
    fechas= pd.Series([1,2,3,4,5,6,7,8,9,10,11,12])




    with output1:
        #fig, axs = plt.subplots(2,1)
        fig, ax = plt.subplots(1,1,figsize=(15,10))

        #2017
        transparencia=0.4
        colorlinea = '#CC4F1B'
        edgecolor= '#CC4F1B'
        facecolor= '#CC4F1B'
        ndvi =  datos2017["NDVI_medio"]
        sdmas =  datos2017["NDVI_medio"]+datos2017["sd_ndvi"]
        sdmenos =  datos2017["NDVI_medio"]-datos2017["sd_ndvi"]

        ax.plot(fechas, ndvi,'k-',color=colorlinea,label=2017)
        ax.fill_between(fechas,sdmenos,sdmas, alpha=transparencia, edgecolor=edgecolor, facecolor=facecolor,
            linewidth=4, antialiased=True)

        #2018
        transparencia=0.4
        colorlinea = '#1B2ACC'
        edgecolor= '#1B2ACC'
        facecolor= '#1B2ACC'
        ndvi =  datos2018["NDVI_medio"]
        sdmas =  datos2018["NDVI_medio"]+datos2018["sd_ndvi"]
        sdmenos =  datos2018["NDVI_medio"]-datos2018["sd_ndvi"]

        ax.plot(fechas, ndvi,'k-',color=colorlinea,label=2018)
        ax.fill_between(fechas,sdmenos,sdmas, alpha=transparencia, edgecolor=edgecolor, facecolor=facecolor,
            linewidth=4, antialiased=True)

        #2019
        transparencia=0.4
        colorlinea = '#3F7F4C'
        edgecolor= '#3F7F4C'
        facecolor= '#3F7F4C'
        ndvi =  datos2019["NDVI_medio"]
        sdmas =  datos2019["NDVI_medio"]+datos2019["sd_ndvi"]
        sdmenos =  datos2019["NDVI_medio"]-datos2019["sd_ndvi"]

        ax.plot(fechas, ndvi,'k-',color=colorlinea,label=2019)
        ax.fill_between(fechas,sdmenos,sdmas, alpha=transparencia, edgecolor=edgecolor, facecolor=facecolor,
            linewidth=4, antialiased=True)

        #2020
        transparencia=0.4
        colorlinea = '#F6FF33'
        edgecolor= '#F6FF33'
        facecolor= '#F6FF33'
        ndvi =  datos2020["NDVI_medio"]
        sdmas =  datos2020["NDVI_medio"]+datos2020["sd_ndvi"]
        sdmenos =  datos2020["NDVI_medio"]-datos2020["sd_ndvi"]

        ax.plot(fechas, ndvi,'k-',color=colorlinea,label=2020)
        ax.fill_between(fechas,sdmenos,sdmas, alpha=transparencia, edgecolor=edgecolor, facecolor=facecolor,
            linewidth=4, antialiased=True)

        ax.set_title('Evolucion de NDVI en KML cargado')
        #ax.set(xlabel="MES",ylabel="NDVI",fontsize=18)
        ax.set_ylabel('NDVI', fontsize = 20.0) # Y label
        ax.set_xlabel('MES', fontsize = 20) # X label
        ax.legend(prop=dict(size=18))

        ax.tick_params(labelsize=15)
        plt.savefig('temp/graficos/NDVI_historico.png',dpi=150)
        
    return

def generar_ambientesgee6(Map1,lista,lote,loteentero,ncluster):
  import ee
  import folium
  import branca.colormap as cm

  ##GRAFICAMOS LO QUE VEMOS
  
  palette = ['#d7191c','#fdae61', '#ffffbf',"#a6d96a","#1a9641"]
  limites = lote.centroid()
  centro = limites.getInfo()
  centro = centro['coordinates']
  centro_array = [centro[1],centro[0]]

  #GRAFICAMOS IMAGENES DE INTERES EN UN MAPA
  # Import the Folium library.
 
  # Define a method for displaying Earth Engine image tiles to folium map.
  def add_ee_layer(self, ee_image_object, vis_params, name):
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
      tiles = map_id_dict['tile_fetcher'].url_format,
      #tiles = "Stamen Terrain",
      attr = 'Map Data &copy; <a href="https://earthengine.google.com/"> Google Earth Engine, @FrancoFrolla, 2020</a>',
      name = name,
      overlay = True,
      control = True
    ).add_to(self)

  # Add EE drawing method to folium.
  folium.Map.add_ee_layer = add_ee_layer

  # Create a folium map object.
  my_map = folium.Map(location=centro_array, zoom_start=14,width='100%',height='100%')


  def ndvi_mediograficar(Map1,ncluster):
    coleccion = ee.ImageCollection(lista)
    coleccion = coleccion.map(lambda img: img.normalizedDifference(['B8', 'B4']).rename('NDVI'))
    
    laimagen = ee.Image(coleccion.select(["NDVI"]).mean())
    laimagen = laimagen.clip(loteentero)
    laimagen = laimagen.reproject(crs="EPSG:4326", scale=10)
    proj = laimagen.projection().getInfo()
    crs = proj['crs']
    laimagen = laimagen.resample('bilinear').reproject(crs=crs, scale=2)
    #probando 1/03
    texture = ee.Image(laimagen).select(["NDVI"]).reduceNeighborhood(reducer=ee.Reducer.median(),kernel=ee.Kernel.circle(30,"meters"))
    texture = ee.Image(laimagen).select(["NDVI"]).reduceNeighborhood(reducer=ee.Reducer.median(),kernel=ee.Kernel.circle(30,"meters"))
    laimagen = laimagen.addBands(texture.select("NDVI_median"))
    
    escenapiola = ee.Image(laimagen)
    
    #Arrancamos con los cluester
    training = escenapiola.sample(region= lote, scale= 2)
    clusterer = ee.Clusterer.wekaKMeans(ncluster).train(training);
    zones = escenapiola.cluster(clusterer);
     
     

    if ncluster == 2:
      paleta = ['ffeda0','f03b20']
      paleta2 = ['#ffeda0','#f03b20']
      listaparametros = [0,1]  
    if ncluster == 3:
      paleta = ['ffeda0','feb24c',"f03b20"]
      paleta2 = ['#ffeda0','#feb24c',"#f03b20"]
      listaparametros = [0,1,2]  
    if ncluster == 4:
      paleta = ['ffffb2','fecc5c',"fd8d3c","e31a1c"]
      paleta2 = ['#ffffb2','#fecc5c',"#fd8d3c","#e31a1c"]
      listaparametros = [0,1,2,3]  
    if ncluster == 5:
      paleta = ['ffffb2','fecc5c',"fd8d3c","f03b20","bd0026"]
      paleta2 = ['#ffffb2','#fecc5c',"#fd8d3c","#f03b20","#bd0026"]
      listaparametros = [0,1,2,3,4]  

 
   
    
    print("1) Se calculo el valor MEDIO de toda las imagenes ingresadas y fueron suavizadas interpolando el valor individual de cada pixel en un radio de 30 metros")
    print("2) El valor que representa cada zona representa a la agrupación echa por Fuzzy K Means")	
    
    vis_params = {
      "bands": ['cluster'],
      'min': 0,
      'max': ncluster,
      'palette': paleta}
    nombre = "Cluster"
    colormap = cm.LinearColormap(colors=paleta2,vmin=0,vmax=ncluster)
    colormap.add_to(my_map)
          

    #Antes era vis_params1
    my_map.add_ee_layer(zones, vis_params, "Productividad")
    Map1.addLayer(zones, vis_params, "Clusters")
    
    link = (zones.getThumbURL({
      "region": lote,
       "bands": ["cluster"],
      "dimensions": '600',
      "min": 0,
      "max": ncluster,
      'palette': paleta,
      "format": 'png'
    }))

    import urllib.request
    nombre = "Productividad"
    ruta = "temp/graficos/"+str(nombre)+".png"
    urllib.request.urlretrieve(link,ruta)

    return zones,listaparametros,coleccion


  zones,listaparametros,coleccion = ndvi_mediograficar(Map1,ncluster)
  # Add a layer control panel to the map.
  my_map.add_child(folium.LayerControl())

  # Display the map
  display(my_map)
  #graficar_barras2(zones,listaparametros,coleccion,lote)
  #zones = zones.rename(["NDVI_median"])
  return zones,listaparametros, coleccion


def graficar_barras2(zonas,listaparametros,coleccion,lote):
  import ee
  import numpy as np
  import matplotlib.pyplot as plt
  from scipy.stats import ttest_ind_from_stats
  from pingouin import pairwise_tukey
  import dataframe_image as dfi  
  #def tukeytest(dataset):
    #from statsmodels.stats.multicomp import pairwise_tukeyhsd
    #me fijo cuantas zonas hay
    #zonasdataset = list(dataset.columns) 
    # reshape the d dataframe suitable for statsmodels package 
    #d_melt = pd.melt(dataset.reset_index(), id_vars=['index'], value_vars=zonasdataset)
    # replace column names
   # d_melt.columns = ['index', 'treatments', 'value']
    # load packages
    # perform multiple pairwise comparison (Tukey HSD)
    #m_comp = pairwise_tukeyhsd(endog=d_melt['value'], groups=d_melt['treatments'], alpha=0.05)
   # print(m_comp)
  print("Para el calculo de medias y SD se tomo se separo calculo el valor medio de NDVI en cada imagen, para cada zona se calculo el NDVI relativo en cada imagen, obteniendo un NDVI normalizado por imagen.") 
  print("Si la zona de manejo tiene un valor cercano o igual a 1 representa el valor mas normal del lote, zonas > 1 indican zonas mas productivas, zonas<1 son de menor producción")
  print("El SD  informado corresponde a la variabilidad de  cada zona normalizada  entre el grupo de imagenes  seleccionadas.")

  def tukeypinguin(df):
    zonasdataset = list(df.columns) 
    # reshape the d dataframe suitable for statsmodels package 
    df = pd.melt(df.reset_index(), id_vars=['index'], value_vars=zonasdataset)
    # replace column names
    pt = pairwise_tukey(data=df, dv='value', between='variable')
    print(pt)


  #zonas = zonas.toInt()

  classes = ee.List(listaparametros)
      #.map(function(n) {
  def vectorizado(n):
    classImage = zonas.eq(ee.Number(n));
    vectors = classImage.updateMask(classImage).reduceToVectors(
      reducer= ee.Reducer.countEvery(), 
      geometry= lote, 
      scale= 5,
      ).geometry();
    feature = ee.Feature(vectors, {"class": n});
    return feature 
    

  #Genero los ambientes vectorizandolos en GEE

  classes = classes.map(lambda n: vectorizado(n))  
  ambientes = ee.FeatureCollection(classes)

  #aca arranca la deconstruccion

  numeroambientes = ambientes.size()

  listaambientes = ambientes.toList(numeroambientes)

  ambiente = listaambientes.get(0)
  
  # funcion para generar una lista donde se acumulan los valores de media
  medias = ee.List([])

  def sacarmedialote(image, lista):
    #Normalizo cada imagen dividiendolo por el valor medio de su lote.
    mediolote = image.select(["NDVI"]).reduceRegion(
      reducer= ee.Reducer.mean(),
      geometry= lote,
      scale= 10,
      maxPixels= 1e9)
    lista = ee.List(lista).add(mediolote.values().get(0));
    return lista 

  def sacarmedia(image, lista):
    #Extraigo el NDVI normalizado para cada region del lote
    mean = image.select(["NDVI"]).reduceRegion(
      reducer= ee.Reducer.mean(),
      geometry= ee.Feature(ambiente).geometry(),
      scale= 10,
      maxPixels= 1e9)
    lista = ee.List(lista).add(mean.values().get(0));
    return lista 
  
  def sacarsd(image, lista):
    #Normalizo cada imagen dividiendolo por el valor medio de su lote.
    #medialote = image.reduce(ee.Reducer.max());
    #image = image.divide(medialote)
    #Extraigo el NDVI normalizado para cada region del lote
    mean = image.select(["NDVI"]).reduceRegion(
      reducer= ee.Reducer.stdDev(),
      geometry= ee.Feature(ambiente).geometry(),
      scale= 10,
      maxPixels= 1e9)
    lista = ee.List(lista).add(mean.values().get(0));
    return lista 

  #Extraigo medias para 2 ambientes
  if numeroambientes.getInfo() == 2:
    ambiente = listaambientes.get(0)

    medias = ee.List([])
    acumulado = ee.List(coleccion.iterate(sacarmedia, medias));
    medias0 = acumulado.getInfo()
    medias = ee.List([])
    acumuladosd = ee.List(coleccion.iterate(sacarsd, medias));
    sd0 = acumuladosd.getInfo()
     

    ambiente = listaambientes.get(1)
    
    medias = ee.List([])
    acumulado = ee.List(coleccion.iterate(sacarmedia, medias));
    medias1 = acumulado.getInfo()
    medias = ee.List([])
    acumuladosd = ee.List(coleccion.iterate(sacarsd, medias));
    sd1 = acumuladosd.getInfo()
    
    #Calculo el valor medio del lote para cada imagen.
    medias = ee.List([])
    acumuladomedialote = ee.List(coleccion.iterate(sacarmedialote, medias));
    mediaslote1 = acumuladomedialote.getInfo()

    #print(medias0)
    #print(medias1)
    #print(mediaslote1)
    medias0 = np.array(medias0)
    medias1 = np.array(medias1)
    mediaslotes1 = np.array(mediaslote1)
    normalizado0 = np.divide(medias0,mediaslotes1)
    normalizado1 = np.divide(medias1,mediaslotes1)
    
 
    #pandas DataFrame
    import pandas as pd
    
    dataset = pd.DataFrame({'Zona1': normalizado0[:,], 'Zona2': normalizado1[:,]})
    
    df_styled = dataset.style.background_gradient() #adding a gradient based on values in cell
    dfi.export(df_styled,"temp/graficos/tabla_extracción.png")
    
    print(dataset)
    tukeypinguin(dataset)

    #aca los array
    media = np.array([np.mean(normalizado0), np.mean(normalizado1)])
    sd = np.array([np.std(normalizado0), np.std(normalizado1)])
    #sd = np.array([np.mean(sd0), np.mean(sd1)])

    
    #count = np.array([np.count(medias0), np.count(medias1)])
    materials = ['Zona 1', 'Zona 2']
    
  #Extraigo medias para 3 ambientes
  if numeroambientes.getInfo() == 3:
    ambiente = listaambientes.get(0)

    medias = ee.List([])
    acumulado = ee.List(coleccion.iterate(sacarmedia, medias));
    medias0 = acumulado.getInfo()
    medias = ee.List([])
    acumuladosd = ee.List(coleccion.iterate(sacarsd, medias));
    sd0 = acumuladosd.getInfo()
    
    ambiente = listaambientes.get(1)
    
    medias = ee.List([])
    acumulado = ee.List(coleccion.iterate(sacarmedia, medias));
    medias1 = acumulado.getInfo()
    medias = ee.List([])
    acumuladosd = ee.List(coleccion.iterate(sacarsd, medias));
    sd1 = acumuladosd.getInfo()

    ambiente = listaambientes.get(2)
    
    medias = ee.List([])
    acumulado = ee.List(coleccion.iterate(sacarmedia, medias));
    medias2 = acumulado.getInfo()
    medias = ee.List([])
    acumuladosd = ee.List(coleccion.iterate(sacarsd, medias));
    sd2 = acumuladosd.getInfo()

   #Calculo el valor medio del lote para cada imagen.
    medias = ee.List([])
    acumuladomedialote = ee.List(coleccion.iterate(sacarmedialote, medias));
    mediaslote1 = acumuladomedialote.getInfo()

    #print(medias0)
    #print(medias1)
    #print(mediaslote1)
    medias0 = np.array(medias0)
    medias1 = np.array(medias1)
    medias2 = np.array(medias2)

    mediaslotes1 = np.array(mediaslote1)
    normalizado0 = np.divide(medias0,mediaslotes1)
    normalizado1 = np.divide(medias1,mediaslotes1)
    normalizado2 = np.divide(medias2,mediaslotes1)

    
 
    #pandas DataFrame
    import pandas as pd
    
    dataset = pd.DataFrame({'Zona1': normalizado0[:,], 'Zona2': normalizado1[:,], 'Zona3': normalizado2[:,]})
    print(dataset)
    df_styled = dataset.style.background_gradient() #adding a gradient based on values in cell
    dfi.export(df_styled,"temp/graficos/tabla_extraccion.png")
    
                   
    tukeypinguin(dataset)

    #aca los array
    media = np.array([np.mean(normalizado0), np.mean(normalizado1), np.mean(normalizado2)])
    sd = np.array([np.std(normalizado0), np.std(normalizado1), np.std(normalizado2)])

    materials = ['Zona 1', 'Zona 2', 'Zona 3']
    

  #Extraigo medias para 4 ambientes
  if numeroambientes.getInfo() == 4:
    ambiente = listaambientes.get(0)

    medias = ee.List([])
    acumulado = ee.List(coleccion.iterate(sacarmedia, medias));
    medias0 = acumulado.getInfo()
    medias = ee.List([])
    acumuladosd = ee.List(coleccion.iterate(sacarsd, medias));
    sd0 = acumuladosd.getInfo()
    
    ambiente = listaambientes.get(1)
    
    medias = ee.List([])
    acumulado = ee.List(coleccion.iterate(sacarmedia, medias));
    medias1 = acumulado.getInfo()
    medias = ee.List([])
    acumuladosd = ee.List(coleccion.iterate(sacarsd, medias));
    sd1 = acumuladosd.getInfo()

    ambiente = listaambientes.get(2)
    
    medias = ee.List([])
    acumulado = ee.List(coleccion.iterate(sacarmedia, medias));
    medias2 = acumulado.getInfo()
    medias = ee.List([])
    acumuladosd = ee.List(coleccion.iterate(sacarsd, medias));
    sd2 = acumuladosd.getInfo()

    ambiente = listaambientes.get(3)
    
    medias = ee.List([])
    acumulado = ee.List(coleccion.iterate(sacarmedia, medias));
    medias3 = acumulado.getInfo()
    medias = ee.List([])
    acumuladosd = ee.List(coleccion.iterate(sacarsd, medias));
    sd3 = acumuladosd.getInfo()

     #pandas DataFrame
     #Calculo el valor medio del lote para cada imagen.
    medias = ee.List([])
    acumuladomedialote = ee.List(coleccion.iterate(sacarmedialote, medias));
    mediaslote1 = acumuladomedialote.getInfo()

    #print(medias0)
    #print(medias1)
    #print(mediaslote1)
    medias0 = np.array(medias0)
    medias1 = np.array(medias1)
    medias2 = np.array(medias2)
    medias3 = np.array(medias3)


    mediaslotes1 = np.array(mediaslote1)
    normalizado0 = np.divide(medias0,mediaslotes1)
    normalizado1 = np.divide(medias1,mediaslotes1)
    normalizado2 = np.divide(medias2,mediaslotes1)
    normalizado3 = np.divide(medias3,mediaslotes1)
  
 
    #pandas DataFrame
    import pandas as pd
    
    dataset = pd.DataFrame({'Zona1': normalizado0[:,], 'Zona2': normalizado1[:,], 'Zona3': normalizado2[:,], 'Zona4': normalizado3[:,]})
    print(dataset)
    df_styled = dataset.style.background_gradient() #adding a gradient based on values in cell
    dfi.export(df_styled,"temp/graficos/tabla_extraccion.png")
    
    
    tukeypinguin(dataset)

    #aca los array
    media = np.array([np.mean(normalizado0), np.mean(normalizado1), np.mean(normalizado2), np.mean(normalizado3)])
    sd = np.array([np.std(normalizado0), np.std(normalizado1), np.std(normalizado2), np.std(normalizado3)])
    
    materials = ['Zona 1', 'Zona 2', 'Zona 3', 'Zona 4']
    

  #Extraigo medias para 5 ambientes
  if numeroambientes.getInfo() == 5:
    ambiente = listaambientes.get(0)

    medias = ee.List([])
    acumulado = ee.List(coleccion.iterate(sacarmedia, medias));
    medias0 = acumulado.getInfo()
    medias = ee.List([])
    acumuladosd = ee.List(coleccion.iterate(sacarsd, medias));
    sd0 = acumuladosd.getInfo()
    
    ambiente = listaambientes.get(1)
    
    medias = ee.List([])
    acumulado = ee.List(coleccion.iterate(sacarmedia, medias));
    medias1 = acumulado.getInfo()
    medias = ee.List([])
    acumuladosd = ee.List(coleccion.iterate(sacarsd, medias));
    sd1 = acumuladosd.getInfo()

    ambiente = listaambientes.get(2)
    
    medias = ee.List([])
    acumulado = ee.List(coleccion.iterate(sacarmedia, medias));
    medias2 = acumulado.getInfo()
    medias = ee.List([])
    acumuladosd = ee.List(coleccion.iterate(sacarsd, medias));
    sd2 = acumuladosd.getInfo()

    ambiente = listaambientes.get(3)
    
    medias = ee.List([])
    acumulado = ee.List(coleccion.iterate(sacarmedia, medias));
    medias3 = acumulado.getInfo()
    medias = ee.List([])
    acumuladosd = ee.List(coleccion.iterate(sacarsd, medias));
    sd3 = acumuladosd.getInfo()

    ambiente = listaambientes.get(4)
    
    medias = ee.List([])
    acumulado = ee.List(coleccion.iterate(sacarmedia, medias));
    medias4 = acumulado.getInfo()
    medias = ee.List([])
    acumuladosd = ee.List(coleccion.iterate(sacarsd, medias));
    sd4 = acumuladosd.getInfo()

    #Calculo el valor medio del lote para cada imagen.
    medias = ee.List([])
    acumuladomedialote = ee.List(coleccion.iterate(sacarmedialote, medias));
    mediaslote1 = acumuladomedialote.getInfo()

    #print(medias0)
    #print(medias1)
    #print(mediaslote1)
    medias0 = np.array(medias0)
    medias1 = np.array(medias1)
    medias2 = np.array(medias2)
    medias3 = np.array(medias3)
    medias4 = np.array(medias4)

    mediaslotes1 = np.array(mediaslote1)
    normalizado0 = np.divide(medias0,mediaslotes1)
    normalizado1 = np.divide(medias1,mediaslotes1)
    normalizado2 = np.divide(medias2,mediaslotes1)
    normalizado3 = np.divide(medias3,mediaslotes1)
    normalizado4 = np.divide(medias4,mediaslotes1)


    #pandas DataFrame
    import pandas as pd
    
    dataset = pd.DataFrame({'Zona1': normalizado0[:,], 'Zona2': normalizado1[:,], 'Zona3': normalizado2[:,], 'Zona4': normalizado3[:,], 'Zona5': normalizado4[:,]})
    print(dataset)
    df_styled = dataset.style.background_gradient() #adding a gradient based on values in cell
    dfi.export(df_styled,"temp/graficos/tabla_extraccion.png")
    
        
    tukeypinguin(dataset)

    #aca los array
    media = np.array([np.mean(normalizado0), np.mean(normalizado1), np.mean(normalizado2), np.mean(normalizado3), np.mean(normalizado4)])
    sd = np.array([np.std(normalizado0), np.std(normalizado1), np.std(normalizado2), np.std(normalizado3), np.std(normalizado4)])


    materials = ['Zona 1', 'Zona 2', 'Zona 3', 'Zona 4','Zona 5']
    
  
  
  
  # Build the plot
  fig, ax = plt.subplots()
  x_pos = np.arange(len(materials))
  ax.bar(x_pos, media, yerr=sd, align='center', alpha=0.5, ecolor='black', capsize=10)
  #ax.errorbar(x_pos, media, yerr=sd,fmt='o', marker='.')

  ax.set_ylabel('NDVI medio de la zona')
  ax.set_xticks(x_pos)
  ax.set_xticklabels(materials)
  ax.set_title('NDVI normalizado por media del lote en cada imagen')

  minimo = (np.ndarray.min(media)-np.ndarray.max(sd))*0.90
  maximo = (np.ndarray.max(media)+np.ndarray.max(sd))*1.10

  ax.set_ylim((minimo, maximo))

  # Save the figure and show
  plt.tight_layout()
  plt.savefig('temp/graficos/estadistica_zonas.png',dpi=150)
  #plt.show()
  return media,sd
  

def salidagrafica(zones,loteentero,df1,resolucion,listaparametros,media,sd,nombrelote):
    import ee
    import geopandas as gpd
    import pandas as pd
    from scipy.interpolate import griddata
    import matplotlib.pyplot as plt
    import numpy as np
    from ambientadorffnp_local2 import grilla_poligonos
    #hay que acomodar que la zona 0 pase a valer 1 y sucesivamente...


    def customRemap(image, upperLimit, newValue): 
     mask = image.eq(upperLimit)
     image = image.where(mask, newValue)
     return image

    bandNames = zones.bandNames().getInfo();

    if bandNames[0] == "NDVI_median":
        if media.size == 2: 
           zones = customRemap(zones.select("NDVI_median"), listaparametros[0], 1)
           zones = customRemap(zones, listaparametros[1], 2)

        if media.size == 3: 
           zones = customRemap(zones.select("NDVI_median"), listaparametros[0], 1)
           zones = customRemap(zones, listaparametros[1], 2)
           zones = customRemap(zones, listaparametros[2], 3) 

        if media.size == 4: 
           zones = customRemap(zones.select("NDVI_median"), listaparametros[0], 1)
           zones = customRemap(zones, listaparametros[1], 2)
           zones = customRemap(zones, listaparametros[2], 3) 
           zones = customRemap(zones, listaparametros[3], 4) 

        if media.size == 5: 
           zones = customRemap(zones.select("NDVI_median"), listaparametros[0], 1)
           zones = customRemap(zones, listaparametros[1], 2)
           zones = customRemap(zones, listaparametros[2], 3) 
           zones = customRemap(zones, listaparametros[3], 4) 
           zones = customRemap(zones, listaparametros[4], 5)

        zones = zones.toInt()

    print("Extrayendo datos de gee")
    vectors = zones.select(bandNames[0]).reduceToVectors(
        geometry= loteentero,
        scale= 5,
        geometryType= 'polygon',
        eightConnected= False,
        labelProperty= 'zone',
        )
      
    data = vectors.getInfo()

    from shapely.geometry import shape
    print("Convirtiendolos a geopandas, generando mascara <1ha")

    data1 = (data["features"])

    for d in data1:
     d['geometry'] = shape(d['geometry'])

    gdf = gpd.GeoDataFrame(data1).set_geometry('geometry')
    datos = gdf["properties"].apply(pd.Series)
    datos.columns = ["count","zona"]
    geometria = gdf["geometry"]
    datos["geometry"] = geometria
    gdf1 = gpd.GeoDataFrame(datos)
    gdf1.crs = "EPSG:4326"
    gdf2 =gdf1.to_crs("EPSG:3857")
    gdf2["area"] = (gdf2['geometry'].area)/10000
    gdf2["mask"] = gdf2["area"] > 1
    ###EN gdf3 se guardan los poligonos con una suerficie interesante.
    mask = gdf2["mask"]
    gdf3 = gdf2.loc[mask]
    gdf3 = gdf3.reset_index(drop=True)


    if bandNames[0] != "NDVI_median":
     gdf3["zona"] = gdf3["zona"] +1 
      
    #genero grilla de muestreo
    grilla = grilla_poligonos(gdf3,resolucion)
    grilla["centros"] = grilla.centroid
    grilla.set_geometry("centros")
      
    grilla.crs = "EPSG:3857"
    gdf3.crs = "EPSG:3857"
      
    print("Extrayendo valores de la capa de gee con la grilla")
    lospuntos = gpd.sjoin(gdf3, grilla,how="right", op='intersects')
    lospuntos = pd.DataFrame.drop_duplicates(lospuntos, subset=["orden"], keep="first", inplace=False)
    x_coords = lospuntos.centroid.x.astype("int64")
    y_coords = lospuntos.centroid.y.astype("int64")
    lospuntos["X"] = x_coords
    lospuntos["Y"] = y_coords
    #del lospuntos["mask"]
    lospuntos = lospuntos[["X","Y","centros","geometry","zona"]]


    #Obtengo la forma de la matriz
    numpyx = lospuntos["X"].values
    numpyx = np.unique(numpyx)
    largo_x = len(numpyx)
    numpyy = lospuntos["Y"].values
    numpyy = np.unique(numpyy)
    largo_y = len(numpyy)
    Y = np.empty((largo_x,largo_y))
    X = np.empty((largo_x,largo_y))
    x_ordenado = np.sort(numpyx)
    #la forma es 88 filas en X y 133 columnas en Y, ahora a rellenar las filas

    for indicador in range(0,largo_x):
     sera = lospuntos['X'] == x_ordenado[indicador]
     X[indicador] = x_ordenado[indicador]
     lospuntos1 = lospuntos[sera]
     numpy_1 = lospuntos1["Y"].values
     y_ordenado_1 = np.sort(numpy_1)
     Y[indicador] = y_ordenado_1

    print("Aplicar filtro de vecindad para suavizar areas")
    from scipy.interpolate import griddata
    import pandas as pd
    lospuntos = pd.DataFrame.dropna(lospuntos)
    y=np.array(lospuntos["Y"]) 
    x=np.array(lospuntos["X"]) 
    z=np.array(lospuntos["zona"])

    lospuntos["zona"].describe()

    points = np.column_stack((x,y))

    grid_z0 = griddata(points, z, (X, Y), method='nearest')


    print("Reconstruyo el dataset con los datos interpolados")
    flatx = (X.flatten())
    flaty = (Y.flatten())
    flatz = (grid_z0.flatten())

    dataset = pd.DataFrame({'X': flatx, 'Y': flaty,'Z': flatz})
    df = gpd.GeoDataFrame(dataset, geometry=gpd.points_from_xy(dataset.X, dataset.Y))
    #df.plot(column="Z",legend=True,missing_kwds={"color": "lightgrey", "edgecolor": "red","hatch": "///","label": "Missing values"},)
    df["ubicacion"] = df["X"] * df ["Y"]
    lospuntos["ubicacion"] = lospuntos["X"] * lospuntos ["Y"]

    print("Coheciona el dataset con la grilla con los valores interpolados")
    x_coords = grilla.centroid.x.astype("int64")
    y_coords = grilla.centroid.y.astype("int64")
    grilla["X"] = x_coords
    grilla["Y"] = y_coords
    grilla['X'].astype(str).astype("int64")
    grilla['Y'].astype(str).astype("int64")
    grilla["zona"] = grilla["X"] * grilla ["Y"]
    grilla.head()


    #lospuntos2 = grilla

    def idxDict(x):
     row = df.loc[df['ubicacion'] == x]
     x1 = row["Z"].values[0]
     return x1
      
    grilla['zonas'] = grilla['zona'].map(idxDict)


    if media.size == 2:
     grilla.loc[grilla['zonas'] <= 1.5, 'zonas'] = media[0]
     grilla.loc[(grilla['zonas'] > 1.5), 'zonas'] = media[1]
     nzonas = ["Zona 1","Zona 2"]
      
    if media.size == 3:
     grilla.loc[grilla['zonas'] <= 1.5, 'zonas'] = media[0]
     grilla.loc[(grilla['zonas'] > 1.5) & (grilla['zonas'] <= 2.5), 'zonas'] = media[1]
     grilla.loc[(grilla['zonas'] > 2.5), 'zonas'] = media[2]
     nzonas = ["Zona 1","Zona 2","Zona 3"]
       
    if media.size == 4:
     grilla.loc[grilla['zonas'] <= 1.5, 'zonas'] = media[0]
     grilla.loc[(grilla['zonas'] > 1.5) & (grilla['zonas'] <= 2.5), 'zonas'] = media[1]
     grilla.loc[(grilla['zonas'] > 2.5) & (grilla['zonas'] <= 3.5), 'zonas'] = media[2]
     grilla.loc[(grilla['zonas'] > 3.5), 'zonas'] = media[3]
     nzonas = ["Zona 1","Zona 2","Zona 3","Zona 4"]

    if media.size == 5:
     grilla.loc[grilla['zonas'] <= 1.5, 'zonas'] = media[0]
     grilla.loc[(grilla['zonas'] > 1.5) & (grilla['zonas'] <= 2.5), 'zonas'] = media[1]
     grilla.loc[(grilla['zonas'] > 2.5) & (grilla['zonas'] <= 3.5), 'zonas'] = media[2]
     grilla.loc[(grilla['zonas'] > 3.5) & (grilla['zonas'] <= 4.5), 'zonas'] = media[3]
     grilla.loc[(grilla['zonas'] > 4.5), 'zonas'] = media[4]
     nzonas = ["Zona 1","Zona 2","Zona 3","Zona 4","Zona 5"]

    df = grilla[["geometry","zonas"]]

    df = df.dissolve(by="zonas",as_index=False)

    print("Recorte area de lote")
    df1 = gpd.GeoDataFrame(df1).set_geometry('geometry')
    df1.crs = "EPSG:3857"
        
    df_cortado = gpd.overlay(df, df1, how='intersection')
    df_cortado.to_crs("EPSG:4326")
    print(df_cortado)

    #df_cortado["sd_ndvi"] = sd
    #df_cortado["Zonas"] = nzonas
    #df_cortado["Zonas"] = ["Zona 1","Zona 2","Zona 3","Zona 4","Zona 5"]
    #nzonas = ["Zona 1","Zona 2","Zona 3","Zona 4","Zona 5"]
    #df_cortado.columns = ['media_ndvi','geometry','sd','Zonas']
    df_cortado['geometry'] =  df_cortado.geometry.buffer(-0.001)
      
    import os
    direccion = os.getcwd()
      
    df_cortado.plot(column="zonas",legend=True)
    rutasalida = direccion+"/" + nombrelote +".shp"
    df_cortado.to_file(rutasalida)
    return df_cortado


def cluster_historicos_s2(lote,mes_inicio,sumar_meses):
    
    import ee
    import config
    
    #Map the function over one year of data.
    coleccionfiltrada = ee.ImageCollection('COPERNICUS/S2').filterBounds(lote.bounds())
    #filtro de duplicadas
    lista = coleccionfiltrada.toList(coleccionfiltrada.size())
    imagen = ee.Image(lista.get(0))
    lista = lista.add(imagen)

    def detectar_duplicador(imagen):
            esduplicado = ee.String("")
            numero = lista.indexOf(imagen)
            imagen1 = ee.Image(lista.get(numero.add(1)))
            #Compare the image(0) in the ImageCollection with the image(1) in the List
            fecha1 = imagen.date().format("Y-M-d")
            fecha2 = imagen1.date().format("Y-M-d")
            estado = ee.Algorithms.IsEqual(fecha1,fecha2)
            esduplicado = ee.String(ee.Algorithms.If(estado,"duplicado","no duplicado"));
            imagen = imagen.set("duplicado", esduplicado)
            return imagen

    coleccionfiltrada = coleccionfiltrada.map(lambda image: detectar_duplicador(image))
    coleccionfiltrada = coleccionfiltrada.filter(ee.Filter.eq('duplicado', "no duplicado"))


    #Recorte de lote
    coleccionfiltrada = coleccionfiltrada.map(lambda img: img.clip(config.lote.dissolve()))

    def agregar_nubes(image): 
          meanDict = image.reduceRegion(
          reducer= ee.Reducer.anyNonZero(),
          geometry= lote,
          scale= 10,
          )
          image = image.set("mascara",meanDict.get("QA60"))
          return image


    print("Pasando por filtro de nubes")
    coleccionfiltrada = coleccionfiltrada.map(lambda image: agregar_nubes(image));
    coleccionfiltrada = coleccionfiltrada.filterMetadata('mascara', 'equals', 0);

    def ndvi(img):
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        img = img.addBands(ndvi)
        return img

    coleccionfiltrada = coleccionfiltrada.map(lambda img: ndvi(img))

    """
    Enmascaro todos los pixeles con NDVI menor a 0.05
    """

    def enmascarar_pixeles(image):
       ndvi = image.select('NDVI');
       #THRESHOLD
       # if NDVI less or equal to 0 => 0 else 1
       mask = ndvi.gte(0.05).rename('mask')
       ndvi_mask = ndvi.updateMask(mask).rename(['NDVI2'])
       image = image.addBands(ndvi_mask);
       return image


    coleccionfiltrada = coleccionfiltrada.map(lambda image: enmascarar_pixeles(image))



    years = ee.List.sequence(2016,2021,1)
    #years = list(range(1984,2013,1))

    def fechas(img):
          fecha = img.date().format("YYYY-MM")
          año = img.date().format("YYYY")
          mes = img.date().format("M")
          img = img.set('month', mes)  
          img = img.set("system:time_piola", fecha) 
          img = img.set("year", año) 
          return img

    coleccionfiltrada = coleccionfiltrada.map(lambda imagen:fechas(imagen))
    coleccionfiltrada = coleccionfiltrada.select(["NDVI2"])
    coleccionfiltradaS2 = coleccionfiltrada

    def listamedias(año):
        startDate = ee.Date.fromYMD(año, mes_inicio, 1)
        endDate = startDate.advance(sumar_meses, 'month')
        filtered = coleccionfiltrada.filter(ee.Filter.date(startDate, endDate))
        imagen = ee.Image(filtered.mean().set('year', año))
        return imagen

    coleccionfiltrada = years.map(lambda año: listamedias(año))
    coleccionfiltrada = ee.ImageCollection(coleccionfiltrada)

    porcentiles = list(range(50,105,5))
    percentile = ee.List.sequence(50,100,5)



    nombres = []

    for quantil in porcentiles:
        nombrebanda = "p"+str(quantil)
        nombres.append(nombrebanda)   

    per = coleccionfiltrada.reduce(ee.Reducer.percentile(percentile,nombres))

    """
    Clasificacion no supervisada 6 custer

    """
    training = per.sample(region= lote, scale= 10)

    clusterer = ee.Clusterer.wekaKMeans(6).train(training);
    result = per.cluster(clusterer);

    #Suavizado capa raster 21/10/2021
    laimagen = result.reproject(crs="EPSG:4326", scale=10)
    proj = laimagen.projection().getInfo()

    crs = proj['crs']
    laimagen = laimagen.resample('bilinear').reproject(crs=crs, scale=5)
    #texture = ee.Image(laimagen).reduceNeighborhood(reducer=ee.Reducer.median(),kernel=ee.Kernel.circle(30,"meters"))
    #texture = ee.Image(laimagen).reduceNeighborhood(reducer=ee.Reducer.median(),kernel=ee.Kernel.circle(30,"meters"))


    result = laimagen.unmask(-9999)
    result6cluster = result.clip(lote)

    """
    Fecha de generacion
    
    """

    from datetime import date
    from datetime import datetime


    #Fecha actual
    now = datetime.now()

    #Datetime
    origen = (str(now.year)+"_"+str(now.month)+"_"+str(now.day)+"_"+str(now.hour)+":"+str(now.minute)+":"+str(now.second))


    """
    Descarga de 6 cluster

    """
    print("Descargando 6 cluster para S2...")
    path = result6cluster.getDownloadUrl({
          'name': str("6_cluster_")+origen,
          'scale' : 10,
          'region' : config.lote})
    print(path)

    """
    Clasificacion no supervisada 3 cluster

    """

    clusterer = ee.Clusterer.wekaKMeans(3).train(training);
    result = per.cluster(clusterer);

     #Suavizado capa raster 21/10/2021
    laimagen = result.reproject(crs="EPSG:4326", scale=10)
    proj = laimagen.projection().getInfo()

    crs = proj['crs']
    laimagen = laimagen.resample('bilinear').reproject(crs=crs, scale=5)

    result = laimagen.unmask(-9999)
    result3cluster = result.clip(lote)

    

    """
    Descarga de 3 cluster

    """
    print("Descargando 3 cluster para S2...")
    path = result3cluster.getDownloadUrl({
          'name': str("3_cluster_")+origen,
          'scale' : 10,
          'region' : config.lote})
    print(path)
    
    return result6cluster,result3cluster
