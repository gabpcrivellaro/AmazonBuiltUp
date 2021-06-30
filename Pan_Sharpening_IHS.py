#####################################################################################################################
#  Algotitmo para fusão de imagens pelo método IHS (DOU et al., 2007) das imagens do sensor WPM a bordo do CBERS-4A #
#####################################################################################################################
#Importar as bibliotecas
import ee
ee.Initialize()
#Cria uma geometria do estado do pará a ser utilizada para adiquirir as imagens VIIRS
shp_f = ee.FeatureCollection("users/gabrielcrivellarog/PA")
geometry=shp_f.geometry().bounds()
#Cria uma geometria de pontos das sedes municipais escolhidas
sedes_fc = ee.FeatureCollection('users/gabrielcrivellarog/sedes_mun_estudo')
sedes = sedes_fc.geometry()
#Define o intervalo de datas para aquisição das imagens CBERS
START_DATE = ee.Date('2020-01-01')
END_DATE = ee.Date('2020-12-31')
#Define o intervalo de datas para aquisição dos dados VIIRS
d1 = '2020-05-01'
d2 = '2020-10-31'
#Carrega a coleção da imagens do CBERS WPM Multiespectral
MULTI_C= ee.ImageCollection('users/gabrielcrivellarog/CBERS_WPM_MULT')
#Carrega a coleção da imagens do CBERS WPM Pancromática
PAN_C= ee.ImageCollection('users/gabrielcrivellarog/CBERS_WPM_PAN')
#Carrega a coleção da imagens VIIRS par o estado do PA e para as datas definidas
col_viirs=ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG').filterBounds(geometry).filterDate(d1,d2) 
#Mostra uma mensagem informado que a região de intersse foi criada e definidas as datas
#Cria Função que adiociona um campo (sede) no vetor poligonos de áreas iluminadas (VIIRS) 
# e atribui 1 para os poligonos que intersectam os pontos das sedes municipais e 0 que não intersectam.
def cruzaSede(ft):
    return ft.set('sede',ft.intersects(sedes))
#Mostra uma mensagem informando que as funções foram criadas
print('Funções ok!')
#Calcula a média das composições mensais do VIIRS dentre o período definido
nighttime=col_viirs.mean().clip(geometry).select('avg_rad')
#Cria uma imagem bináriacom valor de 1 para os pixels com valor maior que 1 e 0 para os pixels com valore menor que 1
zones = nighttime.gt(1)
#Vetoriza a imagem binária do viirs
vectors = zones.addBands(nighttime).reduceToVectors(geometry = geometry,
                                                     crs="EPSG:31981",
                                                     scale = 500,
                                                     geometryType = 'polygon',
                                                     eightConnected = True,
                                                     labelProperty = 'zone',
                                                     reducer = ee.Reducer.mean(),
                                                    maxPixels=1571869600 )
#Utiliza a função cruzaSede identificar os poligonos de áreas iluminadas que intersectam as sedes municipais de interesse
vectors = vectors.map(cruzaSede)
#cria um vetor apenas com os poligonos que possuem valor 1 (True) no campo "sede"
vectors_sedes =vectors.filterMetadata('sede','equals',True)
#Cria uma lista com o ID de cada um dos poligonos do vetor vectors_sedes
sede_id = vectors_sedes.aggregate_array('system:index').getInfo()
#Cria um filtro para identificar a intersecção entre duas geometrias
filter = ee.Filter.intersects(leftField='.geo',rightField='.geo')
#Define uma operação para unir duas camadas vetoriais (sedes e poligonos iluminados) e cria um campo com a palavra 'SEDE'
join = ee.Join.saveFirst(matchKey = 'SEDE')
#Cria uma nova camada vetorial dos poligonos iluminados em sedes municipais 
#com os campos de informações do vetor de pontos de sedes municipais (Nome das sedes e sigla das sedes)
vectors_joined = join.apply(vectors_sedes, sedes_fc,filter)
#Cria uma lista vazia para armazenar os nomes dos municipios das sedes analisadas
mun_nomes_interesse = []
#Cria uma variável de contagem para exibir uma contagem
c=1
#Cria uma geometria das áreas iluminadas nas sedes de estudo
AOI1 = vectors_sedes.geometry() 
#Conta o número de poligonos de áreas iluminadas que intersectaram as sedes que não intersectaram
cruza_sede_result = vectors.aggregate_histogram('sede').getInfo()
#Mostra o total de poligonos de áreas iluminadas que não intersectaram as sedes
print('Total de poligonos de áreas iluminadas: ',cruza_sede_result['false'])
#Mostra o total de poligonos de áreas iluminadas que intersectaram as sedes
print('Total de poligonos de áreas iluminadas em sedes municipais de interesse: ',cruza_sede_result['true'])
#Define um laço para mostrar o nome dos municipios em que os poligonos iluminados intersectaram as sedes
for i in sede_id:
    #Interendo entre os IDs dos poligonos iluminados em sedes
    #cria uma nova camada vetorial com o poligono que possue mesmo ID (index) 
    get_name = vectors_joined.filterMetadata('system:index','equals',i)
    #Cria uma variável string com o nome do municipio ao qual o poligono de área iluminada intersecta a sede
    Sede_nome = ee.Feature(get_name.first().get('SEDE')).get('NAME').getInfo()
    #adiciona o nome do municipio a lista com o nome dos muncipios das sedes
    mun_nomes_interesse.append(Sede_nome)
    #Mostra o nome do municipio com um número de contagem
    print(c,'-',Sede_nome)
    #Adiciona 1 ao valor de c para o contador
    c=c+1
#Mostra uma linha vazia para separar as informações
print()
#Mostra uma mensagem informando que a AOI1 foi criada com sucesso
print('AOI1 pronto!')
#Laço que intera em cada ID presente na lista de ID dos poligonos sede
for i in sede_id:
    print('Exportando imagens!')
    #Cria uma geometria somente com o poligono de uma sede
    AOI_sede = vectors_sedes.filterMetadata('system:index','equals',i).geometry()
    #Cria uma FC com o poligono com os nome da sede
    get_name = vectors_joined.filterMetadata('system:index','equals',i)
    #Cria uma variável string com o nome da sede
    Sede_nome = ee.Feature(get_name.first().get('SEDE')).get('NAME').getInfo()
    #Cria uma variável string com a sigla da sede
    Sede_nome_sigla = ee.Feature(get_name.first().get('SEDE')).get('SIGLA').getInfo()
    s = Sede_nome_sigla
    
    #Mostra uma mensagen informando o nome da sede que iniciou o processamento
    print (Sede_nome, ' Iniciado!')
    
    #Filtra as imagens da coleção CBERS WPM Multiespectral que intersectam a AOI da sede e atendem ao intevalo de datas definido 
    MULTI = MULTI_C.filterBounds(AOI_sede).filterDate(START_DATE,END_DATE).mosaic().clip(AOI1).double().divide(1000)
    #Filtra as imagens da coleção CBERS WPM Pancromática que intersectam a AOI da sede e atendem ao intevalo de datas definido 
    PAN = PAN_C.filterBounds(AOI_sede).filterDate(START_DATE,END_DATE).mosaic().clip(AOI1).double().divide(1000)
    # Converte as bandas RGB para HSV 
    hsv = MULTI.select(['b3','b3', 'b1']).rgbToHsv()
    # Passa para a pancromática e converte para RGB
    sharpened = ee.Image.cat([hsv.select('hue'), hsv.select('saturation'), PAN.select('b1')]).hsvToRgb()
    #Exporta as imagens para o google drive
    name = s + '_CBERS_PSH_B123'
    task_im = ee.batch.Export.image.toDrive(image = sharpened, description=name, folder='PSH', scale=2, crs = 'EPSG:32722',maxPixels=1483386531140)
    task_im.start()
    #Mostra uma mensagem informando que o prcessamento de uma das sedes foi concluído
    print(Sede_nome, 'Pronto!')
print('Todas imagens foram exportadas!')

