#######################################################################################################################
#                                   AmazonBuiltUP - Single Random Forest - SRF                                        #
#######################################################################################################################
#Algoritmo desenvolvido para mapear áreas construídas em sedes municipais de cidades amazônicas                       #
#Esse script foi criado para classificar imaagens Sentinel-2, também são utilizados dados do VIIRS,                   #
#pontos de sedes municipais (IBGE) e amostras de treinamento                                                          #
#O mapa temático final possui 5 classes: Solo Exposto; Área Construída, Água, Vegetação Arbórea e Vegetação Herbácea  #
#Autor: Gabriel Crivellaro Gonçalves                                                                                  #
#######################################################################################################################

#Importar as bibliotecas
import ee
ee.Initialize()
#Mostra uma mensagem informando que as bibliotecas foram importadas com sucesso
print('Imports ok!')
#Cria uma geometria do estado do pará a ser utilizada para adiquirir as imagens VIIRS
shp_f = ee.FeatureCollection("users/gabrielcrivellarog/PA")
geometry=shp_f.geometry().bounds()
#Cria uma geometria de pontos das sedes municipais escolhidas
sedes_fc = ee.FeatureCollection('users/gabrielcrivellarog/sedes_mun_estudo')
sedes = sedes_fc.geometry()
#Define o intervalo de datas para aquisição das imagens sentinel-2
START_DATE = ee.Date('2020-01-01')
END_DATE = ee.Date('2020-12-31')
#Define o intervalo de datas para aquisição dos dados VIIRS
d1 = '2020-05-01'
d2 = '2020-10-31'
#Carrega a coleção da imagens do Sentinel-2 SR
s2Sr_col = ee.ImageCollection('COPERNICUS/S2_SR')
#Carrega a coleção de mascaras de núvens do Sentinel-2
s2Clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
#Carrega a coleção da imagens VIIRS par o estado do PA e para as datas definidas
col_viirs=ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG').filterBounds(geometry).filterDate(d1,d2) 
#Mostra uma mensagem informado que a região de intersse foi criada e definidas as datas
print('ROI e datas ok!')
#Cria Função para o calculo do GLCM
#quant_levels = número de níveis de cinza a ser requantizado
#scale = tamanho do pixel (m)
#size = tamanho da janela kernel (1 = 3x3; 2 = 5x5)
def compute_probabilistic_glcm(img,quant_levels,scale,size,geom):
    qt_img = probabilistic_quantizer(img,quant_levels,scale,geom).rename(['glcm'])
    return qt_img.glcmTexture(size)
#Cria Função para para requantizar a imagem pelo metodo de probabilidades iguais
def probabilistic_quantizer(img,num_levels,scale,geom):
    img = ee.Image(img)
    num_levels = ee.Number(num_levels)
    region = ee.Geometry(geom)
    percentiles = ee.List.sequence(0,num_levels).map(lambda i: ee.Number(i).multiply(100).divide(num_levels))
    reducer = ee.Reducer.percentile(percentiles)
    quantis = img.reduceRegion(reducer = reducer, geometry = region, scale = scale, maxPixels = 25000000000000000, bestEffort = True)
    quantis_values = quantis.values(quantis.keys()).sort()

    def get_fraction(pair):
        pair = ee.List(pair)
        low_bound  = ee.Number(pair.get(0))
        high_bound = ee.Number(pair.get(1))
        return ee.Image(1).updateMask(img.gt(low_bound).And(img.lt(high_bound)))

    img_req_col = ee.ImageCollection(quantis_values.zip(quantis_values.slice(1)).map(get_fraction))

    l=ee.List.sequence(0,num_levels.subtract(1))
    img_req_col_list = img_req_col.toList(num_levels)

    def create_quantized_col(i):
        i = ee.Number(i)
        return ee.Image(img_req_col_list.get(i)).multiply(ee.Number(i).add(1)).toByte()

    mul_col = ee.ImageCollection(l.map(create_quantized_col))

    return mul_col.mosaic()
def spectral_glcm_band_name(glcm_image,spectral_band):
    bandname = spectral_band.bandNames().getInfo()
    bandname = str(bandname[0])
    new_glcm_band_names = []
    glcm_bands_names = glcm_image.bandNames().remove("glcm_maxcorr").getInfo()
    for i in glcm_bands_names:
        band_name_glcm = bandname+'_'+i
        new_glcm_band_names.append(band_name_glcm)
    glcm_image_N = glcm_image.select(glcm_bands_names).rename(new_glcm_band_names)
    return glcm_image_N
#Cria Função que adiociona um campo (sede) no vetor poligonos de áreas iluminadas (VIIRS) 
# e atribui 1 para os poligonos que intersectam os pontos das sedes municipais e 0 que não intersectam.
def cruzaSede(ft):
    return ft.set('sede',ft.intersects(sedes))
#Cria Função para constuir mosaico de imagens Sentinel-2 livre de núvens (RICHTER, 2017)
def maskClouds(img):
    clouds = ee.Image(img.get('cloud_mask')).select('probability')
    isNotCloud = clouds.lt(MAX_CLOUD_PROBABILITY)
    return img.updateMask(isNotCloud)
#Cria Função para corrigir defeitos de bordas das cenas (RICHTER, 2017)
def maskEdges(s2_img):
    return s2_img.updateMask(s2_img.select('B8A').mask().updateMask(s2_img.select('B9').mask()))
#Mostra uma mensagem informando que as funções foram criadas
print('Funções ok!')
#Calcula a média das composições mensais do VIIRS dentre o período definido
nighttime=col_viirs.mean().clip(geometry).select('avg_rad')
#Cria uma imagem bináriacom valor de 1 para os pixels com valor maior que 1 e 0 para os pixels com valore menor que 1
zones = nighttime.gt(0.8)
#Vetoriza a imagem binária do viirs
vectors = zones.addBands(nighttime).reduceToVectors(geometry = geometry,
                                                     crs="EPSG:31981",
                                                     scale = 500,
                                                     geometryType = 'polygon',
                                                     eightConnected = True,
                                                     labelProperty = 'zone',
                                                     reducer = ee.Reducer.mean(),
                                                    maxPixels=15718696000000000 )
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
print('AOI pronto!')
#Filtra as imagens da coleção Sentinel-2 que intersectam a AOI1 e atendem ao intevalo de datas definido 
criteria = s2Sr_col.filterBounds(AOI1).filterDate(START_DATE,END_DATE)
#Aplica a função maskEdges na imagem
s2Sr = criteria.map(maskEdges)
#Filtra as bandas de probilidade de núvens para a area de intersse e datas definidas
s2Clouds = s2Clouds.filterBounds(AOI1).filterDate(START_DATE,END_DATE)
#Junta as imagens Sentinel-2 com a imagem de probabilidade de núvens
s2SrWithCloudMask = ee.Join.saveFirst('cloud_mask').apply(primary= criteria,
                                                           secondary= s2Clouds,
                                                           condition=ee.Filter.equals(leftField = 'system:index', rightField = 'system:index'))
#Define a proprabilidade máxima de presença de núvens
MAX_CLOUD_PROBABILITY = 30
#Seleciona todos os pixels de todas imagens que possuem até 30% de probabilidade de núvens
#E cria um mosaico a partir do valor médio de cada pixel em cada imagem
s2CloudMasked = ee.ImageCollection(s2SrWithCloudMask).map(maskClouds).median()
#Seleciona as bandas espectrais de 10m a serem utilizadas 
s2CloudMasked = s2CloudMasked.clip(AOI1).select('B4','B3','B2','B8')
#Calcula o NDVI
ndvi = ee.Image(s2CloudMasked.normalizedDifference(['B8', 'B4'])).rename('NDVI')
#Calcula o NDWI
ndwi = ee.Image(s2CloudMasked.normalizedDifference(['B3', 'B8'])).rename('NDWI')

#Cria uma banda espectral para cada banda utilizada
B_RED = s2CloudMasked.select('B4').rename('R')
B_GREEN = s2CloudMasked.select('B3').rename('G')
B_BLUE = s2CloudMasked.select('B2').rename('B')
B_NIR = s2CloudMasked.select('B8').rename('N')

#Número de níveis de cinza
dig_levels = 16
#Tamanho da janela kernel
kernel = 1
#Número de árvores de decisão 
trees = 1000
#Mostra o número de níveis de cinza escolhido
print("Número de níveis de cinza: ",dig_levels)
#Mostra o tamanho da janela kernel
if kernel==1:
    w='3X3'
    print("Tmanho da janela: ",w)
else:
    w='5X5'
    print("Tmanho da janela: ",w)
#Mostra o número de árvores de decisão
print("Número de árvores: ",trees)
#cria uma string com todos os hiperparametros escolhidos para identificar os arquivos de saída
s = 'SRF_T'+str(trees)+'_W'+str(w)+'_DL'+str(dig_levels)

#Utiliza a função compute_probabilistic_glcm para requantizar as bandas e gera a matrix GLCM e calcula as métricas
RED_GLCM = compute_probabilistic_glcm(B_RED.clip(AOI1.bounds()),dig_levels,10,kernel,AOI1)
RED_GLCM = spectral_glcm_band_name(RED_GLCM,B_RED)
GREEN_GLCM = compute_probabilistic_glcm(B_GREEN.clip(AOI1.bounds()),dig_levels,10,kernel,AOI1)
GREEN_GLCM = spectral_glcm_band_name(GREEN_GLCM,B_GREEN)
BLUE_GLCM = compute_probabilistic_glcm(B_BLUE.clip(AOI1.bounds()),dig_levels,10,kernel,AOI1)
BLUE_GLCM = spectral_glcm_band_name(BLUE_GLCM,B_BLUE)
NIR_GLCM = compute_probabilistic_glcm(B_NIR.clip(AOI1.bounds()),dig_levels,10,kernel,AOI1)
NIR_GLCM = spectral_glcm_band_name(NIR_GLCM,B_NIR)

#Cria uma imagem com as 4 bandas espectrais, 18 métricas de textura para cada banda espectral, NDVI e NDWI
image_class = RED_GLCM.addBands(GREEN_GLCM).addBands(BLUE_GLCM).addBands(NIR_GLCM).addBands(B_RED).addBands(B_GREEN).addBands(B_BLUE).addBands(B_NIR).addBands(ndvi).addBands(ndwi)
#carregar as amostras
sample = ee.FeatureCollection("users/gabrielcrivellarog/amostra_2020_ALL")
#Cria uma coluna com valores aleatórios entre 0 e 1 para cada polígono
sample = sample.randomColumn(seed=1)
#Definir o percentual a ser utilziado para teste e treinamento
split = 0.6
#Cria uma camada vetorial com as amostras de treinamento
training = sample.filter(ee.Filter.lt('random', split))
#Cria uma camada vetorial com as amostras de teste
validation = sample

#Extrai todas as informações de todas bandas de todos os pixels que intersectam os poligonos
training_values = image_class.sampleRegions(collection= training,properties= ['C_ID'],scale= 10,geometries=True)
#Remove as amostras que podem ter ficado em área sem informação
trainingNoNulls = training_values.filter(ee.Filter.notNull(training_values.first().propertyNames()))
#Exporta para o GoogleDrive uma planillha com os valores amostrados para cada pixel
N_TD= s+'_treino' 
task=ee.batch.Export.table.toDrive(collection = trainingNoNulls,description = N_TD, folder = 'N_Single_classifier')
task.start()

#Cria um classificador random forest e treina com as amostras coletadas.
classifier = ee.Classifier.smileRandomForest(numberOfTrees=trees,bagFraction=1,seed=1).train(features= trainingNoNulls,classProperty='C_ID')
#Classifica a imagem com o Rabdom Forest
classified = image_class.classify(classifier)
#Conta o numero de amostras utilizadas no treinamento
#labels = trainingNoNulls.aggregate_histogram('C_ID').getInfo()
#print ('Labels:',labels)

for i in sede_id:
    
    AOI_sede = vectors_sedes.filterMetadata('system:index','equals',i).geometry()
    get_name = vectors_joined.filterMetadata('system:index','equals',i)
    Sede_nome = ee.Feature(get_name.first().get('SEDE')).get('NAME').getInfo()
    Sede_nome_sigla = ee.Feature(get_name.first().get('SEDE')).get('SIGLA').getInfo()
    cidade_name = Sede_nome_sigla
    print (Sede_nome, ' Iniciado!')

    #Concateda a sigla do municipio com a identificação da imagem de saída
    name = cidade_name + '_' + s +'_CLASS_20'
    #Exporta para o GoogleDrive o mapa temático final
    task_im = ee.batch.Export.image.toDrive(image = classified.clip(AOI_sede), description=name, folder='N_Single_classifier', region=AOI_sede, scale=10, crs = 'EPSG:31981')
    task_im.start()

   
    print(Sede_nome, ': Mapa temático exportado!')

#Cria um dicionário com os resultados do RF
dict_ = classifier.explain()

#Exporta os resultados do classificador em uma tabela no GDrive
explain = ee.FeatureCollection(ee.Feature(None, ee.Dictionary(dict_)))
N_explain = s+'_explain' 
task_explain=ee.batch.Export.table.toDrive(collection = explain, description = N_explain, folder = 'N_Single_classifier') 
task_explain.start() 

#Exporta os a importancia das variáveis do classificador em uma tabela no GDrive
variable_importance = ee.FeatureCollection(ee.Feature(None, ee.Dictionary(dict_).get('importance')))
N_ID = s+'_importance' 
task_i=ee.batch.Export.table.toDrive(collection = variable_importance, description = N_ID, folder = 'N_Single_classifier') 
task_i.start()

#Exporta a matrix de confusão em uma tabela no GDrive
ta = classifier.confusionMatrix()
train_accuracy = ee.Feature(None,{'matrix':ta.array()})
N_AT = s+'_Acuracia_treino' 
task_it=ee.batch.Export.table.toDrive(collection = ee.FeatureCollection(train_accuracy), description = N_AT, folder = 'N_Single_classifier') 
task_it.start()

#Exporta a validação do classificador em uma tabela no GDrive
validated_values = classified.sampleRegions(collection= validation,properties= ['C_ID'],scale= 10,geometries=True)
validated_valuesNoNulls = validated_values.filter(ee.Filter.notNull(validated_values.first().propertyNames()))
N_TDP= s + '_teste_p' 
taskTP=ee.batch.Export.table.toDrive(collection = validated_valuesNoNulls,description = N_TDP, folder = 'N_Single_classifier',fileFormat="SHP")
taskTP.start()  
test_accuracy = validated_valuesNoNulls.errorMatrix('C_ID', 'classification',[1,2,3,4,5])
test_accuracy_fe = ee.Feature(None,{'matrix':(test_accuracy).array()})
N_TA = s + '_Acuracia_teste' 
task_ta=ee.batch.Export.table.toDrive(collection = ee.FeatureCollection(test_accuracy_fe), description = N_TA, folder = 'N_Single_classifier') 
task_ta.start()
#Mostra uma mensagem informando que o prcessamento de uma das sedes foi concluído
print('Pronto!')
print('Matrix de erro da validação: ', test_accuracy.getInfo())
print('Ordem da matrix de erro da validação: ', test_accuracy.order().getInfo())
print('Acurácia global da validação: ', test_accuracy.accuracy().getInfo())
print('Matrix do consumidor da validação: ', test_accuracy.consumersAccuracy().getInfo())
print('Acurácia do produtor da validação: ', test_accuracy.producersAccuracy().getInfo())
print('kappa da validação: ', test_accuracy.kappa().getInfo())
#Mostra que a classificação foi conclúida e mostra o valor das do pixel de cada classe
print('Processamento concluído! - Solo Exposto: 1; Área Construída: 2; Água: 3; Herbácea: 4; Arbórea: 5.')
print('Fim!')
