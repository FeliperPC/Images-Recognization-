# importação das bibliotecas necessárias
# pybrain
from pybrain.datasets.supervised import SupervisedDataSet 
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from PIL import Image

# gráficos 
import matplotlib.pyplot as plt
import numpy as np

# função para carregar os dados de treinameto a partir das imagens
def getDataImage(path):
    #Read image
    img = Image.open(path)
    # create the pixel map
    pixels = img.load() 
    data = []
    pixel = []
    for i in range( img.size[0]):         # for every col do: img.size[0]
        for j in range( img.size[1] ):    # for every row   img.size[1]      
             pixel = pixels[i,j]          # get every pixel
             data.append( pixel[0] )
             data.append( pixel[1] )
             data.append( pixel[2] )

    #Viewing EXIF data embedded in image
    exif_data = img._getexif()
    exif_data
    return data

# Definindo padrão de qnt de pixels por treinamento
size = 40 * 40 * 3

# configurando a rede neural artificial e o dataSet de treinamento
network = buildNetwork( size, 61, 10, 4 )  # define network
dataSet = SupervisedDataSet( size, 4 )      # define dataSet

# load dataSet
dataSet.addSample ( getDataImage( 'correntePY\\correntesLubrificadas\\1l.png' ), (1, 1, 1, 1) )       # Lubrificada
dataSet.addSample ( getDataImage( 'correntePY\\correntesLubrificadas\\2.png' ), (1, 1, 1, 1) )       # Lubrificada
dataSet.addSample ( getDataImage( 'correntePY\\correntesLubrificadas\\3.png' ), (1, 1, 1, 1) )       # Lubrificada
dataSet.addSample ( getDataImage( 'correntePY\\correntesLubrificadas\\4.png' ), (1, 1, 1, 1) )       # Lubrificada
dataSet.addSample ( getDataImage( 'correntePY\\correntesLubrificadas\\5.png' ), (1, 1, 1, 1) )       # Lubrificada
dataSet.addSample ( getDataImage( 'correntePY\\correntesLubrificadas\\6.png' ), (1, 1, 1, 1) )       # Lubrificada
dataSet.addSample ( getDataImage( 'correntePY\\correntesLubrificadas\\7.png' ), (1, 1, 1, 1) )       # Lubrificada
dataSet.addSample ( getDataImage( 'correntePY\\correntesLubrificadas\\8.png' ), (1, 1, 1, 1) )       # Lubrificada
dataSet.addSample ( getDataImage( 'correntePY\\correntesLubrificadas\\9.png' ), (1, 1, 1, 1) )       # Lubrificada
dataSet.addSample ( getDataImage( 'correntePY\\correntesLubrificadas\\10.png' ), (1, 1, 1, 1) )       # Lubrificada
dataSet.addSample ( getDataImage( 'correntePY\\correntesLubrificadas\\11.png' ), (1, 1, 1, 1) )       # Lubrificada
dataSet.addSample ( getDataImage( 'correntePY\\correntesLubrificadas\\12.png' ), (1, 1, 1, 1) )       # Lubrificada

dataSet.addSample ( getDataImage( 'correntePY\\correntesNaoLubrificadas\\13.png' ), (0, 0, 0, 0) )       #N Lubrificada
dataSet.addSample ( getDataImage( 'correntePY\\correntesNaoLubrificadas\\14.png' ), (0, 0, 0, 0) )       #N Lubrificada
dataSet.addSample ( getDataImage( 'correntePY\\correntesNaoLubrificadas\\15.png' ), (0, 0, 0, 0) )       #N Lubrificada
dataSet.addSample ( getDataImage( 'correntePY\\correntesNaoLubrificadas\\16.png' ), (0, 0, 0, 0) )       #N Lubrificada
dataSet.addSample ( getDataImage( 'correntePY\\correntesNaoLubrificadas\\17.png' ), (0, 0, 0, 0) )       #N Lubrificada
dataSet.addSample ( getDataImage( 'correntePY\\correntesNaoLubrificadas\\18.png' ), (0, 0, 0, 0) )       #N Lubrificada
dataSet.addSample ( getDataImage( 'correntePY\\correntesNaoLubrificadas\\19.png' ), (0, 0, 0, 0) )       #N Lubrificada
dataSet.addSample ( getDataImage( 'correntePY\\correntesNaoLubrificadas\\20.png' ), (0, 0, 0, 0) )       #N Lubrificada
dataSet.addSample ( getDataImage( 'correntePY\\correntesNaoLubrificadas\\21.png' ), (0, 0, 0, 0) )       #N Lubrificada
dataSet.addSample ( getDataImage( 'correntePY\\correntesNaoLubrificadas\\22.png' ), (0, 0, 0, 0) )       #N Lubrificada

# trainer
trainer = BackpropTrainer( network, dataSet)
error = 1
iteration = 0
outputs = []
arquivo= open('correntePY\\output.txt','a')
while error > 0.001: 
    error = trainer.train()
    outputs.append( error )
    iteration += 1    
    print ( iteration, error )
    arquivo.write(str(iteration)+"-"+str(error)+"\n");

# plot graph
plt.ioff()
plt.plot( outputs )
plt.xlabel('Iterações')
plt.ylabel('Erro Quadrático')
plt.show()

arquivo.write("\n\nFASE DE TESTE: \n\n");

# Fase de teste
name = ['teste1L.png', 'teste2L.png', 'teste3NL.png','teste4NL.png','teste5NL.png']
for i in range( len(name) ):
    path = "correntePY\\"+ name[i]
    print ( path )
    print ( network.activate( getDataImage( path ) ) )
    arquivo.write(str(i)+" - "+str(network.activate( getDataImage( path )))+"\n");



