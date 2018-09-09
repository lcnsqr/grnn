#!/usr/bin/env python3

import sys
import os
import struct
import numpy as np

if len(sys.argv) < 2:
    exit("Informe o nome do arquivo como argumento")

filename = sys.argv[1]
basename = os.path.splitext(filename)[0]

# Estrutura dos dados importados
dataDict = {};
# Conjunto das variáveis independentes
varIndSet = []
# Conjunto das variáveis dependentes
varDepSet = []
# Ler e converter os dados
with open(filename, "rb") as dataBin:
    # Tipo dos dados
    dataDict['type'] = hex(int.from_bytes(dataBin.read(4), byteorder='little'))
    # Total de dataDict no arquivo
    dataDict['total'] = int.from_bytes(dataBin.read(4), byteorder='little')
    # Número de vertices numa amostra
    dataDict['vertices'] = int.from_bytes(dataBin.read(4), byteorder='little')
    # Número de componentes em cada vertice
    dataDict['dim'] = []
    for d in range(dataDict['vertices']):
        dataDict['dim'].append(int.from_bytes(dataBin.read(4), byteorder='little'))
    # Tamanho do conjunto em bytes
    dataDict['size'] = int.from_bytes(dataBin.read(4), byteorder='little')
    # Par amostral
    varInd = []
    varDep = []
    for i in range(dataDict['total']):
        varInd.clear()
        for c in range(dataDict['dim'][0]):
            dataBin.seek(24 + i * 4 + c * dataDict['total'] * 4)
            varInd.append(struct.unpack('f', dataBin.read(4))[0])
        varDep.clear()
        for c in range(dataDict['dim'][1]):
            dataBin.seek(24 + dataDict['dim'][0] * dataDict['total'] * 4 + i * 4 + c * dataDict['total'] * 4)
            varDep.append(struct.unpack('f', dataBin.read(4))[0])
        varIndSet.append(varInd)
        varDepSet.append(varDep)
    # Salvar conjunto amostral no formato npz
    np.savez(basename, features=varIndSet, labels=varDepSet)
