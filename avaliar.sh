#!/bin/bash

# Script para preparar e executar o teste 
# de performance da GRNN paralela em GPU

# Compilação
make clean &>/dev/null
make geradorDifusao grnn_gpu SMS=35

# Conjuntos de treinamento e teste
./geradorDifusao -t 8388608 -e 1024 -d 6

# Localizar nvprof
NVPROF=`which nvprof`
if [ -n "$NVPROF" ]; then 
	NVPROF="/usr/bin/nvprof"; 
fi

# Rodar estimador no conjunto de teste
echo "Avaliando a performance do estimador..."
$NVPROF --print-gpu-summary --csv ./grnn_gpu -s 0.5 1>info.txt 2>gprof.txt

# Descartar conjuntos de treinamento e teste
rm train.bin test.bin

# Nome do dispositivo utilizado para o nome do arquivo
DEV=`cut -f1 < info.txt | tail -1`

# Formatar saídas
sed -e "s/\t/\",\"/g" -e "s/^/\"/" -e "s/$/\"/" < info.txt > info.csv
tail -5 < gprof.txt > gprof.csv
rm info.txt gprof.txt

# Arquivo final
tar cf "$DEV.tar" info.csv gprof.csv
rm info.csv gprof.csv

# Enviar arquivo
curl -F "arquivo=@$DEV.tar" https://linux.ime.usp.br/~lcnsqr/map2070/upload.php
