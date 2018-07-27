#!/usr/bin/env bash

# Script para preparar e executar o teste 
# de performance da GRNN paralela em GPU

# Compilação
make clean &>/dev/null
make geradorDifusao grnn_gpu SMS=35

# Conjuntos de treinamento e teste
if  [ ! -f train.bin -a ! -f test.bin ]; then
	./geradorDifusao -t 8388608 -e 1024 -d 6
fi

# Localizar nvprof
PATH=$PATH:/usr/local/cuda/bin
NVPROF=`which nvprof`
if [ ! -x "$NVPROF" ]; then 
	echo "Comando nvprof não localizado"
	exit 1
fi

# Rodar estimador no conjunto de teste
echo "Avaliando a performance do estimador..."
$NVPROF --unified-memory-profiling off --print-gpu-summary --csv ./grnn_gpu -s 0.5 1>info.txt 2>gprof.txt

# Descartar conjuntos de treinamento e teste
#rm train.bin test.bin

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
