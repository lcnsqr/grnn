#!/usr/bin/env bash

# Script para executar o teste de performance da GRNN 
# paralela em GPU e calcular a média do tempo de execução

# Compilar comandos se necessário
make geradorMandelbrot geradorDifusao grnn_gpu

# Quantidade de repetições
REP=10

# Testar script
#BOGUS="-b"
# Produção
BOGUS=""

#
# Desempenho para o conjunto Mandelbrot
#

# Avaliar os diferentes tamanhos do conjunto 
# de treinamento para o conjunto mandelbrot
SETS="300x300 700x700 1000x1000"

# Gerar conjuntos se necessário
if [ ! -d mandelbrot ]; then 
  mkdir mandelbrot
fi
for s in $SETS; do
	# Conjunto de treinamento
  TRAIN="mandelbrot/train_${s}.bin"
  # Conjunto de teste
  TEST="mandelbrot/test_${s}.bin"
	if [ ! -e $TRAIN ]; then 
    echo "Gerando conjunto ${s}..."
    ./geradorMandelbrot -w ${s%x*} -h ${s#*x} -t 0.4375 -r -0.75 -b -0.4375  -l -1.625 -s
    mv train.bin $TRAIN
    mv test.bin $TEST
	fi
done

for s in $SETS; do
	# Link para o conjunto de treinamento
	if [ -e train.bin ]; then 
		rm train.bin
	fi
	ln mandelbrot/train_${s}.bin train.bin
  # Link para o conjunto de teste
  if [ -e test.bin ]; then 
    rm test.bin
  fi
  ln mandelbrot/test_${s}.bin test.bin
	# Executar o teste REP vezes 
  for r in `seq -w 0 $(($REP-1))`; do
    echo "Teste $((${r#0}+1)) de $REP: ${s#*_} e 8192"
    ./grnn_gpu $BOGUS -s .01 > "${r}_${s#*_}.dat"
  done
  # Agrupar resultados para cada conjunto de treinamento
  sed -e '2d' < "`seq -w 0 $(($REP-1)) | head -n 1`_${s#*_}.dat" > "mandelbrot/GPU_${s#*_}.dat"
  for r in `seq -w 0 $(($REP-1))`; do
    sed -e '1d' < "${r}_${s#*_}.dat" >> "mandelbrot/GPU_${s#*_}.dat"
    rm "${r}_${s#*_}.dat"
  done
done
rm train.bin test.bin

# Criar arquivo contendo os resultados
tar czf desempenho_GPU_mandelbrot.tar.gz mandelbrot/*dat
echo "Resultado dos testes armazenado em desempenho_mandelbrot_GPU.tar.gz"

#
# Desempenho para a equação do calor
#

## Avaliar os diferentes tamanhos do conjunto de treinamento
#SETS="4_131072 4_262144 4_524288 4_1048576"
## Quantidade de repetições
#REP=10
#for s in $SETS; do
#  # Apagar link para o conjunto de treinamento
#  if [ -e train.bin ]; then 
#  	rm train.bin
#  fi
#  ln train_${s}.bin train.bin
#  # Executar o teste REP vezes para cada conjunto de treinamento
#  for r in `seq -w 0 $(($REP-1))`; do
#    echo "Teste $((${r#0}+1)) de $REP: ${s#*_} e 8192"
#    ./grnn_gpu -s .2 > "${r}_${s#*_}.dat"
#  done
#  # Agrupar resultados para cada connjunto de treinamento
#  sed -e '2d' < "`seq -w 0 $(($REP-1)) | head -n 1`_${s#*_}.dat" > "difusao/GPU_${s#*_}.dat"
#  for r in `seq -w 0 $(($REP-1))`; do
#    sed -e '1d' < "${r}_${s#*_}.dat" >> "difusao/GPU_${s#*_}.dat"
#    rm "${r}_${s#*_}.dat"
#  done
#done
