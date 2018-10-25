#!/usr/bin/env bash

# Script para executar o teste de performance da GRNN 
# paralela em CPU e calcular a média do tempo de execução

# Compilar comandos se necessário
make geradorMandelbrot geradorDifusao grnn_pthreads

MODEL=$(grep '^model name' < /proc/cpuinfo | uniq | sed -e 's/.*: //')
CORES=$(grep '^cpu cores' < /proc/cpuinfo | uniq -c | awk '{print $5}')
PROCS=$(grep '^cpu cores' < /proc/cpuinfo | uniq -c | awk '{print $1}')

# Quantidade de repetições
REP=10

# Threads em paralelo
THREADS=`seq $PROCS`

# Testar script
#BOGUS="-b"
# Produção
BOGUS=""

#
# Desempenho para o conjunto Mandelbrot
#

echo -e "CPU\tNúcleos\tProcessadores" > cpuinfo.dat
echo -e "$MODEL\t$CORES\t$PROCS" >> cpuinfo.dat

# Avaliar os diferentes tamanhos do conjunto 
# de treinamento para o conjunto mandelbrot
SETS="300x300 700x700 1000x1000"

# Gerar conjuntos se necessário
if [ ! -d mandelbrot ]; then 
  mkdir mandelbrot
fi
for s in $SETS; do
	# Link para o conjunto de treinamento
  TRAIN="mandelbrot/train_${s}.bin"
	if [ ! -e $TRAIN ]; then 
    echo "Gerando ${TRAIN}..."
    ./geradorMandelbrot -w ${s%x*} -h ${s#*x} -t 0.4375 -r -0.75 -b -0.4375  -l -1.625 -s
    mv train.bin $TRAIN
	fi
  # Conjunto de teste
	if [ -e test.bin ]; then 
    mv test.bin mandelbrot/
  fi
done

# Link para o conjunto de teste
if [ -e test.bin ]; then 
  rm test.bin
fi
ln mandelbrot/test.bin test.bin
for s in $SETS; do
	# Link para o conjunto de treinamento
	if [ -e train.bin ]; then 
		rm train.bin
	fi
	ln mandelbrot/train_${s}.bin train.bin
	# Executar o teste REP vezes para cada número 
	# de threads e conjunto de treinamento
	for t in $THREADS; do
		for r in `seq -w 0 $(($REP-1))`; do
			echo "Teste $((${r#0}+1)) de $REP: ${s#*_} e 8192 em $t threads"
			./grnn_pthreads $BOGUS -s .005 -p $t > result.dat
			paste cpuinfo.dat result.dat > "${r}_${s#*_}_$t.dat"
		done
		rm result.dat
		# Agrupar resultados para cada quantidade de threads
		sed -e '2d' < "`seq -w 0 $(($REP-1)) | head -n 1`_${s#*_}_$t.dat" > "mandelbrot/CPU_${t}_${s#*_}.dat"
		for r in `seq -w 0 $(($REP-1))`; do
			sed -e '1d' < "${r}_${s#*_}_$t.dat" >> "mandelbrot/CPU_${t}_${s#*_}.dat"
			rm "${r}_${s#*_}_$t.dat"
		done
	done
done
rm train.bin test.bin

#
# Desempenho para a equação do calor
#

# Avaliar os diferentes tamanhos do conjunto 
# de treinamento para a equação do calor
SETS="4_131072 4_262144 4_524288 4_1048576"

# Gerar conjuntos se necessário
if [ ! -d difusao ]; then 
  mkdir difusao
fi
for s in $SETS; do
	# Link para o conjunto de treinamento
  TRAIN="difusao/train_${s}.bin"
	if [ ! -e $TRAIN ]; then 
    echo "Gerando ${TRAIN}..."
    ./geradorDifusao -t ${s#*_} -e 8192 -d 4
    mv train.bin $TRAIN
	fi
  # Conjunto de teste
	if [ -e test.bin ]; then 
    mv test.bin difusao/
  fi
done

# Link para o conjunto de teste
if [ -e test.bin ]; then 
  rm test.bin
fi
ln difusao/test.bin test.bin
for s in $SETS; do
	# Link para o conjunto de treinamento
	if [ -e train.bin ]; then 
		rm train.bin
	fi
	ln difusao/train_${s}.bin train.bin
	# Executar o teste REP vezes para cada número 
	# de threads e conjunto de treinamento
	for t in $THREADS; do
		for r in `seq -w 0 $(($REP-1))`; do
			echo "Teste $((${r#0}+1)) de $REP: ${s#*_} e 8192 em $t threads"
			./grnn_pthreads $BOGUS -s .2 -p $t > result.dat
			paste cpuinfo.dat result.dat > "${r}_${s#*_}_$t.dat"
		done
		rm result.dat
		# Agrupar resultados para cada quantidade de threads
		sed -e '2d' < "`seq -w 0 $(($REP-1)) | head -n 1`_${s#*_}_$t.dat" > "difusao/CPU_${t}_${s#*_}.dat"
		for r in `seq -w 0 $(($REP-1))`; do
			sed -e '1d' < "${r}_${s#*_}_$t.dat" >> "difusao/CPU_${t}_${s#*_}.dat"
			rm "${r}_${s#*_}_$t.dat"
		done
	done
done
rm -f train.bin test.bin
rm cpuinfo.dat

# Criar arquivo contendo os resultados
tar czf desempenho_CPU.tar.gz mandelbrot/*dat difusao/*dat
echo "Resultado dos testes armazenado em desempenho_CPU.tar.gz"
