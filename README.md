Compilar o gerador dos conjuntos de treinamento e teste;

`gcc -o geradorDifusao -lm -O3 geradorDifusao.c`

Compilar a GRNN sequencial:

`gcc -o grnn_cpu -lm -O3 grnn_cpu.c`

Compilar a GRNN paralela:

`make`

Pode ser necessário alterar algumas definições no arquivo `Makefile` antes de compilar a versão paralela:

Localização das bibliotecas Cuda (linha 37): `CUDA_PATH ?= "/usr/local/cuda"`

Includes necessários (linha 234): `INCLUDES  := -I/usr/local/cuda/samples/common/inc`

Nenhum dos comandos `grnn_cpu` e `grnn_gpu` recebe opção, ambos utilizam os mesmos conjuntos de treinamento e teste gerados pelo comando `geradorDifusao`.
