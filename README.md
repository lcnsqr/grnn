Implementação da GRNN com diferentes métodos.

* Sequencial em CPU.
* Paralela em CPU com OpenMP.
* Paralela em CPU com Pthreads.
* Paralelal em GPU com Cuda.

Os conjuntos de treinamento e teste para a equação do calor são gerados pelo comando `geradorDifusao`.

Para compilar todos os comandos, executar `make all`. Para compilar os comandos individualmente:

* `make geradorDifusao`
* `make grnn_cpu`
* `make grnn_omp`
* `make grnn_pthreads`
* `make grnn_gpu`

Nenhum dos comandos `grnn_{cpu,omp,pthreads,gpu}` recebe opção, todos utilizam os mesmos conjuntos de treinamento e teste gerados pelo comando `geradorDifusao`.

Pode ser necessário alterar algumas definições no arquivo `Makefile` antes de compilar a versão GPU com Cuda:

Localização das bibliotecas Cuda (linha 37): `CUDA_PATH ?= "/usr/local/cuda"`

Includes necessários (linha 234): `INCLUDES  := -I/usr/local/cuda/samples/common/inc`

