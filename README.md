Implementação da GRNN com diferentes métodos.

* Sequencial em CPU.
* Paralela em CPU com OpenMP.
* Paralela em CPU com Pthreads.
* Paralela em GPU com Cuda.

Para compilar todos os comandos, executar `make all`. Para compilar os comandos individualmente:

* Gerador do conjuntos para a equação do calor: `make geradorDifusao`
* Gerador do conjuntos para o conjunto Mandelbrot: `make geradorMandelbrot`
* Visualizador das estimativas para o conjunto Mandelbrot: `make verMandelbrot`
* Estimador sequencial usando CPU: `make grnn_cpu`
* Estimador paralelizado com OpenMP: `make grnn_omp`
* Estimador paralelizado com pthreads: `make grnn_pthreads`
* Estimador paralelizado pela GPU: `make grnn_gpu`

Pode ser necessário alterar algumas definições no arquivo `Makefile` antes de compilar a versão GPU com Cuda:

Localização das bibliotecas Cuda (linha 37): `CUDA_PATH ?= "/usr/local/cuda"`

## Utilização

Sem nenhum argumento, os comandos `grnn_{cpu,omp,pthreads,gpu}` apenas estimam o erro das estimativas para o conjunto de  teste. Os conjuntos de treinamento e teste são gerados pelo comando `geradorDifusao` ou `geradorMandelbrot`. São criados os arquivos `train.bin` e `test.bin`, que serão utilizados pelo comando `grnn_{cpu,omp,pthreads,gpu}` para computar as estimativas.

Um nome de arquivo pode ser informado como argumento ao comando `grnn_{cpu,omp,pthreads,gpu}` para que as estimativas sejam armazenadas neste arquivo.

## Conjunto Mandelbrot

O comando `geradorMalndelbrot` abre um tela com a representação gráfica do conjunto Mandelbrot. A tecla **z** aproxima, a tecla **x** afasta e as teclas de direção transladam. A tecla **s** armazena em *train.bin* o conjunto de treinamento para o semiplano exibido e a tecla **t** armazena em *test.bin* o conjunto de teste para o semiplano exibido. Depois de gerar um arquivo de resultados com o comando `grnn_{cpu,omp,pthreads,gpu} result.bin`, a imagem com as estimativas para o conjunto de teste é exibida com o comando `verMaldelbrot result.bin`.