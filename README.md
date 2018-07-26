Implementação da GRNN com diferentes métodos.

* Sequencial em CPU.
* Paralela em CPU com Pthreads.
* Paralela em GPU com Cuda.

Para compilar todos os comandos, executar `make all`. Para compilar os comandos individualmente:

* Gerador do conjuntos para a equação do calor: `make geradorDifusao`
* Estimador sequencial usando CPU: `make grnn_cpu`
* Estimador paralelizado com pthreads: `make grnn_pthreads`
* Estimador paralelizado pela GPU: `make grnn_gpu`

Pode ser necessário alterar algumas definições no arquivo `Makefile` antes de compilar a versão GPU com Cuda:

Localização das bibliotecas Cuda (linha 37): `CUDA_PATH ?= "/usr/local/cuda"`

## Utilização

Sem nenhum argumento, os comandos `grnn_{cpu,pthreads,gpu}` apenas estimam o erro das estimativas para o conjunto de  teste. Os conjuntos de treinamento e teste são gerados pelo comando `geradorDifusao` ou `geradorMandelbrot`. São criados os arquivos `train.bin` e `test.bin`, que serão utilizados pelo comando `grnn_{cpu,pthreads,gpu}` para computar as estimativas.

Um arquivo para armazenar as estimativas pode ser informado com a opção `-o result.bin` para que as estimativas sejam armazenadas no arquivo `result.bin`.

Um escalar para o parâmetro sigma da regressão pode ser informado com a opção `-s ESCALAR`, onde `ESCALAR` é um valor real positivo. O valor padrão é 1.

## Conjunto Mandelbrot

Além dos resultados para a equação do calor, podem ser estimados os pontos do conjunto fractal Mandelbrot. Os comandos `geradorMandelbrot` e `verMandelbrot` são compilados separadamente com o comando `make -f Makefile.mandelbrot`. São necessárias as bibliotecas de desenvolvimento **SDL2** para compilar esses comandos.

O comando `geradorMalndelbrot` abre um tela com a representação gráfica do conjunto Mandelbrot. A tecla **z** aproxima, a tecla **x** afasta e as teclas de direção transladam o semiplano. A tecla **s** armazena em *train.bin* o conjunto de treinamento para o semiplano exibido e a tecla **t** armazena em *test.bin* o conjunto de teste para o semiplano exibido. Depois de gerar um arquivo de resultados com o comando `grnn_{cpu,omp,pthreads,gpu} result.bin`, a imagem com as estimativas para o conjunto de teste é exibida com o comando `verMaldelbrot result.bin`.
