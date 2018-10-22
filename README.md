Implementação da GRNN com diferentes métodos.

* Sequencial em CPU;
* Paralela em CPU com Pthreads;
* Paralela em GPU com Cuda.

Para compilar todos os comandos, executar `make all`. Para compilar os comandos individualmente:

* Gerador dos conjuntos para a equação do calor: `make geradorDifusao`
* Gerador dos conjuntos para o conjunto Mandelbrot: `make geradorDifusao`
* Estimador paralelizado em CPU com pthreads: `make grnn_pthreads`
* Estimador paralelizado pela GPU: `make grnn_gpu`

Pode ser necessário alterar algumas definições no arquivo `Makefile` antes de compilar a versão GPU com Cuda:

Localização das bibliotecas Cuda (linha 37): `CUDA_PATH ?= "/usr/local/cuda"`

## Utilização

Sem nenhum argumento, os comandos `grnn_pthreads` e `grnn_gpu` apenas estimam o erro das estimativas para o conjunto de  teste. Os conjuntos de treinamento e teste são gerados pelo comando `geradorDifusao` ou `geradorMandelbrot`. São criados os arquivos `train.bin` e `test.bin`, que serão utilizados pelos comandos `grnn_pthreads` e `grnn_gpu` para computar as estimativas.

### Conjunto Mandelbrot

Estimativa de pertecimento de um ponto ao conjunto fractal Mandelbrot. São necessárias as bibliotecas de desenvolvimento **SDL2** para compilar esses comandos.

O comando `geradorMandelbrot` abre um tela com a representação gráfica do conjunto Mandelbrot. O parâmetro `-w 700` define a largura em `700` pixels e o parâmetro `-h 700` define a altura em `700` pixels. A quantidade de pontos no connjunto de treinamento corresponde à quantidade de pixels. A tecla **z** aproxima, a tecla **x** afasta e as teclas de direção transladam o semiplano. A tecla **s** armazena em *train.bin* o conjunto de treinamento (o semiplano exibido) e em *test.bin* o conjunto de teste (8192 pontos aleatórios dentro do semiplano exibido). Com a opção `-s` os conjuntos são gerados sem interatividade.

### Equação do Calor

O comando `geradorDifusao` tem três opções: `-t`, `-e` e `-d`. A opção `-t TOTAL` define em `TOTAL` a quantidade de amostras de treinamento e a opção `-t TOTALTEST` define em `TOTALTEST` a quantidade de amostras de teste, onde `TOTAL` e `TOTALTEST` são números inteiros maiores que zero.

A opção `-d` define a quantidade de pontos igualmente espaçados no intervalo da solução, o que também determina o número de dimensões das variáveis independentes e dependentes da regressão.

### Gerador das estimativas 

Os comandos geradores das estimativas `grnn_pthreads` e `grnn_gpu` têm duas opções: `-o` e `-s`.

Um arquivo para armazenar as estimativas pode ser informado com a opção `-o result.bin` para que as estimativas sejam armazenadas no arquivo `result.bin`.

Um escalar para o parâmetro sigma da regressão pode ser informado com a opção `-s ESCALAR`, onde `ESCALAR` é um valor real positivo. O valor padrão é 1.
