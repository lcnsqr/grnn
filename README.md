Compilar gerador dos conjuntos de treinamento e teste;

`gcc -o geradorDifusao -lm -O3 geradorDifusao.c`

Compilar GRNN sequencial:

`gcc -o grnn_cpu -lm -O3 grnn_cpu.c`

Compilar GRNN paralela:

`make`


Nenhum dos comandos recebe opção.
