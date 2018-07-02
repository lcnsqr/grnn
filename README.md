Compilar gerador dos conjuntos de treinamento e teste;

gcc -o geradorDifusao -lm -O3 geradorDifusao.c

Compilar GRNN sequencial:

`gcc -o grnn\_cpu -lm -O3 grnn\_cpu.c`

Compilar GRNN paralela:

`make`


Nenhum dos comandos recebe opção.
