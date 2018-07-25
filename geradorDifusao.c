#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "matrix.h"
#include "difusao.h"
#include "pathio.h"

// Constante da equação do calor
#define ALFA 1

// Gerar as soluções por diferenças finitas, após T segundos
#define T 1e-1

// Valores padrão
// Extremidade inicial e final
#define POS0 0
#define POS1 1

// Valores padrão
// Menor e maior valores de leitura
#define MINVAL 0
#define MAXVAL 1

// Total de amostras
#define TOTAL 20480000

// Total de amostras de teste
#define TOTALTEST 100

// Dimensões da amostra
#define DIM 6

// Arquivo das amostras
#define TRAIN "train.bin"
// Arquivo das amostras de teste
#define TEST "test.bin"

// Variável aleatória uniforme em [0,1)
#define RAND ((float)(rand() >> 1)/((RAND_MAX >> 1) + 1))

// Função de preenchimento da condição inicial.
void inicial(Matrix *w){
	// Gerar uma curva aleatória
	w->_[0] = RAND * MAXVAL;
	for (int i = 1; i < DIM; i++){
		w->_[i] = w->_[i-1] - .25 + .5 * RAND;
		if ( w->_[i] < MINVAL ) w->_[i] = MINVAL;
		if ( w->_[i] > MAXVAL ) w->_[i] = MAXVAL;
	}
	// Valor da derivada em relação a x no extremo 
	// b definido pelo último valor do vetor (coluna)
	w->_[DIM] = 0;
}

int main (int argc, char **argv){
	srand((unsigned int)time(NULL));

	// Conjunto de treinamento
	struct pathSet train;
	// Usar float (4 bytes)
	train.type = 0x0f;
	train.total = TOTAL;
	train.vertices = 2;
	train.dim = (unsigned int*)malloc(2*sizeof(unsigned int));
	train.dim[0] = DIM;
	train.dim[1] = DIM;
	// Tamanho de um caminho (soma das dimensões dos vértices)
	unsigned int sdim = train.dim[0]+train.dim[1];
	train.size = train.total*sdim*sizeof(float);
	train.data.f = (float*)malloc(train.size);
	
	// Vetores antes e depois (condição inicial e vetor resultante)
	Matrix w[2];
	// Alocar e incluir dimensão para a condição de Neuman
	mtrxBuild(&w[0], DIM + 1, 1);
	mtrxBuild(&w[1], DIM + 1, 1);
	// Gerar cada amostra e copiar para o conjunto de treinamento
	printf("Gerando %d amostras de treinamento...\n", train.total);
	for (int i = 0; i < train.total; i++){
		// Preencher condições iniciais
		inicial(&w[0]);
		// Gerar as soluções por diferenças finitas, após T segundos
		gerar(w, ALFA, POS0, POS1, T);
		// Copiar v.i. para posição correspondente na memória do conjunto
		//memcpy(&train.data.f[i*sdim], w[0]._, train.dim[0]*sizeof(float));
		for (int j = 0; j < train.dim[0]; j++){
			train.data.f[i + j * train.total] = w[0]._[j];
		}
		// Variável dependente
		//memcpy(&train.data.f[i*sdim+train.dim[0]], w[1]._, train.dim[1]*sizeof(float));
		for (int j = 0; j < train.dim[1]; j++){
			train.data.f[train.total*train.dim[0] + i + j * train.total] = w[1]._[j];
		}
		// Exibir progresso
		if ( (i+1) % 10000 == 0 ) printf("%d\n", i+1);
	}
	// Salvar conjunto de treinamento
	pathSetSave(TRAIN, &train);
	free(train.dim);
	free(train.data.f);

	// Conjunto de teste
	struct pathSet test;
	// Usar float (4 bytes)
	test.type = 0x0f;
	test.total = TOTALTEST;
	test.vertices = 2;
	test.dim = (unsigned int*)malloc(2*sizeof(unsigned int));
	test.dim[0] = DIM;
	test.dim[1] = DIM;
	test.size = test.total*sdim*sizeof(float);
	test.data.f = (float*)malloc(test.size);
	
	// Gerar cada amostra e copiar para o conjunto de teste
	printf("Gerando %d amostras de teste...\n", test.total);
	for (int i = 0; i < test.total; i++){
		// Preencher condições iniciais
		inicial(&w[0]);
		// Gerar as soluções por diferenças finitas, após T segundos
		gerar(w, ALFA, POS0, POS1, T);
		// Copiar v.i. para posição correspondente na memória do conjunto
		//memcpy(&test.data.f[i*sdim], w[0]._, test.dim[0]*sizeof(float));
		for (int j = 0; j < test.dim[0]; j++){
			test.data.f[i + j * test.total] = w[0]._[j];
		}
		// Variável dependente
		//memcpy(&test.data.f[i*sdim+test.dim[0]], w[1]._, test.dim[1]*sizeof(float));
		for (int j = 0; j < test.dim[1]; j++){
			test.data.f[test.total*test.dim[0] + i + j * test.total] = w[1]._[j];
		}
		// Somente o último componente
		//test.data.f[i*sdim+test.dim[0]] = w[1]._[DIM-1];
		// Exibir progresso
		if ( (i+1) % 10000 == 0 ) printf("%d\n", i+1);
	}

	// Salvar conjunto de treinamento
	pathSetSave(TEST, &test);
	free(test.dim);
	free(test.data.f);

	return 0;
}
