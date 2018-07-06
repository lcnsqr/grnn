#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "pathio.h"

// Arquivo das amostras
#define TRAIN "train.bin"
// Arquivos de teste
#define TEST "test.bin"

// Implementação da distância entre dois vetores
float dist(float *v, float *w, int n){
	// Quadrado da distância euclidiana entre o vetores v e w
	float d = 0;
	for (int i = 0; i < n; i++){
		d += pow(w[i]-v[i], 2);
	}
	return d;
}

// Estimar a variável dependente
// train: Conjunto de treinamento
// total: total de pares no treinamento
// dim: Dimensões da variável independente e dependente
// x: Variável independente lida
// y: Estimativa da variável dependente
// s: Parâmetro da regressão
void estim(float *train, unsigned int total, unsigned int *dim, float *x, float *y, float s){
	// Tamanho de um caminho (soma das dimensões dos vértices)
	unsigned int dims = dim[0]+dim[1];
	// Acumuladores do numerador e denominador do estimador
	// para cada dimensão da variável dependente
	float *numer = (float*)malloc(dim[1]*sizeof(float));
	float *denom = (float*)malloc(dim[1]*sizeof(float));
	for (int c = 0; c < dim[1]; c++){
		numer[c] = 0;
		denom[c] = 0;
	}
	// Iterar para cada componente de y
	#pragma omp parallel for
	for (int c = 0; c < dim[1]; c++){
		// Fator comum para cada amostra
		float f;
		// Iterar em cada amostra de treinamento
		for (int i = 0; i < total; i++){
		// Computar o fator comum da i-esima amostra
		f = exp( -dist(x, &train[i*dims], dim[0]) / s );
			// Numerador da fração para o c-ésimo componente
			numer[c] += train[i*dims + dim[0] + c] * f;
			// Denominador da fração
			denom[c] += f;
		}
	}
	// Estimativa com verificação de divisão por zero
	// para cada dimensão da variável dependente
	for (int c = 0; c < dim[1]; c++){
		if ( denom[c] != 0 ){
			y[c] = numer[c] / denom[c];
		}
		else {
			// Falha na operação
			puts("Divisão por zero\n");
			exit(EXIT_FAILURE);
		}
	}
	free(numer);
	free(denom);
}

void testar(struct pathSet *train, struct pathSet *test, float sigma, float *errsum){
	// Vetor da estimativa 
	float *y = (float*)malloc(sizeof(float)*train->dim[1]);
	// Erro da estimativa
	float err = 0;
	// Tamanho de um caminho (soma das dimensões dos vértices)
	unsigned int dims = train->dim[0]+train->dim[1];
	// Expressão envolvendo sigma no numerador do fator comum é constante
	float s = 2*pow(sigma,2);
	// Iterar em todo o conjunto de teste
	for (int i = 0; i < test->total; i++){
		estim(train->data.f, train->total, train->dim, &test->data.f[i*dims], y, s);
		/*
		// Exibir valores da condição inicial
		printf("Condição inicial\n");
		for (int c = 0; c < test->dim[0]; c++){
			printf("%.6f\n", test->data.f[i*dims + c]);
		}
		// Exibir estimativa e valor observado
		printf("Valor Observado:\n");
		for (int c = 0; c < test->dim[1]; c++){
			printf("%.6f\n", test->data.f[i*dims + test->dim[0] + c]);
		}
		printf("Estimativa: \n");
		for (int c = 0; c < test->dim[1]; c++){
			printf("%.6f\n", y[c]);
		}
		*/
		// Erro da estimativa
		err = sqrt(dist(&test->data.f[i*dims], y, test->dim[1]));
		/*
		printf("Diferença: %f\n", err);
		printf("\n");
		*/
		// Erro acumulado (sem raiz)
		*errsum += err;
		// Mostrar progresso
		putchar('.');
		fflush(stdout);
		if ( (i+1) % 10 == 0 ){
			// Espaço a cada 10 pontos
			putchar(' ');
			fflush(stdout);
		}
	}
	free(y);
}

int main(int argc, char **argv){
	struct pathSet train, test;
	// Carregar arquivo das amostras de treinamento
	pathSetLoad(TRAIN, &train);

	// Arquivo de teste
	pathSetLoad(TEST, &test);

	printf("Conjunto de treinamento: %d amostras.\n", train.total);
	printf("Dimensões da variável independente: %d\n", train.dim[0]);
	printf("Dimensões da variável dependente:   %d\n", train.dim[1]);

	// Parâmetro sigma da regressão
	float sigma;
	// Erro acumulado
	float errsum;

	// Testar
	printf("Calculando estimativas para o conjunto teste (%d amostras).\n\n", test.total);
	puts("O ponto (.) representa a estimativa para cada amostra do conjunto de teste\n");

	// Parâmetro sigma (variância)
	sigma = 1.0/log(train.total);

	// Medir tempo e execução
	clock_t begin, end;

	// Sequencial
	begin = clock();
	errsum = 0;
	testar(&train, &test, sigma, &errsum);
	end = clock();
	//printf("\nTempo: %f segundos\n", (double)(end - begin) / CLOCKS_PER_SEC);
	//printf("Erro médio: %f\n\n", errsum / (float)test.total);

	return 0;
}
