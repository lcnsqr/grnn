#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "pathio.h"

// Arquivo das amostras
#define TRAIN "train.bin"
// Arquivos de teste
#define TEST "test.bin"

// Quantidade de threads
#define NUM_THREADS	8

// Estrutura para o contexto de cada thread que 
// computa uma parcial do conjunto de treinamento
struct TrainPart {
	// Endereço da parte donjunto de treinamento
	float *train;
	// total de pares no treinamento parcial
	unsigned int total;
	// Dimensões da variável independente e dependente
	unsigned int *dim;
	// Variável independente lida
	float *x;
	// Parâmetro da regressão
	float s;
	// Acumuladores do numerador e denominador
	float *numer;
	float *denom;
};

// Implementação da distância entre dois vetores
float dist(float *v, float *w, int n){
	// Quadrado da distância euclidiana entre o vetores v e w
	float d = 0;
	for (int i = 0; i < n; i++){
		d += pow(w[i]-v[i], 2);
	}
	return d;
}

// Thread para computar parcial do conjunto de treinamento
void *estimPart(void *voidTrainPart){
	// Estrutura da parcial do treinamento
	struct TrainPart *tp = (struct TrainPart*)voidTrainPart;
	// Tamanho de um caminho (soma das dimensões dos vértices)
	unsigned int dims = tp->dim[0]+tp->dim[1];
	// Fator comum para cada amostra
	float f;
	// Iterar em cada amostra de treinamento
	for (int i = 0; i < tp->total; i++){
		// Computar o fator comum da i-esima amostra
		f = exp( -dist(tp->x, &tp->train[i*dims], tp->dim[0]) / tp->s );
		// Iterar para cada componente de y
		for (int c = 0; c < tp->dim[1]; c++){
			// Numerador da fração para o c-ésimo componente
			tp->numer[c] += tp->train[i*dims + tp->dim[0] + c] * f;
			// Denominador da fração
			tp->denom[c] += f;
		}
	}
	pthread_exit((void*)voidTrainPart);
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
	// Contexto para cada thread
	struct TrainPart tp[NUM_THREADS];
	// Acumuladores do numerador e denominador 
	// do estimador para cada thread
	float *aNumer[NUM_THREADS];
	float *aDenom[NUM_THREADS];
	// Configurar contexto para cada thread
	for (int t = 0; t < NUM_THREADS; t++){
		// Acumuladores para cada dimensão da variável dependente
		aNumer[t] = (float*)malloc(dim[1]*sizeof(float));
		aDenom[t] = (float*)malloc(dim[1]*sizeof(float));
		for (int c = 0; c < dim[1]; c++){
			aNumer[t][c] = 0;
			aDenom[t][c] = 0;
		}
		tp[t].train = &train[t*total/NUM_THREADS];
		tp[t].total = total/NUM_THREADS;
		tp[t].dim = dim;
		tp[t].x = x;
		tp[t].s = s;
		tp[t].numer = aNumer[t];
		tp[t].denom = aDenom[t];
	}

	// Threads para cada conjunto de treinamento
   pthread_t thread[NUM_THREADS];
   pthread_attr_t attr;
   int pthread_return;
   void *pthread_status;

	// Iniciar threads
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	for(int t = 0; t < NUM_THREADS; t++){
		pthread_return = pthread_create(&thread[t], &attr, estimPart, (void *)&tp[t]); 
		if (pthread_return) {
			printf("ERROR; return code from pthread_create() is %d\n", pthread_return);
			exit(EXIT_FAILURE);
		}
	}

	// Liberar atributo e aguardar threads
	pthread_attr_destroy(&attr);
	for(int t = 0; t < NUM_THREADS; t++){
		pthread_return = pthread_join(thread[t], &pthread_status);
		if (pthread_return){
			printf("ERROR; return code from pthread_join() is %d\n", pthread_return);
			exit(EXIT_FAILURE);
		}
		//printf("Finalizado thread %d\n",t);
	}

	// Acumuladores do numerador e denominador 
	// para a variável dependente estimada
	float *numer = (float*)malloc(dim[1]*sizeof(float));
	float *denom = (float*)malloc(dim[1]*sizeof(float));
	for (int c = 0; c < dim[1]; c++){
		numer[c] = 0;
		denom[c] = 0;
	}
	for (int t = 0; t < NUM_THREADS; t++){
		// Estimativa com verificação de divisão por zero
		// para cada dimensão da variável dependente
		for (int c = 0; c < dim[1]; c++){
			numer[c] += aNumer[t][c];
			denom[c] += aNumer[t][c];
		}
		free(aNumer[t]);
		free(aDenom[t]);
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

void estimar(struct pathSet *train, struct pathSet *test, float *errsum){
	// Vetor da estimativa 
	float *y = (float*)malloc(sizeof(float)*train->dim[1]);
	// Erro da estimativa
	float err = 0;
	// Tamanho de um caminho (soma das dimensões dos vértices)
	unsigned int dims = train->dim[0]+train->dim[1];
	unsigned int *dim = train->dim;
	// Parâmetro sigma (variância)
	float sigma = 1.0/log(train->total);
	// Expressão envolvendo sigma no numerador do fator comum é constante
	float s = 2*pow(sigma,2);

	// Iterar em todo o conjunto de teste
	for (int i = 0; i < test->total; i++){
		estim(train->data.f, train->total, train->dim, &test->data.f[i*dims], y, s);
		// Sem endereço para o erro somado, substituir valores no conjunto estimado
		if ( errsum == NULL ){
			// Escrever estimativa
			memcpy(&test->data.f[i*dims+dim[0]], y, dim[1]*sizeof(float));
		}
		else {
			// Erro da estimativa
			err = sqrt(dist(&test->data.f[i*dims], y, dim[1]));
			// Erro acumulado (sem raiz)
			*errsum += err;
		}
	}
	free(y);
}

int main(int argc, char **argv){
	struct pathSet train, estim;
	// Carregar arquivo das amostras de treinamento
	pathSetLoad(TRAIN, &train);

	// Arquivo de teste
	pathSetLoad(TEST, &estim);

	printf("Conjunto de treinamento: %d amostras.\n", train.total);
	printf("Dimensões da variável independente: %d\n", train.dim[0]);
	printf("Dimensões da variável dependente:   %d\n", train.dim[1]);

	// Calcular o erro ou salvar um arquivo com o resultado
	printf("Estimando %d amostras de teste...\n", estim.total);
	if (argc > 1 ){
		// Salvar resultado no arquivo informado
		estimar(&train, &estim, NULL);
		pathSetSave(argv[1], &estim);
		printf("Resultado salvo em %s\n", argv[1]);
	}
	else {
		// Calcular o erro médio para as estimativas
		float errsum = 0;
		estimar(&train, &estim, &errsum);
		printf("Erro médio: %f\n", errsum / (float)estim.total);
	}

	return 0;
}
