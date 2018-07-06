#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "pathio.h"

// Arquivo das amostras
#define TRAIN "train.bin"
// Arquivos de teste
#define TEST "test.bin"

// Tamanho da memória compartilhada por bloco (GPU)
unsigned int sharedMemPerBlock;
// Número máximo de threads por bloco
unsigned int maxThreadsPerBlock;

// Implementação da distância entre dois vetores
float dist(float *v, float *w, int n){
	// Quadrado da distância euclidiana entre o vetores v e w
	float d = 0;
	for (int i = 0; i < n; i++){
		d += pow(w[i]-v[i], 2);
	}
	return d;
}

// Estimar a variável dependente (kernel)
// train: Conjunto de treinamento
// total: Total de amostras no conjunto de treinamento
// threadsPerBlock: Threads por bloco
// dim: Dimensões da variável independente e dependente
// x: Variável independente lida
// yPart: Acumulador das somas parciais
// s: Parâmetro da regressão
__global__ void estimKernel(float *train, const unsigned int total, const unsigned int threadsPerBlock, unsigned int *dim, float *x, float *yPart, const float s){
	// Ignorar se thread atual ultrapassou total de amostras
	if ( blockIdx.x * threadsPerBlock + threadIdx.x + 1 > total ) return;
	// Tamanho de cada amostra (variável independente e dependente)
	unsigned int dims = dim[0]+dim[1];
	// Fator comum das operações
	float f;
	// Distância estimando-amostra
	float d = 0;

	// Atalhos para a amostra na memória compartilhada
	float *sx;
	float *sy;

	// Carregar amostra na memória compartilhada
	extern __shared__ float sData[];
	sx = &sData[threadIdx.x * dims];
	sy = &sData[threadIdx.x * dims + dim[0]];

	// Guardar diferença entre estimando 
	// e variável independente da amostra
	float dif;

	for(int c = 0; c < dim[0]; c++){
		sx[c] = train[(blockIdx.x * threadsPerBlock + threadIdx.x) * dims + c];
		// Distância entre estimando e variável independente da amostra
		//d += pow( x[c] - sx[c], 2);
		dif = __fsub_rn(x[c], sx[c]);
		d = __fadd_rn(d, __fmul_rn(dif, dif));
	}
	for(int c = 0; c < dim[1]; c++){
		sy[c] = train[(blockIdx.x * threadsPerBlock + threadIdx.x) * dims + dim[0] + c];
	}

	// Sincronizar threads
	__syncthreads();
	
	// Fator comum
	//f = exp( -d / s );
	f = __expf( __fdiv_rn(-d, s));

	// Atalhos para soma parcial
	float *numer = &yPart[blockIdx.x * 2 * dim[1]];
	float *denom = &yPart[blockIdx.x * 2 * dim[1] + dim[1]];
	// Efetuar a soma parcial para cada dimensão da variável dependente
	for(unsigned int c = 0; c < dim[1]; c++){
		// Parcial do numerador
		atomicAdd( &numer[c], sy[c] * f );
		// Parcial do denominador
		atomicAdd( &denom[c], f );
	}
}

void testarDev(struct pathSet *train, struct pathSet *test, float sigma, float *errsum){
	// Registrar conjunto de treinamento na memória para 
	// evitar paginação e agilizar o acesso pela GPU ao
	// mapear a memória entre o host e a memória da GPU
	cudaHostRegister(train->data.f, train->size, cudaHostRegisterMapped);
	// Copiar conjunto de treinamento para memória do dispositivo
	float *trainDataDev;
	//cudaHostGetDevicePointer(&trainDataDev, train->data.f, 0);
	cudaMalloc(&trainDataDev, train->size);
	cudaMemcpy(trainDataDev, train->data.f, train->size, cudaMemcpyHostToDevice);

	// Dimensões da variável independente e dependente
	unsigned int *dim;
	unsigned int *dimDev;
	dim = train->dim;
	cudaMalloc(&dimDev, 2*sizeof(unsigned int));
	cudaMemcpy(dimDev, dim, 2*sizeof(unsigned int), cudaMemcpyHostToDevice);

	// Tamanho de um caminho (soma das dimensões dos vértices)
	unsigned int dims = dim[0]+dim[1];
	// Threads por bloco (quantas amostras cabem num bloco)
	unsigned int threadsPerBlock = sharedMemPerBlock / (dims*sizeof(float));
	if (threadsPerBlock > maxThreadsPerBlock){
		// Número de threads ultrapassou o máximo permitido
		threadsPerBlock = maxThreadsPerBlock;
	}
	// Tamanho da memória compartilhada pelo bloco, 
	// utilizada para armazenar uma amostra por thread
	unsigned int sharedSize = threadsPerBlock * dims * sizeof(float);
	// Total de blocos
	unsigned int blocksPerGrid = train->total / threadsPerBlock;

	// Variável independente para associar à estimativa
	float *xDev;
	cudaMalloc(&xDev, dim[0]*sizeof(float));

	// Cada bloco produz um par de somas parciais 
	// para cada dimensão da variável dependente
	float *yPart;
	float *yPartDev;
	cudaMallocHost(&yPart, 2*blocksPerGrid*dim[1]*sizeof(float));
	cudaMalloc(&yPartDev, 2*blocksPerGrid*dim[1]*sizeof(float));

	// Variáveis para agregar o numerador e o denominador da fração
	// para cada dimensão da variável dependente
	float *numer;
	float *denom;
	cudaMallocHost(&numer, dim[1]*sizeof(float));
	cudaMallocHost(&denom, dim[1]*sizeof(float));

	// Vetor da estimativa da variável dependente
	float *y;
	cudaMallocHost(&y, dim[1]*sizeof(float));

	// Expressão envolvendo sigma no numerador do fator comum é constante
	float s = 2*pow(sigma,2);

	// Erro da estimativa
	float err = 0;

	// Índice de parcela
	unsigned int p;

	// Iterar em todo o conjunto de teste
	for (int i = 0; i < test->total; i++){
		// Copiar variável independente a ser estimada pela GPU
		cudaMemcpy(xDev, &test->data.f[i*dims], dim[0]*sizeof(float), cudaMemcpyHostToDevice);

		// Invocar kernel
		estimKernel<<<blocksPerGrid, threadsPerBlock, sharedSize>>>(trainDataDev, train->total, threadsPerBlock, dimDev, xDev, yPartDev, s);
		
		// Copiar parciais da estimativa geradas
		cudaMemcpy(yPart, yPartDev, 2*blocksPerGrid*dim[1]*sizeof(float), cudaMemcpyDeviceToHost);
	
		// Computar estimativa 
		for (unsigned int d = 0; d < dim[1]; d++){
			numer[d] = 0;
			denom[d] = 0;
			for (unsigned int b = 0; b < blocksPerGrid; b++){
				p = b*2*dim[1]+d;
				numer[d] += yPart[p];
				denom[d] += yPart[p+dim[1]];
				// Zerar parcial para a próxima amostra
				yPart[p] = 0;
				yPart[p+dim[1]] = 0;
			}
		}
		// Apagar parciais no dispositivo
		cudaMemcpy(yPartDev, yPart, 2*blocksPerGrid*dim[1]*sizeof(float), cudaMemcpyHostToDevice);

		// Vetor final da estimativa
		for (unsigned int d = 0; d < dim[1]; d++){
			y[d] = numer[d] / denom[d];
		}

		/*
		// Exibir valores da condição inicial
		printf("Condição inicial\n");
		for (int c = 0; c < dim[0]; c++){
			printf("%.6f\n", test->data.f[i*dims + c]);
		}
		// Exibir estimativa e valor observado
		printf("Valor Observado:\n");
		for (int c = 0; c < dim[1]; c++){
			printf("%.6f\n", test->data.f[i*dims + dim[0] + c]);
		}
		printf("Estimativa: \n");
		for (int c = 0; c < dim[1]; c++){
			printf("%.6f\n", y[c]);
		}
		*/
		// Erro da estimativa
		err = sqrt(dist(&test->data.f[i*dims], y, dim[1]));
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
}

int main(int argc, char **argv){
	// Detectar GPU
	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
	if (error_id != cudaSuccess){
		printf("Falha: cudaGetDeviceCount devolveu %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		exit(EXIT_FAILURE);
	}
	if (deviceCount == 0){
		printf("Nenhum dispositivo CUDA encontrado\n");
		exit(EXIT_FAILURE);
	}
	// Utilizar o primeiro dispositivo encontrado
	cudaSetDevice(0);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	// Tamanho da memória compartilhada por bloco (em bytes)
	sharedMemPerBlock = deviceProp.sharedMemPerBlock;
	// Máximo de threads por bloco
	maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;

	// Mapeamento de memória entre o Host e a GPU
	cudaSetDeviceFlags(cudaDeviceMapHost);

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

	// Paralelo
	begin = clock();
	errsum = 0;
	testarDev(&train, &test, sigma, &errsum);
	end = clock();
	//printf("\nTempo: %f segundos\n", (double)(end - begin) / CLOCKS_PER_SEC);
	printf("Erro médio: %f\n\n", errsum / (float)test.total);

	return 0;
}
