// Definições do dispositivo
cudaDeviceProp deviceProp;

// Estimar a variável dependente (kernel)
// train: Conjunto de treinamento
// total: Total de amostras no conjunto de treinamento
// threadsPerBlock: Threads por bloco
// dim: Dimensões da variável independente e dependente
// x: Variável independente lida
// yPart: Acumulador das somas parciais
// s: Parâmetro da regressão
__global__ void estimKernel(const float* __restrict__ train, const unsigned int total, const unsigned int threadsPerBlock, const unsigned int* __restrict__ dim, const float* __restrict__ x, float* __restrict__ yPart, const float s){
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

	// Copiar cada dimensão da variável independente para memória compartilhada
	for(int c = 0; c < dim[0]; c++){
		sx[c] = train[(blockIdx.x * threadsPerBlock + threadIdx.x) * dims + c];
		// Distância entre estimando e variável independente da amostra
		dif = __fsub_rn(x[c], sx[c]);
		d = __fadd_rn(d, __fmul_rn(dif, dif));
	}
	// Copiar cada dimensão da variável dependente para memória compartilhada
	for(int c = 0; c < dim[1]; c++){
		sy[c] = train[(blockIdx.x * threadsPerBlock + threadIdx.x) * dims + dim[0] + c];
	}

	// Sincronizar threads
	__syncthreads();
	
	// Fator comum
	f = __expf( __fdiv_rn(-d, s));

	// Efetuar a soma parcial para cada dimensão da variável dependente
	for(unsigned int c = 0; c < dim[1]; c++){
		// Parcial do numerador
		atomicAdd( &yPart[blockIdx.x * 2 * dim[1] + c], sy[c] * f );
		// Parcial do denominador
		atomicAdd( &yPart[blockIdx.x * 2 * dim[1] + dim[1] + c], f );
	}
}

// Implementação da distância entre dois vetores para cálculo do erro
float dist(float *v, float *w, int n){
	// Quadrado da distância euclidiana entre o vetores v e w
	float d = 0;
	for (int i = 0; i < n; i++){
		d += pow(w[i]-v[i], 2);
	}
	return d;
}

// Identificar e iniciar o dispositivo
void init_gpu(){
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
	cudaGetDeviceProperties(&deviceProp, 0);

	// Mapeamento de memória entre o Host e a GPU
	cudaSetDeviceFlags(cudaDeviceMapHost);
}

void estimar(struct pathSet *train, struct pathSet *estim, float *errsum){
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
	unsigned int threadsPerBlock = deviceProp.sharedMemPerBlock / (dims*sizeof(float));
	if (threadsPerBlock > deviceProp.maxThreadsPerBlock){
		// Número de threads ultrapassou o máximo permitido
		threadsPerBlock = deviceProp.maxThreadsPerBlock;
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

	// Parâmetro sigma (variância)
	float sigma = 1.0/log(train->total);
	// Expressão envolvendo sigma no numerador do fator comum é constante
	float s = 2*pow(sigma,2);

	// Erro da estimativa
	float err = 0;

	// Índice de parcela
	unsigned int p;

	// Iterar em todo o conjunto a estimar
	for (int i = 0; i < estim->total; i++){
		// Copiar variável independente a ser estimada pela GPU
		cudaMemcpy(xDev, &estim->data.f[i*dims], dim[0]*sizeof(float), cudaMemcpyHostToDevice);

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

		// Sem endereço para o erro somado, substituir valores no conjunto estimado
		if ( errsum == NULL ){
			// Escrever estimativa
			cudaMemcpy(&estim->data.f[i*dims+dim[0]], y, dim[1]*sizeof(float), cudaMemcpyHostToHost);
		}
		else {
			// Erro da estimativa
			err = sqrt(dist(&estim->data.f[i*dims], y, dim[1]));
			// Erro acumulado (sem raiz)
			*errsum += err;
		}
	}
}
