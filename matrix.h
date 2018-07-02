/*
 * Funções para operações matriciais
 */

// Estrutura para vetor/matriz
typedef struct {
	int cols, rows;
	// Dados armazenados na ordem Row-major 
	float *_;
	// Número de elementos
	size_t size;
	// Quantidade de memória ocupada
	size_t memsize;
} Matrix;

void mtrxBuild(Matrix*, const int, const int);
void mtrxBuildNull(Matrix*, const int, const int);
void mtrxRebuild(Matrix*, const int, const int);
void mtrxBuildWith(Matrix*,const int,const int,float (*f)(const int, const int));
void mtrxRebuildWith(Matrix*, const int, const int, float (*f)(const int, const int));
void mtrxDiscard(Matrix*);
void mtrxEqual(Matrix*, Matrix*);
void mtrxScale(Matrix*, Matrix*, const float);
void mtrxScalar(Matrix*, Matrix*, const float);
void mtrxPlus(Matrix*, Matrix*, Matrix*);
void mtrxMinus(Matrix*, Matrix*, Matrix*);
void mtrxMul(Matrix*, Matrix*, Matrix*);
void mtrxTranspose(Matrix*, Matrix*);
void mtrxCopy(Matrix*, Matrix*, const int*);
void mtrxPaste(Matrix*, Matrix*, const int*);
void mtrxCol(Matrix*, Matrix*, const int);
void mtrxRow(Matrix*, Matrix*, const int);
void mtrxPrint(Matrix*, const char*);
void mtrxFillWith(Matrix*,float (*f)(const int, const int));
void mtrxParams(Matrix*,Matrix*,float (*f)(const float));
void mtrxReadFromFile(Matrix* res, const char* file);

void mtrxBuild(Matrix* res, const int rows, const int cols){
	res->size = rows * cols;
	res->memsize = sizeof(float)*res->size;
	res->_ = (float*)malloc(res->memsize);
	res->rows = rows;
	res->cols = cols;
}

void mtrxBuildNull(Matrix* res, const int rows, const int cols){
	res->size = rows * cols;
	res->_ = (float*)NULL;
	res->memsize = sizeof(float)*res->size;
	res->rows = rows;
	res->cols = cols;
}

void mtrxRebuild(Matrix* res, const int rows, const int cols){
	res->size = rows * cols;
	res->memsize = sizeof(float)*res->size;
	res->_ = (float*)realloc(res->_, res->memsize);
	res->rows = rows;
	res->cols = cols;
}

void mtrxBuildWith(Matrix* res , const int rows, const int cols , float (*f)(const int, const int)){
	res->size = rows * cols;
	res->memsize = sizeof(float)*res->size;
	res->_ = (float*)malloc(res->memsize);
	#pragma omp parallel for
	for (size_t k = 0; k < res->size; k++)
		res->_[k] = f(k / cols, k % cols);
	res->rows = rows;
	res->cols = cols;
}

void mtrxRebuildWith(Matrix* res , const int rows, const int cols , float (*f)(const int, const int)){
	res->size = rows * cols;
	res->memsize = sizeof(float)*res->size;
	res->_ = (float*)realloc(res->_, res->memsize);
	#pragma omp parallel for
	for (size_t k = 0; k < res->size; k++)
		res->_[k] = f(k / cols, k % cols);
	res->rows = rows;
	res->cols = cols;
}

void mtrxDiscard(Matrix* res){
	free(res->_);
	res->rows = 0;
	res->cols = 0;
	res->size = 0;
	res->memsize = 0;
}

void mtrxEqual(Matrix* res, Matrix* m){
	res->rows = m->rows;
	res->cols = m->cols;
	res->size = m->size;
	res->memsize = sizeof(float)*res->size;
	res->_ = (float*)realloc(res->_, res->memsize);
	memcpy(res->_, m->_, res->memsize);
}

void mtrxScale(Matrix* res, Matrix* m, const float s){
	#pragma omp parallel for
	for (size_t k = 0; k < res->size; k++)
		res->_[k] = s * m->_[k];
}
void mtrxScalar(Matrix* res, Matrix* m, const float s){
	mtrxScale(res, m, s);
}

void mtrxPlus(Matrix* res, Matrix* m1, Matrix* m2){
	#pragma omp parallel for
	for (size_t k = 0; k < res->size; k++)
		res->_[k] = m1->_[k] + m2->_[k];
}

void mtrxMinus(Matrix* res, Matrix* m1, Matrix* m2){
	#pragma omp parallel for
	for (size_t k = 0; k < res->size; k++)
		res->_[k] = m1->_[k] - m2->_[k];
}

void mtrxMul(Matrix* res, Matrix* m1, Matrix* m2){
	#pragma omp parallel for
	for (size_t k = 0; k < res->size; k++){
		int i = k / res->cols;
		int j = k % res->cols;
		res->_[k] = 0;
		for (int c = 0; c < m1->cols; c++){
			res->_[k] += m1->_[i * m1->cols + c] * m2->_[c * m2->cols + j];
		}
	}
}

void mtrxTranspose(Matrix* res, Matrix* m){
	#pragma omp parallel for
	for (size_t k = 0; k < res->size; k++){
		int i = k / res->cols;
		int j = k % res->cols;
		res->_[k] = m->_[j * m->cols + i];
	}
}

void mtrxCopy(Matrix *res, Matrix *m, const int *P){
	// Copiar de m a partir da posição P no tamanho definido por res
	// P[0]: linha
	// P[1]: coluna
	#pragma omp parallel for
	for (int i = 0; i < res->rows; i++){
		memcpy(res->_ + i * res->cols, m->_ + (P[0] + i) * m->cols + P[1], sizeof(float)*res->cols);
	}
}

void mtrxPaste(Matrix* res, Matrix* m, const int* P){
	// Colar m na posição P da matriz res
	// P[0]: linha
	// P[1]: coluna
	#pragma omp parallel for
	for (int i = 0; i < m->rows; i++){
		memcpy(res->_ + (P[0] + i) * res->cols + P[1], m->_ + i * m->cols, sizeof(float)*m->cols);
	}
}

void mtrxRow(Matrix* res, Matrix* m, const int k){
	memcpy(res->_, m->_ + k * m->cols, sizeof(float)*m->cols);
}

void mtrxCol(Matrix* res, Matrix* m, const int k){
	const int P[] = {0, k};
	mtrxCopy(res, m, P);
}

void mtrxFillWith(Matrix* res, float (*f)(const int, const int)){
	#pragma omp parallel for
	for (size_t k = 0; k < res->size; k++){
		int i = k / res->cols;
		int j = k % res->cols;
		res->_[k] = f(i,j);
	}
}

void mtrxParams(Matrix *res, Matrix *m, float (*f)(const float)){
	// Use each element of m as a parameter in f and put results in res
	#pragma omp parallel for
	for (size_t k = 0; k < m->size; k++){
		res->_[k] = f(m->_[k]);
	}
}

void mtrxReadFromFile(Matrix* res, const char* file){
	// Leitura de matriz (não iniciada) a partir de arquivo
	const int maxW = 4096;
	const int maxH = 4096;
	char** input;
	input = (char**)malloc(maxH*sizeof(char*));
	int i = 0;
	input[i] = (char*)malloc(maxW*sizeof(char));
	char* token;
	int j;
	float** table;
	table = (float**)malloc(maxH*sizeof(float*));
	table[i] = (float*)malloc(maxW*sizeof(float));
	FILE* f = fopen(file, "r");
	while ( NULL != fgets(input[i], maxW, f) && i < maxH ){
		input[i+1] = (char*)malloc(maxW*sizeof(char));
		table[i+1] = (float*)malloc(maxW*sizeof(float));
		token = strtok(input[i], " ");
		if ( token == NULL ) continue;
		j = 0;
		table[i][j++] = atof(token);
		while ( (token = strtok(NULL, " ")) != NULL )
			table[i][j++] = atof(token);
		free(input[i]);
		i++;
	}
	fclose(f);
	free(input);
	mtrxBuild(res, i, j);
	for (int k = 0; k < i; k++){
		memcpy(&res->_[k*j], table[k], j*sizeof(float));
		free(table[k]);
	}
	free(table);
}

void mtrxReadFromSTDIN(Matrix* res){
	// Leitura de matriz (não iniciada) a partir da STDIN
	const int maxW = 4096;
	const int maxH = 4096;
	char** input;
	input = (char**)malloc(maxH*sizeof(char*));
	int i = 0;
	input[i] = (char*)malloc(maxW*sizeof(char));
	char* token;
	int j;
	float** table;
	table = (float**)malloc(maxH*sizeof(float*));
	table[i] = (float*)malloc(maxW*sizeof(float));
	while ( NULL != fgets(input[i], maxW, stdin) && i < maxH ){
		input[i+1] = (char*)malloc(maxW*sizeof(char));
		table[i+1] = (float*)malloc(maxW*sizeof(float));
		token = strtok(input[i], " ");
		if ( token == NULL ) continue;
		j = 0;
		table[i][j++] = atof(token);
		while ( (token = strtok(NULL, " ")) != NULL )
			table[i][j++] = atof(token);
		free(input[i]);
		i++;
	}
	free(input);
	mtrxBuild(res, i, j);
	for (int k = 0; k < i; k++){
		memcpy(&res->_[k*j], table[k], j*sizeof(float));
		free(table[k]);
	}
	free(table);
}

void mtrxPrint(Matrix *res, const char* s){
	printf("%s = [\n", s);
	for (size_t k = 0; k < res->size; k++){
		printf("%.9f ", res->_[k]);
		if ( (k+1) % res->cols == 0 && k+1 < res->size ) printf(";\n");
	}
	printf("]\n");
}
