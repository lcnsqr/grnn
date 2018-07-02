/*
 * Funções para entrada/saída de arquivos 
 * para armazenar pares de variáveis multivariadas
 */

// Estrutura do conjunto de dados
struct pathSet {
	// Formato dos números (magic number)
	// 0x01: char (unsigned byte)
	// 0x04: int (4 bytes)
	// 0x0F: float (4 bytes)
	// 0x0D: double (8 bytes)
	char type;
	// Quantidade total de caminhos
	unsigned int total;
	// Número de vértices em cada caminho
	unsigned int vertices;
	// Número de elementos para cada vértice 
	unsigned int *dim;
	// Ocupação em bytes
	unsigned int size;
	// Ponteiro para os dados
	union {
		char *c;
		int *i;
		float *f;
		double *d;
	} data;
};

void pathSetLoad(const char* filename, struct pathSet *ps){
	// Buffer para ler o formato 
	int *buf = (int*)malloc(4);
	// Ponteiro do arquivo de entrada
	FILE *fp;
	fp = fopen(filename, "r");
	// Tipo dos dados
	fread(buf, 4, 1, fp);
	ps->type = (*buf & 0x000000FF);
	free(buf);
	// Quantidade de caminhos
	fread(&ps->total, 4, 1, fp);
	// Número de vértices no caminho
	fread(&ps->vertices, 4, 1, fp);
	// Número de elementos em cada vértice
	ps->dim = (unsigned int*)malloc(4 * ps->vertices);
	fread(ps->dim, 4, ps->vertices, fp);
	// Total de valores em cada componente de cada vértice
	fread(&ps->size, 4, 1, fp);
	// Ler dados
	switch ( ps->type ){
		case 0x01:
			// unsigned char
			ps->data.c = (char*)malloc(ps->size);
			fread(ps->data.c, ps->size, 1, fp);
		break;
		case 0x04:
			// integer
			ps->data.i = (int*)malloc(ps->size * sizeof(int));
			fread(ps->data.i, ps->size * sizeof(int), 1, fp);
		break;
		case 0x0f:
			// float
			ps->data.f = (float*)malloc(ps->size * sizeof(float));
			fread(ps->data.f, ps->size * sizeof(float), 1, fp);
		break;
		case 0x0d:
			// double
			ps->data.d = (double*)malloc(ps->size * sizeof(double));
			fread(ps->data.d, ps->size * sizeof(double), 1, fp);
		break;
	}
	// Liberar arquivo aberto
	fclose(fp);
}

void pathSetSave(const char* filename, struct pathSet *ps){
	// Buffer para ler o formato 
	int *buf = (int*)malloc(4);
	*buf = ps->type;
	// Ponteiro do arquivo
	FILE *fp;
	fp = fopen(filename, "w");
	// Escrever Magic number 
	fwrite(buf, 4, 1, fp);
	// Liberar buffer
	free(buf);
	// Quantidade de caminhos
	fwrite(&ps->total, 4, 1, fp);
	// Número de vértices no caminho
	fwrite(&ps->vertices, 4, 1, fp);
	// Número de elementos em cada vértice
	fwrite(ps->dim, 4, ps->vertices, fp);
	// Total de valores em cada componente de cada vértice
	fwrite(&ps->size, 4, 1, fp);
	// Escrever dados
	switch ( ps->type ){
		case 0x01:
			// unsigned char
			fwrite(ps->data.c, ps->size, 1, fp);
		break;
		case 0x04:
			// integer
			fwrite(ps->data.i, ps->size * sizeof(int), 1, fp);
		break;
		case 0x0f:
			// float
			fwrite(ps->data.f, ps->size * sizeof(float), 1, fp);
		break;
		case 0x0d:
			// double
			fwrite(ps->data.d, ps->size * sizeof(double), 1, fp);
		break;
	}
	// Fechar ponteiro do arquivo
	fclose(fp);
}
