/*
 * Solução numérica da equação do calor pelo
 * método Crank-Nicolson de diferenças finitas
 */

// Funções auxiliares para preenchimento de matriz
float zeros(int i, int j){
	return 0;
}
float eye(int i, int j){
	return ( i == j ) ? 1 : 0;
}

// Construção da matriz A em Aw_j+1 = Bw_j com 
// a condição de Dirichlet em a e Neuman em b.
// A: Matriz resultante
// lambda: constante de diferenças finitas
void leftA(Matrix* A, const float lambda){
	A->_[0] = 1;
	for (int i = 1; i < A->rows - 2; i++){
		A->_[i*A->cols+i-1] = -lambda/2;
		A->_[i*A->cols+i] = 1+lambda;
		A->_[i*A->cols+i+1] = -lambda/2;
	}
	A->_[(A->rows-2)*A->cols+A->cols-3] = -lambda/2;
	A->_[(A->rows-2)*A->cols+A->cols-2] = 1+lambda/2;
	A->_[(A->rows-2)*A->cols+A->cols-1] = -lambda/2;
	A->_[A->size-1] = 1;
}

// Construção da matriz B em Aw_j+1 = Bw_j com 
// a condição de Dirichlet em a e Neuman em b.
// B: Matriz resultante
// lambda: constante de diferenças finitas
void rightB(Matrix* B, const float lambda){
	B->_[0] = 1;
	for (int i = 1; i < B->rows - 2; i++){
		B->_[i*B->cols+i-1] = lambda/2;
		B->_[i*B->cols+i] = 1-lambda;
		B->_[i*B->cols+i+1] = lambda/2;
	}
	B->_[(B->rows-2)*B->cols+B->cols-3] = lambda/2;
	B->_[(B->rows-2)*B->cols+B->cols-2] = 1-lambda/2;
	B->_[(B->rows-2)*B->cols+B->cols-1] = lambda/2;
	B->_[B->size-1] = 1;
}

// Gerar inversa de matriz tridiagonal
// Res: Matriz resultante (inversa de A)
// A: Matriz tridiagonal
void invTri(Matrix* res, Matrix* A){
	// Matriz res deve ser identidade
	mtrxRebuildWith(res, A->rows, A->cols, &eye);
	// Copiar matriz, inversão altera seu conteúdo
	Matrix M;
	mtrxBuildNull(&M, A->rows, A->cols);
	mtrxEqual(&M, A);
	// Vetores auxiliares
	Matrix row[3], rowInv[3];
	mtrxBuild(&row[0], 1, M.cols);
	mtrxBuild(&row[1], 1, M.cols);
	mtrxBuild(&row[2], 1, M.cols);
	mtrxBuild(&rowInv[0], 1, M.cols);
	mtrxBuild(&rowInv[1], 1, M.cols);
	mtrxBuild(&rowInv[2], 1, M.cols);
	// Posição na matriz
	int p[2];
	p[0] = 0;
	p[1] = 0;
	// Matrix tridiagonal, com diagonal dominante estrita. 
	// Inversão por escalonamento simples.
	// Primeira linha já pronta
	for (int i = 1; i < M.rows - 1; i++){
		// i-ésima linha
		mtrxRow(&row[0], &M, i);
		// Na inversa
		mtrxRow(&rowInv[0], res, i);
		// Dividir pelo elemento em i-1
		mtrxScalar(&row[1], &row[0], 1.0/row[0]._[i-1]);
		// Na inversa
		mtrxScalar(&rowInv[1], &rowInv[0], 1.0/row[0]._[i-1]);
		// Subtrair com a linha anterior
		mtrxRow(&row[0], &M, i-1);
		mtrxMinus(&row[2], &row[1], &row[0]);
		// Na inversa
		mtrxRow(&rowInv[0], res, i-1);
		mtrxMinus(&rowInv[2], &rowInv[1], &rowInv[0]);
		// Dividir pelo elemento em i
		mtrxScalar(&row[0], &row[2], 1.0/row[2]._[i]);
		// Na inversa
		mtrxScalar(&rowInv[0], &rowInv[2], 1.0/row[2]._[i]);
		// Atualizar linha na Matriz
		p[0] = i;
		mtrxPaste(&M, &row[0], p);
		// Na inversa
		mtrxPaste(res, &rowInv[0], p);
	}
	// Finalizar inversão diagonal-superior resultante
	// Última linha já pronta
	for (int i = M.rows - 2; i > 0; i--){
		// i-ésima linha
		mtrxRow(&row[0], &M, i);
		// Na inversa
		mtrxRow(&rowInv[0], res, i);
		// (i+1)-ésima linha
		mtrxRow(&row[1], &M, i+1);
		// Na inversa
		mtrxRow(&rowInv[1], res, i+1);
		// Multiplicar pelo elemento acima na linha superior
		mtrxScalar(&row[2], &row[1], row[0]._[i+1]);
		// Na inversa
		mtrxScalar(&rowInv[2], &rowInv[1], row[0]._[i+1]);
		// Subtrair i-ésima com (i+1)-ésima
		mtrxMinus(&row[1], &row[0], &row[2]);
		// Na inversa
		mtrxMinus(&rowInv[1], &rowInv[0], &rowInv[2]);
		// Atualizar linha na Matriz
		p[0] = i;
		mtrxPaste(&M, &row[1], p);
		// Na inversa
		mtrxPaste(res, &rowInv[1], p);
	}
	// Limpeza
	mtrxDiscard(&M);
	mtrxDiscard(&row[0]);
	mtrxDiscard(&row[1]);
	mtrxDiscard(&row[2]);
	mtrxDiscard(&rowInv[0]);
	mtrxDiscard(&rowInv[1]);
	mtrxDiscard(&rowInv[2]);
}

// Gerar amostras
// alfa: Constante da equação
// a: Extremidade inicial
// b: Extremidade final
/*
 * Vetores antes e depois: w (valores iniciais e valores após)
 * Condição de Dirichlet no extremo a
 * Valor no extremo a definido pelo primeiro valor do vetor
 * Condição de Neuman no extremo b
 * Valor da derivada em relação a x no extremo b definido pelo último valor do vetor 
 */
// t: Estado depois de t segundos
void gerar(Matrix *w, float alfa, float a, float b, float t){

	// Total de pontos entre a e b (inclusivo).
	// Último valor do vetor corresponde à derivada da
	// condição de Neumann e não é contabilizado
	int m = w[0].rows - 1;
	// Copiar valores de antes para depois
	mtrxEqual(&w[1], &w[0]);

	// Constante da equação
	if ( alfa == 0 ){
		fprintf(stderr, "A constante da equação não pode ser nula. \n");
		exit(EXIT_FAILURE);
	}
	// Espaçamento entre pontos 
	float h = (b-a)/m;
	// Tamanho do passo no tempo calculado a partir 
	// da constante alfa e do espaçamento h
	float k = pow(h,2)/((2+1e-4)*alfa);
	// Limitar o passo no tempo em 1 segundo
	k = ( k > 1 ) ? 1 : k;
	// Constante auxiliar lambda
	float lambda = k*alfa/pow(h,2);
	//fprintf(stderr, "h = %.9f\nk = %.9f\n", h, k);

	// Matrizes da operação
	Matrix A, invA, B;
	// Iniciá-las com zeros para posterior preenchimento
	mtrxBuildWith(&A, m+1, m+1, &zeros);
	mtrxBuildWith(&invA, m+1, m+1, &eye);
	mtrxBuildWith(&B, m+1, m+1, &zeros);
	// Preenchimento das matrizes tridiagonais
	leftA(&A, lambda);
	rightB(&B, lambda);
	// Inverter A
	invTri(&invA, &A);
	// Construir a matriz inv(A)*B
	Matrix C;
	mtrxBuild(&C, m+1, m+1);
	mtrxMul(&C, &invA, &B);
	// Matrizes A, B, inv(A) não mais necessárias
	mtrxDiscard(&A);
	mtrxDiscard(&invA);
	mtrxDiscard(&B);

	// Tempo acumulado das iterações
	float ts = 0;
	// Intervalo de tempo
	if ( t <= 0 ){
		fprintf(stderr, "O intervalo de tempo não pode ser menor ou igual a zero. \n");
		exit(EXIT_FAILURE);
	}
	// Buffer para evitar escrever sobre w[0]
	Matrix buf;
	mtrxBuildNull(&buf, 1, 1);
	mtrxEqual(&buf, &w[0]);
	// Contar os passos no tempo até atingir o instante desejado
	while (ts <= t){
		// Próximo resultado a partir do estado atual
		mtrxMul(&w[1], &C, &buf);
		mtrxEqual(&buf, &w[1]);
		ts += k;
	}
	mtrxDiscard(&buf);
	mtrxDiscard(&C);
	// Vetor resultante na saída padrão
	//for (int i = 0; i < m; i++) fprintf(stdout, "%.25f\n", w[0]._[i]);
}
