#include <stdio.h>
#include <malloc.h>
#include <math.h>
// CUDA runtime
#include <cuda_runtime.h>
#include <curand.h>

//this must be smaller than 16 (but is realisticly 3 or 4 or maybe 5)
#define BLK_SIZE 3
#define SUD_SIZE (BLK_SIZE*BLK_SIZE)

//to describe the sudoku in a form that is suitable for fast access, we split
//it in two parts: One part describes the constant structure, i.e. the
//positions of the unknowns and the hints, and is the same for all instances.
//The other parts contains the values of the unknowns, the current energy, and
//space for bookkeeping.

typedef struct{
	//There are three tables that map an idx for each unknown to the entities it
	//belongs, i.e. its row, column and block. The size is chosen to be a safe
	//upper bound for the number of unknowns.
	char unk_row[SUD_SIZE*SUD_SIZE];
	char unk_col[SUD_SIZE*SUD_SIZE];
	char unk_blk[SUD_SIZE*SUD_SIZE];

	//There are three tables that contain the counts for the hints for each
	//symbol for each entity, souch that hnt_*[idx*SUD_SIZE + sym] gives the
	//number of occurences of hints of value sym in entity idx.
	char hnt_row[SUD_SIZE*SUD_SIZE];
	char hnt_col[SUD_SIZE*SUD_SIZE];
	char hnt_blk[SUD_SIZE*SUD_SIZE];

	//Number of unknowns
	unsigned int n_unk;
} sudoku_const_t;

typedef struct {
	//The energy of the current configuration
	int energy;
	//This table maps an idx of a unknown to its value.
	char unk_val[SUD_SIZE*SUD_SIZE];

	//This is space for accumulating counts.  cnt_*[sym*SUD_SIZE + idx] gives
	//the number of occurences of symbol sym in entity idx
	char cnt_row[SUD_SIZE*SUD_SIZE];
	char cnt_col[SUD_SIZE*SUD_SIZE];
	char cnt_blk[SUD_SIZE*SUD_SIZE];
} sudoku_t;

__constant__ sudoku_const_t con[1];
__constant__ int lut_exp[256];

//each int is good for four unknowns, so n_ints should be sufficiently large
__global__ void init(sudoku_t *suds, unsigned int* energies, unsigned int* random, unsigned n_ints){
	int thread_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int i,sym;

	sudoku_t* sud = &(suds[thread_idx]);

	for(i=0;i<con->n_unk;i++){
		sud->unk_val[i] = (random[n_ints*thread_idx + (i/4)] >> (8*(i%4))) % SUD_SIZE;
	}

	//put the hints into the counters
	for(i=0;i<SUD_SIZE*SUD_SIZE;i++){
		sud->cnt_row[i] = con->hnt_row[i];
		sud->cnt_col[i] = con->hnt_col[i];
		sud->cnt_blk[i] = con->hnt_blk[i];
	}

	//count the unknowns
	for(i=0;i<con->n_unk;i++){
		sud->cnt_row[con->unk_row[i]*SUD_SIZE + sud->unk_val[i]]++;
		sud->cnt_col[con->unk_col[i]*SUD_SIZE + sud->unk_val[i]]++;
		sud->cnt_blk[con->unk_blk[i]*SUD_SIZE + sud->unk_val[i]]++;
	}

	//calculate the energy
	sud->energy = 0;
	for(i=0;i<SUD_SIZE;i++){
		for(sym=0;sym<SUD_SIZE;sym++){
			sud->energy += (sud->cnt_row[i*SUD_SIZE + sym] == 0) ? 1 : sud->cnt_row[i*SUD_SIZE + sym] - 1;
			sud->energy += (sud->cnt_col[i*SUD_SIZE + sym] == 0) ? 1 : sud->cnt_col[i*SUD_SIZE + sym] - 1;
			sud->energy += (sud->cnt_blk[i*SUD_SIZE + sym] == 0) ? 1 : sud->cnt_blk[i*SUD_SIZE + sym] - 1;
		}
	}
	energies[thread_idx] = sud->energy;
}

//n_ints is the number of random 32 bit integers per thread supplied through
//random. Each int is good for one MC steps
__global__ void round(sudoku_t *suds, unsigned int* energies, unsigned int* random, unsigned int n_ints){
	int steps,i,sym;
	int delta;
	char idx, val, old_val;
	int rnd;
	int thread_idx = blockIdx.x*blockDim.x + threadIdx.x;

	sudoku_t* sud = &(suds[thread_idx]);

	for(steps=0;steps<n_ints;steps++){
		//dissect the random number
		idx = (random[n_ints*thread_idx + steps] >> 0) % con->n_unk;
		val = (random[n_ints*thread_idx + steps] >> 8) % SUD_SIZE;
		rnd = (random[n_ints*thread_idx + steps] >> 16);

		//save old value, put new value in
		old_val = sud->unk_val[idx];
		sud->unk_val[idx] = val;

		//put the hints into the counters
		for(i=0;i<SUD_SIZE*SUD_SIZE;i++){
			sud->cnt_row[i] = con->hnt_row[i];
			sud->cnt_col[i] = con->hnt_col[i];
			sud->cnt_blk[i] = con->hnt_blk[i];
		}

		//count the unknowns
		for(i=0;i<con->n_unk;i++){
			sud->cnt_row[con->unk_row[i]*SUD_SIZE + sud->unk_val[i]]++;
			sud->cnt_col[con->unk_col[i]*SUD_SIZE + sud->unk_val[i]]++;
			sud->cnt_blk[con->unk_blk[i]*SUD_SIZE + sud->unk_val[i]]++;
		}

		//calculate the energy
		delta = -sud->energy;
		for(i=0;i<SUD_SIZE;i++){
			for(sym=0;sym<SUD_SIZE;sym++){
				delta += (sud->cnt_row[i*SUD_SIZE + sym] == 0) ? 1 : sud->cnt_row[i*SUD_SIZE + sym] - 1;
				delta += (sud->cnt_col[i*SUD_SIZE + sym] == 0) ? 1 : sud->cnt_col[i*SUD_SIZE + sym] - 1;
				delta += (sud->cnt_blk[i*SUD_SIZE + sym] == 0) ? 1 : sud->cnt_blk[i*SUD_SIZE + sym] - 1;
			}
		}

		if((sud->energy == 0) || ((delta > 0) && (lut_exp[delta > 255 ? 255 : delta] < rnd))){
			//discard step
			sud->unk_val[idx] = old_val;
		} else {
			sud->energy += delta;
		}
	}
	energies[thread_idx] = sud->energy;
}

void add_unk(sudoku_const_t* sud, int i, int j){
	int blk = (i / BLK_SIZE)*BLK_SIZE + (j / BLK_SIZE);
	sud->unk_row[sud->n_unk] = i;
	sud->unk_col[sud->n_unk] = j;
	sud->unk_blk[sud->n_unk] = blk;
	sud->n_unk++;
}

void add_hin(sudoku_const_t* sud, int i, int j, int val){
	int blk = (i / BLK_SIZE)*BLK_SIZE + (j / BLK_SIZE);
	sud->hnt_row[val*SUD_SIZE + i]++;
	sud->hnt_col[val*SUD_SIZE + j]++;
	sud->hnt_blk[val*SUD_SIZE + blk]++;
}

void hint1(sudoku_const_t* sud){
	sud->n_unk = 0;
	int i,j;
	for(i=0;i<SUD_SIZE*SUD_SIZE;i++){
		sud->hnt_row[i] = 0;
		sud->hnt_col[i] = 0;
		sud->hnt_blk[i] = 0;
	}
	int vals[81] = {
		0, 9, 0, 7, 0, 3, 0, 6, 0, 0, 5, 0, 4, 0, 8, 0, 1, 0, 1, 0, 8, 0, 5, 0, 9, 0,
		4, 7, 0, 0, 1, 8, 2, 0, 0, 3, 0, 4, 2, 0, 6, 0, 5, 8, 0, 3, 0, 0, 9, 4, 5, 0,
		0, 2, 8, 0, 6, 0, 7, 0, 1, 0, 9, 0, 2, 0, 8, 0, 1, 0, 5, 0, 0, 1, 0, 6, 0, 9,
		0, 4, 0};
	for(i=0;i<SUD_SIZE;i++){
		for(j=0;j<SUD_SIZE;j++){
			if(vals[SUD_SIZE*i+j] == 0){
				add_unk(sud,i,j);
			} else {
				add_hin(sud,i,j,vals[SUD_SIZE*i+j]-1);
			}
		}
	}
}

void empty(sudoku_const_t* sud){
	sud->n_unk = 0;
	int i,j;
	for(i=0;i<SUD_SIZE*SUD_SIZE;i++){
		sud->hnt_row[i] = 0;
		sud->hnt_col[i] = 0;
		sud->hnt_blk[i] = 0;
	}
	for(i=0;i<SUD_SIZE;i++){
		for(j=0;j<SUD_SIZE;j++){
			add_unk(sud,i,j);
		}
	}
}

void hint16(sudoku_const_t* sud){
	sud->n_unk = 0;
	int i,j;
	for(i=0;i<SUD_SIZE*SUD_SIZE;i++){
		sud->hnt_row[i] = 0;
		sud->hnt_col[i] = 0;
		sud->hnt_blk[i] = 0;
	}
	int vals[256] = {
		0, 2, 0, 1, 0, 9, 0, 0, 0, 0, 11, 0, 0, 15, 14, 0, 0, 13, 11, 0, 0, 0, 0, 0, 9, 0, 7, 0, 16, 8, 0, 12, 
		4, 0, 10, 7, 15, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 9, 0, 0, 0, 0, 10, 0, 0, 12, 13, 0, 0, 0, 0, 4, 0, 0, 
		3, 0, 0, 0, 0, 0, 0, 0, 0, 16, 9, 11, 0, 0, 0, 0, 15, 5, 0, 16, 0, 0, 0, 4, 1, 0, 3, 0, 0, 10, 0, 0, 
		1, 0, 9, 12, 13, 15, 0, 0, 0, 6, 10, 0, 4, 0, 0, 0, 0, 11, 0, 0, 2, 6, 0, 0, 0, 0, 0, 0, 13, 0, 9, 8, 
		7, 10, 0, 6, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0, 8, 0, 0, 0, 0, 11, 0, 13, 7, 0, 0, 0, 6, 10, 1, 12, 0, 4, 
		0, 0, 12, 0, 0, 5, 0, 16, 7, 0, 0, 0, 11, 0, 10, 13, 0, 0, 0, 0, 6, 8, 14, 0, 0, 0, 0, 0, 0, 0, 0, 2, 
		0, 0, 3, 0, 0, 0, 0, 13, 16, 0, 0, 12, 0, 0, 0, 0, 13, 0, 0, 4, 0, 0, 0, 0, 11, 0, 0, 14, 2, 1, 0, 3, 
		9, 0, 2, 5, 0, 3, 0, 15, 0, 0, 0, 0, 0, 14, 7, 0, 0, 7, 1, 0, 0, 14, 0, 0, 0, 0, 5, 0, 8, 0, 16, 0};
	for(i=0;i<SUD_SIZE;i++){
		for(j=0;j<SUD_SIZE;j++){
			if(vals[SUD_SIZE*i+j] == 0){
				add_unk(sud,i,j);
			} else {
				add_hin(sud,i,j,vals[SUD_SIZE*i+j]-1);
			}
		}
	}
}

void energy_stats(int round, unsigned int *energies, unsigned int n_threads){
	int max_energy=0, min_energy=100000, total_energy;
	int i;
	for(i=0;i<n_threads;i++){
		max_energy = (energies[i] > max_energy) ? energies[i] : max_energy;
		min_energy = (energies[i] < min_energy) ? energies[i] : min_energy;
		total_energy += energies[i];
	}
	printf("Round: %d Mean: %f Min: %d Max: %d\n",round,float(total_energy)/n_threads, min_energy, max_energy);
}

#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS 4

int main(){
	int i,j;
	int n_ints;

	sudoku_const_t h_sudoku_const[1];
	sudoku_t *d_sudoku;

	int h_lut_exp[256];
	unsigned int *d_random;
	unsigned int *d_energies, *h_energies;

	dim3 threads_per_block(THREADS_PER_BLOCK);
	dim3 num_blocks(NUM_BLOCKS);
	int n_threads = threads_per_block.x*num_blocks.x;

	//set up constant sudoku description
	if(SUD_SIZE==9){
		empty(h_sudoku_const);
	} else {
		hint16(h_sudoku_const);
	}
	cudaMemcpyToSymbol(con,h_sudoku_const,sizeof(sudoku_const_t));
	n_ints = 1 + h_sudoku_const->n_unk/4;
	printf("Unknowns: %d n_ints: %d\n",h_sudoku_const->n_unk,n_ints);

	//set up random number generator
	curandGenerator_t gen;
	curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen,12345ULL);
	cudaMalloc(&d_random,n_threads*n_ints*sizeof(unsigned int));

	//set up exp(delta_E/T) lut
	//256 is a safe upper bound for SUD_SIZE 9 and we clamp delta E for
	//SUD_SIZE 16. But anyway, for useful temperatures it is zero very quickly
	//anyway
	float temp = 0.7;
	for(i=0;i<256;i++){
		h_lut_exp[i] = exp(-i/temp)*(1<<16);
	}
	cudaMemcpyToSymbol(lut_exp,h_lut_exp,256*sizeof(int));

	h_energies = (unsigned int*) malloc(n_threads*sizeof(unsigned int));
	cudaMalloc(&d_energies,n_threads*sizeof(unsigned int));

	//setup and initialize variable sudoku description
	cudaMalloc(&d_sudoku,n_threads*sizeof(sudoku_t));
	curandGenerate(gen,d_random,n_ints*n_threads);
	init<<<num_blocks,threads_per_block>>>(d_sudoku,d_energies,d_random,n_ints);

	for(j=0;j<500;j++){
		curandGenerate(gen,d_random,n_ints*n_threads);
		round<<<num_blocks,threads_per_block>>>(d_sudoku,d_energies,d_random,n_ints);

		//check energies
		cudaMemcpy(h_energies,d_energies,n_threads*sizeof(unsigned int),cudaMemcpyDeviceToHost);
		energy_stats(j,h_energies,n_threads);
	}

	cudaFree(d_random);
	curandDestroyGenerator(gen);
}





