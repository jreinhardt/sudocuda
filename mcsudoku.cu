/* The MIT License (MIT)
 *
 * Copyright (c) 2013 Johannes Reinhardt <jreinhardt@ist-dein-freund.de>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

/* Monte Carlo Sudoku solver in CUDA */


#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

//this must be smaller than 16 (but is realisticly 3 or 4 or maybe 5)
#define BLK_SIZE 3
#define SUD_SIZE (BLK_SIZE*BLK_SIZE)
#define N_ENTRIES (SUD_SIZE*SUD_SIZE)
#define THREADS_PER_BLOCK 256
#define N_BLOCKS 256
#define N_THREADS (THREADS_PER_BLOCK*N_BLOCKS)
#define N_STEPS 100

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//TODO: Why does this not work
//__constant__ int lut_exp[256];

//a form to describe a sudoku that is suitable for fast access, consists of two
//parts: One part describes the constant structure, i.e. the positions of the
//unknowns and the hints, and is the same for all instances.  The other parts
//contains the values of the unknowns, the current energy, and space for
//bookkeeping.

typedef struct {
	//Constant part, applies to all sudokus:

	//There are three tables that map an idx for each unknown to the entities it
	//belongs, i.e. its row, column and block.
	char* unk_row;
	char* unk_col;
	char* unk_blk;

	//There are three tables that contain the counts for the hints for each
	//symbol for each entity, souch that hnt_*[idx*SUD_SIZE + sym] gives the
	//number of occurences of hints of value sym in entity idx.
	char* hnt_row;
	char* hnt_col;
	char* hnt_blk;

	//Number of unknowns (a single number, for memory purposes pointerized)
	unsigned int n_unk;

	//Variable part: contains information about the individual instances
	//The energy of the current configuration
	unsigned int* energy;

	//prng state
	unsigned int* random;
	//This table maps an idx of a unknown to its value.
	char** unk_val;

	//This is space for accumulating counts.  cnt_*[sym*SUD_SIZE + idx] gives
	//the number of occurences of symbol sym in entity idx
	char** cnt_row;
	char** cnt_col;
	char** cnt_blk;

} sudoku_t;

__device__ __host__ void mc_step(sudoku_t sud, int* lut_exp, unsigned int thread_idx){
	char val,old_val;
	unsigned int energy,rnd;
	int i,delta,idx;

	//generate a new random number
	rnd = sud.random[thread_idx];
	rnd = 1664525*rnd + 1013904223;
	sud.random[thread_idx] = rnd;

	//dissect the random number
	idx = (rnd >> 0) % sud.n_unk;
	val = (rnd >> 8) % SUD_SIZE;
	rnd = (rnd >> 16);

	//save old value, put new value in //TODO: Suboptimal access pattern
	old_val = sud.unk_val[idx][thread_idx];
	sud.unk_val[idx][thread_idx] = val;

	//put the hints into the counters
	for(i=0;i<N_ENTRIES;i++){
		sud.cnt_row[i][thread_idx] = sud.hnt_row[i];
		sud.cnt_col[i][thread_idx] = sud.hnt_col[i];
		sud.cnt_blk[i][thread_idx] = sud.hnt_blk[i];
	}

	//count the unknowns TODO: Suboptimal access pattern.
	for(i=0;i<sud.n_unk;i++){
		sud.cnt_row[sud.unk_row[i]*SUD_SIZE + sud.unk_val[i][thread_idx]][thread_idx]++;
		sud.cnt_col[sud.unk_col[i]*SUD_SIZE + sud.unk_val[i][thread_idx]][thread_idx]++;
		sud.cnt_blk[sud.unk_blk[i]*SUD_SIZE + sud.unk_val[i][thread_idx]][thread_idx]++;
	}

	//calculate the energy
	energy = 0;
	for(i=0;i<N_ENTRIES;i++){
		energy += (sud.cnt_row[i][thread_idx] == 0) ? 1 : sud.cnt_row[i][thread_idx] - 1;
		energy += (sud.cnt_col[i][thread_idx] == 0) ? 1 : sud.cnt_col[i][thread_idx] - 1;
		energy += (sud.cnt_blk[i][thread_idx] == 0) ? 1 : sud.cnt_blk[i][thread_idx] - 1;
	}

	delta = energy - sud.energy[thread_idx];
	delta = delta > 255 ? 255 : delta;

	if(((delta > 0) && (lut_exp[delta > 255 ? 255 : delta] < rnd)) || (energy == 0)){
		//discard step TODO: suboptimal access pattern
		sud.unk_val[idx][thread_idx] = old_val;
	} else {
		sud.energy[thread_idx] = energy;
	}
}


//Memory is laid out such that data for different threads is as close together
//as possible to allow coalesced accesses
void sudoku_alloc_host(sudoku_t* sud, int* lut_exp, unsigned int n_unk){
	int i,j;

	sud->unk_row = (char*) malloc(n_unk*sizeof(char));
	sud->unk_col = (char*) malloc(n_unk*sizeof(char));
	sud->unk_blk = (char*) malloc(n_unk*sizeof(char));

	sud->hnt_row = (char*) malloc(N_ENTRIES*sizeof(char));
	sud->hnt_col = (char*) malloc(N_ENTRIES*sizeof(char));
	sud->hnt_blk = (char*) malloc(N_ENTRIES*sizeof(char));

	sud->n_unk = n_unk;

	sud->energy = (unsigned int*) malloc(N_THREADS*sizeof(unsigned int));
	sud->random = (unsigned int*) malloc(N_THREADS*sizeof(unsigned int));

	sud->unk_val = (char**) malloc(n_unk*sizeof(char*));
	for(i=0;i<n_unk;i++){
		sud->unk_val[i] = (char*) malloc(N_THREADS*sizeof(char));
	}

	sud->cnt_row = (char**) malloc(N_ENTRIES*sizeof(char*));
	sud->cnt_col = (char**) malloc(N_ENTRIES*sizeof(char*));
	sud->cnt_blk = (char**) malloc(N_ENTRIES*sizeof(char*));

	for(i=0;i<N_ENTRIES;i++){
		sud->cnt_row[i] = (char*) malloc(N_THREADS*sizeof(char));
		sud->cnt_col[i] = (char*) malloc(N_THREADS*sizeof(char));
		sud->cnt_blk[i] = (char*) malloc(N_THREADS*sizeof(char));
	}

	//Setup a simple prng (coefficients from Numerical recipies)
	for(i=0;i<N_THREADS;i++){
		sud->random[i] = 1664525*i + 1013904223;
		//run it a few times
		for(j=0;j<5;j++){
			sud->random[i] = 1664525*sud->random[i] + 1013904223;
		}
	}

	//Fill the unknowns
	for(i=0;i<sud->n_unk;i++){
		for(j=0;j<N_THREADS;j++){
			sud->unk_val[i][j] = sud->random[j] % SUD_SIZE;
			sud->random[j] = 1664525*sud->random[j] + 1013904223;
		}
	}
	//Set energy to a really high arbitrary value, this will cause the first step to be accepted
	for(i=0;i<N_THREADS;i++){
		sud->energy[i] = 1<<20;
	}
}

void sudoku_free_host(sudoku_t* sud){
	int i;
	free(sud->unk_row);
	free(sud->unk_col);
	free(sud->unk_blk);

	free(sud->hnt_row);
	free(sud->hnt_col);
	free(sud->hnt_blk);

	free(sud->energy);
	free(sud->random);

	for(i=0; i<sud->n_unk;i++){
		free(sud->unk_val[i]);
	}
	free(sud->unk_val);

	for(i=0;i<N_ENTRIES;i++){
		free(sud->cnt_row[i]);
		free(sud->cnt_col[i]);
		free(sud->cnt_blk[i]);
	}

	free(sud->cnt_row);
	free(sud->cnt_col);
	free(sud->cnt_blk);
}

void sudoku_free_device(sudoku_t* sud){
	int i;
	char **tmp_val, **tmp_row, **tmp_col, **tmp_blk;
	cudaFree(sud->unk_row);
	cudaFree(sud->unk_col);
	cudaFree(sud->unk_blk);

	cudaFree(sud->hnt_row);
	cudaFree(sud->hnt_col);
	cudaFree(sud->hnt_blk);

	cudaFree(sud->energy);
	cudaFree(sud->random);

	tmp_val = (char**) malloc(sud->n_unk*sizeof(char*));
	cudaMemcpy(tmp_val,sud->unk_val,sud->n_unk*sizeof(char*),cudaMemcpyDeviceToHost);
	for(i=0; i<sud->n_unk;i++){
		cudaFree(tmp_val[i]);
	}
	free(tmp_val);
	cudaFree(sud->unk_val);

	tmp_col = (char**) malloc(N_ENTRIES*sizeof(char*));
	tmp_row = (char**) malloc(N_ENTRIES*sizeof(char*));
	tmp_blk = (char**) malloc(N_ENTRIES*sizeof(char*));
	cudaMemcpy(tmp_row,sud->cnt_row,N_ENTRIES*sizeof(char*),cudaMemcpyDeviceToHost);
	cudaMemcpy(tmp_col,sud->cnt_col,N_ENTRIES*sizeof(char*),cudaMemcpyDeviceToHost);
	cudaMemcpy(tmp_blk,sud->cnt_blk,N_ENTRIES*sizeof(char*),cudaMemcpyDeviceToHost);
	for(i=0;i<N_ENTRIES;i++){
		cudaFree(tmp_row[i]);
		cudaFree(tmp_col[i]);
		cudaFree(tmp_blk[i]);
	}
	free(tmp_col);
	free(tmp_row);
	free(tmp_blk);
	cudaFree(sud->cnt_row);
	cudaFree(sud->cnt_col);
	cudaFree(sud->cnt_blk);
}


void sudoku_alloc_and_copy_device(sudoku_t* d_sud, sudoku_t* h_sud){
	int i;
	char **tmp_val, **tmp_row, **tmp_col, **tmp_blk;

	int n_unk = h_sud->n_unk;
	//alloc
	gpuErrchk(cudaMalloc(&d_sud->unk_row,n_unk*sizeof(char)));
	gpuErrchk(cudaMalloc(&d_sud->unk_col,n_unk*sizeof(char)));
	gpuErrchk(cudaMalloc(&d_sud->unk_blk,n_unk*sizeof(char)));

	gpuErrchk(cudaMalloc(&d_sud->hnt_row,N_ENTRIES*sizeof(char)));
	gpuErrchk(cudaMalloc(&d_sud->hnt_col,N_ENTRIES*sizeof(char)));
	gpuErrchk(cudaMalloc(&d_sud->hnt_blk,N_ENTRIES*sizeof(char)));

	gpuErrchk(cudaMalloc(&d_sud->energy, N_THREADS*sizeof(unsigned int)));
	gpuErrchk(cudaMalloc(&d_sud->random, N_THREADS*sizeof(unsigned int)));

	//Nested array setup is a bit annoying. The result of cudaMalloc can not
	//directly written to device memory, so we have to cache it in tmp and then
	//copy it to the device

	gpuErrchk(cudaMalloc(&d_sud->unk_val, n_unk*sizeof(char*)));
	tmp_val = (char**) malloc(n_unk*sizeof(char*));
	for(i=0;i<n_unk;i++){
		gpuErrchk(cudaMalloc(&tmp_val[i],N_THREADS*sizeof(char)));
	}
	gpuErrchk(cudaMemcpy(d_sud->unk_val,tmp_val,n_unk*sizeof(char*),cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc(&d_sud->cnt_row,N_ENTRIES*sizeof(char*)));
	gpuErrchk(cudaMalloc(&d_sud->cnt_col,N_ENTRIES*sizeof(char*)));
	gpuErrchk(cudaMalloc(&d_sud->cnt_blk,N_ENTRIES*sizeof(char*)));

	tmp_row = (char**) malloc(N_ENTRIES*sizeof(char*));
	tmp_col = (char**) malloc(N_ENTRIES*sizeof(char*));
	tmp_blk = (char**) malloc(N_ENTRIES*sizeof(char*));

	for(i=0;i<N_ENTRIES;i++){
		gpuErrchk(cudaMalloc(&tmp_row[i],N_THREADS*sizeof(char)));
		gpuErrchk(cudaMalloc(&tmp_col[i],N_THREADS*sizeof(char)));
		gpuErrchk(cudaMalloc(&tmp_blk[i],N_THREADS*sizeof(char)));
	}
	gpuErrchk(cudaMemcpy(d_sud->cnt_row,tmp_row, N_ENTRIES*sizeof(char*),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_sud->cnt_col,tmp_col, N_ENTRIES*sizeof(char*),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_sud->cnt_blk,tmp_blk, N_ENTRIES*sizeof(char*),cudaMemcpyHostToDevice));

  //copy
  //constant
	d_sud->n_unk = h_sud->n_unk;
	gpuErrchk(cudaMemcpy(d_sud->unk_row,h_sud->unk_row,sizeof(char)*n_unk,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_sud->unk_col,h_sud->unk_col,sizeof(char)*n_unk,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_sud->unk_blk,h_sud->unk_blk,sizeof(char)*n_unk,cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(d_sud->hnt_row,h_sud->hnt_row,sizeof(char)*N_ENTRIES,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_sud->hnt_col,h_sud->hnt_col,sizeof(char)*N_ENTRIES,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_sud->hnt_blk,h_sud->hnt_blk,sizeof(char)*N_ENTRIES,cudaMemcpyHostToDevice));

	//variable
	gpuErrchk(cudaMemcpy(d_sud->energy,h_sud->energy,sizeof(unsigned int)*N_THREADS,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_sud->random,h_sud->random,sizeof(unsigned int)*N_THREADS,cudaMemcpyHostToDevice));

	for(i=0;i<h_sud->n_unk;i++){
		gpuErrchk(cudaMemcpy(tmp_val[i],h_sud->unk_val[i],sizeof(char)*N_THREADS,cudaMemcpyHostToDevice));
	}

	for(i=0;i<N_ENTRIES;i++){
		gpuErrchk(cudaMemcpy(tmp_row[i],h_sud->cnt_row[i],sizeof(char)*N_THREADS,cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(tmp_col[i],h_sud->cnt_col[i],sizeof(char)*N_THREADS,cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(tmp_blk[i],h_sud->cnt_blk[i],sizeof(char)*N_THREADS,cudaMemcpyHostToDevice));
	}

	free(tmp_val);
	free(tmp_row);
	free(tmp_col);
	free(tmp_blk);
}

__global__ void step_gpu(sudoku_t sud, int* lut_exp){
	unsigned int thread_idx = blockIdx.x*blockDim.x + threadIdx.x;
	mc_step(sud, lut_exp, thread_idx);
}

void step_cpu(sudoku_t sud, int* lut_exp){
	unsigned int thread_idx;
	for(thread_idx=0; thread_idx < N_THREADS; thread_idx++){
		mc_step(sud, lut_exp, thread_idx);
	}
}


void add_unk(sudoku_t* sud, int i, int j){
	int blk = (i / BLK_SIZE)*BLK_SIZE + (j / BLK_SIZE);
	sud->unk_row[sud->n_unk] = i;
	sud->unk_col[sud->n_unk] = j;
	sud->unk_blk[sud->n_unk] = blk;
	sud->n_unk++;
}

void add_hin(sudoku_t* sud, int i, int j, int val){
	int blk = (i / BLK_SIZE)*BLK_SIZE + (j / BLK_SIZE);
	sud->hnt_row[val*SUD_SIZE + i]++;
	sud->hnt_col[val*SUD_SIZE + j]++;
	sud->hnt_blk[val*SUD_SIZE + blk]++;
}

void hint1(sudoku_t* sud){
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

void empty(sudoku_t* sud){
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

void hint16(sudoku_t* sud){
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
	int max_energy=0, min_energy=100000, total_energy=0;
	int i;
	for(i=0;i<n_threads;i++){
		max_energy = (energies[i] > max_energy) ? energies[i] : max_energy;
		min_energy = (energies[i] < min_energy) ? energies[i] : min_energy;
		total_energy += energies[i];
	}
	printf("Round: %03d Mean: %3.2f Min: %03d Max: %03d\r",round,float(total_energy)/n_threads, min_energy, max_energy);
	fflush(stdout);
}

int main(){
	int i,j;
	long int n_steps;
	float time;

	int *h_lut_exp, *d_lut_exp;
	h_lut_exp = (int*) malloc(256*sizeof(int));
	gpuErrchk(cudaMalloc(&d_lut_exp,256*sizeof(int)));

	unsigned int* energies;

	energies = (unsigned int*) malloc(N_THREADS*sizeof(unsigned int));

	dim3 threads_per_block(THREADS_PER_BLOCK);
	dim3 num_blocks(N_BLOCKS);

	sudoku_t d_sudoku, h_sudoku;

	//set up exp(delta_E/T) lut
	//256 is a safe upper bound for SUD_SIZE 9 and we clamp delta E for
	//SUD_SIZE 16. But anyway, for useful temperatures it is zero very quickly
	//anyway
	float temp = 0.7;
	for(i=0;i<256;i++){
		h_lut_exp[i] = exp(-i/temp)*(1<<16);
	}
	gpuErrchk(cudaMemcpy(d_lut_exp,h_lut_exp,256*sizeof(int),cudaMemcpyHostToDevice));

	//set up sudoku description
	if(SUD_SIZE==9){
		sudoku_alloc_host(&h_sudoku,h_lut_exp,81);
		empty(&h_sudoku);
	} else {
		sudoku_alloc_host(&h_sudoku, h_lut_exp, 165);
		hint16(&h_sudoku);
	}
	printf("Unknowns: %d\n",h_sudoku.n_unk);
	sudoku_alloc_and_copy_device(&d_sudoku,&h_sudoku);

	n_steps = N_THREADS*N_STEPS;
	//GPU Run
	time = clock();

	for(j=0;j<15000;j++){
		step_gpu<<<num_blocks,threads_per_block>>>(d_sudoku,d_lut_exp);
		gpuErrchk(cudaMemcpy(energies,d_sudoku.energy,N_THREADS*sizeof(unsigned int),cudaMemcpyDeviceToHost));
		energy_stats(j,energies,N_THREADS);
	}
	time = (clock() - time)/CLOCKS_PER_SEC;
	printf("\nGPU: Steps: %ld, Time: %f s, %.1f Steps/s\n",n_steps,time,n_steps/time);

	//CPU Run
	time = clock();

	for(j=0;j<N_STEPS;j++){
		step_cpu(h_sudoku,h_lut_exp);
		gpuErrchk(cudaMemcpy(energies,h_sudoku.energy,N_THREADS*sizeof(unsigned int),cudaMemcpyHostToHost));
		energy_stats(j,energies,N_THREADS);
	}
	time = (clock() - time)/CLOCKS_PER_SEC;
	printf("\nCPU: Steps: %ld, Time: %f s, %.1f Steps/s\n",n_steps,time,n_steps/time);

	sudoku_free_host(&h_sudoku);
	sudoku_free_device(&d_sudoku);
	free(energies);
	cudaFree(d_lut_exp);
}
