template<typename scalar_t>
__global__ void matmul_tile_kernel(const scalar_t* a, const scalar_t* b, scalar_t* c, 
    int32_t m, int32_t n, int32_t k, int32_t tile_size){ // Might not need to pass tile_size if blockDim is known to be equivalent
        extern __shared__ scalar_t tile[];
        float* shmem_a = tile;
        float* shmem_b = tile + tile_size*tile_size;

        int32_t th_x = blockIdx.x * blockDim.x + threadIdx.x;
        int32_t th_y = blockIdx.y * blockDim.y + threadIdx.y;
        int32_t num_tiles = (n-1) / tile_size + 1;

        if(th_x < n && th_y < k){
            scalar_t res = 0;
            for(int i = 0; i < n / num_tiles; ++i){
                // TODO: Add guards here
                // Load Tile
                shmem_a[threadIdx.y * tile_size + threadIdx.x] = a[m*th_y + (i*tile_size) + threadIdx.x];
                shmem_b[threadIdx.y * tile_size + threadIdx.x] = b[th_x + (i*tile_size + threadIdx.y)*n];

                // Synchronize
                __syncthreads();

                // Compute
                res += shmem_a[threadIdx.y * tile_size + threadIdx.x] * shmem_b[threadIdx.y * tile_size + threadIdx.x];
            }
            c[th_y*k + th_x] = res;
        }
}

template<typename scalar_t>
__global__ void matmul_tile_gradA_kernel(const scalar_t* grad_out, const scalar_t* b, scalar_t* da, 
    int32_t m, int32_t n, int32_t k, int32_t tile_size){

}

template<typename scalar_t>
__global__ void matmul_tile_gradB_kernel(const scalar_t* grad_out, const scalar_t* a, scalar_t* db, 
    int32_t m, int32_t n, int32_t k, int32_t tile_size){

}

template __global__ void matmul_tile_kernel<float>(const float*, const float*, float*, int32_t, int32_t, int32_t, int32_t);
template __global__ void matmul_tile_gradA_kernel<float>(const float*, const float*, float*, int32_t, int32_t, int32_t, int32_t);
template __global__ void matmul_tile_gradB_kernel<float>(const float*, const float*, float*, int32_t, int32_t, int32_t, int32_t);