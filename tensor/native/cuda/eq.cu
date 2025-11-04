/**
 * Element-wise comparison between two tensors
 * For now we'll just support 2D Matrices.
 * TODO: Extend this to arbitrary tensor sizes.
 * Maybe just use the total size to support arbitrary kernels for this.
 */
template<typename scalar_t> 
__global__ void eq_kernel(const scalar_t* a, const scalar_t* b, bool* c, int32_t size){
    int32_t th_x = blockIdx.x * blockDim.x + threadIdx.x;
    if(th_x < size) c[th_x] = a[th_x] == b[th_x];
}

template __global__ void eq_kernel<float>(const scalar_t*, const scalar_t*, bool*, int32_t, int32_t);
template __global__ void eq_kernel<int32_t>(const scalar_t*, const scalar_t*, bool*, int32_t, int32_t);