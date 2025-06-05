#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>

#define WARP_SIZE 32
#define TILE_SIZE 16
#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__,       \
             __LINE__);                                                        \
      exit(1);                                                                 \
    }                                                                          \
  }

// 添加偏置的内核
__global__ void add_bias_kernel(float *output, const float *bias, 
                                int output_dim, int batch_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < batch_size * output_dim) {
    int neuron_idx = idx % output_dim;
    output[idx] += bias[neuron_idx];
  }
}

// 使用CUDA流实现分流并行
__global__ void linear_forward_kernel(const float *__restrict__ input,
                                      const float *__restrict__ weights,
                                      const float *__restrict__ bias,
                                      float *__restrict__ hidden, int input_dim,
                                      int hidden_dim, int batch_size) {
  // 二维线程布局: (batch, hidden_dim)
  int batch_idx = blockIdx.y;
  int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (batch_idx >= batch_size || neuron_idx >= hidden_dim)
    return;

  extern __shared__ float shared_mem[];
  float *input_tile = shared_mem;
  float *weight_tile = &shared_mem[TILE_SIZE * TILE_SIZE];

  float sum = 0.0f;

  for (int tile = 0; tile < (input_dim + TILE_SIZE - 1) / TILE_SIZE; tile++) {
    int input_idx = tile * TILE_SIZE + threadIdx.x;

    // 协作加载输入块
    if (input_idx < input_dim) {
      input_tile[threadIdx.x] = input[batch_idx * input_dim + input_idx];
    } else {
      input_tile[threadIdx.x] = 0.0f;
    }

    // 协作加载权重块
    int weight_idx = tile * TILE_SIZE + threadIdx.x;
    if (weight_idx < input_dim && neuron_idx < hidden_dim) {
      weight_tile[threadIdx.x] = weights[neuron_idx * input_dim + weight_idx];
    } else {
      weight_tile[threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // 计算点积
    for (int i = 0; i < TILE_SIZE; i++) {
      sum += input_tile[i] * weight_tile[i];
    }

    __syncthreads();
  }

  if (neuron_idx < hidden_dim) {
    sum += bias[neuron_idx];
    hidden[batch_idx * hidden_dim + neuron_idx] = sum;
  }
}

__global__ void linear_backward_kernel(const float *__restrict__ input,
                                       const float *__restrict__ grad_output,
                                       float *__restrict__ grad_weights,
                                       float *__restrict__ grad_bias,
                                       int input_dim, int hidden_dim,
                                       int batch_size) {
  extern __shared__ float shared_mem[];
  float *grad_accum = shared_mem;

  int tid = threadIdx.x;
  int neuron_idx = blockIdx.x;
  int batch_idx = blockIdx.y;

  // 初始化共享内存
  if (tid < input_dim) {
    grad_accum[tid] = 0.0f;
  }
  __syncthreads();

  if (neuron_idx < hidden_dim && batch_idx < batch_size) {
    float grad_val = grad_output[batch_idx * hidden_dim + neuron_idx];

    // 累加权重梯度
    for (int i = tid; i < input_dim; i += blockDim.x) {
      float partial = input[batch_idx * input_dim + i] * grad_val;
      atomicAdd(&grad_accum[i], partial);
    }

    // 累加偏置梯度
    if (tid == 0) {
      atomicAdd(&grad_bias[neuron_idx], grad_val);
    }
  }

  __syncthreads();

  // 原子更新全局内存
  if (tid < input_dim) {
    atomicAdd(&grad_weights[neuron_idx * input_dim + tid], grad_accum[tid]);
  }
}

// actor/critic & head
// (fc+tanh)1 forward kernel
__global__ void fc1_forward_kernel(const float *__restrict__ input,
                                   const float *__restrict__ weights,
                                   const float *__restrict__ bias,
                                   float *__restrict__ hidden, int input_dim,
                                   int hidden_dim, int batch_size) {
  // 二维线程布局: (batch, hidden_dim)
  int batch_idx = blockIdx.y;
  int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (batch_idx >= batch_size || neuron_idx >= hidden_dim)
    return;

  extern __shared__ float shared_mem[];
  float *input_tile = shared_mem;
  float *weight_tile = &shared_mem[TILE_SIZE * TILE_SIZE];

  float sum = 0.0f;

  for (int tile = 0; tile < (input_dim + TILE_SIZE - 1) / TILE_SIZE; tile++) {
    int input_idx = tile * TILE_SIZE + threadIdx.x;

    // 协作加载输入块
    if (input_idx < input_dim) {
      input_tile[threadIdx.x] = input[batch_idx * input_dim + input_idx];
    } else {
      input_tile[threadIdx.x] = 0.0f;
    }

    // 协作加载权重块
    int weight_idx = tile * TILE_SIZE + threadIdx.x;
    if (weight_idx < input_dim && neuron_idx < hidden_dim) {
      weight_tile[threadIdx.x] = weights[neuron_idx * input_dim + weight_idx];
    } else {
      weight_tile[threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // 计算点积
    for (int i = 0; i < TILE_SIZE; i++) {
      sum += input_tile[i] * weight_tile[i];
    }

    __syncthreads();
  }

  if (neuron_idx < hidden_dim) {
    sum += bias[neuron_idx];
    hidden[batch_idx * hidden_dim + neuron_idx] = tanhf(sum); // tanH
  }
}

// (fc+tanh)2 forward kernel
__global__ void fc2_forward_kernel(const float *__restrict__ hidden,
                                   const float *__restrict__ weights,
                                   const float *__restrict__ bias,
                                   float *__restrict__ output, int hidden_dim,
                                   int output_dim, int batch_size) {
  // 计算warp信息
  const int warp_id = threadIdx.x / WARP_SIZE; // Block内的warp索引
  const int lane_id = threadIdx.x % WARP_SIZE; // Warp内的线程索引(0-31)

  // 计算当前warp处理的输出元素索引
  const int output_idx = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id;
  if (output_idx >= batch_size * output_dim)
    return;

  // 分解索引到batch和神经元
  const int batch_idx = output_idx / output_dim;
  const int neuron_idx = output_idx % output_dim;

  // 使用warp级并行计算点积
  float sum = 0.0f;
  for (int i = lane_id; i < hidden_dim; i += WARP_SIZE) {
    sum += hidden[batch_idx * hidden_dim + i] *
           weights[neuron_idx * hidden_dim + i];
  }

  // Warp内归约求和
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
  }

  // 第一个线程写入结果
  if (lane_id == 0) {
    output[output_idx] = sum + bias[neuron_idx];
  }
}

// 改进的fc2反向传播kernel
__global__ void fc2_backward_kernel(const float *__restrict__ hidden,
                                    const float *__restrict__ grad_output,
                                    float *__restrict__ grad_weights,
                                    float *__restrict__ grad_bias,
                                    int hidden_dim, int output_dim,
                                    int batch_size) {
  extern __shared__ float shared_mem[];
  float *grad_accum = shared_mem;

  int tid = threadIdx.x;
  int neuron_idx = blockIdx.x;
  int batch_idx = blockIdx.y;

  // 初始化共享内存
  if (tid < hidden_dim) {
    grad_accum[tid] = 0.0f;
  }
  __syncthreads();

  if (neuron_idx < output_dim && batch_idx < batch_size) {
    float grad_val = grad_output[batch_idx * output_dim + neuron_idx];

    // 累加权重梯度
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
      float partial = hidden[batch_idx * hidden_dim + i] * grad_val;
      atomicAdd(&grad_accum[i], partial);
    }

    // 累加偏置梯度
    if (tid == 0) {
      atomicAdd(&grad_bias[neuron_idx], grad_val);
    }
  }

  __syncthreads();

  // 原子更新全局内存
  if (tid < hidden_dim) {
    atomicAdd(&grad_weights[neuron_idx * hidden_dim + tid], grad_accum[tid]);
  }
}

// 改进的fc1反向传播kernel
__global__ void fc1_backward_kernel(const float *__restrict__ input,
                                    const float *__restrict__ hidden,
                                    const float *__restrict__ grad_output,
                                    const float *__restrict__ fc2_weights,
                                    float *__restrict__ grad_weights,
                                    float *__restrict__ grad_bias,
                                    int input_dim, int hidden_dim,
                                    int output_dim, int batch_size) {
  extern __shared__ float shared_mem[];
  float *delta_shared = shared_mem;

  int j = blockIdx.x; // hidden_dim维度索引
  int tid = threadIdx.x;
  int batch_idx = blockIdx.y; // batch维度索引

  if (j >= hidden_dim || batch_idx >= batch_size)
    return;

  // 计算delta1
  float tanh_val = hidden[batch_idx * hidden_dim + j];
  float tanh_derivative = 1.0f - tanh_val * tanh_val; // tanh导数

  // 计算梯度分量
  float delta1 = 0.0f;
  for (int i = tid; i < output_dim; i += blockDim.x) {
    if (i < output_dim) {
      delta1 += fc2_weights[i * hidden_dim + j] *
                grad_output[batch_idx * output_dim + i];
    }
  }
  delta1 *= tanh_derivative;

  // 存储到共享内存
  delta_shared[tid] = delta1;
  __syncthreads();

  // 归约求和
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      delta_shared[tid] += delta_shared[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    float total_delta = delta_shared[0];
    
    // 更新权重梯度
    for (int k = 0; k < input_dim; k++) {
      float partial_w = input[batch_idx * input_dim + k] * total_delta;
      atomicAdd(&grad_weights[j * input_dim + k], partial_w);
    }
    
    // 更新偏置
    atomicAdd(&grad_bias[j], total_delta);
  }
}

// 共享层反向传播，合并actor和critic的梯度
__global__ void shared_backward_kernel(
    const float *__restrict__ input, 
    const float *__restrict__ grad_actor, 
    const float *__restrict__ grad_critic,
    float *__restrict__ grad_weights, 
    float *__restrict__ grad_bias,
    int input_dim, int hidden_dim, int batch_size) {

  int j = blockIdx.x; // hidden_dim维度索引
  int tid = threadIdx.x;
  int batch_idx = blockIdx.y; // batch维度索引

  if (j >= hidden_dim || batch_idx >= batch_size)
    return;

  // 合并actor和critic的梯度
  float delta = grad_actor[batch_idx * hidden_dim + j] +
                grad_critic[batch_idx * hidden_dim + j];

  // 每个线程处理部分输入维度
  for (int k = tid; k < input_dim; k += blockDim.x) {
    float partial = input[batch_idx * input_dim + k] * delta;
    atomicAdd(&grad_weights[j * input_dim + k], partial);
  }

  // 更新偏置（每个神经元只由一个线程更新）
  if (tid == 0) {
    atomicAdd(&grad_bias[j], delta);
  }
}

// 设备内存池
struct DeviceMemoryPool {
  float *d_input;
  float *d_shared_w, *d_shared_b;
  float *d_actor_fc1_w, *d_actor_fc1_b, *d_actor_fc2_w, *d_actor_fc2_b;
  float *d_actor_head_w, *d_actor_head_b;
  float *d_critic_fc1_w, *d_critic_fc1_b, *d_critic_fc2_w, *d_critic_fc2_b;
  float *d_critic_head_w, *d_critic_head_b;
  float *d_shared_output;
  float *d_actor_hidden, *d_critic_hidden;
  float *d_actor_fc2_output, *d_critic_fc2_output;
  float *d_actor_output, *d_critic_output;

  cudaStream_t actor_stream, critic_stream;
  cublasHandle_t cublas_handle;

  bool initialized = false;

  void initialize(int max_batch, int max_input_dim, int max_hidden_dim,
                  int max_output_dim) {
    if (initialized)
      return;

    CHECK_CUDA(cudaMalloc(&d_input, max_batch * max_input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_shared_w,
                          max_hidden_dim * max_input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_shared_b, max_hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_actor_fc1_w,
                          max_hidden_dim * max_hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_actor_fc1_b, max_hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_actor_fc2_w,
                          max_output_dim * max_hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_actor_fc2_b, max_output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_actor_head_w,
                          max_output_dim * max_output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_actor_head_b, max_output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_fc1_w,
                          max_hidden_dim * max_hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_fc1_b, max_hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_fc2_w,
                          max_output_dim * max_hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_fc2_b, max_output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_head_w,
                          max_output_dim * max_output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_head_b, max_output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_shared_output,
                          max_batch * max_hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_actor_hidden,
                          max_batch * max_hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_hidden,
                          max_batch * max_hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_actor_fc2_output,
                          max_batch * max_output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_fc2_output,
                          max_batch * max_output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_actor_output,
                          max_batch * max_output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_output,
                          max_batch * max_output_dim * sizeof(float)));

    CHECK_CUDA(cudaStreamCreate(&actor_stream));
    CHECK_CUDA(cudaStreamCreate(&critic_stream));
    cublasCreate(&cublas_handle);
    cublasSetStream(cublas_handle, actor_stream); // 为cuBLAS设置流

    initialized = true;
  }

  void destroy() {
    if (!initialized)
      return;

    cudaFree(d_input);
    cudaFree(d_shared_w);
    cudaFree(d_shared_b);
    cudaFree(d_actor_fc1_w);
    cudaFree(d_actor_fc1_b);
    cudaFree(d_actor_fc2_w);
    cudaFree(d_actor_fc2_b);
    cudaFree(d_actor_head_w);
    cudaFree(d_actor_head_b);
    cudaFree(d_critic_fc1_w);
    cudaFree(d_critic_fc1_b);
    cudaFree(d_critic_fc2_w);
    cudaFree(d_critic_fc2_b);
    cudaFree(d_critic_head_w);
    cudaFree(d_critic_head_b);
    cudaFree(d_shared_output);
    cudaFree(d_actor_hidden);
    cudaFree(d_critic_hidden);
    cudaFree(d_actor_fc2_output);
    cudaFree(d_critic_fc2_output);
    cudaFree(d_actor_output);
    cudaFree(d_critic_output);

    cudaStreamDestroy(actor_stream);
    cudaStreamDestroy(critic_stream);
    cublasDestroy(cublas_handle);

    initialized = false;
  }
};

// 全局内存池实例
DeviceMemoryPool g_mem_pool;

extern "C" {
void cuda_init(int max_batch, int max_input_dim, int max_hidden_dim,
               int max_output_dim) {
  g_mem_pool.initialize(max_batch, max_input_dim, max_hidden_dim,
                        max_output_dim);
}

void cuda_cleanup() { g_mem_pool.destroy(); }

void cuda_forward(const float *input, int batch_size, int input_dim,
                  int hidden_dim, int output_dim, const float *shared_w,
                  const float *shared_b, const float *actor_fc1_w,
                  const float *actor_fc1_b, const float *actor_fc2_w,
                  const float *actor_fc2_b, const float *actor_head_w,
                  const float *actor_head_b, const float *critic_fc1_w,
                  const float *critic_fc1_b, const float *critic_fc2_w,
                  const float *critic_fc2_b, const float *critic_head_w,
                  const float *critic_head_b, float *actor_output,
                  float *critic_output, float *shared_output,   
                  float *actor_hidden, float *actor_fc2_output, 
                  float *critic_hidden,
                  float *critic_fc2_output) { 

  float *d_input = g_mem_pool.d_input;
  float *d_shared_w = g_mem_pool.d_shared_w;
  float *d_shared_b = g_mem_pool.d_shared_b;
  float *d_actor_fc1_w = g_mem_pool.d_actor_fc1_w;
  float *d_actor_fc1_b = g_mem_pool.d_actor_fc1_b;
  float *d_actor_fc2_w = g_mem_pool.d_actor_fc2_w;
  float *d_actor_fc2_b = g_mem_pool.d_actor_fc2_b;
  float *d_actor_head_w = g_mem_pool.d_actor_head_w;
  float *d_actor_head_b = g_mem_pool.d_actor_head_b;
  float *d_critic_fc1_w = g_mem_pool.d_critic_fc1_w;
  float *d_critic_fc1_b = g_mem_pool.d_critic_fc1_b;
  float *d_critic_fc2_w = g_mem_pool.d_critic_fc2_w;
  float *d_critic_fc2_b = g_mem_pool.d_critic_fc2_b;
  float *d_critic_head_w = g_mem_pool.d_critic_head_w;
  float *d_critic_head_b = g_mem_pool.d_critic_head_b;
  float *d_shared_output = g_mem_pool.d_shared_output;
  float *d_actor_hidden = g_mem_pool.d_actor_hidden;
  float *d_critic_hidden = g_mem_pool.d_critic_hidden;
  float *d_actor_fc2_output = g_mem_pool.d_actor_fc2_output;
  float *d_critic_fc2_output = g_mem_pool.d_critic_fc2_output;
  float *d_actor_output = g_mem_pool.d_actor_output;
  float *d_critic_output = g_mem_pool.d_critic_output;

  cudaStream_t actor_stream = g_mem_pool.actor_stream;
  cudaStream_t critic_stream = g_mem_pool.critic_stream;

  // 创建同步事件
  cudaEvent_t shared_done;
  CHECK_CUDA(cudaEventCreate(&shared_done));

  // 拷贝数据到设备
  CHECK_CUDA(cudaMemcpyAsync(d_input, input,
                             batch_size * input_dim * sizeof(float),
                             cudaMemcpyHostToDevice, actor_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_shared_w, shared_w,
                             hidden_dim * input_dim * sizeof(float),
                             cudaMemcpyHostToDevice, actor_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_shared_b, shared_b, hidden_dim * sizeof(float),
                             cudaMemcpyHostToDevice, actor_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_actor_fc1_w, actor_fc1_w,
                             hidden_dim * hidden_dim * sizeof(float),
                             cudaMemcpyHostToDevice, actor_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_actor_fc1_b, actor_fc1_b,
                             hidden_dim * sizeof(float), cudaMemcpyHostToDevice,
                             actor_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_actor_fc2_w, actor_fc2_w,
                             output_dim * hidden_dim * sizeof(float),
                             cudaMemcpyHostToDevice, actor_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_actor_fc2_b, actor_fc2_b,
                             output_dim * sizeof(float), cudaMemcpyHostToDevice,
                             actor_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_actor_head_w, actor_head_w,
                             output_dim * output_dim * sizeof(float),
                             cudaMemcpyHostToDevice, actor_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_actor_head_b, actor_head_b,
                             output_dim * sizeof(float), cudaMemcpyHostToDevice,
                             actor_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_critic_fc1_w, critic_fc1_w,
                             hidden_dim * hidden_dim * sizeof(float),
                             cudaMemcpyHostToDevice, critic_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_critic_fc1_b, critic_fc1_b,
                             hidden_dim * sizeof(float), cudaMemcpyHostToDevice,
                             critic_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_critic_fc2_w, critic_fc2_w,
                             output_dim * hidden_dim * sizeof(float),
                             cudaMemcpyHostToDevice, critic_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_critic_fc2_b, critic_fc2_b,
                             output_dim * sizeof(float), cudaMemcpyHostToDevice,
                             critic_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_critic_head_w, critic_head_w,
                             output_dim * output_dim * sizeof(float),
                             cudaMemcpyHostToDevice, critic_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_critic_head_b, critic_head_b,
                             output_dim * sizeof(float), cudaMemcpyHostToDevice,
                             critic_stream));

  // 执行共享层前向传播
  dim3 block0((hidden_dim + TILE_SIZE - 1) / TILE_SIZE, batch_size);
  int shared_mem_size = TILE_SIZE * (TILE_SIZE + 1) * sizeof(float);
  linear_forward_kernel<<<block0, TILE_SIZE, shared_mem_size, actor_stream>>>(
      d_input, d_shared_w, d_shared_b, d_shared_output, input_dim, hidden_dim,
      batch_size);

  // 记录事件确保共享层计算完成
  CHECK_CUDA(cudaEventRecord(shared_done, actor_stream));

  // 等待共享层完成后再开始Actor和Critic分支
  CHECK_CUDA(cudaStreamWaitEvent(actor_stream, shared_done, 0));
  CHECK_CUDA(cudaStreamWaitEvent(critic_stream, shared_done, 0));

  // Actor路径
  dim3 block_a1((hidden_dim + TILE_SIZE - 1) / TILE_SIZE, batch_size);
  fc1_forward_kernel<<<block_a1, TILE_SIZE, shared_mem_size, actor_stream>>>(
      d_shared_output, d_actor_fc1_w, d_actor_fc1_b, d_actor_hidden,
      hidden_dim, hidden_dim, batch_size);

  // 使用cuBLAS优化全连接层
  float alpha = 1.0f;
  float beta = 0.0f;
  cublasSgemm(g_mem_pool.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, output_dim,
              batch_size, hidden_dim, &alpha, d_actor_fc2_w, output_dim,
              d_actor_hidden, hidden_dim, &beta, d_actor_fc2_output,
              output_dim);

  // 添加偏置
  dim3 bias_block(256);
  dim3 bias_grid((batch_size * output_dim + bias_block.x - 1) / bias_block.x);
  add_bias_kernel<<<bias_grid, bias_block, 0, actor_stream>>>(
      d_actor_fc2_output, d_actor_fc2_b, output_dim, batch_size);
  
  // Actor Head层
  dim3 block_actor_head((output_dim + TILE_SIZE - 1) / TILE_SIZE, batch_size);
  linear_forward_kernel<<<block_actor_head, TILE_SIZE, shared_mem_size, actor_stream>>>(
      d_actor_fc2_output, d_actor_head_w, d_actor_head_b, d_actor_output,
      output_dim, output_dim, batch_size);

  // Critic路径
  dim3 block_c1((hidden_dim + TILE_SIZE - 1) / TILE_SIZE, batch_size);
  fc1_forward_kernel<<<block_c1, TILE_SIZE, shared_mem_size, critic_stream>>>(
      d_shared_output, d_critic_fc1_w, d_critic_fc1_b, d_critic_hidden,
      hidden_dim, hidden_dim, batch_size);

  dim3 block_c2((batch_size * output_dim + 255) / 256);
  fc2_forward_kernel<<<block_c2, 256, 0, critic_stream>>>(
      d_critic_hidden, d_critic_fc2_w, d_critic_fc2_b, d_critic_fc2_output,
      hidden_dim, output_dim, batch_size);

  // Critic Head层
  dim3 block_critic_head((output_dim + TILE_SIZE - 1) / TILE_SIZE, batch_size);
  linear_forward_kernel<<<block_critic_head, TILE_SIZE, shared_mem_size, critic_stream>>>(
      d_critic_fc2_output, d_critic_head_w, d_critic_head_b, d_critic_output,
      output_dim, output_dim, batch_size);

  // 等待两个流完成
  CHECK_CUDA(cudaStreamSynchronize(actor_stream));
  CHECK_CUDA(cudaStreamSynchronize(critic_stream));

  // 拷贝所有需要的中间结果回主机
  CHECK_CUDA(cudaMemcpyAsync(shared_output, d_shared_output,
                             batch_size * hidden_dim * sizeof(float),
                             cudaMemcpyDeviceToHost, actor_stream));
  CHECK_CUDA(cudaMemcpyAsync(actor_hidden, d_actor_hidden,
                             batch_size * hidden_dim * sizeof(float),
                             cudaMemcpyDeviceToHost, actor_stream));
  CHECK_CUDA(cudaMemcpyAsync(actor_fc2_output, d_actor_fc2_output,
                             batch_size * output_dim * sizeof(float),
                             cudaMemcpyDeviceToHost, actor_stream));
  CHECK_CUDA(cudaMemcpyAsync(critic_hidden, d_critic_hidden,
                             batch_size * hidden_dim * sizeof(float),
                             cudaMemcpyDeviceToHost, critic_stream));
  CHECK_CUDA(cudaMemcpyAsync(critic_fc2_output, d_critic_fc2_output,
                             batch_size * output_dim * sizeof(float),
                             cudaMemcpyDeviceToHost, critic_stream));

  // 拷贝最终输出回主机
  CHECK_CUDA(cudaMemcpyAsync(actor_output, d_actor_output,
                             batch_size * output_dim * sizeof(float),
                             cudaMemcpyDeviceToHost, actor_stream));
  CHECK_CUDA(cudaMemcpyAsync(critic_output, d_critic_output,
                             batch_size * output_dim * sizeof(float),
                             cudaMemcpyDeviceToHost, critic_stream));

  // 等待拷贝完成
  CHECK_CUDA(cudaStreamSynchronize(actor_stream));
  CHECK_CUDA(cudaStreamSynchronize(critic_stream));

  // 销毁事件
  CHECK_CUDA(cudaEventDestroy(shared_done));
}

// 改进的反向传播函数，支持并行计算
void cuda_backward(const float *input, const float *shared_output,
                   const float *actor_hidden, const float *critic_hidden,
                   const float *actor_fc2_output, const float *actor_output,
                   const float *critic_fc2_output, const float *critic_output,
                   const float *grad_actor_output,
                   const float *grad_critic_output, int batch_size,
                   int input_dim, int hidden_dim, int output_dim,
                   const float *actor_fc2_w, const float *critic_fc2_w,
                   const float *actor_head_w, const float *critic_head_w,
                   float *grad_shared_w, float *grad_shared_b,
                   float *grad_actor_fc1_w, float *grad_actor_fc1_b,
                   float *grad_actor_fc2_w, float *grad_actor_fc2_b,
                   float *grad_actor_head_w, float *grad_actor_head_b,
                   float *grad_critic_fc1_w, float *grad_critic_fc1_b,
                   float *grad_critic_fc2_w, float *grad_critic_fc2_b,
                   float *grad_critic_head_w, float *grad_critic_head_b) {
  // 使用内存池中的设备指针
  float *d_input = g_mem_pool.d_input;
  float *d_shared_output = g_mem_pool.d_shared_output;
  float *d_actor_hidden = g_mem_pool.d_actor_hidden;
  float *d_critic_hidden = g_mem_pool.d_critic_hidden;
  float *d_actor_fc2_output = g_mem_pool.d_actor_fc2_output;
  float *d_critic_fc2_output = g_mem_pool.d_critic_fc2_output;
  float *d_actor_output = g_mem_pool.d_actor_output;
  float *d_critic_output = g_mem_pool.d_critic_output;
  float *d_grad_actor_output = g_mem_pool.d_actor_output;   
  float *d_grad_critic_output = g_mem_pool.d_critic_output; 
  float *d_actor_fc2_w = g_mem_pool.d_actor_fc2_w;
  float *d_critic_fc2_w = g_mem_pool.d_critic_fc2_w;
  float *d_actor_head_w = g_mem_pool.d_actor_head_w;
  float *d_critic_head_w = g_mem_pool.d_critic_head_w;
  float *d_grad_shared_w = g_mem_pool.d_shared_w;                   
  float *d_grad_shared_b = g_mem_pool.d_shared_b;                   
  float *d_grad_actor_fc1_w = g_mem_pool.d_actor_fc1_w;             
  float *d_grad_actor_fc1_b = g_mem_pool.d_actor_fc1_b;             
  float *d_grad_actor_fc2_w = g_mem_pool.d_actor_fc2_w;             
  float *d_grad_actor_fc2_b = g_mem_pool.d_actor_fc2_b;             
  float *d_grad_actor_head_w = g_mem_pool.d_actor_head_w;           
  float *d_grad_actor_head_b = g_mem_pool.d_actor_head_b;           
  float *d_grad_critic_fc1_w = g_mem_pool.d_critic_fc1_w;           
  float *d_grad_critic_fc1_b = g_mem_pool.d_critic_fc1_b;           
  float *d_grad_critic_fc2_w = g_mem_pool.d_critic_fc2_w;           
  float *d_grad_critic_fc2_b = g_mem_pool.d_critic_fc2_b;           
  float *d_grad_critic_head_w = g_mem_pool.d_critic_head_w;         
  float *d_grad_critic_head_b = g_mem_pool.d_critic_head_b;         
  float *d_grad_actor_hidden = g_mem_pool.d_actor_hidden;           
  float *d_grad_critic_hidden = g_mem_pool.d_critic_hidden;         
  float *d_grad_actor_fc2_output = g_mem_pool.d_actor_fc2_output;   
  float *d_grad_critic_fc2_output = g_mem_pool.d_critic_fc2_output; 

  cudaStream_t actor_stream = g_mem_pool.actor_stream;
  cudaStream_t critic_stream = g_mem_pool.critic_stream;

  // 创建同步事件
  cudaEvent_t actor_head_done, critic_head_done;
  CHECK_CUDA(cudaEventCreate(&actor_head_done));
  CHECK_CUDA(cudaEventCreate(&critic_head_done));

  // 拷贝数据到设备
  CHECK_CUDA(cudaMemcpyAsync(d_input, input,
                             batch_size * input_dim * sizeof(float),
                             cudaMemcpyHostToDevice, actor_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_shared_output, shared_output,
                             batch_size * hidden_dim * sizeof(float),
                             cudaMemcpyHostToDevice, actor_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_actor_hidden, actor_hidden,
                             batch_size * hidden_dim * sizeof(float),
                             cudaMemcpyHostToDevice, actor_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_critic_hidden, critic_hidden,
                             batch_size * hidden_dim * sizeof(float),
                             cudaMemcpyHostToDevice, critic_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_actor_fc2_output, actor_fc2_output,
                             batch_size * output_dim * sizeof(float),
                             cudaMemcpyHostToDevice, actor_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_critic_fc2_output, critic_fc2_output,
                             batch_size * output_dim * sizeof(float),
                             cudaMemcpyHostToDevice, critic_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_actor_output, actor_output,
                             batch_size * output_dim * sizeof(float),
                             cudaMemcpyHostToDevice, actor_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_critic_output, critic_output,
                             batch_size * output_dim * sizeof(float),
                             cudaMemcpyHostToDevice, critic_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_grad_actor_output, grad_actor_output,
                             batch_size * output_dim * sizeof(float),
                             cudaMemcpyHostToDevice, actor_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_grad_critic_output, grad_critic_output,
                             batch_size * output_dim * sizeof(float),
                             cudaMemcpyHostToDevice, critic_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_actor_fc2_w, actor_fc2_w,
                             output_dim * hidden_dim * sizeof(float),
                             cudaMemcpyHostToDevice, actor_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_critic_fc2_w, critic_fc2_w,
                             output_dim * hidden_dim * sizeof(float),
                             cudaMemcpyHostToDevice, critic_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_actor_head_w, actor_head_w,
                             output_dim * output_dim * sizeof(float),
                             cudaMemcpyHostToDevice, actor_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_critic_head_w, critic_head_w,
                             output_dim * output_dim * sizeof(float),
                             cudaMemcpyHostToDevice, critic_stream));

  // 初始化梯度为0
  CHECK_CUDA(cudaMemsetAsync(d_grad_shared_w, 0,
                             hidden_dim * input_dim * sizeof(float),
                             actor_stream));
  CHECK_CUDA(cudaMemsetAsync(d_grad_shared_b, 0, hidden_dim * sizeof(float),
                             actor_stream));
  CHECK_CUDA(cudaMemsetAsync(d_grad_actor_fc1_w, 0,
                             hidden_dim * hidden_dim * sizeof(float),
                             actor_stream));
  CHECK_CUDA(cudaMemsetAsync(d_grad_actor_fc1_b, 0, hidden_dim * sizeof(float),
                             actor_stream));
  CHECK_CUDA(cudaMemsetAsync(d_grad_actor_fc2_w, 0,
                             output_dim * hidden_dim * sizeof(float),
                             actor_stream));
  CHECK_CUDA(cudaMemsetAsync(d_grad_actor_fc2_b, 0, output_dim * sizeof(float),
                             actor_stream));
  CHECK_CUDA(cudaMemsetAsync(d_grad_actor_head_w, 0,
                             output_dim * output_dim * sizeof(float),
                             actor_stream));
  CHECK_CUDA(cudaMemsetAsync(d_grad_actor_head_b, 0, output_dim * sizeof(float),
                             actor_stream));
  CHECK_CUDA(cudaMemsetAsync(d_grad_critic_fc1_w, 0,
                             hidden_dim * hidden_dim * sizeof(float),
                             critic_stream));
  CHECK_CUDA(cudaMemsetAsync(d_grad_critic_fc1_b, 0, hidden_dim * sizeof(float),
                             critic_stream));
  CHECK_CUDA(cudaMemsetAsync(d_grad_critic_fc2_w, 0,
                             output_dim * hidden_dim * sizeof(float),
                             critic_stream));
  CHECK_CUDA(cudaMemsetAsync(d_grad_critic_fc2_b, 0, output_dim * sizeof(float),
                             critic_stream));
  CHECK_CUDA(cudaMemsetAsync(d_grad_critic_head_w, 0,
                             output_dim * output_dim * sizeof(float),
                             critic_stream));
  CHECK_CUDA(cudaMemsetAsync(d_grad_critic_head_b, 0,
                             output_dim * sizeof(float), critic_stream));

  // 并行执行Actor和Critic的反向传播
  // Actor路径反向传播
  dim3 head_block(256);
  dim3 head_grid((batch_size * output_dim + head_block.x - 1) / head_block.x);

  // 计算Actor Head层梯度
  dim3 head_grid_dim(output_dim, batch_size);
  linear_backward_kernel<<<head_grid_dim, TILE_SIZE,
                           TILE_SIZE * (TILE_SIZE + 1) * sizeof(float),
                           actor_stream>>>(
      d_actor_fc2_output, d_grad_actor_output,
      d_grad_actor_head_w, d_grad_actor_head_b, output_dim, output_dim,
      batch_size);

  // 记录事件
  CHECK_CUDA(cudaEventRecord(actor_head_done, actor_stream));

  // 使用cuBLAS计算梯度
  float alpha = 1.0f, beta = 0.0f;
  cublasSgemm(g_mem_pool.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, hidden_dim,
              output_dim, batch_size, &alpha, d_actor_hidden, hidden_dim,
              d_grad_actor_output, output_dim, &beta, d_grad_actor_fc2_w,
              hidden_dim);

  // Critic路径反向传播
  dim3 fc2_block(256);
  dim3 fc2_grid((batch_size * output_dim + fc2_block.x - 1) / fc2_block.x);

  // 计算Critic Head层梯度
  linear_backward_kernel<<<head_grid_dim, TILE_SIZE,
                           TILE_SIZE * (TILE_SIZE + 1) * sizeof(float),
                           critic_stream>>>(
      d_critic_fc2_output, d_grad_critic_output,
      d_grad_critic_head_w, d_grad_critic_head_b, output_dim, output_dim,
      batch_size);

  // 记录事件
  CHECK_CUDA(cudaEventRecord(critic_head_done, critic_stream));

  // 使用cuBLAS计算梯度
  cublasSgemm(g_mem_pool.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, hidden_dim,
              output_dim, batch_size, &alpha, d_critic_hidden, hidden_dim,
              d_grad_critic_output, output_dim, &beta, d_grad_critic_fc2_w,
              hidden_dim);

  // 等待Head层梯度计算完成
  CHECK_CUDA(cudaStreamWaitEvent(actor_stream, actor_head_done, 0));
  CHECK_CUDA(cudaStreamWaitEvent(critic_stream, critic_head_done, 0));

  // Actor FC1层反向传播
  dim3 fc1_grid(hidden_dim, batch_size);
  // 使用固定大小的共享内存（256个元素）
  size_t fc1_shared_mem = 256 * sizeof(float);
  fc1_backward_kernel<<<fc1_grid, 256, fc1_shared_mem, actor_stream>>>(
      d_shared_output, d_actor_hidden, d_grad_actor_output, d_actor_fc2_w,
      d_grad_actor_fc1_w, d_grad_actor_fc1_b, hidden_dim, hidden_dim,
      output_dim, batch_size);

  // Critic FC1层反向传播
  fc1_backward_kernel<<<fc1_grid, 256, fc1_shared_mem, critic_stream>>>(
      d_shared_output, d_critic_hidden, d_grad_critic_output, d_critic_fc2_w,
      d_grad_critic_fc1_w, d_grad_critic_fc1_b, hidden_dim, hidden_dim,
      output_dim, batch_size);

  // 等待两个流完成
  CHECK_CUDA(cudaStreamSynchronize(actor_stream));
  CHECK_CUDA(cudaStreamSynchronize(critic_stream));
  
  // 共享层反向传播
  shared_backward_kernel<<<fc1_grid, 256, 0, actor_stream>>>(
      d_input, 
      d_grad_actor_hidden, 
      d_grad_critic_hidden,
      d_grad_shared_w, 
      d_grad_shared_b, 
      input_dim, hidden_dim, batch_size);

  // 等待共享层梯度计算完成
  CHECK_CUDA(cudaStreamSynchronize(actor_stream));

  // 拷贝梯度回主机
  CHECK_CUDA(cudaMemcpyAsync(grad_shared_w, d_grad_shared_w,
                             hidden_dim * input_dim * sizeof(float),
                             cudaMemcpyDeviceToHost, actor_stream));
  CHECK_CUDA(cudaMemcpyAsync(grad_shared_b, d_grad_shared_b,
                             hidden_dim * sizeof(float), cudaMemcpyDeviceToHost,
                             actor_stream));
  CHECK_CUDA(cudaMemcpyAsync(grad_actor_fc1_w, d_grad_actor_fc1_w,
                             hidden_dim * hidden_dim * sizeof(float),
                             cudaMemcpyDeviceToHost, actor_stream));
  CHECK_CUDA(cudaMemcpyAsync(grad_actor_fc1_b, d_grad_actor_fc1_b,
                             hidden_dim * sizeof(float), cudaMemcpyDeviceToHost,
                             actor_stream));
  CHECK_CUDA(cudaMemcpyAsync(grad_actor_fc2_w, d_grad_actor_fc2_w,
                             output_dim * hidden_dim * sizeof(float),
                             cudaMemcpyDeviceToHost, actor_stream));
  CHECK_CUDA(cudaMemcpyAsync(grad_actor_fc2_b, d_grad_actor_fc2_b,
                             output_dim * sizeof(float), cudaMemcpyDeviceToHost,
                             actor_stream));
  CHECK_CUDA(cudaMemcpyAsync(grad_actor_head_w, d_grad_actor_head_w,
                             output_dim * output_dim * sizeof(float),
                             cudaMemcpyDeviceToHost, actor_stream));
  CHECK_CUDA(cudaMemcpyAsync(grad_actor_head_b, d_grad_actor_head_b,
                             output_dim * sizeof(float), cudaMemcpyDeviceToHost,
                             actor_stream));
  CHECK_CUDA(cudaMemcpyAsync(grad_critic_fc1_w, d_grad_critic_fc1_w,
                             hidden_dim * hidden_dim * sizeof(float),
                             cudaMemcpyDeviceToHost, critic_stream));
  CHECK_CUDA(cudaMemcpyAsync(grad_critic_fc1_b, d_grad_critic_fc1_b,
                             hidden_dim * sizeof(float), cudaMemcpyDeviceToHost,
                             critic_stream));
  CHECK_CUDA(cudaMemcpyAsync(grad_critic_fc2_w, d_grad_critic_fc2_w,
                             output_dim * hidden_dim * sizeof(float),
                             cudaMemcpyDeviceToHost, critic_stream));
  CHECK_CUDA(cudaMemcpyAsync(grad_critic_fc2_b, d_grad_critic_fc2_b,
                             output_dim * sizeof(float), cudaMemcpyDeviceToHost,
                             critic_stream));
  CHECK_CUDA(cudaMemcpyAsync(grad_critic_head_w, d_grad_critic_head_w,
                             output_dim * output_dim * sizeof(float),
                             cudaMemcpyDeviceToHost, critic_stream));
  CHECK_CUDA(cudaMemcpyAsync(grad_critic_head_b, d_grad_critic_head_b,
                             output_dim * sizeof(float), cudaMemcpyDeviceToHost,
                             critic_stream));

  // 等待拷贝完成
  CHECK_CUDA(cudaStreamSynchronize(actor_stream));
  CHECK_CUDA(cudaStreamSynchronize(critic_stream));

  // 销毁事件
  CHECK_CUDA(cudaEventDestroy(actor_head_done));
  CHECK_CUDA(cudaEventDestroy(critic_head_done));
}
} // extern "C"