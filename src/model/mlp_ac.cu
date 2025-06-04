#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>

#define WARP_SIZE 32

// 使用CUDA流实现分流并行
__global__ void linear_forward_kernel(const float *input, const float *weights,
                                      const float *bias, float *hidden,
                                      int input_dim, int hidden_dim,
                                      int batch_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size * hidden_dim)
    return;

  int batch_idx = idx / hidden_dim;
  int neuron_idx = idx % hidden_dim;

  float sum = 0.0f;
  for (int i = 0; i < input_dim; ++i) {
    sum +=
        input[batch_idx * input_dim + i] * weights[neuron_idx * input_dim + i];
  }
  sum += bias[neuron_idx];
  hidden[idx] = sum;
}

__global__ void linear_backward_kernel(const float *__restrict__ input,
                                       const float *__restrict__ hidden,
                                       const float *__restrict__ grad_output,
                                       float *__restrict__ grad_weights,
                                       float *__restrict__ grad_bias,
                                       int input_dim, int hidden_dim,
                                       int batch_size) {
  // 三维线程布局: (i, j, b)
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int b = blockIdx.z * blockDim.z + threadIdx.z;

  // 计算权重梯度
  if (i < hidden_dim && j < input_dim && b < batch_size) {
    float partial = input[b * input_dim + j] * grad_output[b * hidden_dim + i];
    atomicAdd(&grad_weights[i * input_dim + j], partial);
  }

  // 计算偏置梯度
  if (i < hidden_dim && b < batch_size && j == 0) { // 使用j==0的线程处理偏置
    atomicAdd(&grad_bias[i], grad_output[b * hidden_dim + i]);
  }
}

// actor/critic & head
// (fc+tanh)1 forward kernel
__global__ void fc1_forward_kernel(const float *input, const float *weights,
                                   const float *bias, float *hidden,
                                   int input_dim, int hidden_dim,
                                   int batch_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size * hidden_dim)
    return;

  int batch_idx = idx / hidden_dim;
  int neuron_idx = idx % hidden_dim;

  float sum = 0.0f;
  for (int i = 0; i < input_dim; ++i) {
    sum +=
        input[batch_idx * input_dim + i] * weights[neuron_idx * input_dim + i];
  }
  sum += bias[neuron_idx];
  hidden[idx] = tanhf(sum); // tanH
}

// (fc+tanh)2 forward kernel
__global__ void fc2_forward_kernel(const float *hidden, const float *weights,
                                   const float *bias, float *output,
                                   int hidden_dim, int output_dim,
                                   int batch_size) {
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
  // 三维线程布局: (i, j, b)
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int b = blockIdx.z * blockDim.z + threadIdx.z;

  // 计算权重梯度
  if (i < output_dim && j < hidden_dim && b < batch_size) {
    float partial =
        hidden[b * hidden_dim + j] * grad_output[b * output_dim + i];
    atomicAdd(&grad_weights[i * hidden_dim + j], partial);
  }

  // 计算偏置梯度
  if (i < output_dim && b < batch_size && j == 0) { // 使用j==0的线程处理偏置
    atomicAdd(&grad_bias[i], grad_output[b * output_dim + i]);
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
  // 三维线程布局: (j, k, b)
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int b = blockIdx.z * blockDim.z + threadIdx.z;

  if (j >= hidden_dim || k >= input_dim || b >= batch_size)
    return;

  __shared__ float shared_delta[256]; // 共享内存用于中间计算

  // 计算delta1
  float tanh_val = hidden[b * hidden_dim + j];
  float tanh_derivative = 1.0f - tanh_val * tanh_val; // tanh导数

  float delta1 = 0.0f;
  for (int i = 0; i < output_dim; i += blockDim.x) {
    int idx = i + threadIdx.x;
    if (idx < output_dim) {
      delta1 +=
          fc2_weights[idx * hidden_dim + j] * grad_output[b * output_dim + idx];
    }
  }
  delta1 *= tanh_derivative; // 乘以tanh导数

  shared_delta[threadIdx.x] = delta1;
  __syncthreads();

  // 归约求和
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_delta[threadIdx.x] += shared_delta[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    delta1 = shared_delta[0];
    float partial_w = input[b * input_dim + k] * delta1;
    atomicAdd(&grad_weights[j * input_dim + k], partial_w);

    if (k == 0) { // 使用k==0的线程处理偏置
      atomicAdd(&grad_bias[j], delta1);
    }
  }
}

// 共享层反向传播，合并actor和critic的梯度
__global__ void shared_backward_kernel(
    const float *__restrict__ input, const float *__restrict__ hidden,
    const float *__restrict__ grad_actor, const float *__restrict__ grad_critic,
    float *__restrict__ grad_weights, float *__restrict__ grad_bias,
    int input_dim, int hidden_dim, int batch_size) {

  int j = blockIdx.x * blockDim.x + threadIdx.x; // hidden_dim维度索引
  int k = blockIdx.y * blockDim.y + threadIdx.y; // input_dim维度索引
  int b = blockIdx.z * blockDim.z + threadIdx.z; // batch维度索引

  if (j >= hidden_dim || k >= input_dim || b >= batch_size)
    return;

  // 合并actor和critic的梯度
  float delta =
      grad_actor[b * hidden_dim + j] + grad_critic[b * hidden_dim + j];

  // 计算权重梯度
  float partial_w = input[b * input_dim + k] * delta;
  atomicAdd(&grad_weights[j * input_dim + k], partial_w);

  // 计算偏置梯度（每个神经元只由一个线程更新）
  if (k == 0) {
    atomicAdd(&grad_bias[j], delta);
  }
}

extern "C" {
void cuda_forward(const float *input, int batch_size, int input_dim,
                  int hidden_dim, int output_dim, const float *shared_w,
                  const float *shared_b, const float *actor_fc1_w,
                  const float *actor_fc1_b, const float *actor_fc2_w,
                  const float *actor_fc2_b, const float *actor_head_w,
                  const float *actor_head_b, const float *critic_fc1_w,
                  const float *critic_fc1_b, const float *critic_fc2_w,
                  const float *critic_fc2_b, const float *critic_head_w,
                  const float *critic_head_b, float *actor_output,
                  float *critic_output) {
  float *d_input, *d_shared_w, *d_shared_b;
  float *d_actor_fc1_w, *d_actor_fc1_b, *d_actor_fc2_w, *d_actor_fc2_b;
  float *d_actor_head_w, *d_actor_head_b; // Actor Head层参数
  float *d_critic_fc1_w, *d_critic_fc1_b, *d_critic_fc2_w, *d_critic_fc2_b;
  float *d_critic_head_w, *d_critic_head_b; // Critic Head层参数
  float *d_shared_output, *d_actor_hidden, *d_critic_hidden;
  float *d_actor_fc2_output, *d_critic_fc2_output;
  float *d_actor_output, *d_critic_output;

  // 创建CUDA流用于并行计算
  cudaStream_t actor_stream, critic_stream;
  cudaStreamCreate(&actor_stream);
  cudaStreamCreate(&critic_stream);

  // 分配设备内存
  cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
  cudaMalloc(&d_shared_w, hidden_dim * input_dim * sizeof(float));
  cudaMalloc(&d_shared_b, hidden_dim * sizeof(float));
  cudaMalloc(&d_actor_fc1_w, hidden_dim * hidden_dim * sizeof(float));
  cudaMalloc(&d_actor_fc1_b, hidden_dim * sizeof(float));
  cudaMalloc(&d_actor_fc2_w, output_dim * hidden_dim * sizeof(float));
  cudaMalloc(&d_actor_fc2_b, output_dim * sizeof(float));
  cudaMalloc(&d_actor_head_w,
             output_dim * output_dim * sizeof(float));     // Actor Head权重
  cudaMalloc(&d_actor_head_b, output_dim * sizeof(float)); // Actor Head偏置

  cudaMalloc(&d_critic_fc1_w, hidden_dim * hidden_dim * sizeof(float));
  cudaMalloc(&d_critic_fc1_b, hidden_dim * sizeof(float));
  cudaMalloc(&d_critic_fc2_w, output_dim * hidden_dim * sizeof(float));
  cudaMalloc(&d_critic_fc2_b, output_dim * sizeof(float));
  cudaMalloc(&d_critic_head_w,
             output_dim * output_dim * sizeof(float));      // Critic Head权重
  cudaMalloc(&d_critic_head_b, output_dim * sizeof(float)); // Critic Head偏置

  // 共享层输出只需分配一次
  cudaMalloc(&d_shared_output, batch_size * hidden_dim * sizeof(float));

  // Actor和Critic的中间层
  cudaMalloc(&d_actor_hidden, batch_size * hidden_dim * sizeof(float));
  cudaMalloc(&d_critic_hidden, batch_size * hidden_dim * sizeof(float));

  // fc2输出
  cudaMalloc(&d_actor_fc2_output, batch_size * output_dim * sizeof(float));
  cudaMalloc(&d_critic_fc2_output, batch_size * output_dim * sizeof(float));

  // 输出层
  cudaMalloc(&d_actor_output, batch_size * output_dim * sizeof(float));
  cudaMalloc(&d_critic_output, batch_size * output_dim * sizeof(float));

  // 拷贝数据到设备
  cudaMemcpy(d_input, input, batch_size * input_dim * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_shared_w, shared_w, hidden_dim * input_dim * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_shared_b, shared_b, hidden_dim * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_actor_fc1_w, actor_fc1_w,
             hidden_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_actor_fc1_b, actor_fc1_b, hidden_dim * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_actor_fc2_w, actor_fc2_w,
             output_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_actor_fc2_b, actor_fc2_b, output_dim * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_actor_head_w, actor_head_w,
             output_dim * output_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_actor_head_b, actor_head_b, output_dim * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_critic_fc1_w, critic_fc1_w,
             hidden_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_critic_fc1_b, critic_fc1_b, hidden_dim * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_critic_fc2_w, critic_fc2_w,
             output_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_critic_fc2_b, critic_fc2_b, output_dim * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_critic_head_w, critic_head_w,
             output_dim * output_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_critic_head_b, critic_head_b, output_dim * sizeof(float),
             cudaMemcpyHostToDevice);

  // 执行共享层前向传播（必须先执行）
  int threads = 256;
  dim3 block0((batch_size * hidden_dim + threads - 1) / threads);
  linear_forward_kernel<<<block0, threads>>>(d_input, d_shared_w, d_shared_b,
                                             d_shared_output, input_dim,
                                             hidden_dim, batch_size);

  // 同步确保共享层计算完成
  cudaDeviceSynchronize();

  // 并行执行Actor和Critic的前向传播
  // Actor路径
  dim3 block_a1((batch_size * hidden_dim + threads - 1) / threads);
  fc1_forward_kernel<<<block_a1, threads, 0, actor_stream>>>(
      d_shared_output, d_actor_fc1_w, d_actor_fc1_b, d_actor_hidden, hidden_dim,
      hidden_dim, batch_size);

  dim3 block_a2((batch_size * output_dim + threads - 1) / threads);
  fc2_forward_kernel<<<block_a2, threads, 0, actor_stream>>>(
      d_actor_hidden, d_actor_fc2_w, d_actor_fc2_b, d_actor_fc2_output,
      hidden_dim, output_dim, batch_size);

  // Actor Head层
  dim3 block_actor_head((batch_size * output_dim + threads - 1) / threads);
  linear_forward_kernel<<<block_actor_head, threads, 0, actor_stream>>>(
      d_actor_fc2_output, d_actor_head_w, d_actor_head_b, d_actor_output,
      output_dim, output_dim, batch_size);

  // Critic路径
  dim3 block_c1((batch_size * hidden_dim + threads - 1) / threads);
  fc1_forward_kernel<<<block_c1, threads, 0, critic_stream>>>(
      d_shared_output, d_critic_fc1_w, d_critic_fc1_b, d_critic_hidden,
      hidden_dim, hidden_dim, batch_size);

  dim3 block_c2((batch_size * output_dim + threads - 1) / threads);
  fc2_forward_kernel<<<block_c2, threads, 0, critic_stream>>>(
      d_critic_hidden, d_critic_fc2_w, d_critic_fc2_b, d_critic_fc2_output,
      hidden_dim, output_dim, batch_size);

  // Critic Head层
  dim3 block_critic_head((batch_size * output_dim + threads - 1) / threads);
  linear_forward_kernel<<<block_critic_head, threads, 0, critic_stream>>>(
      d_critic_fc2_output, d_critic_head_w, d_critic_head_b, d_critic_output,
      output_dim, output_dim, batch_size);

  // 等待两个流完成
  cudaStreamSynchronize(actor_stream);
  cudaStreamSynchronize(critic_stream);

  // 拷贝结果回主机
  cudaMemcpy(actor_output, d_actor_output,
             batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(critic_output, d_critic_output,
             batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost);

  // 释放设备内存
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

  // 销毁CUDA流
  cudaStreamDestroy(actor_stream);
  cudaStreamDestroy(critic_stream);
}

// 改进的反向传播函数，支持并行计算
void cuda_backward(const float *input, const float *shared_output,
                   const float *actor_hidden, const float *critic_hidden,
                   const float *actor_fc2_output,
                   const float *critic_fc2_output, const float *actor_output,
                   const float *critic_output, const float *grad_actor_output,
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
  // 设备内存指针
  float *d_input, *d_shared_output, *d_actor_hidden, *d_critic_hidden;
  float *d_actor_fc2_output, *d_critic_fc2_output;
  float *d_actor_output, *d_critic_output;
  float *d_grad_actor_output, *d_grad_critic_output;
  float *d_actor_fc2_w, *d_critic_fc2_w, *d_actor_head_w, *d_critic_head_w;
  float *d_grad_shared_w, *d_grad_shared_b;
  float *d_grad_actor_fc1_w, *d_grad_actor_fc1_b, *d_grad_actor_fc2_w,
      *d_grad_actor_fc2_b;
  float *d_grad_actor_head_w, *d_grad_actor_head_b;
  float *d_grad_critic_fc1_w, *d_grad_critic_fc1_b, *d_grad_critic_fc2_w,
      *d_grad_critic_fc2_b;
  float *d_grad_critic_head_w, *d_grad_critic_head_b;
  float *d_grad_actor_hidden, *d_grad_critic_hidden;
  float *d_grad_actor_fc2_output, *d_grad_critic_fc2_output;

  // 创建CUDA流用于并行计算
  cudaStream_t actor_stream, critic_stream;
  cudaStreamCreate(&actor_stream);
  cudaStreamCreate(&critic_stream);

  // 分配设备内存
  cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
  cudaMalloc(&d_shared_output, batch_size * hidden_dim * sizeof(float));
  cudaMalloc(&d_actor_hidden, batch_size * hidden_dim * sizeof(float));
  cudaMalloc(&d_critic_hidden, batch_size * hidden_dim * sizeof(float));
  cudaMalloc(&d_actor_fc2_output, batch_size * output_dim * sizeof(float));
  cudaMalloc(&d_critic_fc2_output, batch_size * output_dim * sizeof(float));
  cudaMalloc(&d_actor_output, batch_size * output_dim * sizeof(float));
  cudaMalloc(&d_critic_output, batch_size * output_dim * sizeof(float));
  cudaMalloc(&d_grad_actor_output, batch_size * output_dim * sizeof(float));
  cudaMalloc(&d_grad_critic_output, batch_size * output_dim * sizeof(float));

  cudaMalloc(&d_actor_fc2_w, output_dim * hidden_dim * sizeof(float));
  cudaMalloc(&d_critic_fc2_w, output_dim * hidden_dim * sizeof(float));
  cudaMalloc(&d_actor_head_w,
             output_dim * output_dim * sizeof(float)); // Actor Head权重
  cudaMalloc(&d_critic_head_w,
             output_dim * output_dim * sizeof(float)); // Critic Head权重

  cudaMalloc(&d_grad_shared_w, hidden_dim * input_dim * sizeof(float));
  cudaMalloc(&d_grad_shared_b, hidden_dim * sizeof(float));

  cudaMalloc(&d_grad_actor_fc1_w, hidden_dim * hidden_dim * sizeof(float));
  cudaMalloc(&d_grad_actor_fc1_b, hidden_dim * sizeof(float));
  cudaMalloc(&d_grad_actor_fc2_w, output_dim * hidden_dim * sizeof(float));
  cudaMalloc(&d_grad_actor_fc2_b, output_dim * sizeof(float));
  cudaMalloc(&d_grad_actor_head_w,
             output_dim * output_dim * sizeof(float)); // Actor Head梯度
  cudaMalloc(&d_grad_actor_head_b,
             output_dim * sizeof(float)); // Actor Head梯度

  cudaMalloc(&d_grad_critic_fc1_w, hidden_dim * hidden_dim * sizeof(float));
  cudaMalloc(&d_grad_critic_fc1_b, hidden_dim * sizeof(float));
  cudaMalloc(&d_grad_critic_fc2_w, output_dim * hidden_dim * sizeof(float));
  cudaMalloc(&d_grad_critic_fc2_b, output_dim * sizeof(float));
  cudaMalloc(&d_grad_critic_head_w,
             output_dim * output_dim * sizeof(float)); // Critic Head梯度
  cudaMalloc(&d_grad_critic_head_b,
             output_dim * sizeof(float)); // Critic Head梯度

  cudaMalloc(&d_grad_actor_hidden, batch_size * hidden_dim * sizeof(float));
  cudaMalloc(&d_grad_critic_hidden, batch_size * hidden_dim * sizeof(float));
  cudaMalloc(&d_grad_actor_fc2_output, batch_size * output_dim * sizeof(float));
  cudaMalloc(&d_grad_critic_fc2_output,
             batch_size * output_dim * sizeof(float));

  // 拷贝数据到设备
  cudaMemcpy(d_input, input, batch_size * input_dim * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_shared_output, shared_output,
             batch_size * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_actor_hidden, actor_hidden,
             batch_size * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_critic_hidden, critic_hidden,
             batch_size * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_actor_fc2_output, actor_fc2_output,
             batch_size * output_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_critic_fc2_output, critic_fc2_output,
             batch_size * output_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_actor_output, actor_output,
             batch_size * output_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_critic_output, critic_output,
             batch_size * output_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_grad_actor_output, grad_actor_output,
             batch_size * output_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_grad_critic_output, grad_critic_output,
             batch_size * output_dim * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_actor_fc2_w, actor_fc2_w,
             output_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_critic_fc2_w, critic_fc2_w,
             output_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_actor_head_w, actor_head_w,
             output_dim * output_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_critic_head_w, critic_head_w,
             output_dim * output_dim * sizeof(float), cudaMemcpyHostToDevice);

  // 初始化梯度为0
  cudaMemset(d_grad_shared_w, 0, hidden_dim * input_dim * sizeof(float));
  cudaMemset(d_grad_shared_b, 0, hidden_dim * sizeof(float));

  cudaMemset(d_grad_actor_fc1_w, 0, hidden_dim * hidden_dim * sizeof(float));
  cudaMemset(d_grad_actor_fc1_b, 0, hidden_dim * sizeof(float));
  cudaMemset(d_grad_actor_fc2_w, 0, output_dim * hidden_dim * sizeof(float));
  cudaMemset(d_grad_actor_fc2_b, 0, output_dim * sizeof(float));
  cudaMemset(d_grad_actor_head_w, 0,
             output_dim * output_dim * sizeof(float)); // 初始化Actor Head梯度
  cudaMemset(d_grad_actor_head_b, 0,
             output_dim * sizeof(float)); // 初始化Actor Head梯度

  cudaMemset(d_grad_critic_fc1_w, 0, hidden_dim * hidden_dim * sizeof(float));
  cudaMemset(d_grad_critic_fc1_b, 0, hidden_dim * sizeof(float));
  cudaMemset(d_grad_critic_fc2_w, 0, output_dim * hidden_dim * sizeof(float));
  cudaMemset(d_grad_critic_fc2_b, 0, output_dim * sizeof(float));
  cudaMemset(d_grad_critic_head_w, 0,
             output_dim * output_dim * sizeof(float)); // 初始化Critic Head梯度
  cudaMemset(d_grad_critic_head_b, 0,
             output_dim * sizeof(float)); // 初始化Critic Head梯度

  // 并行执行Actor和Critic的反向传播
  // Actor路径反向传播
  dim3 head_block(8, 8, 4); // 每个block 256 threads
  dim3 head_grid((output_dim + head_block.x - 1) / head_block.x,
                 (output_dim + head_block.y - 1) / head_block.y,
                 (batch_size + head_block.z - 1) / head_block.z);

  // 计算Actor Head层梯度
  linear_backward_kernel<<<head_grid, head_block, 0, actor_stream>>>(
      d_actor_fc2_output, d_actor_output, d_grad_actor_output,
      d_grad_actor_head_w, d_grad_actor_head_b, output_dim, output_dim,
      batch_size);

  dim3 fc2_block(8, 8, 4); // 每个block 256 threads
  dim3 fc2_grid((output_dim + fc2_block.x - 1) / fc2_block.x,
                (hidden_dim + fc2_block.y - 1) / fc2_block.y,
                (batch_size + fc2_block.z - 1) / fc2_block.z);

  // 使用fc2输出的梯度计算fc2权重和偏置梯度
  fc2_backward_kernel<<<fc2_grid, fc2_block, 0, actor_stream>>>(
      d_actor_hidden, d_grad_actor_fc2_output, d_grad_actor_fc2_w,
      d_grad_actor_fc2_b, hidden_dim, output_dim, batch_size);

  dim3 fc1_block(16, 16, 1); // 每个block 256 threads
  dim3 fc1_grid((hidden_dim + fc1_block.x - 1) / fc1_block.x,
                (hidden_dim + fc1_block.y - 1) / fc1_block.y,
                (batch_size + fc1_block.z - 1) / fc1_block.z);

  // 使用fc2输出的梯度计算fc1梯度
  fc1_backward_kernel<<<fc1_grid, fc1_block, 0, actor_stream>>>(
      d_shared_output, d_actor_hidden, d_grad_actor_fc2_output, d_actor_fc2_w,
      d_grad_actor_fc1_w, d_grad_actor_fc1_b, hidden_dim, hidden_dim,
      output_dim, batch_size);

  // Critic路径反向传播
  // 计算Critic Head层梯度
  linear_backward_kernel<<<head_grid, head_block, 0, critic_stream>>>(
      d_critic_fc2_output, d_critic_output, d_grad_critic_output,
      d_grad_critic_head_w, d_grad_critic_head_b, output_dim, output_dim,
      batch_size);

  // 使用fc2输出的梯度计算fc2权重和偏置梯度
  fc2_backward_kernel<<<fc2_grid, fc2_block, 0, critic_stream>>>(
      d_critic_hidden, d_grad_critic_fc2_output, d_grad_critic_fc2_w,
      d_grad_critic_fc2_b, hidden_dim, output_dim, batch_size);

  // 使用fc2输出的梯度计算fc1梯度
  fc1_backward_kernel<<<fc1_grid, fc1_block, 0, critic_stream>>>(
      d_shared_output, d_critic_hidden, d_grad_critic_fc2_output,
      d_critic_fc2_w, d_grad_critic_fc1_w, d_grad_critic_fc1_b, hidden_dim,
      hidden_dim, output_dim, batch_size);

  // 等待两个流完成
  cudaStreamSynchronize(actor_stream);
  cudaStreamSynchronize(critic_stream);

  // 执行共享层的反向传播（需要actor和critic的梯度）
  dim3 shared_block(16, 16, 1);
  dim3 shared_grid((hidden_dim + shared_block.x - 1) / shared_block.x,
                   (input_dim + shared_block.y - 1) / shared_block.y,
                   (batch_size + shared_block.z - 1) / shared_block.z);

  shared_backward_kernel<<<shared_grid, shared_block>>>(
      d_input, d_shared_output, d_grad_actor_hidden, d_grad_critic_hidden,
      d_grad_shared_w, d_grad_shared_b, input_dim, hidden_dim, batch_size);

  // 拷贝梯度结果回主机
  cudaMemcpy(grad_shared_w, d_grad_shared_w,
             hidden_dim * input_dim * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(grad_shared_b, d_grad_shared_b, hidden_dim * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaMemcpy(grad_actor_fc1_w, d_grad_actor_fc1_w,
             hidden_dim * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(grad_actor_fc1_b, d_grad_actor_fc1_b, hidden_dim * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(grad_actor_fc2_w, d_grad_actor_fc2_w,
             output_dim * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(grad_actor_fc2_b, d_grad_actor_fc2_b, output_dim * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(grad_actor_head_w, d_grad_actor_head_w,
             output_dim * output_dim * sizeof(float),
             cudaMemcpyDeviceToHost); // 拷贝Actor Head梯度
  cudaMemcpy(grad_actor_head_b, d_grad_actor_head_b, output_dim * sizeof(float),
             cudaMemcpyDeviceToHost); // 拷贝Actor Head梯度

  cudaMemcpy(grad_critic_fc1_w, d_grad_critic_fc1_w,
             hidden_dim * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(grad_critic_fc1_b, d_grad_critic_fc1_b, hidden_dim * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(grad_critic_fc2_w, d_grad_critic_fc2_w,
             output_dim * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(grad_critic_fc2_b, d_grad_critic_fc2_b, output_dim * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(grad_critic_head_w, d_grad_critic_head_w,
             output_dim * output_dim * sizeof(float),
             cudaMemcpyDeviceToHost); // 拷贝Critic Head梯度
  cudaMemcpy(grad_critic_head_b, d_grad_critic_head_b,
             output_dim * sizeof(float),
             cudaMemcpyDeviceToHost); // 拷贝Critic Head梯度

  // 释放设备内存
  cudaFree(d_input);
  cudaFree(d_shared_output);
  cudaFree(d_actor_hidden);
  cudaFree(d_critic_hidden);
  cudaFree(d_actor_fc2_output);
  cudaFree(d_critic_fc2_output);
  cudaFree(d_actor_output);
  cudaFree(d_critic_output);
  cudaFree(d_grad_actor_output);
  cudaFree(d_grad_critic_output);
  cudaFree(d_actor_fc2_w);
  cudaFree(d_critic_fc2_w);
  cudaFree(d_actor_head_w);
  cudaFree(d_critic_head_w);
  cudaFree(d_grad_shared_w);
  cudaFree(d_grad_shared_b);
  cudaFree(d_grad_actor_fc1_w);
  cudaFree(d_grad_actor_fc1_b);
  cudaFree(d_grad_actor_fc2_w);
  cudaFree(d_grad_actor_fc2_b);
  cudaFree(d_grad_actor_head_w);
  cudaFree(d_grad_actor_head_b);
  cudaFree(d_grad_critic_fc1_w);
  cudaFree(d_grad_critic_fc1_b);
  cudaFree(d_grad_critic_fc2_w);
  cudaFree(d_grad_critic_fc2_b);
  cudaFree(d_grad_critic_head_w);
  cudaFree(d_grad_critic_head_b);
  cudaFree(d_grad_actor_hidden);
  cudaFree(d_grad_critic_hidden);
  cudaFree(d_grad_actor_fc2_output);
  cudaFree(d_grad_critic_fc2_output);

  // 销毁CUDA流
  cudaStreamDestroy(actor_stream);
  cudaStreamDestroy(critic_stream);
}
}