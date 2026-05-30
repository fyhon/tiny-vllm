#include "kernels.cuh"
#include <iostream>

// TODO perhaps share these between main.cpp and kernels.cu to not duplicate them?

constexpr int N_LAYERS = 16; // TODO: hardcoded for llama 3.2 1B, just like any other value for now
constexpr int EMBEDDING_LENGTH = 2048;
constexpr int KV_DIM = 512;
constexpr int HEAD_DIM = 64;
constexpr float SQRT_HEAD_DIM = 8;
constexpr int NUM_Q_HEADS = 32;
constexpr int GQA_Q_TO_K_RATIO = 4;
constexpr int MAX_SEQ_LEN = 2048; // TODO: make it tunable
constexpr int BLOCK_SIZE = 16;    // TODO: tunable as well, defined the size of a single page in pagedattn
constexpr int V_OFFSET = BLOCK_SIZE * KV_DIM * sizeof(__nv_bfloat16);
constexpr int BLOCK_BYTES = V_OFFSET * 2;                    // * 2 because K and V
constexpr int MAX_BLOCKS_PER_SEQ = MAX_SEQ_LEN / BLOCK_SIZE; // 2048 / 16 = 128

// prefill / shared

// =============================================================================
// embeddingGatherKernel — 把 token id 序列转成 embedding 向量
// -----------------------------------------------------------------------------
// 输入：gpu_input_tokens [N]              token id（int 数组）
//       embed_tokens [vocab, 2048]        embedding lookup table（bf16）
// 输出：gpu_input_embeds [N, 2048]        每个 token 对应的 2048 维 embedding
//
// Launch：<<<num_input_tokens, 1024>>>
//   blockIdx.x  = 第几个 token
//   threadIdx.x = 0..1023，每 thread 处理 2 维（共覆盖 2048 维）
//
// 调用站点：main.cpp:159（prefill 入口处）；decode 走的是 embeddingGatherDecode
//
// ----- 学习反问（答案在 kernel 末尾）-----
//   [Q1] 为什么每 thread 处理 2 个元素而不是 1 个？1024 thread 每个处理 1 维不就够 1024 维了？
// =============================================================================
// gpu_input_tokens - N tokens
// gpu_input_embeds - N * sizeof(__nv_bfloat16) * 2048
// embed_tokens - (100000+smth, 2048)
// num_input_tokens - N (just N, not N tokens)
__global__ void embeddingGatherKernel(int *gpu_input_tokens, __nv_bfloat16 *gpu_input_embeds, __nv_bfloat16 *embed_tokens, int num_input_tokens)
{
    int workIndex = threadIdx.x + blockIdx.x * 2048;
    if (workIndex < num_input_tokens * 2048)
    {
        // 同一 block 内所有 thread 共用 blockIdx.x 这个 token id，去 embed_tokens 表里取行
        // 每 thread 取该行中两个位置：threadIdx.x 和 threadIdx.x + 1024
        gpu_input_embeds[workIndex] = embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x];
        gpu_input_embeds[workIndex + 1024] = embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x + 1024];
    }
    // ---------------------------------------------------------------------
    // 反问答案
    // [A1] 因为 RTX 5090 max threads per block = 1024，但 EMBEDDING_LENGTH=2048。
    //      所以每 thread 必须处理 2 个元素覆盖整行。这是硬件约束驱动的写法，作者
    //      在 embeddingGather() 包装函数里也写了注释。
    //      如果改用 grid stride loop（一维 launch <<<N, 1024>>>，每 thread 用
    //      stride=1024 覆盖整行），代码更通用但不影响性能。
    // ---------------------------------------------------------------------
}

void embeddingGather(int *gpu_input_tokens, __nv_bfloat16 *gpu_input_embeds, __nv_bfloat16 *embed_tokens, int num_input_tokens)
{
    // even though embedding is 2048, I can only dispatch 1024 because it's max threads per block on my gpu
    embeddingGatherKernel<<<num_input_tokens, 1024>>>(gpu_input_tokens, gpu_input_embeds, embed_tokens, num_input_tokens);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

// =============================================================================
// rmsNormKernel — RMSNorm（Root Mean Square Normalization）
// -----------------------------------------------------------------------------
// 公式：output[i] = input[i] / sqrt(mean(input²) + eps) × weight[i]
//      （比 LayerNorm 少一个减均值步骤，LLaMA 系列的标配）
//
// 输入：input [num_tokens, 2048]
//       norm_weights [2048]
// 输出：output [num_tokens, 2048]   in-place 安全（input 和 output 可指向同一片）
//
// Launch：<<<num_tokens, 1024>>>，每 block 处理一个 token 的 2048 维向量
//
// 算法核心：block 级 tree reduction（1024 thread 求和 2048 个平方）
//   - 每 thread 算 2 个元素的平方相加 → SMEM[1024]
//   - log2(1024) = 10 级树形 reduce，最终 SMEM[0] = Σ x²
//   - thread 0 算 RMS = sqrt(Σ x² / 2048 + 1e-5)
//   - 所有 thread 读 SMEM[0]（broadcast）→ 各自做归一化和 weight 乘法
//
// 调用站点：main.cpp:167(input_layernorm prefill) / 402(post_attn prefill)
//          / 495(final norm prefill) / 760,902,972(decode 对应)
//
// ----- 学习反问（答案在 kernel 末尾）-----
//   [Q1] 为什么不用 warp shuffle 做 reduce？1024 = 32 warps，warp 内 shfl 比 SMEM 快得多
//   [Q2] 这里的 1024 是 block size 而不是 warp size，为什么 i*=2 这种 stride 模式仍然有效？
//        会不会有 warp divergence？
// =============================================================================
__global__ void rmsNormKernel(__nv_bfloat16 *input, __nv_bfloat16 *output, __nv_bfloat16 *norm_weights, int num_tokens)
{
    __shared__ float rms_vector[1024];
    int workIndex = threadIdx.x + blockIdx.x * 2048;
    if (workIndex < num_tokens * 2048)
    {
        // 每 thread 处理 2 个元素：tid 和 tid+1024。把两个平方先加在一起放进 SMEM[tid]
        // 直接 (float) cast 是为了避免 bf16² 累加时精度爆炸
        rms_vector[threadIdx.x] = (float)input[workIndex] * (float)input[workIndex] + (float)input[workIndex + 1024] * (float)input[workIndex + 1024];
        __syncthreads();
        // tree reduction
        // 第 1 轮 i=1：偶数 thread 加上 tid+1（合并相邻两元素）
        // 第 2 轮 i=2：tid%4==0 的 thread 加上 tid+2
        // ... 第 10 轮 i=512：thread 0 加上 thread 512
        // 最终 rms_vector[0] = Σ x² (i=0..2047)
        for (int i = 1; i < 1024; i = i * 2)
        {
            if (threadIdx.x % (i * 2) == 0)
            {
                rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + i];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0)
        {
            // 注意分母用的是 2048（向量维度）不是 1024（thread 数）
            // 1.0e-5 是 LLaMA 用的 eps；不同模型可能不同（如 Qwen 用 1e-6）
            rms_vector[0] = sqrt(rms_vector[0] / 2048.0 + 1.0e-5);
        }
        __syncthreads();
        // <(^-^)>
        // 此时所有 thread 读 rms_vector[0] = RMS 值（broadcast）
        // 每 thread 写出对应 2 个位置的归一化结果
        output[workIndex] = (__nv_bfloat16)(((float)input[workIndex] / rms_vector[0]) * (float)norm_weights[threadIdx.x]);
        output[workIndex + 1024] = (__nv_bfloat16)(((float)input[workIndex + 1024] / rms_vector[0]) * (float)norm_weights[threadIdx.x + 1024]);
    }
    // ---------------------------------------------------------------------
    // 反问答案
    // [A1] 教学代码统一用 SMEM tree reduction 模式，可读性优先；性能上 warp
    //      shuffle 至少快 2-3×。生产级写法应该是：
    //        1) warp 内 shfl_down 5 级 → 每 warp 的 lane 0 得到 32 个元素之和
    //        2) lane 0 写到 SMEM[warp_id]（共 32 个 partial sum）
    //        3) 第 0 个 warp 再用 shfl_down 5 级把 32 个 partial sum 合并
    //      这样总共只需 2 次 __syncthreads。pagedAttention 的跨 warp reduce
    //      就是这种风格。
    //
    // [A2] tid%(i*2)==0 在 i 较小时（i=1,2,4,8,16）warp 内不同 lane 走不同分支，
    //      存在 warp divergence。但因为：
    //        a) 各 thread 都在等 __syncthreads，divergence 反正会被同步消化
    //        b) 不参与 reduce 的 thread 只是 idle，没有分支内的计算
    //      所以代价是「有些 thread 闲着」，不是真正的指令重发。
    //      shuffle 版本下 stride 1/2/4/8/16 都在 warp 内通过 register 传值，
    //      没有 idle thread，理论上更高效。
    // ---------------------------------------------------------------------
}

// (N, 2048) -> (N, 2048)
void rmsNorm(__nv_bfloat16 *input, __nv_bfloat16 *output, __nv_bfloat16 *norm_weights, int num_tokens)
{
    rmsNormKernel<<<num_tokens, 1024>>>(input, output, norm_weights, num_tokens);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

// =============================================================================
// ropeKernel — 旋转位置编码（Rotary Position Embedding）prefill 版
// -----------------------------------------------------------------------------
// RoPE 的核心想法：把绝对位置编码转成相对位置编码，通过对每对相邻元素 (x_2i, x_2i+1)
// 应用一个二维旋转矩阵 R(m·θ_i) 实现。其中 m 是 token 在 sequence 中的位置，
// θ_i = 1/base^(2i/HEAD_DIM) 是这一对维度的"旋转频率"。
//
// 数学：
//   x_2i'   = x_2i · cos(m·θ) - x_2i+1 · sin(m·θ)
//   x_2i+1' = x_2i · sin(m·θ) + x_2i+1 · cos(m·θ)
//
// 输入/输出：input [num_tokens, proj_dim]   in-place 修改
//   proj_dim = 2048 (Q) 或 512 (K)，对应不同 launch
//
// Launch：<<<num_tokens, proj_dim/2>>>
//   blockIdx.x  = token 在 sequence 中的位置 m
//   threadIdx.x = 处理第几对元素，每 thread 负责一对 (2i, 2i+1)
//
// 关键技巧：θ_i 在 head 内部循环（2i 范围 0..HEAD_DIM-1），不同 head 用同一组 θ。
//   所以代码里 double_i = 2 * (threadIdx.x % 32)，% 32 是因为 HEAD_DIM=64
//   一个 head 有 32 对元素，threadIdx.x 跨过一个 head 后 θ 重新从 0 开始。
//
// base = 500000 是 LLaMA 3 的设定（原版 RoPE 用 10000，长上下文模型加大 base）
//
// 调用站点：main.cpp:245(Q rope) / 246(K rope)，每层 prefill 都调一次
//
// ----- 学习反问（答案在 kernel 末尾）-----
//   [Q1] 为什么 K 也要 RoPE？V 不用？
//   [Q2] 每次 kernel 调用都重算 cos/sin（昂贵的浮点运算），有什么优化空间？
// =============================================================================
__global__ void ropeKernel(__nv_bfloat16 *input, int num_tokens, int proj_dim)
{
    if (2 * threadIdx.x + 1 + blockIdx.x * proj_dim < num_tokens * proj_dim)
    {
        // TODO: precompute thetas, angles and perhaps sin/cos vals and reuse it across all kernel invocations
        // double_i 是当前 thread 负责的"对"在 head 内的 even index（0, 2, 4, ..., 62）
        // %32 让每个 head（32 对元素）独立循环：head 0 thread 0..31 对应 double_i 0..62，
        // head 1 thread 32..63 也对应 double_i 0..62
        int double_i = 2 * (threadIdx.x % 32);
        // θ_i = 1 / base^(2i/HEAD_DIM)，频率随维度递减
        // 高频在低维（i 小，θ 大），低频在高维（i 大，θ 小）—— 类似 sinusoidal PE
        float theta = 1.0 / (pow(500000.0, ((float)double_i / HEAD_DIM)));
        // angle = position × θ_i，位置越大、维度频率越高 → 旋转角度越大
        float angle = blockIdx.x * theta;
        __nv_bfloat16 prev_2i = input[2 * threadIdx.x + blockIdx.x * proj_dim];
        __nv_bfloat16 prev_2i_1 = input[2 * threadIdx.x + 1 + blockIdx.x * proj_dim];
        // 二维旋转矩阵：[cos -sin; sin cos] · [x_2i; x_2i+1]
        input[2 * threadIdx.x + blockIdx.x * proj_dim] = (__nv_bfloat16)((float)prev_2i * cos(angle) - (float)prev_2i_1 * sin(angle));
        input[2 * threadIdx.x + 1 + blockIdx.x * proj_dim] = (__nv_bfloat16)((float)prev_2i * sin(angle) + (float)prev_2i_1 * cos(angle));
    }
    // ---------------------------------------------------------------------
    // 反问答案
    // [A1] Q·K^T 是 attention 的核心，相对位置信息只需要在 Q 和 K 上施加。V 不
    //      参与 score 计算，只是 score 加权累加的对象，不需要位置编码。原始
    //      sinusoidal PE（Vaswani 2017）也只在 Q/K 上加，V 直接传递信息。
    //      数学上：(R_m·q) · (R_n·k) = q · R_{m-n} · k，旋转的差等价于相对位置。
    //
    // [A2] cos(angle) 和 sin(angle) 在以下层面都可以预计算：
    //      a) 同一 (position, dim_pair) 在所有 layer 都重复使用 → 模型生命周期内
    //         可以一次性算完整张表 [MAX_SEQ_LEN, HEAD_DIM/2] × {cos, sin}
    //      b) HuggingFace transformers 的标准做法：把 cos/sin 表存为模型的
    //         buffer（不是 parameter），第一次 forward 时填充
    //      c) 进一步优化：把 cos/sin 直接乘进 weight（但 K 需要 layer-wise，
    //         不一定划算）
    //      作者注释里也写了 TODO 标记。Tiny-vLLM 没做是为了教学简洁。
    // ---------------------------------------------------------------------
}

// proj_dim: q_proj 2048, k_proj 512
// num_threads: I want to use it for both q_proj and k_proj so need to parameterize num_threads (1024 for q_proj and 512 for k_proj)
void rope(__nv_bfloat16 *input, int num_tokens, int proj_dim)
{
    int num_threads = proj_dim / 2;
    if (num_threads > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, RoPE kernel not launched";
        return;
    }

    ropeKernel<<<num_tokens, num_threads>>>(input, num_tokens, proj_dim);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

// =============================================================================
// causalMaskKernel — 给 attention scores 矩阵打因果 mask（仅 prefill 用）
// -----------------------------------------------------------------------------
// 输入/输出：input [NUM_Q_HEADS, num_tokens, num_tokens]   in-place 修改
//   形状是 attention scores 矩阵，行 = query，列 = key
//
// causal mask = 上三角矩阵置 -inf：每个 query 只能看到自己和之前的 token
// 数学：input[h, row, col] = -inf  if col > row
//
// Launch：<<<num_tokens × NUM_Q_HEADS, num_tokens>>>
//   blockIdx.x  = head_id × num_tokens + row  （混合编码）
//   threadIdx.x = column
//   row = blockIdx.x % num_tokens（取模拿到 row 编号）
//
// 调用站点：main.cpp:330（prefill 每层 attention 算完 QK^T 后调用）
// 注意：decode 不需要 mask（每次只生成 1 个 token，自然不会 attend 到未来）—
//       这就是 pagedAttention 没有 mask 步骤的原因
//
// ----- 学习反问（答案在 kernel 末尾）-----
//   [Q1] 为什么把 -inf 而不是 0 写进 input？后面 softmax 怎么用？
//   [Q2] HUGE_VALF 和 -INFINITY 在 fp16/bf16 cast 后会变成什么？还能保证 mask 行为吗？
// =============================================================================
__global__ void causalMaskKernel(__nv_bfloat16 *input, int num_tokens)
{
    if (threadIdx.x + blockIdx.x * blockDim.x >= num_tokens * num_tokens * NUM_Q_HEADS)
    {
        return;
    }

    int column = threadIdx.x;
    // blockIdx.x 编码了 (head_id, row)：每个 head 占 num_tokens 个 block
    // row = blockIdx.x % num_tokens，head_id = blockIdx.x / num_tokens
    // 这里只取 row 是因为 mask 只跟 (row, col) 有关，跨 head 都是同样的上三角图案
    int row = blockIdx.x % num_tokens;
    if (column > row)
    {
        // 上三角部分（未来 token）置 -inf，这样 softmax 后 exp(-inf)=0，对应权重为 0
        input[blockIdx.x * num_tokens + threadIdx.x] = -HUGE_VALF;
    }
    // ---------------------------------------------------------------------
    // 反问答案
    // [A1] -inf 是 softmax-friendly 的"屏蔽值"。softmax 公式：
    //        out[i] = exp(s_i - max) / Σ exp(s_j - max)
    //      mask 位置 s_i = -inf → exp(-inf - max) = 0，自然在分母分子都不贡献。
    //      如果置 0，softmax 后会变成 exp(0 - max)，不是 0，相当于给未来 token
    //      均匀分配了一些权重，破坏了 causal 约束。
    //      "additive mask" 是更通用的写法（在 attention scores 上加 mask 矩阵），
    //      0 表示 attend，-inf 表示屏蔽。
    //
    // [A2] -HUGE_VALF 是 float 的负最大（约 -3.4e38），cast 到 bf16 后会饱和到
    //      bf16 的负最大（约 -3.39e38，因为 bf16 共享 fp32 的 8-bit exponent）。
    //      仍然是绝对值极大的负数，softmax 中 exp(-3.4e38) = 0，mask 行为保持。
    //      若是 fp16（5-bit exponent），-HUGE_VALF cast 后会变成 -inf（fp16 max ≈ 6.5e4），
    //      也仍然 OK。但如果是 int8 量化的 attention，就需要更小心地选屏蔽常数。
    // ---------------------------------------------------------------------
}

void causalMask(__nv_bfloat16 *input, int num_tokens)
{
    if (num_tokens > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, Causal mask kernel not launched";
        return;
    }

    causalMaskKernel<<<num_tokens * NUM_Q_HEADS, num_tokens>>>(input, num_tokens);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

// =============================================================================
// softmaxKernel — 行 softmax（prefill 用，与 pagedAttention 的 online softmax 形成对照）
// -----------------------------------------------------------------------------
// 算法：传统两遍 softmax
//   Pass 1: 找全行最大值（用于数值稳定性，subtract max 后 exp 不会溢出）
//   Pass 2: 计算 exp(x - max)，求和得到分母 d
//   Pass 3: out[i] = exp(x_i - max) / d
//
// 输入/输出：input [NUM_Q_HEADS, num_tokens, num_tokens]   in-place
//   每行就是一个 query 对所有 key 的 attention scores
//
// Launch：<<<num_tokens × NUM_Q_HEADS, num_tokens>>>
//   blockIdx.x  = (head_id × num_tokens + row)
//   threadIdx.x = 列索引（softmax 是按行做的，每个 thread 负责一列元素）
//
// 关键 SMEM：row[1024] 既存原始 token 又存 max/sum 的 reduce 中间结果
//   max 算完后存到独立的 max_val 让 row 数组可以重用做 sum reduce
//
// ----- 与 pagedAttention online softmax 的对照 -----
//   两遍 softmax    online softmax
//   存 scores[N]    3 个寄存器
//   先扫一遍找 max  边算边维护 running max + correction factor
//   不能流式        可以流式（FlashAttention 的数学基础）
//   适合：prefill 一次性算完整矩阵
//   适合：decode 逐 KV token 累加
//
// 调用站点：main.cpp:332（prefill 每层 causalMask 之后）
//
// ----- 学习反问（答案在 kernel 末尾）-----
//   [Q1] 倒数第二行：input[workIndex] = expf(...) / row[0]，为什么不用前面已经算过
//        存进 row[threadIdx.x] 的 exp 值？要再算一次 expf 不浪费吗？
//   [Q2] 这个 kernel 写死了 num_tokens ≤ 1024 的限制，要扩展到 2048 token
//        prefill 怎么改？
// =============================================================================
__global__ void softmaxKernel(__nv_bfloat16 *input, int num_tokens)
{
    // softmaxxing per head
    // might waste a lot of memory by hardcoding the size here but can't use num_tokens directly
    // SMEM 这里浪费比较明显：num_tokens 远小于 1024 时 row 数组大部分闲置
    __shared__ float row[1024]; // row[0] will contain max value after the loop
    __shared__ float max_val;
    // find max of the row to subtract it for numerical stability
    int workIndex = blockIdx.x * num_tokens + threadIdx.x;
    __nv_bfloat16 token = input[workIndex];
    row[threadIdx.x] = (float)token;
    __syncthreads();

    // ---- Pass 1: tree reduction 求 max ----
    // i 控制 stride：1 → 2 → 4 → ... → num_tokens/2
    // 与 rmsNorm 同款 reduce 模式，但用 fmaxf 而不是加法
    for (int i = 1; i < num_tokens; i = i * 2)
    {
        if (threadIdx.x % (i * 2) == 0 && threadIdx.x + i < num_tokens)
        {
            row[threadIdx.x] = fmaxf(row[threadIdx.x], row[threadIdx.x + i]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        max_val = row[0]; // so I don't need to allocate another shared value for max_val
    }
    __syncthreads();

    // ---- Pass 2: 算 exp(token - max)，覆盖回 row 数组 ----
    // turn into exp
    row[threadIdx.x] = expf((float)token - max_val);
    __syncthreads();

    // ---- Pass 3: tree reduction 求 sum ----
    // now I can compute the numerical stable sum, similar pattern - tree reduction
    // reusing row memory
    for (int i = 1; i < num_tokens; i = i * 2)
    {
        if (threadIdx.x % (i * 2) == 0 && threadIdx.x + i < num_tokens)
        {
            row[threadIdx.x] = row[threadIdx.x] + row[threadIdx.x + i];
        }
        __syncthreads();
    }

    // ---- 写出归一化结果 ----
    // 注意：再次计算 expf 而不是用 row[threadIdx.x]！见反问 [A1]
    input[workIndex] = (__nv_bfloat16)(expf((float)token - max_val) / row[0]);
    // ---------------------------------------------------------------------
    // 反问答案
    // [A1] 因为 row[threadIdx.x] 在 Pass 3 的 tree reduction 中已经被覆盖了！
    //      除了 row[0]（汇总分母）外，其他 row[i] 都是中间累加结果，不是当初
    //      存进去的 exp(x_i - max) 了。为了拿回 exp(x_i - max)，要么：
    //        a) 像现在这样再算一次（多 1 次 expf，简洁但浪费）
    //        b) 用第二个 SMEM 数组存原始 exp 值，避免被 reduce 覆盖（多消耗 SMEM）
    //        c) 不在 row 上做 reduce，先把 row 拷到另一个数组再 reduce
    //      生产实现一般用 (b)，因为 expf 是相对昂贵的浮点运算（比 SMEM 多一倍空间换性能）。
    //
    // [A2] 1024 限制来自 RTX 5090 max threads per block。要支持 2048：
    //        a) 每 thread 处理多个元素（grid stride loop），但 reduce 仍按 1024 thread 走
    //        b) 拆 block：一个 row 用多个 block，先各自算 partial max/sum，再用第二个
    //           kernel 合并（two-pass kernel）
    //        c) 升级到更大的 GPU（H100 SM 支持 1024 threads/block 但 cluster 可以扩 multiblock）
    //      vLLM/SGLang 的 fused softmax 走的是 (a) + warp shuffle，能优雅处理任意 row 长。
    //      Tiny-vLLM 用 1024 限制简化代码。
    // ---------------------------------------------------------------------
}

// input are masked attention scores (NUM_Q_HEADS, num_tok, num_tok)
void softmax(__nv_bfloat16 *input, int num_tokens)
{
    if (num_tokens > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, Softmax kernel not launched";
        return;
    }

    softmaxKernel<<<num_tokens * NUM_Q_HEADS, num_tokens>>>(input, num_tokens);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

// =============================================================================
// residualKernel — Residual connection（element-wise add）
// -----------------------------------------------------------------------------
// 公式：input[i] += input_embeds[i]   in-place
// Transformer 标配：attention/FFN 输出 + 输入 residual = 下一层输入
//
// 输入：input [num_tokens, 2048]         attention 或 FFN 的输出
//       input_embeds [num_tokens, 2048]  本层的输入（即上一层的输出）
// 输出：input 被覆盖为 input + input_embeds
//
// Launch：<<<num_tokens, 1024>>>，每 thread 处理 2 个元素覆盖 2048 维
//
// 调用站点：main.cpp:400, 493(prefill) / 900, 969(decode)
//
// ----- 学习反问（答案在 kernel 末尾）-----
//   [Q1] 为什么 residual 要单独写一个 kernel？这种 element-wise 加法不能融到上一个
//        kernel 的 epilogue 里吗？
// =============================================================================
__global__ void residualKernel(__nv_bfloat16 *input, __nv_bfloat16 *input_embeds)
{
    int workIndex = threadIdx.x + blockIdx.x * 2048;
    // 每 thread 写 2 个位置（同 rmsNorm/embeddingGather 模式，2048 / 1024 thread = 2 元素/thread）
    input[workIndex] = input[workIndex] + input_embeds[workIndex];
    input[workIndex + 1024] = input[workIndex + 1024] + input_embeds[workIndex + 1024];
    // ---------------------------------------------------------------------
    // 反问答案
    // [A1] 完全可以而且应该融。Tiny-vLLM 拆开是教学清晰优先。生产实现的常见融合：
    //        a) cublas GEMM epilogue：down_proj 矩阵乘的输出 = down·input + bias + residual
    //           （cublas 11+ 支持 epilogue 融合 bias/relu/gelu，部分版本支持 residual add）
    //        b) torch.compile / TorchInductor pointwise fusion：自动把 GEMM 后续的
    //           elementwise 操作（add, silu, mul）融为一个 kernel
    //        c) vLLM 的 fused_add_rms_norm：residual + rmsNorm 融成一个 kernel，
    //           少一次 GMEM 读写
    //      Tiny-vLLM 单独 launch 这个 kernel 增加了 launch 开销 + 一次 GMEM 写后立即读
    //      （input 写出后下一个 rmsNorm 立刻读回），不利于 memory-bound 性能。
    //      进入"实践验证"阶段时这是一个值得 nsight 测的点。
    // ---------------------------------------------------------------------
}

// (num_tok, 2048) + (num_tok, 2048) -> (num_tok, 2048)
void residualAdd(__nv_bfloat16 *input, __nv_bfloat16 *input_embeds, int num_tokens)
{
    residualKernel<<<num_tokens, 1024>>>(input, input_embeds);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

// =============================================================================
// siluKernel — SwiGLU 的核心：SiLU(gate) × up，element-wise，in-place
// -----------------------------------------------------------------------------
// 公式：a[i] = SiLU(a[i]) × b[i] = a[i] · sigmoid(a[i]) · b[i]
//      = a[i] · (1 / (1 + exp(-a[i]))) · b[i]
//
// LLaMA FFN（SwiGLU 变体）：
//   gate = x @ W_gate         [N, intermediate_size=8192]
//   up   = x @ W_up           [N, 8192]
//   y    = (SiLU(gate) ⊙ up) @ W_down   → [N, 2048]
// 这个 kernel 算的就是 SiLU(gate) ⊙ up（写回 a），下一步会 cublas 做 ⊙ W_down
//
// 输入：a [num_tokens, 8192]   gate（in-place 覆盖为输出）
//       b [num_tokens, 8192]   up
// 输出：a 被覆盖为 SiLU(gate) × up，b 不变
//
// Launch：<<<num_tokens, 1024>>>，每 thread 处理 8 个元素（8192 / 1024）
//
// 调用站点：main.cpp:460(prefill) / 946(decode)
//
// ----- 学习反问（答案在 kernel 末尾）-----
//   [Q1] 为什么 SwiGLU 比标准 FFN（ReLU(x@W1)@W2）效果好？
//   [Q2] sigmoid 用 1/(1+exp(-x)) 在数值上有没有问题？x 很负的时候会怎样？
// =============================================================================
__global__ void siluKernel(__nv_bfloat16 *a, __nv_bfloat16 *b)
{
    int workIndex = threadIdx.x + blockIdx.x * 8192;
    // 每 thread 处理 8 个连续元素（间隔 1024 是因为 8192/8=1024 thread 满载）
    // 不是 grid stride loop，而是 thread 内 stride loop
    for (int i = 0; i < 8192; i += 1024)
    {
        // a · sigmoid(a) · b，全用 float 计算（bf16 精度不够 expf 容易溢出）
        a[workIndex + i] = (__nv_bfloat16)((float)a[workIndex + i] * (1 / (1 + expf(-(float)a[workIndex + i]))) * (float)b[workIndex + i]);
    }
    // ---------------------------------------------------------------------
    // 反问答案
    // [A1] SwiGLU = SiLU + GLU（Gated Linear Unit），有两条 W_gate / W_up 平行的
    //      投影，乘起来形成一个"门控"机制：模型能动态选择哪些通道流过去。
    //        - 比 ReLU 平滑（梯度连续，便于训练）
    //        - 门控让模型自己学"该激活什么"，比固定 ReLU 表达力更强
    //        - 实证上 PaLM/LLaMA/Qwen 都用，比 ReLU FFN 提升 1-2% perplexity
    //      代价：参数量翻倍（W_gate + W_up 两条），所以 intermediate_size 通常
    //      取 2/3 × 4d 而不是 4d 来保持总参数量。
    //
    // [A2] 数值上 SiLU 比 sigmoid 单调，相对稳健：
    //        - x → +∞：sigmoid → 1，SiLU = x·1 = x（线性）
    //        - x → -∞：sigmoid → 0，SiLU = x·0 ≈ 0（saturate）
    //        - x → 0：SiLU ≈ x/2（near-linear）
    //      但 1/(1+expf(-x)) 在 x 很负时 expf 会溢出！例如 x=-100，expf(100)
    //      在 fp32 已经溢出到 inf。教学代码这样写没问题（因为预期 attention
    //      score 不会到那么大），生产实现常用：
    //        sigmoid(x) = 0.5 * (1 + tanhf(0.5 * x))    // 数值稳定形式
    //      或者直接调 cuDNN / cutlass 的融合 SwiGLU kernel。
    // ---------------------------------------------------------------------
}

// in-place, overwriting a
void silu(__nv_bfloat16 *a, __nv_bfloat16 *b, int num_tokens)
{
    siluKernel<<<num_tokens, 1024>>>(a, b);
}

// decode
// =============================================================================
// embeddingGatherKernelDecode — Decode 版本的 embedding lookup
// -----------------------------------------------------------------------------
// 与 prefill 版的关键差异：
//   prefill: 输入是一个 sequence 的 N 个 token，所有 token 都是同一 sequence 的
//   decode:  输入是 batch 中每个活跃 sequence 的"上一个生成的 token"，每 token 来自不同 sequence
//
// 输入：gpu_last_tokens [num_tokens]     batch 中每个 active slot 上次生成的 token id
//       embed_tokens [vocab, 2048]
// 输出：output [num_tokens, 2048]
//
// Launch：<<<num_tokens, 1024>>>，num_tokens = num_active_slots
//
// 调用站点：main.cpp:757（decode 主循环每一轮的入口）
// =============================================================================
__global__ void embeddingGatherKernelDecode(int *gpu_last_tokens, int num_tokens, __nv_bfloat16 *output, __nv_bfloat16 *embed_tokens)
{
    int input_token = gpu_last_tokens[blockIdx.x];
    int workIndex = blockIdx.x * 2048 + threadIdx.x;
    if (workIndex < num_tokens * 2048)
    {
        output[workIndex] = embed_tokens[input_token * 2048 + threadIdx.x];
        output[workIndex + 1024] = embed_tokens[input_token * 2048 + threadIdx.x + 1024];
    }
}

void embeddingGatherDecode(int *gpu_last_tokens, int num_tokens, __nv_bfloat16 *output, __nv_bfloat16 *embed_tokens)
{
    // even though embedding is 2048, I can only dispatch 1024 because it's max threads per block on my gpu
    embeddingGatherKernelDecode<<<num_tokens, 1024>>>(gpu_last_tokens, num_tokens, output, embed_tokens);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

// =============================================================================
// ropeKernelDecode — Decode 版本的 RoPE
// -----------------------------------------------------------------------------
// 与 prefill 版（ropeKernel）的差异：
//   prefill:  blockIdx.x = 0..N-1，对应一个 sequence 的所有 token 位置
//   decode:   一次只处理 1 个 token（grid 是 <<<1, ...>>>），位置由参数显式传入
//
// 关键参数：position_in_sequence —— 当前生成的这个 token 在它的 sequence 里的位置
//   每次调用都不同（随生成进度递增）
//
// Launch：<<<1, proj_dim/2>>>，单 block，每 thread 一对元素
//
// 调用站点：main.cpp:845(Q decode rope) / 846(K decode rope)
//   注意：每个 active slot 都会调一次（main 里有循环），不是 batch 一次性处理
//
// ----- 学习反问（答案在 kernel 末尾）-----
//   [Q1] 为什么 decode RoPE 不批量处理所有 active slot？显然 batch 写一个 kernel
//        效率更高，作者为什么逐 slot 调？
// =============================================================================
__global__ void ropeKernelDecode(__nv_bfloat16 *input, int position_in_sequence, int proj_dim)
{
    if (2 * threadIdx.x + 1 < proj_dim) // TODO: check correctness
    {
        // TODO: precompute thetas, angles and perhaps sin/cos vals and reuse it across all kernel invocations
        // 与 prefill 版完全相同的 θ 计算
        int double_i = 2 * (threadIdx.x % 32);
        float theta = 1.0 / (pow(500000.0, ((float)double_i / HEAD_DIM)));
        // 关键差异：position 不是 blockIdx.x 而是显式传入的参数
        float angle = position_in_sequence * theta;
        __nv_bfloat16 prev_2i = input[2 * threadIdx.x];
        __nv_bfloat16 prev_2i_1 = input[2 * threadIdx.x + 1];
        // 二维旋转矩阵（同 prefill 版）
        input[2 * threadIdx.x] = (__nv_bfloat16)((float)prev_2i * cos(angle) - (float)prev_2i_1 * sin(angle));
        input[2 * threadIdx.x + 1] = (__nv_bfloat16)((float)prev_2i * sin(angle) + (float)prev_2i_1 * cos(angle));
    }
    // ---------------------------------------------------------------------
    // 反问答案
    // [A1] Tiny-vLLM 设计简化的代价。理想批量处理需要：
    //        a) batch 里每个 slot 的 position_in_sequence 不一样（continuous batching
    //           各 slot 独立推进），所以 angle 不能预计算成一个共用常数
    //        b) 需要把 position 数组也传到 GPU，kernel 里通过 blockIdx.x 索引
    //      这其实并不复杂，但作者选择在 main.cpp 里循环 launch（看 845-846 行）。
    //      代价是 N 个 active slot 就要 N 次 launch overhead，几个 µs/次。
    //      vLLM 实际实现会做完整 batch RoPE。这是阶段 B 实践验证的优化候选项之一。
    // ---------------------------------------------------------------------
}

// proj_dim: q_proj 2048, k_proj 512
// num_threads: I want to use it for both q_proj and k_proj so need to parameterize num_threads (1024 for q_proj and 512 for k_proj)
void ropeDecode(__nv_bfloat16 *input, int position_in_sequence, int proj_dim)
{
    int num_threads = proj_dim / 2;
    if (num_threads > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, RoPE kernel not launched";
        return;
    }

    ropeKernelDecode<<<1, num_threads>>>(input, position_in_sequence, proj_dim);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

// =============================================================================
// softmaxKernelDecode — Decode 版本的两遍 softmax（实际上 decode 走 pagedAttention，
// 这个 kernel 在当前主流程未被调用）
// -----------------------------------------------------------------------------
// 与 prefill softmaxKernel 的差异：
//   - row stride = MAX_SEQ_LEN（2048）而不是 num_tokens
//     decode 的 attention scores 每 head 留了 MAX_SEQ_LEN 长度的固定 buffer，
//     当前实际使用 seq_len 个，剩下是 padding
//   - input 形状：[NUM_Q_HEADS, seq_len]，不是 prefill 的方阵 [NUM_Q_HEADS, N, N]
//   - launch grid = NUM_Q_HEADS（不需要 ×num_tokens，因为 decode 一次只生成 1 个 token）
//
// ⚠️ 当前 decode 主路径走的是 pagedAttention（含融合的 online softmax），不会调
// 这个 kernel。它是给"非 paged 版本的 decode attention"留的代码路径，但 main.cpp
// 没在用。可能是早期实现遗留。
//
// Launch：<<<NUM_Q_HEADS, seq_len>>>，每 head 一个 block，block 内每 thread 1 个 token
//
// 调用站点：(grep main.cpp 后) 当前未被调用
//
// 算法本身与 prefill softmax 完全相同：tree reduce 求 max → exp → tree reduce 求 sum
// 详细注释参见上方 softmaxKernel
// =============================================================================
// seq_len increases by 1 with every new token
__global__ void softmaxKernelDecode(__nv_bfloat16 *input, int seq_len)
{
    // softmaxxing per head
    // might waste a lot of memory by hardcoding the size here but can't use num_tokens directly
    __shared__ float row[1024]; // row[0] will contain max value after the loop
    __shared__ float max_val;
    // find max of the row to subtract it for numerical stability
    // 关键差异：workIndex 用 MAX_SEQ_LEN 作为 stride（与 prefill 用 num_tokens 不同）
    int workIndex = blockIdx.x * MAX_SEQ_LEN + threadIdx.x;
    __nv_bfloat16 token = input[workIndex];
    row[threadIdx.x] = (float)token;
    __syncthreads();

    for (int i = 1; i < seq_len; i = i * 2)
    {
        if (threadIdx.x % (i * 2) == 0 && threadIdx.x + i < seq_len)
        {
            row[threadIdx.x] = fmaxf(row[threadIdx.x], row[threadIdx.x + i]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        max_val = row[0]; // so I don't need to allocate another shared value for max_val
    }
    __syncthreads();

    // turn into exp
    row[threadIdx.x] = expf((float)token - max_val);
    __syncthreads();

    // now I can compute the numerical stable sum, similar pattern - tree reduction
    // reusing row memory
    for (int i = 1; i < seq_len; i = i * 2)
    {
        if (threadIdx.x % (i * 2) == 0 && threadIdx.x + i < seq_len)
        {
            row[threadIdx.x] = row[threadIdx.x] + row[threadIdx.x + i];
        }
        __syncthreads();
    }

    input[workIndex] = (__nv_bfloat16)(expf((float)token - max_val) / row[0]);
}

// input are masked attention scores (NUM_Q_HEADS, seq_len)
void softmaxDecode(__nv_bfloat16 *input, int seq_len)
{
    if (seq_len > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, Softmax kernel not launched";
        return;
    }

    softmaxKernelDecode<<<NUM_Q_HEADS, seq_len>>>(input, seq_len);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

// =============================================================================
// pagedAttentionKernel — Decode 阶段的 attention kernel（这个文件最复杂的一个）
// -----------------------------------------------------------------------------
// Launch 配置：<<<dim3(num_active_slots, NUM_Q_HEADS), HEAD_DIM>>>
//   blockIdx.x  = active_slot   batch 中第几个 sequence（continuous batching 中的活跃 slot）
//   blockIdx.y  = q_head_id     哪个 Q head（0..31）
//   threadIdx.x = thread_id     HEAD_DIM 维度的哪一维（0..63），每 thread 负责 1 维
//
// 数据流（每个 KV token 一轮）：
//   1) Q · K 点积：每 thread 贡献 1 维乘积，2 个 warp 共 64 个 thread 总和 = 完整点积
//   2) warp 内 shfl_down 5 级 reduce → 每个 warp 的 lane 0 持有该 warp 32 维之和
//   3) 跨 warp 通过 SMEM dot_products[2] 汇合 → thread 0 合并 + 除以 sqrt(HEAD_DIM)
//   4) online softmax 累加：修正 acc / d，加入新 token 贡献
//
// 关键技术：
//   - online softmax：3 个 float 寄存器（current_max / acc / d）替代 O(seq_len) 的 SMEM
//     传统 softmax 要存完整 scores 数组才能算 max/sum；online 版边算边更新，不存中间结果
//     来源：Milakov & Gimelshein 2018，也是 FlashAttention 的数学核心
//
//   - 跨 warp reduce：HEAD_DIM=64 = 2 warp。warp 内用 shfl 零成本，跨 warp 用 SMEM
//     规则：warp 内永远 shfl，跨 warp 永远 SMEM + __syncthreads
//
//   - GQA：32 Q head : 8 KV head（4:1）。kv_head_idx = q_head_id / GQA_Q_TO_K_RATIO
//     物理上 4 个 Q head 共享同一份 KV，KV cache 显存压缩 4×
//
//   - PagedAttention：block_table[slot][layer][logical_block] → physical_block
//     逻辑连续的 KV cache 通过间接寻址映射到分散的物理块（类比 OS 虚拟内存）
//     物理块布局：[K 部分 BLOCK_SIZE×KV_DIM×bf16 | V 部分 同上]，V_OFFSET 是分隔点
//
// 调用站点：main.cpp:876（decode 主循环每层调用 1 次，共 16 层）
// 对照：vLLM PagedAttention v1 简化版，缺 multi-block parallelism / cp.async
//
// ----- 学习反问（先想再读代码，答案在 kernel 末尾的 [A1]-[A4] 区块）-----
//   [Q1] 为什么 dot_products 用 SMEM 而不是 register？warp 内的 shfl 不就够吗？
//   [Q2] online softmax 的 correction_factor 在 max 不变的时候等于多少？为什么这是 fallback 到普通累加？
//   [Q3] GQA 4:1 在这个 kernel 里实际上节省了什么——显存？带宽？计算？
//   [Q4] 如果把 HEAD_DIM 改成 128（一个 warp 装不下），整个跨 warp reduce 部分要怎么改？
// =============================================================================
__global__ void pagedAttentionKernel(int layer, int num_active_slots, __nv_bfloat16 *q_proj, __nv_bfloat16 *kv_cache, int *block_table_gpu, int *gpu_seq_lens, int *gpu_active_slots, __nv_bfloat16 *output)
{
    // 跨 warp 汇合点：dot_products[0] = warp 0 partial sum，[1] = warp 1 的
    __shared__ float dot_products[2];
    int active_slot = blockIdx.x; // active_slot == seq_id
    int slot = gpu_active_slots[active_slot];
    int q_head_id = blockIdx.y;
    int thread_id = threadIdx.x;
    // GQA 4:1 —— 4 个 Q head 共享 1 个 KV head（NUM_Q_HEADS=32, KV head 数=8）
    int kv_head_idx = q_head_id / GQA_Q_TO_K_RATIO;
    // 这个 thread 负责的 Q 元素（HEAD_DIM 维度中的某一维）
    __nv_bfloat16 q = q_proj[active_slot * EMBEDDING_LENGTH + q_head_id * HEAD_DIM + thread_id];
    int seq_len = gpu_seq_lens[active_slot];
    int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE; // ceil(seq_len / BLOCK_SIZE)

    // ---- online softmax 状态变量 ----
    // 数学：output = Σ exp(s_i - max) · V_i / Σ exp(s_i - max)
    // 逐 token 维护：acc = 当前分子（加权 V 累加），d = 当前分母，current_max = 历史最大 score
    // for online softmax https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
    float current_max = -INFINITY;
    float acc = 0.0f;
    float d = 0.0f; // denominator, same name as in paper above

    // 遍历这个 sequence 的所有 KV blocks
    for (int logical_block_idx = 0; logical_block_idx < num_blocks; ++logical_block_idx)
    {
        // block_table 三维索引：[slot][layer][logical_block] → physical_block_id
        // 这是 PagedAttention 的核心——逻辑连续的 KV 通过间接寻址映射到分散物理块
        int physical_block = block_table_gpu[slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + logical_block_idx];
        // 最后一个块可能不满，取 min（block 内有效 token 数）
        int tokens_in_block = min(seq_len - logical_block_idx * BLOCK_SIZE, BLOCK_SIZE);
        for (int token = 0; token < tokens_in_block; ++token)
        {
            // ---- K / V 寻址 ----
            // 物理块字节布局：[K: BLOCK_SIZE×KV_DIM×bf16][V: BLOCK_SIZE×KV_DIM×bf16]
            //   physical_block × BLOCK_BYTES        : 跳到该物理块起点
            //   token × KV_DIM × bf16                : 跳到块内第 token 个 KV
            //   kv_head_idx × HEAD_DIM × bf16        : 跳到对应 KV head
            //   thread_id × bf16                     : 跳到 HEAD_DIM 中的某一维
            // V 寻址同 K，但加 V_OFFSET（K 部分总字节数）
            __nv_bfloat16 *k = (__nv_bfloat16 *)((char *)kv_cache + physical_block * BLOCK_BYTES + token * KV_DIM * sizeof(__nv_bfloat16) + kv_head_idx * HEAD_DIM * sizeof(__nv_bfloat16) + thread_id * sizeof(__nv_bfloat16));
            __nv_bfloat16 *v = (__nv_bfloat16 *)((char *)kv_cache + physical_block * BLOCK_BYTES + V_OFFSET + token * KV_DIM * sizeof(__nv_bfloat16) + kv_head_idx * HEAD_DIM * sizeof(__nv_bfloat16) + thread_id * sizeof(__nv_bfloat16));
            // 每 thread 贡献 1 维乘积，待 reduce 后得到完整 Q·K
            float qk = (float)q * (float)*k;
            // ---- Step 1: warp 内 reduce ----
            // 32 thread 通过 shfl_down_sync 5 级树形求和，warp 内零开销不需 SMEM
            // __shfl_down_sync(mask, val, delta)：取 lane (lane+delta) 的 val
            // tree reduction within current warp, thread 0 gets sum of all 32 elements within warp
            // could be done with __syncthreads but accessing memory of other threads in warp is op
            qk += __shfl_down_sync(0xffffffff, qk, 16);
            qk += __shfl_down_sync(0xffffffff, qk, 8);
            qk += __shfl_down_sync(0xffffffff, qk, 4);
            qk += __shfl_down_sync(0xffffffff, qk, 2);
            qk += __shfl_down_sync(0xffffffff, qk, 1);
            // 至此 lane 0 持有该 warp 32 个 thread 的 qk 之和
            // ---- Step 2: 跨 warp 通过 SMEM 汇合 ----
            // warp 0 lane 0 (id=0) / warp 1 lane 0 (id=32) 各写一格
            if (thread_id == 0)
            {
                dot_products[0] = qk;
            }
            if (thread_id == 32)
            {
                dot_products[1] = qk;
            }
            __syncthreads();
            // ---- Step 3: thread 0 合并两个 warp 的 partial sum 并缩放 ----
            if (thread_id == 0)
            {
                dot_products[0] = (dot_products[0] + dot_products[1]) / SQRT_HEAD_DIM;
            }
            __syncthreads();
            // 所有 thread 读 SMEM[0]，相当于 broadcast
            float dot_product = dot_products[0];
            // ---- Online Softmax 核心 ----
            // 当 max 更新时：所有历史 exp(s_i - old_max) 需 × exp(old_max - new_max)
            //   等价于把所有历史项的 base 从 old_max 换成 new_max
            // online softmax
            float new_max = current_max;
            if (dot_product > current_max)
            {
                new_max = dot_product;
            }
            float correction_factor = expf(current_max - new_max);
            current_max = new_max;
            float exp_score = expf(dot_product - current_max);
            d = d * correction_factor + exp_score;                 // 修正分母 + 新项
            acc = acc * correction_factor + exp_score * (float)*v; // 修正分子 + 新贡献（加权 V）
        }
    }
    // 写出 attention 输出的对应一维
    output[active_slot * EMBEDDING_LENGTH + q_head_id * HEAD_DIM + thread_id] = acc / d;

    // =====================================================================
    // 反问答案（对应顶部 [Q1]-[Q4]）
    // ---------------------------------------------------------------------
    // [A1] dot_products 跨 2 个 warp 共享，shfl_down_sync 只能在同 warp 32 lane
    //      之间传递，跨 warp 必须通过 SMEM（或 GMEM）+ __syncthreads。
    //      register 是线程私有，warp shuffle 也只在 warp 内有效——这是硬件层
    //      面的边界，不是写法选择问题。
    //
    // [A2] max 不变时 correction_factor = exp(0) = 1，acc 和 d 都不缩放，等价
    //      于普通累加 acc += exp(s) · V, d += exp(s)。online softmax 在数值
    //      意义上是普通累加的"超集"，多了 max 升级时的修正分支。这就是为什么
    //      它在最坏情况下也只比普通版多两次 expf。
    //
    // [A3] 节省的是 KV Cache 显存（KV head 数 8 而不是 32，物理存储 4×↓）和
    //      KV 加载带宽（每 KV head 仍被 4 个 Q head 各读一次，但因为 KV cache
    //      只存 8 头，HBM→SM 加载量降了 4×）。Q·K 计算 FLOPs 不变（仍然是 32
    //      个 Q head 各做完整点积）。
    //      所以 GQA 是 memory-bound 优化，不影响 compute roofline 的纵轴位置，
    //      只把 arithmetic intensity 推到右边（计算量不变，访存量降，AI 提升）。
    //
    // [A4] HEAD_DIM=128 = 4 warp。需要：
    //      1) dot_products[4]，每个 warp 的 lane 0（thread_id = 0/32/64/96）
    //         各写一格；
    //      2) 最后 thread 0 累加 4 项再除以 sqrt(128)；
    //      3) 或者用 log2(num_warps) 级树状 reduce（4 → 2 → 1）效率更高。
    //      reduce 树深度 = log2(num_warps)，代码会更长但思路不变。
    // =====================================================================
}

void pagedAttention(int layer, int num_active_slots, __nv_bfloat16 *q_proj, __nv_bfloat16 *kv_cache, int *block_table_gpu, int *gpu_seq_lens, int *gpu_active_slots, __nv_bfloat16 *output)
{
    pagedAttentionKernel<<<dim3(num_active_slots, NUM_Q_HEADS), HEAD_DIM>>>(layer, num_active_slots, q_proj, kv_cache, block_table_gpu, gpu_seq_lens, gpu_active_slots, output);
}