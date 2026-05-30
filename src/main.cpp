// =============================================================================
// Tiny-vLLM main.cpp — 推理引擎主入口
// -----------------------------------------------------------------------------
// 职责：
//   1) 加载 Llama 3.2 1B-Instruct 模型权重（SafeTensors 格式）
//   2) 分配 GPU buffer（attention/FFN 中间结果、KV cache、block table）
//   3) 主推理循环：从 stdin 读 prompt → tokenize → prefill → decode 直到 EOS
//   4) Continuous batching：BATCH_SIZE 个 sequence 槽位，prefill 和 decode 混合调度
//
// 文件结构（按位置）：
//   L1-39    模型常量定义（Llama 3.2 1B 硬编码维度）
//   L41-63   GPU 信息打印
//   L65-78   Weights struct（所有权重的 GPU 指针）
//   L80-148  loadWeights() — SafeTensors 解析 + 一次性 cudaMemcpy
//   L150-554 prefill() — 处理一个 prompt，跑完整 16 层 transformer，生成第一个 token
//   L556-1042 main() — 启动 / KV cache 分配 / decode 主循环
//
// 依赖的 kernels（kernels.cu）：
//   prefill 用：embeddingGather / rmsNorm / rope / causalMask / softmax / silu / residualAdd
//   decode 用：embeddingGatherDecode / rmsNorm / ropeDecode / pagedAttention / silu / residualAdd
//   矩阵乘均通过 cublasGemmEx（不自己写 GEMM kernel，cublas 性能已经接近 peak）
// =============================================================================
#include <iostream>
#include <numeric>
#include <fstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <queue>
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "json.hpp"
#include "kernels.cuh"

using json = nlohmann::json;

// =============================================================================
// 模型常量（Llama 3.2 1B-Instruct 硬编码）
// -----------------------------------------------------------------------------
// 这些数字都来自 HuggingFace 上 Llama 3.2 1B 的 config.json，目前没参数化。
// 改其他模型必须重新编译并改这一段（这是 tiny-vllm 教学性质的明确取舍）。
// =============================================================================
constexpr int MAX_NEW_TOKENS_GENERATED = 20; // TODO: parameterize it with program arguments
constexpr int B_TO_MB = 1024 * 1024;
constexpr int B_TO_GB = 1024 * 1024 * 1024;
constexpr int N_LAYERS = 16; // TODO: hardcoded for llama 3.2 1B, just like any other value for now
constexpr int EMBEDDING_LENGTH = 2048;       // hidden dim, 即 d_model
constexpr int HIDDEN_DIM = 8192;             // FFN intermediate size，4× embedding（标准 Transformer 比例）
constexpr int KV_DIM = 512;                  // = NUM_K_HEADS × HEAD_DIM = 8 × 64
constexpr int HEAD_DIM = 64;                 // 每个 attention head 的维度
constexpr int NUM_Q_HEADS = 32;              // Query head 数（= EMBEDDING_LENGTH / HEAD_DIM = 2048/64）
constexpr int NUM_K_HEADS = 8;               // GQA：Key head 数 = Q heads / 4
constexpr int NUM_V_HEADS = 8;               // GQA：Value head 数（与 K head 一致）
constexpr int GQA_Q_TO_K_RATIO = 4;          // 32 Q heads 共享 8 KV heads，比例 4:1
constexpr int GQA_ATTN_SCORES_TO_V_RATIO = 4; // 同上，attention scores → V 的归约比例
constexpr int VOCAB_SIZE = 128256;           // Llama 3 词表（比 Llama 2 的 32k 大很多，BPE 多语言扩充）
constexpr int END_OF_TEXT_TOKEN_ID = 128001; // <|end_of_text|>
constexpr int EOT_ID_TOKEN_ID = 128009;      // <|eot_id|>，instruct 模型用的对话结束标记
constexpr int MAX_SEQ_LEN = 2048;            // TODO: make it tunable

// ---- Continuous batching 参数 ----
constexpr int BATCH_SIZE = 2;                // TODO: not even close to being good, it's just here to have batching
constexpr int MAX_PROMPT_LEN = 512;          // TODO: arbitrary, tunable
constexpr int MAX_BUFFER_SIZE = std::max(MAX_PROMPT_LEN, BATCH_SIZE);

// ---- PagedAttention 参数 ----
// BLOCK_SIZE = 16 token/页（vLLM 论文用 16），太小→寻址开销大，太大→内存碎片多
constexpr int BLOCK_SIZE = 16;                                    // TODO: tunable as well, defined the size of a single page in pagedattn
// 一个物理块的字节布局：[K (BLOCK_SIZE × KV_DIM × bf16) | V (同上)]
// V_OFFSET = 16 × 512 × 2 = 16384 字节 = 16KB（K 部分总大小）
constexpr int V_OFFSET = BLOCK_SIZE * KV_DIM * sizeof(__nv_bfloat16);
constexpr int BLOCK_BYTES = V_OFFSET * 2;                         // * 2 because K and V，总 32KB/块
constexpr size_t KV_CACHE_SIZE_BYTES = 2ULL * 1024 * 1024 * 1024; // TODO: 2GB
constexpr int MAX_BLOCKS_PER_SEQ = MAX_SEQ_LEN / BLOCK_SIZE;      // 2048 / 16 = 128
// 65536 个物理块 × 32KB = 2GB（与 KV_CACHE_SIZE_BYTES 一致），可容纳的 token 总量 = 65536 × 16 = 1M token
constexpr int NUM_BLOCKS = KV_CACHE_SIZE_BYTES / BLOCK_BYTES;     // 2*1024*1024*1024/(16*512*2*2) = 65536
constexpr int MAX_SEQUENCES = BATCH_SIZE;

// =============================================================================
// 学习反问（main.cpp 整体设计层面）
// ----- [Q] -----
//   [QM1] 为什么 KV cache 一开始就分配 2GB 固定大小？vLLM 不是按需扩展吗？
//   [QM2] BATCH_SIZE=2 这么小也算 batching 吗？典型生产配置是多少？
//   [QM3] BLOCK_SIZE=16 / NUM_BLOCKS=65536 看起来"够用"，但内存碎片在哪里？
// ----- [A]（答案在 main() 入口附近，搜索 [AM1]-[AM3]） -----
// =============================================================================

int checkGPUStatus()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0)
    {
        std::cerr << "No CUDA devices found\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Global memory: " << prop.totalGlobalMem / B_TO_MB << " MB\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    size_t free_mem;
    size_t total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Free memory: " << free_mem / B_TO_GB << "GB, total memory: " << total_mem / B_TO_GB << "GB\n";
    return 0;
}

// =============================================================================
// Weights — 模型权重在 GPU 上的访问入口
// -----------------------------------------------------------------------------
// 设计：所有权重共享一大块连续 GPU 显存（loadWeights 里一次性 cudaMalloc + memcpy），
// 这个 struct 只存指向不同位置的指针偏移。
//
// 这是 SafeTensors 设计的好处：tensor 在文件里是连续二进制，加载就是 host→device
// 的一次大块 memcpy，然后按 header 里的 offset 切出每个 tensor 的指针。
//
// 内存布局（按 N_LAYERS=16 计算）：
//   embed_tokens         [128256, 2048] bf16   ≈ 525MB（vocab × hidden）
//   per layer × 16:
//     input_layernorm    [2048]                ≈ 4KB
//     w_q                [2048, 2048]          ≈ 8MB
//     w_k                [512, 2048]           ≈ 2MB（GQA 4:1，所以 K 比 Q 小 4×）
//     w_v                [512, 2048]           ≈ 2MB
//     w_o                [2048, 2048]          ≈ 8MB
//     post_attn_layernorm[2048]                ≈ 4KB
//     mlp_gate_proj      [8192, 2048]          ≈ 32MB
//     mlp_up_proj        [8192, 2048]          ≈ 32MB
//     mlp_down_proj      [2048, 8192]          ≈ 32MB
//   norm                 [2048]                ≈ 4KB
//   总计 ~1.74GB（Llama 3.2 1B bf16 实际 model.safetensors 文件大小）
// =============================================================================
struct Weights
{
    __nv_bfloat16 *embed_tokens;
    __nv_bfloat16 *input_layernorm[N_LAYERS];
    __nv_bfloat16 *mlp_gate_proj[N_LAYERS];
    __nv_bfloat16 *mlp_up_proj[N_LAYERS];
    __nv_bfloat16 *mlp_down_proj[N_LAYERS];
    __nv_bfloat16 *post_attn_layernorms[N_LAYERS];
    __nv_bfloat16 *w_k[N_LAYERS];
    __nv_bfloat16 *w_o[N_LAYERS];
    __nv_bfloat16 *w_q[N_LAYERS];
    __nv_bfloat16 *w_v[N_LAYERS];
    __nv_bfloat16 *norm;
};

// =============================================================================
// loadWeights — 加载 model.safetensors 到 GPU
// -----------------------------------------------------------------------------
// SafeTensors 文件格式：
//   [0:8]               header_size (uint64, little-endian)
//   [8:8+header_size]   JSON header（描述每个 tensor 的 dtype / shape / offset）
//   [8+header_size:]    所有 tensor 的连续二进制数据
//
// JSON header 示例（部分）：
//   {
//     "model.embed_tokens.weight": {
//       "dtype": "BF16",
//       "shape": [128256, 2048],
//       "data_offsets": [0, 525336576]
//     },
//     "model.layers.0.self_attn.q_proj.weight": {
//       "dtype": "BF16", "shape": [2048, 2048],
//       "data_offsets": [525336576, 533725184]
//     },
//     ...
//   }
//
// 加载流程：
//   1) 读 header_size（前 8 字节）
//   2) 读 JSON header
//   3) 解析 JSON，记录每个 tensor 的 offset，找到 max_offset
//   4) 按 max_offset cudaMalloc 一大块 GPU 显存
//   5) 从文件读 max_offset 字节到 host vector
//   6) 一次 cudaMemcpy 整体上传到 GPU
//   7) 按 offset 算出每个 tensor 在 GPU 上的指针，存进 Weights struct
//
// ----- 学习反问（答案在 loadWeights 末尾） -----
//   [Q1] 为什么不用 mmap + cudaHostRegister 直接零拷贝加载？
//   [Q2] 如果 GPU 显存不够装下整个模型，能怎么改？
// =============================================================================
int loadWeights(Weights &weights)
{
    if (checkGPUStatus() != 0)
    {
        return 1;
    }

    // READ SAFETENSORS
    std::ifstream safetensors_file("model.safetensors", std::ios_base::binary); // TODO: use args to provide the path or smth
    if (!safetensors_file.is_open())
    {
        std::cout << "Can't open model.safetensors file\n";
        safetensors_file.close();
        return 1;
    }

    // READ SAFETENSORS HEADER SIZE
    // SafeTensors 标准：前 8 字节小端序的 uint64，记录后续 JSON header 的字节数
    uint64_t header_size;
    safetensors_file.read(reinterpret_cast<char *>(&header_size), 8);
    // READ SAFETENSORS HEADER
    std::string header;
    header.resize(header_size);
    safetensors_file.read(header.data(), header_size);
    // READ OFFSETS OF EVERY LAYER (TENSOR) TO KNOW WHERE EVERY LAYER STARTS AND ENDS IN THE MEMORY
    // 解析 JSON header，提取每个 tensor 的起始 offset；找最大 end offset 决定要分配多大显存
    std::unordered_map<std::string, uint64_t> offsets;
    json header_json = json::parse(header);
    uint64_t max_offset = 0;
    for (auto &[key, value] : header_json.items())
    {
        if (key == "__metadata__")
        {
            continue;
        }
        uint64_t offset_end = value["data_offsets"].at(1).get<uint64_t>();
        if (offset_end > max_offset)
        {
            max_offset = offset_end;
        }
        offsets[key] = value["data_offsets"].at(0).get<uint64_t>();
    }

    // 一次性分配整块模型显存。max_offset 是所有 tensor 的最右边界。
    // 注意：data_offsets 是相对于 binary 数据段（不是文件起点）的偏移，所以
    // GPU 这里直接用 max_offset 大小就够了。
    void *model_weights;
    cudaMalloc(&model_weights, max_offset); // max_offset tells where the model weights end in the memory

    // CPU 暂存：从文件读到 host 内存，再一次 cudaMemcpy 全量上传 GPU
    // 不做分块/流式：1.7GB 在 PCIe 4.0 x16 上 ~250ms 内完成，可以接受
    std::vector<char> model_weights_cpu;
    model_weights_cpu.resize(max_offset);
    safetensors_file.read(model_weights_cpu.data(), max_offset);

    cudaMemcpy(model_weights, model_weights_cpu.data(), max_offset, cudaMemcpyHostToDevice);
    safetensors_file.close();
    // BASICALLY A HELPER STRUCT TO HAVE AN EASY ACCESS TO ANY MODEL WEIGHTS ON GPU
    // TODO: right now I know the model structure since it's always llama 3.2 1B-Instruct, but maybe it would be convenient
    //       to store dimensions somewhere for even easier access?
    // 按 offset 算出每个 tensor 在 GPU 上的指针。所有指针都指向 model_weights 这块连续显存的不同位置
    weights.embed_tokens = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.embed_tokens.weight"));
    weights.norm = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.norm.weight"));
    for (int i = 0; i < N_LAYERS; ++i)
    {
        // HuggingFace 命名约定：model.layers.{i}.{component}.weight
        weights.input_layernorm[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".input_layernorm.weight"));
        weights.mlp_down_proj[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".mlp.down_proj.weight"));
        weights.mlp_gate_proj[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"));
        weights.mlp_up_proj[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".mlp.up_proj.weight"));
        weights.post_attn_layernorms[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"));
        weights.w_k[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".self_attn.k_proj.weight"));
        weights.w_o[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".self_attn.o_proj.weight"));
        weights.w_q[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".self_attn.q_proj.weight"));
        weights.w_v[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".self_attn.v_proj.weight"));
    }
    // ---------------------------------------------------------------------
    // 反问答案
    // [A1] 完全可以而且更高效。SafeTensors 的设计就是 mmap-friendly：tensor
    //      在文件里连续，按 dtype 自然对齐。生产实现（如 vLLM、HuggingFace
    //      transformers + accelerate）的做法：
    //        1) mmap 文件 → 拿到 CPU 虚拟地址
    //        2) cudaHostRegister 把 mmap 区域 pin 住（让 cudaMemcpyAsync 走 DMA）
    //        3) cudaMemcpyAsync 异步拷贝（甚至可以做按需分层加载）
    //      这样省去：a) std::vector<char> 中间拷贝（节省 1.7GB host 内存）
    //              b) 同步等待全量加载完成
    //      Tiny-vLLM 用同步 read + memcpy 是教学简洁优先。
    //
    // [A2] 几种渐进方案：
    //        a) 量化（bf16 → int8/int4）：1.7GB → 0.85GB / 0.43GB（参考 Lab 4 量化对比）
    //        b) 只把当前需要的 layer 权重放 GPU，其他流式加载（layer offloading）
    //        c) CPU offloading：weight 留 CPU pinned memory，每层 forward 前 prefetch
    //        d) 多 GPU tensor parallelism：weight 切片到多卡（vLLM tp 实现）
    //      Tiny-vLLM 假设单 GPU 显存够用，没做这些。Llama 3.2 1B bf16 在
    //      RTX 5090（32GB）上完全装得下。
    // ---------------------------------------------------------------------
    return 0;
}

// =============================================================================
// prefill — 处理一个 prompt，跑完整 16 层 transformer，生成第一个 token
// -----------------------------------------------------------------------------
// 这是 LLM 推理的"预填充"阶段：把整个 prompt 的所有 token 一次性送进模型。
// 与 decode 的本质差异：
//   prefill：N 个 token 并行，可以打满 GPU 算力（compute-bound）
//   decode：每次只生成 1 个 token，受 KV cache 加载带宽限制（memory-bound）
//
// ----- 完整数据流 -----
//   prompt (N tokens)
//     │
//     ▼ embeddingGather
//   hidden_state [N, 2048]
//     │
//     ├─── for layer 0..15: ─────────────────────────────────────────────┐
//     │   rmsNorm                                                          │
//     │   Q, K, V 投影 (cublas)            ← Q [N,2048]  K/V [N,512]      │
//     │   RoPE on Q and K                                                  │
//     │   把 K, V 写入 PagedAttention 物理块（per-layer per-slot）         │
//     │   per-head loop × 32:                                              │
//     │     attention_scores[h] = Q_h × K_h^T (cublas)  ← [N,N]            │
//     │   causalMask + softmax                                             │
//     │   per-head loop × 32:                                              │
//     │     attn_v[h] = scores[h] × V_h (cublas)        ← [N,64]           │
//     │   O_proj (cublas) + residualAdd                                    │
//     │   rmsNorm (post-attn)                                              │
//     │   gate, up 投影 (cublas)            ← [N,8192]                     │
//     │   silu(gate) ⊙ up                                                 │
//     │   down 投影 (cublas) + residualAdd                                 │
//     │                                                                    │
//     ▼ ────────────────────────────────────────────────────────────────┘
//   final rmsNorm
//     │
//     ▼ cublas (× embed_tokens^T)
//   logits [N, 128256]
//     │
//     ▼ argmax on last token only
//   max_token_idx → generated_tokens[slot]
//     │
//     ▼
//   sync block_table to GPU（让后续 decode 能用）
//
// ----- 巨型参数列表的来源 -----
// 这个函数有 50+ 个参数，是因为：
//   1) 没用 OOP（Engine class）封装上下文，所有 GPU buffer 都从 main() 传入
//   2) 引用传递让 prefill 能修改 buffer 指针（如 q_proj、o_proj 等会被指到不同 buf）
//   3) cublas alpha/beta 标量也按引用传，避免每次 cublasGemmEx 都构造临时变量
// 作者自己也吐槽（见函数声明上方 TODO 注释）。生产实现会封装 Engine class，
// 在 ctor 里分配并持有所有 buffer。
//
// ----- cublas 列主序 trick（重要：贯穿整个函数）-----
// PyTorch / 我们的代码：行主序（row-major），矩阵 A[m,n] 内存布局是 m 行 × n 列连续
// cuBLAS：列主序（column-major），它把同一片内存解读为 n 行 × m 列
// 数学等价：cublas 看到的 = 我们矩阵的转置
//
// trick：要算 C = A × B（A,B,C 都是行主序）
//   等价于  C^T = B^T × A^T  （转置律）
//   cublas 看到的 A,B,C 已经是 A^T, B^T, C^T，所以直接调用 cublas(B, A) → 输出
//   这块内存 cublas 看作 C^T，我们读作 C，无需显式 transpose
// 这就是为什么下面所有 cublasGemmEx 都把 weight 当第一参数（CUBLAS_OP_T）、
// activation 当第二参数（CUBLAS_OP_N）—— activation 已经"被 cublas 看作转置"。
//
// ----- 学习反问（答案在函数末尾 [AP1]-[AP5] 区块）-----
//   [QP1] 为什么 attention 算 32 个 head 用 for loop 而不是 batched gemm？
//   [QP2] 为什么 q_proj 指向 buf_2048_1，o_proj 指向 buf_2048_2？为什么要两个 buffer？
//   [QP3] argmax 在 CPU 上做（cudaMemcpy 整个 logits 回 CPU）有多大代价？
//   [QP4] block_table 末尾整体 cudaMemcpy 同步到 GPU，效率低在哪？
//   [QP5] 这个 prefill 实现什么时候会是 memory-bound、什么时候 compute-bound？
// =============================================================================
// TODO: clean up this mess lol XD (I mean, the arguments list is so long, but maybe that's unavoidable, I don't know yet)
void prefill(std::vector<int> &prompt, std::queue<std::vector<int>> &queue, int &prompt_len, std::vector<bool> &is_slot_free, int slot, int *gpu_input_tokens, nv_bfloat16 *input_embeddings, Weights &weights, nv_bfloat16 *hidden_state, nv_bfloat16 *rms_norms, nv_bfloat16 *&q_proj, nv_bfloat16 *buf_2048_1, cublasHandle_t cublas_handle, float &q_proj_alpha, float &q_proj_beta, float &k_proj_alpha, float &k_proj_beta, float &v_proj_alpha, float &v_proj_beta, nv_bfloat16 *prefill_attn_scores, float &attn_alpha, float &attn_beta, nv_bfloat16 *&attn_scores_v, float &attn_scores_v_alpha, float &attn_scores_v_beta, nv_bfloat16 *&o_proj, nv_bfloat16 *buf_2048_2, float &o_proj_alpha, float &o_proj_beta, float &gate_alpha, float &gate_beta, nv_bfloat16 *gate, float &up_alpha, float &up_beta, nv_bfloat16 *up, nv_bfloat16 *&down, float &down_alpha, float &down_beta, float &embed_alpha, float &embed_beta, nv_bfloat16 *embed_proj, std::vector<nv_bfloat16> &embed_proj_cpu, std::vector<std::vector<int>> &generated_tokens, std::vector<int> &last_generated_tokens, std::vector<int> &current_prompt_len, __nv_bfloat16 *k_proj_temp_buf, __nv_bfloat16 *v_proj_temp_buf, std::vector<int> &block_table, int *block_table_gpu, std::vector<int> &free_blocks, __nv_bfloat16 *kv_cache)
{
    // ---- Stage 1: 从队列取 prompt → 上传 token id 到 GPU → embedding lookup ----
    prompt = queue.front();
    prompt_len = prompt.size();
    queue.pop();
    is_slot_free[slot] = false;  // 标记这个 slot 已被占用

    // prompt token ids（CPU vector）→ GPU int 数组
    cudaMemcpy(gpu_input_tokens, prompt.data(), prompt_len * sizeof(int), cudaMemcpyHostToDevice);
    // tokens [N] → embeddings [N, 2048]
    embeddingGather(gpu_input_tokens, input_embeddings, weights.embed_tokens, prompt_len);

    // 复制 input_embeddings → hidden_state，因为：
    //   - 后续 rmsNorm 是 hidden_state → rms_norms 的"读 hidden 写 norm"，不会破坏 hidden_state
    //   - residualAdd 需要 hidden_state（保留原值）+ attn_output → 新 hidden_state
    //   - 如果只用一份 buffer，rmsNorm 之后原始 embedding 就没了，residual 用不上
    cudaMemcpy(hidden_state,
               input_embeddings,
               prompt_len * EMBEDDING_LENGTH * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToDevice);
    // ---- Stage 2: 16 层 transformer 主循环 ----
    for (int layer = 0; layer < N_LAYERS; ++layer)
    {
        // ---- 2.1 input rmsNorm: hidden_state → rms_norms ----
        // 注意：保留 hidden_state 不动，rmsNorm 输出到独立的 rms_norms buffer
        rmsNorm(hidden_state, rms_norms, weights.input_layernorm[layer], prompt_len);

        // ---- 2.2 Q / K / V 投影 ----
        // 数学：Q = rms_norms × W_q^T，dim = (N, 2048) × (2048, 2048) → (N, 2048)
        //       K = rms_norms × W_k^T，dim = (N, 2048) × (2048, 512)  → (N, 512)
        //       V = rms_norms × W_v^T，dim = (N, 2048) × (2048, 512)  → (N, 512)
        // GQA：K/V dim 是 Q 的 1/4（512 = 8 KV heads × 64，Q 是 32 heads × 64）
        // Q = inputs * wq^T; my matrices are row-major, cublas expects column-major
        // it perceives my matrices as transposed
        // there's a trick where C = A * B == C^T = B^T * A^T
        // so in my scenario cublas sees now: Q = inputs^T * wq^T^T = inputs ^T * wq
        // so I need to do: Q^T = wq ^T * inputs
        // the beauty is that we don't need to transpose Q^T back to Q
        // because cublas sees the output as column-major
        // so it's in fact transposed
        // final dim (num_tok, EMBEDDING_LENGTH)
        q_proj = buf_2048_1;  // 复用 buf_2048_1 作为 Q 投影输出
        cublasStatus_t q_proj_status = cublasGemmEx(cublas_handle,
                                                    CUBLAS_OP_T,           // weight 转置（与列主序解读一起做了"两次转置 = 不转置"）
                                                    CUBLAS_OP_N,           // activation 不转置（cublas 看到的就是我们的转置）
                                                    EMBEDDING_LENGTH,      // m = 输出 col 数（行主序看的话就是行数）= 2048
                                                    prompt_len,            // n = 输出 row 数 = N
                                                    EMBEDDING_LENGTH,      // k = 共享维度 = 2048
                                                    &q_proj_alpha,         // = 1.0
                                                    weights.w_q[layer],    // [2048, 2048] bf16
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,      // lda
                                                    rms_norms,             // [N, 2048]
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,      // ldb
                                                    &q_proj_beta,          // = 0.0
                                                    q_proj,                // 输出 [N, 2048]
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,      // ldc
                                                    CUBLAS_COMPUTE_32F,    // 用 fp32 累加（避免 bf16 累加精度损失）
                                                    CUBLAS_GEMM_DEFAULT);

        // K 投影：与 Q 同模式，但输出维度变 KV_DIM=512（GQA 4×↓）
        // input = (num_tokens, EMBEDDING_LENGTH), weights = (KV_DIM, EMBEDDING_LENGTH)
        // after trick: (KV_DIM, EMBEDDING_LENGTH) * (EMBEDDING_LENGTH, num_tokens) -> (KV_DIM, num_tokens), which really is (num_tok, KV_DIM)
        // lda: EMBEDDING_LENGTH, ldb: EMBEDDING_LENGTH, ldc: KV_DIM
        cublasStatus_t k_proj_status = cublasGemmEx(cublas_handle,
                                                    CUBLAS_OP_T,
                                                    CUBLAS_OP_N,
                                                    KV_DIM,                // 输出 col = 512
                                                    prompt_len,
                                                    EMBEDDING_LENGTH,
                                                    &k_proj_alpha,
                                                    weights.w_k[layer],    // [512, 2048] bf16
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    rms_norms,
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    &k_proj_beta,
                                                    k_proj_temp_buf,       // K 临时 buffer [N, 512]，待 RoPE 后写 KV cache
                                                    CUDA_R_16BF,
                                                    KV_DIM,
                                                    CUBLAS_COMPUTE_32F,
                                                    CUBLAS_GEMM_DEFAULT);

        // V 投影：与 K 同模式
        // same as K projection
        cublasStatus_t v_proj_status = cublasGemmEx(cublas_handle,
                                                    CUBLAS_OP_T,
                                                    CUBLAS_OP_N,
                                                    KV_DIM,
                                                    prompt_len,
                                                    EMBEDDING_LENGTH,
                                                    &v_proj_alpha,
                                                    weights.w_v[layer],
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    rms_norms,
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    &v_proj_beta,
                                                    v_proj_temp_buf,       // V 临时 buffer [N, 512]，待写 KV cache
                                                    CUDA_R_16BF,
                                                    KV_DIM,
                                                    CUBLAS_COMPUTE_32F,
                                                    CUBLAS_GEMM_DEFAULT);

        // ---- 2.3 RoPE on Q and K（V 不需要 RoPE，见 kernels.cu ropeKernel 反问 [A1]）----
        // RoPE now

        rope(q_proj, prompt_len, EMBEDDING_LENGTH);          // Q：proj_dim=2048，每 token 32 个 head 各做旋转
        rope(k_proj_temp_buf, prompt_len, KV_DIM);            // K：proj_dim=512，每 token 8 个 head 各做旋转

        // ---- 2.4 把 K, V 写入 PagedAttention 物理块 ----
        // PagedAttention - scatter K and V into blocks
        // slot - index within batch
        // layer - index of layer
        // ceil(prompt_len/BLOCK_SIZE) = number of blocks needed to allocate in block table
        //
        // 这里是 prefill 与 decode 分歧的"关键写入点"：
        //   prefill 把 [N, 512] 的 K/V 按 BLOCK_SIZE=16 切片，每片占用一个物理块
        //   decode 时 pagedAttention kernel 内部按 block_table 索引读这些物理块
        //
        // 物理块布局（per block，BLOCK_BYTES=32KB）：
        //   [K: BLOCK_SIZE × KV_DIM × bf16 = 16 × 512 × 2 = 16KB][V: 同上]
        //
        // block_table 三维：[slot, layer, logical_block_idx] → physical_block_id
        for (int token_idx = 0; token_idx < prompt_len; token_idx += BLOCK_SIZE)
        {
            // 这个 block 实际写入的 token 数（最后一块可能不满 16）
            int num_tokens_to_copy = prompt_len - token_idx;
            if (num_tokens_to_copy > BLOCK_SIZE)
            {
                num_tokens_to_copy = BLOCK_SIZE;
            }
            // ---- 物理块分配协议 ----
            // 1) 看 block_table[slot][layer][logical_block_idx] 是不是 -1（未分配）
            // 2) 是 -1 → 从 free_blocks 栈顶取一个物理块编号，写回 block_table
            // 3) 不是 -1 → 此块已分配（prefill 不该出现这种情况，否则 assert）
            // read index of physical block from logical block_table
            // if -1, then need to allocate the new block
            // pop from free_blocks
            // write its value to block_table on the same position we read from
            // compute address of this block table in kv_cache
            // write tokens to it
            int block_idx = token_idx / BLOCK_SIZE;
            int block = block_table[slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + block_idx];
            if (block == -1)
            {
                // free_blocks 是 LIFO 栈（vector + back/pop_back）
                // 设计上没有"按访问局部性优化"，物理块是随机散布在 KV cache 大池子里
                int physical_block_idx = free_blocks.back();
                free_blocks.pop_back();
                block = physical_block_idx;
                block_table[slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + block_idx] = block;
            }
            else
            {
                // prefill 阶段 block 不可能是已分配状态：每个 prompt 都是新进来的
                // 这个 assert 是防御性编程，理论上不会触发
                assert(false && "block must be -1 during prefill - what happened?");
                // probably in prefill this doesn't make a lot of sense? but will matter in decode
            }

            // ---- 把 K 写入物理块的 K 部分 ----
            // store K
            __nv_bfloat16 *k_cache_ptr = (__nv_bfloat16 *)((char *)kv_cache + block * BLOCK_BYTES);
            __nv_bfloat16 *k_proj_ptr = k_proj_temp_buf + token_idx * KV_DIM;
            cudaMemcpy(k_cache_ptr, k_proj_ptr, num_tokens_to_copy * KV_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);

            // ---- 把 V 写入物理块的 V 部分（V_OFFSET 之后）----
            // store V
            __nv_bfloat16 *v_cache_ptr = (__nv_bfloat16 *)((char *)kv_cache + block * BLOCK_BYTES + V_OFFSET);
            __nv_bfloat16 *v_proj_ptr = v_proj_temp_buf + token_idx * KV_DIM;
            cudaMemcpy(v_cache_ptr, v_proj_ptr, num_tokens_to_copy * KV_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
        }

        // ---- 2.5 Attention scores: per-head QK^T 计算 ----
        // 数学：score_h = Q_h × K_h^T / sqrt(64)，dim (N, 64) × (64, N) = (N, N)
        // 全部 32 个 head 输出形状 (32, N, N)
        //
        // ⚠️ 这里用 for loop 顺序调用 32 次 cublasGemmEx，每次小 GEMM
        // 反问 [QP1] 答案：用 cublasGemmStridedBatchedEx 一次 launch 处理 32 head
        // 性能更好，但教学版用 loop 让每个 head 的内存计算清晰可见
        // attention scores
        // per head, 64 elements each
        // so total 32 heads
        // Q (num_tok, 2048)
        // K (num_tok, 512)
        // GQA grouping reuses 1 K head per 4 consecutive Q heads
        // Q_head (num_tok, 64)
        // K_head (num_tok, 64)
        // attn_score_head = Q_head * K_head^T / sqrt(64)
        // so: head output dims (num_tok, num_tok)
        // total output (32, num_tok, num_tok)
        for (int i = 0; i < NUM_Q_HEADS; ++i)
        {
            // GQA：4 个 Q head 共享 1 个 K head（i=0,1,2,3 → k_head=0；i=4,5,6,7 → k_head=1...）
            int k_head_idx = i / GQA_Q_TO_K_RATIO;
            // Q_head 在 q_proj 中的偏移：每个 head 64 维，依次排列
            // q_proj 形状 [N, 2048] = [N, 32 heads × 64]，head i 起点 = i * 64
            __nv_bfloat16 *q_head = q_proj + i * HEAD_DIM;
            __nv_bfloat16 *k_head = k_proj_temp_buf + k_head_idx * HEAD_DIM;
            // attn_scores 形状 [32, N, N]，head i 的 [N,N] 子矩阵起点
            __nv_bfloat16 *attn_score_head = prefill_attn_scores + prompt_len * prompt_len * i;

            // 注意 lda/ldb：q_head/k_head 在大 buffer 里，stride 不是 HEAD_DIM 而是
            //   q_proj 的 stride = EMBEDDING_LENGTH = 2048 (32 head 拼接)
            //   k_proj_temp_buf 的 stride = KV_DIM = 512 (8 head 拼接)
            // attn_alpha = 1/sqrt(64) = 0.125（在 main 里设置），cublas 直接乘进去
            cublasStatus_t attn_score_status = cublasGemmEx(cublas_handle,
                                                            CUBLAS_OP_T,
                                                            CUBLAS_OP_N,
                                                            prompt_len,         // m = N (输出行)
                                                            prompt_len,         // n = N (输出列)
                                                            HEAD_DIM,           // k = 64 (HEAD_DIM)
                                                            &attn_alpha,        // = 1/sqrt(64) ≈ 0.125
                                                            k_head,
                                                            CUDA_R_16BF,
                                                            KV_DIM,             // lda = K 大 buffer 的 stride
                                                            q_head,
                                                            CUDA_R_16BF,
                                                            EMBEDDING_LENGTH,   // ldb = Q 大 buffer 的 stride
                                                            &attn_beta,         // = 0
                                                            attn_score_head,
                                                            CUDA_R_16BF,
                                                            prompt_len,         // ldc = 输出 stride
                                                            CUBLAS_COMPUTE_32F,
                                                            CUBLAS_GEMM_DEFAULT);
        }

        // ---- 2.6 Causal mask + softmax ----
        // 对 [32, N, N] 的 scores 矩阵做：
        //   1) causalMask：上三角置 -inf（每个 query 只能 attend 到自己和之前 token）
        //   2) softmax：每行归一化为概率分布
        // 详见 kernels.cu causalMaskKernel / softmaxKernel
        causalMask(prefill_attn_scores, prompt_len);

        softmax(prefill_attn_scores, prompt_len);

        // ---- 2.7 attn_scores × V：per-head 加权累加 ----
        // 数学：output_h = scores_h × V_h，dim (N, N) × (N, 64) = (N, 64)
        // 全部 32 个 head 拼成 (N, 32 × 64) = (N, 2048)，作为 O 投影的输入
        //
        // GQA：4 个 Q head（也就是 4 个 attn_scores head）共享 1 个 V head
        // attn scores * V
        // (32, num_tok, num_tok) * (num_tok, 512)
        // GQA - 4 Q heads share 1 V head
        // attn_scores dim (32, num_tok, num_tok)
        // attn_scores head dim (num_tok, num_tok)
        // V dim (num_tok, 512)
        // NUM_V_HEADS is 8 -> 512 / 8 = 64
        // V_head dim (num_tok, 64)
        // output head dim: scores head * V head -> (num_tok, num_tok) * (num_tok, 64) = (num_tok, 64)
        // in total 32 output heads: so (num_tok, 64 * 32) = (num_tok, 2048)
        attn_scores_v = buf_2048_1;  // 复用 buf_2048_1（之前装 q_proj，现在装 attn_v 输出，前者用完了）
        for (int i = 0; i < NUM_Q_HEADS; ++i)
        {
            int v_head_idx = i / GQA_ATTN_SCORES_TO_V_RATIO;
            // i * prompt_under_prefill.size() * prompt_under_prefill.size(),  because attn scores is (32, num_tok, num_tok)
            __nv_bfloat16 *attn_scores_head = prefill_attn_scores + i * prompt_len * prompt_len;
            __nv_bfloat16 *v_head = v_proj_temp_buf + v_head_idx * HEAD_DIM;
            __nv_bfloat16 *output_attn_scores_head = attn_scores_v + i * HEAD_DIM;

            cublasStatus_t attn_score_status = cublasGemmEx(cublas_handle,
                                                            CUBLAS_OP_N,            // 注意：这里 V 不转置（与上面 K 转置不同）
                                                            CUBLAS_OP_N,
                                                            HEAD_DIM,               // m = 64
                                                            prompt_len,             // n = N
                                                            prompt_len,             // k = N (scores 矩阵的列数 = K row 数)
                                                            &attn_scores_v_alpha,
                                                            v_head,
                                                            CUDA_R_16BF,
                                                            KV_DIM,                 // lda = V 大 buffer stride
                                                            attn_scores_head,
                                                            CUDA_R_16BF,
                                                            prompt_len,
                                                            &attn_scores_v_beta,
                                                            output_attn_scores_head,
                                                            CUDA_R_16BF,
                                                            EMBEDDING_LENGTH,       // ldc = 输出大 buffer stride（32 head 拼接）
                                                            CUBLAS_COMPUTE_32F,
                                                            CUBLAS_GEMM_DEFAULT);
        }

        // ---- 2.8 O 投影：把 attention output 投回 hidden_size ----
        // 数学：o_proj = attn_v × W_o^T，dim (N, 2048) × (2048, 2048) → (N, 2048)
        // 与 Q 投影同模式
        // output projection, it will be an input for MLP blocks
        // attn_scores_v * w_o^T
        // (num_tok, 2048) * (2048, 2048) -> (num_tok, 2048)
        // same as Q projection, so copy paste
        o_proj = buf_2048_2;  // 用 buf_2048_2 不复用 buf_2048_1，因为 attn_scores_v 还在用
        cublasStatus_t o_proj_status = cublasGemmEx(cublas_handle,
                                                    CUBLAS_OP_T,
                                                    CUBLAS_OP_N,
                                                    EMBEDDING_LENGTH,
                                                    prompt_len,
                                                    EMBEDDING_LENGTH,
                                                    &o_proj_alpha,
                                                    weights.w_o[layer],
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    attn_scores_v,
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    &o_proj_beta,
                                                    o_proj,
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    CUBLAS_COMPUTE_32F,
                                                    CUBLAS_GEMM_DEFAULT);

        // ---- 2.9 Residual + post-attn rmsNorm ----
        // hidden_state = hidden_state + o_proj（attention residual）
        // 注意 hidden_state 现在变成"attention 输出加 residual"，但还没经 FFN
        // (num_tok, 2048) + (num_tok, 2048) -> (num_tok, 2048)
        residualAdd(hidden_state, o_proj, prompt_len);
        // post attention RMS Norm
        // 把 residual 后的 hidden 再 norm 一次，作为 FFN 输入
        rmsNorm(hidden_state, rms_norms, weights.post_attn_layernorms[layer], prompt_len);

        // ---- 2.10 FFN: SwiGLU = SiLU(gate) ⊙ up，再过 down ----
        // 三个 GEMM + 一个 elementwise kernel：
        //   gate = rms_norms × W_gate^T  → (N, 8192)
        //   up   = rms_norms × W_up^T    → (N, 8192)
        //   gate = SiLU(gate) ⊙ up        in-place（kernels.cu siluKernel）
        //   down = gate × W_down^T        → (N, 2048)
        //
        // FFN 比 attention 更占 FLOPs（3 个 8192×2048 GEMM vs attention 4 个 2048×2048）
        // SwiGLU time - just MLP + SiLU
        // gate = hidden_state (rms-normed) * mlp_gate_proj ^ T
        // HIDDEN_DIM = 8192
        // (num_tok, 2048) * (2048, 8192) -> (num_tok, 8192)
        // my data is row major so transpose trick
        // gate ^T = (mlp_gate_proj ^ T)^T * hidden_state^T
        // gate ^T = mlp_gate_proj * hidden_state^T
        // (num_tok, 8192)^T = (8192, 2048) * (2048, num_tok)
        // but data is perceived as column major so I need to transpose mlp_gate_proj
        // to make it work
        // m 8192 n num_tok k 2048 lda 2048 ldb 2048 ldc 8192
        cublasStatus_t gate_status = cublasGemmEx(cublas_handle,
                                                  CUBLAS_OP_T,
                                                  CUBLAS_OP_N,
                                                  HIDDEN_DIM,                      // m = 8192 (输出维度)
                                                  prompt_len,
                                                  EMBEDDING_LENGTH,
                                                  &gate_alpha,
                                                  weights.mlp_gate_proj[layer],    // [8192, 2048]
                                                  CUDA_R_16BF,
                                                  EMBEDDING_LENGTH,
                                                  rms_norms,
                                                  CUDA_R_16BF,
                                                  EMBEDDING_LENGTH,
                                                  &gate_beta,
                                                  gate,                            // 输出 [N, 8192]
                                                  CUDA_R_16BF,
                                                  HIDDEN_DIM,
                                                  CUBLAS_COMPUTE_32F,
                                                  CUBLAS_GEMM_DEFAULT);

        // up 投影：与 gate 同模式，权重不同
        // up, the same dims as gate
        cublasStatus_t up_status = cublasGemmEx(cublas_handle,
                                                CUBLAS_OP_T,
                                                CUBLAS_OP_N,
                                                HIDDEN_DIM,
                                                prompt_len,
                                                EMBEDDING_LENGTH,
                                                &up_alpha,
                                                weights.mlp_up_proj[layer],
                                                CUDA_R_16BF,
                                                EMBEDDING_LENGTH,
                                                rms_norms,
                                                CUDA_R_16BF,
                                                EMBEDDING_LENGTH,
                                                &up_beta,
                                                up,
                                                CUDA_R_16BF,
                                                HIDDEN_DIM,
                                                CUBLAS_COMPUTE_32F,
                                                CUBLAS_GEMM_DEFAULT);

        // SiLU(gate) ⊙ up，结果 in-place 写回 gate
        // 详见 kernels.cu siluKernel
        // SiLU
        // after_silu = SiLU(gate) * up (element-wise multication)
        // after_silu = gate * (1 / (1 + e^(-gate))) * up
        // gate is dim (num_tok, 8192), up too
        silu(gate, up, prompt_len); // gate = after_silu now

        // ---- down 投影：把 8192 维投回 2048 hidden ----
        // 数学：down = gate × W_down^T，dim (N, 8192) × (8192, 2048) → (N, 2048)
        // down projection
        // output = post-silu * down_proj^T
        // dims: (num_tok, 8192) * (2048, 8192) ^ T = (num_tok, 8192) * (8192, 2048) = (num_tok, 2048)
        // output^T = (down_proj^T)^T * post-silu^T
        // output^T = down_proj * post-silu^T
        // cublas sees them already as transposed so only down_proj I need to transpose
        // dims = (2048, 8192) * (8192, num_tok) = (2048, num_tok)
        // m: 2048 n: num_tok, k: 8192
        // lda: 8192, ldb: 8192, ldc: 2048
        down = buf_2048_2;  // 复用 buf_2048_2（之前装 o_proj 已用完）
        cublasStatus_t down_status = cublasGemmEx(cublas_handle,
                                                  CUBLAS_OP_T,
                                                  CUBLAS_OP_N,
                                                  EMBEDDING_LENGTH,
                                                  prompt_len,
                                                  HIDDEN_DIM,
                                                  &down_alpha,
                                                  weights.mlp_down_proj[layer],
                                                  CUDA_R_16BF,
                                                  HIDDEN_DIM,
                                                  gate,
                                                  CUDA_R_16BF,
                                                  HIDDEN_DIM,
                                                  &down_beta,
                                                  down,
                                                  CUDA_R_16BF,
                                                  EMBEDDING_LENGTH,
                                                  CUBLAS_COMPUTE_32F,
                                                  CUBLAS_GEMM_DEFAULT);

        // ---- 2.11 FFN residual + 进入下一层 ----
        // hidden_state = hidden_state + down，至此一层 transformer 完成
        // (num_tok, 2048) + (num_tok, 2048) -> (num_tok, 2048)
        residualAdd(hidden_state, down, prompt_len);
    }
    // ---- Stage 3: Final rmsNorm（所有 16 层结束后的最后归一化）----
    rmsNorm(hidden_state, rms_norms, weights.norm, prompt_len);

    // ---- Stage 4: LM head: hidden → logits over vocab ----
    // 数学：logits = rms_norms × embed_tokens^T，(N, 2048) × (2048, 128256) → (N, 128256)
    // weight tying：复用 embedding 矩阵作为 LM head（Llama 没有独立的 lm_head weight）
    // logits = rms_norms * weights.embed_tokens^T
    // dim rms_norms: (num_tok, 2048), dim embed_tokens: (128256, 2048)
    // logits dim = (num_tok, 2048) * (2048, 128256) = (num_tok, 128256) => m = num_tok, n = 128256, k = 2048
    // I leave this comment above because it shows a bug in my thinking
    // because I use the cublas trick, logits are transposed so m and n should be swapped
    // so m 128256, n num_tok
    // data is row major so we treat it as transposed and use the trick
    // logits^T = ((weights.embed_tokens^T)^T * rms_norms^T
    // logits^T = weights.embed_tokens * rms_norms^T
    // so we need to transpose embed_tokens, because rms_norms already
    // appears to cublas as transposed
    // lda = 2048, ldb = 2048, ldc = 128256

    cublasStatus_t embed_status = cublasGemmEx(cublas_handle,
                                               CUBLAS_OP_T,
                                               CUBLAS_OP_N,
                                               VOCAB_SIZE,             // 128256（Llama 3 词表）
                                               prompt_len,
                                               EMBEDDING_LENGTH,
                                               &embed_alpha,
                                               weights.embed_tokens,   // weight tying：与 embedding 共享权重
                                               CUDA_R_16BF,
                                               EMBEDDING_LENGTH,
                                               rms_norms,
                                               CUDA_R_16BF,
                                               EMBEDDING_LENGTH,
                                               &embed_beta,
                                               embed_proj,             // 输出 [N, 128256] = logits
                                               CUDA_R_16BF,
                                               VOCAB_SIZE,
                                               CUBLAS_COMPUTE_32F,
                                               CUBLAS_GEMM_DEFAULT);

    // ---- Stage 5: Argmax（CPU 上做，不优雅但简洁）----
    // 拷整个 logits [N, 128256] 回 CPU 是浪费——其实只需要最后一个 token 的
    // [128256] 那一行，因为只有最后 token 用来生成下一个。
    // 这是一个明显的优化点（反问 [QP3] 答案）：
    //   1) 只 cudaMemcpy 最后一个 token 的 128256 个 logits（256KB 而不是 N×256KB）
    //   2) 或者 GPU 上写一个 argmax kernel，只回传 1 个 int
    cudaMemcpy(embed_proj_cpu.data(), embed_proj, sizeof(__nv_bfloat16) * prompt_len * VOCAB_SIZE, cudaMemcpyDeviceToHost);
    // argmax to get the output token
    // TODO: write a proper kernel for it
    // for now just a simple CPU function
    int last_token_offset = (prompt_len - 1) * VOCAB_SIZE;  // 只看最后一个 token 的 logits
    float max_token = (float)embed_proj_cpu[last_token_offset];
    int max_token_idx = 0;
    for (int token_idx = 0; token_idx < VOCAB_SIZE; ++token_idx)
    {
        if ((float)embed_proj_cpu[token_idx + last_token_offset] > max_token)
        {
            max_token = embed_proj_cpu[token_idx + last_token_offset];
            max_token_idx = token_idx;
        }
    }
    std::cout << "Output token: " << (float)max_token << ", token index: " << std::to_string(max_token_idx) << std::endl;

    // ---- Stage 6: 把生成的 token 记录到 batch 状态 ----
    generated_tokens[slot].push_back(max_token_idx);     // 该 slot 累计生成的 token list
    last_generated_tokens[slot] = max_token_idx;          // decode 下一轮的 input
    current_prompt_len[slot] = prompt_len;                // 该 slot 当前 KV cache 占用长度

    // ---- Stage 7: block_table CPU → GPU 同步 ----
    // 因为 prefill 的 block 分配是在 CPU vector block_table 上做的（用 free_blocks
    // 栈分配），但 decode 时 pagedAttention kernel 读 GPU 上的 block_table_gpu。
    // 所以 prefill 末尾必须把整张 block_table 同步到 GPU。
    //
    // ⚠️ 这里整张表都拷贝（MAX_SEQUENCES × N_LAYERS × MAX_BLOCKS_PER_SEQ × 4B
    // = 2 × 16 × 128 × 4 = 16KB），即使只有这个 slot 的部分变了。
    // 反问 [QP4] 答案：可以只拷该 slot 的子矩阵 N_LAYERS × MAX_BLOCKS_PER_SEQ
    // = 16 × 128 × 4 = 8KB，省一半。但因为表本来就很小（< 64KB），优化不紧迫。
    //
    // 真正的瓶颈是这是个 host→device 同步拷贝，prefill 必须等它完成才能返回。
    // 优化方向：在 GPU 上直接维护 block_table，free_blocks 也用 GPU lock-free 队列
    // synchronize state of block_table with block_table_gpu
    // TODO: do it more clever and not copy full table unnecessarily
    cudaMemcpy(block_table_gpu, block_table.data(), MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ * sizeof(int), cudaMemcpyHostToDevice);

    // =====================================================================
    // 反问答案（对应函数顶部 [QP1]-[QP5]）
    // ---------------------------------------------------------------------
    // [AP1] for loop 32 次小 GEMM 是教学清晰但性能次优。优化层级：
    //         a) cublasGemmStridedBatchedEx：一次 launch 处理 32 head，
    //            cublas 内部自动 batch dispatch。GQA 需要 broadcast K/V（4 Q : 1 K），
    //            可以用 stride=0 的技巧让 4 个 Q head 共用同一份 K head
    //         b) 直接用 FlashAttention：Q×K^T + softmax + ×V 三步融成一个 kernel，
    //            还省了 [N,N] scores 矩阵的 GMEM 写入和读出（这是 prefill 长 prompt 的
    //            主要瓶颈）
    //         c) FA-2 / xformers / TransformerEngine 都是这条路线
    //
    // [AP2] q_proj 和 o_proj 是 transformer 一层中"还需要保留的中间结果"：
    //         - Q 算完后要做 RoPE（in-place）+ 喂给 attention scores
    //         - O 算完后要 residualAdd（读 hidden + o_proj 算和）
    //       attention scores 和 attn_v 用同一个 buf_2048_1 是因为 attn_v 阶段
    //       Q 已经"消化"完了（K 已写 KV cache，scores 也算完）。
    //       buffer 调度很微妙——这就是为什么生产 Engine 会把这些封装成 BufferPool。
    //       Tiny-vLLM 用 2 个交替 buf 是最小化 GPU 显存占用的取舍。
    //
    // [AP3] 拷整个 [N, 128256] logits 回 CPU 在长 prompt 时浪费明显：
    //         - prompt_len=512 时 logits = 512 × 128256 × 2B = 128MB 回拷
    //         - 真正用的只有最后一行 128256 × 2B = 256KB
    //       PCIe 4.0 x16 单向 32GB/s，128MB → 4ms 的延迟，不可忽略。
    //       优化：
    //         a) cudaMemcpy 时只拷 last_token_offset 起的 256KB
    //         b) GPU 上写 argmax kernel（用 block reduce + atomicMax），只回传 int
    //       生产实现一定走 (b)，因为还要支持 top-k/top-p 采样，不能离开 GPU。
    //
    // [AP4] 整张 block_table 同步拷贝两个低效点：
    //         a) 即使只有一个 slot 改了，也拷整张（浪费 ~50% 带宽）
    //         b) 同步拷贝阻塞 prefill 返回（cudaMemcpy 默认是同步的）
    //       优化：
    //         a) 只拷该 slot 的子表（[slot:slot+1] × N_LAYERS × MAX_BLOCKS_PER_SEQ）
    //         b) 用 cudaMemcpyAsync + 后续 stream sync，让 CPU 早返回
    //         c) 终极方案：block_table 直接维护在 GPU 上（用 unified memory 或 pinned host）
    //       这里影响小是因为 block_table 就 16KB，不是瓶颈。但学习它的低效性
    //       对理解"为什么生产 vLLM 用 GPU 上的 BlockManager"很重要。
    //
    // [AP5] prefill 的 compute/memory 平衡随 prompt_len 变化：
    //       - 短 prompt（N < 32）：每次 GEMM 都是窄矩阵（如 [32, 2048] × [2048, 2048]），
    //         arithmetic intensity 低，memory-bound（受 HBM 带宽限制）
    //       - 中 prompt（N ≈ 128~512）：GEMM 形状接近"方阵"，cublas 能打满 Tensor Core，
    //         compute-bound（受峰值 FLOPS 限制）
    //       - 长 prompt（N > 1024）：QK^T 和 scores×V 的 [N, N] 矩阵尺寸爆炸，
    //         GMEM 读写占主导，又变 memory-bound（这是 FlashAttention 解决的核心问题）
    //       Lab 1 (prefill vs decode profiling) 量化过这种过渡点。
    //       Roofline 分析（Lab 2 / wiki/track-b/phase-5/roofline-model.md）能精确定位。
    // =====================================================================
}

int main(int argc, char *argv[])
{
    cublasHandle_t cublas_handle;
    cublasStatus_t status = cublasCreate(&cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS init failed, status: " << status << "\n";
        return 1;
    }

    Weights weights{};
    if (loadWeights(weights) != 0)
    {
        return 1;
    }

    // allocator for pagedattn
    __nv_bfloat16 *kv_cache;
    cudaMalloc(&kv_cache, KV_CACHE_SIZE_BYTES);
    std::vector<int> free_blocks(NUM_BLOCKS);
    std::iota(free_blocks.begin(), free_blocks.end(), 0);
    std::vector<int> block_table(MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ, -1);
    int *block_table_gpu;
    cudaMalloc(&block_table_gpu, MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ * sizeof(int));

    // PROMPT 0 (What is 2+2?) - length 17
    std::queue<std::vector<int>> queue;
    queue.push({128000, 128006, 882, 128007, 271, 3923, 374, 220, 17, 10, 17, 30, 128009, 128006, 78191, 128007, 271});

    // PROMPT 1 (Name a color.) - length 14
    queue.push({128000, 128006, 882, 128007, 271, 678, 264, 1933, 13, 128009, 128006, 78191, 128007, 271});

    // PROMPT 2 (Say hello.) - length 13
    queue.push({128000, 128006, 882, 128007, 271, 46864, 24748, 13, 128009, 128006, 78191, 128007, 271});

    // PROMPT 3 (Capital of France?) - length 14
    queue.push({128000, 128006, 882, 128007, 271, 64693, 315, 9822, 30, 128009, 128006, 78191, 128007, 271});

    // BATCH
    std::vector<bool> is_slot_free(BATCH_SIZE, true); // set to false when slot taken, set to true when free

    std::vector<std::vector<int>> generated_tokens(BATCH_SIZE);
    std::vector<int> last_generated_tokens(BATCH_SIZE);
    std::vector<int> current_prompt_len(BATCH_SIZE, 0);

    // needed to provide contiguous data for decode
    std::vector<int> active_slots;
    std::vector<int> active_tokens;

    int *gpu_active_slots;
    cudaMalloc(&gpu_active_slots, BATCH_SIZE * sizeof(int));
    int *gpu_seq_lens;
    cudaMalloc(&gpu_seq_lens, BATCH_SIZE * sizeof(int));

    // TODO: recalculate input_tokens_size and prompt_lengths always when there is a change to prompt_under_prefill
    // TODO: right now I handle input manually, it's the least interesting part, will come back to it when continuous batching and pagedattn works

    std::vector<int> prompt;
    int prompt_len;
    int *gpu_input_tokens;
    cudaMalloc(&gpu_input_tokens, MAX_PROMPT_LEN * sizeof(int));
    __nv_bfloat16 *input_embeddings;
    cudaMalloc(&input_embeddings, MAX_PROMPT_LEN * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);

    __nv_bfloat16 *hidden_state;
    cudaMalloc(&hidden_state, MAX_BUFFER_SIZE * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);

    __nv_bfloat16 *rms_norms;
    cudaMalloc(&rms_norms, MAX_BUFFER_SIZE * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);

    __nv_bfloat16 *buf_2048_1; // shared between q_proj and attn_scores_v
    cudaMalloc(&buf_2048_1, MAX_BUFFER_SIZE * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);
    __nv_bfloat16 *q_proj;
    float q_proj_alpha = 1.0f;
    float q_proj_beta = 0.0f;

    // K and V cache
    __nv_bfloat16 *k_proj_temp_buf;
    cudaMalloc(&k_proj_temp_buf, MAX_PROMPT_LEN * KV_DIM * sizeof(__nv_bfloat16));

    __nv_bfloat16 *v_proj_temp_buf;
    cudaMalloc(&v_proj_temp_buf, MAX_PROMPT_LEN * KV_DIM * sizeof(__nv_bfloat16));

    float k_proj_alpha = 1.0f;
    float k_proj_beta = 0.0f;

    float v_proj_alpha = 1.0f;
    float v_proj_beta = 0.0f;

    __nv_bfloat16 *prefill_attn_scores;
    cudaMalloc(&prefill_attn_scores, MAX_PROMPT_LEN * MAX_PROMPT_LEN * sizeof(__nv_bfloat16) * NUM_Q_HEADS);
    float attn_alpha = 1.0f / 8.0f;
    float attn_beta = 0.0f;

    __nv_bfloat16 *attn_scores_v;
    float attn_scores_v_alpha = 1.0f;
    float attn_scores_v_beta = 0.0f;

    __nv_bfloat16 *buf_2048_2; // shared between o_proj and down
    cudaMalloc(&buf_2048_2, MAX_BUFFER_SIZE * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);
    __nv_bfloat16 *o_proj;
    float o_proj_alpha = 1.0f;
    float o_proj_beta = 0.0f;

    __nv_bfloat16 *gate;
    cudaMalloc(&gate, MAX_BUFFER_SIZE * sizeof(__nv_bfloat16) * HIDDEN_DIM);
    float gate_alpha = 1.0f;
    float gate_beta = 0.0f;

    __nv_bfloat16 *up;
    cudaMalloc(&up, MAX_BUFFER_SIZE * sizeof(__nv_bfloat16) * HIDDEN_DIM);
    float up_alpha = 1.0f;
    float up_beta = 0.0f;

    __nv_bfloat16 *down;
    float down_alpha = 1.0f;
    float down_beta = 0.0f;

    __nv_bfloat16 *embed_proj;
    cudaMalloc(&embed_proj, sizeof(__nv_bfloat16) * MAX_BUFFER_SIZE * VOCAB_SIZE);
    float embed_alpha = 1.0f;
    float embed_beta = 0.0f;

    std::vector<__nv_bfloat16> embed_proj_cpu;
    embed_proj_cpu.resize(MAX_BUFFER_SIZE * VOCAB_SIZE);
    // decode-only allocation
    int *gpu_last_tokens;
    cudaMalloc(&gpu_last_tokens, BATCH_SIZE * sizeof(int));
    // TODO: move argmax to GPU and get rid of these CPU<->GPU tokens moves

    // reused temporary buffers for K and V cache computation during decode
    __nv_bfloat16 *k_proj_batched_buffer;
    cudaMalloc(&k_proj_batched_buffer, BATCH_SIZE * sizeof(__nv_bfloat16) * KV_DIM);

    __nv_bfloat16 *v_proj_batched_buffer;
    cudaMalloc(&v_proj_batched_buffer, BATCH_SIZE * sizeof(__nv_bfloat16) * KV_DIM);

    for (int slot = 0; slot < is_slot_free.size() && !queue.empty(); ++slot)
    {
        if (!is_slot_free[slot])
        {
            continue; // slot taken, skip
        }
        prefill(prompt, queue, prompt_len, is_slot_free, slot, gpu_input_tokens, input_embeddings, weights, hidden_state, rms_norms, q_proj, buf_2048_1, cublas_handle, q_proj_alpha, q_proj_beta, k_proj_alpha, k_proj_beta, v_proj_alpha, v_proj_beta, prefill_attn_scores, attn_alpha, attn_beta, attn_scores_v, attn_scores_v_alpha, attn_scores_v_beta, o_proj, buf_2048_2, o_proj_alpha, o_proj_beta, gate_alpha, gate_beta, gate, up_alpha, up_beta, up, down, down_alpha, down_beta, embed_alpha, embed_beta, embed_proj, embed_proj_cpu, generated_tokens, last_generated_tokens, current_prompt_len, k_proj_temp_buf, v_proj_temp_buf, block_table, block_table_gpu, free_blocks, kv_cache);

        // // after prefill:
        // int first_token = -1; // TODO just a stub
        // last_generated_tokens[slot] = first_token;
        // current_prompt_len[slot] = prompt.size();
        // generated_tokens[slot].push_back(first_token);
    }

    // INFERENCE STARTS HERE! =]
    // I have the same amount of embeddings as input tokens
    // it's just every embedding is EMBEDDING_LENGTH length bf16 vector
    // retrieved from model weights based on token's value

    // PREFILL

    // DECODE
    // since now I operate always on index 0 for all values and for current_position_token for new K and V

    while (true) // exit condition irrelevant for now, since it's an inference server that's supposed to run foreveeer!!!
    {
        active_slots.clear();
        active_tokens.clear();
        for (int slot = 0; slot < BATCH_SIZE; ++slot)
        {
            if (is_slot_free[slot])
            {
                if (queue.empty())
                {
                    continue;
                }
                generated_tokens[slot].clear();
                prefill(prompt, queue, prompt_len, is_slot_free, slot, gpu_input_tokens, input_embeddings, weights, hidden_state, rms_norms, q_proj, buf_2048_1, cublas_handle, q_proj_alpha, q_proj_beta, k_proj_alpha, k_proj_beta, v_proj_alpha, v_proj_beta, prefill_attn_scores, attn_alpha, attn_beta, attn_scores_v, attn_scores_v_alpha, attn_scores_v_beta, o_proj, buf_2048_2, o_proj_alpha, o_proj_beta, gate_alpha, gate_beta, gate, up_alpha, up_beta, up, down, down_alpha, down_beta, embed_alpha, embed_beta, embed_proj, embed_proj_cpu, generated_tokens, last_generated_tokens, current_prompt_len, k_proj_temp_buf, v_proj_temp_buf, block_table, block_table_gpu, free_blocks, kv_cache);
            }
            active_slots.push_back(slot);
            active_tokens.push_back(last_generated_tokens[slot]);
        }
        int num_active_slots = active_slots.size();
        if (num_active_slots == 0)
        {
            if (queue.empty())
            {
                break; // TODO: continue will make sense when I will finally write to queue, for now it has predefined size so break instead
            }
            continue;
        }

        // copy useful data to gpu
        cudaMemcpy(gpu_last_tokens, active_tokens.data(), num_active_slots * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_active_slots, active_slots.data(), num_active_slots * sizeof(int), cudaMemcpyHostToDevice);
        std::vector<int> seq_lens(num_active_slots);
        for (int slot = 0; slot < num_active_slots; ++slot)
        {
            int active_slot = active_slots[slot];
            seq_lens[slot] = current_prompt_len[active_slot] + 1;
        }
        cudaMemcpy(gpu_seq_lens, seq_lens.data(), seq_lens.size() * sizeof(int), cudaMemcpyHostToDevice);

        embeddingGatherDecode(gpu_last_tokens, num_active_slots, hidden_state, weights.embed_tokens);
        for (int layer = 0; layer < N_LAYERS; ++layer)
        {
            rmsNorm(hidden_state, rms_norms, weights.input_layernorm[layer], num_active_slots);
            q_proj = buf_2048_1;
            // q proj (num_prompts, 2048)
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         EMBEDDING_LENGTH, // m
                         num_active_slots, // n
                         EMBEDDING_LENGTH, // k
                         &q_proj_alpha,
                         weights.w_q[layer], // A
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH, // lda
                         rms_norms,        // B
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH, // ldb
                         &q_proj_beta,
                         q_proj, // C
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH, // ldc
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);
            // k proj (1, 512), writing output to next position in current layer's K cache
            // K proj = rms_norms (num_prompt, 2048) * W_k (512, 2048)
            // W_k is actually stored as 512, 2048 (out features, in features)
            // so that's why we need to transpose it
            // all the data is stored in row major and cublas reads it as column major
            // so all the data appears as transposed
            // so data actually apppears as (2048, num_prompt) * (2048, 512)
            // the output of matmul will also be produced as transposed, so we can say that
            // in our mental model we talk about K_proj^T
            // and to get K_proj^T we can do transposition trick and write the cublas call as
            // W_k^T * rms_nroms
            // so we end up with: K_proj^T = W_k^T (512, 2048) * rms_norms (2048, num_prompt)
            // result dim is K_proj^T = (512, num_prompt)
            // but it's transposed, so in fact we get correct output dimension (num_prompt, 512)
            // it was great for num_prompt=1, but the problem is that prompts have different length
            // that's why we have vector of current_prompt_len, but also we can't write to K_proj
            // directly, so I write to temp buffer kv_proj_batched_buffer and the scatter
            // output to K_proj in a loop
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         KV_DIM,           // m = 512
                         num_active_slots, // n = num prompts
                         EMBEDDING_LENGTH, // k = 2048
                         &k_proj_alpha,
                         weights.w_k[layer], // A
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH, // lda 2048, because W_k is in memory as 512, 2048
                         // so the gap between subsequent elements is 2048
                         rms_norms, // B
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH, // ldb, same reason for rms_norms
                         &k_proj_beta,
                         k_proj_batched_buffer, // TODO C
                         CUDA_R_16BF,
                         KV_DIM, // ldc = 512
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            // same
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         KV_DIM,
                         num_active_slots,
                         EMBEDDING_LENGTH,
                         &v_proj_alpha,
                         weights.w_v[layer],
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         rms_norms,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         &v_proj_beta,
                         v_proj_batched_buffer,
                         CUDA_R_16BF,
                         KV_DIM,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            for (int slot = 0; slot < num_active_slots; ++slot)
            {
                int active_slot = active_slots[slot];
                ropeDecode(&q_proj[slot * EMBEDDING_LENGTH], current_prompt_len[active_slot], EMBEDDING_LENGTH);
                ropeDecode(k_proj_batched_buffer + slot * KV_DIM, current_prompt_len[active_slot], KV_DIM);
            }

            // PagedAttn scatter k and v from a temp buffer, like in the prefill
            for (int slot = 0; slot < num_active_slots; ++slot)
            {
                int active_slot = active_slots[slot];
                int seq_len = current_prompt_len[active_slot]; // + generated tokens?
                int logical_block_idx = seq_len / BLOCK_SIZE;
                int token_in_block_idx = seq_len % BLOCK_SIZE;
                int block = block_table[active_slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + logical_block_idx];
                if (token_in_block_idx == 0)
                {
                    int physical_block_idx = free_blocks.back();
                    free_blocks.pop_back();
                    block = physical_block_idx;
                    block_table[active_slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + logical_block_idx] = block;
                }
                __nv_bfloat16 *k_cache_ptr = (__nv_bfloat16 *)((char *)kv_cache + block * BLOCK_BYTES + token_in_block_idx * KV_DIM * sizeof(__nv_bfloat16));
                __nv_bfloat16 *k_proj_ptr = k_proj_batched_buffer + slot * KV_DIM;
                cudaMemcpy(k_cache_ptr, k_proj_ptr, KV_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);

                __nv_bfloat16 *v_cache_ptr = (__nv_bfloat16 *)((char *)kv_cache + block * BLOCK_BYTES + V_OFFSET + token_in_block_idx * KV_DIM * sizeof(__nv_bfloat16));
                __nv_bfloat16 *v_proj_ptr = v_proj_batched_buffer + slot * KV_DIM;
                cudaMemcpy(v_cache_ptr, v_proj_ptr, KV_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
            }

            // synchronize block table on cpu with block table on gpu (for attention)
            cudaMemcpy(block_table_gpu, block_table.data(), MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ * sizeof(int), cudaMemcpyHostToDevice);

            pagedAttention(layer, num_active_slots, q_proj, kv_cache, block_table_gpu, gpu_seq_lens, gpu_active_slots, buf_2048_1);

            o_proj = buf_2048_2;
            // (1, 2048) * (2048, 2048) -> (1, 2048)
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         EMBEDDING_LENGTH, // m
                         num_active_slots, // n
                         EMBEDDING_LENGTH, // k
                         &o_proj_alpha,
                         weights.w_o[layer],
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         buf_2048_1,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         &o_proj_beta,
                         o_proj,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            residualAdd(hidden_state, o_proj, num_active_slots);

            rmsNorm(hidden_state, rms_norms, weights.post_attn_layernorms[layer], num_active_slots);

            // MLP
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         HIDDEN_DIM,       // m
                         num_active_slots, // n
                         EMBEDDING_LENGTH, // k
                         &gate_alpha,
                         weights.mlp_gate_proj[layer],
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         rms_norms,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         &gate_beta,
                         gate,
                         CUDA_R_16BF,
                         HIDDEN_DIM,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            // (1, 2048) * (2048, 8192) -> (1, 8192)
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         HIDDEN_DIM,       // m
                         num_active_slots, // n
                         EMBEDDING_LENGTH, // k
                         &up_alpha,
                         weights.mlp_up_proj[layer],
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         rms_norms,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         &up_beta,
                         up,
                         CUDA_R_16BF,
                         HIDDEN_DIM,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            silu(gate, up, num_active_slots);

            down = buf_2048_2;
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         EMBEDDING_LENGTH, // m
                         num_active_slots, // n
                         HIDDEN_DIM,       // k
                         &down_alpha,
                         weights.mlp_down_proj[layer],
                         CUDA_R_16BF,
                         HIDDEN_DIM,
                         gate,
                         CUDA_R_16BF,
                         HIDDEN_DIM,
                         &down_beta,
                         down,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            residualAdd(hidden_state, down, num_active_slots);
        }

        rmsNorm(hidden_state, rms_norms, weights.norm, num_active_slots);

        cublasGemmEx(cublas_handle,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     VOCAB_SIZE,       // m
                     num_active_slots, // n
                     EMBEDDING_LENGTH, // k
                     &embed_alpha,
                     weights.embed_tokens,
                     CUDA_R_16BF,
                     EMBEDDING_LENGTH,
                     rms_norms,
                     CUDA_R_16BF,
                     EMBEDDING_LENGTH,
                     &embed_beta,
                     embed_proj,
                     CUDA_R_16BF,
                     VOCAB_SIZE,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT);

        cudaMemcpy(embed_proj_cpu.data(), embed_proj, sizeof(__nv_bfloat16) * num_active_slots * VOCAB_SIZE, cudaMemcpyDeviceToHost);

        float max_token = 0.0;
        int max_token_idx = 0;
        for (int slot = 0; slot < num_active_slots; ++slot)
        {
            int active_slot = active_slots[slot];
            max_token = (float)embed_proj_cpu[slot * VOCAB_SIZE]; // TODO: verify if float is good enough in place of nvbf16
            max_token_idx = 0;
            for (int token_idx = 0; token_idx < VOCAB_SIZE; ++token_idx)
            {
                if ((float)embed_proj_cpu[slot * VOCAB_SIZE + token_idx] > max_token)
                {
                    max_token = embed_proj_cpu[slot * VOCAB_SIZE + token_idx];
                    max_token_idx = token_idx;
                }
            }
            // TODO: wrap with #ifdef DEBUG
            std::cout << "Output token: " << (float)max_token << ", token index: " << std::to_string(max_token_idx) << std::endl;
            if (max_token_idx == END_OF_TEXT_TOKEN_ID || max_token_idx == EOT_ID_TOKEN_ID || current_prompt_len[active_slot] == MAX_SEQ_LEN - 1)
            {
                is_slot_free[active_slot] = true;
                for (int layer = 0; layer < N_LAYERS; ++layer)
                {
                    for (int logical_block_idx = 0; logical_block_idx < MAX_BLOCKS_PER_SEQ; ++logical_block_idx)
                    {
                        int block_idx = active_slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + logical_block_idx;
                        if (block_table[block_idx] != -1)
                        {
                            free_blocks.push_back(block_table[block_idx]);
                            block_table[block_idx] = -1;
                        }
                    }
                }
                cudaMemcpy(block_table_gpu, block_table.data(), MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ * sizeof(int), cudaMemcpyHostToDevice);
            }
            else
            {
                last_generated_tokens[active_slot] = max_token_idx;
                generated_tokens[active_slot].push_back(max_token_idx);
                current_prompt_len[active_slot] = current_prompt_len[active_slot] + 1;
            }
        }
    }
    std::cout << "\nOk bye!\n";
    cublasDestroy(cublas_handle);
    cudaDeviceSynchronize();
    return 0;
}
