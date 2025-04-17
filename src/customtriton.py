
import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offset_q: tl.constexpr,
    offset_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    if STAGE == 1:
        low, high = 0, block_index_q * BLOCK_SIZE_Q

    elif STAGE == 2:
        # block transition between non causal and causal keys (the diagonal block)
        low, high = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q

        low = tl.multiple_of(low, BLOCK_SIZE_Q)

    else:
        #non causal
        low, high = 0, SEQ_LEN
    
    K_block_ptr = tl.advance(K_block_ptr, (0, low))
    V_block_ptr = tl.advance(V_block_ptr, (low, 0))
    
    for start_kv in range(low, high, BLOCK_SIZE_KV):
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)

        if STAGE == 2:
            mask = offset_q[:, None] >= (start_kv + offset_kv[None, :])
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block -= m_ij[:, None]

        else:
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block * softmax_scale - m_ij[:, None]

        # compute the exp(qk_ij - m_ij)
        P_block = tl.math.exp(QK_block)

        # compute the normilization factor (of the current block)
        l_ij = tl.sum(P_block, 1)

        alpha = tl.math.exp(m_i - m_ij)

        l_i = l_i * alpha + l_ij

        V_block = tl.load(V_block_ptr)

        P_block = P_block.to(tl.float16)

        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)
        
        m_i = m_ij

        #* Next K, V block
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0)) # V[SEQ_LEN, HEAD_DIM]
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV)) # K[HEAD_DIM, SEQ_LEN]

    return O_block, l_i, m_i   

@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [64, 128]
        for BLOCK_SIZE_KV in [32, 64]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    softmax_scale,
    M,
    O,
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,   
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)
    # Block in the sequence length to process
    block_index_q = tl.program_id(0)

    # Which head and batch to process
    index_batch_head = tl.program_id(1)

    # Indicate which batch this program is associated with (each batch has NUM_HEADS heads)
    index_batch = index_batch_head // NUM_HEADS

    # Position of the head in the batch
    index_head =  index_batch_head % NUM_HEADS

    # Get the SEQ_LEN, HEAD_DIM, block in the Q, K, V by selecting indexing it by batch and head
    qkv_offset = (
        index_batch.to(tl.int64) * stride_Q_batch + 
        index_head.to(tl.int64) * stride_Q_head
    )

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )


    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )


    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(
            stride_K_dim,
            stride_K_seq,
        ),  
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1),
    )
    
    O_block_ptr = tl.make_block_ptr(
        base=O + qkv_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    offset_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)

    offset_kv = tl.arange(0, BLOCK_SIZE_KV)

    # running maximum
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")

    # running sum
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0

    O_block = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), dtype=tl.float32)

    Q_block = tl.load(Q_block_ptr)

     

    #* STAGE 1: QK^T 

    if STAGE == 1 or STAGE == 3:
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            offset_q,
            offset_kv,
            SEQ_LEN,
        )

    if STAGE == 3:
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            offset_q,
            offset_kv,
            SEQ_LEN,
        )
    
    m_i += tl.math.log(l_i)

    O_block = O_block / l_i[:, None]
    m_ptrs = M + index_batch_head * SEQ_LEN + offset_q

    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))

@triton.jit
def _attn_bwd_preprocess(
    O,
    dO,
    D,
    SEQ_LEN,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
):
    block_index_q = tl.program_id(0)
    offset_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    index_batch_head = tl.program_id(1)
    offset_dim = tl.arange(0, HEAD_DIM)

    O_block = tl.load(
        O
        + index_batch_head * HEAD_DIM * SEQ_LEN
        + offset_q[:, None] * HEAD_DIM
        + offset_dim[None, :]
    )
    # Load a single block of BLOCK_SIZE_Q rows of dO
    dO_block = tl.load(
        dO
        + index_batch_head * HEAD_DIM * SEQ_LEN
        + offset_q[:, None] * HEAD_DIM
        + offset_dim[None, :]
    ).to(tl.float32)

    # Compute the D block
    D_block = tl.sum(dO_block * O_block, axis=1)  #

    D_block_ptrs = D + index_batch_head * SEQ_LEN + offset_q
    tl.store(D_block_ptrs, D_block)


@triton.jit
def _attn_bwd_dk_dv(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(tl.int64)

    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head

    M += offset_batch_head_seq
    D += offset_batch_head_seq

    offset_dim = tl.arange(0, HEAD_DIM)

    index_block_kv = tl.program_id(0)
    start_kv = index_block_kv * BLOCK_KV

    offset_kv = start_kv + tl.arange(0, BLOCK_KV)

    dV_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dK_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    K_block = tl.load(
        K + offset_kv[:, None] * stride_seq + offset_dim[None, :] * stride_dim
    ) #* BLOCK_KV1, HEAD_DIM
    
    V_block = tl.load(
        V + offset_kv[:, None] * stride_seq + offset_dim[None, :] * stride_dim
    ) #* BLOCK_KV1, HEAD_DIM

    offset_q = tl.arange(0, BLOCK_Q)

    qT_ptrs = Q + offset_q[None, :] * stride_seq + offset_dim[:, None] * stride_dim
    dO_ptrs = dO + offset_q[:, None] * stride_seq + offset_dim[None, :] * stride_dim

    current_q = 0
    num_steps = SEQ_LEN // BLOCK_Q
    for i in range(num_steps):
        qT_block = tl.load(qT_ptrs)

        offset_q = current_q + tl.arange(0, BLOCK_Q)
        m = tl.load(M + offset_q)

        QK_T_block = softmax_scale * tl.dot(K_block, qT_block)
        P_T_block = tl.math.exp(QK_T_block - m[None, :])

        if STAGE == 3:
            mask_block = (
                offset_q[None, :] >= offset_kv[:, None]
            )

            P_T_block = tl.where(mask_block, P_T_block, 0.0)
        
        dO_block = tl.load(dO_ptrs)

        dV_block += tl.dot(P_T_block.to(tl.float16), dO_block)

        Di = tl.load(D + offset_q)


        dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)

        dS_T_block = P_T_block * (dpT_block - Di[None, :])
        dS_T_block = dS_T_block.to(tl.float16)

        dK_block += softmax_scale * tl.dot(dS_T_block, tl.trans(qT_block))
        
        current_q += BLOCK_Q
        qT_ptrs += BLOCK_Q * stride_seq
        dO_ptrs += BLOCK_Q * stride_seq

    dV_block_ptrs = dV + offset_kv[:, None] * stride_seq + offset_dim[None, :] * stride_dim
    tl.store(dV_block_ptrs, dV_block)

    dK_block_ptrs = dK + offset_kv[:, None] * stride_seq + offset_dim[None, :] * stride_dim
    tl.store(dK_block_ptrs, dK_block)


@triton.jit
def _attn_bwd_dq(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(tl.int64)

    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head

    M += offset_batch_head_seq
    D += offset_batch_head_seq

    offset_dim = tl.arange(0, HEAD_DIM)

    index_block_kv = tl.program_id(0)

    start_q = index_block_kv * BLOCK_Q
    offset_q = start_q + tl.arange(0, BLOCK_Q)

    Q_block = tl.load(Q + offset_q[:, None] * stride_seq + offset_dim[None, :] * stride_dim)
    dQ_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    dO_block = tl.load(dO + offset_q[:, None] * stride_seq + offset_dim[None, :] * stride_dim)

    M_block = tl.load(M + offset_q)
    M_block = M_block[:, None]

    offset_kv = tl.arange(0, BLOCK_KV)

    kT_ptrs = K + offset_kv[None, :] * stride_seq + offset_dim[:, None] * stride_dim
    vT_ptrs = V + offset_kv[None, :] * stride_seq + offset_dim[:, None] * stride_dim

    Di = tl.load(D + offset_q)

    current_kv = 0
    num_steps = SEQ_LEN // BLOCK_KV

    for i in range(num_steps):
        K_T_block = tl.load(kT_ptrs)
        V_T_block = tl.load(vT_ptrs)

        QK_block = softmax_scale * tl.dot(Q_block, K_T_block)
        P_block = tl.math.exp(QK_block - M_block)

        if STAGE == 3:
            offset_kv = current_kv + tl.arange(0, BLOCK_KV)
            mask_block = offset_q[:, None] >= offset_kv[None, :]
            P_block = tl.where(mask_block, P_block, 0.0)

        
        dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
        dS_block = P_block * (dP_block - Di[:, None])
        dS_block = dS_block.to(tl.float16)

        dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block))

        current_kv += BLOCK_KV
        kT_ptrs += BLOCK_KV * stride_seq
        vT_ptrs += BLOCK_KV * stride_seq
    
    dQ_block_ptrs = dQ + offset_q[:, None] * stride_seq + offset_dim[None, :] * stride_dim
    tl.store(dQ_block_ptrs, dQ_block)


class FlashAttention(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = Q.shape[-1], K.shape[-1], V.shape[-1]

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        O = torch.empty_like(Q)
        stage = 3 if causal else 1

        grid = lambda args: (
            # cell(SEQ_LEN / BLOCK_SIZE_Q) = how many blocks of Q we have
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]), # which group of queries
            BATCH_SIZE * NUM_HEADS, # which head of which batch size
            1, # Z dimension in CUDA launch grid
        )

        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        )

        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,

            softmax_scale=softmax_scale,
            M=M,
            O=O,

            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),

            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),

            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),

            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),

            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage,
        )

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, M = ctx.saved_tensors

        assert dO.is_contiguous()
        assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
        NUM_WARPS, NUM_STAGES = 4, 3
        BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 128

        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        preprocess_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE * NUM_HEADS)
        D = torch.empty_like(M)

        #* Triton kernel
        _attn_bwd_preprocess[preprocess_grid](
            O=O,
            dO=dO,
            D=D,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=ctx.HEAD_DIM,
            BLOCK_SIZE_Q=BLOCK_SIZE_MACRO,
        )

        grid = (SEQ_LEN // BLOCK_SIZE_MACRO, 1, BATCH_SIZE * NUM_HEADS)

        stage = 3 if ctx.causal else 1
        
        _attn_bwd_dk_dv[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MICRO,
            BLOCK_KV=BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

        _attn_bwd_dq[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MACRO,
            BLOCK_KV=BLOCK_SIZE_MICRO,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

        return dQ, dK, dV, None, None



def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    Q = (torch.empty(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    K = (torch.empty(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )   
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    V = (torch.empty(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )   
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    softmax_scale = 1 / (HEAD_DIM ** 0.5) #* QK^t/sqrt(HEAD_DIM)
    dO = torch.randn_like(Q)

    #* Reference
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale

    if causal:
        P[:,:,MASK==0] = float("-inf")
    
    P = torch.softmax(P.float(), dim=-1).half()

    ref_O = torch.matmul(P, V)

    ref_O.backward(dO)

    ref_dQ, Q.grad = Q.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dV, V.grad = V.grad.clone(), None


    #* Triton
    tri_out = FlashAttention.apply(Q, K, V, causal, softmax_scale).half()
    tri_out.backward(dO)
    
    tri_dQ, Q.grad = Q.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dV, V.grad = V.grad.clone(), None

    # compare
    rtol = 0.0
    atol = 1e-2

    assert torch.allclose(ref_O, tri_out, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)


if __name__ == "__main__":
    BATCH_SIZE = 8
    NUM_HEADS = 16
    SEQ_LEN = 256
    HEAD_DIM = 64

    test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal=True)
    test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal=False)

    print("All tests passed!")