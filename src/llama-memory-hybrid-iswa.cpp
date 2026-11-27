#include "llama-memory-hybrid-iswa.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-context.h"

//
// llama_memory_hybrid_iswa
//

llama_memory_hybrid_iswa::llama_memory_hybrid_iswa(
        const llama_model & model,
                      bool   no_alloc,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   swa_full,
                 uint32_t   kv_size,
                 uint32_t   n_ubatch,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                 uint32_t   n_layer_use) :
    hparams(model.hparams),
    mem_attn(new llama_kv_cache_iswa(
        model,
        no_alloc,
        type_k,
        type_v,
        v_trans,
        offload,
        swa_full,
        unified,
        kv_size,
        n_seq_max,
        n_ubatch,
        nullptr,
        filter_attn == nullptr ?
            [&](int32_t il) { return !hparams.is_recr(il); }
            : filter_attn,
        nullptr,
        nullptr,
        n_layer_use
    )),
    mem_recr(new llama_memory_recurrent(
        model,
        no_alloc,
        type_r,
        type_s,
        offload,
        rs_size,
        n_seq_max,
        n_rs_seq,
        filter_recr == nullptr ?
            [&](int32_t il) { return hparams.is_recr(il); }
            : filter_recr
    )) {}

llama_memory_context_ptr llama_memory_hybrid_iswa::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (mem_attn->get_base()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = mem_recr->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!mem_recr->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache (iswa version returns both base and swa slot infos)
        auto sinfos_base = mem_attn->get_base()->prepare(ubatches);
        if (sinfos_base.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention base ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        auto sinfos_swa = mem_attn->get_swa()->prepare(ubatches);
        if (sinfos_swa.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention swa ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        return std::make_unique<llama_memory_hybrid_iswa_context>(
                this, std::move(sinfos_base), std::move(sinfos_swa), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_iswa::init_full() {
    return std::make_unique<llama_memory_hybrid_iswa_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_iswa::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_iswa_context>(this, lctx, optimize);
}

void llama_memory_hybrid_iswa::set_allow_replay(bool value) {
    mem_attn->set_allow_replay(value);
}

bool llama_memory_hybrid_iswa::get_can_shift() const {
    // Shifting is trivially supported for recurrent
    return mem_attn->get_can_shift();
}

void llama_memory_hybrid_iswa::clear(bool data) {
    mem_attn->clear(data);
    mem_recr->clear(data);
}

bool llama_memory_hybrid_iswa::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // Try removing from the recurrent cache first since it may fail. If it does
    // fail, the cache will not have been mutated.
    if (!mem_recr->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    return mem_attn->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_iswa::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    mem_attn->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    mem_recr->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_hybrid_iswa::seq_keep(llama_seq_id seq_id) {
    mem_attn->seq_keep(seq_id);
    mem_recr->seq_keep(seq_id);
}

void llama_memory_hybrid_iswa::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    mem_attn->seq_add(seq_id, p0, p1, shift);
    mem_recr->seq_add(seq_id, p0, p1, shift);
}

void llama_memory_hybrid_iswa::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    mem_attn->seq_div(seq_id, p0, p1, d);
    mem_recr->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_hybrid_iswa::seq_pos_min(llama_seq_id seq_id) const {
    // the min of the total cache is the max of the two caches' min values
    return std::max(mem_attn->seq_pos_min(seq_id), mem_recr->seq_pos_min(seq_id));
}

llama_pos llama_memory_hybrid_iswa::seq_pos_max(llama_seq_id seq_id) const {
    // the max of the total cache is the min of the two caches' max values
    return std::min(mem_attn->seq_pos_max(seq_id), mem_recr->seq_pos_max(seq_id));
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_iswa::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = mem_attn->memory_breakdown();
    for (const auto & buft_size : mem_recr->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

void llama_memory_hybrid_iswa::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    mem_attn->state_write(io, seq_id, flags);
    mem_recr->state_write(io, seq_id, flags);
}

void llama_memory_hybrid_iswa::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    mem_attn->state_read(io, seq_id, flags);
    mem_recr->state_read(io, seq_id, flags);
}

llama_kv_cache_iswa * llama_memory_hybrid_iswa::get_mem_attn() const {
    return mem_attn.get();
}

// inkling2_loop (V08): pass-slot substitution. the loop core occupies the physical layers
//   [n_loop_start, n_loop_end); pass 0 keeps them at their own cache slot and pass p > 0 keeps the
//   j-th of them at n_loop_phys + (p - 1)*(n_loop_end - n_loop_start) + j (see layer_phys()).
//   copying the K/V rows of a range of positions between two pass slots makes the attention of a
//   pass read the K/V that another pass wrote; the rows of new positions still go to the pass's own
//   slot, because only the existing cells [p0, p1) are replaced.
int32_t llama_memory_hybrid_iswa::copy_loop_pass_kv(uint32_t pass_src, uint32_t pass_dst, llama_seq_id seq_id,
        llama_pos p0, llama_pos p1) {
    if (hparams.n_loop_phys == 0) {
        return -1;
    }
    if (p0 < 0 || p1 <= p0) {
        return -1;
    }

    auto * kv_base = mem_attn->get_base();
    auto * kv_swa  = mem_attn->get_swa();

    const uint32_t n_phys = hparams.n_loop_phys;
    const uint32_t start  = hparams.n_loop_start;
    const uint32_t end    = hparams.n_loop_end;

    auto slot = [&](uint32_t pass, uint32_t j) -> int32_t {
        return pass == 0 ? (int32_t) (start + j)
                         : (int32_t) (n_phys + (pass - 1)*(end - start) + j);
    };

    int32_t n_layers = 0;

    for (uint32_t j = 0; j < end - start; ++j) {
        const int32_t il_src = slot(pass_src, j);
        const int32_t il_dst = slot(pass_dst, j);

        if (il_src == il_dst) {
            ++n_layers;
            continue;
        }
        if (kv_base->copy_kv_rows(seq_id, p0, p1, il_src, il_dst)) {
            ++n_layers;
            continue;
        }
        if (kv_swa->copy_kv_rows(seq_id, p0, p1, il_src, il_dst)) {
            ++n_layers;
            continue;
        }

        LLAMA_LOG_WARN("%s: pass %u -> %u, core layer %u: no cache holds both slots (%d, %d)\n",
                __func__, pass_src, pass_dst, j, il_src, il_dst);
    }

    LLAMA_LOG_INFO("%s: seq %d: pass %u -> %u over [%d, %d): %d layers\n",
            __func__, seq_id, pass_src, pass_dst, (int) p0, (int) p1, n_layers);

    return n_layers;
}

llama_memory_recurrent * llama_memory_hybrid_iswa::get_mem_recr() const {
    return mem_recr.get();
}

//
// llama_memory_hybrid_iswa_context
//

llama_memory_hybrid_iswa_context::llama_memory_hybrid_iswa_context(llama_memory_status status) : status(status) {}

llama_memory_hybrid_iswa_context::llama_memory_hybrid_iswa_context(llama_memory_hybrid_iswa * mem) :
    ctx_attn(mem->get_mem_attn()->init_full()),
    ctx_recr(mem->get_mem_recr()->init_full()),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_iswa_context::llama_memory_hybrid_iswa_context(
        llama_memory_hybrid_iswa * mem,
                   llama_context * lctx,
                            bool   optimize) :
    ctx_attn(mem->get_mem_attn()->init_update(lctx, optimize)),
    ctx_recr(mem->get_mem_recr()->init_update(lctx, optimize)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_iswa_context::llama_memory_hybrid_iswa_context(
           llama_memory_hybrid_iswa * mem,
                    slot_info_vec_t   sinfos_base,
                    slot_info_vec_t   sinfos_swa,
          std::vector<llama_ubatch>   ubatches) :
    ubatches(std::move(ubatches)),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_attn(new llama_kv_cache_iswa_context(mem->get_mem_attn(), std::move(sinfos_base), std::move(sinfos_swa), this->ubatches)),
    ctx_recr(new llama_memory_recurrent_context(mem->get_mem_recr(), this->ubatches)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

bool llama_memory_hybrid_iswa_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_attn->next();
    ctx_recr->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_hybrid_iswa_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_attn->apply();
    res = res & ctx_recr->apply();

    return res;
}

llama_memory_status llama_memory_hybrid_iswa_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_hybrid_iswa_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const llama_kv_cache_iswa_context * llama_memory_hybrid_iswa_context::get_attn() const {
    return static_cast<const llama_kv_cache_iswa_context *>(ctx_attn.get());
}

const llama_memory_recurrent_context * llama_memory_hybrid_iswa_context::get_recr() const {
    return static_cast<const llama_memory_recurrent_context *>(ctx_recr.get());
}
