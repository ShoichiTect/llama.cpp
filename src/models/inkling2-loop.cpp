#include "models.h"
#include "../llama-kv-cache-iswa.h"
#include "../llama-memory-recurrent.h"

int llama_model_inkling2_loop::layer_phys(int il) const {
    return il < n_phys ? il : core_start + (il - n_phys) % (core_end - core_start);
}

void llama_model_inkling2_loop::load_arch_hparams(llama_model_loader & ml) {
    llama_model_inkling2::load_arch_hparams(ml);

    ml.get_key(LLM_KV_LOOP_START,     core_start);
    ml.get_key(LLM_KV_LOOP_END,       core_end);
    ml.get_key(LLM_KV_LOOP_COUNT,     n_passes);
    ml.get_key(LLM_KV_LOOP_GATE_RANK, gate_rank);
    ml.get_key(LLM_KV_LOOP_VARYING,   varying, false);

    n_phys = hparams.n_layer();
    const uint32_t block = hparams.attn_res_block_size;

    // expose the loop geometry so a context can pick a smaller number of passes
    hparams.n_loop_passes = n_passes;
    hparams.n_loop_phys   = n_phys;
    hparams.n_loop_start  = core_start;
    hparams.n_loop_end    = core_end;
    // a fixed loop needs at least 2 passes, a variable loop at least 1
    if (n_passes < (varying ? 1u : 2u) || !gate_rank || !core_start || core_start >= core_end || core_end > (uint32_t) n_phys ||
        core_start % block || core_end % block || hparams.n_layer_nextn) {
        throw std::runtime_error(format("inkling2_loop: invalid loop core [%u, %u) with block size %u, %u passes or gate rank %u for %d layers",
                core_start, core_end, block, n_passes, gate_rank, n_phys));
    }
    const size_t n_slot = n_phys + (size_t) (n_passes - 1)*(core_end - core_start);
    if (n_slot > LLAMA_MAX_LAYERS) {
        throw std::runtime_error(format("inkling2_loop: %zu cache slots exceed the maximum of %d", n_slot, LLAMA_MAX_LAYERS));
    }

    // expand to the cache slots before load_tensors() allocates the layers and the memory
    hparams.n_layer_all = n_slot;
    for (int il = n_phys; il < (int) n_slot; ++il) {
        const int p = layer_phys(il);
        hparams.n_head_arr[il]        = hparams.n_head_arr[p];
        hparams.n_head_kv_arr[il]     = hparams.n_head_kv_arr[p];
        hparams.n_ff_arr[il]          = hparams.n_ff_arr[p];
        hparams.n_ff_exp_arr[il]      = hparams.n_ff_exp_arr[p];
        hparams.n_expert_used_arr[il] = hparams.n_expert_used_arr[p];
        hparams.is_swa_impl[il]       = hparams.is_swa_impl[p];
        hparams.is_recr_impl[il]      = hparams.is_recr_impl[p];
    }
}

void llama_model_inkling2_loop::load_arch_tensors(llama_model_loader & ml) {
    llama_model_inkling2::load_arch_tensors(ml);

    const int64_t n_embd = hparams.n_embd;
    const int64_t rank   = gate_rank;
    const llm_tensor down[LOOP_CTRL_COUNT]   = { LLM_TENSOR_LOOP_Q_CTRL_DOWN,   LLM_TENSOR_LOOP_K_CTRL_DOWN,   LLM_TENSOR_LOOP_V_CTRL_DOWN,   LLM_TENSOR_LOOP_FFN_CTRL_DOWN   };
    const llm_tensor gain[LOOP_CTRL_COUNT]   = { LLM_TENSOR_LOOP_Q_CTRL_GAIN,   LLM_TENSOR_LOOP_K_CTRL_GAIN,   LLM_TENSOR_LOOP_V_CTRL_GAIN,   LLM_TENSOR_LOOP_FFN_CTRL_GAIN   };
    const llm_tensor forget[LOOP_CTRL_COUNT] = { LLM_TENSOR_LOOP_Q_CTRL_FORGET, LLM_TENSOR_LOOP_K_CTRL_FORGET, LLM_TENSOR_LOOP_V_CTRL_FORGET, LLM_TENSOR_LOOP_FFN_CTRL_FORGET };

    loop_layers.resize(core_end - core_start);
    for (uint32_t il = core_start; il < core_end; ++il) {
        auto & l = loop_layers[il - core_start];
        // the q, k, v and ffn message queries
        l.msg_query = create_tensor(tn(LLM_TENSOR_LOOP_MSG_QUERY, "weight", il), {n_embd, LOOP_CTRL_COUNT}, 0);
        for (int c = 0; c < LOOP_CTRL_COUNT; ++c) {
            l.ctrl_down[c]   = create_tensor(tn(down[c],   "weight", il), {2*n_embd, rank}, 0);
            l.ctrl_gain[c]   = create_tensor(tn(gain[c],   "weight", il), {rank, n_embd}, 0);
            l.ctrl_forget[c] = create_tensor(tn(forget[c], "weight", il), {rank, n_embd}, 0);
        }
    }
}

void llama_model_inkling2_loop::post_arch_tensors(llama_model_loader &) {
    // share the physical weights, each cache slot keeps its own KV and conv state
    for (int il = n_phys; il < (int) hparams.n_layer(); ++il) {
        layers[il] = layers[layer_phys(il)];
    }
}

std::unique_ptr<llm_graph_context> llama_model_inkling2_loop::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

llama_model_inkling2_loop::graph::graph(const llm_graph_params & params) : llama_model_inkling2::graph(params, no_build{}) {
    const auto & m = static_cast<const llama_model_inkling2_loop &>(model);
    const int start = m.core_start;
    const int end   = m.core_end;

    auto * cur = build_inputs();
    std::vector<ggml_tensor *> blocks;

    // inkling2_loop (S28): the activation bank. mode 1 stores what the next pass would consume,
    //   mode 2 feeds those tensors back in and runs the deepest pass alone.
    const uint32_t bank_mode = params.bank_mode;
    const bool has_bank   = !params.bank.empty();
    const bool bank_read  = bank_mode == 2 && has_bank;
    const bool bank_write = bank_mode == 1 && has_bank;
    // mode 3: the same inputs as mode 2, but the passes below are still built (a parity check of the
    //   bank data against the in-graph values, without the graph-shape difference)
    const bool bank_keep  = bank_mode == 3 && has_bank;
    // mode 4 (debug): store the tensors the current pass consumes, captured at the start of that pass
    const bool bank_in    = bank_mode == 4 && has_bank;

    ggml_tensor * bank_pos = nullptr;
    if ((bank_read || bank_write || bank_keep || bank_in) && has_bank) {
        if (params.bank[0]->ne[0] != hparams.n_embd) {
            throw std::runtime_error(format("inkling2_loop: bank width %d does not match n_embd %d", (int) params.bank[0]->ne[0], (int) hparams.n_embd));
        }

        auto inp = std::make_unique<llm_graph_input_bank_pos>();
        inp->pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        ggml_set_name(inp->pos, "bank_pos");
        ggml_set_input(inp->pos);
        bank_pos = inp->pos;
        res->add_input(std::move(inp));
    }

    // one band of the bank, gathered at the positions of this ubatch
    auto bank_rows = [&](uint32_t slot) -> ggml_tensor * {
        GGML_ASSERT(bank_pos != nullptr);
        GGML_ASSERT(slot < (uint32_t) params.bank.size());

        ggml_tensor * rows = ggml_get_rows(ctx0, params.bank[slot], bank_pos);

        return ggml_reshape_2d(ctx0, rows, hparams.n_embd, ubatch.n_tokens);
    };

    if (bank_read) {
        // the prelude is not run: its blocks (the normalized embeddings) are rebuilt from the graph
        // inputs, its output and the blocks of the pass below come from the bank
        blocks.push_back(cur);
        cur = bank_rows(0);
    } else {
        for (int il = 0; il < start; ++il) {
            cur = build_layer(il, il, cur, blocks);
        }
    }

    // the context may run fewer passes than the model was loaded with
    const uint32_t n_passes = params.cparams.n_passes > 0 ? params.cparams.n_passes : m.n_passes;
    const uint32_t n_bank   = (uint32_t) params.bank.size();

    // every pass starts from the prelude output
    auto * prelude = cur;
    const auto prelude_blocks = blocks;
    for (uint32_t pass = bank_read ? n_passes - 1 : 0; pass < n_passes; ++pass) {
        // the bank of the previous pass: its blocks after the first core block, and its output
        std::vector<ggml_tensor *> bank;
        ggml_tensor * bank_out = nullptr;
        const bool from_bank = bank_read || (bank_keep && pass + 1 == n_passes);
        if (from_bank) {
            GGML_ASSERT(n_bank >= 3);
            for (uint32_t slot = 1; slot + 1 < n_bank; ++slot) {
                bank.push_back(bank_rows(slot));
            }
            bank_out = bank_rows(n_bank - 1);
            if (bank_keep) {
                cur    = bank_rows(0);
                blocks = prelude_blocks;
            }
        } else if (pass > 0) {
            bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
            bank_out = cur;
            cur = prelude;
            blocks = prelude_blocks;

            if (bank_in && pass + 1 == n_passes) {
                // the tensors this pass consumes, exactly as they are here
                std::vector<ggml_tensor *> store;
                store.push_back(prelude);
                for (auto * t : bank) {
                    store.push_back(t);
                }
                store.push_back(bank_out);

                if ((uint32_t) store.size() == n_bank) {
                    for (uint32_t slot = 0; slot < n_bank; ++slot) {
                        ggml_tensor * src = ggml_reshape_2d(ctx0, store[slot], hparams.n_embd, ubatch.n_tokens);
                        src = ggml_cast(ctx0, src, params.bank[slot]->type);
                        ggml_build_forward_expand(gf, ggml_set_rows(ctx0, params.bank[slot], src, bank_pos));
                    }
                }
            }
        }
        for (int ip = start; ip < end; ++ip) {
            const int il = pass == 0 ? ip : m.n_phys + (pass - 1)*(end - start) + (ip - start);
            if (pass == 0) {
                cur = build_layer(il, ip, cur, blocks);
                continue;
            }
            const auto & loop = m.loop_layers[ip - start];
            cur = build_layer(il, ip, cur, blocks, mix(bank, bank_out, loop.msg_query), &loop);
        }
    }

    if (bank_write) {
        // the tensors the next pass would consume: the prelude output, the cross-depth blocks of this
        // pass (its blocks after the first core block) and its output
        std::vector<ggml_tensor *> store;
        store.push_back(prelude);
        for (size_t i = prelude_blocks.size() + 1; i < blocks.size(); ++i) {
            store.push_back(blocks[i]);
        }
        store.push_back(cur);

        if ((uint32_t) store.size() != n_bank) {
            throw std::runtime_error(format("inkling2_loop: bank has %u slots, the graph stores %u", n_bank, (uint32_t) store.size()));
        }

        for (uint32_t slot = 0; slot < n_bank; ++slot) {
            ggml_tensor * src = ggml_reshape_2d(ctx0, store[slot], hparams.n_embd, ubatch.n_tokens);
            src = ggml_cast(ctx0, src, params.bank[slot]->type);
            ggml_build_forward_expand(gf, ggml_set_rows(ctx0, params.bank[slot], src, bank_pos));
        }
    }

    // no repair: this graph only runs the slots of the first `n_passes` passes. a cell of a
    // remaining pass slot at a position of this ubatch would otherwise keep the K/V of an earlier
    // run (another sequence, or the same sequence at a deeper depth) and the shared KQ mask would
    // treat it as valid. reset those cells to the neutral value a fresh cache has, so the positions
    // stay usable when the depth is raised later on but carry no content of another series.
    for (uint32_t pass = n_passes; pass < params.cparams.n_passes_max; ++pass) {
        for (int ip = start; ip < end; ++ip) {
            const int il = m.n_phys + (pass - 1)*(end - start) + (ip - start);

            const bool is_swa = hparams.is_swa(il);

            const llama_kv_cache_context * kvctx = is_swa ? attn->mctx->get_swa() : attn->mctx->get_base();
            ggml_tensor * k_idxs = is_swa ? attn->get_k_idxs_swa() : attn->get_k_idxs();
            ggml_tensor * v_idxs = is_swa ? attn->get_v_idxs_swa() : attn->get_v_idxs();
            if (!k_idxs || !v_idxs) {
                continue;
            }

            const int64_t n_embd_head = hparams.n_embd_head_k(il);
            const int64_t n_head_kv   = hparams.n_head_kv(il);
            const int64_t n_tok       = ubatch.n_tokens;

            ggml_tensor * zero_k = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd_head, n_head_kv, n_tok), 0.0f);
            ggml_tensor * zero_v = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd_head, n_head_kv, n_tok), 0.0f);
            ggml_set_name(zero_k, "inkling2_nopass_k");
            ggml_set_name(zero_v, "inkling2_nopass_v");

            ggml_build_forward_expand(gf, kvctx->cpy_k(ctx0, zero_k, k_idxs, il));
            ggml_build_forward_expand(gf, kvctx->cpy_v(ctx0, zero_v, v_idxs, il));

            // the conv/recurrent state of the slot must not keep the history of an earlier series
            // either: the next pass slot resumes from the state written at this cell
            ggml_tensor * r = recr->mctx->get_r_l(il);
            if (r) {
                const size_t row = r->nb[1];
                const size_t off = (size_t) recr->mctx->get_head()*row;
                if (off + ubatch.n_seqs*row <= ggml_nbytes(r)) {
                    ggml_tensor * cell = ggml_view_2d(ctx0, r, hparams.n_embd_r(), ubatch.n_seqs, row, off);
                    ggml_tensor * zero_r = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_r(), ubatch.n_seqs), 0.0f);
                    ggml_set_name(zero_r, "inkling2_nopass_r");
                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, zero_r, cell));
                }
            }
        }
    }

    for (int il = end; il < m.n_phys; ++il) {
        cur = build_layer(il, il, cur, blocks);
    }
    build_output(cur, blocks);
}
