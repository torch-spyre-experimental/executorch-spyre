// Copyright 2026 The Torch-Spyre / ExecuTorch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Validation harness: drive an --llm-wrapper Spyre PTE through the *standard*
// ExecuTorch LLM runner stack (extension/llm/runner), with NO Spyre-specific
// runner logic:
//
//   Module -> get_llm_metadata -> get_eos_ids -> IOManager
//          -> TextDecoderRunner -> TextPrefiller + TextTokenGenerator
//          -> TextLLMRunner (and the stock create_text_llm_runner factory)
//
// The only non-stock code is TracingDecoderRunner, a thin TextDecoderRunner
// subclass that forwards step() to the base class unchanged and records
// (start_pos, input token, argmax, top-5, logits) for the token trace.
//
// Usage:
//   spyre_llm_runner --pte <file.pte> (--tokenizer <tokenizer.json> |
//   --stub-tokenizer)
//                  (--prompt <text> | --prompt-ids a,b,c)  [--max-new N]
//                  [--mode components|runner|factory] [--dump-dir DIR]

#include <executorch/extension/llm/runner/io_manager/io_manager.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/text_decoder_runner.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/extension/llm/runner/text_prefiller.h>
#include <executorch/extension/llm/runner/text_token_generator.h>
#include <executorch/extension/module/module.h>
#include <pytorch/tokenizers/tokenizer.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::extension::llm::GenerationConfig;
using executorch::extension::llm::IOManager;
using executorch::extension::llm::Stats;
using executorch::extension::llm::TextDecoderRunner;
using executorch::extension::llm::TextLLMRunner;
using executorch::extension::llm::TextPrefiller;
using executorch::extension::llm::TextTokenGenerator;
using executorch::runtime::Error;

namespace {

// ---------------------------------------------------------------------------
// Byte-level stub tokenizer (vocab 256) for synthetic PTEs that have no
// tokenizer.  id == byte value.  Only used with --stub-tokenizer.
// ---------------------------------------------------------------------------
class StubTokenizer : public tokenizers::Tokenizer {
 public:
  StubTokenizer() {
    initialized_ = true;
    vocab_size_ = 256;
    bos_tok_ = 1;
    eos_tok_ = 0;
  }
  tokenizers::Error load(const std::string&) override {
    return tokenizers::Error::Ok;
  }
  tokenizers::Result<std::string> id_to_piece(uint64_t t) const override {
    return piece(t);
  }
  tokenizers::Result<uint64_t> piece_to_id(
      const std::string& s) const override {
    return s.empty() ? 0 : static_cast<uint64_t>(static_cast<uint8_t>(s[0]));
  }
  tokenizers::Result<std::vector<uint64_t>> encode(
      const std::string& in, int8_t bos = 0, int8_t eos = 0) const override {
    std::vector<uint64_t> out;
    for (int8_t i = 0; i < bos; ++i) out.push_back(bos_tok_);
    for (unsigned char c : in) out.push_back(c);
    for (int8_t i = 0; i < eos; ++i) out.push_back(eos_tok_);
    return out;
  }
  tokenizers::Result<std::string> decode(uint64_t, uint64_t t,
                                         bool = false) const override {
    return piece(t);
  }

 private:
  static std::string piece(uint64_t t) {
    if (t >= 33 && t < 127) return std::string(1, static_cast<char>(t));
    return "<" + std::to_string(t) + ">";
  }
};

// ---------------------------------------------------------------------------
// Trace record + tracing decoder runner (forwards to the stock step()).
// ---------------------------------------------------------------------------
struct StepRecord {
  int call;
  int64_t start_pos;
  int64_t n_in;
  std::vector<int64_t> in_tokens;
  int64_t argmax;
  std::vector<int64_t> top5;
  std::string dtype;
  std::vector<int64_t> shape;
  double ms;
};

float elem_to_float(const executorch::aten::Tensor& t, int64_t i) {
  switch (t.scalar_type()) {
    case executorch::aten::ScalarType::Half:
      return static_cast<float>(t.const_data_ptr<executorch::aten::Half>()[i]);
    case executorch::aten::ScalarType::BFloat16:
      return static_cast<float>(
          t.const_data_ptr<executorch::aten::BFloat16>()[i]);
    case executorch::aten::ScalarType::Float:
      return t.const_data_ptr<float>()[i];
    default:
      std::fprintf(stderr, "unsupported logits dtype\n");
      std::exit(2);
  }
}

class TracingDecoderRunner : public TextDecoderRunner {
 public:
  TracingDecoderRunner(Module* m, IOManager* io, std::string method,
                       std::string dump_dir)
      : TextDecoderRunner(m, io, std::move(method)),
        dump_dir_(std::move(dump_dir)) {}

  executorch::runtime::Result<executorch::aten::Tensor> step(
      TensorPtr& tokens, int64_t start_pos) override {
    StepRecord r;
    r.call = static_cast<int>(trace.size());
    r.start_pos = start_pos;
    r.n_in = tokens->numel();
    for (int64_t i = 0; i < tokens->numel(); ++i)
      r.in_tokens.push_back(tokens->const_data_ptr<int64_t>()[i]);

    auto t0 = std::chrono::steady_clock::now();
    // Diagnostic only (EXECUTORCH_SPYRE_FORCE_POS): override input_pos to probe
    // the delegate.
    if (const char* fp = std::getenv("EXECUTORCH_SPYRE_FORCE_POS"))
      start_pos = std::atoll(fp);
    auto res = TextDecoderRunner::step(tokens, start_pos);  // STOCK path
    auto t1 = std::chrono::steady_clock::now();
    r.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (!res.ok()) return res;

    const auto& lg = res.get();
    r.dtype = executorch::runtime::toString(lg.scalar_type());
    for (int d = 0; d < lg.dim(); ++d) r.shape.push_back(lg.size(d));
    const int64_t V = lg.size(lg.dim() - 1);
    const int64_t base = (lg.dim() == 3) ? (lg.size(1) - 1) * V : 0;
    std::vector<int64_t> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(
        idx.begin(), idx.begin() + 5, idx.end(), [&](int64_t a, int64_t b) {
          return elem_to_float(lg, base + a) > elem_to_float(lg, base + b);
        });
    r.argmax = idx[0];
    r.top5.assign(idx.begin(), idx.begin() + 5);
    if (!dump_dir_.empty()) {
      std::vector<float> f(V);
      for (int64_t i = 0; i < V; ++i) f[i] = elem_to_float(lg, base + i);
      std::ofstream o(
          dump_dir_ + "/logits_call" + std::to_string(r.call) + ".f32",
          std::ios::binary);
      o.write(reinterpret_cast<const char*>(f.data()), V * sizeof(float));
    }
    trace.push_back(std::move(r));
    return res;
  }

  std::vector<StepRecord> trace;

 private:
  std::string dump_dir_;
};

std::vector<uint64_t> parse_ids(const std::string& s) {
  std::vector<uint64_t> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) v.push_back(std::stoull(tok));
  return v;
}

void dump_trace(const std::vector<StepRecord>& tr) {
  std::printf("\n---- TOKEN TRACE (every TextDecoderRunner::step call) ----\n");
  std::printf("%4s %9s %5s %-14s %9s  %-28s %8s\n", "call", "start_pos", "n_in",
              "in_tokens", "argmax", "top5", "ms");
  for (const auto& r : tr) {
    std::string in, t5;
    for (size_t i = 0; i < r.in_tokens.size(); ++i)
      in += (i ? "," : "") + std::to_string(r.in_tokens[i]);
    for (size_t i = 0; i < r.top5.size(); ++i)
      t5 += (i ? "," : "") + std::to_string(r.top5[i]);
    std::printf(
        "%4d %9" PRId64 " %5" PRId64 " %-14s %9" PRId64 "  %-28s %8.2f\n",
        r.call, r.start_pos, r.n_in, in.c_str(), r.argmax, t5.c_str(), r.ms);
  }
  std::printf("TRACE_ARGMAX:");
  for (const auto& r : tr) std::printf(" %" PRId64, r.argmax);
  std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string pte, tok_path, prompt, prompt_ids_s, mode = "components",
                                                   dump_dir;
  bool stub = false;
  int max_new = 8;
  int repeat = 1;              // number of generate() calls (runner mode)
  bool reset_between = false;  // call runner.reset() between generate() calls
  std::string prompt2;         // if set, generate() call #2+ uses this prompt
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", a.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--pte")
      pte = next();
    else if (a == "--tokenizer")
      tok_path = next();
    else if (a == "--stub-tokenizer")
      stub = true;
    else if (a == "--prompt")
      prompt = next();
    else if (a == "--prompt-ids")
      prompt_ids_s = next();
    else if (a == "--max-new")
      max_new = std::atoi(next().c_str());
    else if (a == "--mode")
      mode = next();
    else if (a == "--prompt2")
      prompt2 = next();
    else if (a == "--repeat")
      repeat = std::atoi(next().c_str());
    else if (a == "--reset-between")
      reset_between = true;
    else if (a == "--dump-dir")
      dump_dir = next();
    else {
      std::fprintf(stderr, "unknown arg %s\n", a.c_str());
      return 2;
    }
  }
  if (pte.empty() || (!stub && tok_path.empty())) {
    std::fprintf(stderr, "need --pte and (--tokenizer | --stub-tokenizer)\n");
    return 2;
  }

  std::unique_ptr<tokenizers::Tokenizer> tokenizer;
  if (stub)
    tokenizer = std::make_unique<StubTokenizer>();
  else
    tokenizer = executorch::extension::llm::load_tokenizer(tok_path);
  if (!tokenizer || !tokenizer->is_loaded()) {
    std::fprintf(stderr, "ERROR: tokenizer failed to load\n");
    return 2;
  }
  std::printf("RUNTIME_PATH mode=%s\n", mode.c_str());
  std::printf("tokenizer: vocab_size=%d bos=%" PRIu64 " eos=%" PRIu64 "\n",
              tokenizer->vocab_size(), tokenizer->bos_tok(),
              tokenizer->eos_tok());

  // ---- stock factory path (no tracing; proves create_text_llm_runner works)
  // ----
  if (mode == "factory") {
    auto runner = executorch::extension::llm::create_text_llm_runner(
        pte, std::move(tokenizer), std::nullopt, /*temperature=*/0.0f);
    if (!runner) {
      std::fprintf(stderr, "ERROR: create_text_llm_runner returned null\n");
      return 3;
    }
    GenerationConfig cfg;
    cfg.echo = false;
    cfg.temperature = 0.0f;
    cfg.ignore_eos = true;
    cfg.max_new_tokens = max_new;
    std::string text;
    Error e =
        runner->generate(prompt, cfg, [&](const std::string& s) { text += s; });
    std::printf("\nFACTORY_RESULT error=0x%x\nFACTORY_TEXT:[%s]\n", (unsigned)e,
                text.c_str());
    return e == Error::Ok ? 0 : 4;
  }

  // ---- manual assembly: identical to create_text_llm_runner, plus tracing
  // ----
  auto module =
      std::make_unique<Module>(pte, Module::LoadMode::MmapUseMlockIgnoreErrors);

  // Task 1/2: raw constant methods + get_llm_metadata().
  auto names = module->method_names();
  if (!names.ok()) {
    std::fprintf(stderr, "ERROR: method_names failed\n");
    return 3;
  }
  std::printf("\n---- METHODS (constant_methods + entry points) ----\n");
  std::vector<std::string> sorted(names->begin(), names->end());
  std::sort(sorted.begin(), sorted.end());
  for (const auto& n : sorted) {
    if (n == "forward") {
      std::printf("  %-26s <entry point>\n", n.c_str());
      continue;
    }
    auto v = module->get(n);
    if (!v.ok()) {
      std::printf("  %-26s <get() failed 0x%x>\n", n.c_str(),
                  (unsigned)v.error());
      continue;
    }
    const auto& ev = v.get();
    if (ev.isInt())
      std::printf("  %-26s Int  %" PRId64 "\n", n.c_str(), ev.toInt());
    else if (ev.isBool())
      std::printf("  %-26s Bool %d\n", n.c_str(), (int)ev.toBool());
    else
      std::printf("  %-26s tag=%d\n", n.c_str(), (int)ev.tag);
  }
  {
    auto meta = module->method_meta("forward");
    if (!meta.ok()) {
      std::fprintf(stderr, "ERROR: method_meta(forward)\n");
      return 3;
    }
    std::printf(
        "\n---- forward() signature ----\n  num_inputs=%zu num_outputs=%zu "
        "num_backends=%zu num_attributes=%zu\n",
        meta->num_inputs(), meta->num_outputs(), meta->num_backends(),
        meta->num_attributes());
    for (size_t i = 0; i < meta->num_inputs(); ++i) {
      auto ti = meta->input_tensor_meta(i);
      if (!ti.ok()) continue;
      std::string sz;
      for (auto s : ti->sizes()) sz += std::to_string(s) + " ";
      std::printf("  input[%zu]  %s [ %s]\n", i,
                  executorch::runtime::toString(ti->scalar_type()), sz.c_str());
    }
    for (size_t i = 0; i < meta->num_outputs(); ++i) {
      auto ti = meta->output_tensor_meta(i);
      if (!ti.ok()) continue;
      std::string sz;
      for (auto s : ti->sizes()) sz += std::to_string(s) + " ";
      std::printf("  output[%zu] %s [ %s]\n", i,
                  executorch::runtime::toString(ti->scalar_type()), sz.c_str());
    }
    for (size_t i = 0; i < meta->num_backends(); ++i) {
      auto bn = meta->get_backend_name(i);
      if (bn.ok()) std::printf("  backend[%zu] = %s\n", i, bn.get());
    }
  }

  auto metadata_result = executorch::extension::llm::get_llm_metadata(
      tokenizer.get(), module.get());
  if (metadata_result.error() != Error::Ok) {
    std::fprintf(stderr, "FAIL get_llm_metadata() -> 0x%x\n",
                 (unsigned)metadata_result.error());
    return 3;
  }
  auto metadata = metadata_result.get();
  std::printf("\n---- get_llm_metadata() = OK ----\n");
  {
    std::vector<std::pair<std::string, int64_t>> kv(metadata.begin(),
                                                    metadata.end());
    std::sort(kv.begin(), kv.end());
    for (auto& p : kv)
      std::printf("  %-26s %" PRId64 "\n", p.first.c_str(), p.second);
  }

  auto eos_ids = std::make_unique<std::unordered_set<uint64_t>>(
      executorch::extension::llm::get_eos_ids(tokenizer.get(), module.get()));
  std::printf("  eos_ids:");
  for (auto e : *eos_ids) std::printf(" %" PRIu64, e);
  std::printf("\n");

  auto io_manager = std::make_unique<IOManager>(*module);
  auto decoder = std::make_unique<TracingDecoderRunner>(
      module.get(), io_manager.get(), "forward", dump_dir);
  auto* decoder_raw = decoder.get();

  using namespace executorch::extension::llm;
  auto prefiller = std::make_unique<TextPrefiller>(
      decoder.get(), metadata.at(kUseKVCache), metadata.at(kEnableDynamicShape),
      metadata.at(kMaxSeqLen));
  auto stats = std::make_unique<Stats>();
  auto generator = std::make_unique<TextTokenGenerator>(
      tokenizer.get(), decoder.get(), metadata.at(kUseKVCache),
      std::move(eos_ids), stats.get());

  std::vector<uint64_t> prompt_ids;
  if (!prompt_ids_s.empty())
    prompt_ids = parse_ids(prompt_ids_s);
  else {
    auto enc = tokenizer->encode(prompt, 0, 0);
    if (!enc.ok()) {
      std::fprintf(stderr, "ERROR: encode failed\n");
      return 3;
    }
    prompt_ids = enc.get();
  }
  std::printf("\nPROMPT_IDS:");
  for (auto t : prompt_ids) std::printf(" %" PRIu64, t);
  std::printf("\n");

  std::string text;
  Error err = Error::Ok;
  if (mode == "components") {
    // Task 3/4 at component level: TextPrefiller + TextTokenGenerator only.
    ET_LOG(Info, "components: TextDecoderRunner::load");
    if ((err = decoder->load()) != Error::Ok) {
      std::fprintf(stderr, "load failed 0x%x\n", (unsigned)err);
      return 4;
    }
    int64_t pos = 0;
    auto pf = prefiller->prefill(prompt_ids, pos);
    if (!pf.ok()) {
      std::fprintf(stderr, "prefill failed 0x%x\n", (unsigned)pf.error());
      return 4;
    }
    uint64_t first = pf.get();
    std::printf("PREFILL_DONE next_token=%" PRIu64 " pos=%" PRId64 "\n", first,
                pos);
    generator->set_ignore_eos(true);
    std::vector<uint64_t> toks = {first};
    auto gr = generator->generate(toks, pos, max_new - 1, 0.0f,
                                  [&](const std::string& s) { text += s; });
    if (!gr.ok()) {
      std::fprintf(stderr, "generate failed 0x%x\n", (unsigned)gr.error());
      return 4;
    }
    std::printf("GENERATE_DONE new_tokens=%" PRId64 "\n", gr.get());
  } else if (mode == "runner") {
    // Task 3/4 at full runner level: TextLLMRunner::generate (text prompt).
    TextLLMRunner runner(metadata, std::move(tokenizer), std::move(module),
                         std::move(decoder), std::move(prefiller),
                         std::move(io_manager), std::move(generator),
                         std::move(stats), /*temperature=*/0.0f);
    GenerationConfig cfg;
    cfg.echo = false;
    cfg.temperature = 0.0f;
    cfg.ignore_eos = true;
    cfg.max_new_tokens = max_new;
    for (int r = 0; r < repeat; ++r) {
      if (r > 0 && reset_between) runner.reset();
      std::printf("\nGENERATE_CALL %d (reset_before=%d)\n", r,
                  (int)(r > 0 && reset_between));
      text += "|";
      err = runner.generate((r > 0 && !prompt2.empty()) ? prompt2 : prompt, cfg,
                            [&](const std::string& s) { text += s; });
      if (err != Error::Ok) break;
    }
    std::printf("\nRUNNER_RESULT error=0x%x\n", (unsigned)err);
    dump_trace(decoder_raw->trace);
    std::printf("GENERATED_TEXT:[%s]\n", text.c_str());
    return err == Error::Ok ? 0 : 4;
  } else {
    std::fprintf(stderr, "unknown --mode %s\n", mode.c_str());
    return 2;
  }
  dump_trace(decoder_raw->trace);
  std::printf("GENERATED_TEXT:[%s]\n", text.c_str());
  return 0;
}
