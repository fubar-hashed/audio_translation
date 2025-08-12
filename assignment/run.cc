// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stddef.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>
#define DR_WAV_IMPLEMENTATION
#include "../../dr_wav.h"

// Placeholder for internal header, do not modify.
#include "gemma/gemma.h"
#include "gemma/tokenizer.h"
#include "util/app.h"  // LoaderArgs
#include "util/args.h"
#include "util/threading.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

int main(int argc, char** argv) {
  {
    // Placeholder for internal init, do not modify.
  }

  gcpp::LoaderArgs loader(argc, argv);
  gcpp::InferenceArgs inference(argc, argv);
  gcpp::AppArgs app(argc, argv);
  if (gcpp::HasHelp(argc, argv)) {
    loader.Help();
    return 0;
  } else if (const char* error = loader.Validate()) {
    loader.Help();
    HWY_ABORT("\nInvalid args: %s", error);
  }

  // Demonstrate constrained decoding by never outputting certain tokens.
  std::set<int> reject_tokens;
  for (int arg = 0; arg < argc; ++arg) {
    // Find a --reject flag and consume everything after it.
    if (strcmp(argv[arg], "--reject") == 0) {
      while (++arg < argc) reject_tokens.insert(atoi(argv[arg]));
    }
  }

  // Instantiate model and KV Cache
  gcpp::BoundedTopology topology(gcpp::CreateTopology(app));
  gcpp::NestedPools pools = gcpp::CreatePools(topology, app);
  gcpp::MatMulEnv env(topology, pools);
  gcpp::Gemma model = gcpp::CreateGemma(loader, env);
  gcpp::KVCache kv_cache =
      gcpp::KVCache::Create(model.GetModelConfig(),
                            inference.prefill_tbatch_size);
  size_t generated = 0;

  // Initialize random number generator
  std::mt19937 gen;
  std::random_device rd;
  gen.seed(rd());


  // read wav file.
  

  // Open WAV file and read PCM samples
  drwav wav;
  if (!drwav_init_file(&wav, "../../roses-are.wav", NULL)) {
    printf("Failed to open WAV file\n");
    return -1;
  }

  int16_t* pSampleData = (int16_t*)malloc((size_t)wav.totalPCMFrameCount * wav.channels * sizeof(int16_t));
  size_t framesRead = drwav_read_pcm_frames_s16(&wav, wav.totalPCMFrameCount, pSampleData);

  drwav_uint64 total_samples = wav.totalPCMFrameCount * wav.channels;



  // ... pSampleData now contains the audio PCM samples
  
  std::cout << "Sample rate : " << wav.sampleRate << "\n";
  std::cout << "Channels : " << wav.channels << "\n";


  // Tokenize instructions.
  //
  //

  std::string prompt = "<bos><start_of_turn>user: You are an experience spanish to english translator. The following is a text in english, translate and give the corresponding sentence in spanish. Do not add anything else from your end. text : Roses are red, Violets are blue<end_of_turn><start_of_turn>model: ";


/*  std::string transcribe_prompt = std::string("<bos><start_of_turn> user: You are an expert speech to text transcriber. The following is a segment of raw PCM audio data extracted from a .wav file. 1. Sample rate: 48000 Hz \n 2. Channels :: mono 3. sample format :: 16 bit signed integers 4. values are sequential waveform samples. Transcribe any spoken word in this segment into plain text. Output only the spoken word and do not add anything else from your end.\n<audio>");
  for (drwav_uint64 i = 0; i < total_samples; i++) {
	  prompt = prompt + std::to_string(*pSampleData);
	  pSampleData++;
  }
  transcribe_prompt = transcribe_prompt +"</audio><end_of_turn><start_of_turn>model :";//"Write a greeting to the world.";*/
  std::cout << "prompt :: "<< prompt << "\n" ;
  const std::vector<int> tokens = gcpp::WrapAndTokenize(
      model.Tokenizer(), loader.Info(), generated, prompt);
  const size_t prompt_size = tokens.size();
  drwav_uninit(&wav);

  // This callback function gets invoked every time a token is generated
  auto stream_token = [&generated, &prompt_size, &model](int token, float) {
    ++generated;
    if (generated < prompt_size) {
      // print feedback
    } else if (!model.GetModelConfig().IsEOS(token)) {
      std::string token_text;
      HWY_ASSERT(model.Tokenizer().Decode({token}, &token_text));
      std::cout << token_text << std::flush;
    }
    return true;
  };

  gcpp::TimingInfo timing_info;
  gcpp::RuntimeConfig runtime_config = {
      .max_generated_tokens = 4096,
      .temperature = 1.0,
      .gen = &gen,
      .verbosity = 0,
      .stream_token = stream_token,
      .accept_token =
          [&](int token, float /* prob */) {
            return !reject_tokens.contains(token);
          },
  };
  model.Generate(runtime_config, tokens, 0, kv_cache, timing_info);
}
