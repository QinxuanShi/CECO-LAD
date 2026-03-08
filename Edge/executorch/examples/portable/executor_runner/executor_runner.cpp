/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 *
 * This tool can run ExecuTorch model files that only use operators that
 * are covered by the portable kernels, with possible delegate to the
 * test_backend_compiler_lib.
 *
 * It sets all input tensor data to ones, and assumes that the outputs are
 * all fp32 tensors.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <memory>
#include <chrono>

#include <gflags/gflags.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

static uint8_t method_allocator_pool[36 * 1024U * 1024U]; // 4 MB

DEFINE_string(
    model_path,
    "model.pte",
    "Model serialized in flatbuffer format.");

DEFINE_string(
    data_path,
    "os_thre_W150B64L6_sample.txt",
    "Input data for executorch program.");
    
DEFINE_string(
    model_name,
    "Openstack_e3_k1_l8_b64",
    "Input data for executorch program.");
    
DEFINE_int32(
    win_size, 
    100, 
    "Window size for executorch program.");

using namespace torch::executor;
using torch::executor::util::FileDataLoader;
using torch::executor::EValue;
using torch::executor::testing::TensorFactory;

int main(int argc, char** argv) {
  using clock = std::chrono::steady_clock;
  runtime_init();

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc > 2) {
    std::string msg = "Extra commandline args:";
    for (int i = 2 /* skip argv[0] (program name) */; i < argc; i++) {
      msg += std::string(" ") + argv[i];
    }
    ET_LOG(Error, "%s", msg.c_str());
    return 1;
  }

  // Create a loader to get the data of the program file. There are other
  // DataLoaders that use mmap() or point to data that's already in memory, and
  // users can create their own DataLoaders to load from arbitrary sources.
  const char* model_path = FLAGS_model_path.c_str();
  Result<FileDataLoader> loader = FileDataLoader::from(model_path);
  ET_CHECK_MSG(
      loader.ok(),
      "FileDataLoader::from() failed: 0x%" PRIx32,
      (uint32_t)loader.error());

  // Parse the program file. This is immutable, and can also be reused between
  // multiple execution invocations across multiple threads.
  Result<Program> program = Program::load(&loader.get());
  if (!program.ok()) {
    ET_LOG(Error, "Failed to parse model file %s", model_path);
    return 1;
  }
  ET_LOG(Info, "Model file %s is loaded.", model_path);

  // Use the first method in the program.
  const char* method_name = nullptr;
  {
    const auto method_name_result = program->get_method_name(0);
    ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
    method_name = *method_name_result;
  }
  ET_LOG(Info, "Using method %s", method_name);

  // MethodMeta describes the memory requirements of the method.
  Result<MethodMeta> method_meta = program->method_meta(method_name);
  ET_CHECK_MSG(
      method_meta.ok(),
      "Failed to get method_meta for %s: 0x%" PRIx32,
      method_name,
      (uint32_t)method_meta.error());

  //
  // The runtime does not use malloc/new; it allocates all memory using the
  // MemoryManger provided by the client. Clients are responsible for allocating
  // the memory ahead of time, or providing MemoryAllocator subclasses that can
  // do it dynamically.
  //

  // The method allocator is used to allocate all dynamic C++ metadata/objects
  // used to represent the loaded method. This allocator is only used during
  // loading a method of the program, which will return an error if there was
  // not enough memory.
  //
  // The amount of memory required depends on the loaded method and the runtime
  // code itself. The amount of memory here is usually determined by running the
  // method and seeing how much memory is actually used, though it's possible to
  // subclass MemoryAllocator so that it calls malloc() under the hood (see
  // MallocMemoryAllocator).
  //
  // In this example we use a statically allocated memory pool.
  MemoryAllocator method_allocator{
      MemoryAllocator(sizeof(method_allocator_pool), method_allocator_pool)};

  // The memory-planned buffers will back the mutable tensors used by the
  // method. The sizes of these buffers were determined ahead of time during the
  // memory-planning pasees.
  //
  // Each buffer typically corresponds to a different hardware memory bank. Most
  // mobile environments will only have a single buffer. Some embedded
  // environments may have more than one for, e.g., slow/large DRAM and
  // fast/small SRAM, or for memory associated with particular cores.
  std::vector<std::unique_ptr<uint8_t[]>> planned_buffers; // Owns the memory
  std::vector<Span<uint8_t>> planned_spans; // Passed to the allocator
  size_t num_memory_planned_buffers = method_meta->num_memory_planned_buffers();
  for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
    // .get() will always succeed because id < num_memory_planned_buffers.
    size_t buffer_size =
        static_cast<size_t>(method_meta->memory_planned_buffer_size(id).get());
    ET_LOG(Info, "Setting up planned buffer %zu, size %zu.", id, buffer_size);
    planned_buffers.push_back(std::make_unique<uint8_t[]>(buffer_size));
    planned_spans.push_back({planned_buffers.back().get(), buffer_size});
  }
  HierarchicalAllocator planned_memory(
      {planned_spans.data(), planned_spans.size()});

  // Assemble all of the allocators into the MemoryManager that the Executor
  // will use.
  MemoryManager memory_manager(&method_allocator, &planned_memory);

  //
  // Load the method from the program, using the provided allocators. Running
  // the method can mutate the memory-planned buffers, so the method should only
  // be used by a single thread at at time, but it can be reused.
  //

  Result<Method> method = program->load_method(method_name, &memory_manager);
  ET_CHECK_MSG(
      method.ok(),
      "Loading of method %s failed with status 0x%" PRIx32,
      method_name,
      (uint32_t)method.error());
  ET_LOG(Info, "Method loaded.");

  // Allocate input tensors and set all of their elements to 1. The `inputs`
  // variable owns the allocated memory and must live past the last call to
  // `execute()`.

  TensorFactory<ScalarType::Float> tf;
  
  std::vector<float> data_test;
  std::string line;

  const char* data_path = FLAGS_data_path.c_str();
  
  const char* model_name = FLAGS_model_name.c_str();
  
  std::cout << data_path << std::endl;
  
  std::ifstream file(data_path);
  
  std::string x_file = std::string("./ensemble_edge/output/output_x_") + model_name + ".txt";
  std::string s_file = std::string("./ensemble_edge/output/output_series_") + model_name + ".txt";
  std::string p_file = std::string("./ensemble_edge/output/output_prior_") + model_name + ".txt";
  
  std::ofstream outFile(x_file, std::ios::app);
  std::ostringstream oss;
  std::ofstream outFile_series(s_file, std::ios::app);
  std::ostringstream oss_series;
  std::ofstream outFile_prior(p_file, std::ios::app);
  std::ostringstream oss_prior;
  // std::cout << torch::executor::util::evalue_edge_items(500);
  if (!outFile) {
    std::cerr << "Error opening file for writing!" << std::endl;
    return 1;
  }
  int count = 0;
  int sample_num = 0;
  int window = FLAGS_win_size; 
  if (file.is_open()) {
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        while (std::getline(iss, token, ' ')) {
            try {
              float value = std::stof(token); // Convert string to float
              data_test.push_back(value);
              count++;
              if(count == 1*window*10){
                sample_num++;
                std::cout << "Float data read from file:\n";
                std::cout << "current sample number is " << sample_num << std::endl;

                // set input to method and predict
                EValue value_from_input(tf.make({1, window, 10}, data_test));

                Error status_t = method->set_input(value_from_input, 0);
                ET_CHECK(status_t == Error::Ok);
                ET_LOG(Info, "Inputs set successfully.");
                // EValue input_check = method->get_input(test_input_size-1);
                // std::cout << "input check " << input_check << std::endl;

                // auto inputs = util::prepare_input_tensors(*method);
                // ET_CHECK_MSG(
                //     inputs.ok(),
                //     "Could not prepare inputs: 0x%" PRIx32,
                //     (uint32_t)inputs.error());
                // ET_LOG(Info, "Inputs prepared.");

                auto t0 = clock::now();

                // Run the model.
                Error status = method->execute();
                ET_CHECK_MSG(
                    status == Error::Ok,
                    "Execution of method %s failed with status 0x%" PRIx32,
                    method_name,
                    (uint32_t)status);
                    
                auto t1 = clock::now();
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
                std::cout << "Elapsed: " << ms << " ms\n";
                
                ET_LOG(Info, "Model executed successfully.");

                // Print the outputs.
                std::vector<EValue> outputs(method->outputs_size());
                ET_LOG(Info, "%zu outputs: ", outputs.size());
                status = method->get_outputs(outputs.data(), outputs.size());
                ET_CHECK(status == Error::Ok);
                // Print the first and last 500 elements of long lists of scalars.
                // std::cout << torch::executor::util::evalue_edge_items(500);
                
                oss.str("");
                oss.clear();
                oss << torch::executor::util::evalue_edge_items(1*window*10/2) << outputs[0];
                outFile << oss.str();
                
                for (int i = 1; i < 4; ++i) {
                  oss_series.str("");  // Clear the stream
                  oss_series.clear();
                  oss_series << torch::executor::util::evalue_edge_items(8*window*window/2) << outputs[i];
                  outFile_series << oss_series.str();
                }
                
                for (int j = 4; j < 7; ++j) {
                  oss_prior.str(""); // Clear the stream
                  oss_prior.clear();
                  oss_prior << torch::executor::util::evalue_edge_items(8*window*window/2) << outputs[j];
                  outFile_prior << oss_prior.str();
                }

                // reset
                count = 0;
                data_test.clear();
              }
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid argument: " << e.what() << std::endl;
            } catch (const std::out_of_range& e) {
                std::cerr << "Out of range: " << e.what() << std::endl;
            }
        }
    }
    
    file.close();
    outFile.close();
    outFile_series.close();
    outFile_prior.close();
  } else {
      ET_LOG(Info, "Unable to open file.");
  }

  return 0;
}
