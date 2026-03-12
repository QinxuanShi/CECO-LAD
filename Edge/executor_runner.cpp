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

#include <algorithm>
#include <iostream>
#include <memory>

#include <gflags/gflags.h>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>

static uint8_t method_allocator_pool[4 * 1024U * 1024U]; // 4 MB

DEFINE_string(
    model_path,
    "Openstack_e3_k3_l3_b32.pte",
    "Model serialized in flatbuffer format.");

DEFINE_string(
    data_path,
    "os_processed_data.txt",
    "Input data (train or test) for executorch program.");

DEFINE_string(
    mode,
    "test",
    "Run mode: 'train' or 'test' (controls data usage and output name).");

DEFINE_string(
    model_name,
    "Openstack_e3_k3_l3_b32",
    "Input data for executorch program.");

DEFINE_int32(win_size, 100, "Window size for executorch program.");

using namespace torch::executor;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;
using torch::executor::util::FileDataLoader;
// using torch::executor::native;
//  using executorch::runtime::Error;
//  using executorch::runtime::EValue;
//  using executorch::runtime::HierarchicalAllocator;
//  using executorch::runtime::MemoryAllocator;
//  using executorch::runtime::MemoryManager;
//  using executorch::runtime::Method;
//  using executorch::runtime::MethodMeta;
//  using executorch::runtime::Program;
//  using executorch::runtime::Result;
//  using executorch::runtime::Span;

std::vector<float> convert_to_vector(torch::executor::Tensor &exeTensor)
{
  // Get the pointer of tensor
  auto sizes = exeTensor.sizes();
  std::vector<int64_t> shape(sizes.begin(), sizes.end());
  float *data_ptr = exeTensor.data_ptr<float>();

  // Copy the tensor to array
  std::vector<float> array(exeTensor.numel());
  std::memcpy(array.data(), data_ptr, exeTensor.numel() * sizeof(float));
  return array;
}

int main(int argc, char **argv)
{
  runtime_init();

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc != 1)
  {
    std::string msg = "Extra commandline args:";
    for (int i = 1 /* skip argv[0] (program name) */; i < argc; i++)
    {
      msg += std::string(" ") + argv[i];
    }
    ET_LOG(Error, "%s", msg.c_str());
    return 1;
  }

  // Create a loader to get the data of the program file. There are other
  // DataLoaders that use mmap() or point to data that's already in memory, and
  // users can create their own DataLoaders to load from arbitrary sources.
  const char *model_path = FLAGS_model_path.c_str();
  Result<FileDataLoader> loader = FileDataLoader::from(model_path);
  ET_CHECK_MSG(
      loader.ok(),
      "FileDataLoader::from() failed: 0x%" PRIx32,
      (uint32_t)loader.error());

  const char *data_path = FLAGS_data_path.c_str();
  const char *model_name_flag = FLAGS_model_name.c_str();
  std::string mode = FLAGS_mode;

  if (mode != "train" && mode != "test")
  {
    std::cerr << "Invalid mode: " << mode
              << ". Expected 'train' or 'test'." << std::endl;
    return 1;
  }

  // Parse the program file. This is immutable, and can also be reused between
  // multiple execution invocations across multiple threads.
  Result<Program> program = Program::load(&loader.get());
  if (!program.ok())
  {
    ET_LOG(Error, "Failed to parse model file %s", model_path);
    return 1;
  }
  ET_LOG(Info, "Model file %s is loaded.", model_path);

  // Use the first method in the program.
  const char *method_name = nullptr;
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
  std::vector<Span<uint8_t>> planned_spans;                // Passed to the allocator
  size_t num_memory_planned_buffers = method_meta->num_memory_planned_buffers();
  for (size_t id = 0; id < num_memory_planned_buffers; ++id)
  {
    // .get() will always succeed because id < num_memory_planned_buffers.
    size_t buffer_size =
        static_cast<size_t>(method_meta->memory_planned_buffer_size(id)
                                .get()); // define the ram of the program
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

  size_t test_input_size = method->inputs_size();
  ET_LOG(Info, "input size is %zu.", test_input_size);

  std::string datafilename = std::string(data_path);
  // Output anomaly scores will be saved to this file, using model name and
  // mode (train/test)
  std::string scorefilename = std::string("prediction_results/") +
                              std::string(model_name_flag) +
                              (mode == "train" ? std::string("_train_score.txt")
                                               : std::string("_test_score.txt"));

  int window_size = FLAGS_win_size;

  // Track latency only for test, not train
  int test_iterations = 0;
  double test_totalDuration = 0.0; // total time for test only
  TensorFactory<ScalarType::Float> tf;

  if (mode == "train")
  {
    std::cout << "compute the train energy" << std::endl;
    std::ifstream dataFile(datafilename);

    if (!dataFile.is_open())
    {
      std::cerr << "Failed to open file: " << datafilename << std::endl;
      return 1;
    }
    std::string line;
    std::vector<float> train_energy;

    int count = 0;
    int times = 0;
    std::vector<float> data_train;
    while (std::getline(dataFile, line))
    {
      std::stringstream s(line);
      std::string word;
      while (getline(s, word, ','))
      {
        data_train.push_back(std::stof(word));
        count++;
        if (count == 1 * window_size * 10)
        {
          times++;
          EValue value_from_input(tf.make({1, window_size, 10}, data_train));
          Error status_t = method->set_input(value_from_input, 0);
          ET_CHECK(status_t == Error::Ok);

          Error status = method->execute();

          ET_CHECK_MSG(
              status == Error::Ok,
              "Execution of method %s failed with status 0x%" PRIx32,
              method_name,
              (uint32_t)status);
          std::vector<EValue> outputs(method->outputs_size());
          status = method->get_outputs(outputs.data(), outputs.size());
          std::vector<float> output = convert_to_vector(outputs[0].toTensor());

          train_energy.insert(train_energy.end(), output.begin(), output.end());

          // reset
          count = 0;
          data_train.clear();
        }
      }
    }

    std::ofstream trainScoreFile(scorefilename);
    if (!trainScoreFile.is_open())
    {
      std::cerr << "Failed to open train score file: " << scorefilename
                << std::endl;
      return 1;
    }
    for (const auto &score : train_energy)
    {
      trainScoreFile << score << '\n';
    }
    trainScoreFile.close();

    std::cout << "Saved train scores to "
              << scorefilename << std::endl;
  }
  else if (mode == "test")
  {
    std::cout << "compute the test energy" << std::endl;
    std::ifstream dataFile(datafilename);

    if (!dataFile.is_open())
    {
      std::cerr << "Failed to open file: " << datafilename << std::endl;
      return 1;
    }
    std::string line;
    std::vector<float> test_energy;

    int count = 0;
    std::vector<float> data_test;
    while (std::getline(dataFile, line))
    {
      std::stringstream s(line);
      std::string word;
      while (getline(s, word, ','))
      {
        data_test.push_back(std::stof(word));
        count++;
        if (count == 1 * window_size * 10)
        {
          EValue value_from_input(tf.make({1, window_size, 10}, data_test));
          Error status_t = method->set_input(value_from_input, 0);
          ET_CHECK(status_t == Error::Ok);

          auto start = std::chrono::high_resolution_clock::now();
          Error status = method->execute();
          auto end = std::chrono::high_resolution_clock::now();

          std::chrono::duration<double> duration = end - start;
          test_totalDuration += duration.count();
          test_iterations++;

          ET_CHECK_MSG(
              status == Error::Ok,
              "Execution of method %s failed with status 0x%" PRIx32,
              method_name,
              (uint32_t)status);
          std::vector<EValue> outputs(method->outputs_size());
          status = method->get_outputs(outputs.data(), outputs.size());
          std::vector<float> output = convert_to_vector(outputs[0].toTensor());

          test_energy.insert(test_energy.end(), output.begin(), output.end());

          // reset
          count = 0;
          data_test.clear();
        }
      }
    }

    std::ofstream testScoreFile(scorefilename);
    if (!testScoreFile.is_open())
    {
      std::cerr << "Failed to open test score file: " << scorefilename
                << std::endl;
      return 1;
    }
    for (const auto &score : test_energy)
    {
      testScoreFile << score << '\n';
    }
    testScoreFile.close();

    std::cout << "Saved test scores to "
              << scorefilename << std::endl;

    std::cout << "the latency of the model on test data is " << std::endl;
    std::cout << "total time: " << test_totalDuration
              << "s   with the number is " << test_iterations << std::endl;
    if (test_iterations > 0)
    {
      std::cout << "Average time: " << test_totalDuration / test_iterations
                << "s" << std::endl;
    }
  }

  return 0;
}
