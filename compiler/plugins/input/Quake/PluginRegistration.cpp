// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "quake/Dialect/Quake/QuakeDialect.h"  // Adjust this include path as necessary
#include "quake/Conversion/Passes.h"           // Assumed header for conversion pipeline

namespace mlir::iree_compiler::quake {

namespace {

struct QuakeOptions {
  void bindOptions(OptionsBinder &binder) {
    // No CLI options yet
  }
};

// Plugin for Quake (CUDA-Q) input.
struct QuakeSession
    : public PluginSession<QuakeSession, QuakeOptions,
                           PluginActivationPolicy::DefaultActivated> {
  static void registerPasses() {
    // Register passes needed for Quake conversion.
    registerQuakeConversionPasses(); // You need to implement this
  }

  void onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<quake::QuakeDialect>();
  }

  bool extendCustomInputConversionPassPipeline(
      OpPassManager &passManager, std::string_view typeMnemonic) override {
    if (typeMnemonic == "quake") {
      buildQuakeInputConversionPassPipeline(passManager);
      return true;
    }
    return false;
  }

  void populateCustomInputConversionTypes(StringSet<> &typeMnemonics) override {
    typeMnemonics.insert("quake");
  }

  void populateDetectedCustomInputConversionTypes(
      ModuleOp &module, StringSet<> &typeMnemonics) override {
    auto *ctx = module.getContext();
    const Dialect *quakeDialect = ctx->getLoadedDialect("quake");

    bool hasQuakeOps = false;
    module.walk([&](Operation *op) {
      if (op->getDialect() == quakeDialect) {
        hasQuakeOps = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (hasQuakeOps) {
      typeMnemonics.insert("quake");
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::quake


// This macro registers command-line options for your plugin using 
// LLVM’s command-line flag system.
IREE_DEFINE_COMPILER_OPTION_FLAGS(::mlir::iree_compiler::quake::QuakeOptions);


// This is the entry point IREE will look for when loading 
// quake plugin at runtime.
extern "C" bool iree_register_compiler_plugin_input_quake(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<::mlir::iree_compiler::quake::QuakeSession>(
      "input_quake");
  return true;
}
