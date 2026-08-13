//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "scalehls/Transforms/Passes.h"

namespace mlir {
namespace scalehls {
#define GEN_PASS_DEF_TOSAFAKEQUANTIZE
#include "scalehls/Transforms/Passes.h.inc"
} // namespace scalehls
} // namespace mlir


using namespace mlir;
using namespace scalehls;

static Type getQuantizeType(Type type) {
  auto i8Type = IntegerType::get(type.getContext(), 8);
  if (isa<Float32Type>(type))
    return i8Type;

  if (auto tensorType = dyn_cast<RankedTensorType>(type))
    if (isa<Float32Type>(tensorType.getElementType()))
      return RankedTensorType::get(tensorType.getShape(), i8Type);

  return nullptr;
}

namespace {
/// This pass is only for testing use!!! To really support quantized model,
/// first we need to have front-ends, such as Torch-MLIR, to support the model
/// quantization, which has not came true unfortunately.
struct TosaFakeQuantize : public scalehls::impl::TosaFakeQuantizeBase<TosaFakeQuantize> {
  void runOnOperation() override {
    auto module = getOperation();

    // Convert the type of block arguments.
    module.walk([&](Block *block) {
      for (auto arg : block->getArguments())
        if (auto quantType = getQuantizeType(arg.getType()))
          arg.setType(quantType);
    });

    // Convert the type of operation results. Also, handle function, constant,
    // conv2d, and matmul operations.
    int8_t fakeIdx = 1;
    module.walk([&](Operation *op) {
      for (auto result : op->getResults())
        if (auto quantType = getQuantizeType(result.getType())) {
          result.setType(quantType);

          if (auto constant = dyn_cast<tosa::ConstOp>(op)) {
            // Because we are not trying to really quantize the model, here we
            // just assign a fake value to the constant operation.
            SmallVector<int8_t, 64> list(constant.getValues().size(), fakeIdx++);
            // for (auto value : constant.valueAttr().getValues<float>())
            //   list.push_back(value);

            auto quantValue = DenseIntElementsAttr::get(cast<ShapedType>(quantType), list);
            constant->setAttr(constant.getValuesAttrName(), quantValue);
          }

          // Upstream TOSA removed the quantization_info attribute in
          // favour of explicit zero-point operands; the zero-valued
          // fake quantization info is simply dropped here.
        }

      // As we have updated the type of all values in the function, we can
      // safely convert the function type as well.
      if (auto func = dyn_cast<func::FuncOp>(op))
        func.setType(FunctionType::get(
            func.getContext(), func.front().getArgumentTypes(),
            func.back().getTerminator()->getOperandTypes()));
    });
  }
};
} // namespace

std::unique_ptr<Pass> scalehls::createTosaFakeQuantizePass() {
  return std::make_unique<TosaFakeQuantize>();
}
