//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "scalehls/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace scalehls {
#define GEN_PASS_DEF_LINALGANALYZEMODEL
#include "scalehls/Transforms/Passes.h.inc"
} // namespace scalehls
} // namespace mlir


using namespace mlir;
using namespace scalehls;

namespace {
struct LinalgAnalyzeModel : public scalehls::impl::LinalgAnalyzeModelBase<LinalgAnalyzeModel> {
  void runOnOperation() override {
    auto func = getOperation();

    // auto getIntArray = [&](ArrayAttr arrayAttr) {
    //   SmallVector<int64_t, 4> array;
    //   for (auto value : arrayAttr)
    //     array.push_back(cast<IntegerAttr>(value).getInt());
    //   return array;
    // };

    unsigned layerIdx = 0;
    unsigned long numOps = 0;

    for (auto &op : func.getOps()) {
      if (auto convOp = dyn_cast<linalg::Conv2DNchwFchwOp>(op)) {
        auto weightShape =
            cast<RankedTensorType>(convOp.filter().getType()).getShape();
        auto outputShape =
            cast<RankedTensorType>(convOp->getResult(0).getType()).getShape();

        auto batch = outputShape[0];
        auto height = outputShape[2];
        auto width = outputShape[3];
        auto in_filter = weightShape[0];
        auto out_filter = weightShape[1];
        auto hkernel = weightShape[2];
        auto wkernel = weightShape[3];

        auto ops = 2 * batch * height * width * in_filter * out_filter *
                   hkernel * wkernel;
        numOps += ops;

        llvm::dbgs() << "(\"conv" << layerIdx << "\", Workload(" << batch
                     << ", " << height << ", " << width << ", " << in_filter
                     << ", " << out_filter << ", " << hkernel << ", " << wkernel
                     << "), Complexity(" << ops << ")),\n";
        ++layerIdx;

      } else if (auto depthOp =
                     dyn_cast<linalg::DepthwiseConv2DNchwChwOp>(op)) {
        auto weightShape =
            cast<RankedTensorType>(depthOp.filter().getType()).getShape();
        auto outputShape =
            cast<RankedTensorType>(depthOp->getResult(0).getType()).getShape();

        auto batch = outputShape[0];
        auto height = outputShape[2];
        auto width = outputShape[3];
        auto filter = weightShape[0];
        auto hkernel = weightShape[1];
        auto wkernel = weightShape[2];

        auto ops = 2 * batch * height * width * filter * hkernel * wkernel;
        numOps += ops;

        llvm::dbgs() << "(\"depth_conv" << layerIdx << "\", Workload(" << batch
                     << ", " << height << ", " << width << ", " << filter
                     << ", " << hkernel << ", " << wkernel << "), Complexity("
                     << ops << ")),\n";
        ++layerIdx;

      } else if (auto gemmOp = dyn_cast<linalg::MatmulOp>(op)) {
        auto inputShape =
            cast<RankedTensorType>(gemmOp.getOperand(0).getType()).getShape();
        auto weightShape =
            cast<RankedTensorType>(gemmOp.getOperand(1).getType()).getShape();

        auto batch = inputShape[0];
        auto in_filter = weightShape[0];
        auto out_filter = weightShape[1];

        auto ops = 2 * batch * in_filter * out_filter;
        numOps += ops;

        llvm::dbgs() << "(\"matmul" << layerIdx << "\", Workload(" << batch
                     << ", " << in_filter << ", " << out_filter
                     << "), Complexity(" << ops << ")),\n";
        ++layerIdx;
      }
    }

    llvm::dbgs() << "TOTAL OPS NUM: " << numOps << "\n";
  }
};
} // namespace

std::unique_ptr<Pass> scalehls::createLinalgAnalyzeModelPass() {
  return std::make_unique<LinalgAnalyzeModel>();
}
