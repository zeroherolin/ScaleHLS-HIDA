//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "scalehls/Transforms/Passes.h"

namespace mlir {
namespace scalehls {
#define GEN_PASS_DEF_FOLDAFFINESUBVIEWACCESS
#include "scalehls/Transforms/Passes.h.inc"
} // namespace scalehls
} // namespace mlir

using namespace mlir;
using namespace scalehls;

/// Materializes the index values of an affine access map.
static SmallVector<Value> expandAccessIndices(PatternRewriter &rewriter,
                                              Location loc, AffineMap map,
                                              ValueRange mapOperands) {
  SmallVector<Value> indices;
  for (auto expr : map.getResults()) {
    auto subMap = AffineMap::get(map.getNumDims(), map.getNumSymbols(), expr);
    indices.push_back(
        rewriter.create<affine::AffineApplyOp>(loc, subMap, mapOperands));
  }
  return indices;
}

namespace {
/// The upstream FoldMemRefAliasOps pass no longer folds affine accesses,
/// but the HLS C++ emitter cannot express subviews. Rewrite affine
/// loads/stores of subviews into memref accesses of the base buffer.
struct FoldAffineLoadOfSubView
    : public OpRewritePattern<affine::AffineLoadOp> {
  using OpRewritePattern<affine::AffineLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineLoadOp load,
                                PatternRewriter &rewriter) const override {
    auto subview = load.getMemRef().getDefiningOp<memref::SubViewOp>();
    if (!subview || !subview.hasUnitStride())
      return failure();

    auto indices = expandAccessIndices(rewriter, load.getLoc(),
                                       load.getAffineMap(),
                                       load.getMapOperands());
    SmallVector<Value> sourceIndices;
    affine::resolveIndicesIntoOpWithOffsetsAndStrides(
        rewriter, load.getLoc(), subview.getMixedOffsets(),
        subview.getMixedStrides(), subview.getDroppedDims(), indices,
        sourceIndices);
    rewriter.replaceOpWithNewOp<memref::LoadOp>(load, subview.getSource(),
                                                sourceIndices);
    return success();
  }
};

struct FoldAffineStoreOfSubView
    : public OpRewritePattern<affine::AffineStoreOp> {
  using OpRewritePattern<affine::AffineStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineStoreOp store,
                                PatternRewriter &rewriter) const override {
    auto subview = store.getMemRef().getDefiningOp<memref::SubViewOp>();
    if (!subview || !subview.hasUnitStride())
      return failure();

    auto indices = expandAccessIndices(rewriter, store.getLoc(),
                                       store.getAffineMap(),
                                       store.getMapOperands());
    SmallVector<Value> sourceIndices;
    affine::resolveIndicesIntoOpWithOffsetsAndStrides(
        rewriter, store.getLoc(), subview.getMixedOffsets(),
        subview.getMixedStrides(), subview.getDroppedDims(), indices,
        sourceIndices);
    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        store, store.getValueToStore(), subview.getSource(), sourceIndices);
    return success();
  }
};
} // namespace

namespace {
struct FoldAffineSubViewAccess
    : public scalehls::impl::FoldAffineSubViewAccessBase<
          FoldAffineSubViewAccess> {
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();

    mlir::RewritePatternSet patterns(context);
    patterns.add<FoldAffineLoadOfSubView>(context);
    patterns.add<FoldAffineStoreOfSubView>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> scalehls::createFoldAffineSubViewAccessPass() {
  return std::make_unique<FoldAffineSubViewAccess>();
}
