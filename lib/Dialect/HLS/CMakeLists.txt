add_mlir_dialect_library(MLIRHLS
  HLS.cpp
  Analysis.cpp
  Utils.cpp

  DEPENDS
  MLIRHLSOpsIncGen
  MLIRHLSEnumsIncGen
  MLIRHLSAttributesIncGen
  MLIRHLSInterfacesIncGen

  # Declare the analysis dependencies explicitly so that single-pass GNU ld
  # archive resolution orders them after libMLIRHLS.a.
  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRAffineAnalysis
  MLIRPresburger
  )
