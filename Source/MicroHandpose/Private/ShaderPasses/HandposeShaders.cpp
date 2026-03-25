#include "ShaderPasses/HandposeShaders.h"

IMPLEMENT_GLOBAL_SHADER(FLetterboxResizeCS,  "/Plugin/MicroHandpose/Private/ImagePreprocess.usf", "LetterboxResize", SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FConv5x5Stride2CS,   "/Plugin/MicroHandpose/Private/Conv5x5Stride2.usf",  "Conv5x5Stride2",  SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FConv3x3Stride2CS,   "/Plugin/MicroHandpose/Private/Conv3x3Stride2.usf",  "Conv3x3Stride2",  SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FDepthwiseConv5x5CS, "/Plugin/MicroHandpose/Private/DepthwiseConv5x5.usf","DepthwiseConv5x5", SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FPointwiseConvCS,    "/Plugin/MicroHandpose/Private/PointwiseConv.usf",   "PointwiseConv",   SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FUpsample2xCS,       "/Plugin/MicroHandpose/Private/Upsample2x.usf",     "Upsample2x",     SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FFusedDwPwCS,        "/Plugin/MicroHandpose/Private/FusedDwPw.usf",       "FusedDwPw",       SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FOutputHeadsCS,      "/Plugin/MicroHandpose/Private/OutputHeads.usf",     "OutputHeads",     SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FAffineCropCS,       "/Plugin/MicroHandpose/Private/AffineCrop.usf",      "AffineCrop",      SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FElementwiseAddCS,   "/Plugin/MicroHandpose/Private/ElementwiseOps.usf",  "ElementwiseAdd",  SF_Compute);

// EfficientNet-specific shaders
IMPLEMENT_GLOBAL_SHADER(FDepthwiseConvBnRelu6CS,   "/Plugin/MicroHandpose/Private/DepthwiseConvBnRelu6.usf",  "DepthwiseConvBnRelu6",  SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FConv1x1BnCS,              "/Plugin/MicroHandpose/Private/Conv1x1Bn.usf",             "Conv1x1Bn",             SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FGlobalAvgPoolCS,           "/Plugin/MicroHandpose/Private/GlobalAvgPool.usf",         "GlobalAvgPool",         SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FFCMatMulCS,                "/Plugin/MicroHandpose/Private/FCMatMul.usf",              "FCMatMul",              SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FConv3x3Stride2BnRelu6CS,  "/Plugin/MicroHandpose/Private/Conv3x3Stride2BnRelu6.usf", "Conv3x3Stride2BnRelu6", SF_Compute);
