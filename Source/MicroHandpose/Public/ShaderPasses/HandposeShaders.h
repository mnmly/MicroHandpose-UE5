#pragma once

#include "CoreMinimal.h"
#include "GlobalShader.h"
#include "ShaderParameterStruct.h"

// -----------------------------------------------------------------------
// Image Preprocessing — Letterbox resize + texture → CHW float buffer
// -----------------------------------------------------------------------

class FLetterboxResizeCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FLetterboxResizeCS);
	SHADER_USE_PARAMETER_STRUCT(FLetterboxResizeCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER(int32, SrcWidth)
		SHADER_PARAMETER(int32, SrcHeight)
		SHADER_PARAMETER(int32, DstSize)
		SHADER_PARAMETER(float, ScaleX)
		SHADER_PARAMETER(float, ScaleY)
		SHADER_PARAMETER(float, OffsetX)
		SHADER_PARAMETER(float, OffsetY)
		SHADER_PARAMETER_RDG_TEXTURE_SRV(Texture2D<float4>, InputTexture)
		SHADER_PARAMETER_SAMPLER(SamplerState, InputSampler)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<float>, OutputBuffer)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

// -----------------------------------------------------------------------
// 5x5 Stride-2 Convolution + PReLU (Palm detection initial conv)
// -----------------------------------------------------------------------

class FConv5x5Stride2CS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FConv5x5Stride2CS);
	SHADER_USE_PARAMETER_STRUCT(FConv5x5Stride2CS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER(int32, InWidth)
		SHADER_PARAMETER(int32, InHeight)
		SHADER_PARAMETER(int32, InChannels)
		SHADER_PARAMETER(int32, OutWidth)
		SHADER_PARAMETER(int32, OutHeight)
		SHADER_PARAMETER(int32, OutChannels)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, InputBuffer)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, Weights)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, Bias)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, Alpha)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<float>, OutputBuffer)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

// -----------------------------------------------------------------------
// 3x3 Stride-2 Convolution + ReLU (Landmark initial conv)
// -----------------------------------------------------------------------

class FConv3x3Stride2CS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FConv3x3Stride2CS);
	SHADER_USE_PARAMETER_STRUCT(FConv3x3Stride2CS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER(int32, InWidth)
		SHADER_PARAMETER(int32, InHeight)
		SHADER_PARAMETER(int32, InChannels)
		SHADER_PARAMETER(int32, OutWidth)
		SHADER_PARAMETER(int32, OutHeight)
		SHADER_PARAMETER(int32, OutChannels)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, InputBuffer)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, Weights)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, Bias)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<float>, OutputBuffer)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

// -----------------------------------------------------------------------
// Depthwise 5x5 Convolution (no activation)
// -----------------------------------------------------------------------

class FDepthwiseConv5x5CS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FDepthwiseConv5x5CS);
	SHADER_USE_PARAMETER_STRUCT(FDepthwiseConv5x5CS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER(int32, InWidth)
		SHADER_PARAMETER(int32, InHeight)
		SHADER_PARAMETER(int32, Channels)
		SHADER_PARAMETER(int32, OutWidth)
		SHADER_PARAMETER(int32, OutHeight)
		SHADER_PARAMETER(int32, Stride)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, InputBuffer)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, Weights)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, Bias)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<float>, OutputBuffer)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

// -----------------------------------------------------------------------
// Pointwise 1x1 Convolution + Skip + Activation (PReLU/ReLU/None)
// -----------------------------------------------------------------------

class FPointwiseConvCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FPointwiseConvCS);
	SHADER_USE_PARAMETER_STRUCT(FPointwiseConvCS, FGlobalShader);

	// Permutation domain for activation mode
	class FActivationMode : SHADER_PERMUTATION_INT("ACTIVATION_MODE", 3); // 0=None, 1=ReLU, 2=PReLU
	class FHasSkip : SHADER_PERMUTATION_BOOL("HAS_SKIP");
	using FPermutationDomain = TShaderPermutationDomain<FActivationMode, FHasSkip>;

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER(int32, InWidth)
		SHADER_PARAMETER(int32, InHeight)
		SHADER_PARAMETER(int32, InChannels)
		SHADER_PARAMETER(int32, OutWidth)
		SHADER_PARAMETER(int32, OutHeight)
		SHADER_PARAMETER(int32, OutChannels)
		SHADER_PARAMETER(int32, Stride)
		SHADER_PARAMETER(int32, ChannelPad)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, InputBuffer)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, SkipBuffer)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, Weights)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, Bias)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, Alpha)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<float>, OutputBuffer)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

// -----------------------------------------------------------------------
// Bilinear 2x Upsample + Optional Add
// -----------------------------------------------------------------------

class FUpsample2xCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FUpsample2xCS);
	SHADER_USE_PARAMETER_STRUCT(FUpsample2xCS, FGlobalShader);

	class FHasAdd : SHADER_PERMUTATION_BOOL("HAS_ADD");
	using FPermutationDomain = TShaderPermutationDomain<FHasAdd>;

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER(int32, InWidth)
		SHADER_PARAMETER(int32, InHeight)
		SHADER_PARAMETER(int32, Channels)
		SHADER_PARAMETER(int32, OutWidth)
		SHADER_PARAMETER(int32, OutHeight)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, InputBuffer)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, AddBuffer)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<float>, OutputBuffer)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

// -----------------------------------------------------------------------
// Fused Depthwise + Pointwise (shared memory, for landmark model layers 24-42)
// -----------------------------------------------------------------------

class FFusedDwPwCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FFusedDwPwCS);
	SHADER_USE_PARAMETER_STRUCT(FFusedDwPwCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER(int32, SpatialWidth)
		SHADER_PARAMETER(int32, SpatialHeight)
		SHADER_PARAMETER(int32, Channels)
		SHADER_PARAMETER(int32, Stride)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, InputBuffer)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, SkipBuffer)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, DwWeights)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, DwBias)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, PwWeights)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, PwBias)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<float>, OutputBuffer)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

// -----------------------------------------------------------------------
// Output Heads — FC + Sigmoid or /256
// -----------------------------------------------------------------------

class FOutputHeadsCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FOutputHeadsCS);
	SHADER_USE_PARAMETER_STRUCT(FOutputHeadsCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER(int32, InChannels)
		SHADER_PARAMETER(int32, InSpatialSize)
		SHADER_PARAMETER(int32, NumOutputs)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, InputBuffer)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, Weights)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, Bias)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<float>, OutputBuffer)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

// -----------------------------------------------------------------------
// Affine Crop — ROI crop with rotation + bilinear interpolation
// -----------------------------------------------------------------------

class FAffineCropCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FAffineCropCS);
	SHADER_USE_PARAMETER_STRUCT(FAffineCropCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER(int32, DstSize)
		SHADER_PARAMETER(FMatrix44f, AffineMatrix)
		SHADER_PARAMETER_RDG_TEXTURE_SRV(Texture2D<float4>, InputTexture)
		SHADER_PARAMETER_SAMPLER(SamplerState, InputSampler)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<float>, OutputBuffer)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

// -----------------------------------------------------------------------
// Depthwise Conv (variable kernel) + BN + ReLU6 (EfficientNet)
// -----------------------------------------------------------------------

class FDepthwiseConvBnRelu6CS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FDepthwiseConvBnRelu6CS);
	SHADER_USE_PARAMETER_STRUCT(FDepthwiseConvBnRelu6CS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER(int32, InWidth)
		SHADER_PARAMETER(int32, InHeight)
		SHADER_PARAMETER(int32, Channels)
		SHADER_PARAMETER(int32, OutWidth)
		SHADER_PARAMETER(int32, OutHeight)
		SHADER_PARAMETER(int32, Stride)
		SHADER_PARAMETER(int32, KernelSize)
		SHADER_PARAMETER(int32, Pad)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, InputBuffer)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, Weights)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, Bias)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<float>, OutputBuffer)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

// -----------------------------------------------------------------------
// Expand/Project 1x1 Conv + BN + optional ReLU6 (EfficientNet MBConv)
// -----------------------------------------------------------------------

class FConv1x1BnCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FConv1x1BnCS);
	SHADER_USE_PARAMETER_STRUCT(FConv1x1BnCS, FGlobalShader);

	class FHasReLU6 : SHADER_PERMUTATION_BOOL("HAS_RELU6");
	using FPermutationDomain = TShaderPermutationDomain<FHasReLU6>;

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER(int32, Width)
		SHADER_PARAMETER(int32, Height)
		SHADER_PARAMETER(int32, InChannels)
		SHADER_PARAMETER(int32, OutChannels)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, InputBuffer)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, Weights)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, Bias)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<float>, OutputBuffer)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

// -----------------------------------------------------------------------
// Global Average Pooling (EfficientNet)
// -----------------------------------------------------------------------

class FGlobalAvgPoolCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FGlobalAvgPoolCS);
	SHADER_USE_PARAMETER_STRUCT(FGlobalAvgPoolCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER(int32, Channels)
		SHADER_PARAMETER(int32, SpatialSize)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, InputBuffer)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<float>, OutputBuffer)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

// -----------------------------------------------------------------------
// FC MatMul + optional Sigmoid (EfficientNet output heads)
// -----------------------------------------------------------------------

class FFCMatMulCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FFCMatMulCS);
	SHADER_USE_PARAMETER_STRUCT(FFCMatMulCS, FGlobalShader);

	class FHasSigmoid : SHADER_PERMUTATION_BOOL("HAS_SIGMOID");
	using FPermutationDomain = TShaderPermutationDomain<FHasSigmoid>;

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER(int32, InFeatures)
		SHADER_PARAMETER(int32, OutFeatures)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, InputBuffer)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, Weights)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, Bias)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<float>, OutputBuffer)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

// -----------------------------------------------------------------------
// Conv3x3 Stride-2 + BN + ReLU6 (EfficientNet initial conv)
// -----------------------------------------------------------------------

class FConv3x3Stride2BnRelu6CS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FConv3x3Stride2BnRelu6CS);
	SHADER_USE_PARAMETER_STRUCT(FConv3x3Stride2BnRelu6CS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER(int32, InWidth)
		SHADER_PARAMETER(int32, InHeight)
		SHADER_PARAMETER(int32, InChannels)
		SHADER_PARAMETER(int32, OutWidth)
		SHADER_PARAMETER(int32, OutHeight)
		SHADER_PARAMETER(int32, OutChannels)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, InputBuffer)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, Weights)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, Bias)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<float>, OutputBuffer)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

// -----------------------------------------------------------------------
// Elementwise Add
// -----------------------------------------------------------------------

class FElementwiseAddCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FElementwiseAddCS);
	SHADER_USE_PARAMETER_STRUCT(FElementwiseAddCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER(int32, Count)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, BufferA)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, BufferB)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<float>, OutputBuffer)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};
