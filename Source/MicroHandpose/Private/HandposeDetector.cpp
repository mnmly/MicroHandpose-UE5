#include "HandposeDetector.h"
#include "MicroHandposeModule.h"
#include "WeightLoader.h"
#include "PalmDetectionPostProcess.h"
#include "LandmarkPostProcess.h"
#include "ShaderPasses/HandposeShaders.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphUtils.h"
#include "RHIResourceUtils.h"
#include "GlobalShader.h"
#include "ShaderParameterStruct.h"
#include "Interfaces/IPluginManager.h"

DECLARE_GPU_STAT_NAMED(HandposeInference, TEXT("Handpose Inference"));

static int32 DivUp(int32 A, int32 B) { return (A + B - 1) / B; }

// ============================================================================
// Constructor / Destructor
// ============================================================================

FHandposeDetector::FHandposeDetector()
{
	InitPalmBlockSpecs();
	InitLandmarkBlockSpecs();
	PalmPostProcess = MakeUnique<FPalmDetectionPostProcess>();

	PalmCls16Readback = new FRHIGPUBufferReadback(TEXT("PalmCls16Readback"));
	PalmReg16Readback = new FRHIGPUBufferReadback(TEXT("PalmReg16Readback"));
	PalmCls8Readback  = new FRHIGPUBufferReadback(TEXT("PalmCls8Readback"));
	PalmReg8Readback  = new FRHIGPUBufferReadback(TEXT("PalmReg8Readback"));
}

FHandposeDetector::~FHandposeDetector()
{
	delete PalmCls16Readback;
	delete PalmReg16Readback;
	delete PalmCls8Readback;
	delete PalmReg8Readback;

	for (FRHIGPUBufferReadback* Rb : LandmarkReadbacks)
	{
		delete Rb;
	}
	LandmarkReadbacks.Empty();
}

// ============================================================================
// Block specifications
// ============================================================================

void FHandposeDetector::InitPalmBlockSpecs()
{
	struct { int32 InCh, OutCh, Stride, InH; } Specs[] = {
		{32, 32, 1, 96}, {32, 32, 1, 96}, {32, 32, 1, 96}, {32, 32, 1, 96},
		{32, 64, 2, 96},
		{64, 64, 1, 48}, {64, 64, 1, 48}, {64, 64, 1, 48}, {64, 64, 1, 48},
		{64, 128, 2, 48},
		{128, 128, 1, 24}, {128, 128, 1, 24}, {128, 128, 1, 24}, {128, 128, 1, 24},
		{128, 256, 2, 24},
		{256, 256, 1, 12}, {256, 256, 1, 12}, {256, 256, 1, 12}, {256, 256, 1, 12},
		{256, 256, 2, 12},
		{256, 256, 1, 6}, {256, 256, 1, 6}, {256, 256, 1, 6}, {256, 256, 1, 6},
	};

	PalmBlocks.Reserve(24);
	for (int32 i = 0; i < 24; i++)
	{
		FPalmBackboneBlock Block;
		Block.DwIdx = i;
		Block.PwIdx = i + 1;
		Block.InCh = Specs[i].InCh;
		Block.OutCh = Specs[i].OutCh;
		Block.Stride = Specs[i].Stride;
		Block.InH = Specs[i].InH;
		PalmBlocks.Add(Block);
	}
}

void FHandposeDetector::InitLandmarkBlockSpecs()
{
	struct { int32 InCh, ExpandCh, DwKernel, Stride, OutCh; bool Res, Proj; int32 SpatOut; } Specs[] = {
		{24,   24,   3, 1,  16,   false, true,  112},
		{16,   64,   3, 2,  24,   false, true,   56},
		{24,  144,   3, 1,  24,   true,  true,   56},
		{24,  144,   5, 2,  40,   false, true,   28},
		{40,  240,   5, 1,  40,   true,  true,   28},
		{40,  240,   3, 2,  80,   false, true,   14},
		{80,  480,   3, 1,  80,   true,  true,   14},
		{80,  480,   3, 1,  80,   true,  true,   14},
		{80,  480,   5, 1, 112,   false, true,   14},
		{112, 672,   5, 1, 112,   true,  true,   14},
		{112, 672,   5, 1, 112,   true,  true,   14},
		{112, 672,   5, 2, 192,   false, true,    7},
		{192, 1152,  5, 1, 192,   true,  true,    7},
		{192, 1152,  5, 1, 192,   true,  true,    7},
		{192, 1152,  5, 1, 192,   true,  true,    7},
		{192, 1152,  3, 1, 1152,  false, false,   7},
	};

	LandmarkBlocks.Reserve(16);
	for (int32 i = 0; i < 16; i++)
	{
		FMBConvBlock Block;
		Block.InCh = Specs[i].InCh;
		Block.ExpandCh = Specs[i].ExpandCh;
		Block.DwKernel = Specs[i].DwKernel;
		Block.Stride = Specs[i].Stride;
		Block.OutCh = Specs[i].OutCh;
		Block.bHasResidual = Specs[i].Res;
		Block.bHasProject = Specs[i].Proj;
		Block.SpatialOut = Specs[i].SpatOut;
		LandmarkBlocks.Add(Block);
	}
}

// ============================================================================
// Weight loading
// ============================================================================

FGPUWeightBuffer FHandposeDetector::UploadWeightToGPU(const TArray<float>& Data, const FString& DebugName)
{
	FGPUWeightBuffer Result;
	Result.NumElements = Data.Num();

	if (Data.Num() == 0) return Result;

	uint32 ByteSize = Data.Num() * sizeof(float);

	const FRHIBufferCreateDesc CreateDesc =
		FRHIBufferCreateDesc::CreateStructured(*DebugName, ByteSize, sizeof(float))
		.AddUsage(EBufferUsageFlags::ShaderResource)
		.SetInitialState(ERHIAccess::SRVMask);

	FRHICommandListBase& RHICmdList = FRHICommandListImmediate::Get();
	FBufferRHIRef Buffer = UE::RHIResourceUtils::CreateBufferWithArray(RHICmdList, CreateDesc, Data);

	Result.PooledBuffer = new FRDGPooledBuffer(RHICmdList, Buffer, FRDGBufferDesc::CreateStructuredDesc(sizeof(float), Data.Num()), Data.Num(), *DebugName);

	return Result;
}

bool FHandposeDetector::LoadPalmWeights(const FString& PluginBaseDir)
{
	FString JsonPath = FPaths::Combine(PluginBaseDir, TEXT("Resources/Models/palm_detection_weights.json"));
	FString BinPath = FPaths::Combine(PluginBaseDir, TEXT("Resources/Models/palm_detection_weights.bin"));

	FWeightCollection Weights = FWeightLoader::LoadWeights(JsonPath, BinPath);
	if (Weights.Tensors.Num() == 0)
	{
		UE_LOG(LogMicroHandpose, Error, TEXT("[MicroHandpose] Failed to load palm detection weights"));
		return false;
	}

	for (auto& Pair : Weights.Tensors)
	{
		const FWeightTensor& Tensor = Pair.Value;

		bool bNeedTranspose = Pair.Key.Contains(TEXT("depthwise")) &&
			Tensor.Shape.Num() == 4 && Tensor.Shape[0] == 1;

		if (bNeedTranspose)
		{
			TArray<float> Transposed = FWeightLoader::TransposeDepthwiseWeights(Tensor);
			PalmWeightBuffers.Add(Pair.Key, UploadWeightToGPU(Transposed, Pair.Key));
		}
		else
		{
			PalmWeightBuffers.Add(Pair.Key, UploadWeightToGPU(Tensor.Data, Pair.Key));
		}
	}

	UE_LOG(LogMicroHandpose, Log, TEXT("[MicroHandpose] Palm detection: %d weight buffers uploaded to GPU"), PalmWeightBuffers.Num());
	return true;
}

bool FHandposeDetector::LoadLandmarkWeights(const FString& PluginBaseDir)
{
	FString JsonPath = FPaths::Combine(PluginBaseDir, TEXT("Resources/Models/weights_f16_full.json"));
	FString BinPath = FPaths::Combine(PluginBaseDir, TEXT("Resources/Models/weights_f16_full.bin"));

	FWeightCollection Weights = FWeightLoader::LoadWeights(JsonPath, BinPath);
	if (Weights.Tensors.Num() == 0)
	{
		UE_LOG(LogMicroHandpose, Error, TEXT("[MicroHandpose] Failed to load landmark weights"));
		return false;
	}

	for (auto& Pair : Weights.Tensors)
	{
		const FWeightTensor& Tensor = Pair.Value;

		bool bNeedTranspose = Pair.Key.Contains(TEXT("FusedBatchNormV3")) &&
			Tensor.Shape.Num() == 4 && Tensor.Shape[0] == 1;

		if (bNeedTranspose)
		{
			TArray<float> Transposed = FWeightLoader::TransposeDepthwiseWeights(Tensor);
			LandmarkWeightBuffers.Add(Pair.Key, UploadWeightToGPU(Transposed, Pair.Key));
		}
		else
		{
			LandmarkWeightBuffers.Add(Pair.Key, UploadWeightToGPU(Tensor.Data, Pair.Key));
		}
	}

	UE_LOG(LogMicroHandpose, Log, TEXT("[MicroHandpose] Landmark model: %d weight buffers uploaded to GPU"), LandmarkWeightBuffers.Num());
	return true;
}

bool FHandposeDetector::Initialize(const FString& PluginBaseDir)
{
	if (bInitialized) return true;

	if (!LoadPalmWeights(PluginBaseDir))
	{
		return false;
	}

	if (!LoadLandmarkWeights(PluginBaseDir))
	{
		return false;
	}

	// Create zero-bias buffer for palm DW convolutions (bias folded into weights)
	TArray<float> Zeros;
	Zeros.SetNumZeroed(256);
	ZeroBias256 = UploadWeightToGPU(Zeros, TEXT("ZeroBias256"));

	bInitialized = true;
	UE_LOG(LogMicroHandpose, Log, TEXT("[MicroHandpose] Detector initialized successfully"));
	return true;
}

// ============================================================================
// Weight SRV helpers
// ============================================================================

FRDGBufferSRVRef FHandposeDetector::GetWeightSRV(FRDGBuilder& GraphBuilder, const TMap<FString, FGPUWeightBuffer>& WeightMap, const FString& ExactKey) const
{
	const FGPUWeightBuffer* Found = WeightMap.Find(ExactKey);
	if (!Found || !Found->PooledBuffer.IsValid())
	{
		UE_LOG(LogMicroHandpose, Error, TEXT("[MicroHandpose] Weight key not found: %s"), *ExactKey);
		return nullptr;
	}
	FRDGBufferRef Buf = GraphBuilder.RegisterExternalBuffer(Found->PooledBuffer);
	return GraphBuilder.CreateSRV(Buf);
}

FRDGBufferSRVRef FHandposeDetector::FindWeightSRV(FRDGBuilder& GraphBuilder, const TMap<FString, FGPUWeightBuffer>& WeightMap, const TArray<FString>& Substrings) const
{
	FString BestSimpleKey;
	int32 BestSimpleLen = INT32_MAX;
	FString BestAnyKey;
	int32 BestAnyLen = INT32_MAX;

	for (const auto& Pair : WeightMap)
	{
		bool bAllMatch = true;
		for (const FString& Sub : Substrings)
		{
			if (!Pair.Key.Contains(Sub))
			{
				bAllMatch = false;
				break;
			}
		}
		if (!bAllMatch) continue;

		bool bHasSemicolon = Pair.Key.Contains(TEXT(";"));

		if (!bHasSemicolon && Pair.Key.Len() < BestSimpleLen)
		{
			BestSimpleKey = Pair.Key;
			BestSimpleLen = Pair.Key.Len();
		}
		if (Pair.Key.Len() < BestAnyLen)
		{
			BestAnyKey = Pair.Key;
			BestAnyLen = Pair.Key.Len();
		}
	}

	FString ChosenKey = !BestSimpleKey.IsEmpty() ? BestSimpleKey : BestAnyKey;
	if (ChosenKey.IsEmpty())
	{
		FString SubStr;
		for (const FString& S : Substrings) { SubStr += S + TEXT(", "); }
		UE_LOG(LogMicroHandpose, Error, TEXT("[MicroHandpose] FindWeightSRV: no key matches [%s]"), *SubStr);
		return nullptr;
	}

	return GetWeightSRV(GraphBuilder, WeightMap, ChosenKey);
}

FRDGBufferSRVRef FHandposeDetector::GetZeroBiasSRV(FRDGBuilder& GraphBuilder) const
{
	FRDGBufferRef Buf = GraphBuilder.RegisterExternalBuffer(ZeroBias256.PooledBuffer);
	return GraphBuilder.CreateSRV(Buf);
}

// ============================================================================
// SSD output reorganization (CHW → HWC anchor order)
// ============================================================================

void FHandposeDetector::ReorganizeSSDOutput(
	const TArray<float>& Cls16, const TArray<float>& Reg16,
	const TArray<float>& Cls8, const TArray<float>& Reg8,
	TArray<float>& OutScores, TArray<float>& OutRegressors) const
{
	OutScores.SetNum(2016);
	OutRegressors.SetNum(2016 * 18);

	int32 AnchorIdx = 0;

	for (int32 Y = 0; Y < 12; Y++)
	{
		for (int32 X = 0; X < 12; X++)
		{
			for (int32 A = 0; A < 6; A++)
			{
				OutScores[AnchorIdx] = Cls16[A * 144 + Y * 12 + X];
				for (int32 V = 0; V < 18; V++)
				{
					int32 Ch = A * 18 + V;
					OutRegressors[AnchorIdx * 18 + V] = Reg16[Ch * 144 + Y * 12 + X];
				}
				AnchorIdx++;
			}
		}
	}

	for (int32 Y = 0; Y < 24; Y++)
	{
		for (int32 X = 0; X < 24; X++)
		{
			for (int32 A = 0; A < 2; A++)
			{
				OutScores[AnchorIdx] = Cls8[A * 576 + Y * 24 + X];
				for (int32 V = 0; V < 18; V++)
				{
					int32 Ch = A * 18 + V;
					OutRegressors[AnchorIdx * 18 + V] = Reg8[Ch * 576 + Y * 24 + X];
				}
				AnchorIdx++;
			}
		}
	}

	check(AnchorIdx == 2016);
}

// ============================================================================
// Palm Detection Dispatch
// ============================================================================

void FHandposeDetector::DispatchPalmDetection(FRDGBuilder& GraphBuilder, FRDGTexture* InputTexture)
{
	RDG_GPU_STAT_SCOPE(GraphBuilder, HandposeInference);

	auto* ShaderMap = GetGlobalShaderMap(GMaxRHIFeatureLevel);
	FRDGBufferSRVRef ZeroBias = GetZeroBiasSRV(GraphBuilder);

	// ---- Helpers for palm weight key construction ----
	auto PalmDwKey = [](int32 Idx) -> FString {
		return Idx == 0 ? TEXT("depthwise_conv2d/") : FString::Printf(TEXT("depthwise_conv2d_%d/"), Idx);
	};
	auto PalmConvKey = [](int32 Idx) -> FString {
		return Idx == 0 ? TEXT("conv2d/") : FString::Printf(TEXT("conv2d_%d/"), Idx);
	};
	auto PalmBnKey = [](int32 Idx) -> FString {
		return Idx == 0 ? TEXT("batch_normalization/FusedBatchNormV3") : FString::Printf(TEXT("batch_normalization_%d/FusedBatchNormV3"), Idx);
	};
	auto PalmPreluKey = [](int32 Idx) -> FString {
		return Idx == 0 ? TEXT("p_re_lu/") : FString::Printf(TEXT("p_re_lu_%d/"), Idx);
	};

	UE_LOG(LogMicroHandpose, Verbose, TEXT("Palm detection: dispatching (input %dx%d)"),
		InputTexture->Desc.Extent.X, InputTexture->Desc.Extent.Y);

	// ---- Store source dimensions for post-processing ----
	FRDGTextureDesc TexDesc = InputTexture->Desc;
	SrcWidth = TexDesc.Extent.X;
	SrcHeight = TexDesc.Extent.Y;

	// ---- 1. Letterbox resize → 192×192 CHW ----
	constexpr int32 PalmSize = 192;
	float Scale = FMath::Max(SrcWidth, SrcHeight) / (float)PalmSize;
	float ScaledW = SrcWidth / Scale;
	float ScaledH = SrcHeight / Scale;
	LetterboxPadX = (PalmSize - ScaledW) / 2.0f;
	LetterboxPadY = (PalmSize - ScaledH) / 2.0f;

	FRDGBufferRef PalmInput = GraphBuilder.CreateBuffer(
		FRDGBufferDesc::CreateStructuredDesc(sizeof(float), 3 * PalmSize * PalmSize), TEXT("PalmInput"));

	{
		auto* P = GraphBuilder.AllocParameters<FLetterboxResizeCS::FParameters>();
		P->SrcWidth = SrcWidth;
		P->SrcHeight = SrcHeight;
		P->DstSize = PalmSize;
		P->ScaleX = Scale;
		P->ScaleY = Scale;
		P->OffsetX = LetterboxPadX;
		P->OffsetY = LetterboxPadY;
		P->InputTexture = GraphBuilder.CreateSRV(FRDGTextureSRVDesc(InputTexture));
		P->InputSampler = TStaticSamplerState<SF_Bilinear, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
		P->OutputBuffer = GraphBuilder.CreateUAV(PalmInput);

		TShaderMapRef<FLetterboxResizeCS> CS(ShaderMap);
		FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("PalmLetterbox"), CS, P,
			FIntVector(DivUp(PalmSize, 16), DivUp(PalmSize, 16), 1));
	}

	// ---- 2. Initial conv: 192×192×3 → 96×96×32 ----
	constexpr int32 InitOutH = 96;
	constexpr int32 InitOutCh = 32;

	FRDGBufferRef Conv0Out = GraphBuilder.CreateBuffer(
		FRDGBufferDesc::CreateStructuredDesc(sizeof(float), InitOutCh * InitOutH * InitOutH), TEXT("PalmConv0"));

	{
		auto* P = GraphBuilder.AllocParameters<FConv5x5Stride2CS::FParameters>();
		P->InWidth = PalmSize;
		P->InHeight = PalmSize;
		P->InChannels = 3;
		P->OutWidth = InitOutH;
		P->OutHeight = InitOutH;
		P->OutChannels = InitOutCh;
		P->InputBuffer = GraphBuilder.CreateSRV(PalmInput);
		P->Weights = FindWeightSRV(GraphBuilder, PalmWeightBuffers, {TEXT("conv2d/Conv2D")});
		P->Bias = FindWeightSRV(GraphBuilder, PalmWeightBuffers, {PalmBnKey(0)});
		P->Alpha = FindWeightSRV(GraphBuilder, PalmWeightBuffers, {PalmPreluKey(0)});
		P->OutputBuffer = GraphBuilder.CreateUAV(Conv0Out);

		TShaderMapRef<FConv5x5Stride2CS> CS(ShaderMap);
		FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("PalmInitConv"), CS, P,
			FIntVector(DivUp(InitOutH, 8), DivUp(InitOutH, 8), InitOutCh));
	}

	// ---- 3. 24 backbone blocks ----
	FRDGBufferRef CurrBuf = Conv0Out;
	int32 CurrCh = InitOutCh;
	int32 CurrH = InitOutH;
	FRDGBufferRef Skip24Buf = nullptr;
	FRDGBufferRef Skip12Buf = nullptr;

	for (int32 i = 0; i < 24; i++)
	{
		const FPalmBackboneBlock& B = PalmBlocks[i];
		int32 DwOutH = CurrH / B.Stride;
		FRDGBufferRef BlockInput = CurrBuf;

		// -- Depthwise 5×5 --
		FRDGBufferRef DwOut = GraphBuilder.CreateBuffer(
			FRDGBufferDesc::CreateStructuredDesc(sizeof(float), B.InCh * DwOutH * DwOutH),
			*FString::Printf(TEXT("PalmDW%d"), B.DwIdx));

		{
			auto* P = GraphBuilder.AllocParameters<FDepthwiseConv5x5CS::FParameters>();
			P->InWidth = CurrH;
			P->InHeight = CurrH;
			P->Channels = B.InCh;
			P->OutWidth = DwOutH;
			P->OutHeight = DwOutH;
			P->Stride = B.Stride;
			P->InputBuffer = GraphBuilder.CreateSRV(CurrBuf);
			P->Weights = FindWeightSRV(GraphBuilder, PalmWeightBuffers, {PalmDwKey(B.DwIdx)});
			P->Bias = ZeroBias;
			P->OutputBuffer = GraphBuilder.CreateUAV(DwOut);

			TShaderMapRef<FDepthwiseConv5x5CS> CS(ShaderMap);
			FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("PalmDW%d", B.DwIdx), CS, P,
				FIntVector(DivUp(DwOutH, 8), DivUp(DwOutH, 8), B.InCh));
		}

		// -- Pointwise 1×1 + skip + PReLU --
		bool bHasSkip = true;
		int32 ChannelPad = FMath::Min(B.InCh, B.OutCh);

		FRDGBufferRef PwOut = GraphBuilder.CreateBuffer(
			FRDGBufferDesc::CreateStructuredDesc(sizeof(float), B.OutCh * DwOutH * DwOutH),
			*FString::Printf(TEXT("PalmPW%d"), B.PwIdx));

		{
			auto* P = GraphBuilder.AllocParameters<FPointwiseConvCS::FParameters>();
			P->InWidth = DwOutH;
			P->InHeight = DwOutH;
			P->InChannels = B.InCh;
			P->OutWidth = DwOutH;
			P->OutHeight = DwOutH;
			P->OutChannels = B.OutCh;
			P->Stride = B.Stride;
			P->ChannelPad = ChannelPad;
			P->InputBuffer = GraphBuilder.CreateSRV(DwOut);
			P->SkipBuffer = GraphBuilder.CreateSRV(BlockInput);
			P->Weights = FindWeightSRV(GraphBuilder, PalmWeightBuffers, {PalmConvKey(B.PwIdx)});
			P->Bias = FindWeightSRV(GraphBuilder, PalmWeightBuffers, {PalmBnKey(B.PwIdx)});
			P->Alpha = FindWeightSRV(GraphBuilder, PalmWeightBuffers, {PalmPreluKey(B.PwIdx)});
			P->OutputBuffer = GraphBuilder.CreateUAV(PwOut);

			FPointwiseConvCS::FPermutationDomain PD;
			PD.Set<FPointwiseConvCS::FActivationMode>(2); // PReLU
			PD.Set<FPointwiseConvCS::FHasSkip>(bHasSkip);
			TShaderMapRef<FPointwiseConvCS> CS(ShaderMap, PD);
			FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("PalmPW%d", B.PwIdx), CS, P,
				FIntVector(DivUp(DwOutH, 8), DivUp(DwOutH, 8), B.OutCh));
		}

		// Save skip features
		if (i == 13) Skip24Buf = PwOut; // 128ch 24×24
		if (i == 18) Skip12Buf = PwOut; // 256ch 12×12

		CurrBuf = PwOut;
		CurrCh = B.OutCh;
		CurrH = DwOutH;
	}

	// ---- Helper lambda for FPN DW+PW blocks ----
	auto DispatchFpnDwPw = [&](FRDGBufferRef InBuf, int32 Ch, int32 Spatial, int32 DwIdx, int32 PwIdx) -> FRDGBufferRef
	{
		// DW
		FRDGBufferRef DwOut = GraphBuilder.CreateBuffer(
			FRDGBufferDesc::CreateStructuredDesc(sizeof(float), Ch * Spatial * Spatial),
			*FString::Printf(TEXT("FpnDW%d"), DwIdx));
		{
			auto* P = GraphBuilder.AllocParameters<FDepthwiseConv5x5CS::FParameters>();
			P->InWidth = Spatial; P->InHeight = Spatial; P->Channels = Ch;
			P->OutWidth = Spatial; P->OutHeight = Spatial; P->Stride = 1;
			P->InputBuffer = GraphBuilder.CreateSRV(InBuf);
			P->Weights = FindWeightSRV(GraphBuilder, PalmWeightBuffers, {PalmDwKey(DwIdx)});
			P->Bias = ZeroBias;
			P->OutputBuffer = GraphBuilder.CreateUAV(DwOut);
			TShaderMapRef<FDepthwiseConv5x5CS> CS(ShaderMap);
			FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("FpnDW%d", DwIdx), CS, P,
				FIntVector(DivUp(Spatial, 8), DivUp(Spatial, 8), Ch));
		}
		// PW (no skip for FPN blocks)
		FRDGBufferRef PwOut = GraphBuilder.CreateBuffer(
			FRDGBufferDesc::CreateStructuredDesc(sizeof(float), Ch * Spatial * Spatial),
			*FString::Printf(TEXT("FpnPW%d"), PwIdx));
		{
			auto* P = GraphBuilder.AllocParameters<FPointwiseConvCS::FParameters>();
			P->InWidth = Spatial; P->InHeight = Spatial; P->InChannels = Ch;
			P->OutWidth = Spatial; P->OutHeight = Spatial; P->OutChannels = Ch;
			P->Stride = 1; P->ChannelPad = Ch;
			P->InputBuffer = GraphBuilder.CreateSRV(DwOut);
			P->SkipBuffer = GraphBuilder.CreateSRV(InBuf);
			P->Weights = FindWeightSRV(GraphBuilder, PalmWeightBuffers, {PalmConvKey(PwIdx)});
			P->Bias = FindWeightSRV(GraphBuilder, PalmWeightBuffers, {PalmBnKey(PwIdx)});
			P->Alpha = FindWeightSRV(GraphBuilder, PalmWeightBuffers, {PalmPreluKey(PwIdx)});
			P->OutputBuffer = GraphBuilder.CreateUAV(PwOut);

			FPointwiseConvCS::FPermutationDomain PD;
			PD.Set<FPointwiseConvCS::FActivationMode>(2);
			PD.Set<FPointwiseConvCS::FHasSkip>(true);
			TShaderMapRef<FPointwiseConvCS> CS(ShaderMap, PD);
			FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("FpnPW%d", PwIdx), CS, P,
				FIntVector(DivUp(Spatial, 8), DivUp(Spatial, 8), Ch));
		}
		return PwOut;
	};

	// ---- Helper lambda for standalone PW project ----
	auto DispatchFpnProject = [&](FRDGBufferRef InBuf, int32 InCh, int32 OutCh, int32 Spatial, int32 PwIdx) -> FRDGBufferRef
	{
		FRDGBufferRef Out = GraphBuilder.CreateBuffer(
			FRDGBufferDesc::CreateStructuredDesc(sizeof(float), OutCh * Spatial * Spatial),
			*FString::Printf(TEXT("FpnProj%d"), PwIdx));

		auto* P = GraphBuilder.AllocParameters<FPointwiseConvCS::FParameters>();
		P->InWidth = Spatial; P->InHeight = Spatial; P->InChannels = InCh;
		P->OutWidth = Spatial; P->OutHeight = Spatial; P->OutChannels = OutCh;
		P->Stride = 1; P->ChannelPad = 0;
		P->InputBuffer = GraphBuilder.CreateSRV(InBuf);
		P->SkipBuffer = ZeroBias; // unused
		P->Weights = FindWeightSRV(GraphBuilder, PalmWeightBuffers, {PalmConvKey(PwIdx)});
		P->Bias = FindWeightSRV(GraphBuilder, PalmWeightBuffers, {PalmBnKey(PwIdx)});
		P->Alpha = FindWeightSRV(GraphBuilder, PalmWeightBuffers, {PalmPreluKey(PwIdx)});
		P->OutputBuffer = GraphBuilder.CreateUAV(Out);

		FPointwiseConvCS::FPermutationDomain PD;
		PD.Set<FPointwiseConvCS::FActivationMode>(2); // PReLU
		PD.Set<FPointwiseConvCS::FHasSkip>(false);
		TShaderMapRef<FPointwiseConvCS> CS(ShaderMap, PD);
		FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("FpnProj%d", PwIdx), CS, P,
			FIntVector(DivUp(Spatial, 8), DivUp(Spatial, 8), OutCh));
		return Out;
	};

	// ---- Helper lambda for Upsample 2× ----
	auto DispatchUpsample = [&](FRDGBufferRef InBuf, int32 Ch, int32 InH, const TCHAR* Name) -> FRDGBufferRef
	{
		int32 OutH = InH * 2;
		FRDGBufferRef Out = GraphBuilder.CreateBuffer(
			FRDGBufferDesc::CreateStructuredDesc(sizeof(float), Ch * OutH * OutH), Name);

		auto* P = GraphBuilder.AllocParameters<FUpsample2xCS::FParameters>();
		P->InWidth = InH; P->InHeight = InH; P->Channels = Ch;
		P->OutWidth = OutH; P->OutHeight = OutH;
		P->InputBuffer = GraphBuilder.CreateSRV(InBuf);
		P->AddBuffer = ZeroBias; // unused
		P->OutputBuffer = GraphBuilder.CreateUAV(Out);

		FUpsample2xCS::FPermutationDomain PD;
		PD.Set<FUpsample2xCS::FHasAdd>(false);
		TShaderMapRef<FUpsample2xCS> CS(ShaderMap, PD);
		FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("Upsample_%s", Name), CS, P,
			FIntVector(DivUp(OutH, 8), DivUp(OutH, 8), Ch));
		return Out;
	};

	// ---- Helper lambda for Elementwise Add ----
	auto DispatchAdd = [&](FRDGBufferRef A, FRDGBufferRef B, int32 Count, const TCHAR* Name) -> FRDGBufferRef
	{
		FRDGBufferRef Out = GraphBuilder.CreateBuffer(
			FRDGBufferDesc::CreateStructuredDesc(sizeof(float), Count), Name);

		auto* P = GraphBuilder.AllocParameters<FElementwiseAddCS::FParameters>();
		P->Count = Count;
		P->BufferA = GraphBuilder.CreateSRV(A);
		P->BufferB = GraphBuilder.CreateSRV(B);
		P->OutputBuffer = GraphBuilder.CreateUAV(Out);

		TShaderMapRef<FElementwiseAddCS> CS(ShaderMap);
		FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("Add_%s", Name), CS, P,
			FIntVector(DivUp(Count, 256), 1, 1));
		return Out;
	};

	// ---- 4. FPN Level 1: 6×6→12×12 ----
	FRDGBufferRef Ups12 = DispatchUpsample(CurrBuf, 256, 6, TEXT("Ups12"));
	FRDGBufferRef Proj25 = DispatchFpnProject(Ups12, 256, 256, 12, 25);
	FRDGBufferRef Add12 = DispatchAdd(Proj25, Skip12Buf, 256 * 12 * 12, TEXT("FpnAdd12"));
	FRDGBufferRef Fpn12 = DispatchFpnDwPw(Add12, 256, 12, 24, 26);
	Fpn12 = DispatchFpnDwPw(Fpn12, 256, 12, 25, 27);

	// ---- 5. FPN Level 2: 12×12→24×24 ----
	FRDGBufferRef Ups24 = DispatchUpsample(Fpn12, 256, 12, TEXT("Ups24"));
	FRDGBufferRef Proj28 = DispatchFpnProject(Ups24, 256, 128, 24, 28);
	FRDGBufferRef Add24 = DispatchAdd(Proj28, Skip24Buf, 128 * 24 * 24, TEXT("FpnAdd24"));
	FRDGBufferRef Fpn24 = DispatchFpnDwPw(Add24, 128, 24, 26, 29);
	Fpn24 = DispatchFpnDwPw(Fpn24, 128, 24, 27, 30);

	// ---- 6. SSD heads ----
	auto DispatchSsdHead = [&](FRDGBufferRef InBuf, int32 InCh, int32 OutCh, int32 Spatial,
		const FString& WeightSub, const FString& BiasSub, const TCHAR* Name) -> FRDGBufferRef
	{
		FRDGBufferRef Out = GraphBuilder.CreateBuffer(
			FRDGBufferDesc::CreateStructuredDesc(sizeof(float), OutCh * Spatial * Spatial), Name);

		auto* P = GraphBuilder.AllocParameters<FPointwiseConvCS::FParameters>();
		P->InWidth = Spatial; P->InHeight = Spatial; P->InChannels = InCh;
		P->OutWidth = Spatial; P->OutHeight = Spatial; P->OutChannels = OutCh;
		P->Stride = 1; P->ChannelPad = 0;
		P->InputBuffer = GraphBuilder.CreateSRV(InBuf);
		P->SkipBuffer = ZeroBias;
		P->Weights = FindWeightSRV(GraphBuilder, PalmWeightBuffers, {WeightSub, TEXT("Conv2D")});
		P->Bias = FindWeightSRV(GraphBuilder, PalmWeightBuffers, {BiasSub, TEXT("BiasAdd")});
		P->Alpha = ZeroBias; // unused
		P->OutputBuffer = GraphBuilder.CreateUAV(Out);

		FPointwiseConvCS::FPermutationDomain PD;
		PD.Set<FPointwiseConvCS::FActivationMode>(0); // None
		PD.Set<FPointwiseConvCS::FHasSkip>(false);
		TShaderMapRef<FPointwiseConvCS> CS(ShaderMap, PD);
		FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("SSD_%s", Name), CS, P,
			FIntVector(DivUp(Spatial, 8), DivUp(Spatial, 8), OutCh));
		return Out;
	};

	FRDGBufferRef Cls16Buf = DispatchSsdHead(Fpn12, 256,   6, 12, TEXT("classifier_palm_16"), TEXT("classifier_palm_16"), TEXT("Cls16"));
	FRDGBufferRef Reg16Buf = DispatchSsdHead(Fpn12, 256, 108, 12, TEXT("regressor_palm_16"),  TEXT("regressor_palm_16"),  TEXT("Reg16"));
	FRDGBufferRef Cls8Buf  = DispatchSsdHead(Fpn24, 128,   2, 24, TEXT("classifier_palm_8"),  TEXT("classifier_palm_8"),  TEXT("Cls8"));
	FRDGBufferRef Reg8Buf  = DispatchSsdHead(Fpn24, 128,  36, 24, TEXT("regressor_palm_8"),   TEXT("regressor_palm_8"),   TEXT("Reg8"));

	// ---- 7. Readback SSD outputs ----
	AddEnqueueCopyPass(GraphBuilder, PalmCls16Readback, Cls16Buf,   6 * 144 * sizeof(float));
	AddEnqueueCopyPass(GraphBuilder, PalmReg16Readback, Reg16Buf, 108 * 144 * sizeof(float));
	AddEnqueueCopyPass(GraphBuilder, PalmCls8Readback,  Cls8Buf,    2 * 576 * sizeof(float));
	AddEnqueueCopyPass(GraphBuilder, PalmReg8Readback,  Reg8Buf,   36 * 576 * sizeof(float));
}

// ============================================================================
// Landmark Inference Dispatch
// ============================================================================

void FHandposeDetector::DispatchLandmarkInference(FRDGBuilder& GraphBuilder, FRDGTexture* InputTexture, const FLandmarkPostProcess::FPixelROI& PixelROI, int32 HandIndex)
{
	UE_LOG(LogMicroHandpose, Verbose, TEXT("Landmark inference: dispatching hand %d (PixelROI center=%.1f,%.1f size=%.1f rot=%.2f)"),
		HandIndex, PixelROI.CenterXPx, PixelROI.CenterYPx, PixelROI.SizePx, PixelROI.Rotation);

	auto* ShaderMap = GetGlobalShaderMap(GMaxRHIFeatureLevel);

	constexpr int32 CropSize = 224;

	// ---- 1. AffineCrop ----
	FMatrix44f AffineMat = FLandmarkPostProcess::ComputeAffineMatrix(PixelROI, SrcWidth, SrcHeight, CropSize);

	FRDGBufferRef CropBuf = GraphBuilder.CreateBuffer(
		FRDGBufferDesc::CreateStructuredDesc(sizeof(float), 3 * CropSize * CropSize), TEXT("LmCrop"));

	{
		auto* P = GraphBuilder.AllocParameters<FAffineCropCS::FParameters>();
		P->DstSize = CropSize;
		P->AffineMatrix = AffineMat;
		P->InputTexture = GraphBuilder.CreateSRV(FRDGTextureSRVDesc(InputTexture));
		P->InputSampler = TStaticSamplerState<SF_Bilinear, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
		P->OutputBuffer = GraphBuilder.CreateUAV(CropBuf);

		TShaderMapRef<FAffineCropCS> CS(ShaderMap);
		FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("LmAffineCrop"), CS, P,
			FIntVector(DivUp(CropSize, 16), DivUp(CropSize, 16), 1));
	}

	// ---- 2. Initial conv: 224×224×3 → 112×112×24 ----
	FRDGBufferRef CurrBuf = GraphBuilder.CreateBuffer(
		FRDGBufferDesc::CreateStructuredDesc(sizeof(float), 24 * 112 * 112), TEXT("LmInitConv"));

	{
		auto* P = GraphBuilder.AllocParameters<FConv3x3Stride2BnRelu6CS::FParameters>();
		P->InWidth = CropSize; P->InHeight = CropSize; P->InChannels = 3;
		P->OutWidth = 112; P->OutHeight = 112; P->OutChannels = 24;
		P->InputBuffer = GraphBuilder.CreateSRV(CropBuf);
		P->Weights = GetWeightSRV(GraphBuilder, LandmarkWeightBuffers, TEXT("conv2d"));
		P->Bias = GetWeightSRV(GraphBuilder, LandmarkWeightBuffers, TEXT("batch_normalization"));
		P->OutputBuffer = GraphBuilder.CreateUAV(CurrBuf);

		TShaderMapRef<FConv3x3Stride2BnRelu6CS> CS(ShaderMap);
		FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("LmInitConv"), CS, P,
			FIntVector(DivUp(112, 8), DivUp(112, 8), 24));
	}

	// ---- 3. 16 MBConv blocks ----
	int32 ConvIdx = 1;
	int32 BnIdx = 1;
	int32 CurrSpatial = 112;

	for (int32 BlockIdx = 0; BlockIdx < 16; BlockIdx++)
	{
		const FMBConvBlock& B = LandmarkBlocks[BlockIdx];
		bool bHasExpand = (B.ExpandCh != B.InCh);
		int32 SpatialIn = CurrSpatial;

		FRDGBufferRef BlockInput = CurrBuf;
		int32 ActiveCh = B.InCh;

		// -- Expand 1×1 (if needed) --
		if (bHasExpand)
		{
			FString ExpandConvKey = FString::Printf(TEXT("conv2d_%d"), ConvIdx);
			FString ExpandBnKey = BnIdx == 0 ? TEXT("batch_normalization") : FString::Printf(TEXT("batch_normalization_%d"), BnIdx);

			FRDGBufferRef ExpandOut = GraphBuilder.CreateBuffer(
				FRDGBufferDesc::CreateStructuredDesc(sizeof(float), B.ExpandCh * SpatialIn * SpatialIn),
				*FString::Printf(TEXT("LmExp%d"), BlockIdx));

			auto* P = GraphBuilder.AllocParameters<FConv1x1BnCS::FParameters>();
			P->Width = SpatialIn; P->Height = SpatialIn;
			P->InChannels = B.InCh; P->OutChannels = B.ExpandCh;
			P->InputBuffer = GraphBuilder.CreateSRV(CurrBuf);
			P->Weights = GetWeightSRV(GraphBuilder, LandmarkWeightBuffers, ExpandConvKey);
			P->Bias = GetWeightSRV(GraphBuilder, LandmarkWeightBuffers, ExpandBnKey);
			P->OutputBuffer = GraphBuilder.CreateUAV(ExpandOut);

			FConv1x1BnCS::FPermutationDomain PD;
			PD.Set<FConv1x1BnCS::FHasReLU6>(true);
			TShaderMapRef<FConv1x1BnCS> CS(ShaderMap, PD);
			FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("LmExpand%d", BlockIdx), CS, P,
				FIntVector(DivUp(SpatialIn, 8), DivUp(SpatialIn, 8), B.ExpandCh));

			CurrBuf = ExpandOut;
			ActiveCh = B.ExpandCh;
			ConvIdx++;
			BnIdx++;
		}

		// -- Depthwise conv --
		FString DwWeightKey = FString::Printf(TEXT("batch_normalization_%d/FusedBatchNormV3"), BnIdx);
		FString DwBiasKey = BnIdx == 0 ? TEXT("batch_normalization") : FString::Printf(TEXT("batch_normalization_%d"), BnIdx);

		int32 DwOutSpatial = B.SpatialOut;

		int32 PadTotal = FMath::Max(0, (DwOutSpatial - 1) * B.Stride + B.DwKernel - SpatialIn);
		int32 PadBefore = PadTotal / 2;

		FRDGBufferRef DwOut = GraphBuilder.CreateBuffer(
			FRDGBufferDesc::CreateStructuredDesc(sizeof(float), ActiveCh * DwOutSpatial * DwOutSpatial),
			*FString::Printf(TEXT("LmDW%d"), BlockIdx));

		{
			auto* P = GraphBuilder.AllocParameters<FDepthwiseConvBnRelu6CS::FParameters>();
			P->InWidth = SpatialIn; P->InHeight = SpatialIn; P->Channels = ActiveCh;
			P->OutWidth = DwOutSpatial; P->OutHeight = DwOutSpatial;
			P->Stride = B.Stride; P->KernelSize = B.DwKernel; P->Pad = PadBefore;
			P->InputBuffer = GraphBuilder.CreateSRV(CurrBuf);
			P->Weights = GetWeightSRV(GraphBuilder, LandmarkWeightBuffers, DwWeightKey);
			P->Bias = GetWeightSRV(GraphBuilder, LandmarkWeightBuffers, DwBiasKey);
			P->OutputBuffer = GraphBuilder.CreateUAV(DwOut);

			TShaderMapRef<FDepthwiseConvBnRelu6CS> CS(ShaderMap);
			FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("LmDW%d", BlockIdx), CS, P,
				FIntVector(DivUp(DwOutSpatial, 8), DivUp(DwOutSpatial, 8), ActiveCh));
		}

		CurrBuf = DwOut;
		BnIdx++;

		// -- Project 1×1 (if needed) --
		if (B.bHasProject)
		{
			FString ProjConvKey = FString::Printf(TEXT("conv2d_%d"), ConvIdx);
			FString ProjBnKey = FString::Printf(TEXT("batch_normalization_%d/FusedBatchNormV3"), BnIdx);

			FRDGBufferRef ProjOut = GraphBuilder.CreateBuffer(
				FRDGBufferDesc::CreateStructuredDesc(sizeof(float), B.OutCh * DwOutSpatial * DwOutSpatial),
				*FString::Printf(TEXT("LmProj%d"), BlockIdx));

			auto* P = GraphBuilder.AllocParameters<FConv1x1BnCS::FParameters>();
			P->Width = DwOutSpatial; P->Height = DwOutSpatial;
			P->InChannels = ActiveCh; P->OutChannels = B.OutCh;
			P->InputBuffer = GraphBuilder.CreateSRV(CurrBuf);
			P->Weights = GetWeightSRV(GraphBuilder, LandmarkWeightBuffers, ProjConvKey);
			P->Bias = GetWeightSRV(GraphBuilder, LandmarkWeightBuffers, ProjBnKey);
			P->OutputBuffer = GraphBuilder.CreateUAV(ProjOut);

			FConv1x1BnCS::FPermutationDomain PD;
			PD.Set<FConv1x1BnCS::FHasReLU6>(false);
			TShaderMapRef<FConv1x1BnCS> CS(ShaderMap, PD);
			FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("LmProj%d", BlockIdx), CS, P,
				FIntVector(DivUp(DwOutSpatial, 8), DivUp(DwOutSpatial, 8), B.OutCh));

			CurrBuf = ProjOut;
			ConvIdx++;
			BnIdx++;
		}

		// -- Residual add --
		if (B.bHasResidual && B.Stride == 1 && B.InCh == B.OutCh)
		{
			int32 Count = B.OutCh * DwOutSpatial * DwOutSpatial;
			FRDGBufferRef ResOut = GraphBuilder.CreateBuffer(
				FRDGBufferDesc::CreateStructuredDesc(sizeof(float), Count),
				*FString::Printf(TEXT("LmRes%d"), BlockIdx));

			auto* P = GraphBuilder.AllocParameters<FElementwiseAddCS::FParameters>();
			P->Count = Count;
			P->BufferA = GraphBuilder.CreateSRV(CurrBuf);
			P->BufferB = GraphBuilder.CreateSRV(BlockInput);
			P->OutputBuffer = GraphBuilder.CreateUAV(ResOut);

			TShaderMapRef<FElementwiseAddCS> CS(ShaderMap);
			FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("LmRes%d", BlockIdx), CS, P,
				FIntVector(DivUp(Count, 256), 1, 1));

			CurrBuf = ResOut;
		}

		CurrSpatial = DwOutSpatial;
	}

	// ---- 4. Global Average Pooling: 1152ch×7×7 → 1152 ----
	constexpr int32 GapCh = 1152;
	constexpr int32 GapSpatial = 7;

	FRDGBufferRef GapOut = GraphBuilder.CreateBuffer(
		FRDGBufferDesc::CreateStructuredDesc(sizeof(float), GapCh), TEXT("LmGAP"));

	{
		auto* P = GraphBuilder.AllocParameters<FGlobalAvgPoolCS::FParameters>();
		P->Channels = GapCh;
		P->SpatialSize = GapSpatial * GapSpatial;
		P->InputBuffer = GraphBuilder.CreateSRV(CurrBuf);
		P->OutputBuffer = GraphBuilder.CreateUAV(GapOut);

		TShaderMapRef<FGlobalAvgPoolCS> CS(ShaderMap);
		FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("LmGAP"), CS, P,
			FIntVector(DivUp(GapCh, 256), 1, 1));
	}

	// ---- 5. FC heads ----
	// Landmarks: 1152→63, no sigmoid
	FRDGBufferRef LmOut = GraphBuilder.CreateBuffer(
		FRDGBufferDesc::CreateStructuredDesc(sizeof(float), 63), TEXT("LmFC_Landmarks"));
	{
		auto* P = GraphBuilder.AllocParameters<FFCMatMulCS::FParameters>();
		P->InFeatures = GapCh; P->OutFeatures = 63;
		P->InputBuffer = GraphBuilder.CreateSRV(GapOut);
		P->Weights = GetWeightSRV(GraphBuilder, LandmarkWeightBuffers, TEXT("conv_landmarks"));
		P->Bias = GetWeightSRV(GraphBuilder, LandmarkWeightBuffers, TEXT("Identity"));
		P->OutputBuffer = GraphBuilder.CreateUAV(LmOut);

		FFCMatMulCS::FPermutationDomain PD;
		PD.Set<FFCMatMulCS::FHasSigmoid>(false);
		TShaderMapRef<FFCMatMulCS> CS(ShaderMap, PD);
		FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("LmFC_Landmarks"), CS, P,
			FIntVector(1, 1, 1));
	}

	// Handflag: 1152→1, sigmoid
	FRDGBufferRef HfOut = GraphBuilder.CreateBuffer(
		FRDGBufferDesc::CreateStructuredDesc(sizeof(float), 1), TEXT("LmFC_Handflag"));
	{
		auto* P = GraphBuilder.AllocParameters<FFCMatMulCS::FParameters>();
		P->InFeatures = GapCh; P->OutFeatures = 1;
		P->InputBuffer = GraphBuilder.CreateSRV(GapOut);
		P->Weights = GetWeightSRV(GraphBuilder, LandmarkWeightBuffers, TEXT("conv_handflag"));
		P->Bias = GetWeightSRV(GraphBuilder, LandmarkWeightBuffers, TEXT("Identity_1"));
		P->OutputBuffer = GraphBuilder.CreateUAV(HfOut);

		FFCMatMulCS::FPermutationDomain PD;
		PD.Set<FFCMatMulCS::FHasSigmoid>(true);
		TShaderMapRef<FFCMatMulCS> CS(ShaderMap, PD);
		FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("LmFC_Handflag"), CS, P,
			FIntVector(1, 1, 1));
	}

	// Handedness: 1152→1, sigmoid
	FRDGBufferRef HdOut = GraphBuilder.CreateBuffer(
		FRDGBufferDesc::CreateStructuredDesc(sizeof(float), 1), TEXT("LmFC_Handedness"));
	{
		auto* P = GraphBuilder.AllocParameters<FFCMatMulCS::FParameters>();
		P->InFeatures = GapCh; P->OutFeatures = 1;
		P->InputBuffer = GraphBuilder.CreateSRV(GapOut);
		P->Weights = GetWeightSRV(GraphBuilder, LandmarkWeightBuffers, TEXT("conv_handedness"));
		P->Bias = GetWeightSRV(GraphBuilder, LandmarkWeightBuffers, TEXT("Identity_2"));
		P->OutputBuffer = GraphBuilder.CreateUAV(HdOut);

		FFCMatMulCS::FPermutationDomain PD;
		PD.Set<FFCMatMulCS::FHasSigmoid>(true);
		TShaderMapRef<FFCMatMulCS> CS(ShaderMap, PD);
		FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("LmFC_Handedness"), CS, P,
			FIntVector(1, 1, 1));
	}

	// ---- 6. Readback (3 buffers per hand) ----
	int32 BaseIdx = HandIndex * 3;
	// Ensure readback objects exist
	while (LandmarkReadbacks.Num() <= BaseIdx + 2)
	{
		LandmarkReadbacks.Add(new FRHIGPUBufferReadback(
			*FString::Printf(TEXT("LmReadback_%d"), LandmarkReadbacks.Num())));
	}

	AddEnqueueCopyPass(GraphBuilder, LandmarkReadbacks[BaseIdx + 0], LmOut, 63 * sizeof(float));
	AddEnqueueCopyPass(GraphBuilder, LandmarkReadbacks[BaseIdx + 1], HfOut, 1 * sizeof(float));
	AddEnqueueCopyPass(GraphBuilder, LandmarkReadbacks[BaseIdx + 2], HdOut, 1 * sizeof(float));
}

// ============================================================================
// RunPipeline — async multi-frame state machine
// ============================================================================

TArray<FHandposeResult> FHandposeDetector::RunPipeline(
	FRDGBuilder& GraphBuilder,
	FRDGTexture* InputTexture,
	int32 MaxHands,
	float ScoreThreshold,
	float PalmScoreThreshold)
{
	if (!bInitialized)
	{
		UE_LOG(LogMicroHandpose, Warning, TEXT("[MicroHandpose] RunPipeline called before Initialize"));
		return {};
	}

	switch (Phase)
	{
	case EPipelinePhase::Idle:
	{
		UE_LOG(LogMicroHandpose, Verbose, TEXT("Pipeline: Idle -> dispatching palm detection"));
		DispatchPalmDetection(GraphBuilder, InputTexture);
		Phase = EPipelinePhase::PalmDispatched;
		break;
	}

	case EPipelinePhase::PalmDispatched:
	{
		if (!PalmCls16Readback->IsReady())
		{
			UE_LOG(LogMicroHandpose, VeryVerbose, TEXT("Pipeline: PalmDispatched, readback not ready yet"));
			break;
		}

		// Read back SSD outputs
		auto ReadBuffer = [](FRHIGPUBufferReadback* Rb, int32 Count) -> TArray<float>
		{
			TArray<float> Data;
			Data.SetNumUninitialized(Count);
			const float* Src = (const float*)Rb->Lock(Count * sizeof(float));
			FMemory::Memcpy(Data.GetData(), Src, Count * sizeof(float));
			Rb->Unlock();
			return Data;
		};

		TArray<float> Cls16 = ReadBuffer(PalmCls16Readback,   6 * 144);
		TArray<float> Reg16 = ReadBuffer(PalmReg16Readback, 108 * 144);
		TArray<float> Cls8  = ReadBuffer(PalmCls8Readback,    2 * 576);
		TArray<float> Reg8  = ReadBuffer(PalmReg8Readback,   36 * 576);

		// Reorganize CHW→HWC, decode, NMS
		TArray<float> Scores, Regressors;
		ReorganizeSSDOutput(Cls16, Reg16, Cls8, Reg8, Scores, Regressors);

		TArray<FPalmDetection> Detections = PalmPostProcess->DecodeDetections(Scores, Regressors, PalmScoreThreshold);
		UE_LOG(LogMicroHandpose, Verbose, TEXT("Pipeline: palm readback complete — %d raw detections (threshold=%.2f)"),
			Detections.Num(), PalmScoreThreshold);

		FPalmDetectionPostProcess::RemoveLetterboxPadding(Detections, LetterboxPadX, LetterboxPadY);
		TArray<FPalmDetection> NmsDetections = FPalmDetectionPostProcess::WeightedNMS(Detections, 0.3f);
		UE_LOG(LogMicroHandpose, Verbose, TEXT("Pipeline: %d detections after NMS"), NmsDetections.Num());

		// Convert to ROIs and dispatch landmark inference
		PendingROIs.Empty();
		LandmarkROIs.Empty();
		int32 NumHands = FMath::Min(NmsDetections.Num(), MaxHands);

		for (int32 i = 0; i < NumHands; i++)
		{
			FLandmarkPostProcess::FPixelROI PixelROI = FLandmarkPostProcess::DetectionToPixelROI(NmsDetections[i], SrcWidth, SrcHeight);
			PendingROIs.Add(PixelROI);
			LandmarkROIs.Add(PixelROI);
			DispatchLandmarkInference(GraphBuilder, InputTexture, PixelROI, i);
		}

		if (NumHands > 0)
		{
			UE_LOG(LogMicroHandpose, Verbose, TEXT("Pipeline: dispatched landmark inference for %d hand(s)"), NumHands);
			Phase = EPipelinePhase::LandmarkDispatched;
		}
		else
		{
			UE_LOG(LogMicroHandpose, Verbose, TEXT("Pipeline: no hands found, returning to Idle"));
			Phase = EPipelinePhase::Idle;
		}
		break;
	}

	case EPipelinePhase::LandmarkDispatched:
	{
		if (LandmarkReadbacks.Num() == 0 || !LandmarkReadbacks[0]->IsReady())
		{
			UE_LOG(LogMicroHandpose, VeryVerbose, TEXT("Pipeline: LandmarkDispatched, readback not ready yet"));
			break;
		}

		CachedResults.Empty();
		int32 NumHands = LandmarkROIs.Num();

		for (int32 h = 0; h < NumHands; h++)
		{
			int32 BaseIdx = h * 3;

			const float* LmData = (const float*)LandmarkReadbacks[BaseIdx + 0]->Lock(63 * sizeof(float));
			const float* HfData = (const float*)LandmarkReadbacks[BaseIdx + 1]->Lock(1 * sizeof(float));
			const float* HdData = (const float*)LandmarkReadbacks[BaseIdx + 2]->Lock(1 * sizeof(float));

			TArray<float> RawLandmarks;
			RawLandmarks.SetNumUninitialized(63);
			FMemory::Memcpy(RawLandmarks.GetData(), LmData, 63 * sizeof(float));

			float HandFlag = HfData[0];
			float Handedness = HdData[0];

			LandmarkReadbacks[BaseIdx + 0]->Unlock();
			LandmarkReadbacks[BaseIdx + 1]->Unlock();
			LandmarkReadbacks[BaseIdx + 2]->Unlock();

			if (HandFlag >= ScoreThreshold)
			{
				FHandposeResult Result = FLandmarkPostProcess::DenormalizeLandmarks(
					RawLandmarks, HandFlag, Handedness, LandmarkROIs[h], SrcWidth, SrcHeight);
				CachedResults.Add(Result);
				UE_LOG(LogMicroHandpose, Verbose, TEXT("Pipeline: hand %d accepted (flag=%.3f, handedness=%.3f -> %s)"),
					h, HandFlag, Handedness, Handedness > 0.5f ? TEXT("Right") : TEXT("Left"));
			}
			else
			{
				UE_LOG(LogMicroHandpose, Verbose, TEXT("Pipeline: hand %d rejected (flag=%.3f < threshold %.2f)"),
					h, HandFlag, ScoreThreshold);
			}
		}

		UE_LOG(LogMicroHandpose, Verbose, TEXT("Pipeline: landmark readback complete — %d hand(s) returned, back to Idle"),
			CachedResults.Num());
		Phase = EPipelinePhase::Idle;
		break;
	}
	}

	return CachedResults;
}
