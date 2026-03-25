#include "WeightLoader.h"
#include "MicroHandposeModule.h"
#include "Misc/FileHelper.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

const FWeightTensor* FWeightCollection::FindBySubstrings(const TArray<FString>& Substrings) const
{
	for (const auto& Pair : Tensors)
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
		if (bAllMatch) return &Pair.Value;
	}
	return nullptr;
}

float FWeightLoader::Float16ToFloat32(uint16 h)
{
	uint32 Sign = (h >> 15) & 0x1;
	uint32 Exponent = (h >> 10) & 0x1F;
	uint32 Mantissa = h & 0x3FF;

	if (Exponent == 0)
	{
		if (Mantissa == 0)
		{
			// Zero
			uint32 Result = Sign << 31;
			return *reinterpret_cast<float*>(&Result);
		}
		// Denormalized
		float M = static_cast<float>(Mantissa) / 1024.0f;
		float Value = FMath::Pow(2.0f, -14.0f) * M;
		return Sign ? -Value : Value;
	}

	if (Exponent == 0x1F)
	{
		if (Mantissa == 0)
		{
			// Infinity
			uint32 Result = (Sign << 31) | 0x7F800000;
			return *reinterpret_cast<float*>(&Result);
		}
		// NaN
		return std::numeric_limits<float>::quiet_NaN();
	}

	// Normalized
	int32 E = static_cast<int32>(Exponent) - 15;
	float M = 1.0f + static_cast<float>(Mantissa) / 1024.0f;
	float Value = FMath::Pow(2.0f, static_cast<float>(E)) * M;
	return Sign ? -Value : Value;
}

FWeightCollection FWeightLoader::LoadWeights(const FString& JsonPath, const FString& BinPath)
{
	FWeightCollection Collection;

	// Load JSON metadata
	FString JsonString;
	if (!FFileHelper::LoadFileToString(JsonString, *JsonPath))
	{
		UE_LOG(LogMicroHandpose, Error, TEXT("[MicroHandpose] Failed to load weight metadata: %s"), *JsonPath);
		return Collection;
	}

	TSharedPtr<FJsonObject> JsonObj;
	TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonString);
	if (!FJsonSerializer::Deserialize(Reader, JsonObj) || !JsonObj.IsValid())
	{
		UE_LOG(LogMicroHandpose, Error, TEXT("[MicroHandpose] Failed to parse weight JSON: %s"), *JsonPath);
		return Collection;
	}

	// Parse metadata fields
	const TArray<TSharedPtr<FJsonValue>>* Keys;
	const TArray<TSharedPtr<FJsonValue>>* Shapes;
	const TArray<TSharedPtr<FJsonValue>>* Offsets;
	if (!JsonObj->TryGetArrayField(TEXT("keys"), Keys) ||
		!JsonObj->TryGetArrayField(TEXT("shapes"), Shapes) ||
		!JsonObj->TryGetArrayField(TEXT("offsets"), Offsets))
	{
		UE_LOG(LogMicroHandpose, Error, TEXT("[MicroHandpose] Weight JSON missing required fields (keys/shapes/offsets)"));
		return Collection;
	}

	FString DType = TEXT("float32");
	JsonObj->TryGetStringField(TEXT("dtype"), DType);
	bool bIsFloat16 = DType == TEXT("float16");

	int32 NumTensors = Keys->Num();
	if (Shapes->Num() != NumTensors || Offsets->Num() != NumTensors)
	{
		UE_LOG(LogMicroHandpose, Error, TEXT("[MicroHandpose] Weight JSON arrays have mismatched lengths"));
		return Collection;
	}

	// Load binary data
	TArray<uint8> BinData;
	if (!FFileHelper::LoadFileToArray(BinData, *BinPath))
	{
		UE_LOG(LogMicroHandpose, Error, TEXT("[MicroHandpose] Failed to load weight binary: %s"), *BinPath);
		return Collection;
	}

	UE_LOG(LogMicroHandpose, Log, TEXT("[MicroHandpose] Loading %d weight tensors from %s (dtype=%s, %d bytes)"),
		NumTensors, *BinPath, *DType, BinData.Num());

	// Parse each tensor
	for (int32 i = 0; i < NumTensors; i++)
	{
		FWeightTensor Tensor;
		Tensor.Key = (*Keys)[i]->AsString();

		// Parse shape
		const TArray<TSharedPtr<FJsonValue>>* ShapeArray;
		if ((*Shapes)[i]->TryGetArray(ShapeArray))
		{
			for (const auto& Dim : *ShapeArray)
			{
				Tensor.Shape.Add(static_cast<int32>(Dim->AsNumber()));
			}
		}
		else
		{
			// Single scalar shape
			Tensor.Shape.Add(1);
		}

		int32 Offset = static_cast<int32>((*Offsets)[i]->AsNumber());
		int32 NumElements = Tensor.NumElements();

		// Read data
		Tensor.Data.SetNumUninitialized(NumElements);

		if (bIsFloat16)
		{
			int32 ByteSize = NumElements * 2;
			if (Offset + ByteSize > BinData.Num())
			{
				UE_LOG(LogMicroHandpose, Error, TEXT("[MicroHandpose] Weight tensor '%s' overflows binary (offset=%d, size=%d, file=%d)"),
					*Tensor.Key, Offset, ByteSize, BinData.Num());
				continue;
			}

			const uint16* SrcPtr = reinterpret_cast<const uint16*>(BinData.GetData() + Offset);
			for (int32 j = 0; j < NumElements; j++)
			{
				Tensor.Data[j] = Float16ToFloat32(SrcPtr[j]);
			}
		}
		else
		{
			int32 ByteSize = NumElements * 4;
			if (Offset + ByteSize > BinData.Num())
			{
				UE_LOG(LogMicroHandpose, Error, TEXT("[MicroHandpose] Weight tensor '%s' overflows binary (offset=%d, size=%d, file=%d)"),
					*Tensor.Key, Offset, ByteSize, BinData.Num());
				continue;
			}

			FMemory::Memcpy(Tensor.Data.GetData(), BinData.GetData() + Offset, ByteSize);
		}

		Collection.Tensors.Add(Tensor.Key, MoveTemp(Tensor));
	}

	UE_LOG(LogMicroHandpose, Log, TEXT("[MicroHandpose] Successfully loaded %d weight tensors"), Collection.Tensors.Num());
	return Collection;
}

TArray<float> FWeightLoader::TransposeDepthwiseWeights(const FWeightTensor& Input)
{
	// Input: [1, kH, kW, channels] → Output: [channels, kH*kW]
	check(Input.Shape.Num() == 4);
	int32 kH = Input.Shape[1];
	int32 kW = Input.Shape[2];
	int32 Ch = Input.Shape[3];
	int32 KernelSize = kH * kW;

	TArray<float> Result;
	Result.SetNumUninitialized(Ch * KernelSize);

	for (int32 c = 0; c < Ch; c++)
	{
		for (int32 ky = 0; ky < kH; ky++)
		{
			for (int32 kx = 0; kx < kW; kx++)
			{
				// TFLite layout: [1, ky, kx, c] → index = ky * kW * Ch + kx * Ch + c
				// Output layout: [c, ky * kW + kx]
				Result[c * KernelSize + ky * kW + kx] = Input.Data[ky * kW * Ch + kx * Ch + c];
			}
		}
	}

	return Result;
}
