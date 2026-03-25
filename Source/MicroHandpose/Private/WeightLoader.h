#pragma once

#include "CoreMinimal.h"
#include "RHIResources.h"

/**
 * Represents a single weight tensor loaded from the binary weight file.
 * Data is stored as float32 (converted from f16 if needed).
 */
struct FWeightTensor
{
	FString Key;
	TArray<int32> Shape;
	TArray<float> Data;

	int32 NumElements() const
	{
		int32 Count = 1;
		for (int32 Dim : Shape) Count *= Dim;
		return Count;
	}
};

/**
 * Collection of weight tensors loaded from a JSON metadata + binary file pair.
 */
struct FWeightCollection
{
	TMap<FString, FWeightTensor> Tensors;

	const FWeightTensor* Find(const FString& Key) const { return Tensors.Find(Key); }

	/** Find a tensor whose key contains all of the given substrings */
	const FWeightTensor* FindBySubstrings(const TArray<FString>& Substrings) const;
};

/**
 * Loads neural network weights from JSON metadata + binary file pairs.
 * Handles both float32 and float16 binary formats.
 */
class FWeightLoader
{
public:
	/**
	 * Load weights from a JSON metadata file + binary data file.
	 * @param JsonPath Absolute path to the .json metadata file
	 * @param BinPath  Absolute path to the .bin binary data file
	 * @return Loaded weight collection, or empty on failure
	 */
	static FWeightCollection LoadWeights(const FString& JsonPath, const FString& BinPath);

	/**
	 * Transpose depthwise convolution weights from TFLite layout [1,kH,kW,ch] to [ch,25].
	 * @param Input The original weight tensor
	 * @return Transposed float array
	 */
	static TArray<float> TransposeDepthwiseWeights(const FWeightTensor& Input);

private:
	/** Convert IEEE 754 half-precision float to single-precision */
	static float Float16ToFloat32(uint16 HalfFloat);
};
