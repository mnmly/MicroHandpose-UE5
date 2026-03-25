#pragma once

#include "CoreMinimal.h"
#include "HandposeTypes.generated.h"

/** Which hand was detected */
UENUM(BlueprintType)
enum class EHandedness : uint8
{
	Left UMETA(DisplayName = "Left"),
	Right UMETA(DisplayName = "Right"),
};

/** Landmark indices for the 21 hand keypoints */
UENUM(BlueprintType)
enum class EHandLandmark : uint8
{
	Wrist = 0,
	ThumbCMC = 1,
	ThumbMCP = 2,
	ThumbIP = 3,
	ThumbTip = 4,
	IndexMCP = 5,
	IndexPIP = 6,
	IndexDIP = 7,
	IndexTip = 8,
	MiddleMCP = 9,
	MiddlePIP = 10,
	MiddleDIP = 11,
	MiddleTip = 12,
	RingMCP = 13,
	RingPIP = 14,
	RingDIP = 15,
	RingTip = 16,
	PinkyMCP = 17,
	PinkyPIP = 18,
	PinkyDIP = 19,
	PinkyTip = 20,
};

static constexpr int32 NUM_HAND_LANDMARKS = 21;

/** A single 3D landmark point */
USTRUCT(BlueprintType)
struct MICROHANDPOSE_API FHandLandmark
{
	GENERATED_BODY()

	/** Position normalized to [0,1] relative to input image (X=right, Y=down, Z=depth) */
	UPROPERTY(BlueprintReadOnly, Category = "Handpose")
	FVector Position = FVector::ZeroVector;
};

/** Detection result for a single hand */
USTRUCT(BlueprintType)
struct MICROHANDPOSE_API FHandposeResult
{
	GENERATED_BODY()

	/** Detection confidence score (0-1) */
	UPROPERTY(BlueprintReadOnly, Category = "Handpose")
	float Score = 0.0f;

	/** Left or right hand */
	UPROPERTY(BlueprintReadOnly, Category = "Handpose")
	EHandedness Handedness = EHandedness::Left;

	/** 21 landmark points */
	UPROPERTY(BlueprintReadOnly, Category = "Handpose")
	TArray<FHandLandmark> Landmarks;

	/** Get a specific landmark by index */
	const FHandLandmark& GetLandmark(EHandLandmark Index) const
	{
		return Landmarks[static_cast<int32>(Index)];
	}

	const FHandLandmark& GetWrist() const { return GetLandmark(EHandLandmark::Wrist); }
	const FHandLandmark& GetThumbTip() const { return GetLandmark(EHandLandmark::ThumbTip); }
	const FHandLandmark& GetIndexTip() const { return GetLandmark(EHandLandmark::IndexTip); }
	const FHandLandmark& GetMiddleTip() const { return GetLandmark(EHandLandmark::MiddleTip); }
	const FHandLandmark& GetRingTip() const { return GetLandmark(EHandLandmark::RingTip); }
	const FHandLandmark& GetPinkyTip() const { return GetLandmark(EHandLandmark::PinkyTip); }
};

/** Delegate for hand pose updates */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnHandposeUpdated, const TArray<FHandposeResult>&, Results);
