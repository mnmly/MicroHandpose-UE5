#include "HandposeComponent.h"
#include "HandposeSubsystem.h"

UHandposeComponent::UHandposeComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
	PrimaryComponentTick.bStartWithTickEnabled = true;
}

void UHandposeComponent::BeginPlay()
{
	Super::BeginPlay();
	bWasTracking = false;
}

void UHandposeComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	Super::EndPlay(EndPlayReason);
}

void UHandposeComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	UHandposeSubsystem* Subsystem = UHandposeSubsystem::GetInstance();
	if (!Subsystem)
	{
		return;
	}

	HandResults = Subsystem->GetLatestResults();

	bool bIsNowTracking = HasTrackedHand();

	if (bIsNowTracking && !bWasTracking)
	{
		OnHandDetected.Broadcast(HandResults[TrackedHandIndex]);
	}
	else if (!bIsNowTracking && bWasTracking)
	{
		OnHandLost.Broadcast();
	}

	bWasTracking = bIsNowTracking;
}

FVector UHandposeComponent::GetLandmarkPosition(EHandLandmark Landmark) const
{
	if (HasTrackedHand())
	{
		const FHandposeResult& Hand = HandResults[TrackedHandIndex];
		int32 Idx = static_cast<int32>(Landmark);
		if (Hand.Landmarks.IsValidIndex(Idx))
		{
			return Hand.Landmarks[Idx].Position;
		}
	}
	return FVector::ZeroVector;
}

bool UHandposeComponent::HasTrackedHand() const
{
	return HandResults.IsValidIndex(TrackedHandIndex);
}
