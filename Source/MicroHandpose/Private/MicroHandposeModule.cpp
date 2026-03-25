#include "MicroHandposeModule.h"
#include "Interfaces/IPluginManager.h"

DEFINE_LOG_CATEGORY(LogMicroHandpose);

#define LOCTEXT_NAMESPACE "FMicroHandposeModule"

void FMicroHandposeModule::StartupModule()
{
	FString PluginName = TEXT("MicroHandpose");
	TSharedPtr<IPlugin> Plugin = IPluginManager::Get().FindPlugin(PluginName);

	if (Plugin.IsValid())
	{
		FString PluginShaderDir = FPaths::Combine(Plugin->GetBaseDir(), TEXT("Shaders"));
		AddShaderSourceDirectoryMapping(TEXT("/Plugin/MicroHandpose"), PluginShaderDir);
		UE_LOG(LogMicroHandpose, Log, TEXT("[MicroHandpose] Shaders registered from: %s"), *PluginShaderDir);
	}
	else
	{
		UE_LOG(LogMicroHandpose, Error, TEXT("[MicroHandpose] Could not find plugin '%s'. Shaders will not load!"), *PluginName);
	}
}

void FMicroHandposeModule::ShutdownModule()
{
	ResetAllShaderSourceDirectoryMappings();
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FMicroHandposeModule, MicroHandpose)
