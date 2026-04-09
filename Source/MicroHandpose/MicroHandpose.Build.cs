using System.IO;
using UnrealBuildTool;

public class MicroHandpose : ModuleRules
{
	public MicroHandpose(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicIncludePaths.AddRange(
			new string[] {
				Path.Combine(GetModuleDirectory("Renderer"), "Private"),
				GetModuleDirectory("MicroHandpose"),
			}
		);

		PrivateIncludePaths.AddRange(
			new string[] {
			}
		);

		// UE 5.6+ moved some Renderer headers to Internal
		if (Target.Version is { MajorVersion: 5, MinorVersion: > 5 })
		{
			PrivateIncludePaths.AddRange(
				new string[] {
					Path.Combine(GetModuleDirectory("Renderer"), "Internal"),
				}
			);
		}

		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"Engine",
			}
		);

		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"CoreUObject",
				"Engine",
				"Slate",
				"SlateCore",
				"Projects",
				"RHI",
				"Renderer",
				"RenderCore",
				"Json",
				"JsonUtilities",
				"MediaAssets",
			}
		);
	}
}
