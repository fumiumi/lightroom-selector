return {
  LrSdkVersion      = 12.0,
  LrSdkMinimumVersion = 10.0,
  LrToolkitIdentifier = 'jp.fumiya.autorate',
  LrPluginName      = "Auto-Rate (JSON)",
  LrPluginInfoUrl   = "https://github.com/fumiumi/lightroom-selector",
  LrDescription     = "rating_map.json から星を自動付与します。",
  -- メニューに登録
  LrExportMenuItems = {
    {
      title = "Apply Ratings from JSON...",
      file  = "AutoRate.lua",   -- 呼び出し先
    },
  },
}
