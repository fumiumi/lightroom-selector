local LrDialogs  = import 'LrDialogs'
local LrPathUtils= import 'LrPathUtils'
local LrTasks    = import 'LrTasks'
local LrFileUtils= import 'LrFileUtils'
local catalog    = import 'LrApplication'.activeCatalog()

-- Lua に JSON パーサが無いので同梱
local json = require 'dkjson'  -- github.com/LuaDist/dkjson をフォルダに置く

local function chooseJson()
  return LrDialogs.runOpenPanel{
    title = "Choose rating_map.json",
    canChooseFiles = true,
    allowedFileTypes = { "json" },
  }
end

local function applyRatings(ratingMap)
  local applied, missed = 0, 0
  catalog:withWriteAccessDo("Auto-Rate", function()
    for _, photo in ipairs(catalog:getAllPhotos()) do
      local base = photo:getFormattedMetadata('fileName'):match('^(.*)%.')
      local rating = ratingMap[base]
      if rating then
        photo:setRawMetadata('rating', rating)
        applied = applied + 1
      else
        missed = missed + 1
      end
    end
  end)
  return applied, missed
end

LrTasks.startAsyncTask(function()
  local sel = chooseJson()
  if not (sel and #sel > 0) then return end
  local jsonPath = sel[1]
  local txt = LrFileUtils.readFile(jsonPath)
  local map, pos, err = json.decode(txt)
  if not map then
    LrDialogs.message("JSON parse error", err, "critical")
    return
  end
  local ok, applied, missed = pcall(applyRatings, map)
  if ok then
    LrDialogs.message("Auto-Rate",
      string.format("Applied ★ to %d photos (missed %d)", applied, missed))
  else
    LrDialogs.message("Error", applied, "critical")
  end
end)
