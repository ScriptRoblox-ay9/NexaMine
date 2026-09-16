local ffi = require("ffi")

local cfg = {
    auto_profile = true,
    force_profile = nil,
    grid_step = 2.0,
    map_size_local = 36,
    map_update_rate = 90,
    map_update_speed = 60,
    map_update_time_budget_ms = 3.0,
    sliding_window = true,
    prefer_forward = true,
    behind_discard_dist = 1.35,
    max_height_layers = 1,
    layer_min_gap = 2.5,
    scan_height_start = 800.0,
    scan_height_range = 1200.0,
    neighbor_z_tolerance = 3.5,
    max_step_height = 0.9,
    headroom = 1.8,
    max_slope = 0.8,
    slope_sample_radius = 1.5,
    slope_sample_range = 6.0,
    slope_min_ok_neighbors = 2,
    slope_avg_factor = 1.5,
    slope_max_factor = 2.0,
    step_min_ok_neighbors = 4,
    los_sample_heights = {0.2, 0.9, 1.6},
    los_side_offset = 0.35,
    height_heuristic_weight = 0.75,
    a_star_weight = 1.25,
    los_check_buildings = true,
    los_check_objects = true,
    los_check_dummies = true,
    los_check_vehicles = true,
    los_check_peds = false,
    los_see_through = false,
    los_ignore_camera = false,
    los_shoot_through = false,
    cache_max_size = 5000,
    cache_cleanup_threshold = 4000,
    link_cache_max_size = 50000,
    link_cache_cleanup_threshold = 40000,
    rrt_enabled = true,
    rrt_max_iters = 600,
    rrt_step = 4.5,
    rrt_goal_bias = 0.15,
    rrt_radius = 10.0,
    rrt_goal_radius = 6.0,
    rrt_rewire_limit = 12,
    rrt_nearest_candidates = 40,
    rrt_min_distance = 20.0,
    path_string_pull = true,
    path_smooth = false,          
    path_smooth_step = 1.0,       
    point_query_cache_radius = 6.0,
    point_snap_dist = 16.0,
    gpu_enabled = true,
    gpu_fallback_cpu = true,
    gpu_batch_size = 256,
    gpu_use_for_distances = true,
    gpu_use_for_priority = true,
}

local PROFILES = {
    low = {
        grid_step = 2.5,
        map_size_local = 28,
        map_update_rate = 120,
        map_update_speed = 40,
        map_update_time_budget_ms = 2.5,
        max_height_layers = 1,
        rrt_max_iters = 300,
        rrt_nearest_candidates = 20,
        cache_max_size = 3000,
        link_cache_max_size = 25000,
    },
    medium = {
        grid_step = 2.0,
        map_size_local = 36,
        map_update_rate = 90,
        map_update_speed = 60,
        map_update_time_budget_ms = 3.0,
        max_height_layers = 1,
        rrt_max_iters = 500,
        rrt_nearest_candidates = 30,
        cache_max_size = 5000,
        link_cache_max_size = 50000,
    },
    high = {
        grid_step = 1.5,
        map_size_local = 48,
        map_update_rate = 60,
        map_update_speed = 120,
        map_update_time_budget_ms = 4.5,
        max_height_layers = 1,
        rrt_max_iters = 600,
        rrt_nearest_candidates = 40,
        cache_max_size = 8000,
        link_cache_max_size = 80000,
    },
}

package.loaded["navmoon.config"] = cfg

local hw = {
    cpu_name = "unknown",
    cpu_cores = 0,
    ram_mb = 0,
    gpu_name = "unknown",
    gpu_ready = false,
    profile = "medium",
    profile_source = "default",
}

pcall(function()
    ffi.cdef[[
        typedef unsigned long DWORD;
        typedef unsigned long long DWORDLONG;
        typedef void* HKEY;
        typedef long LONG;

        typedef struct _MEMORYSTATUSEX {
            DWORD dwLength;
            DWORD dwMemoryLoad;
            DWORDLONG ullTotalPhys;
            DWORDLONG ullAvailPhys;
            DWORDLONG ullTotalPageFile;
            DWORDLONG ullAvailPageFile;
            DWORDLONG ullTotalVirtual;
            DWORDLONG ullAvailVirtual;
            DWORDLONG ullAvailExtendedVirtual;
        } MEMORYSTATUSEX;

        void GetSystemInfo(void* lpSystemInfo);
        int GlobalMemoryStatusEx(MEMORYSTATUSEX* lpBuffer);

        LONG RegOpenKeyExA(HKEY hKey, const char* lpSubKey, DWORD ulOptions, DWORD samDesired, HKEY* phkResult);
        LONG RegQueryValueExA(HKEY hKey, const char* lpValueName, DWORD* lpReserved, DWORD* lpType, unsigned char* lpData, DWORD* lpcbData);
        LONG RegCloseKey(HKEY hKey);
    ]]
end)

local function detect_cpu_ram()
    pcall(function()
        local kernel32 = ffi.load("kernel32")
        local si = ffi.new("char[64]")
        kernel32.GetSystemInfo(si)

        local c1 = tonumber(ffi.cast("DWORD*", si + 20)[0]) or 0
        local c2 = tonumber(ffi.cast("DWORD*", si + 32)[0]) or 0
        if c1 >= 1 and c1 <= 256 then
            hw.cpu_cores = c1
        elseif c2 >= 1 and c2 <= 256 then
            hw.cpu_cores = c2
        else
            hw.cpu_cores = 0
        end

        local ms = ffi.new("MEMORYSTATUSEX")
        ms.dwLength = ffi.sizeof("MEMORYSTATUSEX")
        if kernel32.GlobalMemoryStatusEx(ms) ~= 0 then
            hw.ram_mb = math.floor(tonumber(ms.ullTotalPhys) / (1024 * 1024))
        end
    end)

    pcall(function()
        local advapi = ffi.load("advapi32")
        local HKEY_LOCAL_MACHINE = ffi.cast("HKEY", 0x80000002)
        local KEY_READ = 0x20019
        local hkey = ffi.new("HKEY[1]")
        local path = "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"
        if advapi.RegOpenKeyExA(HKEY_LOCAL_MACHINE, path, 0, KEY_READ, hkey) == 0 then
            local buf = ffi.new("unsigned char[512]")
            local size = ffi.new("DWORD[1]", 512)
            local typ = ffi.new("DWORD[1]")
            if advapi.RegQueryValueExA(hkey[0], "ProcessorNameString", nil, typ, buf, size) == 0 then
                hw.cpu_name = ffi.string(buf):gsub("^%s+", ""):gsub("%s+$", "")
            end
            advapi.RegCloseKey(hkey[0])
        end
    end)
end

local function score_hardware()

    local score = 0
    local cpu = (hw.cpu_name or ""):lower()
    local cores = hw.cpu_cores or 0
    local ram = hw.ram_mb or 0

    if cores >= 12 then score = score + 3
    elseif cores >= 6 then score = score + 2
    elseif cores >= 4 then score = score + 1
    end

    if ram >= 16000 then score = score + 3
    elseif ram >= 8000 then score = score + 2
    elseif ram >= 4000 then score = score + 1
    end

    if cpu:find("n100", 1, true) or cpu:find("n200", 1, true)
       or cpu:find("n95", 1, true) or cpu:find("n305", 1, true)
       or cpu:find("celeron", 1, true) or cpu:find("pentium", 1, true)
       or cpu:find("atom", 1, true) or cpu:find("a4-", 1, true)
       or cpu:find("a6-", 1, true) then
        score = score - 3
    end

    if cpu:find("i3-", 1, true) or cpu:find("ryzen 3", 1, true)
       or cpu:find("i5-7", 1, true) or cpu:find("i5-8", 1, true)
       or cpu:find("i5-9", 1, true) then
        score = score - 0
    end

    if cpu:find("i7-", 1, true) or cpu:find("i9-", 1, true)
       or cpu:find("ryzen 7", 1, true) or cpu:find("ryzen 9", 1, true)
       or cpu:find("ryzen 5", 1, true) then
        score = score + 1
    end

    local gpu = (hw.gpu_name or ""):lower()
    if gpu:find("uhd", 1, true) or gpu:find("iris", 1, true)
       or gpu:find("hd graphics", 1, true) or gpu:find("vega", 1, true) then

        score = score - 1
    end
    if gpu:find("rtx", 1, true) or gpu:find("gtx", 1, true)
       or gpu:find("radeon rx", 1, true) or gpu:find("nvidia", 1, true) then
        score = score + 2
    end

    if score <= 1 then return "low"
    elseif score <= 5 then return "medium"
    else return "high" end
end

local function apply_profile(name)
    local p = PROFILES[name]
    if not p then return end
    for k, v in pairs(p) do
        cfg[k] = v
    end
    hw.profile = name
end

local function resolve_profile()
    if cfg.force_profile and PROFILES[cfg.force_profile] then
        apply_profile(cfg.force_profile)
        hw.profile_source = "force"
        return hw.profile
    end
    if cfg.auto_profile then
        local name = score_hardware()
        apply_profile(name)
        hw.profile_source = "auto"
        return hw.profile
    end
    hw.profile = "manual"
    hw.profile_source = "manual"
    return hw.profile
end

local math_floor = math.floor
local line_of_sight_cache, cache_size = {}, 0

local function los_clear()
    line_of_sight_cache, cache_size = {}, 0
end

local function los_process(x1, y1, z1, x2, y2, z2, b, v, p, o, d, s, i, sh)
    local key = table.concat({
        math_floor(x1 * 10 + 0.5), math_floor(y1 * 10 + 0.5), math_floor(z1 * 10 + 0.5),
        math_floor(x2 * 10 + 0.5), math_floor(y2 * 10 + 0.5), math_floor(z2 * 10 + 0.5),
        b and 1 or 0, v and 1 or 0, p and 1 or 0, o and 1 or 0,
        d and 1 or 0, s and 1 or 0, i and 1 or 0, sh and 1 or 0
    }, ",")
    local c = line_of_sight_cache[key]
    if c then return c[1], c[2] end
    if cache_size > cfg.cache_cleanup_threshold or cache_size > cfg.cache_max_size then los_clear() end
    local r, col = processLineOfSight(x1, y1, z1, x2, y2, z2, b, v, p, o, d, s, i, sh)
    line_of_sight_cache[key] = {r, col}
    cache_size = cache_size + 1
    return r, col
end

package.loaded["navmoon.los"] = { clear = los_clear, process = los_process }

local Core = require("navmoon.core")
Core.config = cfg
Core.los = package.loaded["navmoon.los"]
Core.hw = hw
Core.profiles = PROFILES

local _new = Core.new
function Core.new(...)
    local nav = _new(...)
    return nav
end

function Core.set_profile(name, nav)
    if not PROFILES[name] then return false end
    cfg.force_profile = name
    apply_profile(name)
    hw.profile_source = "force"
    if nav and nav.map_side then
        local step = cfg.grid_step
        nav.map_size = 3000 - 3000 % step
        cfg.map_size_local = cfg.map_size_local - cfg.map_size_local % step
        nav.map_side = nav.map_size * 2 / step + 1
        nav.grid_range = step * cfg.map_size_local
        nav.map_center = (nav.map_side - 1) / 2
        nav.inv_grid_step = 1 / step
    end
    return true
end

local _init = Core.init
function Core:init(...)
    detect_cpu_ram()

    _init(self, ...)

    hw.gpu_ready = self.gpu_ready and true or false
    if self.gpu_info and self.gpu_info ~= "" then
        hw.gpu_name = tostring(self.gpu_info)
    elseif not hw.gpu_ready then
        hw.gpu_name = "none / OpenCL unavailable"
    end

    resolve_profile()

    local step = cfg.grid_step
    self.map_size = 3000 - 3000 % step
    cfg.map_size_local = cfg.map_size_local - cfg.map_size_local % step
    self.map_side = self.map_size * 2 / step + 1
    self.grid_range = step * cfg.map_size_local
    self.map_center = (self.map_side - 1) / 2
    self.inv_grid_step = 1 / step

    print(string.format(
        "[NavMoon] profile=%s (%s) | CPU=%s | cores=%s | RAM=%sMB | GPU=%s",
        hw.profile, hw.profile_source, hw.cpu_name, tostring(hw.cpu_cores),
        tostring(hw.ram_mb), hw.gpu_name
    ))
end

return Core
