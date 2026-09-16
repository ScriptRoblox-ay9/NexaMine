local cfg      = require("navmoon.config")
local los_cache = require("navmoon.los")

local gpu
do
    local ffi = require("ffi")

    local M = {
        ready       = false,
        info        = "",
        last_error  = nil,
    }

    local mooncl = nil
    local kernel_dist = nil
    local kernel_priority = nil

    local KERNEL_DIST_SRC = [[
    __kernel void batch_dist(
        __global const float4* a,   // positions A
        __global const float4* b,   // positions B
        __global float* out,        // distances
        const int n)
    {
        int i = get_global_id(0);
        if (i >= n) return;
        float4 d = a[i] - b[i];
        out[i] = sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
    }
    ]]

    local KERNEL_PRIORITY_SRC = [[
    // Score nodes for scan priority.
    // Higher score = more important (closer + in front of player movement).
    __kernel void scan_priority(
        __global const float4* nodes,   // xyz = node pos, w unused
        __global float* scores,
        const float4 player,            // xyz + heading (radians) in w
        const float prefer_forward,
        const int n)
    {
        int i = get_global_id(0);
        if (i >= n) return;

        float4 p = nodes[i];
        float dx = p.x - player.x;
        float dy = p.y - player.y;
        float dist = sqrt(dx*dx + dy*dy) + 0.001f;

        // forward factor
        float heading = player.w;
        float fx = sin(heading);
        float fy = cos(heading);
        float dot = (dx * fx + dy * fy) / dist;   // -1 .. 1

        float score = 1.0f / dist;
        if (prefer_forward > 0.5f) {
            score *= (1.0f + 0.85f * fmax(dot, 0.0f));
        }
        scores[i] = score;
    }
    ]]

    local function try_load_mooncl()

        local candidates = {
            "navmoon.mooncl",
            "navmoon.mooncl.init",
            "mooncl",
            "lib.mooncl",
        }
        local last_err
        for i = 1, #candidates do
            local ok, mod = pcall(require, candidates[i])
            if ok and mod then
                return mod
            end
            last_err = mod
        end
        return nil, "mooncl not found (expected inside navmoon/mooncl/). Last error: " .. tostring(last_err)
    end

    function M.init()
        M.ready = false
        M.info = "CPU"
        M.last_error = nil
        M.device_name = nil

        if not cfg.gpu_enabled then
            M.info = "GPU disabled in config"
            return false, M.info
        end

        local ok, err = pcall(function()
            local mod, load_err = try_load_mooncl()
            if not mod then
                error(load_err or "mooncl require failed")
            end
            mooncl = mod

            local inited, info = mooncl.init()
            if not inited then
                error(tostring(info or "mooncl.init failed"))
            end
            M.info = tostring(info or "OpenCL ready")

            M.device_name = M.info

            local k, log = mooncl.compile(KERNEL_DIST_SRC, "batch_dist")
            if k then
                kernel_dist = k
            else
                M.last_error = "batch_dist: " .. tostring(log)
            end

            k, log = mooncl.compile(KERNEL_PRIORITY_SRC, "scan_priority")
            if k then
                kernel_priority = k
            else
                M.last_error = (M.last_error or "") .. " scan_priority: " .. tostring(log)
            end

            if mooncl.is_ready and mooncl.is_ready() then
                M.ready = true
            elseif kernel_dist or kernel_priority then
                M.ready = true
            else

                M.ready = true
            end
        end)

        if not ok then
            M.ready = false
            M.last_error = tostring(err)
            M.info = "CPU only (" .. tostring(err) .. ")"

            pcall(function()
                if kernel_dist then kernel_dist:free() end
                if kernel_priority then kernel_priority:free() end
                if mooncl and mooncl.cleanup then mooncl.cleanup() end
            end)
            kernel_dist, kernel_priority, mooncl = nil, nil, nil
            return false, M.info
        end

        return true, M.info
    end

    function M.is_ready()
        return M.ready and mooncl and mooncl.is_ready()
    end

    function M.cleanup()
        if kernel_dist then kernel_dist:free() kernel_dist = nil end
        if kernel_priority then kernel_priority:free() kernel_priority = nil end
        if mooncl then
            pcall(mooncl.cleanup)
        end
        M.ready = false
    end

    function M.batch_distances(a_pos, b_pos)
        if not M.is_ready() or not kernel_dist or not cfg.gpu_use_for_distances then
            return nil
        end

        local n = #a_pos
        if n == 0 or n ~= #b_pos then return nil end

        local ok, result = pcall(function()
            local a = mooncl.new_float4_array(n)
            local b = mooncl.new_float4_array(n)
            for i = 1, n do
                local p = a_pos[i]
                a[i-1] = ffi.new("mooncl_float4", p[1], p[2], p[3], 0)
                p = b_pos[i]
                b[i-1] = ffi.new("mooncl_float4", p[1], p[2], p[3], 0)
            end

            local buf_a = mooncl.create_buffer(n * 16)
            local buf_b = mooncl.create_buffer(n * 16)
            local buf_o = mooncl.create_buffer(n * 4)

            buf_a:write(a, n * 16)
            buf_b:write(b, n * 16)

            kernel_dist:set_buffer(0, buf_a)
            kernel_dist:set_buffer(1, buf_b)
            kernel_dist:set_buffer(2, buf_o)
            kernel_dist:set_int(3, n)

            if not kernel_dist:run(n) then
                error("kernel run failed")
            end

            local out = ffi.new("float[?]", n)
            buf_o:read(out, n * 4)

            local distances = {}
            for i = 1, n do
                distances[i] = out[i-1]
            end

            buf_a:free(); buf_b:free(); buf_o:free()
            return distances
        end)

        if ok then return result end
        M.last_error = tostring(result)
        return nil
    end

    function M.scan_priorities(nodes, player)
        if not M.is_ready() or not kernel_priority or not cfg.gpu_use_for_priority then
            return nil
        end

        local n = #nodes
        if n == 0 then return nil end

        local ok, result = pcall(function()
            local arr = mooncl.new_float4_array(n)
            for i = 1, n do
                local p = nodes[i]
                arr[i-1] = ffi.new("mooncl_float4", p[1], p[2], p[3], 0)
            end

            local buf_n = mooncl.create_buffer(n * 16)
            local buf_s = mooncl.create_buffer(n * 4)
            buf_n:write(arr, n * 16)

            kernel_priority:set_buffer(0, buf_n)
            kernel_priority:set_buffer(1, buf_s)
            kernel_priority:set_float4(2, player[1], player[2], player[3], player[4] or 0)
            kernel_priority:set_float(3, cfg.prefer_forward and 1.0 or 0.0)
            kernel_priority:set_int(4, n)

            if not kernel_priority:run(n) then
                error("priority kernel failed")
            end

            local out = ffi.new("float[?]", n)
            buf_s:read(out, n * 4)

            local scores = {}
            for i = 1, n do
                scores[i] = out[i-1]
            end

            buf_n:free(); buf_s:free()
            return scores
        end)

        if ok then return result end
        M.last_error = tostring(result)
        return nil
    end

    function M.batch_distances_cpu(a_pos, b_pos)
        local n = #a_pos
        local d = {}
        for i = 1, n do
            local ax, ay, az = a_pos[i][1], a_pos[i][2], a_pos[i][3]
            local bx, by, bz = b_pos[i][1], b_pos[i][2], b_pos[i][3]
            local dx, dy, dz = ax-bx, ay-by, az-bz
            d[i] = math.sqrt(dx*dx + dy*dy + dz*dz)
        end
        return d
    end
    gpu = M
end

local math_floor = math.floor
local math_abs   = math.abs
local math_sqrt  = math.sqrt
local math_randomseed = math.randomseed
local math_random = math.random
local pairs = pairs
local os_time = os.time
local os_clock = os.clock

local MAP_SIZE = 3000

local NavMoon = {}
NavMoon.__index = NavMoon

NavMoon.NODE_TIMESTAMP = 1
NavMoon.NODE_POINTS    = 2
NavMoon.NODE_NEIGHBORS = 3
NavMoon.NAV_NEI_OFFSETS_X = {0, 1, 1, 1, 0, -1, -1, -1}
NavMoon.NAV_NEI_OFFSETS_Y = {1, 1, 0, -1, -1, -1, 0, 1}
NavMoon.NAV_NEI_COUNT = 8

function NavMoon.new()
    local obj = setmetatable({}, NavMoon)
    obj.mesh = {}
    obj.map_size = MAP_SIZE
    obj.map_side = 0
    obj.last_points_n = {}
    obj.last_check = -1
    obj.last_node = 0
    obj.last_points_local = {}
    obj.last_edges = {}
    obj.last_id_to_pos = {}
    obj.grid_range = 0
    obj.map_center = 0
    obj.inv_grid_step = 0
    obj.scan_queue = {}
    obj.scan_queue_idx = 1
    obj.stats = {scan_calls = 0, scan_time = 0, update_calls = 0, update_time = 0, gpu_hits = 0, gpu_misses = 0}
    obj.gpu_ready = false
    obj.player_heading = 0
    obj.last_point_query = {id = nil, x = 0, y = 0, z = 0}
    obj.link_cache = {}
    obj.link_cache_size = 0
    obj.link_cache_hits = 0
    obj.link_cache_misses = 0
    obj.next_point_id = 1
    obj.pending_dirty = {}
    return obj
end

function NavMoon:init()
    local step = cfg.grid_step
    math_randomseed(os_time())
    self.map_size = MAP_SIZE - MAP_SIZE % step
    cfg.map_size_local = cfg.map_size_local - cfg.map_size_local % step
    self.map_side = self.map_size * 2 / step + 1
    self.grid_range = step * cfg.map_size_local
    self.map_center = (self.map_side - 1) / 2
    self.inv_grid_step = 1 / step

    self.gpu_ready = false
    self.player_heading = 0
    self.gpu_info = "CPU"

    if cfg.gpu_enabled then
        local call_ok, a, b = pcall(function()
            return gpu.init()
        end)
        if call_ok then
            self.gpu_ready = a and true or false
            self.gpu_info = tostring(b or a or "")
        else
            self.gpu_ready = false
            self.gpu_info = tostring(a)
        end
        if self.gpu_ready then
            print("[NavMoon] GPU: " .. self.gpu_info)
        else
            print("[NavMoon] CPU mode: " .. self.gpu_info)
        end
    else
        print("[NavMoon] CPU mode (gpu_enabled=false)")
    end
end

function NavMoon:shutdown()
    gpu.cleanup()
end

function NavMoon:get_node_offsets_from_coords(x, y)
    local step = cfg.grid_step
    local half = step * 0.5
    local offset_x = x + half
    local offset_y = y + half
    offset_x = offset_x - (offset_x % step)
    offset_y = offset_y - (offset_y % step)
    return offset_x, offset_y
end

function NavMoon:get_node_from_offsets(x, y)
    local inv = self.inv_grid_step
    local center = self.map_center
    local idx_x = math_floor(center + x * inv)
    local idx_y = math_floor(center + y * inv)
    return idx_x * self.map_side + idx_y + 1
end

function NavMoon:get_coords_from_index(index, width)
    local idx = index - 1
    return idx % width + 1, math_floor(idx / width) + 1
end

function NavMoon:get_coords_from_node(node)
    local x, y = self:get_coords_from_index(node, self.map_side)
    local step = cfg.grid_step
    return -(self.map_size - y * step) - step, -(self.map_size - x * step) - step
end

function NavMoon:get_neighbor_at(index, width, offset_idx)
    local idx = index - 1
    local x = idx % width + 1
    local y = math_floor(idx / width) + 1
    local nx = x + NavMoon.NAV_NEI_OFFSETS_X[offset_idx]
    local ny = y + NavMoon.NAV_NEI_OFFSETS_Y[offset_idx]
    return (ny - 1) * width + nx
end

function NavMoon:clear_neighbor_links(idx, map_side)
    local mesh = self.mesh
    for i = 1, NavMoon.NAV_NEI_COUNT do
        local idx_nei = self:get_neighbor_at(idx, map_side, i)
        local neighbor_node = mesh[idx_nei]
        if neighbor_node then neighbor_node[NavMoon.NODE_NEIGHBORS][idx] = nil end
    end
end

function NavMoon:touch_grid(mx, my, ts)
    local points_local = {}
    local dirty = {}
    local mesh = self.mesh
    local map_side = self.map_side
    local step = cfg.grid_step
    local range = self.grid_range
    local local_size = cfg.map_size_local
    local update_rate = cfg.map_update_rate
    local local_size_sq = local_size * local_size
    local heading = self.player_heading or 0
    local fx = math.sin(heading)
    local fy = math.cos(heading)
    local behind_mult = cfg.behind_discard_dist or 1.35

    for step_x = -range, range, step do
        local dx_sq = step_x * step_x
        for step_y = -range, range, step do
            local dist_sq = dx_sq + step_y * step_y
            local in_range = dist_sq < local_size_sq

            if cfg.sliding_window and in_range then
                local dist = math_sqrt(dist_sq)
                if dist > 0.01 then
                    local dot = (step_x * fx + step_y * fy) / dist
                    if dot < -0.25 and dist > local_size * behind_mult then
                        in_range = false
                    end
                end
            end

            if in_range then
                local idx = self:get_node_from_offsets(mx + step_x, my + step_y)
                points_local[idx] = true
                local node = mesh[idx]
                local needs_reset = not node or ts - node[NavMoon.NODE_TIMESTAMP] > update_rate
                if needs_reset then
                    mesh[idx] = {
                        [NavMoon.NODE_TIMESTAMP] = 0,
                        [NavMoon.NODE_POINTS] = {},
                        [NavMoon.NODE_NEIGHBORS] = {}
                    }

                    self:clear_neighbor_links(idx, map_side)
                    dirty[idx] = true
                end
            end
        end
    end
    return points_local, dirty
end

function NavMoon:build_scan_queue(points_local, px, py)
    local items = {}
    local n = 0
    local mesh = self.mesh
    for idx in pairs(points_local) do
        local node = mesh[idx]

        if node and node[NavMoon.NODE_TIMESTAMP] == 0 then
            local x, y = self:get_coords_from_node(idx)
            n = n + 1
            items[n] = {idx = idx, x = x, y = y}
        end
    end

    if n == 0 then
        self.scan_queue = {}
        self.scan_queue_idx = 1
        return
    end

    local scores = nil
    if self.gpu_ready and cfg.gpu_use_for_priority then
        local nodes = {}
        for i = 1, n do
            nodes[i] = {items[i].x, items[i].y, 0}
        end
        local ok, result = pcall(gpu.scan_priorities, nodes, {px, py, 0, self.player_heading or 0})
        if ok and result then
            scores = result
            self.stats.gpu_hits = (self.stats.gpu_hits or 0) + 1
        else
            self.stats.gpu_misses = (self.stats.gpu_misses or 0) + 1

        end
    end

    if scores then
        for i = 1, n do
            items[i].score = scores[i] or 0
        end
    else
        local heading = self.player_heading or 0
        local fx = math.sin(heading)
        local fy = math.cos(heading)
        for i = 1, n do
            local dx = items[i].x - px
            local dy = items[i].y - py
            local dist = math_sqrt(dx * dx + dy * dy) + 0.001
            local dot = (dx * fx + dy * fy) / dist
            local score = 1.0 / dist
            if cfg.prefer_forward then
                score = score * (1.0 + 0.85 * math.max(dot, 0))
            end
            items[i].score = score
        end
    end

    table.sort(items, function(a, b) return a.score > b.score end)
    local queue = {}
    for i = 1, n do
        queue[i] = items[i].idx
    end
    self.scan_queue = queue
    self.scan_queue_idx = 1
end

function NavMoon:check_vertical_clearance(x, y, z)
    local top_z = z + cfg.headroom
    local hit = los_cache.process(x, y, z + 0.2, x, y, top_z, cfg.los_check_buildings, cfg.los_check_vehicles, cfg.los_check_peds, cfg.los_check_objects, cfg.los_check_dummies, cfg.los_see_through, cfg.los_ignore_camera, cfg.los_shoot_through)
    return not hit
end

function NavMoon:get_ground_height(x, y)
    local start_z = cfg.scan_height_start
    local end_z = start_z - cfg.scan_height_range
    local hit, col_point = los_cache.process(x, y, start_z, x, y, end_z, true, false, false, true, false, false, false, false)
    if hit and col_point and col_point.pos then return col_point.pos[3] end
    return nil
end

function NavMoon:get_ground_height_near(x, y, base_z, range)
    local start_z = base_z + range
    local end_z = base_z - range
    local hit_down, col_down = los_cache.process(x, y, start_z, x, y, end_z, true, false, false, true, false, false, false, false)
    local hit_up, col_up = los_cache.process(x, y, end_z, x, y, start_z, true, false, false, true, false, false, false, false)
    local z_down = hit_down and col_down and col_down.pos and col_down.pos[3] or nil
    local z_up = hit_up and col_up and col_up.pos and col_up.pos[3] or nil
    if z_down and z_up then
        if math_abs(z_down - base_z) <= math_abs(z_up - base_z) then
            return z_down
        end
        return z_up
    end
    return z_down or z_up
end

function NavMoon:is_too_steep(x, y, base_z)
    local r = cfg.slope_sample_radius
    local max_slope = cfg.max_slope
    local range = cfg.slope_sample_range
    local min_ok = cfg.slope_min_ok_neighbors
    local avg_factor = cfg.slope_avg_factor
    local max_factor = cfg.slope_max_factor
    local step_min_ok = cfg.step_min_ok_neighbors
    local max_step = cfg.max_step_height
    local offsets = {
        {r, 0},
        {-r, 0},
        {0, r},
        {0, -r}
    }
    local step_offsets = {
        {r, 0},
        {-r, 0},
        {0, r},
        {0, -r},
        {r, r},
        {r, -r},
        {-r, r},
        {-r, -r}
    }
    local ok = 0
    local sum_diff = 0
    local max_diff = 0
    for i = 1, #offsets do
        local ox, oy = offsets[i][1], offsets[i][2]
        local hz = self:get_ground_height_near(x + ox, y + oy, base_z, range)
        if not hz then return true end
        local diff = math_abs(hz - base_z)
        if diff <= max_slope then ok = ok + 1 end
        sum_diff = sum_diff + diff
        if diff > max_diff then max_diff = diff end
    end
    local ok_step = 0
    for i = 1, #step_offsets do
        local ox, oy = step_offsets[i][1], step_offsets[i][2]
        local hz = self:get_ground_height_near(x + ox, y + oy, base_z, range)
        if not hz then return true end
        local diff = math_abs(hz - base_z)
        if diff <= max_step then ok_step = ok_step + 1 end
    end
    local avg_diff = sum_diff / #offsets
    if ok_step < step_min_ok then return true end
    if ok < min_ok then return true end
    if ok == 1 and avg_diff > max_slope * avg_factor then return true end
    if max_diff > max_slope * max_factor and ok <= 1 then return true end
    return false
end

function NavMoon:can_connect_points(x1, y1, z1, x2, y2, z2, want_reason)
    local dz = math_abs(z2 - z1)
    if dz > cfg.max_step_height then
        local dxs, dys = x2 - x1, y2 - y1
        local len2s = dxs * dxs + dys * dys
        if len2s > 0.01 then
            local len = math_sqrt(len2s)
            local step_len = cfg.grid_step * 0.5
            local steps = math_floor(len / step_len)
            if steps > 1 then
                local ok_slope = true
                local prev_z = z1
                for i = 1, steps do
                    local t = i / steps
                    local sx = x1 + dxs * t
                    local sy = y1 + dys * t
                    local sz = z1 + (z2 - z1) * t
                    if math_abs(sz - prev_z) > cfg.max_step_height then
                        ok_slope = false
                        break
                    end

                    if not self:check_vertical_clearance(sx, sy, sz) then
                        ok_slope = false
                        break
                    end

                    prev_z = sz
                end

                if not ok_slope then return false, "step_height" end
            else
                return false, "step_height"
            end
        else
            return false, "step_height"
        end
    end

    if not self:check_vertical_clearance(x1, y1, z1) then return false, "headroom_start" end
    if not self:check_vertical_clearance(x2, y2, z2) then return false, "headroom_end" end
    local dx, dy = x2 - x1, y2 - y1
    local len2 = dx * dx + dy * dy
    if len2 < 0.0001 then return true, "ok" end
    local inv_len = 1 / math_sqrt(len2)
    local off = cfg.los_side_offset
    local ox, oy = -dy * inv_len * off, dx * inv_len * off
    local heights = cfg.los_sample_heights
    local flags = {cfg.los_check_buildings, cfg.los_check_vehicles, cfg.los_check_peds, cfg.los_check_objects, cfg.los_check_dummies, cfg.los_see_through, cfg.los_ignore_camera, cfg.los_shoot_through}
    local function ray(tag, sx, sy, sz, ex, ey, ez)
        local hit, col = los_cache.process(sx, sy, sz, ex, ey, ez, flags[1], flags[2], flags[3], flags[4], flags[5], flags[6], flags[7], flags[8])
        if hit then
            if want_reason and col and col.pos then return false, tag .. "@" .. string.format("%.2f %.2f %.2f", col.pos[1], col.pos[2], col.pos[3]) end
            return false, tag
        end
        return true, "ok"
    end

    for i = 1, #heights do
        local h = heights[i]
        local ok, reason = ray("center_h" .. i, x1, y1, z1 + h, x2, y2, z2 + h)
        if not ok then return false, reason end
    end

    local mid_h = heights[2] or 0.9
    local ok_left, reason_left = ray("side_left", x1 + ox, y1 + oy, z1 + mid_h, x2 + ox, y2 + oy, z2 + mid_h)
    if not ok_left then return false, reason_left end
    local ok_right, reason_right = ray("side_right", x1 - ox, y1 - oy, z1 + mid_h, x2 - ox, y2 - oy, z2 + mid_h)
    if not ok_right then return false, reason_right end
    return true, "ok"
end

function NavMoon:scan_node(idx, ts)
    local node = self.mesh[idx]
    if node[NavMoon.NODE_TIMESTAMP] ~= 0 then return false end
    node[NavMoon.NODE_TIMESTAMP] = ts
    local x1, y1 = self:get_coords_from_node(idx)
    local points = node[NavMoon.NODE_POINTS]
    local max_layers = cfg.max_height_layers or 1
    local min_gap = cfg.layer_min_gap
    local start_z = cfg.scan_height_start
    local scan_range = cfg.scan_height_range
    local current_z = start_z
    local end_z = start_z - scan_range

    local raw = {}
    local raw_n = 0
    local raw_limit = math.max(max_layers, 8)
    while current_z > end_z and raw_n < raw_limit do
        local result, col_point = los_cache.process(
            x1, y1, current_z,
            x1, y1, current_z - 100.0,
            true, false, false, true, false, false, false, false
        )
        if result and col_point and col_point.pos then
            local hit_z = col_point.pos[3]
            local is_new = true
            for i = 1, raw_n do
                if math_abs(raw[i] - hit_z) < min_gap then
                    is_new = false
                    break
                end
            end
            if is_new then
                local candidate_z = hit_z + 0.5
                if not self:is_too_steep(x1, y1, hit_z) and self:check_vertical_clearance(x1, y1, candidate_z) then
                    raw_n = raw_n + 1
                    raw[raw_n] = candidate_z
                end
            end
            current_z = hit_z - min_gap
        else
            current_z = current_z - 50.0
        end
    end

    if raw_n == 0 then return true end

    for i = 1, raw_n - 1 do
        for j = i + 1, raw_n do
            if raw[i] < raw[j] then raw[i], raw[j] = raw[j], raw[i] end
        end
    end

    local layer_count = 0
    local start_i = raw_n - max_layers + 1
    if start_i < 1 then start_i = 1 end
    for i = start_i, raw_n do
        layer_count = layer_count + 1
        points[layer_count] = raw[i]
    end

    for i = 1, layer_count - 1 do
        for j = i + 1, layer_count do
            if points[i] < points[j] then points[i], points[j] = points[j], points[i] end
        end
    end

    local map_side = self.map_side
    local mesh = self.mesh
    local z_tolerance = cfg.neighbor_z_tolerance
    for i = 1, NavMoon.NAV_NEI_COUNT do
        local idx_nei = self:get_neighbor_at(idx, map_side, i)
        local neighbor_node = mesh[idx_nei]
        if neighbor_node and neighbor_node[NavMoon.NODE_TIMESTAMP] ~= 0 and not neighbor_node[NavMoon.NODE_NEIGHBORS][idx] then
            neighbor_node[NavMoon.NODE_NEIGHBORS][idx] = {}
            node[NavMoon.NODE_NEIGHBORS][idx_nei] = {}
            local x2, y2 = self:get_coords_from_node(idx_nei)
            local neighbor_points = neighbor_node[NavMoon.NODE_POINTS]
            local neighbor_count = #neighbor_points
            for k1 = 1, layer_count do
                local z1 = points[k1]
                for k2 = 1, neighbor_count do
                    local z2 = neighbor_points[k2]
                    local dz = math_abs(z2 - z1)
                    local can_connect = dz < z_tolerance and self:can_connect_points(x1, y1, z1, x2, y2, z2, false)
                    if can_connect then
                        local dx, dy = x2 - x1, y2 - y1
                        local dist = math_sqrt(dx * dx + dy * dy + dz * dz)
                        local conn1 = node[NavMoon.NODE_NEIGHBORS][idx_nei]
                        local conn2 = neighbor_node[NavMoon.NODE_NEIGHBORS][idx]
                        conn1[#conn1 + 1] = {k1, k2, dist}
                        conn2[#conn2 + 1] = {k2, k1, dist}
                    end
                end
            end
        end
    end
    return true
end

local function add_edge(edges, pn_idx, pn_nei, data)
    local k1, k2 = pn_idx[data[1]], pn_nei[data[2]]
    if k1 and k2 then edges[k1][k2] = data[3] end
end

function NavMoon:rebuild_points_and_edges(points_local, dirty)
    local mesh = self.mesh
    local id_to_pos = self.last_id_to_pos
    local points_n = self.last_points_n
    local edges = self.last_edges
    local next_id = self.next_point_id or 1

    if dirty then
        for idx in pairs(dirty) do
            local pn = points_n[idx]
            if pn then
                for k = 1, #pn do
                    local pid = pn[k]
                    if pid then
                        if edges[pid] then
                            for other in pairs(edges[pid]) do
                                if edges[other] then edges[other][pid] = nil end
                            end
                            edges[pid] = nil
                        end
                        id_to_pos[pid] = nil
                    end
                end
                points_n[idx] = nil
            end
        end
    end

    local rebuild_idxs = dirty or {}
    if not dirty then
        rebuild_idxs = points_local
        points_n = {}
        id_to_pos = {}
        edges = {}
        next_id = 1
    end

    for idx in pairs(rebuild_idxs) do
        if points_local[idx] then
            local node = mesh[idx]
            if node then
                local node_points = node[NavMoon.NODE_POINTS]
                local count = #node_points
                if count > 0 then
                    points_n[idx] = {}
                    local x, y = self:get_coords_from_node(idx)
                    local pn = points_n[idx]
                    for k = 1, count do
                        local pid = next_id
                        next_id = next_id + 1
                        edges[pid] = {}
                        pn[k] = pid
                        id_to_pos[pid] = {x, y, node_points[k]}
                    end
                end
            end
        end
    end

    local edge_src = rebuild_idxs
    for idx in pairs(edge_src) do
        local pn_idx = points_n[idx]
        if pn_idx then
            local neighbors = mesh[idx] and mesh[idx][NavMoon.NODE_NEIGHBORS]
            if neighbors then
                for idx_nei, neighbor_data in pairs(neighbors) do
                    local pn_nei = points_n[idx_nei]
                    if pn_nei then
                        for i = 1, #neighbor_data do
                            local data = neighbor_data[i]
                            add_edge(edges, pn_idx, pn_nei, data)
                            add_edge(edges, pn_nei, pn_idx, {data[2], data[1], data[3]})
                        end
                    end
                end
            end
        end
    end

    self.last_id_to_pos = id_to_pos
    self.last_points_n = points_n
    self.last_edges = edges
    self.next_point_id = next_id
end

function NavMoon:update_mesh()
    local update_start = os_clock()
    local px, py, pz = getCharCoordinates(PLAYER_PED)
    local heading = getCharHeading(PLAYER_PED)
    if heading then self.player_heading = math.rad(heading) end
    local mx, my = self:get_node_offsets_from_coords(px, py)
    local my_node = self:get_node_from_offsets(mx, my)
    local ts = os_time()
    local clock = os_clock()
    local grid_changed = false
    local points_local
    local dirty = self.pending_dirty or {}
    self.pending_dirty = dirty

    if my_node ~= self.last_node or clock - self.last_check > 1.0 then
        local touch_dirty
        points_local, touch_dirty = self:touch_grid(mx, my, ts)
        self.last_points_local = points_local
        self.last_check = clock
        grid_changed = true
        for idx in pairs(touch_dirty) do
            dirty[idx] = true
        end
        for idx in pairs(self.last_points_n or {}) do
            if not points_local[idx] then
                dirty[idx] = true
                grid_changed = true
            end
        end
        self:build_scan_queue(points_local, px, py)
    else
        points_local = self.last_points_local
    end

    self.last_node = my_node
    local points_count = 0
    local max_speed = cfg.map_update_speed
    local queue = self.scan_queue
    local qidx = self.scan_queue_idx
    local budget_ms = cfg.map_update_time_budget_ms or 0
    while points_count < max_speed and qidx <= #queue do
        if budget_ms > 0 and points_count > 0 then
            local elapsed_ms = (os_clock() - update_start) * 1000
            if elapsed_ms >= budget_ms then break end
        end
        local idx = queue[qidx]
        qidx = qidx + 1
        local scan_start = os_clock()
        if self:scan_node(idx, ts) then
            grid_changed = true
            dirty[idx] = true
            points_count = points_count + 1
        end
        local scan_elapsed = os_clock() - scan_start
        self.stats.scan_calls = self.stats.scan_calls + 1
        self.stats.scan_time = self.stats.scan_time + scan_elapsed
    end

    self.scan_queue_idx = qidx
    if grid_changed then
        self:rebuild_points_and_edges(points_local, dirty)
        self.pending_dirty = {}
    end
    self.stats.update_calls = self.stats.update_calls + 1
    self.stats.update_time = self.stats.update_time + (os_clock() - update_start)
    return my_node
end

function NavMoon:get_node_k_from_coords(x, y, z)
    local o_x, o_y = self:get_node_offsets_from_coords(x, y)
    local node = self:get_node_from_offsets(o_x, o_y)
    local mesh_node = self.mesh[node]
    if not (self.last_points_local[node] and mesh_node and mesh_node[NavMoon.NODE_TIMESTAMP] ~= 0) then return nil, nil end
    local points = mesh_node[NavMoon.NODE_POINTS]
    local count = #points
    if count == 0 then return nil, nil end
    local best_k = 1
    local best_diff = math_abs(z - points[1])
    for k = 2, count do
        local diff = math_abs(z - points[k])
        if diff < best_diff then
            best_diff = diff
            best_k = k
        end
    end

    if best_diff < 5.0 then return node, best_k end
    return nil, nil
end

function NavMoon:get_point_id_from_coords(x, y, z)
    local node, k = self:get_node_k_from_coords(x, y, z)
    if node and k then
        local pn = self.last_points_n[node]
        if pn and pn[k] then
            return pn[k]
        end
    end

    local id_to_pos = self.last_id_to_pos
    local cache = self.last_point_query
    if cache and cache.id then
        local cached_pos = id_to_pos[cache.id]
        if cached_pos then
            local dx, dy, dz = cached_pos[1] - x, cached_pos[2] - y, cached_pos[3] - z
            local d2 = dx * dx + dy * dy + dz * dz
            local r = cfg.point_query_cache_radius or 0
            if r > 0 and d2 <= r * r then return cache.id end
        end
    end

    local best_id = nil
    local best_d2 = nil
    for id, pos in pairs(id_to_pos) do
        local dx, dy, dz = pos[1] - x, pos[2] - y, pos[3] - z
        local d2 = dx * dx + dy * dy + dz * dz
        if not best_d2 or d2 < best_d2 then
            best_d2 = d2
            best_id = id
        end
    end

    if not best_id then return nil end
    local max_dist = cfg.point_snap_dist or (cfg.grid_step * 8)
    if best_d2 and best_d2 > max_dist * max_dist then return nil end
    self.last_point_query = {id = best_id, x = x, y = y, z = z}
    return best_id
end

function NavMoon:get_stats()
    return {
        scan_calls = self.stats.scan_calls,
        scan_time = self.stats.scan_time,
        update_calls = self.stats.update_calls,
        update_time = self.stats.update_time,
        link_cache_size = self.link_cache_size,
        link_cache_hits = self.link_cache_hits,
        link_cache_misses = self.link_cache_misses,
        gpu_ready = self.gpu_ready,
        gpu_hits = self.stats.gpu_hits or 0,
        gpu_misses = self.stats.gpu_misses or 0
    }
end

function NavMoon:reset_stats()
    self.stats.scan_calls = 0
    self.stats.scan_time = 0
    self.stats.update_calls = 0
    self.stats.update_time = 0
    self.link_cache_hits = 0
    self.link_cache_misses = 0
    self.stats.gpu_hits = 0
    self.stats.gpu_misses = 0
end

local function make_link_key(a, b)
    if a < b then
        return a .. ":" .. b
    end
    return b .. ":" .. a
end

function NavMoon:can_connect_ids(a, b)
    local key = make_link_key(a, b)
    local cached = self.link_cache[key]
    if cached ~= nil then
        self.link_cache_hits = self.link_cache_hits + 1
        return cached
    end

    if self.link_cache_size > cfg.link_cache_cleanup_threshold or self.link_cache_size > cfg.link_cache_max_size then
        self.link_cache = {}
        self.link_cache_size = 0
    end

    local id_to_pos = self.last_id_to_pos
    local pa, pb = id_to_pos[a], id_to_pos[b]
    if not pa or not pb then return false end
    local ok = self:can_connect_points(pa[1], pa[2], pa[3], pb[1], pb[2], pb[3], false)
    self.link_cache[key] = ok
    self.link_cache_size = self.link_cache_size + 1
    self.link_cache_misses = self.link_cache_misses + 1
    return ok
end

function NavMoon:collect_sample_ids()
    local ids = {}
    local n = 0
    for id in pairs(self.last_id_to_pos) do
        n = n + 1
        ids[n] = id
    end
    return ids, n
end

local function create_min_heap()
    local heap = {}
    local size = 0
    local function sift_up(idx)
        local item = heap[idx]
        local f = item.f
        while idx > 1 do
            local parent_idx = math_floor(idx * 0.5)
            local parent = heap[parent_idx]
            if f >= parent.f then break end
            heap[idx] = parent
            idx = parent_idx
        end

        heap[idx] = item
    end

    local function sift_down(idx)
        local item = heap[idx]
        local f = item.f
        local half = math_floor(size * 0.5)
        while idx <= half do
            local left = idx * 2
            local right = left + 1
            local smallest = left
            local smallest_f = heap[left].f
            if right <= size and heap[right].f < smallest_f then
                smallest = right
                smallest_f = heap[right].f
            end

            if f <= smallest_f then break end
            heap[idx] = heap[smallest]
            idx = smallest
        end

        heap[idx] = item
    end
    return {
        push = function(node)
            size = size + 1
            heap[size] = node
            sift_up(size)
        end,
        pop = function()
            if size == 0 then return nil end
            local root = heap[1]
            heap[1] = heap[size]
            heap[size] = nil
            size = size - 1
            if size > 0 then sift_down(1) end
            return root
        end,
        size = function() return size end
    }
end

function NavMoon:a_star(start, goal, edges)
    local id_to_pos = self.last_id_to_pos
    local goal_pos = id_to_pos[goal]
    if not goal_pos then return false, {} end
    local gx, gy, gz = goal_pos[1], goal_pos[2], goal_pos[3]
    local weight = cfg.a_star_weight or 1.0
    local function heuristic(a)
        local pos = id_to_pos[a]
        local dx, dy, dz = pos[1] - gx, pos[2] - gy, pos[3] - gz
        local wz = cfg.height_heuristic_weight
        return math_sqrt(dx * dx + dy * dy + (dz * wz) * (dz * wz))
    end

    local open_set = create_min_heap()
    local came_from = {}
    local g_score = {}
    local closed = {}
    g_score[start] = 0
    open_set.push({
        node = start,
        f = heuristic(start) * weight
    })

    while open_set.size() > 0 do
        local current = open_set.pop().node
        if not closed[current] then
            if current == goal then return self:reconstruct_path(came_from, current, g_score[goal]) end
            closed[current] = true
            self:expand_node(current, goal, edges, g_score, came_from, closed, open_set, heuristic)
        end
    end
    return false, {}
end

function NavMoon:reconstruct_path(came_from, current, total_dist)
    local path = {current}
    local len = 1
    while came_from[current] do
        current = came_from[current]
        len = len + 1
        path[len] = current
    end

    local reversed = {}
    for i = len, 1, -1 do
        reversed[len - i + 1] = path[i]
    end
    return true, {
        full_distance = total_dist,
        path = reversed
    }
end

function NavMoon:expand_node(current, goal, edges, g_score, came_from, closed, open_set, heuristic)
    local node_edges = edges[current]
    if not node_edges then return end
    local current_g = g_score[current]
    local weight = cfg.a_star_weight or 1.25
    for neighbor, dist in pairs(node_edges) do
        local dominated = closed[neighbor] or (g_score[neighbor] and g_score[neighbor] <= current_g + dist)
        if not dominated then
            came_from[neighbor] = current
            g_score[neighbor] = current_g + dist
            open_set.push({
                node = neighbor,
                f = g_score[neighbor] + heuristic(neighbor) * weight
            })
        end
    end
end

------------------------------------------------------------------------------
-- Path post-process: string pulling + optional curve smooth
------------------------------------------------------------------------------

-- Greedy string-pull: keep furthest point still visible from last kept waypoint.
-- Removes grid staircase / redundant RRT samples where LOS allows a shortcut.
function NavMoon:string_pull_path(path)
    if not path or #path < 3 then return path end
    local out = {{path[1][1], path[1][2], path[1][3]}}
    local n = #path
    local i = 1
    while i < n do
        local best = i + 1
        local ax, ay, az = path[i][1], path[i][2], path[i][3]
        for j = i + 2, n do
            local b = path[j]
            if self:can_connect_points(ax, ay, az, b[1], b[2], b[3], false) then
                best = j
            end
        end
        local p = path[best]
        out[#out + 1] = {p[1], p[2], p[3]}
        i = best
    end
    return out
end

local function dist3(a, b)
    local dx, dy, dz = a[1] - b[1], a[2] - b[2], a[3] - b[3]
    return math_sqrt(dx * dx + dy * dy + dz * dz)
end

local function catmull_rom(p0, p1, p2, p3, t)
    local t2 = t * t
    local t3 = t2 * t
    local function c(i)
        return 0.5 * ((2 * p1[i])
            + (-p0[i] + p2[i]) * t
            + (2 * p0[i] - 5 * p1[i] + 4 * p2[i] - p3[i]) * t2
            + (-p0[i] + 3 * p1[i] - 3 * p2[i] + p3[i]) * t3)
    end
    return {c(1), c(2), c(3)}
end

function NavMoon:smooth_path_catmull(path, step)
    if not path or #path < 2 then return path end
    step = step or 1.0
    if step < 0.2 then step = 0.2 end
    local n = #path
    local out = {{path[1][1], path[1][2], path[1][3]}}
    for i = 1, n - 1 do
        local p0 = path[i > 1 and (i - 1) or 1]
        local p1 = path[i]
        local p2 = path[i + 1]
        local p3 = path[(i + 2) <= n and (i + 2) or n]
        local seg_len = dist3(p1, p2)
        local samples = math.max(1, math_floor(seg_len / step))
        for s = 1, samples do
            local t = s / samples
            out[#out + 1] = catmull_rom(p0, p1, p2, p3, t)
        end
    end
    return out
end

function NavMoon:postprocess_path(path)
    if not path or type(path) ~= "table" or #path == 0 then
        return path
    end
    if cfg.path_string_pull ~= false then
        path = self:string_pull_path(path)
    end
    local mode = cfg.path_smooth
    if mode == true or mode == "catmull" then
        path = self:smooth_path_catmull(path, cfg.path_smooth_step or 1.0)
    end
    return path
end

function NavMoon:generate_path(from, to)
    local result, data = self:a_star(from, to, self.last_edges)
    if not result then return false end
    local id_to_pos = self.last_id_to_pos
    local data_path = data.path
    local path = {}
    for i = 2, #data_path do
        local pos = id_to_pos[data_path[i]]
        if pos then
            path[#path + 1] = {pos[1], pos[2], pos[3]}
        end
    end
    if #path == 0 then return false end
    return self:postprocess_path(path)
end

function NavMoon:rrt_star_path(start_id, goal_id)
    local id_to_pos = self.last_id_to_pos
    local start_pos = id_to_pos[start_id]
    local goal_pos = id_to_pos[goal_id]
    if not start_pos or not goal_pos then return nil end

    local sample_ids, sample_count = self:collect_sample_ids()
    if sample_count == 0 then return nil end

    local tree = {start_id}
    local parent = {[start_id] = false}
    local cost = {[start_id] = 0}

    local function dist(a, b)
        local pa, pb = id_to_pos[a], id_to_pos[b]
        local dx, dy, dz = pa[1] - pb[1], pa[2] - pb[2], pa[3] - pb[3]
        return math_sqrt(dx * dx + dy * dy + dz * dz)
    end

    local function can_link(a, b)
        return self:can_connect_ids(a, b)
    end

    if can_link(start_id, goal_id) then return {start_id, goal_id} end

    local function nearest_node(sample_id)
        local best_id = tree[1]
        local best_d = dist(best_id, sample_id)
        local limit = cfg.rrt_nearest_candidates
        local candidates = {}
        local cand_n = 0

        if #tree <= limit then
            for i = 1, #tree do
                cand_n = cand_n + 1
                candidates[cand_n] = tree[i]
            end
        else
            for i = 1, limit do
                cand_n = cand_n + 1
                candidates[cand_n] = tree[math_random(1, #tree)]
            end
        end

        for i = 1, cand_n do
            local id = candidates[i]
            local d = dist(id, sample_id)
            if d < best_d then
                best_d = d
                best_id = id
            end
        end
        return best_id, best_d
    end

    local function steer(from_id, to_id)
        local from_pos = id_to_pos[from_id]
        local to_pos = id_to_pos[to_id]
        local dx, dy, dz = to_pos[1] - from_pos[1], to_pos[2] - from_pos[2], to_pos[3] - from_pos[3]
        local len = math_sqrt(dx * dx + dy * dy + dz * dz)
        if len < 0.001 then return nil end
        if len <= cfg.rrt_step then return to_id end
        local t = cfg.rrt_step / len
        local sx = from_pos[1] + dx * t
        local sy = from_pos[2] + dy * t
        local sz = from_pos[3] + dz * t
        local pid = self:get_point_id_from_coords(sx, sy, sz)
        return pid
    end

    local function collect_near_nodes(new_id)
        local near = {}
        local n = 0
        local radius = cfg.rrt_radius
        local max_n = cfg.rrt_rewire_limit
        for i = 1, #tree do
            local id = tree[i]
            if dist(id, new_id) <= radius then
                n = n + 1
                near[n] = id
                if n >= max_n then break end
            end
        end
        return near, n
    end

    for iter = 1, cfg.rrt_max_iters do
        local sample_id
        if math_random() < cfg.rrt_goal_bias then
            sample_id = goal_id
        else
            sample_id = sample_ids[math_random(1, sample_count)]
        end

        local near_id = nearest_node(sample_id)
        local new_id = steer(near_id, sample_id)
        if new_id and new_id ~= start_id and not parent[new_id] and can_link(near_id, new_id) then
            local best_parent = near_id
            local best_cost = cost[near_id] + dist(near_id, new_id)

            local near_nodes = collect_near_nodes(new_id)
            for i = 1, #near_nodes do
                local nid = near_nodes[i]
                local cand_cost = cost[nid] + dist(nid, new_id)
                if cand_cost < best_cost and can_link(nid, new_id) then
                    best_cost = cand_cost
                    best_parent = nid
                end
            end

            parent[new_id] = best_parent
            cost[new_id] = best_cost
            tree[#tree + 1] = new_id

            for i = 1, #near_nodes do
                local nid = near_nodes[i]
                local cand_cost = best_cost + dist(new_id, nid)
                if cand_cost < (cost[nid] or math.huge) and can_link(new_id, nid) then
                    parent[nid] = new_id
                    cost[nid] = cand_cost
                end
            end

            local d_goal = dist(new_id, goal_id)
            if d_goal <= cfg.rrt_goal_radius and can_link(new_id, goal_id) then
                parent[goal_id] = new_id
                cost[goal_id] = best_cost + d_goal
                local path = {goal_id}
                local cur = goal_id
                while parent[cur] do
                    cur = parent[cur]
                    path[#path + 1] = cur
                end
                local rev = {}
                for i = #path, 1, -1 do rev[#rev + 1] = path[i] end
                return rev
            end
        end
    end
    return nil
end

function NavMoon:generate_path_hybrid(sx, sy, sz, gx, gy, gz)
    local start_id = self:get_point_id_from_coords(sx, sy, sz)
    local goal_id = self:get_point_id_from_coords(gx, gy, gz)
    if not start_id or not goal_id then
        return false, "no_mesh_point"
    end

    if start_id == goal_id then
        local pos = self.last_id_to_pos[goal_id]
        if pos then return {{pos[1], pos[2], pos[3]}} end
    end

    if self:can_connect_ids(start_id, goal_id) then
        local b = self.last_id_to_pos[goal_id]
        if b then
            return {{b[1], b[2], b[3]}}
        end
    end

    local a_star_path = self:generate_path(start_id, goal_id)
    if a_star_path and type(a_star_path) == "table" and #a_star_path > 0 then
        return a_star_path
    end

    if not cfg.rrt_enabled then
        return false
    end

    local min_rrt = cfg.rrt_min_distance or 20.0
    local sp = self.last_id_to_pos[start_id]
    local gp = self.last_id_to_pos[goal_id]
    if sp and gp then
        local dx, dy, dz = sp[1] - gp[1], sp[2] - gp[2], sp[3] - gp[3]
        local d = math_sqrt(dx * dx + dy * dy + dz * dz)
        if d < min_rrt then
            return false
        end
    end

    local rrt_path = self:rrt_star_path(start_id, goal_id)
    if not rrt_path or #rrt_path < 2 then
        return false
    end

    local id_to_pos = self.last_id_to_pos
    local full_path = {}
    for i = 1, #rrt_path - 1 do
        local ok, data = self:a_star(rrt_path[i], rrt_path[i + 1], self.last_edges)
        if ok and data and data.path then
            local data_path = data.path
            for j = 2, #data_path do
                local pos = id_to_pos[data_path[j]]
                if pos then
                    full_path[#full_path + 1] = {pos[1], pos[2], pos[3]}
                end
            end
        else
            local pos = id_to_pos[rrt_path[i + 1]]
            if pos then
                full_path[#full_path + 1] = {pos[1], pos[2], pos[3]}
            end
        end
    end

    if #full_path == 0 then return false end
    return self:postprocess_path(full_path)
end

return NavMoon
