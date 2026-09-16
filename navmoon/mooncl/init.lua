script_version("0.2.0")
local ffi = require("ffi")

ffi.cdef[[
    typedef void* cl_mem;
    typedef void* cl_event;
    typedef struct CustomKernelHandle CustomKernelHandle;
    typedef struct { float x, y, z, w; } mooncl_float4;

    int  mooncl_init(char* out_info, int max_len);
    void mooncl_cleanup();
    int  mooncl_is_ready();

    CustomKernelHandle* mooncl_compile_kernel(const char* source, const char* kernel_name, char* out_log, int max_log);
    CustomKernelHandle* mooncl_get_builtin_kernel(const char* kernel_name, char* out_log, int max_log);
    void                mooncl_free_kernel(CustomKernelHandle* handle);

    cl_mem   mooncl_create_buffer(size_t size);
    void     mooncl_free_buffer(cl_mem buf);
    int      mooncl_write_buffer(cl_mem buf, const void* data, size_t size);
    int      mooncl_read_buffer(cl_mem buf, void* data, size_t size);
    cl_event mooncl_read_buffer_async(cl_mem buf, void* data, size_t size);

    int      mooncl_set_arg_buffer(CustomKernelHandle* handle, int index, cl_mem buf);
    int      mooncl_set_arg_val(CustomKernelHandle* handle, int index, const void* val_ptr, size_t size);
    int      mooncl_run_kernel(CustomKernelHandle* handle, size_t global_size);
    cl_event mooncl_run_kernel_async(CustomKernelHandle* handle, size_t global_size);
    int      mooncl_run_kernel_nd(CustomKernelHandle* handle, int work_dim, const size_t* global_work_size);
    cl_event mooncl_run_kernel_nd_async(CustomKernelHandle* handle, int work_dim, const size_t* global_work_size);

    int      mooncl_event_is_done(cl_event evt);
    int      mooncl_event_wait(cl_event evt);
    void     mooncl_event_free(cl_event evt);
]]

local Event = {}
Event.__index = Event

local Buffer = {}
Buffer.__index = Buffer

local Kernel = {}
Kernel.__index = Kernel

local INIT_STAGES = {
    [-1] = "clGetPlatformIDs",
    [-2] = "clGetDeviceIDs",
    [-3] = "clCreateContext",
    [-4] = "clCreateCommandQueue",
}

local M = {}

local cl = nil

local function get_dll()
    if not cl then
        local candidates = {
            getWorkingDirectory() .. "\\lib\\navmoon\\mooncl\\mooncl.dll",
            getWorkingDirectory() .. "\\lib\\mooncl\\mooncl.dll",
            getWorkingDirectory() .. "\\mooncl\\mooncl.dll",
        }
        local last_err
        for i = 1, #candidates do
            local ok, lib = pcall(ffi.load, candidates[i])
            if ok and lib then
                cl = lib
                break
            end
            last_err = lib
        end
        if not cl then
            error("[MoonCL] cannot load mooncl.dll. Tried:\n  " .. table.concat(candidates, "\n  ") .. "\n" .. tostring(last_err))
        end
    end
    return cl
end

local function wrap_event(handle)
    local dll = get_dll()
    ffi.gc(handle, dll.mooncl_event_free)
    return setmetatable({ handle = handle }, Event)
end

function Event:is_done()
    if not self.handle then return true end
    local dll = get_dll()
    return dll.mooncl_event_is_done(self.handle) == 1
end

function Event:is_ready()
    return self:is_done()
end

function Event:wait()
    if not self.handle then return false end
    local dll = get_dll()
    return dll.mooncl_event_wait(self.handle) == 1
end

function Event:free()
    if self.handle then
        local dll = get_dll()
        ffi.gc(self.handle, nil)
        dll.mooncl_event_free(self.handle)
        self.handle = nil
    end
end

function M.init()
    local dll = get_dll()
    local info_buf = ffi.new("char[2048]")
    local res = dll.mooncl_init(info_buf, 2048)

    local msg = ffi.string(info_buf)
    if res == 1 then
        return true, msg
    else
        local stage_name = INIT_STAGES[res] or "mooncl_init"
        local err_text = string.format("%s (%d): %s",
            stage_name,
            res,
            msg ~= "" and msg or "unknown initialization error"
        )
        return false, err_text
    end
end

function M.is_ready()
    if not cl then return false end
    return cl.mooncl_is_ready() == 1
end

function M.cleanup()
    if cl then
        cl.mooncl_cleanup()
    end
end

function Kernel:set_float4(arg_idx, x, y, z, w)
    assert(self.handle ~= nil, "[MoonCL] kernel is not initialized!")

    local v
    if type(x) == "cdata" then
        return self:set_raw(arg_idx, x, 16)
    elseif type(x) == "table" then
        v = ffi.new("mooncl_float4", x[1] or x.x or 0, x[2] or x.y or 0, x[3] or x.z or 0, x[4] or x.w or 0)
    else
        v = ffi.new("mooncl_float4", x or 0, y or 0, z or 0, w or 0)
    end

    local dll = get_dll()
    return dll.mooncl_set_arg_val(self.handle, arg_idx, v, 16) == 1
end

function M.new_float4_array(count)
    return ffi.new("mooncl_float4[?]", count)
end

function M.create_buffer(size_bytes)
    assert(M.is_ready(), "[MoonCL] GPU is not initialized!")
    local dll = get_dll()
    local mem = dll.mooncl_create_buffer(size_bytes)
    assert(mem ~= nil, "[MoonCL] failed to allocate buffer in VRAM!")

    ffi.gc(mem, dll.mooncl_free_buffer)

    return setmetatable({ handle = mem, size = size_bytes }, Buffer)
end

function Buffer:write(cdata, size)
    assert(self.handle ~= nil, "[MoonCL] write to a freed buffer!")
    local dll = get_dll()
    return dll.mooncl_write_buffer(self.handle, cdata, size or self.size) == 1
end

function Buffer:read(cdata, size)
    assert(self.handle ~= nil, "[MoonCL] read from a freed buffer!")
    local dll = get_dll()
    return dll.mooncl_read_buffer(self.handle, cdata, size or self.size) == 1
end

function Buffer:read_async(cdata, size)
    assert(self.handle ~= nil, "[MoonCL] read from a freed buffer!")
    local dll = get_dll()
    local evt = dll.mooncl_read_buffer_async(self.handle, cdata, size or self.size)
    if evt == nil then return nil end
    return wrap_event(evt)
end

function Buffer:free()
    if self.handle then
        local dll = get_dll()
        ffi.gc(self.handle, nil)
        dll.mooncl_free_buffer(self.handle)
        self.handle = nil
    end
end

function M.compile(source, kernel_name)
    assert(M.is_ready(), "[MoonCL] GPU is not initialized!")
    local dll = get_dll()
    local log_buf = ffi.new("char[4096]")
    local handle = dll.mooncl_compile_kernel(source, kernel_name, log_buf, 4096)

    if handle == nil then
        return nil, ffi.string(log_buf)
    end

    ffi.gc(handle, dll.mooncl_free_kernel)

    return setmetatable({ handle = handle, name = kernel_name }, Kernel), nil
end

function M.get_kernel(kernel_name)
    assert(M.is_ready(), "[MoonCL] GPU is not initialized!")
    local dll = get_dll()
    local log_buf = ffi.new("char[4096]")
    local handle = dll.mooncl_get_builtin_kernel(kernel_name, log_buf, 4096)

    if handle == nil then
        return nil, ffi.string(log_buf)
    end

    ffi.gc(handle, dll.mooncl_free_kernel)

    return setmetatable({ handle = handle, name = kernel_name }, Kernel), nil
end

function Kernel:set_buffer(arg_idx, buffer_obj)
    assert(self.handle ~= nil, "[MoonCL] kernel is not initialized!")
    local dll = get_dll()

    local h = buffer_obj
    if type(buffer_obj) == "table" then
        h = buffer_obj.handle
    end

    assert(h ~= nil, "[MoonCL] null buffer or freed buffer!")

    return dll.mooncl_set_arg_buffer(self.handle, arg_idx, h) == 1
end

function Kernel:set_float(arg_idx, val)
    assert(self.handle ~= nil, "[MoonCL] kernel is not initialized!")
    local dll = get_dll()
    local v = ffi.new("float[1]", val)
    return dll.mooncl_set_arg_val(self.handle, arg_idx, v, 4) == 1
end

function Kernel:set_int(arg_idx, val)
    assert(self.handle ~= nil, "[MoonCL] kernel is not initialized!")
    local dll = get_dll()
    local v = ffi.new("int[1]", val)
    return dll.mooncl_set_arg_val(self.handle, arg_idx, v, 4) == 1
end

function Kernel:set_raw(arg_idx, cdata_ptr, type_size)
    assert(self.handle ~= nil, "[MoonCL] kernel is not initialized!")
    local dll = get_dll()
    return dll.mooncl_set_arg_val(self.handle, arg_idx, cdata_ptr, type_size) == 1
end

function Kernel:run(global_size)
    assert(self.handle ~= nil, "[MoonCL] kernel is not initialized!")
    local dll = get_dll()
    return dll.mooncl_run_kernel(self.handle, global_size) == 1
end

function Kernel:run_async(global_size)
    assert(self.handle ~= nil, "[MoonCL] kernel is not initialized!")
    local dll = get_dll()
    local evt = dll.mooncl_run_kernel_async(self.handle, global_size)
    if evt == nil then return nil end
    return wrap_event(evt)
end

function Kernel:run_nd(work_dim, sizes_table)
    assert(self.handle ~= nil, "[MoonCL] kernel is not initialized!")
    local dll = get_dll()
    local c_sizes = ffi.new("size_t[?]", work_dim, sizes_table)
    return dll.mooncl_run_kernel_nd(self.handle, work_dim, c_sizes) == 1
end

function Kernel:run_nd_async(work_dim, sizes_table)
    assert(self.handle ~= nil, "[MoonCL] kernel is not initialized!")
    local dll = get_dll()
    local c_sizes = ffi.new("size_t[?]", work_dim, sizes_table)
    local evt = dll.mooncl_run_kernel_nd_async(self.handle, work_dim, c_sizes)
    if evt == nil then return nil end
    return wrap_event(evt)
end

function Kernel:free()
    if self.handle then
        local dll = get_dll()
        ffi.gc(self.handle, nil)
        dll.mooncl_free_kernel(self.handle)
        self.handle = nil
    end
end

return M
