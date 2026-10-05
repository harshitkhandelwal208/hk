`src/vk/c.zig` holds the part of the Vulkan API the GPU backend uses, extracted from the Khronos
headers so the build needs no Vulkan SDK. To add something, add its name to `roots.txt`, then:

    git clone --depth 1 https://github.com/KhronosGroup/Vulkan-Headers /tmp/vkh
    cp -r /tmp/vkh/include/vulkan/vk_platform.h /tmp/vkh/include/vulkan/vulkan_core.h /tmp/vkh/include/vk_video .
    zig translate-c -lc -I. vulkan_core.h > vk_all.zig
    python3 tools/vk/extract.py          # reads vk_all.zig and roots.txt, writes vk_min.zig

then copy `vk_min.zig` over `src/vk/c.zig` without the two `VK_API_VERSION_1_*` lines (they use a
macro that does not translate; `src/vk/vk.zig` defines the version helper itself).
