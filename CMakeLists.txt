# 设置CMake的最低版本
cmake_minimum_required(VERSION 3.10)

# 设置项目名称和支持的语言
project(FasterGEMM)

#检查指令集
include(CheckCXXSourceCompiles)

set(AVX_CXX_CODE "
#include <immintrin.h>
int main(){
    __m128 a = _mm_set1_ps(1.0f);
    return 0;
}
")
check_cxx_source_compiles("${AVX_CXX_CODE}" AVX_SUPPORTED)
if(AVX_SUPPORTED)
    message("AVX is supported")
    add_compile_options(-mavx)
else()
    message("AVX is not supported")
endif()

set(AVX2_CXX_CODE "
#include <immintrin.h>
int main() { 
return 0; 
}")
check_cxx_source_compiles("${AVX2_CXX_CODE}" AVX2_SUPPORTED)
if(AVX2_SUPPORTED)
    message("AVX2 is supported")
    add_compile_options(-mavx2)
else()
    message("AVX2 is not supported")
endif()

set(FMA_CXX_CODE "
#include <immintrin.h>
int main() { 
return 0; 
}")
check_cxx_source_compiles("${FMA_CXX_CODE}" FMA_SUPPORTED)
if(AVX2_SUPPORTED)
    message("FMA is supported")
    add_compile_options(-mfma)
else()
    message("FMA is not supported")
endif()

# 设置包含目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 添加库目标
add_library(labmat STATIC src/sgemm.cpp src/tools.cpp src/transpose.cpp src/vecxmat.cpp src/vecxvec.cpp)  # 静态库
add_library(lab SHARED src/sgemm.cpp src/tools.cpp src/transpose.cpp src/vecxmat.cpp src/vecxvec.cpp)  # 动态库

# 设置输出目录
# 设置生成的库文件的输出目录
set_target_properties(labmat PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib  # 静态库输出目录
    
)

set_target_properties(lab PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib  # 动态库输出目录
)