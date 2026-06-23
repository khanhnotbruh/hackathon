cmake_minimum_required(VERSION 3.16)
project(hackathon LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG        master 
)

# set(GGML_CUDA ON CACHE BOOL "Build with CUDA" FORCE)
FetchContent_MakeAvailable(llama_cpp)
add_executable(engine simdjson/simdjson.cpp main.cpp)
target_link_libraries(engine PRIVATE llama)
target_include_directories(engine PRIVATE simdjson)
