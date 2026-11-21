cmake_minimum_required(VERSION 3.20)
project(Davi_3.0.0_Joker CXX)

set(CMAKE_CXX_STANDARD 20)

# LibTorch (CUDA build). Defaults to the copy inside the santorini2025 engine repo.
set(LIBTORCH_DIR "C:/Users/rafae/CLionProjects/santorini2025/extern/libtorch" CACHE PATH "LibTorch root")
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_DIR}")
find_package(Torch REQUIRED)

add_executable(Davi_3.0.0_Joker src/main.cpp src/board.cpp src/moves.cpp)
target_link_libraries(Davi_3.0.0_Joker PRIVATE "${TORCH_LIBRARIES}")

if(MSVC)
    target_compile_definitions(Davi_3.0.0_Joker PRIVATE NOMINMAX)
    target_compile_options(Davi_3.0.0_Joker PRIVATE /bigobj /permissive-)
    # The Windows LibTorch CUDA link drops the CUDA backend unless this symbol is forced in.
    target_link_options(Davi_3.0.0_Joker PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()

# Copy LibTorch DLLs and the released network next to the executable.
file(GLOB TORCH_DLLS "${LIBTORCH_DIR}/lib/*.dll")
add_custom_command(TARGET Davi_3.0.0_Joker POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} $<TARGET_FILE_DIR:Davi_3.0.0_Joker>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/../weights/joker-3.0.0/santorini_evaluator.pt"
                $<TARGET_FILE_DIR:Davi_3.0.0_Joker>)
