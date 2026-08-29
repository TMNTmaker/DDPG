cmake_minimum_required(VERSION 3.15)

project(ddpg_from_scratch
    VERSION 1.0.0
    DESCRIPTION "DDPG implemented from scratch without any external library"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(ddpg
    DDPG/DDPG/main.cpp
    DDPG/DDPG/unit.cpp
)

target_include_directories(ddpg PRIVATE DDPG/DDPG)

if(MSVC)
    # BOM なしの UTF-8 ソースをコードページに依存せず読ませる
    target_compile_options(ddpg PRIVATE /utf-8 /W3)
    # プリコンパイル済みヘッダー生成用の翻訳単位（Visual Studio 側の構成と揃える）
    target_sources(ddpg PRIVATE DDPG/DDPG/stdafx.cpp)
else()
    target_compile_options(ddpg PRIVATE -Wall)
    target_link_libraries(ddpg PRIVATE m)
endif()
