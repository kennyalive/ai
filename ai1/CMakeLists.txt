cmake_minimum_required(VERSION 3.10)
project(ai1)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SOURCE
    main.cpp
    random.h
    pcg_basic.c
    pcg_basic.h
)
add_executable(ai1 ${SOURCE})
