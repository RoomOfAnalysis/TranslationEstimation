cmake_minimum_required(VERSION 3.21)

project(chi2shift LANGUAGES CXX)

if(POLICY CMP0167)
   cmake_policy(SET CMP0167 OLD)
endif()

set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE INTERNAL "" FORCE)  # -Wno-dev

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
else()
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

find_package(xtl CONFIG REQUIRED)
find_package(xtensor CONFIG REQUIRED)
find_package(xtensor-blas CONFIG REQUIRED)
find_package(xsimd CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(argparse CONFIG REQUIRED)
find_package(FFTW3 CONFIG REQUIRED)

find_path(XTENSOR_FFTW_INCLUDE_DIRS "xtensor-fftw/basic.hpp")

function(addTest name)
    set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    add_executable(${name} ${SRC_DIR}/${name}.cpp)
    target_include_directories(${name}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
	    PRIVATE ${XTENSOR_FFTW_INCLUDE_DIRS}
    )
    target_link_libraries(${name}
	    PRIVATE xtl
    	PRIVATE xtensor xtensor::optimize xtensor::use_xsimd
    	PRIVATE xtensor-blas
    	PRIVATE xsimd
        PRIVATE ${OpenCV_LIBS}
    	PRIVATE FFTW3::fftw3
        PRIVATE argparse::argparse
    )
    if(MSVC)
        target_compile_options(${name} PRIVATE /EHsc /MP /bigobj)
    endif()
endfunction()

addTest(chi2_shift_test)
