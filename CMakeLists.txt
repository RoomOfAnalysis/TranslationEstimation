cmake_minimum_required(VERSION 3.21)

project(translation_estimation LANGUAGES CXX)

if(POLICY CMP0167)
   cmake_policy(SET CMP0167 OLD)
endif()

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};D:/source/lib/libtorch_cpu")
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

find_package(OpenMP REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
find_package(Torch CONFIG REQUIRED)

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
        target_compile_options(${name} PRIVATE /utf-8 /EHsc /MP /bigobj)
        set_target_properties(${name} PROPERTIES
            LINK_FLAGS "/MANIFEST:EMBED /MANIFESTINPUT:${CMAKE_CURRENT_SOURCE_DIR}/utf8.manifest"
        )
    endif()
endfunction()

function(addTorchTest name)
    set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    add_executable(${name} ${SRC_DIR}/${name}.cpp)
    target_include_directories(${name}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(${name}
	    PRIVATE ${TORCH_LIBRARIES}
        PRIVATE ${OpenCV_LIBS}
        PRIVATE OpenMP::OpenMP_CXX
        PRIVATE argparse::argparse
    )
    if(MSVC)
        target_compile_options(${name} PRIVATE /utf-8 /EHsc /MP /bigobj)
        set_target_properties(${name} PROPERTIES
            LINK_FLAGS "/MANIFEST:EMBED /MANIFESTINPUT:${CMAKE_CURRENT_SOURCE_DIR}/utf8.manifest"
        )
    endif()
endfunction()

addTest(chi2_shift_test)
addTest(phase_cross_correlation_test)

addTorchTest(chi2_shift_torch_test)
addTorchTest(phase_cross_correlation_torch_test)
addTorchTest(seam_find_test)
