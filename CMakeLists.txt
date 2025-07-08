cmake_minimum_required(VERSION 3.8)
project(oak VERSION 0.0.3)

set(CMAKE_CXX_STANDARD 23)

add_subdirectory(extern/lrsnash EXCLUDE_FROM_ALL)
add_subdirectory(extern/eigen EXCLUDE_FROM_ALL)

function(create_library target mode dir)
  add_library(${target} STATIC IMPORTED)
  set_target_properties(${target} PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/libpkmn/${mode}/${dir}/libpkmn-showdown.a)
  target_include_directories(${target} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/extern/engine/zig-out/include)
endfunction()

create_library(libpkmn_client release client)
create_library(libpkmn_chance release chance)
create_library(libpkmn_fast release fast)

target_compile_definitions(libpkmn_client INTERFACE LOG CHANCE CALC)
target_compile_definitions(libpkmn_chance INTERFACE CHANCE CALC)
# target_compile_definitions(libpkmn_fast INTERFACE)

add_custom_target(build_client_debug ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/libpkmn/debug/client/libpkmn-showdown.a)
add_custom_target(build_chance_debug ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/libpkmn/debug/chance/libpkmn-showdown.a)
add_custom_target(build_fast_debug ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/libpkmn/debug/fast/libpkmn-showdown.a)
add_custom_target(build_client_release ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/libpkmn/release/client/libpkmn-showdown.a)
add_custom_target(build_chance_release ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/libpkmn/release/chance/libpkmn-showdown.a)
add_custom_target(build_fast_release ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/libpkmn/release/fast/libpkmn-showdown.a)

add_executable(benchmark src/benchmark.cc)
add_dependencies(benchmark build_chance_release)
target_include_directories(benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(benchmark PRIVATE libpkmn_chance lrslib128)

add_executable(debug-log src/debug-log.cc)
add_dependencies(debug-log build_client_release)
target_include_directories(debug-log PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(debug-log PRIVATE libpkmn_client)

add_executable(chall src/chall.cc)
add_dependencies(chall build_chance_release)
target_include_directories(chall PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(chall PRIVATE libpkmn_chance lrslib128)

add_executable(generate src/generate.cc)
add_dependencies(generate build_chance_release)
target_include_directories(generate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(generate PRIVATE libpkmn_chance lrslib128 eigen)

add_executable(decompress src/decompress.cc)
add_dependencies(decompress build_chance_release)
target_include_directories(decompress PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(decompress PRIVATE libpkmn_chance)