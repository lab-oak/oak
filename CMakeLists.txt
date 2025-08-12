cmake_minimum_required(VERSION 3.8)
project(oak VERSION 0.0.3)

set(CMAKE_CXX_STANDARD 23)

add_subdirectory(extern/lrsnash EXCLUDE_FROM_ALL)
add_subdirectory(extern/eigen EXCLUDE_FROM_ALL)

# Libpkmn

function(create_library target mode dir)
  add_library(${target} STATIC IMPORTED)
  set_target_properties(${target} PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/libpkmn/${mode}/${dir}/libpkmn-showdown.a
  )
  target_include_directories(${target} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/extern/engine/zig-out/include)
endfunction()

create_library(libpkmn_client_release release client)
create_library(libpkmn_client_debug debug client)
create_library(libpkmn_chance_release release chance)
create_library(libpkmn_chance_debug debug chance)
create_library(libpkmn_fast_release release fast)
create_library(libpkmn_fast_debug debug fast)

add_library(libpkmn_client INTERFACE)
target_link_libraries(libpkmn_client INTERFACE
  $<$<CONFIG:Debug>:libpkmn_client_debug>
  $<$<CONFIG:Release>:libpkmn_client_release>
)
add_library(libpkmn_chance INTERFACE)
target_link_libraries(libpkmn_chance INTERFACE
  $<$<CONFIG:Debug>:libpkmn_chance_debug>
  $<$<CONFIG:Release>:libpkmn_chance_release>
)
add_library(libpkmn_fast INTERFACE)
target_link_libraries(libpkmn_fast INTERFACE
  $<$<CONFIG:Debug>:libpkmn_fast_debug>
  $<$<CONFIG:Release>:libpkmn_fast_release>
)

target_compile_definitions(libpkmn_client INTERFACE LOG CHANCE CALC)
target_compile_definitions(libpkmn_chance INTERFACE CHANCE CALC)

# Targets

add_executable(benchmark src/benchmark.cc)
target_include_directories(benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(benchmark PRIVATE libpkmn_chance lrslib128 eigen)

add_executable(dur src/dur.cc)
target_include_directories(dur PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(dur PRIVATE libpkmn_chance lrslib128)

add_executable(debug-log src/debug-log.cc)
target_include_directories(debug-log PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(debug-log PRIVATE libpkmn_client)

add_executable(chall src/chall.cc)
target_include_directories(chall PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(chall PRIVATE libpkmn_chance lrslib128)

add_executable(generate src/generate.cc)
target_include_directories(generate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(generate PRIVATE libpkmn_chance lrslib128 eigen)

add_library(pyoak SHARED src/pyoak.cc)
target_include_directories(pyoak PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(pyoak PRIVATE libpkmn_chance)

add_executable(vs src/vs.cc)
target_include_directories(vs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vs PRIVATE libpkmn_chance lrslib128 eigen)
