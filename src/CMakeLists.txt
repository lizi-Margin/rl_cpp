file(GLOB_RECURSE SOURCE_FILES 
    "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
)

target_sources(rl
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
    ${SOURCE_FILES}
)