cmake_minimum_required(VERSION 3.20)
project(shop)

# Register one test executable.
function(add_shop_test name)
  add_executable(${name} ${name}.cpp)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

macro(enable_warnings tgt)
  target_compile_options(${tgt} PRIVATE -Wall)
endmacro()

add_shop_test(pricing_test)
